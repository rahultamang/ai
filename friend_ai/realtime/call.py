from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import List

import numpy as np

from friend_ai.audio import AudioPlayer
from friend_ai.config import ConfigLoader
from friend_ai.llm import LocalLLM, Message
from friend_ai.memory import MemoryStore
from friend_ai.stt import RealtimeTranscriber, TranscriptionEvent
from friend_ai.tts import CoquiXTTS


@dataclass
class DialogueTurn:
    role: str
    text: str


class CallSession:
    def __init__(self):
        self.cfg = ConfigLoader.load()
        self.store = MemoryStore(
            persist_dir=self.cfg.app.db_dir,
            collection_name=self.cfg.memory.collection_name,
            embedding_model=self.cfg.memory.embedding_model,
        )
        self.llm = LocalLLM(
            model_path=self.cfg.llm.model_path,
            temperature=self.cfg.llm.temperature,
            top_p=self.cfg.llm.top_p,
            max_tokens=self.cfg.llm.max_tokens,
        )
        self.tts = CoquiXTTS(
            model_name=self.cfg.tts.model_name,
            device=self.cfg.app.device,
            default_sample_rate=self.cfg.tts.sample_rate,
        )
        self.transcriber = RealtimeTranscriber(
            model_size=self.cfg.stt.whisper_model_size,
            vad_aggressiveness=self.cfg.stt.vad_aggressiveness,
            sample_rate=16000,
        )
        self.player = AudioPlayer()
        self.history: List[DialogueTurn] = []
        self._stop = threading.Event()
        self._resp_thread = threading.Thread(target=self._response_loop, daemon=True)

    def start(self):
        self._stop.clear()
        self.transcriber.start()
        self._resp_thread.start()

    def stop(self):
        self._stop.set()
        self.transcriber.stop()
        self.player.stop()

    def _response_loop(self):
        while not self._stop.is_set():
            try:
                evt: TranscriptionEvent = self.transcriber.events_q.get(timeout=0.1)
            except Exception:
                continue
            if not evt.is_final:
                continue
            user_text = evt.text.strip()
            if not user_text:
                continue
            # Barge-in: stop any current playback
            self.player.stop()
            self.history.append(DialogueTurn(role="user", text=user_text))
            # Persist user memory
            self.store.add(user_text, metadata={"from": "user", "ts": time.time()})
            # Retrieve related memories to ground reply
            related = self.store.query(user_text, top_k=self.cfg.memory.top_k_default)
            mem_context = "\n".join(f"- {m.text}" for m in related)
            system_prompt = (
                "You are a caring, concise AI friend. Personalize replies based on prior memories."
            )
            aug_user = user_text
            if mem_context:
                aug_user += f"\n\nRelevant memories:\n{mem_context}"
            messages = [
                Message(role="system", content=system_prompt),
                *[Message(role=t.role, content=t.text) for t in self.history[-6:]],
                Message(role="user", content=aug_user),
            ]
            reply = self.llm.generate(messages)
            self.history.append(DialogueTurn(role="assistant", text=reply))
            # TTS
            audio = self._synthesize_np(reply)
            self.player.play(audio=audio, sample_rate=self.cfg.tts.sample_rate)

    def _synthesize_np(self, text: str) -> np.ndarray:
        # Use Coqui to get numpy audio by saving then reading, or extend TTS class to return np
        # Here we call a hidden method by generating file then loading, but we prefer memory. We'll generate file then load.
        out_path = os.path.join(
            self.cfg.app.audio_out_dir, f"call-{int(time.time())}.wav"
        )
        res = self.tts.synthesize_to_file(
            text=text,
            speaker_ref_wav=self.cfg.tts.speaker_ref_wav,
            language=self.cfg.tts.voice_clone_language,
            output_path=out_path,
            sample_rate=self.cfg.tts.sample_rate,
        )
        import soundfile as sf
        audio, sr = sf.read(res.audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        return audio.astype(np.float32)