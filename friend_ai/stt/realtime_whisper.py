from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd
import webrtcvad

try:
    from faster_whisper import WhisperModel  # type: ignore
    _USE_FASTER = True
except Exception:
    import whisper  # type: ignore
    _USE_FASTER = False


@dataclass
class TranscriptionEvent:
    text: str
    is_final: bool


class RealtimeTranscriber:
    def __init__(self, model_size: str = "small", vad_aggressiveness: int = 2, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.frame_ms = 30
        self.frame_bytes = int(self.sample_rate * self.frame_ms / 1000)
        self.buffer = bytearray()
        self.audio_q: "queue.Queue[bytes]" = queue.Queue(maxsize=50)
        self.events_q: "queue.Queue[TranscriptionEvent]" = queue.Queue()
        self._stop = threading.Event()
        self._listening = threading.Event()
        if _USE_FASTER:
            self.model = WhisperModel(model_size, device="auto")
        else:
            self.model = whisper.load_model(model_size)
        self._in_stream: Optional[sd.InputStream] = None
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)

    def start(self):
        if self._worker_thread.is_alive():
            return
        self._stop.clear()
        self._listening.set()
        self._in_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            callback=self._on_audio,
        )
        self._in_stream.start()
        self._worker_thread.start()

    def stop(self):
        self._listening.clear()
        self._stop.set()
        try:
            if self._in_stream:
                self._in_stream.stop()
                self._in_stream.close()
        finally:
            self._in_stream = None

    def _on_audio(self, indata, frames, time_info, status):
        if status:
            pass
        if not self._listening.is_set():
            return
        pcm16 = indata[:, 0].astype(np.int16).tobytes()
        self.buffer.extend(pcm16)
        while len(self.buffer) >= self.frame_bytes * 2:
            chunk = bytes(self.buffer[: self.frame_bytes * 2])
            del self.buffer[: self.frame_bytes * 2]
            self.audio_q.put(chunk)

    def _worker_loop(self):
        voiced = False
        cur_chunks: list[bytes] = []
        last_voice_time = time.time()
        silence_timeout_s = 0.7
        while not self._stop.is_set():
            try:
                chunk = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                chunk = None
            if chunk is not None:
                is_speech = False
                try:
                    is_speech = self.vad.is_speech(chunk, self.sample_rate)
                except Exception:
                    is_speech = False
                if is_speech:
                    cur_chunks.append(chunk)
                    voiced = True
                    last_voice_time = time.time()
                elif voiced:
                    if time.time() - last_voice_time > silence_timeout_s:
                        audio_bytes = b"".join(cur_chunks)
                        cur_chunks.clear()
                        voiced = False
                        self._transcribe_and_emit(audio_bytes)
            else:
                if voiced and (time.time() - last_voice_time > silence_timeout_s):
                    audio_bytes = b"".join(cur_chunks)
                    cur_chunks.clear()
                    voiced = False
                    self._transcribe_and_emit(audio_bytes)

    def _transcribe_and_emit(self, audio_bytes: bytes):
        if not audio_bytes:
            return
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if _USE_FASTER:
            segments, _ = self.model.transcribe(audio_np, language="en")
            text_parts = [seg.text for seg in segments]
            text = " ".join(text_parts).strip()
        else:
            # whisper expects 16k float32 mono
            import torch
            with torch.no_grad():
                result = self.model.transcribe(audio_np, language="en")
            text = result.get("text", "").strip()
        if text:
            self.events_q.put(TranscriptionEvent(text=text, is_final=True))