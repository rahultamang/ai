from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import soundfile as sf


@dataclass
class TTSResult:
    audio_path: str
    sample_rate: int
    duration_s: float


class CoquiXTTS:
    def __init__(self, model_name: str, device: str = "auto", default_sample_rate: int = 22050):
        try:
            from TTS.api import TTS  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "Coqui TTS is not installed. Install with 'pip install TTS' (use Python 3.10/3.11) "
                "or 'pip install git+https://github.com/coqui-ai/TTS.git'."
            ) from exc
        if device == "auto":
            device = None  # type: ignore
        self.tts = TTS(model_name=model_name, progress_bar=False, gpu=device == "cuda")
        self.default_sample_rate = default_sample_rate

    def synthesize_to_file(
        self,
        text: str,
        speaker_ref_wav: str,
        language: str = "en",
        output_path: Optional[str] = None,
        sample_rate: Optional[int] = None,
    ) -> TTSResult:
        assert os.path.exists(speaker_ref_wav), f"Missing speaker reference wav: {speaker_ref_wav}"
        sample_rate = sample_rate or self.default_sample_rate
        wav = self.tts.tts(
            text=text,
            speaker_wav=speaker_ref_wav,
            language=language,
        )
        audio = np.array(wav, dtype=np.float32)
        if output_path is None:
            ts = time.strftime("%Y%m%d-%H%M%S")
            output_path = f"output-{ts}.wav"
        sf.write(output_path, audio, sample_rate)
        duration = len(audio) / float(sample_rate)
        return TTSResult(audio_path=output_path, sample_rate=sample_rate, duration_s=duration)