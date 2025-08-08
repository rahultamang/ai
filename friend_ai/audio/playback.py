from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import sounddevice as sd


class AudioPlayer:
    def __init__(self):
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.OutputStream] = None

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        self.stop()
        self._stop_event.clear()

        def run():
            with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
                self._stream = stream
                idx = 0
                block = 1024
                while not self._stop_event.is_set() and idx < len(audio):
                    chunk = audio[idx : idx + block]
                    if chunk.ndim == 1:
                        stream.write(chunk.reshape(-1, 1))
                    else:
                        stream.write(chunk)
                    idx += block

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            if self._stream:
                try:
                    self._stream.abort()
                except Exception:
                    pass
            self._thread.join(timeout=1.0)
        self._thread = None
        self._stream = None