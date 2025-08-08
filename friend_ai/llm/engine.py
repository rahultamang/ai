from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Optional

from friend_ai.config import ConfigLoader

try:
    from llama_cpp import Llama
except Exception:
    Llama = None  # type: ignore


@dataclass
class Message:
    role: str  # 'system' | 'user' | 'assistant'
    content: str


class LocalLLM:
    def __init__(self, model_path: str, temperature: float = 0.6, top_p: float = 0.95, max_tokens: int = 512):
        self.model_path = model_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.model: Optional[object] = None
        if Llama is not None and os.path.exists(self.model_path):
            self.model = Llama(model_path=self.model_path, n_ctx=4096, n_threads=None)

    def generate(self, messages: List[Message]) -> str:
        if self.model is None:
            # Minimal fallback
            last_user = next((m.content for m in reversed(messages) if m.role == "user"), "" )
            return f"I heard you say: '{last_user}'. I'm not fully set up yet. Please add a local GGUF model to {self.model_path}."
        # llama.cpp chat completion
        formatted = [
            {"role": m.role, "content": m.content} for m in messages
        ]
        out = self.model.create_chat_completion(
            messages=formatted,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        return out["choices"][0]["message"]["content"].strip()