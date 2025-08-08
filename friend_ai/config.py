import os
import yaml
from dataclasses import dataclass
from typing import Any, Dict


DEFAULT_CONFIG_PATH = os.path.join("/workspace", "config.yaml")


@dataclass
class AppConfig:
    data_dir: str
    audio_out_dir: str
    db_dir: str
    device: str


@dataclass
class MemoryConfig:
    embedding_model: str
    collection_name: str
    top_k_default: int


@dataclass
class STTConfig:
    whisper_model_size: str
    vad_aggressiveness: int


@dataclass
class TTSConfig:
    provider: str
    model_name: str
    speaker_ref_wav: str
    sample_rate: int
    voice_clone_language: str


@dataclass
class LLMConfig:
    engine: str
    model_path: str
    temperature: float
    top_p: float
    max_tokens: int


@dataclass
class Config:
    app: AppConfig
    memory: MemoryConfig
    stt: STTConfig
    tts: TTSConfig
    llm: LLMConfig


class ConfigLoader:
    @staticmethod
    def load(config_path: str = DEFAULT_CONFIG_PATH) -> Config:
        with open(config_path, "r") as f:
            raw: Dict[str, Any] = yaml.safe_load(f)
        app = AppConfig(**raw["app"])  
        memory = MemoryConfig(**raw["memory"])  
        stt = STTConfig(**raw["stt"])  
        tts = TTSConfig(**raw["tts"])  
        llm = LLMConfig(**raw["llm"])  
        # Ensure dirs exist
        os.makedirs(app.data_dir, exist_ok=True)
        os.makedirs(app.audio_out_dir, exist_ok=True)
        os.makedirs(app.db_dir, exist_ok=True)
        return Config(app=app, memory=memory, stt=stt, tts=tts, llm=llm)