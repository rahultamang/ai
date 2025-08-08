# AI Friend (Local, Persistent, Adaptive)

This project is a modular, local-first AI assistant designed to be persistent, adaptive, and voice-enabled.

Step 1 delivers:
- Local vector memory with ChromaDB + SentenceTransformers
- Voice cloning TTS with Coqui TTS XTTS-v2 (zero-shot from your voice sample)

Later steps will add:
- Local LLM (llama.cpp) for conversation
- Whisper STT for real-time transcription
- Full-duplex audio (barge-in), notifications, and a TUI/GUI

## Quickstart

1) Create a Python 3.11 venv and install dependencies:

```bash
sudo apt-get install -y python3.11 python3.11-venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
```

2) Place a short clean WAV of your voice at `voice_samples/your_voice.wav` (10–30s works well).

3) Configure `config.yaml` as needed, then test memory and voice cloning:

```bash
# Add a memory fact
python -m friend_ai.scripts.memory_demo --text "I love oolong tea and hiking at sunrise."

# Query the memory store
python -m friend_ai.scripts.memory_demo --query "What tea do I like?" --top_k 3

# Synthesize a sample phrase in your cloned voice
python -m friend_ai.scripts.voice_test --text "Hey, it's me. I'm your AI friend."
```

Generated audio will be saved under `data/audio/`.

## Config
See `config.yaml` for tunables like model names, device, and paths.

## Notes
- XTTS-v2 runs CPU-only but benefits from a GPU. The first run will download model weights.
- For best cloning quality, record your sample in a quiet room (16 kHz or 22.05 kHz WAV, mono is fine).
