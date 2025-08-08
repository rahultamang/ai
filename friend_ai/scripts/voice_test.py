import argparse
import os
import time
from rich import print

from friend_ai.config import ConfigLoader
from friend_ai.tts import CoquiXTTS


def main():
    parser = argparse.ArgumentParser(description="Generate TTS audio using a cloned voice sample")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--out", type=str, default=None, help="Output wav path")
    args = parser.parse_args()

    cfg = ConfigLoader.load()

    tts = CoquiXTTS(
        model_name=cfg.tts.model_name,
        device=cfg.app.device,
        default_sample_rate=cfg.tts.sample_rate,
    )

    out_path = args.out or os.path.join(
        cfg.app.audio_out_dir, f"tts-{time.strftime('%Y%m%d-%H%M%S')}.wav"
    )

    result = tts.synthesize_to_file(
        text=args.text,
        speaker_ref_wav=cfg.tts.speaker_ref_wav,
        language=cfg.tts.voice_clone_language,
        output_path=out_path,
        sample_rate=cfg.tts.sample_rate,
    )

    print({"audio_path": result.audio_path, "duration_s": round(result.duration_s, 2)})


if __name__ == "__main__":
    main()