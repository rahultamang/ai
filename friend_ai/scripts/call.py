import argparse
import time
from rich import print

from friend_ai.realtime import CallSession


def main():
    parser = argparse.ArgumentParser(description="Start a full-duplex voice call with your AI friend")
    parser.add_argument("--duration", type=int, default=0, help="Optional max duration in seconds (0 = until Ctrl+C)")
    args = parser.parse_args()

    session = CallSession()
    print("[bold green]Starting call. Speak any time. Press Ctrl+C to end.[/bold green]")
    session.start()
    try:
        if args.duration > 0:
            time.sleep(args.duration)
        else:
            while True:
                time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        session.stop()
        print("[bold yellow]Call ended.[/bold yellow]")


if __name__ == "__main__":
    main()