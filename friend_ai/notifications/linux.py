import shutil
import subprocess


def notify(title: str, message: str) -> bool:
    if not shutil.which("notify-send"):
        return False
    try:
        subprocess.run(["notify-send", title, message], check=False)
        return True
    except Exception:
        return False