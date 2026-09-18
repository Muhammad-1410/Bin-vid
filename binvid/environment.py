from __future__ import annotations

import os
import shutil
import sys


def find_ffmpeg() -> str:
    """Locate the ffmpeg binary on the system.

    Searches PATH via shutil.which, falling back to imageio_ffmpeg if installed.
    Raises RuntimeError with installation instructions if neither works.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return os.path.abspath(ffmpeg_path)

    # Fallback to imageio_ffmpeg if available
    try:
        import imageio_ffmpeg  # type: ignore

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return os.path.abspath(exe)
    except ImportError:
        pass

    raise RuntimeError(
        "FFmpeg binary not found on this system.\n"
        "Please install FFmpeg and ensure it is added to your PATH, e.g.:\n"
        "  - Windows: winget install Gyan.FFmpeg or choco install ffmpeg\n"
        "  - macOS: brew install ffmpeg\n"
        "  - Linux: sudo apt update && sudo apt install ffmpeg\n"
        "Or install the imageio-ffmpeg Python fallback package via:\n"
        "  pip install imageio-ffmpeg"
    )


def find_monospace_font() -> str:
    """Search common platform paths and matplotlib for a monospace TrueType font.

    Raises RuntimeError listing all candidate paths that were checked if none found.
    """
    candidates: list[str] = []

    if sys.platform.startswith("win"):
        windir = os.environ.get("WINDIR", r"C:\Windows")
        candidates = [
            os.path.join(windir, "Fonts", f)
            for f in ("consola.ttf", "cour.ttf", "lucon.ttf", "CascadiaMono.ttf", "CascadiaCode.ttf")
        ]
    elif sys.platform == "darwin":
        candidates = [
            "/System/Library/Fonts/Menlo.ttc",
            "/Library/Fonts/Courier New.ttf",
            "/System/Library/Fonts/Monaco.ttf",
            "/System/Library/Fonts/SFNSMono.ttf",
        ]
    else:
        candidates = [
            f"/usr/share/fonts/truetype/{sub}"
            for sub in ("dejavu/DejaVuSansMono.ttf", "liberation/LiberationMono-Regular.ttf", "ubuntu/UbuntuMono-R.ttf", "freefont/FreeMono.ttf")
        ]

    for font_path in candidates:
        if os.path.isfile(font_path):
            return os.path.abspath(font_path)

    try:
        import matplotlib  # type: ignore
        mpl_font = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSansMono.ttf")
        candidates.append(mpl_font)
        if os.path.isfile(mpl_font):
            return os.path.abspath(mpl_font)
    except Exception:
        pass

    candidates_str = "\n".join(f"  - {p}" for p in candidates)
    raise RuntimeError(
        "No monospace font found on system. Candidate paths searched:\n"
        f"{candidates_str}\n"
        "Please provide a valid font_path in RenderConfig or install a monospace font."
    )


def main() -> None:
    """Print ffmpeg and monospace font discovery results."""
    print("=== binvid Environment Verification ===")

    try:
        ffmpeg_bin = find_ffmpeg()
        print(f"FFmpeg binary : {ffmpeg_bin}")
    except RuntimeError as err:
        print(f"FFmpeg binary : NOT FOUND\n  {err}")

    try:
        font_path = find_monospace_font()
        print(f"Monospace font: {font_path}")
    except RuntimeError as err:
        print(f"Monospace font: NOT FOUND\n  {err}")


if __name__ == "__main__":
    main()
