from __future__ import annotations
import os
from pathlib import Path
import shutil
import sys

def find_ffmpeg() -> str:
    if getattr(sys, 'frozen', False):
        base_dir = Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent))
        candidates = [base_dir / 'bin' / 'ffmpeg.exe', base_dir / 'ffmpeg.exe', Path(sys.executable).parent / 'bin' / 'ffmpeg.exe', Path(sys.executable).parent / 'ffmpeg.exe']
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate.resolve())
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        return os.path.abspath(ffmpeg_path)
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return os.path.abspath(exe)
    except ImportError:
        pass
    raise RuntimeError('FFmpeg binary not found on this system.\nPlease install FFmpeg and ensure it is added to your PATH, e.g.:\n  - Windows: winget install Gyan.FFmpeg or choco install ffmpeg\n  - macOS: brew install ffmpeg\n  - Linux: sudo apt update && sudo apt install ffmpeg\nOr install the imageio-ffmpeg Python fallback package via:\n  pip install imageio-ffmpeg')

def find_monospace_font() -> str:
    if getattr(sys, 'frozen', False):
        base_dir = Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent))
        candidates_frozen = [base_dir / 'fonts' / 'consola.ttf', base_dir / 'consola.ttf', Path(sys.executable).parent / 'fonts' / 'consola.ttf', Path(sys.executable).parent / 'consola.ttf']
        for font_candidate in candidates_frozen:
            if font_candidate.is_file():
                return str(font_candidate.resolve())
    candidates: list[str] = []
    if sys.platform.startswith('win'):
        windir = os.environ.get('WINDIR', 'C:\\Windows')
        candidates = [os.path.join(windir, 'Fonts', f) for f in ('consola.ttf', 'cour.ttf', 'lucon.ttf', 'CascadiaMono.ttf', 'CascadiaCode.ttf')]
    elif sys.platform == 'darwin':
        candidates = ['/System/Library/Fonts/Menlo.ttc', '/Library/Fonts/Courier New.ttf', '/System/Library/Fonts/Monaco.ttf', '/System/Library/Fonts/SFNSMono.ttf']
    else:
        candidates = [f'/usr/share/fonts/truetype/{sub}' for sub in ('dejavu/DejaVuSansMono.ttf', 'liberation/LiberationMono-Regular.ttf', 'ubuntu/UbuntuMono-R.ttf', 'freefont/FreeMono.ttf')]
    for font_path in candidates:
        if os.path.isfile(font_path):
            return os.path.abspath(font_path)
    try:
        import matplotlib
        mpl_font = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', 'DejaVuSansMono.ttf')
        candidates.append(mpl_font)
        if os.path.isfile(mpl_font):
            return os.path.abspath(mpl_font)
    except Exception:
        pass
    candidates_str = '\n'.join((f'  - {p}' for p in candidates))
    raise RuntimeError(f'No monospace font found on system. Candidate paths searched:\n{candidates_str}\nPlease provide a valid font_path in RenderConfig or install a monospace font.')

def main() -> None:
    print('=== binvid Environment Verification ===')
    try:
        ffmpeg_bin = find_ffmpeg()
        print(f'FFmpeg binary : {ffmpeg_bin}')
    except RuntimeError as err:
        print(f'FFmpeg binary : NOT FOUND\n  {err}')
    try:
        font_path = find_monospace_font()
        print(f'Monospace font: {font_path}')
    except RuntimeError as err:
        print(f'Monospace font: NOT FOUND\n  {err}')
if __name__ == '__main__':
    main()
