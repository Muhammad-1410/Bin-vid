from __future__ import annotations
import collections
import os
import subprocess
import threading
from types import TracebackType
from typing import Deque
import numpy as np
_CFR_FLAG: list[str] | None = None

def _get_cfr_flag(ffmpeg_bin: str) -> list[str]:
    global _CFR_FLAG
    if _CFR_FLAG is not None:
        return _CFR_FLAG
    import re
    try:
        res = subprocess.run([ffmpeg_bin, '-version'], capture_output=True, text=True, check=False)
        match = re.search('ffmpeg version (\\d+)', res.stdout)
        if match and int(match.group(1)) >= 7:
            _CFR_FLAG = ['-fps_mode', 'cfr']
            return _CFR_FLAG
    except Exception:
        pass
    _CFR_FLAG = ['-vsync', 'cfr']
    return _CFR_FLAG

class FrameSink:

    def __init__(self, source_path: str, output_path: str, width: int, height: int, fps: float=30.0, crf: int=18, ffmpeg_path: str | None=None) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f'Dimensions width and height must be positive, got ({width}, {height}).')
        if fps <= 0:
            raise ValueError(f'fps must be positive, got {fps}.')
        if not 0 <= crf <= 51:
            raise ValueError(f'crf must be between 0 and 51, got {crf}.')
        self.source_path = source_path
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.crf = crf
        self.ffmpeg_path = ffmpeg_path
        self._proc: subprocess.Popen | None = None
        self._stderr_buffer: Deque[str] = collections.deque(maxlen=100)
        self._reader_thread: threading.Thread | None = None

    def __enter__(self) -> FrameSink:
        if self.ffmpeg_path is None:
            from binvid.environment import find_ffmpeg
            self.ffmpeg_path = find_ffmpeg()
        cfr_flag = _get_cfr_flag(self.ffmpeg_path)
        cmd = [self.ffmpeg_path, '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{self.width}x{self.height}', '-r', str(self.fps), '-i', '-', '-i', self.source_path, '-map', '0:v:0', '-map', '1:a:0?', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', str(self.crf), '-preset', 'medium', *cfr_flag, '-c:a', 'aac', '-b:a', '192k', '-shortest', self.output_path]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        def _stderr_reader() -> None:
            if self._proc is None or self._proc.stderr is None:
                return
            for line in iter(self._proc.stderr.readline, b''):
                self._stderr_buffer.append(line.decode('utf-8', errors='replace'))
            self._proc.stderr.close()
        self._reader_thread = threading.Thread(target=_stderr_reader, daemon=True)
        self._reader_thread.start()
        return self

    def _get_stderr_tail(self, max_lines: int=40) -> str:
        lines = list(self._stderr_buffer)[-max_lines:]
        return ''.join(lines) if lines else '<No stderr captured>'

    def write(self, frame: np.ndarray) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("FrameSink is not active. Use within a 'with' block.")
        if not isinstance(frame, np.ndarray):
            raise TypeError(f'Expected np.ndarray frame, got {type(frame).__name__}.')
        if frame.dtype != np.uint8:
            raise ValueError(f'Frame dtype must be uint8, got {frame.dtype}.')
        expected_shape = (self.height, self.width, 3)
        if frame.shape != expected_shape:
            raise ValueError(f'Frame shape {frame.shape} does not match expected {expected_shape}.')
        try:
            self._proc.stdin.write(memoryview(frame))
        except (BrokenPipeError, OSError) as exc:
            if self._reader_thread and self._reader_thread.is_alive():
                self._reader_thread.join(timeout=0.5)
            stderr_tail = self._get_stderr_tail(40)
            raise RuntimeError(f'Broken pipe writing to FFmpeg (encoder exited prematurely):\n{stderr_tail}') from exc

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> bool | None:
        if self._proc is None:
            return None
        if self._proc.stdin and (not self._proc.stdin.closed):
            try:
                self._proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        retcode = self._proc.wait()
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)
        if exc_type is not None:
            return False
        if retcode != 0:
            stderr_tail = self._get_stderr_tail(40)
            raise RuntimeError(f'FFmpeg encoding failed with exit code {retcode}:\n{stderr_tail}')
        return None

def main() -> None:
    import shutil
    import tempfile
    from binvid.environment import find_ffmpeg
    from binvid.probe import probe
    ffmpeg_bin = find_ffmpeg()
    temp_dir = tempfile.mkdtemp()
    source_video = os.path.join(temp_dir, 'source_ref.mp4')
    output_video = os.path.join(temp_dir, 'static_demo.mp4')
    print(f'Creating 2-second reference source video with audio tone...')
    subprocess.run([ffmpeg_bin, '-y', '-f', 'lavfi', '-i', 'testsrc=duration=2:size=320x240:rate=30', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2', '-c:v', 'libx264', '-c:a', 'aac', source_video], capture_output=True, check=True)
    width, height, fps = (320, 240, 30.0)
    num_frames = 60
    print(f'Piping {num_frames} frames of generated static at {fps:.0f} FPS into FrameSink...')
    rng = np.random.default_rng(42)
    with FrameSink(source_video, output_video, width=width, height=height, fps=fps, crf=18) as sink:
        for i in range(num_frames):
            static_frame = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
            sink.write(static_frame)
    print('Encoding completed successfully. Probing output...')
    info = probe(output_video)
    print('--- Encoded Video Probe ---')
    print(f'  Dimensions    : {info.width}x{info.height}')
    print(f'  FPS           : {info.fps:.2f}')
    print(f'  Frame count   : {info.frame_count}')
    print(f'  Duration      : {info.duration_seconds:.2f} s')
    print(f'  Has Audio     : {info.has_audio}')
    if info.frame_count == num_frames and info.has_audio:
        print('Verification: SUCCESS (60 frames encoded with audio preserved)')
    else:
        print(f'Verification: WARNING (Expected {num_frames} frames with audio, got {info})')
    shutil.rmtree(temp_dir, ignore_errors=True)
if __name__ == '__main__':
    main()
