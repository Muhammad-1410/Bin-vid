from __future__ import annotations
import argparse
import math
import os
import re
import shutil
import subprocess
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator
import cv2
import numpy as np

@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    has_audio: bool
    rotation: int = 0

def get_video_rotation(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    ffprobe_bin = shutil.which('ffprobe')
    if ffprobe_bin:
        try:
            res = subprocess.run([ffprobe_bin, '-v', 'quiet', '-print_format', 'json', '-select_streams', 'v:0', '-show_streams', path], capture_output=True, text=True, check=False)
            if res.returncode == 0 and res.stdout.strip():
                import json
                data = json.loads(res.stdout)
                streams = data.get('streams', [])
                if streams:
                    s = streams[0]
                    tags = s.get('tags', {})
                    if 'rotate' in tags:
                        return int(float(tags['rotate'])) % 360
                    for side_data in s.get('side_data_list', []):
                        if 'rotation' in side_data:
                            return -int(float(side_data['rotation'])) % 360
        except Exception:
            pass
    try:
        from binvid.environment import find_ffmpeg
        ffmpeg_bin = find_ffmpeg()
        res = subprocess.run([ffmpeg_bin, '-i', path], capture_output=True, text=True, check=False)
        match = re.search('rotate\\s*:\\s*(-?\\d+)', res.stderr, re.IGNORECASE)
        if match:
            return int(float(match.group(1))) % 360
        match_dm = re.search('displaymatrix:\\s*rotation of (-?\\d+)', res.stderr, re.IGNORECASE)
        if match_dm:
            return -int(float(match_dm.group(1))) % 360
    except Exception:
        pass
    return 0

def check_has_audio(path: str) -> bool:
    ffprobe_bin = shutil.which('ffprobe')
    if ffprobe_bin:
        try:
            res = subprocess.run([ffprobe_bin, '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=codec_type', '-of', 'default=noprint_wrappers=1:nokey=1', path], capture_output=True, text=True, check=False)
            if res.returncode == 0 and 'audio' in res.stdout.lower():
                return True
        except Exception:
            pass
    try:
        from binvid.environment import find_ffmpeg
        ffmpeg_bin = find_ffmpeg()
        res = subprocess.run([ffmpeg_bin, '-i', path], capture_output=True, text=True, check=False)
        for line in res.stderr.splitlines():
            line_str = line.strip()
            if line_str.startswith('Stream #') and 'Audio:' in line_str:
                return True
    except Exception:
        pass
    return False

def probe(path: str) -> VideoInfo:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video file not found: '{path}'")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video file: '{path}'. The file may be corrupt or unreadable.")
    try:
        raw_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        raw_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        raw_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        cap.release()
    rotation = get_video_rotation(path)
    if rotation in (90, 270):
        width = int(raw_height)
        height = int(raw_width)
    else:
        width = int(raw_width)
        height = int(raw_height)
    fps = float(raw_fps)
    if fps <= 0 or math.isnan(fps) or math.isinf(fps):
        warnings.warn(f"Invalid or missing FPS ({raw_fps}) reported for '{path}'. Defaulting to 30.0 FPS.", UserWarning, stacklevel=2)
        fps = 30.0
    frame_count = int(raw_frame_count) if not math.isnan(raw_frame_count) and (not math.isinf(raw_frame_count)) else 0
    if frame_count <= 0:
        warnings.warn(f"Invalid or missing frame count ({raw_frame_count}) reported for '{path}'.", UserWarning, stacklevel=2)
        frame_count = max(0, frame_count)
    duration_seconds = frame_count / fps if frame_count > 0 and fps > 0 else 0.0
    has_audio = check_has_audio(path)
    return VideoInfo(width=width, height=height, fps=fps, frame_count=frame_count, duration_seconds=duration_seconds, has_audio=has_audio, rotation=rotation)

@contextmanager
def _open_capture(path: str) -> Iterator[cv2.VideoCapture]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video file not found: '{path}'")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video file: '{path}'. The file may be corrupt or unreadable.")
    try:
        yield cap
    finally:
        cap.release()

def frame_iter(path: str, rotation: int | None=None) -> Iterator[np.ndarray]:
    if rotation is None:
        rotation = get_video_rotation(path)
    rot_code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(rotation)
    with _open_capture(path) as cap:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if rot_code is not None:
                frame = cv2.rotate(frame, rot_code)
            yield frame

def main() -> None:
    parser = argparse.ArgumentParser(description='Probe a video file and verify actual frame count.')
    parser.add_argument('path', help='Path to the video file')
    args = parser.parse_args()
    video_path = args.path
    print(f'Probing video: {video_path}')
    info = probe(video_path)
    print('\n--- VideoInfo ---')
    print(f'  Width           : {info.width} px')
    print(f'  Height          : {info.height} px')
    print(f'  FPS             : {info.fps:.2f}')
    print(f'  Reported Frames : {info.frame_count}')
    print(f'  Duration (s)    : {info.duration_seconds:.2f} s')
    print(f'  Has Audio       : {info.has_audio}')
    print('\nIterating through frames...')
    actual_count = 0
    for _ in frame_iter(video_path):
        actual_count += 1
    print('\n--- Frame Count Verification ---')
    print(f'  Reported count  : {info.frame_count}')
    print(f'  Actual count    : {actual_count}')
    if actual_count == info.frame_count:
        print('  Match           : YES (Reported frame count matches actual frame count)')
    else:
        diff = actual_count - info.frame_count
        print(f'  Match           : NO (Difference: {diff:+d})')
if __name__ == '__main__':
    main()
