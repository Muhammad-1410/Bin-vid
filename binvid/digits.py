from __future__ import annotations
import numpy as np

class DigitField:

    def __init__(self, rows: int, cols: int, seed: int=42, refresh: int=8, partial: bool=False, flip_ratio: float=0.15) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError(f'rows and cols must be positive, got rows={rows}, cols={cols}.')
        if refresh < 0:
            raise ValueError(f'refresh must be non-negative (>= 0), got {refresh}.')
        if not 0.0 <= flip_ratio <= 1.0:
            raise ValueError(f'flip_ratio must be between 0.0 and 1.0, got {flip_ratio}.')
        self.rows = rows
        self.cols = cols
        self.seed = seed
        self.refresh = refresh
        self.partial = partial
        self.flip_ratio = flip_ratio
        self._reset()

    def _reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._current = self._rng.integers(0, 2, size=(self.rows, self.cols), dtype=np.uint8)
        self._current_period = 0

    def for_frame(self, frame_index: int) -> np.ndarray:
        if frame_index < 0:
            raise ValueError(f'frame_index must be non-negative, got {frame_index}.')
        if self.refresh == 0:
            return self._current
        target_period = frame_index // self.refresh
        if target_period == self._current_period:
            return self._current
        if target_period < self._current_period:
            self._reset()
        while self._current_period < target_period:
            self._current_period += 1
            if self.partial:
                flip_mask = self._rng.random(size=(self.rows, self.cols)) < self.flip_ratio
                self._current = self._current.copy()
                self._current[flip_mask] ^= 1
            else:
                self._current = self._rng.integers(0, 2, size=(self.rows, self.cols), dtype=np.uint8)
        return self._current

class ScrollField:

    def __init__(self, rows: int, cols: int, seed: int=42, min_speed: float=0.5, max_speed: float=2.0) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError(f'rows and cols must be positive, got rows={rows}, cols={cols}.')
        if min_speed <= 0 or max_speed < min_speed:
            raise ValueError(f'Invalid speed range: min_speed={min_speed}, max_speed={max_speed}.')
        self.rows = rows
        self.cols = cols
        self.seed = seed
        self.min_speed = min_speed
        self.max_speed = max_speed
        self._rng = np.random.default_rng(seed)
        self._buffer_h = rows * 3
        self._buffer: np.ndarray = self._rng.integers(0, 2, size=(self._buffer_h, cols), dtype=np.uint8)
        self._speeds: np.ndarray = self._rng.uniform(min_speed, max_speed, size=cols)

    def for_frame(self, frame_index: int) -> np.ndarray:
        if frame_index < 0:
            raise ValueError(f'frame_index must be non-negative, got {frame_index}.')
        col_offsets = (frame_index * self._speeds).astype(int) % self._buffer_h
        row_indices = (np.arange(self.rows)[:, None] + col_offsets[None, :]) % self._buffer_h
        col_indices = np.arange(self.cols)[None, :]
        return self._buffer[row_indices, col_indices]
