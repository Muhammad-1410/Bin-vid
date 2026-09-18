from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RenderConfig:
    source_path: str
    output_path: str
    cols: int = 200
    cell_w: int = 8
    cell_h: int = 14
    font_path: str | None = None
    font_size: int = 12
    digit_refresh: int = 8  # regenerate digit field every N frames; 0 = never
    seed: int = 42
    crf: int = 18
    mode: str = "color"  # "color", "mono", "flat"
    digit_mode: str = "refresh"  # "static", "refresh", "scroll"
    workers: int = 1  # number of worker processes for rendering (>= 1)
    limit: int | None = None  # process only first N frames (None = all)
    invert: bool = False  # dark digits on light background
    bg: str = "#000000"  # background color hex (#RRGGBB) or RGB (R,G,B)
    gamma: float = 1.0  # gamma correction on cell brightness (> 0, e.g. 0.7 for dark scenes)
    contrast_boost: bool = False  # per-frame histogram equalization of downsampled brightness
    scanlines: bool = False  # darken every other output row slightly (CRT style)
    edge_emphasis: bool = False  # Sobel edge filter biasing high-gradient cells toward '1'
    tint_color: str = "#00FF46"  # tint color hex (#RRGGBB) or RGB (R,G,B) for mono/flat modes

    def get_bg_bgr(self) -> tuple[int, int, int]:
        """Return the background color as a (B, G, R) integer tuple in [0, 255]."""
        return parse_color_bgr(self.bg)

    def get_tint_bgr(self) -> tuple[int, int, int]:
        """Return the tint color as a (B, G, R) integer tuple in [0, 255]."""
        return parse_color_bgr(self.tint_color)

    def validate(self) -> None:
        """Validate configuration settings, raising ValueError on invalid parameters."""
        if not self.source_path or not str(self.source_path).strip():
            raise ValueError("source_path must not be empty.")

        if not self.output_path or not str(self.output_path).strip():
            raise ValueError("output_path must not be empty.")

        if self.cols < 10:
            raise ValueError(f"cols must be at least 10, got {self.cols}.")

        if self.cell_w < 4:
            raise ValueError(f"cell_w must be at least 4, got {self.cell_w}.")

        if self.cell_h < 4:
            raise ValueError(f"cell_h must be at least 4, got {self.cell_h}.")

        if self.font_size < 1:
            raise ValueError(f"font_size must be at least 1, got {self.font_size}.")

        if self.digit_refresh < 0:
            raise ValueError(f"digit_refresh must be non-negative (>= 0), got {self.digit_refresh}.")

        if not (0 <= self.crf <= 51):
            raise ValueError(f"crf must be between 0 and 51, got {self.crf}.")

        if self.font_path is not None and not os.path.isfile(self.font_path):
            raise ValueError(f"font_path specified does not exist or is not a file: {self.font_path}")

        if self.mode not in ("color", "mono", "flat"):
            raise ValueError(f"mode must be one of 'color', 'mono', 'flat', got '{self.mode}'.")

        if self.digit_mode not in ("static", "refresh", "scroll"):
            raise ValueError(
                f"digit_mode must be one of 'static', 'refresh', 'scroll', got '{self.digit_mode}'."
            )

        if self.workers < 1:
            raise ValueError(f"workers must be at least 1, got {self.workers}.")

        if self.limit is not None and self.limit <= 0:
            raise ValueError(f"limit must be greater than 0 if specified, got {self.limit}.")

        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive (> 0), got {self.gamma}.")

        # Validate background and tint color formats
        parse_color_bgr(self.bg)
        parse_color_bgr(self.tint_color)


def parse_color_bgr(color_str: str) -> tuple[int, int, int]:
    """Parse a hex (#RRGGBB / RRGGBB) or comma-separated RGB (R,G,B) string into BGR."""
    c = color_str.strip()
    if c.startswith("#"):
        c = c[1:]

    # Check for hex format (6 hex characters)
    if len(c) == 6:
        try:
            r = int(c[0:2], 16)
            g = int(c[2:4], 16)
            b = int(c[4:6], 16)
            return (b, g, r)
        except ValueError:
            pass

    # Check for comma-separated R,G,B format
    if "," in c:
        parts = c.split(",")
        if len(parts) == 3:
            try:
                r = int(parts[0].strip())
                g = int(parts[1].strip())
                b = int(parts[2].strip())
                if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
                    return (b, g, r)
            except ValueError:
                pass

    raise ValueError(
        f"Invalid background color '{color_str}'. Expected hex (#RRGGBB) or comma-separated RGB (R,G,B)."
    )
