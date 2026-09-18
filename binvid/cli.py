from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from binvid.config import RenderConfig
from binvid.pipeline import convert


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser for binvid."""
    parser = argparse.ArgumentParser(
        prog="binvid",
        description="Convert video into ASCII-style binary art made of 0 and 1 digits.",
    )
    parser.add_argument(
        "input",
        help="Path to the input source video file",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path to the output video file (defaults to <input-stem>_binary.mp4 next to input)",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=200,
        help="Number of character columns across the grid (default: 200)",
    )
    parser.add_argument(
        "--cell-w",
        type=int,
        default=8,
        help="Pixel width per character cell (default: 8)",
    )
    parser.add_argument(
        "--cell-h",
        type=int,
        default=14,
        help="Pixel height per character cell (default: 14)",
    )
    parser.add_argument(
        "--font",
        default=None,
        help="Path to custom monospace TTF/OTF font file (default: auto-detect)",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=12,
        help="Font size in points/pixels (default: 12)",
    )
    parser.add_argument(
        "--mode",
        choices=["color", "mono", "flat"],
        default="color",
        help="Rendering style: color, mono, or flat (default: color)",
    )
    parser.add_argument(
        "--digit-mode",
        choices=["static", "refresh", "scroll"],
        default="refresh",
        help="Digit pattern strategy: static, refresh, or scroll (default: refresh)",
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=8,
        help="Regenerate digit field every N frames; 0 = never (default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible digit placement (default: 42)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="Constant Rate Factor for H.264 video quality [0-51] (default: 18)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for rendering (default: 1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N frames (default: all)",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert rendering: dark digits on a light/colored background",
    )
    parser.add_argument(
        "--bg",
        type=str,
        default="#000000",
        help="Background canvas color hex (#RRGGBB) or RGB (R,G,B) (default: #000000)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma correction on cell brightness (> 0, e.g. 0.7 for dark scenes; default: 1.0)",
    )
    parser.add_argument(
        "--contrast-boost",
        action="store_true",
        help="Per-frame histogram equalization to preserve mid-tones",
    )
    parser.add_argument(
        "--scanlines",
        action="store_true",
        help="Darken alternating output rows slightly (CRT style)",
    )
    parser.add_argument(
        "--edge-emphasis",
        action="store_true",
        help="Sobel edge detection biasing high-gradient cells toward digit '1'",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for binvid."""
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.is_file():
        print(
            f"Error: Input video file not found: '{args.input}'. Please check the path and try again.",
            file=sys.stderr,
        )
        return 1

    # Default output path to <input-stem>_binary.mp4 next to the input
    if args.output is None:
        output_path = str(input_path.with_name(f"{input_path.stem}_binary.mp4"))
    else:
        output_path = args.output

    try:
        config = RenderConfig(
            source_path=str(input_path.resolve()),
            output_path=os.path.abspath(output_path),
            cols=args.cols,
            cell_w=args.cell_w,
            cell_h=args.cell_h,
            font_path=args.font,
            font_size=args.font_size,
            digit_refresh=args.refresh,
            seed=args.seed,
            crf=args.crf,
            mode=args.mode,
            digit_mode=args.digit_mode,
            workers=args.workers,
            limit=args.limit,
            invert=args.invert,
            bg=args.bg,
            gamma=args.gamma,
            contrast_boost=args.contrast_boost,
            scanlines=args.scanlines,
            edge_emphasis=args.edge_emphasis,
        )
    except ValueError as exc:
        print(f"Error: Invalid configuration: {exc}", file=sys.stderr)
        return 1

    try:
        convert(config, progress=not args.no_progress)
        return 0
    except (FileNotFoundError, PermissionError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: An unexpected error occurred: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
