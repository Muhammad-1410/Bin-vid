"""binvid: Convert video into binary ASCII-style art made of 0 and 1 digits."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from binvid.config import RenderConfig
    from binvid.digits import DigitField, ScrollField
    from binvid.encoder import FrameSink
    from binvid.environment import find_ffmpeg, find_monospace_font
    from binvid.geometry import GridSpec, compute_grid, fit_cols_to_width
    from binvid.glyphs import build_atlas, preview_atlas
    from binvid.pipeline import convert
    from binvid.probe import VideoInfo, frame_iter, probe
    from binvid.render import render_frame

__version__ = "0.1.0"

__all__ = [
    "DigitField",
    "FrameSink",
    "GridSpec",
    "RenderConfig",
    "ScrollField",
    "VideoInfo",
    "build_app",
    "build_atlas",
    "compute_grid",
    "convert",
    "find_ffmpeg",
    "find_monospace_font",
    "fit_cols_to_width",
    "frame_iter",
    "preview_atlas",
    "probe",
    "render_frame",
]


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "build_app": ("binvid.app", "build_app"),
    "RenderConfig": ("binvid.config", "RenderConfig"),
    "find_ffmpeg": ("binvid.environment", "find_ffmpeg"),
    "find_monospace_font": ("binvid.environment", "find_monospace_font"),
    "VideoInfo": ("binvid.probe", "VideoInfo"),
    "probe": ("binvid.probe", "probe"),
    "frame_iter": ("binvid.probe", "frame_iter"),
    "GridSpec": ("binvid.geometry", "GridSpec"),
    "compute_grid": ("binvid.geometry", "compute_grid"),
    "fit_cols_to_width": ("binvid.geometry", "fit_cols_to_width"),
    "build_atlas": ("binvid.glyphs", "build_atlas"),
    "preview_atlas": ("binvid.glyphs", "preview_atlas"),
    "DigitField": ("binvid.digits", "DigitField"),
    "ScrollField": ("binvid.digits", "ScrollField"),
    "render_frame": ("binvid.render", "render_frame"),
    "FrameSink": ("binvid.encoder", "FrameSink"),
    "convert": ("binvid.pipeline", "convert"),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        import importlib
        mod_name, attr = _LAZY_EXPORTS[name]
        return getattr(importlib.import_module(mod_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
