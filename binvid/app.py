from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import subprocess
import tempfile
import time
import uuid
from typing import Any

import cv2
import gradio as gr
import numpy as np

from binvid.config import RenderConfig, parse_color_bgr
from binvid.digits import DigitField, ScrollField
from binvid.environment import find_ffmpeg, find_monospace_font
from binvid.geometry import compute_grid
from binvid.glyphs import build_atlas
from binvid.pipeline import convert
from binvid.probe import probe
from binvid.render import render_frame

_PREVIEW_FRAME_CACHE: dict[str, tuple[float, np.ndarray, int, int]] = {}
TEMP_OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "binvid_outputs")

CUSTOM_CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

:root {
  color-scheme: dark;
  --binvid-accent: #00ff46;
  --binvid-accent-soft: rgba(0, 255, 70, 0.24);
  --binvid-canvas: #050a07;
  --binvid-text: #edfff2;
  --binvid-muted: #b7d9c0;
  --binvid-glass: rgba(9, 24, 15, 0.18);
  --binvid-glass-strong: rgba(10, 29, 18, 0.2);
  --binvid-edge: rgba(237, 255, 242, 0.2);
  --binvid-edge-subtle: rgba(237, 255, 242, 0.12);
  --binvid-shadow: rgba(0, 0, 0, 0.52);
  --binvid-radius: 18px;

  --body-background-fill: transparent !important;
  --background-fill-primary: transparent !important;
  --background-fill-secondary: transparent !important;
  --block-background-fill: transparent !important;
  --block-border-width: 0px !important;
  --block-border-color: transparent !important;
  --panel-border-width: 0px !important;
  --panel-border-color: transparent !important;
  --input-background-fill: rgba(4, 13, 8, 0.34) !important;
  --input-border-color: var(--binvid-edge-subtle) !important;
  --input-border-color-focus: var(--binvid-accent) !important;
  --color-accent: var(--binvid-accent) !important;
  --color-accent-soft: var(--binvid-accent-soft) !important;
  --slider-color: var(--binvid-accent) !important;
  --button-primary-text-color: var(--binvid-text) !important;
  --button-primary-background-fill: var(--binvid-glass-strong) !important;
  --button-primary-background-fill-hover: rgba(0, 255, 70, 0.16) !important;
  --button-primary-border-color: rgba(0, 255, 70, 0.5) !important;
}

* {
  box-sizing: border-box;
}

html,
body {
  min-height: 100dvh !important;
  margin: 0 !important;
  padding: 0 !important;
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  justify-content: flex-start !important;
  color: var(--binvid-text) !important;
  background:
    radial-gradient(60rem 44rem at 8% -10%, rgba(0, 255, 70, 0.17), transparent 58%),
    radial-gradient(44rem 36rem at 103% 20%, rgba(0, 255, 70, 0.1), transparent 60%),
    radial-gradient(34rem 26rem at 44% 105%, rgba(0, 255, 70, 0.08), transparent 66%),
    var(--binvid-canvas) !important;
  font-family: 'Plus Jakarta Sans', ui-sans-serif, system-ui, sans-serif !important;
}

body {
  overflow-x: hidden !important;
}

.gradio-container,
.gradio-container > .main,
.gradio-container .contain,
div[class*="gradio-container"],
#root,
.app,
gradio-app {
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
  margin: 0 auto !important;
  padding: 0 !important;
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  justify-content: flex-start !important;
}

.app-shell {
  width: 100% !important;
  max-width: 100% !important;
  min-height: 100dvh !important;
  margin: 0 auto !important;
  padding: 40px 24px 64px !important;
  position: relative !important;
  isolation: isolate;
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  justify-content: flex-start !important;
}

.editorial-header {
  width: 100% !important;
  max-width: 920px !important;
  margin: 0 auto 32px auto !important;
  text-align: center !important;
}

.editorial-header h1 {
  margin: 0 auto !important;
  text-align: center !important;
  color: var(--binvid-text) !important;
  font-size: clamp(2.2rem, 4.2vw, 3.6rem) !important;
  font-weight: 800 !important;
  line-height: 1.15 !important;
  letter-spacing: -0.04em !important;
  text-wrap: balance;
  text-shadow: 0 0 32px rgba(0, 255, 70, 0.22);
}

/* Common container behind all the boxes combined: centrally aligned, glassmorphic */
.common-container,
div.common-container,
.gradio-container .common-container {
  width: min(100%, 1280px) !important;
  max-width: 1280px !important;
  margin: 0 auto !important;
  padding: 36px 36px 40px !important;
  background:
    radial-gradient(120% 100% at 50% 0%, rgba(0, 255, 70, 0.08) 0%, transparent 60%),
    linear-gradient(180deg, rgba(8, 26, 17, 0.72) 0%, rgba(3, 14, 8, 0.86) 100%) !important;
  backdrop-filter: blur(24px) saturate(140%) !important;
  -webkit-backdrop-filter: blur(24px) saturate(140%) !important;
  border: 1px solid rgba(0, 255, 70, 0.25) !important;
  border-radius: 24px !important;
  box-shadow:
    0 24px 64px rgba(0, 0, 0, 0.75),
    0 0 45px rgba(0, 255, 70, 0.08),
    inset 0 1px 0 rgba(237, 255, 242, 0.14) !important;
  position: relative !important;
  isolation: isolate;
  align-self: center !important;
}

.workspace-grid-row,
.gradio-container .workspace-grid-row {
  gap: 32px !important;
}

/* Remove outer boxes, borders, backgrounds, and shadows from internal containers */
.sub-card,
.drop-zone-card,
.settings-card,
.preview-subcard,
.output-subcard,
.app-shell,
div[class*="sub-card"],
.gradio-container .group,
.gradio-container .form,
.gradio-container .block.col:not(.common-container),
.gradio-container .block.row,
.gradio-container div[class*="col"]:not(.common-container),
.gradio-container [data-testid="column"]:not(.common-container),
.gradio-container [data-testid="group"] {
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
  backdrop-filter: none !important;
  -webkit-backdrop-filter: none !important;
}

.sub-card {
  padding: 0 0 16px 0 !important;
  margin-bottom: 20px !important;
}

.sub-card > .styler,
.accordion-card > .styler,
.gradio-container .group > .styler {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

.card-title {
  display: flex !important;
  align-items: center !important;
  gap: 10px !important;
  min-height: 28px !important;
  margin: 0 0 14px !important;
  color: var(--binvid-text) !important;
  font-size: 1.05rem !important;
  font-weight: 700 !important;
  letter-spacing: -0.015em !important;
}

.card-title::before {
  content: "" !important;
  display: inline-block !important;
  width: 8px !important;
  height: 8px !important;
  border-radius: 50% !important;
  background: var(--binvid-accent) !important;
  box-shadow: 0 0 10px var(--binvid-accent) !important;
}

.video-uploader,
.preview-box,
.result-video {
  overflow: hidden !important;
  background:
    linear-gradient(135deg, rgba(237, 255, 242, 0.08), transparent 44%),
    var(--binvid-glass) !important;
  border: 1px solid var(--binvid-edge) !important;
  border-radius: var(--binvid-radius) !important;
  box-shadow:
    0 16px 40px var(--binvid-shadow),
    inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
  backdrop-filter: blur(14px) saturate(135%) !important;
  -webkit-backdrop-filter: blur(14px) saturate(135%) !important;
}

.video-uploader :is(.upload-container, .empty, [class*="upload"], [class*="empty"]) {
  min-height: 220px;
  padding: 24px 16px !important;
  color: var(--binvid-text) !important;
  background: rgba(3, 13, 8, 0.12) !important;
  border: 1px dashed rgba(237, 255, 242, 0.32) !important;
  border-radius: calc(var(--binvid-radius) - 3px) !important;
  transition: border-color 180ms ease, background-color 180ms ease, box-shadow 180ms ease !important;
}

.video-uploader :is(.upload-container, .empty, [class*="upload"], [class*="empty"]):hover {
  background: rgba(0, 255, 70, 0.08) !important;
  border-color: rgba(0, 255, 70, 0.7) !important;
  box-shadow: inset 0 0 30px rgba(0, 255, 70, 0.1) !important;
}

.video-uploader :is(p, span, label) {
  color: var(--binvid-text) !important;
}

.settings-card :is(.block, [data-testid="block"], [data-testid*="slider"], [data-testid="dropdown"], [data-testid="colorpicker"]),
.settings-card .form,
.accordion-card .form {
  background: transparent !important;
}

.accordion-card {
  margin: 20px 0 24px !important;
  overflow: hidden !important;
  background:
    linear-gradient(135deg, rgba(237, 255, 242, 0.08), transparent 44%),
    rgba(8, 25, 15, 0.16) !important;
}

.accordion-card > button,
.accordion-card summary,
div[data-testid="accordion"] > button {
  min-height: 48px !important;
  padding: 12px 16px !important;
  color: var(--binvid-text) !important;
  background: transparent !important;
  border: 0 !important;
}

.accordion-card > button:hover,
div[data-testid="accordion"] > button:hover {
  background: rgba(0, 255, 70, 0.08) !important;
}

label,
.label,
span[data-testid="block-label"],
.block label {
  color: var(--binvid-text) !important;
  font-size: 0.9rem !important;
  font-weight: 600 !important;
}

.block-info,
span[data-testid="block-info"],
p.info,
.info-text {
  color: var(--binvid-muted) !important;
  font-size: 0.8rem !important;
  line-height: 1.5 !important;
}

/* Slider header alignment and numeric input boxes */
.head,
[class*="head"] {
  align-items: center !important;
}

.tab-like-container,
[class*="tab-like-container"],
.head .tab-like-container {
  height: 32px !important;
  min-height: 32px !important;
  max-height: 32px !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  border-radius: 8px !important;
  border: 1px solid var(--binvid-edge-subtle) !important;
  background: rgba(4, 18, 10, 0.6) !important;
  overflow: hidden !important;
  box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.4) !important;
  vertical-align: middle !important;
}

/* Numeric inputs: numbers centered vertically & horizontally, not sunken */
.tab-like-container input[type="number"],
[class*="tab-like-container"] input[type="number"],
.settings-card .tab-like-container input[type="number"],
.settings-card input[data-testid="number-input"],
.settings-card input[type="number"] {
  height: 32px !important;
  min-height: 32px !important;
  max-height: 32px !important;
  line-height: 32px !important;
  padding: 0 8px !important;
  margin: 0 !important;
  font-size: 0.95rem !important;
  font-weight: 600 !important;
  text-align: center !important;
  display: block !important;
  vertical-align: middle !important;
  box-sizing: border-box !important;
  border: none !important;
  border-radius: 0 !important;
  background: transparent !important;
  color: var(--binvid-text) !important;
  box-shadow: none !important;
  -moz-appearance: textfield !important;
}

.tab-like-container input[type="number"]::-webkit-inner-spin-button,
.tab-like-container input[type="number"]::-webkit-outer-spin-button,
.settings-card input[type="number"]::-webkit-inner-spin-button,
.settings-card input[type="number"]::-webkit-outer-spin-button {
  -webkit-appearance: none !important;
  margin: 0 !important;
}

.tab-like-container .reset-button,
[class*="tab-like-container"] .reset-button,
[class*="reset-button"] {
  height: 32px !important;
  min-height: 32px !important;
  max-height: 32px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  padding: 0 8px !important;
  margin: 0 !important;
  background: transparent !important;
  border: none !important;
  border-left: 1px solid var(--binvid-edge-subtle) !important;
  color: var(--binvid-muted) !important;
  cursor: pointer !important;
  transition: background-color 150ms ease, color 150ms ease !important;
}

.tab-like-container .reset-button:hover,
[class*="tab-like-container"] .reset-button:hover,
[class*="reset-button"]:hover {
  background: rgba(0, 255, 70, 0.15) !important;
  color: var(--binvid-accent) !important;
}

.settings-card :is(input[type="text"], textarea, select),
.settings-card :is(.wrap, .secondary-wrap):not(.tab-like-container):not([data-testid="dropdown"] *) {
  color: var(--binvid-text) !important;
  background-color: rgba(3, 13, 8, 0.25) !important;
  border-color: var(--binvid-edge-subtle) !important;
  border-radius: 12px !important;
}

.settings-card :is(input[type="text"], textarea) {
  min-height: 40px !important;
  padding: 8px 12px !important;
  line-height: 1.4 !important;
}

input[type="range"] {
  height: 6px !important;
  accent-color: var(--binvid-accent) !important;
  cursor: pointer !important;
}

input[type="range"]::-webkit-slider-runnable-track {
  height: 6px !important;
  background: rgba(237, 255, 242, 0.2) !important;
  border-radius: 999px !important;
}

input[type="range"]::-webkit-slider-thumb {
  width: 18px !important;
  height: 18px !important;
  margin-top: -6px !important;
  background: var(--binvid-accent) !important;
  border: 2px solid #dffff0 !important;
  border-radius: 50% !important;
  box-shadow: 0 0 12px rgba(0, 255, 70, 0.62) !important;
}

input[type="range"]::-moz-range-track {
  height: 6px !important;
  background: rgba(237, 255, 242, 0.2) !important;
  border-radius: 999px !important;
}

input[type="range"]::-moz-range-progress {
  height: 6px !important;
  background: var(--binvid-accent) !important;
  border-radius: 999px !important;
}

input[type="range"]::-moz-range-thumb {
  width: 16px !important;
  height: 16px !important;
  background: var(--binvid-accent) !important;
  border: 2px solid #dffff0 !important;
  border-radius: 50% !important;
  box-shadow: 0 0 12px rgba(0, 255, 70, 0.62) !important;
}

.settings-card :is([class*="range"] [class*="fill"], [class*="slider"] [class*="fill"], [class*="slider"] .progress) {
  background: var(--binvid-accent) !important;
  box-shadow: 0 0 10px rgba(0, 255, 70, 0.42) !important;
}

.settings-card :is([role="slider"], [class*="thumb"], [class*="handle"]) {
  background: var(--binvid-accent) !important;
  border-color: #dffff0 !important;
  box-shadow: 0 0 12px rgba(0, 255, 70, 0.62) !important;
}

input[type="checkbox"] {
  width: 18px !important;
  height: 18px !important;
  accent-color: var(--binvid-accent) !important;
}

/* Dropdown component & trigger wrapper styling */
.settings-card [data-testid="dropdown"],
.settings-card div[class*="dropdown"] {
  position: relative !important;
  z-index: 50 !important;
  overflow: visible !important;
}

.settings-card [data-testid="dropdown"] :is(.wrap, .wrap-inner, .secondary-wrap) {
  overflow: visible !important;
  min-height: 44px !important;
  height: 44px !important;
  background-color: rgba(4, 18, 10, 0.7) !important;
  border-radius: 12px !important;
  border: 1px solid var(--binvid-edge-subtle) !important;
  display: flex !important;
  align-items: center !important;
  box-sizing: border-box !important;
  transition: border-color 150ms ease, box-shadow 150ms ease !important;
}

.settings-card [data-testid="dropdown"] :is(.wrap, .wrap-inner):focus-within {
  border-color: var(--binvid-accent) !important;
  box-shadow: 0 0 14px rgba(0, 255, 70, 0.35) !important;
}

.settings-card [role="combobox"],
.settings-card input[role="combobox"],
.settings-card button[role="combobox"],
[data-testid="dropdown"] input[role="combobox"] {
  min-height: 44px !important;
  height: 44px !important;
  line-height: 44px !important;
  color: var(--binvid-text) !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  padding: 0 14px !important;
  background: transparent !important;
  border: none !important;
  cursor: pointer !important;
  display: flex !important;
  align-items: center !important;
}

.settings-card [data-testid="dropdown"] .icon-wrap,
[data-testid="dropdown"] .icon-wrap {
  color: var(--binvid-accent) !important;
  opacity: 0.9 !important;
}

/* Dropdown Popup Menu / Options List: fully visible, opaque, high-contrast, above other layers */
.options,
div[class*="options"],
.gradio-container [role="listbox"],
.gradio-container .option-list,
.gradio-container [data-testid="dropdown"] [class*="options"] {
  z-index: 2147483647 !important;
  color: #edfff2 !important;
  background: #07190e !important;
  border: 1.5px solid rgba(0, 255, 70, 0.55) !important;
  border-radius: 12px !important;
  box-shadow: 0 18px 48px rgba(0, 0, 0, 0.92), 0 0 24px rgba(0, 255, 70, 0.22) !important;
  backdrop-filter: none !important;
  -webkit-backdrop-filter: none !important;
  overflow: hidden !important;
  padding: 6px 0 !important;
}

.gradio-container .option-list {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
}

/* Dropdown Options items */
.item,
[data-testid="dropdown-option"],
li[role="option"],
.gradio-container [role="option"],
.gradio-container [data-testid="dropdown"] li {
  background: transparent !important;
  color: #edfff2 !important;
  font-size: 0.95rem !important;
  font-weight: 500 !important;
  padding: 10px 16px !important;
  cursor: pointer !important;
  display: flex !important;
  align-items: center !important;
  transition: background-color 150ms ease, color 150ms ease !important;
}

.item:hover,
[data-testid="dropdown-option"]:hover,
li[role="option"]:hover,
.gradio-container [role="option"]:hover,
.active,
.active[role="option"],
.item[aria-selected="true"],
li[role="option"][aria-selected="true"],
.gradio-container [role="option"][aria-selected="true"] {
  background: rgba(0, 255, 70, 0.22) !important;
  color: #00ff46 !important;
  font-weight: 700 !important;
}

.item .inner-item,
[data-testid="dropdown-option"] span,
li[role="option"] span {
  color: #00ff46 !important;
  margin-right: 8px !important;
}

.gradio-container :is(button, input, select, textarea, [role="slider"], [role="combobox"]):focus-visible {
  outline: 2px solid var(--binvid-accent) !important;
  outline-offset: 2px !important;
  box-shadow: 0 0 0 4px rgba(0, 255, 70, 0.2) !important;
}

button.primary-btn,
.primary-btn > button,
button.download-btn,
.download-btn > button {
  min-height: 48px !important;
  padding: 12px 20px !important;
  color: var(--binvid-text) !important;
  background:
    linear-gradient(135deg, rgba(237, 255, 242, 0.12), transparent 52%),
    rgba(4, 18, 10, 0.2) !important;
  border: 1px solid rgba(0, 255, 70, 0.42) !important;
  border-radius: 14px !important;
  box-shadow:
    0 12px 28px rgba(0, 0, 0, 0.34),
    inset 0 1px 0 rgba(255, 255, 255, 0.14) !important;
  backdrop-filter: blur(14px) saturate(135%) !important;
  -webkit-backdrop-filter: blur(14px) saturate(135%) !important;
  font-weight: 700 !important;
  letter-spacing: -0.01em !important;
  cursor: pointer !important;
  touch-action: manipulation;
  transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease, background-color 180ms ease !important;
}

button.primary-btn:hover,
.primary-btn > button:hover,
button.download-btn:hover,
.download-btn > button:hover {
  background:
    linear-gradient(135deg, rgba(237, 255, 242, 0.16), transparent 52%),
    rgba(0, 255, 70, 0.16) !important;
  border-color: var(--binvid-accent) !important;
  box-shadow:
    0 14px 34px rgba(0, 0, 0, 0.42),
    0 0 22px rgba(0, 255, 70, 0.28),
    inset 0 1px 0 rgba(255, 255, 255, 0.18) !important;
  transform: translateY(-1px) !important;
}

button.primary-btn:active,
.primary-btn > button:active,
button.download-btn:active,
.download-btn > button:active {
  transform: translateY(0) scale(0.985) !important;
}

button.primary-btn,
.primary-btn > button {
  width: 100% !important;
}

.preview-box :is(img, canvas),
.result-video video {
  border-radius: calc(var(--binvid-radius) - 2px) !important;
}

.info-text-box {
  margin-top: 12px !important;
  color: var(--binvid-muted) !important;
  font-size: 0.875rem !important;
  line-height: 1.55 !important;
}

.info-text-box * {
  color: var(--binvid-muted) !important;
}

.info-text-box strong,
.info-text-box code,
.summary-box strong,
.summary-box code {
  color: var(--binvid-accent) !important;
}

.summary-box {
  margin-top: 16px !important;
  padding: 16px 18px !important;
  color: var(--binvid-text) !important;
  background: rgba(3, 13, 8, 0.13) !important;
  border: 1px solid var(--binvid-edge-subtle) !important;
  border-radius: 14px !important;
  line-height: 1.6 !important;
}

footer,
.footer,
[data-testid="footer"],
div[class*="footer"],
.gradio-container footer,
body > footer {
  display: none !important;
}

@media (max-width: 768px) {
  .app-shell {
    padding: 24px 16px 40px !important;
  }

  .common-container,
  div.common-container,
  .gradio-container .common-container {
    padding: 20px 16px 24px !important;
    border-radius: 18px !important;
  }

  .workspace-grid-row,
  .gradio-container .workspace-grid-row {
    gap: 20px !important;
  }

  .editorial-header {
    margin-bottom: 24px !important;
  }

  .sub-card {
    padding: 0 0 12px 0 !important;
    margin-bottom: 16px !important;
  }

  .card-title {
    margin-bottom: 16px !important;
  }
}

@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    scroll-behavior: auto !important;
    transition-duration: 0.01ms !important;
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
  }
}

@supports not ((-webkit-backdrop-filter: blur(1px)) or (backdrop-filter: blur(1px))) {
  .common-container,
  .accordion-card,
  .video-uploader,
  .preview-box,
  .result-video,
  button.primary-btn,
  .primary-btn > button,
  button.download-btn,
  .download-btn > button {
    background-color: rgba(5, 16, 10, 0.88) !important;
  }
}
"""


def get_temp_output_dir() -> str:
    """Return the output directory path, ensuring it exists."""
    os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)
    return TEMP_OUTPUT_DIR


def cleanup_temp_files(directory: str, max_age_seconds: int = 3600, max_files: int = 20) -> int:
    """Remove video files older than max_age_seconds or exceeding max_files."""
    if not os.path.exists(directory):
        return 0

    now, deleted_count, candidate_files = time.time(), 0, []
    for entry in os.scandir(directory):
        if entry.is_file() and entry.name.lower().endswith((".mp4", ".mkv", ".mov", ".webm", ".avi")):
            try:
                mtime = entry.stat().st_mtime
                if now - mtime > max_age_seconds:
                    os.remove(entry.path)
                    deleted_count += 1
                else:
                    candidate_files.append((entry.path, mtime))
            except OSError:
                pass

    if len(candidate_files) > max_files:
        candidate_files.sort(key=lambda x: x[1])
        for path, _ in candidate_files[: len(candidate_files) - max_files]:
            try:
                os.remove(path)
                deleted_count += 1
            except OSError:
                pass

    return deleted_count


def resolve_video_path(val: Any) -> str | None:
    """Safely extract string filesystem path from any Gradio video payload format."""
    if not val:
        return None
    if isinstance(val, str):
        path = val
    elif isinstance(val, dict):
        path = val.get("path") or val.get("video") or val.get("name")
    elif hasattr(val, "path"):
        path = getattr(val, "path")
    elif isinstance(val, (list, tuple)) and len(val) > 0:
        path = resolve_video_path(val[0])
    else:
        try:
            path = str(val)
        except Exception:
            return None

    if path and isinstance(path, str) and os.path.isfile(path):
        return os.path.abspath(path)
    return None


def extract_preview_frame(video_path: Any) -> tuple[np.ndarray, int, int] | None:
    """Extract and cache the raw BGR frame at 25% of the video duration."""
    resolved = resolve_video_path(video_path)
    if not resolved:
        return None

    try:
        current_mtime = os.path.getmtime(resolved)
    except OSError:
        return None

    if resolved in _PREVIEW_FRAME_CACHE:
        cached_mtime, cached_frame, w, h = _PREVIEW_FRAME_CACHE[resolved]
        if cached_mtime == current_mtime:
            return cached_frame, w, h

    try:
        info = probe(resolved)
    except Exception:
        return None

    cap = cv2.VideoCapture(resolved)
    ret, frame = False, None
    if cap.isOpened():
        target_idx = int(0.25 * info.frame_count) if info.frame_count > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        cap.release()

    if not ret or frame is None:
        try:
            ffmpeg_bin = find_ffmpeg()
            target_time = max(0.0, float(0.25 * info.duration_seconds)) if info.duration_seconds > 0 else 0.0
            cmd = [
                ffmpeg_bin,
                "-ss",
                str(target_time),
                "-i",
                resolved,
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "-",
            ]
            res = subprocess.run(cmd, capture_output=True, check=False)
            if res.returncode == 0 and res.stdout:
                frame = cv2.imdecode(np.frombuffer(res.stdout, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    ret = True
        except Exception:
            pass

    if not ret or frame is None:
        return None

    rot_code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(info.rotation)
    if rot_code is not None:
        frame = cv2.rotate(frame, rot_code)

    h, w = frame.shape[:2]
    _PREVIEW_FRAME_CACHE[resolved] = (current_mtime, frame, w, h)
    return frame, w, h


def render_preview_still(
    video_path: Any,
    cols: int,
    font_size: int,
    gamma: float,
    mode: str,
    digit_mode: str,
    tint_color: str,
    invert: bool,
    bg_color: str,
    contrast_boost: bool,
    scanlines: bool,
    edge_emphasis: bool,
) -> tuple[np.ndarray | None, str]:
    """Render a single still image preview of the 25% frame with current settings."""
    resolved = resolve_video_path(video_path)
    if not resolved:
        return None, "Upload a video to see live preview dimensions."

    extracted = extract_preview_frame(resolved)
    if extracted is None:
        return None, "Unable to read video frames. Ensure file is a valid video format."

    frame, src_w, src_h = extracted
    grid = compute_grid(src_w, src_h, cols=max(10, int(cols)), cell_w=8, cell_h=14)

    try:
        atlas = build_atlas(find_monospace_font(), max(4, int(font_size)), grid.cell_w, grid.cell_h)
    except Exception as exc:
        return None, f"Font error: {exc}"

    field = ScrollField(grid.rows, grid.cols, seed=42) if digit_mode == "scroll" else DigitField(grid.rows, grid.cols, seed=42, refresh=0)
    digits = field.for_frame(0)

    try:
        tint_bgr = parse_color_bgr(tint_color or "#00FF46")
    except Exception:
        tint_bgr = (70, 255, 0)

    try:
        bg_bgr = parse_color_bgr(bg_color or "#000000")
    except Exception:
        bg_bgr = (0, 0, 0)

    rendered_bgr = render_frame(
        bgr_frame=frame,
        grid=grid,
        atlas=atlas,
        digits=digits,
        mode=mode,
        tint_color=tint_bgr,
        bg_color=bg_bgr,
        invert=invert,
        gamma=float(gamma),
        contrast_boost=contrast_boost,
        scanlines=scanlines,
        edge_emphasis=edge_emphasis,
    )

    rendered_rgb = cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2RGB)
    info_text = (
        f"**Grid**: {grid.cols} cols × {grid.rows} rows "
        f"({grid.cols * grid.rows:,} characters) | "
        f"**Output Resolution**: {grid.out_w} × {grid.out_h} px"
    )
    return rendered_rgb, info_text


def run_conversion(
    video_path: Any,
    cols: int,
    font_size: int,
    gamma: float,
    mode: str,
    digit_mode: str,
    tint_color: str,
    invert: bool,
    bg_color: str,
    contrast_boost: bool,
    scanlines: bool,
    edge_emphasis: bool,
    refresh: int,
    crf: int,
    workers: int,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[str, str, Any]:
    """Execute video conversion with progress tracking and return video path, summary, and download button."""
    resolved = resolve_video_path(video_path)
    if not resolved:
        raise gr.Error("Please upload a source video file first.")

    out_dir = get_temp_output_dir()
    cleanup_temp_files(out_dir, max_age_seconds=3600, max_files=20)

    out_name = f"binvid_{int(time.time())}_{uuid.uuid4().hex[:6]}.mp4"
    out_path = os.path.join(out_dir, out_name)

    config = RenderConfig(
        source_path=resolved,
        output_path=out_path,
        cols=int(cols),
        font_size=int(font_size),
        gamma=float(gamma),
        mode=mode,
        digit_mode=digit_mode,
        tint_color=tint_color or "#00FF46",
        invert=bool(invert),
        bg=bg_color or "#000000",
        contrast_boost=bool(contrast_boost),
        scanlines=bool(scanlines),
        edge_emphasis=bool(edge_emphasis),
        digit_refresh=int(refresh),
        crf=int(crf),
        workers=int(workers),
    )

    def on_progress(done: int, total: int | None) -> None:
        if total and total > 0:
            progress(done / total, desc=f"Rendering frame {done}/{total}...")
        else:
            progress(None, desc=f"Rendering frame {done}...")

    progress(0, desc="Initializing conversion...")
    try:
        convert(config=config, progress=True, progress_callback=on_progress)
    except Exception as exc:
        raise gr.Error(f"Conversion failed: {exc}") from exc

    if not os.path.isfile(out_path):
        raise gr.Error("Output file was not created. Check console logs.")

    summary_text = (
        f" **Conversion Complete!**\n\n"
        f"- **Output File**: `{os.path.basename(out_path)}`\n"
        f"- **Grid**: {config.cols} cols | **Mode**: {config.mode}"
    )

    return str(out_path), summary_text, gr.DownloadButton(value=str(out_path), visible=True)


convert_video_gradio = run_conversion


def build_app() -> gr.Blocks:
    """Construct and configure the Gradio Blocks application."""
    with gr.Blocks(title="binvid — Binary ASCII Video Art", fill_width=True) as demo:
        gr.HTML(f"<style>{CUSTOM_CSS}</style>", visible=False)
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(
                """
                <div class="editorial-header">
                  <h1>Transform Video into Binary ASCII Art</h1>
                </div>
                """
            )

            with gr.Column(elem_classes=["common-container"]):
                with gr.Row(elem_classes=["workspace-grid-row"]):
                    with gr.Column(scale=1):
                        with gr.Column(elem_classes=["sub-card", "drop-zone-card"]):
                            gr.HTML('<div class="card-title">Source Video</div>')
                            video_input = gr.Video(label="Source Video", sources=["upload"], elem_classes=["video-uploader"])

                        with gr.Column(elem_classes=["sub-card", "settings-card"]):
                            gr.HTML('<div class="card-title">Grid & Style Settings</div>')
                            with gr.Row():
                                cols_slider = gr.Slider(20, 400, value=200, step=2, label="Grid Columns (cols)", info="Character column density")
                                font_size_slider = gr.Slider(6, 32, value=12, step=1, label="Font Size (px)", info="Rendered glyph size")

                            with gr.Row():
                                mode_dropdown = gr.Dropdown(["color", "mono", "flat"], value="color", label="Render Mode")
                                digit_mode_dropdown = gr.Dropdown(["refresh", "static", "scroll"], value="refresh", label="Digit Mode")

                            with gr.Row():
                                gamma_slider = gr.Slider(1.0, 2.5, value=1.0, step=0.05, label="Gamma Correction")
                                tint_picker = gr.ColorPicker(value="#00FF46", label="Mono / Flat Tint Color")

                            with gr.Accordion("Advanced Styles & Performance", open=False, elem_classes=["accordion-card"]):
                                with gr.Row():
                                    invert_chk = gr.Checkbox(value=False, label="Invert (Dark on Light)")
                                    contrast_chk = gr.Checkbox(value=False, label="Contrast Boost (HistEq)")
                                with gr.Row():
                                    scanlines_chk = gr.Checkbox(value=False, label="CRT Scanlines")
                                    edge_chk = gr.Checkbox(value=False, label="Sobel Edge Emphasis")
                                bg_picker = gr.ColorPicker(value="#000000", label="Canvas Background Color")
                                with gr.Row():
                                    refresh_slider = gr.Slider(0, 60, value=8, step=1, label="Digit Refresh Interval")
                                    crf_slider = gr.Slider(0, 51, value=18, step=1, label="H.264 CRF Quality")
                                    workers_slider = gr.Slider(1, 8, value=2, step=1, label="Multiprocessing Workers")

                            convert_btn = gr.Button("Convert Video", variant="primary", size="lg", elem_classes=["primary-btn"])

                    with gr.Column(scale=1):
                        with gr.Column(elem_classes=["sub-card", "preview-subcard"]):
                            gr.HTML('<div class="card-title">Live Preview (25% Frame Still)</div>')
                            preview_image = gr.Image(label="Live Preview", type="numpy", interactive=False, elem_classes=["preview-box"])
                            info_text = gr.Markdown("Upload a video to see live preview dimensions.", elem_classes=["info-text-box"])

                        with gr.Column(elem_classes=["sub-card", "output-subcard"]):
                            gr.HTML('<div class="card-title">Rendered Output Video</div>')
                            video_out = gr.Video(label="Result Video", interactive=False, autoplay=True, elem_classes=["result-video"])
                            download_btn = gr.DownloadButton(label="Download video", visible=False, elem_classes=["download-btn"])
                            summary_md = gr.Markdown(visible=True, elem_classes=["summary-box"])

        preview_inputs = [
            video_input,
            cols_slider,
            font_size_slider,
            gamma_slider,
            mode_dropdown,
            digit_mode_dropdown,
            tint_picker,
            invert_chk,
            bg_picker,
            contrast_chk,
            scanlines_chk,
            edge_chk,
        ]

        # Bind both upload and change events to guarantee preview renders on any video input
        video_input.upload(fn=render_preview_still, inputs=preview_inputs, outputs=[preview_image, info_text])
        video_input.change(fn=render_preview_still, inputs=preview_inputs, outputs=[preview_image, info_text])
        video_input.clear(
            fn=lambda: (None, "Upload a video to see live preview dimensions."),
            inputs=None,
            outputs=[preview_image, info_text],
        )

        for control in preview_inputs[1:]:
            handler = control.release if hasattr(control, "release") else control.change
            handler(fn=render_preview_still, inputs=preview_inputs, outputs=[preview_image, info_text])

        convert_inputs = preview_inputs + [refresh_slider, crf_slider, workers_slider]
        convert_btn.click(fn=run_conversion, inputs=convert_inputs, outputs=[video_out, summary_md, download_btn])

    return demo


def main() -> None:
    """CLI launcher for the binvid Gradio web interface."""
    mp.freeze_support()
    parser = argparse.ArgumentParser(description="binvid — Interactive Gradio Web Interface")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Port number to bind (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--inbrowser", action="store_true", default=False, help="Automatically open browser on launch")
    args = parser.parse_args()

    demo = build_app()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
