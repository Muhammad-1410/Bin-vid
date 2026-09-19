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
TEMP_OUTPUT_DIR = os.path.join(tempfile.gettempdir(), 'binvid_outputs')
CUSTOM_CSS = '\n@import url(\'https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap\');\n\n:root {\n  color-scheme: dark;\n  --binvid-accent: #00ff46;\n  --binvid-accent-soft: rgba(0, 255, 70, 0.24);\n  --binvid-canvas: #050a07;\n  --binvid-text: #edfff2;\n  --binvid-muted: #b7d9c0;\n  --binvid-glass: rgba(9, 24, 15, 0.18);\n  --binvid-glass-strong: rgba(10, 29, 18, 0.2);\n  --binvid-edge: rgba(237, 255, 242, 0.2);\n  --binvid-edge-subtle: rgba(237, 255, 242, 0.12);\n  --binvid-shadow: rgba(0, 0, 0, 0.52);\n  --binvid-radius: 18px;\n\n  --body-background-fill: transparent !important;\n  --background-fill-primary: transparent !important;\n  --background-fill-secondary: transparent !important;\n  --block-background-fill: transparent !important;\n  --block-border-width: 0px !important;\n  --block-border-color: transparent !important;\n  --panel-border-width: 0px !important;\n  --panel-border-color: transparent !important;\n  --input-background-fill: rgba(4, 13, 8, 0.34) !important;\n  --input-border-color: var(--binvid-edge-subtle) !important;\n  --input-border-color-focus: var(--binvid-accent) !important;\n  --color-accent: var(--binvid-accent) !important;\n  --color-accent-soft: var(--binvid-accent-soft) !important;\n  --slider-color: var(--binvid-accent) !important;\n  --button-primary-text-color: var(--binvid-text) !important;\n  --button-primary-background-fill: var(--binvid-glass-strong) !important;\n  --button-primary-background-fill-hover: rgba(0, 255, 70, 0.16) !important;\n  --button-primary-border-color: rgba(0, 255, 70, 0.5) !important;\n}\n\n* {\n  box-sizing: border-box;\n}\n\nhtml,\nbody {\n  min-height: 100dvh !important;\n  margin: 0 !important;\n  padding: 0 !important;\n  display: flex !important;\n  flex-direction: column !important;\n  align-items: center !important;\n  justify-content: flex-start !important;\n  color: var(--binvid-text) !important;\n  background:\n    radial-gradient(60rem 44rem at 8% -10%, rgba(0, 255, 70, 0.17), transparent 58%),\n    radial-gradient(44rem 36rem at 103% 20%, rgba(0, 255, 70, 0.1), transparent 60%),\n    radial-gradient(34rem 26rem at 44% 105%, rgba(0, 255, 70, 0.08), transparent 66%),\n    var(--binvid-canvas) !important;\n  font-family: \'Plus Jakarta Sans\', ui-sans-serif, system-ui, sans-serif !important;\n}\n\nbody {\n  overflow-x: hidden !important;\n}\n\n.gradio-container,\n.gradio-container > .main,\n.gradio-container .contain,\ndiv[class*="gradio-container"],\n#root,\n.app,\ngradio-app {\n  width: 100% !important;\n  max-width: 100% !important;\n  min-width: 0 !important;\n  margin: 0 auto !important;\n  padding: 0 !important;\n  display: flex !important;\n  flex-direction: column !important;\n  align-items: center !important;\n  justify-content: flex-start !important;\n}\n\n.app-shell {\n  width: 100% !important;\n  max-width: 100% !important;\n  min-height: 100dvh !important;\n  margin: 0 auto !important;\n  padding: 40px 24px 64px !important;\n  position: relative !important;\n  isolation: isolate;\n  display: flex !important;\n  flex-direction: column !important;\n  align-items: center !important;\n  justify-content: flex-start !important;\n}\n\n.editorial-header {\n  width: 100% !important;\n  max-width: 920px !important;\n  margin: 0 auto 32px auto !important;\n  text-align: center !important;\n}\n\n.editorial-header h1 {\n  margin: 0 auto !important;\n  text-align: center !important;\n  color: var(--binvid-text) !important;\n  font-size: clamp(2.2rem, 4.2vw, 3.6rem) !important;\n  font-weight: 800 !important;\n  line-height: 1.15 !important;\n  letter-spacing: -0.04em !important;\n  text-wrap: balance;\n  text-shadow: 0 0 32px rgba(0, 255, 70, 0.22);\n}\n\n/* Common container behind all the boxes combined: centrally aligned, glassmorphic */\n.common-container,\ndiv.common-container,\n.gradio-container .common-container {\n  width: min(100%, 1280px) !important;\n  max-width: 1280px !important;\n  margin: 0 auto !important;\n  padding: 36px 36px 40px !important;\n  background:\n    radial-gradient(120% 100% at 50% 0%, rgba(0, 255, 70, 0.08) 0%, transparent 60%),\n    linear-gradient(180deg, rgba(8, 26, 17, 0.72) 0%, rgba(3, 14, 8, 0.86) 100%) !important;\n  backdrop-filter: blur(24px) saturate(140%) !important;\n  -webkit-backdrop-filter: blur(24px) saturate(140%) !important;\n  border: 1px solid rgba(0, 255, 70, 0.25) !important;\n  border-radius: 24px !important;\n  box-shadow:\n    0 24px 64px rgba(0, 0, 0, 0.75),\n    0 0 45px rgba(0, 255, 70, 0.08),\n    inset 0 1px 0 rgba(237, 255, 242, 0.14) !important;\n  position: relative !important;\n  isolation: isolate;\n  align-self: center !important;\n}\n\n.workspace-grid-row,\n.gradio-container .workspace-grid-row {\n  gap: 32px !important;\n}\n\n/* Remove outer boxes, borders, backgrounds, and shadows from internal containers */\n.sub-card,\n.drop-zone-card,\n.settings-card,\n.preview-subcard,\n.output-subcard,\n.app-shell,\ndiv[class*="sub-card"],\n.gradio-container .group,\n.gradio-container .form,\n.gradio-container .block.col:not(.common-container),\n.gradio-container .block.row,\n.gradio-container div[class*="col"]:not(.common-container),\n.gradio-container [data-testid="column"]:not(.common-container),\n.gradio-container [data-testid="group"] {\n  border: none !important;\n  box-shadow: none !important;\n  background: transparent !important;\n  backdrop-filter: none !important;\n  -webkit-backdrop-filter: none !important;\n}\n\n.sub-card {\n  padding: 0 0 16px 0 !important;\n  margin-bottom: 20px !important;\n}\n\n.sub-card > .styler,\n.accordion-card > .styler,\n.gradio-container .group > .styler {\n  background: transparent !important;\n  border: none !important;\n  box-shadow: none !important;\n}\n\n.card-title {\n  display: flex !important;\n  align-items: center !important;\n  gap: 10px !important;\n  min-height: 28px !important;\n  margin: 0 0 14px !important;\n  color: var(--binvid-text) !important;\n  font-size: 1.05rem !important;\n  font-weight: 700 !important;\n  letter-spacing: -0.015em !important;\n}\n\n.card-title::before {\n  content: "" !important;\n  display: inline-block !important;\n  width: 8px !important;\n  height: 8px !important;\n  border-radius: 50% !important;\n  background: var(--binvid-accent) !important;\n  box-shadow: 0 0 10px var(--binvid-accent) !important;\n}\n\n.video-uploader,\n.preview-box,\n.result-video {\n  overflow: hidden !important;\n  background:\n    linear-gradient(135deg, rgba(237, 255, 242, 0.08), transparent 44%),\n    var(--binvid-glass) !important;\n  border: 1px solid var(--binvid-edge) !important;\n  border-radius: var(--binvid-radius) !important;\n  box-shadow:\n    0 16px 40px var(--binvid-shadow),\n    inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;\n  backdrop-filter: blur(14px) saturate(135%) !important;\n  -webkit-backdrop-filter: blur(14px) saturate(135%) !important;\n}\n\n.video-uploader :is(.upload-container, .empty, [class*="upload"], [class*="empty"]) {\n  min-height: 220px;\n  padding: 24px 16px !important;\n  color: var(--binvid-text) !important;\n  background: rgba(3, 13, 8, 0.12) !important;\n  border: 1px dashed rgba(237, 255, 242, 0.32) !important;\n  border-radius: calc(var(--binvid-radius) - 3px) !important;\n  transition: border-color 180ms ease, background-color 180ms ease, box-shadow 180ms ease !important;\n}\n\n.video-uploader :is(.upload-container, .empty, [class*="upload"], [class*="empty"]):hover {\n  background: rgba(0, 255, 70, 0.08) !important;\n  border-color: rgba(0, 255, 70, 0.7) !important;\n  box-shadow: inset 0 0 30px rgba(0, 255, 70, 0.1) !important;\n}\n\n.video-uploader :is(p, span, label) {\n  color: var(--binvid-text) !important;\n}\n\n.settings-card :is(.block, [data-testid="block"], [data-testid*="slider"], [data-testid="dropdown"], [data-testid="colorpicker"]),\n.settings-card .form,\n.accordion-card .form {\n  background: transparent !important;\n}\n\n.accordion-card {\n  margin: 20px 0 24px !important;\n  overflow: hidden !important;\n  background:\n    linear-gradient(135deg, rgba(237, 255, 242, 0.08), transparent 44%),\n    rgba(8, 25, 15, 0.16) !important;\n}\n\n.accordion-card > button,\n.accordion-card summary,\ndiv[data-testid="accordion"] > button {\n  min-height: 48px !important;\n  padding: 12px 16px !important;\n  color: var(--binvid-text) !important;\n  background: transparent !important;\n  border: 0 !important;\n}\n\n.accordion-card > button:hover,\ndiv[data-testid="accordion"] > button:hover {\n  background: rgba(0, 255, 70, 0.08) !important;\n}\n\nlabel,\n.label,\nspan[data-testid="block-label"],\n.block label {\n  color: var(--binvid-text) !important;\n  font-size: 0.9rem !important;\n  font-weight: 600 !important;\n}\n\n.block-info,\nspan[data-testid="block-info"],\np.info,\n.info-text {\n  color: var(--binvid-muted) !important;\n  font-size: 0.8rem !important;\n  line-height: 1.5 !important;\n}\n\n/* Slider header alignment and numeric input boxes */\n.head,\n[class*="head"] {\n  align-items: center !important;\n}\n\n.tab-like-container,\n[class*="tab-like-container"],\n.head .tab-like-container {\n  height: 32px !important;\n  min-height: 32px !important;\n  max-height: 32px !important;\n  display: inline-flex !important;\n  align-items: center !important;\n  justify-content: center !important;\n  border-radius: 8px !important;\n  border: 1px solid var(--binvid-edge-subtle) !important;\n  background: rgba(4, 18, 10, 0.6) !important;\n  overflow: hidden !important;\n  box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.4) !important;\n  vertical-align: middle !important;\n}\n\n/* Numeric inputs: numbers centered vertically & horizontally, not sunken */\n.tab-like-container input[type="number"],\n[class*="tab-like-container"] input[type="number"],\n.settings-card .tab-like-container input[type="number"],\n.settings-card input[data-testid="number-input"],\n.settings-card input[type="number"] {\n  height: 32px !important;\n  min-height: 32px !important;\n  max-height: 32px !important;\n  line-height: 32px !important;\n  padding: 0 8px !important;\n  margin: 0 !important;\n  font-size: 0.95rem !important;\n  font-weight: 600 !important;\n  text-align: center !important;\n  display: block !important;\n  vertical-align: middle !important;\n  box-sizing: border-box !important;\n  border: none !important;\n  border-radius: 0 !important;\n  background: transparent !important;\n  color: var(--binvid-text) !important;\n  box-shadow: none !important;\n  -moz-appearance: textfield !important;\n}\n\n.tab-like-container input[type="number"]::-webkit-inner-spin-button,\n.tab-like-container input[type="number"]::-webkit-outer-spin-button,\n.settings-card input[type="number"]::-webkit-inner-spin-button,\n.settings-card input[type="number"]::-webkit-outer-spin-button {\n  -webkit-appearance: none !important;\n  margin: 0 !important;\n}\n\n.tab-like-container .reset-button,\n[class*="tab-like-container"] .reset-button,\n[class*="reset-button"] {\n  height: 32px !important;\n  min-height: 32px !important;\n  max-height: 32px !important;\n  display: flex !important;\n  align-items: center !important;\n  justify-content: center !important;\n  padding: 0 8px !important;\n  margin: 0 !important;\n  background: transparent !important;\n  border: none !important;\n  border-left: 1px solid var(--binvid-edge-subtle) !important;\n  color: var(--binvid-muted) !important;\n  cursor: pointer !important;\n  transition: background-color 150ms ease, color 150ms ease !important;\n}\n\n.tab-like-container .reset-button:hover,\n[class*="tab-like-container"] .reset-button:hover,\n[class*="reset-button"]:hover {\n  background: rgba(0, 255, 70, 0.15) !important;\n  color: var(--binvid-accent) !important;\n}\n\n.settings-card :is(input[type="text"], textarea, select),\n.settings-card :is(.wrap, .secondary-wrap):not(.tab-like-container):not([data-testid="dropdown"] *) {\n  color: var(--binvid-text) !important;\n  background-color: rgba(3, 13, 8, 0.25) !important;\n  border-color: var(--binvid-edge-subtle) !important;\n  border-radius: 12px !important;\n}\n\n.settings-card :is(input[type="text"], textarea) {\n  min-height: 40px !important;\n  padding: 8px 12px !important;\n  line-height: 1.4 !important;\n}\n\ninput[type="range"] {\n  height: 6px !important;\n  accent-color: var(--binvid-accent) !important;\n  cursor: pointer !important;\n}\n\ninput[type="range"]::-webkit-slider-runnable-track {\n  height: 6px !important;\n  background: rgba(237, 255, 242, 0.2) !important;\n  border-radius: 999px !important;\n}\n\ninput[type="range"]::-webkit-slider-thumb {\n  width: 18px !important;\n  height: 18px !important;\n  margin-top: -6px !important;\n  background: var(--binvid-accent) !important;\n  border: 2px solid #dffff0 !important;\n  border-radius: 50% !important;\n  box-shadow: 0 0 12px rgba(0, 255, 70, 0.62) !important;\n}\n\ninput[type="range"]::-moz-range-track {\n  height: 6px !important;\n  background: rgba(237, 255, 242, 0.2) !important;\n  border-radius: 999px !important;\n}\n\ninput[type="range"]::-moz-range-progress {\n  height: 6px !important;\n  background: var(--binvid-accent) !important;\n  border-radius: 999px !important;\n}\n\ninput[type="range"]::-moz-range-thumb {\n  width: 16px !important;\n  height: 16px !important;\n  background: var(--binvid-accent) !important;\n  border: 2px solid #dffff0 !important;\n  border-radius: 50% !important;\n  box-shadow: 0 0 12px rgba(0, 255, 70, 0.62) !important;\n}\n\n.settings-card :is([class*="range"] [class*="fill"], [class*="slider"] [class*="fill"], [class*="slider"] .progress) {\n  background: var(--binvid-accent) !important;\n  box-shadow: 0 0 10px rgba(0, 255, 70, 0.42) !important;\n}\n\n.settings-card :is([role="slider"], [class*="thumb"], [class*="handle"]) {\n  background: var(--binvid-accent) !important;\n  border-color: #dffff0 !important;\n  box-shadow: 0 0 12px rgba(0, 255, 70, 0.62) !important;\n}\n\ninput[type="checkbox"] {\n  width: 18px !important;\n  height: 18px !important;\n  accent-color: var(--binvid-accent) !important;\n}\n\n/* Dropdown component & trigger wrapper styling */\n.settings-card [data-testid="dropdown"],\n.settings-card div[class*="dropdown"] {\n  position: relative !important;\n  z-index: 50 !important;\n  overflow: visible !important;\n}\n\n.settings-card [data-testid="dropdown"] :is(.wrap, .wrap-inner, .secondary-wrap) {\n  overflow: visible !important;\n  min-height: 44px !important;\n  height: 44px !important;\n  background-color: rgba(4, 18, 10, 0.7) !important;\n  border-radius: 12px !important;\n  border: 1px solid var(--binvid-edge-subtle) !important;\n  display: flex !important;\n  align-items: center !important;\n  box-sizing: border-box !important;\n  transition: border-color 150ms ease, box-shadow 150ms ease !important;\n}\n\n.settings-card [data-testid="dropdown"] :is(.wrap, .wrap-inner):focus-within {\n  border-color: var(--binvid-accent) !important;\n  box-shadow: 0 0 14px rgba(0, 255, 70, 0.35) !important;\n}\n\n.settings-card [role="combobox"],\n.settings-card input[role="combobox"],\n.settings-card button[role="combobox"],\n[data-testid="dropdown"] input[role="combobox"] {\n  min-height: 44px !important;\n  height: 44px !important;\n  line-height: 44px !important;\n  color: var(--binvid-text) !important;\n  font-weight: 600 !important;\n  font-size: 0.95rem !important;\n  padding: 0 14px !important;\n  background: transparent !important;\n  border: none !important;\n  cursor: pointer !important;\n  display: flex !important;\n  align-items: center !important;\n}\n\n.settings-card [data-testid="dropdown"] .icon-wrap,\n[data-testid="dropdown"] .icon-wrap {\n  color: var(--binvid-accent) !important;\n  opacity: 0.9 !important;\n}\n\n/* Dropdown Popup Menu / Options List: fully visible, opaque, high-contrast, above other layers */\n.options,\ndiv[class*="options"],\n.gradio-container [role="listbox"],\n.gradio-container .option-list,\n.gradio-container [data-testid="dropdown"] [class*="options"] {\n  z-index: 2147483647 !important;\n  color: #edfff2 !important;\n  background: #07190e !important;\n  border: 1.5px solid rgba(0, 255, 70, 0.55) !important;\n  border-radius: 12px !important;\n  box-shadow: 0 18px 48px rgba(0, 0, 0, 0.92), 0 0 24px rgba(0, 255, 70, 0.22) !important;\n  backdrop-filter: none !important;\n  -webkit-backdrop-filter: none !important;\n  overflow: hidden !important;\n  padding: 6px 0 !important;\n}\n\n.gradio-container .option-list {\n  background: transparent !important;\n  border: none !important;\n  box-shadow: none !important;\n  padding: 0 !important;\n}\n\n/* Dropdown Options items */\n.item,\n[data-testid="dropdown-option"],\nli[role="option"],\n.gradio-container [role="option"],\n.gradio-container [data-testid="dropdown"] li {\n  background: transparent !important;\n  color: #edfff2 !important;\n  font-size: 0.95rem !important;\n  font-weight: 500 !important;\n  padding: 10px 16px !important;\n  cursor: pointer !important;\n  display: flex !important;\n  align-items: center !important;\n  transition: background-color 150ms ease, color 150ms ease !important;\n}\n\n.item:hover,\n[data-testid="dropdown-option"]:hover,\nli[role="option"]:hover,\n.gradio-container [role="option"]:hover,\n.active,\n.active[role="option"],\n.item[aria-selected="true"],\nli[role="option"][aria-selected="true"],\n.gradio-container [role="option"][aria-selected="true"] {\n  background: rgba(0, 255, 70, 0.22) !important;\n  color: #00ff46 !important;\n  font-weight: 700 !important;\n}\n\n.item .inner-item,\n[data-testid="dropdown-option"] span,\nli[role="option"] span {\n  color: #00ff46 !important;\n  margin-right: 8px !important;\n}\n\n.gradio-container :is(button, input, select, textarea, [role="slider"], [role="combobox"]):focus-visible {\n  outline: 2px solid var(--binvid-accent) !important;\n  outline-offset: 2px !important;\n  box-shadow: 0 0 0 4px rgba(0, 255, 70, 0.2) !important;\n}\n\nbutton.primary-btn,\n.primary-btn > button,\nbutton.download-btn,\n.download-btn > button {\n  min-height: 48px !important;\n  padding: 12px 20px !important;\n  color: var(--binvid-text) !important;\n  background:\n    linear-gradient(135deg, rgba(237, 255, 242, 0.12), transparent 52%),\n    rgba(4, 18, 10, 0.2) !important;\n  border: 1px solid rgba(0, 255, 70, 0.42) !important;\n  border-radius: 14px !important;\n  box-shadow:\n    0 12px 28px rgba(0, 0, 0, 0.34),\n    inset 0 1px 0 rgba(255, 255, 255, 0.14) !important;\n  backdrop-filter: blur(14px) saturate(135%) !important;\n  -webkit-backdrop-filter: blur(14px) saturate(135%) !important;\n  font-weight: 700 !important;\n  letter-spacing: -0.01em !important;\n  cursor: pointer !important;\n  touch-action: manipulation;\n  transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease, background-color 180ms ease !important;\n}\n\nbutton.primary-btn:hover,\n.primary-btn > button:hover,\nbutton.download-btn:hover,\n.download-btn > button:hover {\n  background:\n    linear-gradient(135deg, rgba(237, 255, 242, 0.16), transparent 52%),\n    rgba(0, 255, 70, 0.16) !important;\n  border-color: var(--binvid-accent) !important;\n  box-shadow:\n    0 14px 34px rgba(0, 0, 0, 0.42),\n    0 0 22px rgba(0, 255, 70, 0.28),\n    inset 0 1px 0 rgba(255, 255, 255, 0.18) !important;\n  transform: translateY(-1px) !important;\n}\n\nbutton.primary-btn:active,\n.primary-btn > button:active,\nbutton.download-btn:active,\n.download-btn > button:active {\n  transform: translateY(0) scale(0.985) !important;\n}\n\nbutton.primary-btn,\n.primary-btn > button {\n  width: 100% !important;\n}\n\n.preview-box :is(img, canvas),\n.result-video video {\n  border-radius: calc(var(--binvid-radius) - 2px) !important;\n}\n\n.info-text-box {\n  margin-top: 12px !important;\n  color: var(--binvid-muted) !important;\n  font-size: 0.875rem !important;\n  line-height: 1.55 !important;\n}\n\n.info-text-box * {\n  color: var(--binvid-muted) !important;\n}\n\n.info-text-box strong,\n.info-text-box code,\n.summary-box strong,\n.summary-box code {\n  color: var(--binvid-accent) !important;\n}\n\n.summary-box {\n  margin-top: 16px !important;\n  padding: 16px 18px !important;\n  color: var(--binvid-text) !important;\n  background: rgba(3, 13, 8, 0.13) !important;\n  border: 1px solid var(--binvid-edge-subtle) !important;\n  border-radius: 14px !important;\n  line-height: 1.6 !important;\n}\n\nfooter,\n.footer,\n[data-testid="footer"],\ndiv[class*="footer"],\n.gradio-container footer,\nbody > footer {\n  display: none !important;\n}\n\n@media (max-width: 768px) {\n  .app-shell {\n    padding: 24px 16px 40px !important;\n  }\n\n  .common-container,\n  div.common-container,\n  .gradio-container .common-container {\n    padding: 20px 16px 24px !important;\n    border-radius: 18px !important;\n  }\n\n  .workspace-grid-row,\n  .gradio-container .workspace-grid-row {\n    gap: 20px !important;\n  }\n\n  .editorial-header {\n    margin-bottom: 24px !important;\n  }\n\n  .sub-card {\n    padding: 0 0 12px 0 !important;\n    margin-bottom: 16px !important;\n  }\n\n  .card-title {\n    margin-bottom: 16px !important;\n  }\n}\n\n@media (prefers-reduced-motion: reduce) {\n  *,\n  *::before,\n  *::after {\n    scroll-behavior: auto !important;\n    transition-duration: 0.01ms !important;\n    animation-duration: 0.01ms !important;\n    animation-iteration-count: 1 !important;\n  }\n}\n\n@supports not ((-webkit-backdrop-filter: blur(1px)) or (backdrop-filter: blur(1px))) {\n  .common-container,\n  .accordion-card,\n  .video-uploader,\n  .preview-box,\n  .result-video,\n  button.primary-btn,\n  .primary-btn > button,\n  button.download-btn,\n  .download-btn > button {\n    background-color: rgba(5, 16, 10, 0.88) !important;\n  }\n}\n'

def get_temp_output_dir() -> str:
    os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)
    return TEMP_OUTPUT_DIR

def cleanup_temp_files(directory: str, max_age_seconds: int=3600, max_files: int=20) -> int:
    if not os.path.exists(directory):
        return 0
    now, deleted_count, candidate_files = (time.time(), 0, [])
    for entry in os.scandir(directory):
        if entry.is_file() and entry.name.lower().endswith(('.mp4', '.mkv', '.mov', '.webm', '.avi')):
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
        for path, _ in candidate_files[:len(candidate_files) - max_files]:
            try:
                os.remove(path)
                deleted_count += 1
            except OSError:
                pass
    return deleted_count

def resolve_video_path(val: Any) -> str | None:
    if not val:
        return None
    if isinstance(val, str):
        path = val
    elif isinstance(val, dict):
        path = val.get('path') or val.get('video') or val.get('name')
    elif hasattr(val, 'path'):
        path = getattr(val, 'path')
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
            return (cached_frame, w, h)
    try:
        info = probe(resolved)
    except Exception:
        return None
    cap = cv2.VideoCapture(resolved)
    ret, frame = (False, None)
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
            cmd = [ffmpeg_bin, '-ss', str(target_time), '-i', resolved, '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-']
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
    return (frame, w, h)

def render_preview_still(video_path: Any, cols: int, font_size: int, gamma: float, mode: str, digit_mode: str, tint_color: str, invert: bool, bg_color: str, contrast_boost: bool, scanlines: bool, edge_emphasis: bool) -> tuple[np.ndarray | None, str]:
    resolved = resolve_video_path(video_path)
    if not resolved:
        return (None, 'Upload a video to see live preview dimensions.')
    extracted = extract_preview_frame(resolved)
    if extracted is None:
        return (None, 'Unable to read video frames. Ensure file is a valid video format.')
    frame, src_w, src_h = extracted
    grid = compute_grid(src_w, src_h, cols=max(10, int(cols)), cell_w=8, cell_h=14)
    try:
        atlas = build_atlas(find_monospace_font(), max(4, int(font_size)), grid.cell_w, grid.cell_h)
    except Exception as exc:
        return (None, f'Font error: {exc}')
    field = ScrollField(grid.rows, grid.cols, seed=42) if digit_mode == 'scroll' else DigitField(grid.rows, grid.cols, seed=42, refresh=0)
    digits = field.for_frame(0)
    try:
        tint_bgr = parse_color_bgr(tint_color or '#00FF46')
    except Exception:
        tint_bgr = (70, 255, 0)
    try:
        bg_bgr = parse_color_bgr(bg_color or '#000000')
    except Exception:
        bg_bgr = (0, 0, 0)
    rendered_bgr = render_frame(bgr_frame=frame, grid=grid, atlas=atlas, digits=digits, mode=mode, tint_color=tint_bgr, bg_color=bg_bgr, invert=invert, gamma=float(gamma), contrast_boost=contrast_boost, scanlines=scanlines, edge_emphasis=edge_emphasis)
    rendered_rgb = cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2RGB)
    info_text = f'**Grid**: {grid.cols} cols × {grid.rows} rows ({grid.cols * grid.rows:,} characters) | **Output Resolution**: {grid.out_w} × {grid.out_h} px'
    return (rendered_rgb, info_text)

def run_conversion(video_path: Any, cols: int, font_size: int, gamma: float, mode: str, digit_mode: str, tint_color: str, invert: bool, bg_color: str, contrast_boost: bool, scanlines: bool, edge_emphasis: bool, refresh: int, crf: int, workers: int, progress: gr.Progress=gr.Progress(track_tqdm=True)) -> tuple[str, str, Any]:
    resolved = resolve_video_path(video_path)
    if not resolved:
        raise gr.Error('Please upload a source video file first.')
    out_dir = get_temp_output_dir()
    cleanup_temp_files(out_dir, max_age_seconds=3600, max_files=20)
    out_name = f'binvid_{int(time.time())}_{uuid.uuid4().hex[:6]}.mp4'
    out_path = os.path.join(out_dir, out_name)
    config = RenderConfig(source_path=resolved, output_path=out_path, cols=int(cols), font_size=int(font_size), gamma=float(gamma), mode=mode, digit_mode=digit_mode, tint_color=tint_color or '#00FF46', invert=bool(invert), bg=bg_color or '#000000', contrast_boost=bool(contrast_boost), scanlines=bool(scanlines), edge_emphasis=bool(edge_emphasis), digit_refresh=int(refresh), crf=int(crf), workers=int(workers))

    def on_progress(done: int, total: int | None) -> None:
        if total and total > 0:
            progress(done / total, desc=f'Rendering frame {done}/{total}...')
        else:
            progress(None, desc=f'Rendering frame {done}...')
    progress(0, desc='Initializing conversion...')
    try:
        convert(config=config, progress=True, progress_callback=on_progress)
    except Exception as exc:
        raise gr.Error(f'Conversion failed: {exc}') from exc
    if not os.path.isfile(out_path):
        raise gr.Error('Output file was not created. Check console logs.')
    summary_text = f' **Conversion Complete!**\n\n- **Output File**: `{os.path.basename(out_path)}`\n- **Grid**: {config.cols} cols | **Mode**: {config.mode}'
    return (str(out_path), summary_text, gr.DownloadButton(value=str(out_path), visible=True))
convert_video_gradio = run_conversion

def build_app() -> gr.Blocks:
    with gr.Blocks(title='binvid — Binary ASCII Video Art', fill_width=True) as demo:
        gr.HTML(f'<style>{CUSTOM_CSS}</style>', visible=False)
        with gr.Column(elem_classes=['app-shell']):
            gr.HTML('\n                <div class="editorial-header">\n                  <h1>Transform Video into Binary ASCII Art</h1>\n                </div>\n                ')
            with gr.Column(elem_classes=['common-container']):
                with gr.Row(elem_classes=['workspace-grid-row']):
                    with gr.Column(scale=1):
                        with gr.Column(elem_classes=['sub-card', 'drop-zone-card']):
                            gr.HTML('<div class="card-title">Source Video</div>')
                            video_input = gr.Video(label='Source Video', sources=['upload'], elem_classes=['video-uploader'])
                        with gr.Column(elem_classes=['sub-card', 'settings-card']):
                            gr.HTML('<div class="card-title">Grid & Style Settings</div>')
                            with gr.Row():
                                cols_slider = gr.Slider(20, 400, value=200, step=2, label='Grid Columns (cols)', info='Character column density')
                                font_size_slider = gr.Slider(6, 32, value=12, step=1, label='Font Size (px)', info='Rendered glyph size')
                            with gr.Row():
                                mode_dropdown = gr.Dropdown(['color', 'mono', 'flat'], value='color', label='Render Mode')
                                digit_mode_dropdown = gr.Dropdown(['refresh', 'static', 'scroll'], value='refresh', label='Digit Mode')
                            with gr.Row():
                                gamma_slider = gr.Slider(1.0, 2.5, value=1.0, step=0.05, label='Gamma Correction')
                                tint_picker = gr.ColorPicker(value='#00FF46', label='Mono / Flat Tint Color')
                            with gr.Accordion('Advanced Styles & Performance', open=False, elem_classes=['accordion-card']):
                                with gr.Row():
                                    invert_chk = gr.Checkbox(value=False, label='Invert (Dark on Light)')
                                    contrast_chk = gr.Checkbox(value=False, label='Contrast Boost (HistEq)')
                                with gr.Row():
                                    scanlines_chk = gr.Checkbox(value=False, label='CRT Scanlines')
                                    edge_chk = gr.Checkbox(value=False, label='Sobel Edge Emphasis')
                                bg_picker = gr.ColorPicker(value='#000000', label='Canvas Background Color')
                                with gr.Row():
                                    refresh_slider = gr.Slider(0, 60, value=8, step=1, label='Digit Refresh Interval')
                                    crf_slider = gr.Slider(0, 51, value=18, step=1, label='H.264 CRF Quality')
                                    workers_slider = gr.Slider(1, 8, value=2, step=1, label='Multiprocessing Workers')
                            convert_btn = gr.Button('Convert Video', variant='primary', size='lg', elem_classes=['primary-btn'])
                    with gr.Column(scale=1):
                        with gr.Column(elem_classes=['sub-card', 'preview-subcard']):
                            gr.HTML('<div class="card-title">Live Preview (25% Frame Still)</div>')
                            preview_image = gr.Image(label='Live Preview', type='numpy', interactive=False, elem_classes=['preview-box'])
                            info_text = gr.Markdown('Upload a video to see live preview dimensions.', elem_classes=['info-text-box'])
                        with gr.Column(elem_classes=['sub-card', 'output-subcard']):
                            gr.HTML('<div class="card-title">Rendered Output Video</div>')
                            video_out = gr.Video(label='Result Video', interactive=False, autoplay=True, elem_classes=['result-video'])
                            download_btn = gr.DownloadButton(label='Download video', visible=False, elem_classes=['download-btn'])
                            summary_md = gr.Markdown(visible=True, elem_classes=['summary-box'])
        preview_inputs = [video_input, cols_slider, font_size_slider, gamma_slider, mode_dropdown, digit_mode_dropdown, tint_picker, invert_chk, bg_picker, contrast_chk, scanlines_chk, edge_chk]
        video_input.upload(fn=render_preview_still, inputs=preview_inputs, outputs=[preview_image, info_text])
        video_input.change(fn=render_preview_still, inputs=preview_inputs, outputs=[preview_image, info_text])
        video_input.clear(fn=lambda: (None, 'Upload a video to see live preview dimensions.'), inputs=None, outputs=[preview_image, info_text])
        for control in preview_inputs[1:]:
            handler = control.release if hasattr(control, 'release') else control.change
            handler(fn=render_preview_still, inputs=preview_inputs, outputs=[preview_image, info_text])
        convert_inputs = preview_inputs + [refresh_slider, crf_slider, workers_slider]
        convert_btn.click(fn=run_conversion, inputs=convert_inputs, outputs=[video_out, summary_md, download_btn])
    return demo

def main() -> None:
    mp.freeze_support()
    parser = argparse.ArgumentParser(description='binvid — Interactive Gradio Web Interface')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host address to bind (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=7860, help='Port number to bind (default: 7860)')
    parser.add_argument('--share', action='store_true', help='Create a public Gradio share link')
    parser.add_argument('--inbrowser', action='store_true', default=False, help='Automatically open browser on launch')
    args = parser.parse_args()
    demo = build_app()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, inbrowser=args.inbrowser, css=CUSTOM_CSS)
if __name__ == '__main__':
    mp.freeze_support()
    main()
