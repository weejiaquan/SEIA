from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from ctypes import wintypes
from typing import Any, Callable, Dict, Optional, Tuple

import ctypes

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.mapper import (
    find_window,
    get_client_origin_and_size,
    is_window_minimized,
    set_window_client_size,
    set_window_topmost,
)


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    config_path = path or DEFAULT_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_config_with_fallback(primary: Optional[str], fallback: Optional[str] = None) -> Dict[str, Any]:
    if primary and os.path.exists(primary):
        return load_config(primary)
    return load_config(fallback or DEFAULT_CONFIG_PATH)


def get_capture_method(config: Dict[str, Any]) -> str:
    capture_cfg = config.get("capture", {})
    method = str(capture_cfg.get("method", "auto")).lower()
    if method not in {"auto", "screen", "window", "window_raise"}:
        return "auto"
    return method


def get_ref_size(config: Dict[str, Any]) -> Tuple[int, int]:
    ref = config.get("reference_resolution", {})
    return int(ref.get("w", 1920)), int(ref.get("h", 1080))


def compute_render_rect(
    client_origin: Tuple[int, int],
    client_size: Tuple[int, int],
    ref_size: Tuple[int, int],
    render_cfg: Dict[str, Any],
) -> Tuple[int, int, int, int]:
    mode = str(render_cfg.get("mode", "stretch")).lower()
    client_w, client_h = client_size
    ref_w, ref_h = ref_size
    if mode == "manual":
        offset = render_cfg.get("offset", {})
        size = render_cfg.get("size", {})
        width = int(size.get("w", 0)) or client_w
        height = int(size.get("h", 0)) or client_h
        off_x = int(offset.get("x", 0))
        off_y = int(offset.get("y", 0))
        return client_origin[0] + off_x, client_origin[1] + off_y, width, height
    if mode == "fit":
        if ref_w <= 0 or ref_h <= 0 or client_w <= 0 or client_h <= 0:
            return client_origin[0], client_origin[1], client_w, client_h
        scale = min(client_w / ref_w, client_h / ref_h)
        render_w = int(round(ref_w * scale))
        render_h = int(round(ref_h * scale))
        off_x = int(round((client_w - render_w) / 2))
        off_y = int(round((client_h - render_h) / 2))
        return client_origin[0] + off_x, client_origin[1] + off_y, render_w, render_h
    return client_origin[0], client_origin[1], client_w, client_h


def get_render_context(
    config: Dict[str, Any],
) -> Tuple[int, Tuple[int, int], Tuple[int, int, int, int], float, float]:
    target = config.get("target", {})
    title_substring = target.get("window_title_substring", "")
    process_name = target.get("process_name", "") or None
    hwnd = find_window(title_substring, process_name)
    if hwnd is None:
        raise RuntimeError("Target window not found.")
    window_cfg = config.get("window", {})
    if bool(window_cfg.get("force_topmost", False)):
        set_window_topmost(hwnd, True)
    ref_size = get_ref_size(config)
    if bool(window_cfg.get("auto_resize", False)):
        _apply_startup_resize(hwnd, ref_size)
    origin_x, origin_y, client_w, client_h = get_client_origin_and_size(hwnd)
    render_cfg = config.get("render_area", {})
    render_rect = compute_render_rect((origin_x, origin_y), (client_w, client_h), ref_size, render_cfg)
    render_w, render_h = render_rect[2], render_rect[3]
    scale_x = render_w / ref_size[0] if ref_size[0] else 1.0
    scale_y = render_h / ref_size[1] if ref_size[1] else 1.0
    return hwnd, ref_size, render_rect, scale_x, scale_y


def _apply_startup_resize(hwnd: int, ref_size: Tuple[int, int]) -> None:
    if is_window_minimized(hwnd):
        logging.info("Window is minimized; skipping auto-resize.")
        return
    try:
        _origin_x, _origin_y, client_w, client_h = get_client_origin_and_size(hwnd)
    except Exception as exc:
        logging.warning("Failed to read window metrics for resize: %s", exc)
        return
    ref_w, ref_h = ref_size
    if ref_w <= 0 or ref_h <= 0:
        return
    if client_w == ref_w and client_h == ref_h:
        return
    if set_window_client_size(hwnd, ref_w, ref_h):
        logging.info("Auto-resized window client to %dx%d", ref_w, ref_h)
    else:
        logging.warning("Auto-resize failed to set client size %dx%d", ref_w, ref_h)


def get_cursor_pos() -> Tuple[int, int]:
    point = wintypes.POINT()
    if ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
        return int(point.x), int(point.y)
    return 0, 0


def start_mouse_poller(
    on_click: Callable[[], None],
    debounce_s: float = 0.3,
    interval_s: float = 0.05,
) -> threading.Thread:
    last_click = {"time": 0.0}

    def poll() -> None:
        while True:
            time.sleep(interval_s)
            if ctypes.windll.user32.GetAsyncKeyState(0x01) & 0x8000:
                now = time.time()
                if now - last_click["time"] > debounce_s:
                    last_click["time"] = now
                    on_click()

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    return thread
