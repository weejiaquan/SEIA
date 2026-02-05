from __future__ import annotations

import os
import sys
import ctypes
import json
import logging
import queue
import time
import msvcrt
from ctypes import wintypes
from typing import Any, Dict, Optional, Tuple

import cv2
import keyboard
import numpy as np
import pydirectinput

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from engine.capture import ScreenCapture
from engine.detector import FeatureDetector, RoiDeltaDetector, TemplateDetector, parse_pixel_points, pixel_points_check
from engine.mapper import (
    find_window,
    get_client_origin_and_size,
    is_window_minimized,
    map_point,
    map_rect,
    set_process_dpi_awareness,
    set_window_client_size,
    set_window_topmost,
)


ROOT_DIR = PARENT_DIR
CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")
CLI_DEBOUNCE_S = 0.5
RELOAD_HOTKEY = "ctrl+f4"
MONITOR_HOTKEY = "ctrl+f8"
EXIT_HOTKEY = "ctrl+f10"
DEFAULT_TRACK_MOUSE_HOTKEY = "ctrl+shift+f5"
MOUSE_TRACK_INTERVAL_S = 0.2


def _sample_config() -> Dict[str, Any]:
    return {
        "poll_interval_ms": 50,
        "target": {
            "window_title_substring": "Blue Archive",
            "process_name": "",
        },
        "reference_resolution": {"w": 1920, "h": 1080},
        "render_area": {
            "mode": "fit",
            "offset": {"x": 0, "y": 0},
            "size": {"w": 0, "h": 0},
        },
        "rois": {
            "marker_roi": {"x": 860, "y": 500, "w": 200, "h": 100},
            "change_roi": {"x": 20, "y": 20, "w": 300, "h": 200},
        },
        "marker_detection": {
            "method": "template",
            "template_path": "templates/marker.png",
            "threshold": 0.85,
            "min_score_delta": 0.0,
            "bypass_threshold": False,
            "pixel_points": [
                {"x": 900, "y": 520, "rgb": [255, 255, 255], "tolerance": 10}
            ],
        },
        "screen_change_detection": {
            "method": "roi_delta",
            "delta_threshold": 0.01,
            "smoothing": 8,
            "pixel_threshold": 20,
            "mask_image": None,
        },
        "debug": {
            "save_frames": False,
            "save_on_toggle_only": True,
            "save_every_n": 10,
            "output_dir": "debug_out",
            "overlay": True,
        },
        "test_input": {
            "enabled": True,
            "hotkeys": {
                "click_point": "ctrl+f6",
                "press_esc": "ctrl+f7",
                "scroll_up": "ctrl+z",
                "scroll_down": "ctrl+x",
                "set_scroll_amount": "ctrl+shift+f2",
                "drag": "ctrl+shift+d",
                "set_drag_start": "ctrl+alt+1",
                "set_drag_end": "ctrl+alt+2",
                "set_marker_near": "ctrl+f1",
                "click_marker": "ctrl+f9",
                "click_mouse": "ctrl+f11",
                "auto_click": "z",
                "auto_esc": "x",
                "log_mouse": "ctrl+f5",
                "track_mouse": "ctrl+shift+f5",
                "set_marker_offset": "ctrl+f3"
            },
            "click_point": {"x": 960, "y": 540},
            "scroll_amount": 500,
            "scroll_events": 1,
            "auto_click_interval_s": 3.0,
            "auto_esc_interval_s": 3.0,
            "global_key_poll": False,
            "drag_start": {"x": 960, "y": 540},
            "drag_end": {"x": 960, "y": 740},
            "drag_hold_s": 0.0,
            "drag_duration_s": 0.25,
            "drag_button": "left",
            "drag_use_ref": True,
            "marker_click_offset": {"x": 0, "y": 0},
            "marker_near": {"x": 0, "y": 0},
            "marker_radius": 0
        },
    }


def load_or_create_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        config = _sample_config()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        logging.info("Created sample config: %s", path)
        return config
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(ROOT_DIR, path))


def ensure_templates_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        logging.info("Created templates directory: %s", path)


def _get_ref_size(config: Dict[str, Any]) -> Tuple[int, int]:
    ref = config.get("reference_resolution", {})
    return int(ref.get("w", 1920)), int(ref.get("h", 1080))


def _get_capture_method(config: Dict[str, Any]) -> str:
    capture_cfg = config.get("capture", {})
    method = str(capture_cfg.get("method", "auto")).lower()
    if method not in {"auto", "screen", "window", "window_raise"}:
        return "auto"
    return method


def _get_force_topmost(config: Dict[str, Any]) -> bool:
    window_cfg = config.get("window", {})
    return bool(window_cfg.get("force_topmost", False))


def _get_auto_resize(config: Dict[str, Any]) -> bool:
    window_cfg = config.get("window", {})
    return bool(window_cfg.get("auto_resize", False))


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


def _get_roi(config: Dict[str, Any], key: str) -> Tuple[int, int, int, int]:
    roi = config.get("rois", {}).get(key, {})
    return int(roi.get("x", 0)), int(roi.get("y", 0)), int(roi.get("w", 0)), int(roi.get("h", 0))


def _compute_render_rect(
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


def _status_line(
    origin: Tuple[int, int],
    size: Tuple[int, int],
    render_size: Tuple[int, int],
    scale_x: float,
    scale_y: float,
    marker_present: bool,
    marker_score: float,
    change_score: float,
    change_detected: bool,
    fps: float,
    marker_bypass_threshold: bool,
) -> str:
    scale_text = f"{scale_x:.3f}" if abs(scale_x - scale_y) < 0.001 else f"{scale_x:.3f}/{scale_y:.3f}"
    render_text = ""
    if render_size != size:
        render_text = f" Render {render_size[0]}x{render_size[1]}"
    bypass_text = f"Byp {int(marker_bypass_threshold)}"
    return (
        f"Origin {origin[0]},{origin[1]} Client {size[0]}x{size[1]}{render_text} Scale {scale_text} | "
        f"Marker {int(marker_present)} {marker_score:.2f} | "
        f"Change {change_score:.2f} {int(change_detected)} | "
        f"{bypass_text} | "
        f"{fps:.1f} fps"
    )




def _get_cursor_pos() -> Tuple[int, int]:
    point = wintypes.POINT()
    if ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
        return int(point.x), int(point.y)
    return 0, 0


def _scroll_wheel(delta: int) -> None:
    if delta == 0:
        return
    ctypes.windll.user32.mouse_event(0x0800, 0, 0, int(delta), 0)


def _vk_from_key_name(key: str) -> Optional[int]:
    cleaned = key.strip().lower()
    if not cleaned or "+" in cleaned:
        return None
    if len(cleaned) == 1 and cleaned.isalnum():
        return ord(cleaned.upper())
    if cleaned.startswith("f") and cleaned[1:].isdigit():
        number = int(cleaned[1:])
        if 1 <= number <= 24:
            return 0x70 + (number - 1)
    aliases = {
        "esc": 0x1B,
        "escape": 0x1B,
        "space": 0x20,
        "tab": 0x09,
        "enter": 0x0D,
    }
    return aliases.get(cleaned)


def _key_is_down(vk: int) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(int(vk)) & 0x8000)


def _parse_hotkey(hotkey: str) -> Optional[Tuple[set[str], int]]:
    cleaned = hotkey.strip().lower().replace(" ", "")
    if not cleaned:
        return None
    parts = [part for part in cleaned.split("+") if part]
    if not parts:
        return None
    modifiers: set[str] = set()
    key_name: Optional[str] = None
    for part in parts:
        if part in {"ctrl", "control"}:
            modifiers.add("ctrl")
        elif part == "shift":
            modifiers.add("shift")
        elif part == "alt":
            modifiers.add("alt")
        elif part in {"win", "windows", "cmd", "command"}:
            modifiers.add("win")
        else:
            if key_name is not None:
                return None
            key_name = part
    if key_name is None:
        return None
    vk = _vk_from_key_name(key_name)
    if vk is None:
        return None
    return modifiers, vk


def _modifiers_down(modifiers: set[str]) -> bool:
    if "ctrl" in modifiers and not _key_is_down(0x11):
        return False
    if "shift" in modifiers and not _key_is_down(0x10):
        return False
    if "alt" in modifiers and not _key_is_down(0x12):
        return False
    if "win" in modifiers and not (_key_is_down(0x5B) or _key_is_down(0x5C)):
        return False
    return True


def _parse_offset_text(text: str) -> Optional[Tuple[int, int]]:
    cleaned = text.strip().replace("(", "").replace(")", "")
    if not cleaned:
        return None
    for sep in [",", " "]:
        if sep in cleaned:
            parts = [p for p in cleaned.replace(",", " ").split() if p]
            break
    else:
        parts = [cleaned]
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _parse_int(text: str) -> Optional[int]:
    cleaned = text.strip()
    if not cleaned:
        return None
    try:
        return int(cleaned)
    except ValueError:
        return None


def _near_to_roi(
    near: Optional[Tuple[int, int]],
    radius: int,
    ref_size: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    if near is None or radius <= 0:
        return None
    ref_w, ref_h = ref_size
    use_radius = max(1, int(radius))
    x = max(0, int(near[0]) - use_radius)
    y = max(0, int(near[1]) - use_radius)
    w = min(ref_w - x, use_radius * 2)
    h = min(ref_h - y, use_radius * 2)
    return (x, y, w, h)


def _save_debug_frame(
    capture: ScreenCapture,
    hwnd: int,
    capture_method: str,
    output_dir: str,
    frame_index: int,
    client_origin: Tuple[int, int],
    client_size: Tuple[int, int],
    render_rect: Tuple[int, int, int, int],
    marker_rect: Tuple[int, int, int, int],
    change_rect: Tuple[int, int, int, int],
    overlay: bool,
    marker_score: float,
    change_score: float,
    marker_present: bool,
    change_detected: bool,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    full_rect = (client_origin[0], client_origin[1], client_size[0], client_size[1])
    frame = capture.grab_auto(full_rect, hwnd, capture_method)
    if frame is None:
        return
    if overlay:
        r_left = render_rect[0] - client_origin[0]
        r_top = render_rect[1] - client_origin[1]
        r_w = render_rect[2]
        r_h = render_rect[3]
        m_left = marker_rect[0] - client_origin[0]
        m_top = marker_rect[1] - client_origin[1]
        m_w = marker_rect[2]
        m_h = marker_rect[3]
        c_left = change_rect[0] - client_origin[0]
        c_top = change_rect[1] - client_origin[1]
        c_w = change_rect[2]
        c_h = change_rect[3]
        if r_w > 0 and r_h > 0 and (r_w != client_size[0] or r_h != client_size[1]):
            cv2.rectangle(frame, (r_left, r_top), (r_left + r_w, r_top + r_h), (255, 0, 0), 2)
        cv2.rectangle(frame, (m_left, m_top), (m_left + m_w, m_top + m_h), (0, 255, 0), 2)
        cv2.rectangle(frame, (c_left, c_top), (c_left + c_w, c_top + c_h), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"Marker {marker_score:.2f} {int(marker_present)}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Change {change_score:.2f} {int(change_detected)}",
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    filename = f"frame_{frame_index:06d}_m{int(marker_present)}_c{int(change_detected)}.png"
    path = os.path.join(output_dir, filename)
    try:
        cv2.imwrite(path, frame)
    except Exception as exc:
        logging.warning("Failed to save debug frame: %s", exc)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    set_process_dpi_awareness()

    config = load_or_create_config(CONFIG_PATH)
    ensure_templates_dir(os.path.join(ROOT_DIR, "templates"))

    capture_method = _get_capture_method(config)
    force_topmost = _get_force_topmost(config)
    auto_resize = _get_auto_resize(config)
    poll_interval_ms = int(config.get("poll_interval_ms", 50))
    target = config.get("target", {})
    title_substring = target.get("window_title_substring", "")
    process_name = target.get("process_name", "") or None

    hwnd = find_window(title_substring, process_name)
    if hwnd is None:
        logging.error("Target window not found for title substring: %s", title_substring)
        return
    ref_size = _get_ref_size(config)
    if force_topmost:
        set_window_topmost(hwnd, True)
    if auto_resize:
        _apply_startup_resize(hwnd, ref_size)
    marker_roi_ref = _get_roi(config, "marker_roi")
    change_roi_ref = _get_roi(config, "change_roi")
    render_cfg = config.get("render_area", {})

    marker_cfg = config.get("marker_detection", {})
    marker_method = marker_cfg.get("method", "template")
    marker_threshold = float(marker_cfg.get("threshold", 0.85))
    marker_min_score_delta = float(marker_cfg.get("min_score_delta", 0.0))
    marker_bypass_threshold = bool(marker_cfg.get("bypass_threshold", False))
    template_path = _resolve_path(marker_cfg.get("template_path", "templates/marker.png"))
    if marker_method == "feature":
        marker_detector: TemplateDetector | FeatureDetector = FeatureDetector(template_path)
    else:
        marker_detector = TemplateDetector(template_path)
    pixel_points = parse_pixel_points(marker_cfg.get("pixel_points"))

    change_cfg = config.get("screen_change_detection", {})
    change_method = change_cfg.get("method", "roi_delta")
    change_delta_threshold = float(change_cfg.get("delta_threshold", 0.01))
    change_smoothing = int(change_cfg.get("smoothing", 0) or 0)
    change_pixel_threshold = int(change_cfg.get("pixel_threshold", 20))
    change_mask_image = change_cfg.get("mask_image") or None
    change_mask: np.ndarray | None = None
    if change_mask_image:
        mask_path = _resolve_path(change_mask_image)
        raw_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if raw_mask is not None and raw_mask.ndim == 3 and raw_mask.shape[2] == 4:
            change_mask = (raw_mask[:, :, 3] > 0).astype(np.uint8) * 255
            logging.info("Loaded change detection mask: %s", mask_path)
        else:
            logging.warning("Change detection mask_image has no alpha or unreadable: %s", mask_path)
    delta_detector = RoiDeltaDetector(change_delta_threshold, change_smoothing, change_pixel_threshold, change_mask)
    if change_method != "roi_delta":
        logging.error("Unknown screen_change_detection method: %s", change_method)

    debug_cfg = config.get("debug", {})
    debug_save_frames = bool(debug_cfg.get("save_frames", False))
    debug_save_on_toggle_only = bool(debug_cfg.get("save_on_toggle_only", True))
    debug_save_every_n = int(debug_cfg.get("save_every_n", 10))
    debug_output_dir = debug_cfg.get("output_dir", "debug_out")
    debug_overlay = bool(debug_cfg.get("overlay", True))

    test_input_cfg = config.get("test_input", {})
    test_enabled = bool(test_input_cfg.get("enabled", True))
    test_hotkeys = test_input_cfg.get("hotkeys", {})
    test_click_hotkey = test_hotkeys.get("click_point", "ctrl+f6")
    test_esc_hotkey = test_hotkeys.get("press_esc", "ctrl+f7")
    test_scroll_up_hotkey = test_hotkeys.get("scroll_up", "ctrl+z")
    test_scroll_hotkey = test_hotkeys.get("scroll_down", "ctrl+x")
    test_set_scroll_amount_hotkey = test_hotkeys.get("set_scroll_amount", "ctrl+shift+f2")
    test_drag_hotkey = test_hotkeys.get("drag", "ctrl+shift+d")
    test_set_drag_start_hotkey = test_hotkeys.get("set_drag_start", "ctrl+alt+1")
    test_set_drag_end_hotkey = test_hotkeys.get("set_drag_end", "ctrl+alt+2")
    test_set_marker_near_hotkey = test_hotkeys.get("set_marker_near", "ctrl+f1")
    test_click_marker_hotkey = test_hotkeys.get("click_marker", "ctrl+f9")
    test_click_mouse_hotkey = test_hotkeys.get("click_mouse", "ctrl+f11")
    test_auto_click_hotkey = test_hotkeys.get("auto_click", "z")
    test_auto_esc_hotkey = test_hotkeys.get("auto_esc", "x")
    global_key_poll = bool(test_input_cfg.get("global_key_poll", False))
    test_log_mouse_hotkey = test_hotkeys.get("log_mouse", "ctrl+f5")
    test_track_mouse_hotkey = test_hotkeys.get("track_mouse", DEFAULT_TRACK_MOUSE_HOTKEY)
    test_set_marker_offset_hotkey = test_hotkeys.get("set_marker_offset", "ctrl+f3")
    test_click_point = test_input_cfg.get("click_point", {"x": 0, "y": 0})
    scroll_amount = int(test_input_cfg.get("scroll_amount", 500))
    scroll_events = max(1, int(test_input_cfg.get("scroll_events", 1)))
    auto_click_interval_s = max(0.1, float(test_input_cfg.get("auto_click_interval_s", 3.0)))
    auto_esc_interval_s = max(0.1, float(test_input_cfg.get("auto_esc_interval_s", 3.0)))
    drag_start_cfg = test_input_cfg.get("drag_start", {"x": 0, "y": 0})
    drag_end_cfg = test_input_cfg.get("drag_end", {"x": 0, "y": 0})
    drag_start = (int(drag_start_cfg.get("x", 0)), int(drag_start_cfg.get("y", 0)))
    drag_end = (int(drag_end_cfg.get("x", 0)), int(drag_end_cfg.get("y", 0)))
    drag_hold_s = max(0.0, float(test_input_cfg.get("drag_hold_s", 0.0)))
    drag_duration_s = max(0.0, float(test_input_cfg.get("drag_duration_s", 0.25)))
    drag_button = str(test_input_cfg.get("drag_button", "left") or "left")
    drag_use_ref = bool(test_input_cfg.get("drag_use_ref", True))
    marker_click_offset_cfg = test_input_cfg.get("marker_click_offset", {"x": 0, "y": 0})
    marker_click_offset = (int(marker_click_offset_cfg.get("x", 0)), int(marker_click_offset_cfg.get("y", 0)))
    marker_near_cfg = test_input_cfg.get("marker_near", {"x": 0, "y": 0})
    marker_near = (
        int(marker_near_cfg.get("x", 0)),
        int(marker_near_cfg.get("y", 0)),
    )
    marker_radius = int(test_input_cfg.get("marker_radius", 0))
    auto_click_enabled = False
    auto_click_next = 0.0
    auto_esc_enabled = False
    auto_esc_next = 0.0
    global_hotkeys: Dict[str, Tuple[set[str], int]] = {}
    global_hotkey_state: Dict[str, bool] = {}

    state = {"monitoring": False, "running": True}
    mouse_state = {"track": False, "last_report": 0.0}
    last_size_warned = False
    last_minimized_warned = False

    capture = ScreenCapture()
    action_queue: queue.SimpleQueue[str] = queue.SimpleQueue()

    def log_mouse_position(prefix: str = "Mouse") -> None:
        x, y = _get_cursor_pos()
        if is_window_minimized(hwnd):
            logging.warning("Window is minimized; mouse log skipped.")
            return
        try:
            origin_x, origin_y, client_w, client_h = get_client_origin_and_size(hwnd)
        except Exception as exc:
            logging.error("Failed to read window metrics: %s", exc)
            return
        render_x, render_y, render_w, render_h = _compute_render_rect(
            (origin_x, origin_y), (client_w, client_h), ref_size, render_cfg
        )
        scale_x = render_w / ref_size[0] if ref_size[0] else 1.0
        scale_y = render_h / ref_size[1] if ref_size[1] else 1.0
        ref_x = int(round((x - render_x) / scale_x)) if scale_x else 0
        ref_y = int(round((y - render_y) / scale_y)) if scale_y else 0
        
        # Sample pixel color at cursor position
        try:
            pixel_img = capture.grab_auto((x, y, 1, 1), hwnd, capture_method)
            if pixel_img is not None and pixel_img.size > 0:
                b, g, r = pixel_img[0, 0]
                rgb = (int(r), int(g), int(b))
                hex_color = f"#{r:02X}{g:02X}{b:02X}"
                
                # ANSI color escape for terminal visualization
                ansi_bg = f"\033[48;2;{r};{g};{b}m"
                ansi_reset = "\033[0m"
                color_block = f"{ansi_bg}    {ansi_reset}"
                
                logging.info(
                    "%s screen %d,%d -> ref %d,%d | RGB%s %s %s (render origin %d,%d, scale %.3f/%.3f)",
                    prefix,
                    x,
                    y,
                    ref_x,
                    ref_y,
                    rgb,
                    hex_color,
                    color_block,
                    render_x,
                    render_y,
                    scale_x,
                    scale_y,
                )
            else:
                logging.info(
                    "%s screen %d,%d -> ref %d,%d (render origin %d,%d, scale %.3f/%.3f)",
                    prefix,
                    x,
                    y,
                    ref_x,
                    ref_y,
                    render_x,
                    render_y,
                    scale_x,
                    scale_y,
                )
        except Exception:
            logging.info(
                "%s screen %d,%d -> ref %d,%d (render origin %d,%d, scale %.3f/%.3f)",
                prefix,
                x,
                y,
                ref_x,
                ref_y,
                render_x,
                render_y,
                scale_x,
                scale_y,
            )

    def apply_config(new_config: Dict[str, Any]) -> None:
        nonlocal poll_interval_ms
        nonlocal ref_size, marker_roi_ref, change_roi_ref, render_cfg
        nonlocal marker_method, marker_threshold, marker_min_score_delta, marker_bypass_threshold, template_path, pixel_points, marker_detector
        nonlocal change_method, change_delta_threshold, change_smoothing, change_pixel_threshold, change_mask_image, change_mask, delta_detector
        nonlocal debug_save_frames, debug_save_on_toggle_only, debug_save_every_n, debug_output_dir, debug_overlay
        nonlocal test_enabled, test_click_point, scroll_amount, scroll_events
        nonlocal auto_click_interval_s, auto_click_enabled, auto_click_next
        nonlocal auto_esc_interval_s, auto_esc_enabled, auto_esc_next
        nonlocal global_key_poll, global_hotkeys, global_hotkey_state
        nonlocal drag_start, drag_end, drag_hold_s, drag_duration_s, drag_button, drag_use_ref
        nonlocal marker_click_offset, marker_near, marker_radius
        nonlocal auto_resize

        poll_interval_ms = int(new_config.get("poll_interval_ms", poll_interval_ms))
        ref_size = _get_ref_size(new_config)
        marker_roi_ref = _get_roi(new_config, "marker_roi")
        change_roi_ref = _get_roi(new_config, "change_roi")
        render_cfg = new_config.get("render_area", render_cfg)
        auto_resize = _get_auto_resize(new_config)

        marker_cfg = new_config.get("marker_detection", {})
        marker_method = marker_cfg.get("method", marker_method)
        marker_threshold = float(marker_cfg.get("threshold", marker_threshold))
        marker_min_score_delta = float(marker_cfg.get("min_score_delta", marker_min_score_delta))
        marker_bypass_threshold = bool(marker_cfg.get("bypass_threshold", marker_bypass_threshold))
        template_path = _resolve_path(marker_cfg.get("template_path", template_path))
        if marker_method == "feature":
            if not isinstance(marker_detector, FeatureDetector):
                marker_detector = FeatureDetector(template_path)
            else:
                marker_detector.reload(template_path)
        else:
            if not isinstance(marker_detector, TemplateDetector):
                marker_detector = TemplateDetector(template_path)
            else:
                marker_detector.reload(template_path)
        pixel_points = parse_pixel_points(marker_cfg.get("pixel_points"))

        change_cfg = new_config.get("screen_change_detection", {})
        change_method = change_cfg.get("method", change_method)
        change_delta_threshold = float(change_cfg.get("delta_threshold", change_delta_threshold))
        change_smoothing = int(change_cfg.get("smoothing", change_smoothing) or 0)
        change_pixel_threshold = int(change_cfg.get("pixel_threshold", change_pixel_threshold))
        new_mask_image = change_cfg.get("mask_image") or None
        change_mask = None
        if new_mask_image:
            mask_path = _resolve_path(new_mask_image)
            raw_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if raw_mask is not None and raw_mask.ndim == 3 and raw_mask.shape[2] == 4:
                change_mask = (raw_mask[:, :, 3] > 0).astype(np.uint8) * 255
                logging.info("Loaded change detection mask: %s", mask_path)
            else:
                logging.warning("Change detection mask_image has no alpha or unreadable: %s", mask_path)
        change_mask_image = new_mask_image
        delta_detector = RoiDeltaDetector(change_delta_threshold, change_smoothing, change_pixel_threshold, change_mask)
        if change_method != "roi_delta":
            logging.error("Unknown screen_change_detection method: %s", change_method)

        debug_cfg = new_config.get("debug", {})
        debug_save_frames = bool(debug_cfg.get("save_frames", debug_save_frames))
        debug_save_on_toggle_only = bool(debug_cfg.get("save_on_toggle_only", debug_save_on_toggle_only))
        debug_save_every_n = int(debug_cfg.get("save_every_n", debug_save_every_n))
        debug_output_dir = debug_cfg.get("output_dir", debug_output_dir)
        debug_overlay = bool(debug_cfg.get("overlay", debug_overlay))

        test_input_cfg = new_config.get("test_input", {})
        test_enabled = bool(test_input_cfg.get("enabled", test_enabled))
        new_global_key_poll = bool(test_input_cfg.get("global_key_poll", global_key_poll))
        if new_global_key_poll != global_key_poll:
            logging.info(
                "global_key_poll change requires restart; keeping %s",
                "ON" if global_key_poll else "OFF",
            )
        test_click_point = test_input_cfg.get("click_point", test_click_point)
        scroll_amount = int(test_input_cfg.get("scroll_amount", scroll_amount))
        scroll_events = max(1, int(test_input_cfg.get("scroll_events", scroll_events)))
        auto_click_interval_s = max(0.1, float(test_input_cfg.get("auto_click_interval_s", auto_click_interval_s)))
        auto_esc_interval_s = max(0.1, float(test_input_cfg.get("auto_esc_interval_s", auto_esc_interval_s)))
        drag_start_cfg = test_input_cfg.get("drag_start", {"x": drag_start[0], "y": drag_start[1]})
        drag_end_cfg = test_input_cfg.get("drag_end", {"x": drag_end[0], "y": drag_end[1]})
        drag_start = (
            int(drag_start_cfg.get("x", drag_start[0])),
            int(drag_start_cfg.get("y", drag_start[1])),
        )
        drag_end = (
            int(drag_end_cfg.get("x", drag_end[0])),
            int(drag_end_cfg.get("y", drag_end[1])),
        )
        drag_hold_s = max(0.0, float(test_input_cfg.get("drag_hold_s", drag_hold_s)))
        drag_duration_s = max(0.0, float(test_input_cfg.get("drag_duration_s", drag_duration_s)))
        drag_button = str(test_input_cfg.get("drag_button", drag_button) or drag_button)
        drag_use_ref = bool(test_input_cfg.get("drag_use_ref", drag_use_ref))
        marker_click_offset_cfg = test_input_cfg.get("marker_click_offset", {"x": 0, "y": 0})
        marker_click_offset = (
            int(marker_click_offset_cfg.get("x", marker_click_offset[0])),
            int(marker_click_offset_cfg.get("y", marker_click_offset[1])),
        )
        near_x = marker_near[0] if marker_near is not None else 0
        near_y = marker_near[1] if marker_near is not None else 0
        marker_near_cfg = test_input_cfg.get("marker_near", {"x": near_x, "y": near_y})
        marker_near = (
            int(marker_near_cfg.get("x", near_x)),
            int(marker_near_cfg.get("y", near_y)),
        )
        marker_radius = int(test_input_cfg.get("marker_radius", marker_radius))
        if not test_enabled and auto_click_enabled:
            auto_click_enabled = False
            auto_click_next = 0.0
            logging.info("Auto-click OFF (test input disabled)")
        if not test_enabled and auto_esc_enabled:
            auto_esc_enabled = False
            auto_esc_next = 0.0
            logging.info("Auto-ESC OFF (test input disabled)")
        _build_global_hotkeys()

    def _build_global_hotkeys() -> None:
        nonlocal global_hotkeys, global_hotkey_state
        global_hotkeys = {}
        if not global_key_poll:
            global_hotkey_state = {}
            return

        def register(action: str, hotkey: str) -> None:
            parsed = _parse_hotkey(hotkey)
            if parsed is None:
                logging.warning("Global key poll: unsupported hotkey '%s' for %s", hotkey, action)
                return
            global_hotkeys[action] = parsed

        register("reload_config", RELOAD_HOTKEY)
        register("toggle_monitoring_hotkey", MONITOR_HOTKEY)
        register("exit", EXIT_HOTKEY)

        if test_enabled:
            register("click_point", test_click_hotkey)
            register("press_esc", test_esc_hotkey)
            register("scroll_up", test_scroll_up_hotkey)
            register("scroll_down", test_scroll_hotkey)
            register("set_scroll_amount", test_set_scroll_amount_hotkey)
            register("drag", test_drag_hotkey)
            register("set_drag_start", test_set_drag_start_hotkey)
            register("set_drag_end", test_set_drag_end_hotkey)
            register("set_marker_near", test_set_marker_near_hotkey)
            register("click_marker", test_click_marker_hotkey)
            register("click_mouse", test_click_mouse_hotkey)
            register("toggle_auto_click", test_auto_click_hotkey)
            register("toggle_auto_esc", test_auto_esc_hotkey)
            register("log_mouse", test_log_mouse_hotkey)
            register("toggle_track_mouse", test_track_mouse_hotkey)
            register("set_marker_offset", test_set_marker_offset_hotkey)

        global_hotkey_state = {name: False for name in global_hotkeys}

    def reload_config() -> None:
        try:
            new_config = load_or_create_config(CONFIG_PATH)
            apply_config(new_config)
            nonlocal_vars["last_marker_present"] = None
            nonlocal_vars["last_change_detected"] = None
            if auto_resize:
                _apply_startup_resize(hwnd, ref_size)
            logging.info("Reloaded config and template from %s", CONFIG_PATH)
        except Exception as exc:
            logging.error("Failed to reload config: %s", exc)

    def toggle_monitoring(reason: str) -> None:
        state["monitoring"] = not state["monitoring"]
        status = "ON" if state["monitoring"] else "OFF"
        logging.info("Monitoring %s (source=%s)", status, reason)
        if state["monitoring"]:
            delta_detector.reset()
            nonlocal_vars["frame_index"] = 0
            nonlocal_vars["last_marker_present"] = None
            nonlocal_vars["last_change_detected"] = None

    def stop() -> None:
        state["running"] = False

    def toggle_bypass() -> None:
        nonlocal marker_bypass_threshold
        marker_bypass_threshold = not marker_bypass_threshold
        logging.info("Marker threshold bypass %s", "ON" if marker_bypass_threshold else "OFF")

    def enqueue_action(action: str) -> None:
        action_queue.put(action)

    def do_test_click() -> None:
        if test_enabled:
            enqueue_action("click_point")

    def do_test_esc() -> None:
        if test_enabled:
            enqueue_action("press_esc")

    def do_scroll_down() -> None:
        if test_enabled:
            enqueue_action("scroll_down")

    def do_scroll_up() -> None:
        if test_enabled:
            enqueue_action("scroll_up")

    def do_drag() -> None:
        if test_enabled:
            enqueue_action("drag")

    def do_set_drag_start() -> None:
        if test_enabled:
            enqueue_action("set_drag_start")

    def do_set_drag_end() -> None:
        if test_enabled:
            enqueue_action("set_drag_end")

    def do_set_scroll_amount() -> None:
        if test_enabled:
            enqueue_action("set_scroll_amount")

    def do_set_marker_near() -> None:
        if test_enabled:
            enqueue_action("set_marker_near")

    def do_click_marker() -> None:
        if test_enabled:
            enqueue_action("click_marker")

    def do_click_mouse() -> None:
        if test_enabled:
            enqueue_action("click_mouse")

    def do_toggle_auto_click() -> None:
        if test_enabled:
            enqueue_action("toggle_auto_click")

    def do_toggle_auto_esc() -> None:
        if test_enabled:
            enqueue_action("toggle_auto_esc")

    def do_log_mouse() -> None:
        if test_enabled:
            enqueue_action("log_mouse")

    def do_toggle_track_mouse() -> None:
        if test_enabled:
            enqueue_action("toggle_track_mouse")

    def do_set_marker_offset() -> None:
        if test_enabled:
            enqueue_action("set_marker_offset")

    def do_reload_config() -> None:
        enqueue_action("reload_config")

    def do_toggle_monitoring() -> None:
        enqueue_action("toggle_monitoring_hotkey")

    last_cli_action: Dict[str, float] = {}

    def _cli_allowed(key: str) -> bool:
        now = time.monotonic()
        last = last_cli_action.get(key, 0.0)
        if now - last < CLI_DEBOUNCE_S:
            return False
        last_cli_action[key] = now
        return True

    def handle_cli_keys() -> None:
        keys: list[str] = []
        while msvcrt.kbhit():
            ch = msvcrt.getwch()
            if not ch:
                continue
            if ch in ("\x00", "\xe0"):
                if msvcrt.kbhit():
                    msvcrt.getwch()
                continue
            keys.append(ch.lower())
        if not keys:
            return
        if "h" in keys and _cli_allowed("h"):
            enqueue_action("show_help")
            keys = [key for key in keys if key not in {"a", "b", "c", "h"}]
        for key in keys:
            if key == "a" and _cli_allowed("a"):
                enqueue_action("toggle_monitoring_cli")
            elif key == "b" and _cli_allowed("b"):
                enqueue_action("toggle_bypass")
            elif key == "c" and _cli_allowed("c"):
                enqueue_action("reload_config")

    def log_help() -> None:
        logging.info("CLI keys:")
        logging.info("  A  toggle monitoring")
        logging.info("  B  toggle bypass threshold")
        logging.info("  C  reload config/template")
        logging.info("  H  help")
        logging.info("Hotkeys:")
        logging.info("  %s  reload config/template", RELOAD_HOTKEY)
        logging.info("  %s  toggle monitoring", MONITOR_HOTKEY)
        logging.info("  %s  exit", EXIT_HOTKEY)
        logging.info("Test hotkeys:")
        logging.info("  %s  click point", test_click_hotkey)
        logging.info("  %s  press ESC", test_esc_hotkey)
        logging.info("  %s  scroll up", test_scroll_up_hotkey)
        logging.info("  %s  scroll down", test_scroll_hotkey)
        logging.info("  %s  set scroll amount", test_set_scroll_amount_hotkey)
        logging.info("  %s  drag", test_drag_hotkey)
        logging.info("  %s  set drag start (use mouse or input)", test_set_drag_start_hotkey)
        logging.info("  %s  set drag end (use mouse or input)", test_set_drag_end_hotkey)
        logging.info("  %s  set marker near", test_set_marker_near_hotkey)
        logging.info("  %s  click marker", test_click_marker_hotkey)
        logging.info("  %s  click mouse", test_click_mouse_hotkey)
        logging.info("  %s  toggle auto-click (every %.2fs)", test_auto_click_hotkey, auto_click_interval_s)
        logging.info("  %s  toggle auto-ESC (every %.2fs)", test_auto_esc_hotkey, auto_esc_interval_s)
        logging.info("  %s  log mouse", test_log_mouse_hotkey)
        logging.info("  %s  toggle mouse tracking", test_track_mouse_hotkey)
        logging.info("  %s  set marker offset", test_set_marker_offset_hotkey)
        logging.info("Scroll amount current: %d events %d", scroll_amount, scroll_events)
        drag_space = "ref" if drag_use_ref else "screen"
        logging.info(
            "Drag current: %d,%d -> %d,%d %s hold %.2fs duration %.2fs button %s",
            drag_start[0],
            drag_start[1],
            drag_end[0],
            drag_end[1],
            drag_space,
            drag_hold_s,
            drag_duration_s,
            drag_button,
        )
        logging.info(
            "Marker offset current: %d,%d",
            marker_click_offset[0],
            marker_click_offset[1],
        )
        marker_near_label = "none" if marker_near is None else f"{marker_near[0]},{marker_near[1]}"
        logging.info("Marker near current: %s radius %d", marker_near_label, marker_radius)
        logging.info("Global key poll %s (all hotkeys)", "ON" if global_key_poll else "OFF")
        logging.info("Monitoring currently %s", "ON" if state["monitoring"] else "OFF")

    nonlocal_vars = {
        "frame_index": 0,
        "last_marker_present": None,
        "last_change_detected": None,
    }

    _build_global_hotkeys()

    if not global_key_poll or "reload_config" not in global_hotkeys:
        keyboard.add_hotkey(RELOAD_HOTKEY, do_reload_config)
    if not global_key_poll or "toggle_monitoring_hotkey" not in global_hotkeys:
        keyboard.add_hotkey(MONITOR_HOTKEY, do_toggle_monitoring)
    if not global_key_poll or "exit" not in global_hotkeys:
        keyboard.add_hotkey(EXIT_HOTKEY, stop)
    if test_enabled:
        if not global_key_poll or "click_point" not in global_hotkeys:
            keyboard.add_hotkey(test_click_hotkey, do_test_click)
        if not global_key_poll or "press_esc" not in global_hotkeys:
            keyboard.add_hotkey(test_esc_hotkey, do_test_esc)
        if not global_key_poll or "scroll_up" not in global_hotkeys:
            keyboard.add_hotkey(test_scroll_up_hotkey, do_scroll_up)
        if not global_key_poll or "scroll_down" not in global_hotkeys:
            keyboard.add_hotkey(test_scroll_hotkey, do_scroll_down)
        if not global_key_poll or "set_scroll_amount" not in global_hotkeys:
            keyboard.add_hotkey(test_set_scroll_amount_hotkey, do_set_scroll_amount)
        if not global_key_poll or "drag" not in global_hotkeys:
            keyboard.add_hotkey(test_drag_hotkey, do_drag)
        if not global_key_poll or "set_drag_start" not in global_hotkeys:
            keyboard.add_hotkey(test_set_drag_start_hotkey, do_set_drag_start)
        if not global_key_poll or "set_drag_end" not in global_hotkeys:
            keyboard.add_hotkey(test_set_drag_end_hotkey, do_set_drag_end)
        if not global_key_poll or "set_marker_near" not in global_hotkeys:
            keyboard.add_hotkey(test_set_marker_near_hotkey, do_set_marker_near)
        if not global_key_poll or "click_marker" not in global_hotkeys:
            keyboard.add_hotkey(test_click_marker_hotkey, do_click_marker)
        if not global_key_poll or "click_mouse" not in global_hotkeys:
            keyboard.add_hotkey(test_click_mouse_hotkey, do_click_mouse)
        if not global_key_poll or "toggle_auto_click" not in global_hotkeys:
            keyboard.add_hotkey(test_auto_click_hotkey, do_toggle_auto_click)
        if not global_key_poll or "toggle_auto_esc" not in global_hotkeys:
            keyboard.add_hotkey(test_auto_esc_hotkey, do_toggle_auto_esc)
        if not global_key_poll or "log_mouse" not in global_hotkeys:
            keyboard.add_hotkey(test_log_mouse_hotkey, do_log_mouse)
        if not global_key_poll or "toggle_track_mouse" not in global_hotkeys:
            keyboard.add_hotkey(test_track_mouse_hotkey, do_toggle_track_mouse)
        if not global_key_poll or "set_marker_offset" not in global_hotkeys:
            keyboard.add_hotkey(test_set_marker_offset_hotkey, do_set_marker_offset)

    logging.info("Ready.")
    log_help()

    try:
        while state["running"]:
            if global_key_poll and global_hotkeys:
                for action, (modifiers, vk) in global_hotkeys.items():
                    is_down = _modifiers_down(modifiers) and _key_is_down(vk)
                    was_down = global_hotkey_state.get(action, False)
                    if is_down and not was_down:
                        if action == "exit":
                            stop()
                        else:
                            enqueue_action(action)
                    global_hotkey_state[action] = is_down
            handle_cli_keys()
            while not action_queue.empty():
                action = action_queue.get()
                if action == "reload_config":
                    reload_config()
                    continue
                if action == "toggle_monitoring":
                    toggle_monitoring("queue")
                    continue
                if action == "toggle_monitoring_cli":
                    toggle_monitoring("cli")
                    continue
                if action == "toggle_monitoring_hotkey":
                    toggle_monitoring("hotkey")
                    continue
                if action == "toggle_bypass":
                    toggle_bypass()
                    continue
                if action == "toggle_track_mouse":
                    mouse_state["track"] = not mouse_state["track"]
                    status = "ON" if mouse_state["track"] else "OFF"
                    mouse_state["last_report"] = 0.0
                    logging.info("Mouse tracking %s", status)
                    continue
                if action == "toggle_auto_click":
                    auto_click_enabled = not auto_click_enabled
                    auto_click_next = 0.0
                    status = "ON" if auto_click_enabled else "OFF"
                    if auto_click_enabled:
                        auto_click_next = time.monotonic()
                    logging.info("Auto-click %s (interval %.2fs)", status, auto_click_interval_s)
                    continue
                if action == "toggle_auto_esc":
                    auto_esc_enabled = not auto_esc_enabled
                    auto_esc_next = 0.0
                    status = "ON" if auto_esc_enabled else "OFF"
                    if auto_esc_enabled:
                        auto_esc_next = time.monotonic()
                    logging.info("Auto-ESC %s (interval %.2fs)", status, auto_esc_interval_s)
                    continue
                if action == "show_help":
                    print("")
                    log_help()
                    continue
                if action == "press_esc":
                    logging.info("Test ESC keypress")
                    pydirectinput.press("esc")
                    continue
                if action == "scroll_down":
                    amount = max(1, abs(scroll_amount))
                    events = max(1, int(scroll_events))
                    logging.info("Scroll down by %d events %d", amount, events)
                    for _ in range(events):
                        _scroll_wheel(-amount)
                    continue
                if action == "scroll_up":
                    amount = max(1, abs(scroll_amount))
                    events = max(1, int(scroll_events))
                    logging.info("Scroll up by %d events %d", amount, events)
                    for _ in range(events):
                        _scroll_wheel(amount)
                    continue
                if action == "drag":
                    if is_window_minimized(hwnd):
                        logging.warning("Window is minimized; drag skipped.")
                        continue
                    try:
                        origin_x, origin_y, client_w, client_h = get_client_origin_and_size(hwnd)
                    except Exception as exc:
                        logging.error("Failed to read window metrics: %s", exc)
                        continue
                    if client_w == 0 or client_h == 0:
                        logging.warning("Client size is zero; drag skipped.")
                        continue
                    render_x, render_y, render_w, render_h = _compute_render_rect(
                        (origin_x, origin_y), (client_w, client_h), ref_size, render_cfg
                    )
                    if drag_use_ref:
                        start_x, start_y = map_point(
                            drag_start,
                            ref_size,
                            (render_x, render_y),
                            (render_w, render_h),
                        )
                        end_x, end_y = map_point(
                            drag_end,
                            ref_size,
                            (render_x, render_y),
                            (render_w, render_h),
                        )
                        space = "ref"
                    else:
                        start_x, start_y = drag_start
                        end_x, end_y = drag_end
                        space = "screen"
                    logging.info(
                        "Dragging %d,%d -> %d,%d (%s) hold %.2fs duration %.2fs button %s",
                        start_x,
                        start_y,
                        end_x,
                        end_y,
                        space,
                        drag_hold_s,
                        drag_duration_s,
                        drag_button,
                    )
                    pydirectinput.moveTo(int(start_x), int(start_y), duration=0.0)
                    try:
                        pydirectinput.mouseDown(button=drag_button)
                    except TypeError:
                        pydirectinput.mouseDown(int(start_x), int(start_y))
                    if drag_hold_s > 0:
                        time.sleep(drag_hold_s)
                    pydirectinput.moveTo(
                        int(end_x),
                        int(end_y),
                        duration=max(0.0, float(drag_duration_s)),
                    )
                    try:
                        pydirectinput.mouseUp(button=drag_button)
                    except TypeError:
                        pydirectinput.mouseUp(int(end_x), int(end_y))
                    continue
                if action in {"set_drag_start", "set_drag_end"}:
                    was_monitoring = state["monitoring"]
                    state["monitoring"] = False
                    label = "start" if action == "set_drag_start" else "end"
                    current = drag_start if action == "set_drag_start" else drag_end
                    space_label = "ref" if drag_use_ref else "screen"
                    print("")
                    prompt = (
                        f"Enter drag {label} x,y ({space_label} coords). "
                        "Blank = use current mouse: "
                    )
                    entry = input(prompt)
                    parsed = _parse_offset_text(entry)
                    if parsed is None and not entry.strip():
                        x, y = _get_cursor_pos()
                        if drag_use_ref:
                            if is_window_minimized(hwnd):
                                logging.warning("Window is minimized; drag %s unchanged.", label)
                            else:
                                try:
                                    origin_x, origin_y, client_w, client_h = get_client_origin_and_size(hwnd)
                                except Exception as exc:
                                    logging.error("Failed to read window metrics: %s", exc)
                                else:
                                    render_x, render_y, render_w, render_h = _compute_render_rect(
                                        (origin_x, origin_y), (client_w, client_h), ref_size, render_cfg
                                    )
                                    scale_x = render_w / ref_size[0] if ref_size[0] else 0.0
                                    scale_y = render_h / ref_size[1] if ref_size[1] else 0.0
                                    if scale_x > 0 and scale_y > 0:
                                        parsed = (
                                            int(round((x - render_x) / scale_x)),
                                            int(round((y - render_y) / scale_y)),
                                        )
                                    else:
                                        logging.warning("Render scale invalid; drag %s unchanged.", label)
                        else:
                            parsed = (x, y)
                    if parsed is None:
                        logging.info(
                            "Drag %s unchanged: %d,%d (%s coords)",
                            label,
                            current[0],
                            current[1],
                            space_label,
                        )
                    else:
                        if action == "set_drag_start":
                            drag_start = parsed
                        else:
                            drag_end = parsed
                        logging.info(
                            "Drag %s set to %d,%d (%s coords)",
                            label,
                            parsed[0],
                            parsed[1],
                            space_label,
                        )
                    if was_monitoring:
                        state["monitoring"] = True
                    continue
                if action == "set_scroll_amount":
                    was_monitoring = state["monitoring"]
                    state["monitoring"] = False
                    print("")
                    prompt = f"Enter scroll amount (current {scroll_amount}, blank keep): "
                    entry = input(prompt)
                    parsed = _parse_int(entry)
                    if parsed is None:
                        logging.info("Scroll amount unchanged: %d", scroll_amount)
                    else:
                        scroll_amount = max(1, abs(parsed))
                        logging.info("Scroll amount set to %d", scroll_amount)
                    if was_monitoring:
                        state["monitoring"] = True
                    continue
                if action == "click_mouse":
                    x, y = _get_cursor_pos()
                    logging.info("Clicking current mouse position at %d,%d", x, y)
                    pydirectinput.click(x, y)
                    continue
                if action == "log_mouse":
                    log_mouse_position("Mouse")
                    continue
                if action == "set_marker_offset":
                    was_monitoring = state["monitoring"]
                    state["monitoring"] = False
                    print("")
                    prompt = (
                        f"Enter marker click offset (x,y in ref coords). Current {marker_click_offset[0]},"
                        f"{marker_click_offset[1]} (blank keep): "
                    )
                    entry = input(prompt)
                    parsed = _parse_offset_text(entry)
                    if parsed is None:
                        logging.info("Marker click offset unchanged: %d,%d", marker_click_offset[0], marker_click_offset[1])
                    else:
                        marker_click_offset = parsed
                        logging.info("Marker click offset set to %d,%d (ref coords)", parsed[0], parsed[1])
                    if was_monitoring:
                        state["monitoring"] = True
                    continue
                if action == "set_marker_near":
                    was_monitoring = state["monitoring"]
                    state["monitoring"] = False
                    print("")
                    near_x = marker_near[0] if marker_near is not None else 0
                    near_y = marker_near[1] if marker_near is not None else 0
                    prompt_near = (
                        f"Enter marker near x,y (ref coords). Current {near_x},{near_y} "
                        "(blank keep, 'clear' to disable): "
                    )
                    entry = input(prompt_near)
                    if entry.strip().lower() == "clear":
                        marker_near = None
                        marker_radius = 0
                        logging.info("Marker near cleared.")
                    else:
                        parsed = _parse_offset_text(entry)
                        if parsed is not None:
                            marker_near = parsed
                        prompt_radius = f"Enter marker radius (current {marker_radius}, 0 disables): "
                        radius_entry = input(prompt_radius)
                        parsed_radius = _parse_int(radius_entry)
                        if parsed_radius is not None:
                            marker_radius = max(0, parsed_radius)
                        near_label = "none" if marker_near is None else f"{marker_near[0]},{marker_near[1]}"
                        logging.info(
                            "Marker near set to %s radius %d",
                            near_label,
                            marker_radius,
                        )
                    if was_monitoring:
                        state["monitoring"] = True
                    continue
                if action in {"click_point", "click_marker"}:
                    if is_window_minimized(hwnd):
                        logging.warning("Window is minimized; click skipped.")
                        continue
                    try:
                        origin_x, origin_y, client_w, client_h = get_client_origin_and_size(hwnd)
                    except Exception as exc:
                        logging.error("Failed to read window metrics: %s", exc)
                        continue
                    if client_w == 0 or client_h == 0:
                        logging.warning("Client size is zero; click skipped.")
                        continue
                    render_x, render_y, render_w, render_h = _compute_render_rect(
                        (origin_x, origin_y), (client_w, client_h), ref_size, render_cfg
                    )
                    if action == "click_point":
                        screen_x, screen_y = map_point(
                            (int(test_click_point.get("x", 0)), int(test_click_point.get("y", 0))),
                            ref_size,
                            (render_x, render_y),
                            (render_w, render_h),
                        )
                        logging.info("Test click at %d,%d", screen_x, screen_y)
                        pydirectinput.click(screen_x, screen_y)
                        continue
                    if marker_method not in {"template", "feature"}:
                        logging.warning("Marker click requires template or feature mode.")
                        continue
                    scale_x = render_w / ref_size[0] if ref_size[0] else 1.0
                    scale_y = render_h / ref_size[1] if ref_size[1] else 1.0
                    roi_ref = _near_to_roi(marker_near, marker_radius, ref_size) or marker_roi_ref
                    marker_rect = map_rect(roi_ref, ref_size, (render_x, render_y), (render_w, render_h))
                    marker_img = capture.grab_auto(marker_rect, hwnd, capture_method)
                    if marker_img is None:
                        logging.warning("Marker capture failed; no click performed.")
                        continue
                    score, loc, t_size, second_best = marker_detector.match_with_location(
                        marker_img, scale_x, scale_y
                    )
                    if loc is None or t_size is None:
                        logging.warning("Marker match failed; no click performed.")
                        continue
                    if not marker_bypass_threshold:
                        if marker_min_score_delta > 0.0 and second_best is not None:
                            delta = score - second_best
                            if delta < marker_min_score_delta:
                                logging.info(
                                    "Marker delta %.3f below min %.3f; no click.",
                                    delta,
                                    marker_min_score_delta,
                                )
                                continue
                        if score < marker_threshold:
                            logging.info(
                                "Marker score %.2f below threshold %.2f; no click.", score, marker_threshold
                            )
                            continue
                    delta = score - second_best if second_best is not None else 0.0
                    click_x = marker_rect[0] + loc[0] + t_size[0] // 2
                    click_y = marker_rect[1] + loc[1] + t_size[1] // 2
                    if marker_click_offset != (0, 0):
                        click_x += int(round(marker_click_offset[0] * scale_x))
                        click_y += int(round(marker_click_offset[1] * scale_y))
                    logging.info(
                        "Clicking marker at %d,%d (score %.2f, delta %.3f, offset %d,%d, bypass %d)",
                        click_x,
                        click_y,
                        score,
                        delta,
                        marker_click_offset[0],
                        marker_click_offset[1],
                        int(marker_bypass_threshold),
                    )
                    pydirectinput.click(click_x, click_y)

            if mouse_state["track"]:
                now = time.monotonic()
                if now - mouse_state["last_report"] >= MOUSE_TRACK_INTERVAL_S:
                    log_mouse_position("Mouse")
                    mouse_state["last_report"] = now

            if auto_click_enabled:
                now = time.monotonic()
                if now >= auto_click_next:
                    x, y = _get_cursor_pos()
                    logging.info("Auto-click at %d,%d", x, y)
                    pydirectinput.click(x, y)
                    auto_click_next = now + auto_click_interval_s

            if auto_esc_enabled:
                now = time.monotonic()
                if now >= auto_esc_next:
                    logging.info("Auto-ESC")
                    pydirectinput.press("esc")
                    auto_esc_next = now + auto_esc_interval_s

            if not state["monitoring"]:
                time.sleep(0.1)
                continue

            loop_start = time.perf_counter()

            if is_window_minimized(hwnd):
                if not last_minimized_warned:
                    logging.warning("Window is minimized; waiting for restore.")
                    last_minimized_warned = True
                time.sleep(0.2)
                continue
            last_minimized_warned = False

            try:
                origin_x, origin_y, client_w, client_h = get_client_origin_and_size(hwnd)
            except Exception as exc:
                logging.error("Failed to read window metrics: %s", exc)
                time.sleep(0.2)
                continue

            if client_w == 0 or client_h == 0:
                time.sleep(0.2)
                continue

            if not last_size_warned and (abs(client_w - ref_size[0]) > 2 or abs(client_h - ref_size[1]) > 2):
                logging.warning(
                    "Client size %dx%d differs from reference %dx%d",
                    client_w,
                    client_h,
                    ref_size[0],
                    ref_size[1],
                )
                last_size_warned = True

            render_x, render_y, render_w, render_h = _compute_render_rect(
                (origin_x, origin_y), (client_w, client_h), ref_size, render_cfg
            )
            scale_x = render_w / ref_size[0] if ref_size[0] else 1.0
            scale_y = render_h / ref_size[1] if ref_size[1] else 1.0

            roi_ref = _near_to_roi(marker_near, marker_radius, ref_size) or marker_roi_ref
            marker_rect = map_rect(roi_ref, ref_size, (render_x, render_y), (render_w, render_h))
            change_rect = map_rect(change_roi_ref, ref_size, (render_x, render_y), (render_w, render_h))

            marker_img = capture.grab_auto(marker_rect, hwnd, capture_method)
            change_img = capture.grab_auto(change_rect, hwnd, capture_method)
            if marker_img is None or change_img is None:
                time.sleep(0.05)
                continue

            marker_present = False
            marker_score = 0.0
            if marker_method in {"template", "feature"}:
                marker_score, _loc, _t_size, second_best = marker_detector.match_with_location(
                    marker_img, scale_x, scale_y
                )
                if marker_bypass_threshold:
                    marker_present = _loc is not None
                else:
                    marker_present = marker_score >= marker_threshold
                    if marker_present and marker_min_score_delta > 0.0 and second_best is not None:
                        marker_present = (marker_score - second_best) >= marker_min_score_delta
            elif marker_method == "pixel":
                marker_present, marker_score = pixel_points_check(
                    pixel_points,
                    marker_img,
                    marker_rect,
                    ref_size,
                    (render_x, render_y),
                    (render_w, render_h),
                    capture,
                )
            else:
                logging.error("Unknown marker detection method: %s", marker_method)

            change_score, change_detected = delta_detector.update(change_img)

            elapsed = time.perf_counter() - loop_start
            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            status = _status_line(
                (origin_x, origin_y),
                (client_w, client_h),
                (render_w, render_h),
                scale_x,
                scale_y,
                marker_present,
                marker_score,
                change_score,
                change_detected,
                fps,
                marker_bypass_threshold,
            )
            print(status)

            should_save = False
            if debug_save_frames:
                if debug_save_on_toggle_only:
                    if nonlocal_vars["last_marker_present"] is None:
                        should_save = True
                    elif nonlocal_vars["last_marker_present"] != marker_present:
                        should_save = True
                    if nonlocal_vars["last_change_detected"] is None:
                        should_save = True
                    elif nonlocal_vars["last_change_detected"] != change_detected:
                        should_save = True
                else:
                    if debug_save_every_n > 0 and nonlocal_vars["frame_index"] % debug_save_every_n == 0:
                        should_save = True

            if should_save:
                _save_debug_frame(
                    capture,
                    hwnd,
                    capture_method,
                    debug_output_dir,
                    nonlocal_vars["frame_index"],
                    (origin_x, origin_y),
                    (client_w, client_h),
                    (render_x, render_y, render_w, render_h),
                    marker_rect,
                    change_rect,
                    debug_overlay,
                    marker_score,
                    change_score,
                    marker_present,
                    change_detected,
                )

            nonlocal_vars["last_marker_present"] = marker_present
            nonlocal_vars["last_change_detected"] = change_detected
            nonlocal_vars["frame_index"] += 1

            sleep_time = max(0.0, poll_interval_ms / 1000.0 - (time.perf_counter() - loop_start))
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        print("")
        keyboard.unhook_all_hotkeys()
        logging.info("Exited.")


if __name__ == "__main__":
    main()
