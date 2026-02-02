from __future__ import annotations

import atexit
import ctypes
import inspect
import json
import logging
import os
import sys
import threading
import time
from typing import Callable, Iterable, Optional, Tuple

import cv2
import keyboard
import numpy as np
import pydirectinput

from .core import AutomationEngine
from .mapper import set_window_topmost
from .verify import add_pixel_verification


# ANSI color codes for terminal
class Color:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def _colored(text: str, color: str) -> str:
    """Add color to text if terminal supports it"""
    if sys.platform == 'win32':
        # Enable ANSI colors on Windows 10+
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass
    return f"{color}{text}{Color.RESET}"


BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
_template_dirs = [
    os.path.join(os.getcwd(), "templates"),
    os.path.join(ROOT_DIR, "templates"),
]

_engine: AutomationEngine | None = None
_stop_event = threading.Event()
_restart_event = threading.Event()
_debug_log_enabled: bool | None = None
_debug_log_mtime: float | None = None
_debug_log_path: str | None = None
_focused_once = False
_global_hotkey_poll_enabled: bool | None = None
_global_hotkey_thread: threading.Thread | None = None
_global_hotkey_stop = threading.Event()
_global_hotkey_lock = threading.Lock()
_global_hotkeys: dict[str, tuple[set[str], int]] = {}
_global_hotkey_state: dict[str, bool] = {}
_global_hotkey_callbacks: dict[str, Callable[[], None]] = {}
_cleanup_done = False

DEFAULT_TIMEOUT_S = 1000.0
DEFAULT_STEP_DELAY_S = 1.0
DEFAULT_LOOP_TIMEOUT_S = 30.0
DEFAULT_NEAR_RADIUS = 200
REF_W = 1920
REF_H = 1080


def _resolve_config_path() -> str:
    override = os.environ.get("SEIA_CONFIG_PATH")
    if override:
        return override
    cwd_path = os.path.join(os.getcwd(), "config.json")
    if os.path.exists(cwd_path):
        return cwd_path
    root_path = os.path.join(ROOT_DIR, "config.json")
    if os.path.exists(root_path):
        return root_path
    return cwd_path


def _ensure_engine() -> AutomationEngine:
    global _engine
    global _focused_once
    if _engine is None:
        _engine = AutomationEngine(config_path=_resolve_config_path())
        _focused_once = False
    if not _focused_once:
        try:
            if _engine.focus_window():
                _focused_once = True
                _log_action("focused target window")
        except Exception as exc:
            _log_action(f"focus window failed: {exc}")
    return _engine


def set_template_dir(path: str) -> None:
    if path:
        _template_dirs.insert(0, path)


def set_marker_dir(path: str) -> None:
    set_template_dir(path)


def _resolve_path(name: str) -> str:
    if os.path.isabs(name) or os.path.exists(name):
        return name
    for base in _template_dirs:
        candidate = os.path.join(base, name)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(_template_dirs[0], name)


def _get_config_path() -> str:
    if _engine is not None:
        return _engine.config_path
    return _resolve_config_path()


def _debug_log_enabled_now() -> bool:
    global _debug_log_enabled, _debug_log_mtime, _debug_log_path
    path = _get_config_path()
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return False
    if _debug_log_path == path and _debug_log_mtime == mtime and _debug_log_enabled is not None:
        return _debug_log_enabled
    try:
        with open(path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except Exception:
        return False
    enabled = bool(config.get("debug", {}).get("log", False))
    _debug_log_enabled = enabled
    _debug_log_mtime = mtime
    _debug_log_path = path
    return enabled


def _log_action(message: str) -> None:
    if _debug_log_enabled_now():
        print(f"[runtime] {message}", flush=True)


def _log_standby_context() -> None:
    if not _debug_log_enabled_now():
        return
    path = _get_config_path()
    _log_action(f"standby config_path={path}")
    config: dict[str, object] | None = None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except Exception as exc:
        _log_action(f"standby config_load_failed={exc}")
    if isinstance(config, dict):
        target = config.get("target", {}) if isinstance(config.get("target", {}), dict) else {}
        render_cfg = config.get("render_area", {}) if isinstance(config.get("render_area", {}), dict) else {}
        window_cfg = config.get("window", {}) if isinstance(config.get("window", {}), dict) else {}
        capture_cfg = config.get("capture", {}) if isinstance(config.get("capture", {}), dict) else {}
        ref = config.get("reference_resolution", {}) if isinstance(config.get("reference_resolution", {}), dict) else {}
        _log_action(
            "standby config "
            f"target_title={target.get('window_title_substring', '')} "
            f"target_process={target.get('process_name', '')} "
            f"ref={int(ref.get('w', 0))}x{int(ref.get('h', 0))} "
            f"render_mode={render_cfg.get('mode', '')} "
            f"render_offset={render_cfg.get('offset', {})} "
            f"render_size={render_cfg.get('size', {})} "
            f"auto_resize={window_cfg.get('auto_resize', False)} "
            f"force_topmost={window_cfg.get('force_topmost', False)} "
            f"capture_method={capture_cfg.get('method', '')}"
        )
    try:
        engine = _ensure_engine()
        (origin_x, origin_y), (client_w, client_h), render_rect, scale_x, scale_y = engine._get_window_metrics()
        _log_action(
            "standby window "
            f"origin={origin_x},{origin_y} "
            f"client={client_w}x{client_h} "
            f"render_rect={render_rect[0]},{render_rect[1]},{render_rect[2]},{render_rect[3]} "
            f"scale={scale_x:.3f}/{scale_y:.3f}"
        )
    except Exception as exc:
        _log_action(f"standby window_metrics_failed={exc}")


def _caller_context(action: str) -> dict[str, object]:
    runtime_path = os.path.abspath(__file__)
    for frame_info in inspect.stack()[2:]:
        filename = os.path.abspath(frame_info.filename)
        if filename != runtime_path:
            return {
                "action": action,
                "caller_file": filename,
                "caller_line": int(frame_info.lineno),
                "caller_func": frame_info.function,
            }
    return {"action": action}


def _format_match(match: object) -> tuple[str, str, str, str, str]:
    score_text = "n/a"
    delta_text = "n/a"
    ref_text = "n/a"
    screen_text = "n/a"
    extra_text = ""
    if hasattr(match, "score") and hasattr(match, "ref_center") and hasattr(match, "center"):
        score = float(getattr(match, "score", 0.0))
        score_text = f"{score:.2f} ({score * 100.0:.0f}%)"
        delta = getattr(match, "score_delta", None)
        if delta is not None:
            delta_text = f"{float(delta):.3f}"
        ref_center = getattr(match, "ref_center", None)
        if ref_center is not None:
            ref_text = f"{ref_center[0]},{ref_center[1]}"
        screen_center = getattr(match, "center", None)
        if screen_center is not None:
            screen_text = f"{screen_center[0]},{screen_center[1]}"
        parts: list[str] = []
        method = getattr(match, "method", None)
        if method:
            parts.append(f"method={method}")
        size = getattr(match, "template_size", None)
        if size is not None:
            parts.append(f"size={size[0]}x{size[1]}")
        second_best = getattr(match, "second_best", None)
        if second_best is not None:
            parts.append(f"second={float(second_best):.2f}")
        match_count = getattr(match, "match_count", None)
        match_total = getattr(match, "match_total", None)
        if match_count is not None and match_total is not None:
            parts.append(f"matches={int(match_count)}/{int(match_total)}")
        extra_text = " ".join(parts)
    return score_text, delta_text, ref_text, screen_text, extra_text


def _locate_match(
    image_name: str,
    use_default_threshold: bool,
    near: Tuple[int, int] | None,
    radius: int | None,
    roi_ref: Tuple[int, int, int, int] | None,
    min_score_delta: float | None = None,
    debug_context: dict[str, object] | None = None,
):
    engine = _ensure_engine()
    image_path = _resolve_path(image_name)
    if not os.path.exists(image_path):
        _fail(f"Image not found: {image_path}")
    roi = roi_ref if roi_ref is not None else _near_to_roi(near, radius)
    return engine.locate_image(
        image_path,
        use_default_threshold=use_default_threshold,
        roi_ref=roi,
        min_score_delta=min_score_delta,
        debug_context=debug_context,
    )


def _near_to_roi(near: Tuple[int, int] | None, radius: int | None) -> Tuple[int, int, int, int] | None:
    if near is None:
        return None
    use_radius = DEFAULT_NEAR_RADIUS if radius is None else int(radius)
    use_radius = max(1, use_radius)
    x = max(0, int(near[0]) - use_radius)
    y = max(0, int(near[1]) - use_radius)
    w = min(REF_W - x, use_radius * 2)
    h = min(REF_H - y, use_radius * 2)
    return (x, y, w, h)


def _vk_from_key_name(key: str) -> int | None:
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


def _parse_hotkey(hotkey: str) -> tuple[set[str], int] | None:
    cleaned = hotkey.strip().lower().replace(" ", "")
    if not cleaned:
        return None
    parts = [part for part in cleaned.split("+") if part]
    if not parts:
        return None
    modifiers: set[str] = set()
    key_name: str | None = None
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


def _global_hotkey_poll_enabled_now() -> bool:
    global _global_hotkey_poll_enabled
    if _global_hotkey_poll_enabled is not None:
        return _global_hotkey_poll_enabled
    override = os.environ.get("SEIA_GLOBAL_HOTKEY_POLL")
    if override is not None:
        _global_hotkey_poll_enabled = override.strip().lower() in {"1", "true", "yes", "on"}
        return _global_hotkey_poll_enabled
    path = _get_config_path()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except Exception:
        _global_hotkey_poll_enabled = False
        return _global_hotkey_poll_enabled
    runtime_cfg = config.get("runtime", {})
    if isinstance(runtime_cfg, dict) and "global_key_poll" in runtime_cfg:
        _global_hotkey_poll_enabled = bool(runtime_cfg.get("global_key_poll", False))
        return _global_hotkey_poll_enabled
    _global_hotkey_poll_enabled = bool(config.get("test_input", {}).get("global_key_poll", False))
    return _global_hotkey_poll_enabled


def _start_global_hotkey_thread() -> None:
    global _global_hotkey_thread
    if _global_hotkey_thread is not None and _global_hotkey_thread.is_alive():
        return

    def _poll_loop() -> None:
        while not _global_hotkey_stop.is_set():
            callbacks: list[Callable[[], None]] = []
            with _global_hotkey_lock:
                for action, (modifiers, vk) in _global_hotkeys.items():
                    is_down = _modifiers_down(modifiers) and _key_is_down(vk)
                    was_down = _global_hotkey_state.get(action, False)
                    if is_down and not was_down:
                        callback = _global_hotkey_callbacks.get(action)
                        if callback is not None:
                            callbacks.append(callback)
                    _global_hotkey_state[action] = is_down
            for callback in callbacks:
                try:
                    callback()
                except Exception:
                    pass
            time.sleep(0.02)

    _global_hotkey_stop.clear()
    _global_hotkey_thread = threading.Thread(target=_poll_loop, name="runtime-hotkey-poll", daemon=True)
    _global_hotkey_thread.start()


def _register_global_hotkey(action: str, hotkey: str, callback: Callable[[], None]) -> bool:
    parsed = _parse_hotkey(hotkey)
    if parsed is None:
        return False
    with _global_hotkey_lock:
        _global_hotkeys[action] = parsed
        _global_hotkey_callbacks[action] = callback
        _global_hotkey_state[action] = False
    _start_global_hotkey_thread()
    return True


def hotkey_stop(key: str = "F10") -> None:
    if _global_hotkey_poll_enabled_now():
        if _register_global_hotkey("stop", key, _stop_event.set):
            return
    keyboard.add_hotkey(key, _stop_event.set)


def hotkey_restart(key: str = "F5") -> None:
    if _global_hotkey_poll_enabled_now():
        if _register_global_hotkey("restart", key, _restart_event.set):
            return
    keyboard.add_hotkey(key, _restart_event.set)


def hotkey_callback(key: str, callback: Callable[[], None], name: str | None = None) -> None:
    if _global_hotkey_poll_enabled_now():
        action = name or f"callback_{id(callback)}"
        if _register_global_hotkey(action, key, callback):
            return
    keyboard.add_hotkey(key, callback)


def stop_requested() -> bool:
    return _stop_event.is_set()


def reset_stop() -> None:
    _stop_event.clear()


def cleanup() -> None:
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True
    _clear_force_topmost()
    keyboard.unhook_all_hotkeys()
    _global_hotkey_stop.set()
    if _global_hotkey_thread is not None and _global_hotkey_thread.is_alive():
        _global_hotkey_thread.join(timeout=0.5)


def _clear_force_topmost() -> None:
    engine = _engine
    if engine is None:
        return
    if not getattr(engine, "_force_topmost", False):
        return
    hwnd = getattr(engine, "_hwnd", None)
    if not hwnd:
        return
    try:
        set_window_topmost(hwnd, False)
    except Exception as exc:
        logging.debug("Failed to clear topmost on exit: %s", exc)


atexit.register(cleanup)


def sleep(seconds: float) -> None:
    remaining = max(0.0, float(seconds))
    end_time = time.monotonic() + remaining
    while True:
        if stop_requested():
            raise SystemExit(0)
        now = time.monotonic()
        if now >= end_time:
            return
        time.sleep(min(0.1, end_time - now))


def wait_for_restart() -> None:
    if _restart_event.is_set():
        _restart_event.clear()
        return
    _log_standby_context()
    _log_action("standby waiting for restart")
    while True:
        if _restart_event.wait(0.1):
            _restart_event.clear()
            return


def press(key: str, repeat: int = 1, delay_between: float = 0.0) -> None:
    """Press a key one or more times.
    
    Args:
        key: Key to press
        repeat: Number of times to press (default 1)
        delay_between: Delay in seconds between each press (default 0)
    """
    for i in range(repeat):
        _log_action(f"press key={key} ({i+1}/{repeat})")
        pydirectinput.press(key)
        if i < repeat - 1 and delay_between > 0:
            sleep(delay_between)


def hold_key(key: str, duration: float, repeat: int = 1, delay_between: float = 0.0) -> None:
    """Hold a key down for a specified duration, optionally repeating.
    
    Args:
        key: Key to hold
        duration: How long to hold in seconds
        repeat: Number of times to repeat the hold (default 1)
        delay_between: Delay in seconds between each hold (default 0)
    """
    for i in range(repeat):
        _log_action(f"hold key={key} duration={duration:.2f}s ({i+1}/{repeat})")
        pydirectinput.keyDown(key)
        sleep(duration)
        pydirectinput.keyUp(key)
        if i < repeat - 1 and delay_between > 0:
            sleep(delay_between)


def _fail(message: str, exit_code: int = 1) -> None:
    _stop_event.set()
    raise SystemExit(message)


def action(
    image_name: str,
    coords_offset: Tuple[int, int] = (0, 0),
    click_duration: float = 0.0,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    use_default_threshold: bool = True,
    require: bool = True,
    near: Tuple[int, int] | None = None,
    radius: int | None = None,
    roi_ref: Tuple[int, int, int, int] | None = None,
    verify_pixels: list[tuple] | None = None,
) -> bool:
    if stop_requested():
        raise SystemExit(0)

    # Add pixel verification if provided
    if verify_pixels:
        add_pixel_verification(image_name, verify_pixels)

    engine = _ensure_engine()
    image_path = _resolve_path(image_name)
    if not os.path.exists(image_path):
        _fail(f"Image not found: {image_path}")
    roi_ref = roi_ref if roi_ref is not None else _near_to_roi(near, radius)
    min_delta = engine.default_min_score_delta()
    _log_action(
        "action "
        f"image={image_name} "
        f"offset={coords_offset[0]},{coords_offset[1]} "
        f"duration={float(click_duration):.2f}s "
        f"timeout={float(timeout_s):.2f}s "
        f"near={near} radius={radius} "
        f"min_delta={min_delta:.3f}"
    )
    debug_context = _caller_context("action")
    match = engine.action_with_match(
        image_path,
        coords_offset=coords_offset,
        click_duration=click_duration,
        timeout_s=timeout_s,
        roi_ref=roi_ref,
        use_default_threshold=use_default_threshold,
        stop_check=stop_requested,
        debug_context=debug_context,
    )
    if stop_requested():
        raise SystemExit(0)
    ok = match is not None
    sleep(DEFAULT_STEP_DELAY_S)
    if not ok and require:
        _fail(f"Image not found within timeout: {image_name}")
    score_text = "n/a"
    delta_text = "n/a"
    ref_text = "n/a"
    screen_text = "n/a"
    extra_text = ""
    if match is not None:
        score_text, delta_text, ref_text, screen_text, extra_text = _format_match(match)
    _log_action(
        "action result "
        f"image={image_name} "
        f"ok={int(ok)} "
        f"score={score_text} "
        f"delta={delta_text} "
        f"ref={ref_text} "
        f"screen={screen_text} "
        f"{extra_text}"
    )
    if ok:
        print(_colored(f"[runtime] ✓ action result image={image_name} ok=1", Color.GREEN + Color.BOLD), flush=True)
    else:
        print(_colored(f"[runtime] ✗ action result image={image_name} ok=0", Color.RED + Color.BOLD), flush=True)
    return ok


def present(
    image_name: str,
    use_default_threshold: bool = True,
    near: Tuple[int, int] | None = None,
    radius: int | None = None,
    roi_ref: Tuple[int, int, int, int] | None = None,
    log: bool = True,
) -> bool:
    engine = _ensure_engine()
    image_path = _resolve_path(image_name)
    if not os.path.exists(image_path):
        _fail(f"Image not found: {image_path}")
    roi_ref = roi_ref if roi_ref is not None else _near_to_roi(near, radius)
    debug_context = _caller_context("present")
    ok = engine.image_present(
        image_path,
        use_default_threshold=use_default_threshold,
        roi_ref=roi_ref,
        debug_context=debug_context,
    )
    if log:
        _log_action(f"present image={image_name} ok={int(ok)}")
    return ok


def verify_pixels(
    points: Iterable[object],
    roi_ref: Tuple[int, int, int, int] | None = None,
    require: bool = False,
    log: bool = True,
) -> bool:
    """
    Verify pixel colors directly using reference-space coordinates.

    Args:
        points: Iterable of pixel points (PixelPoint, dict, or tuple forms).
        roi_ref: Optional ROI in reference coords to limit capture.
        require: If True, raise on failure.
        log: If True, emit debug log line.
    """
    engine = _ensure_engine()
    debug_context = _caller_context("verify_pixels")
    ok, score = engine.verify_pixels(points, roi_ref=roi_ref, debug_context=debug_context)
    if log:
        _log_action(f"verify_pixels ok={int(ok)} score={score:.2f}")
    if require and not ok:
        _fail("Pixel verification failed")
    return ok


def wait_for(
    image_name: str,
    timeout_s: float = DEFAULT_LOOP_TIMEOUT_S,
    step_delay_s: float = DEFAULT_STEP_DELAY_S,
    use_default_threshold: bool = True,
    require: bool = True,
    near: Tuple[int, int] | None = None,
    radius: int | None = None,
    roi_ref: Tuple[int, int, int, int] | None = None,
    verify_pixels: list[tuple] | None = None,
) -> bool:
    _log_action(
        "wait_for "
        f"image={image_name} "
        f"timeout={float(timeout_s):.2f}s "
        f"step_delay={float(step_delay_s):.2f}s"
    )
    debug_context = _caller_context("wait_for")

    # Add pixel verification if provided
    if verify_pixels:
        add_pixel_verification(image_name, verify_pixels)
    
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        if stop_requested():
            raise SystemExit(0)
        match = _locate_match(
            image_name,
            use_default_threshold=use_default_threshold,
            near=near,
            radius=radius,
            roi_ref=roi_ref,
            min_score_delta=None,
            debug_context=debug_context,
        )
        if match is not None:
            score_text, delta_text, ref_text, screen_text, extra_text = _format_match(match)
            _log_action(
                "wait_for match "
                f"image={image_name} "
                f"score={score_text} "
                f"delta={delta_text} "
                f"ref={ref_text} "
                f"screen={screen_text} "
                f"{extra_text}"
            )
            print(_colored(f"[runtime] ✓ wait_for result image={image_name} ok=1", Color.GREEN + Color.BOLD), flush=True)
            return True
        sleep(step_delay_s)
        if time.monotonic() >= deadline:
            if require:
                _fail(f"Timeout waiting for: {image_name}")
            print(_colored(f"[runtime] ✗ wait_for result image={image_name} ok=0", Color.RED + Color.BOLD), flush=True)
            return False


def wait_for_image_optional(
    image_name: str,
    timeout_s: float = DEFAULT_LOOP_TIMEOUT_S,
    step_delay_s: float = DEFAULT_STEP_DELAY_S,
    use_default_threshold: bool = True,
    near: Tuple[int, int] | None = None,
    radius: int | None = None,
    roi_ref: Tuple[int, int, int, int] | None = None,
) -> bool:
    _log_action(
        "wait_for_optional "
        f"image={image_name} "
        f"timeout={float(timeout_s):.2f}s "
        f"step_delay={float(step_delay_s):.2f}s"
    )
    debug_context = _caller_context("wait_for_optional")
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        if stop_requested():
            raise SystemExit(0)
        match = _locate_match(
            image_name,
            use_default_threshold=use_default_threshold,
            near=near,
            radius=radius,
            roi_ref=roi_ref,
            min_score_delta=None,
            debug_context=debug_context,
        )
        if match is not None:
            score_text, delta_text, ref_text, screen_text, extra_text = _format_match(match)
            _log_action(
                "wait_for_optional match "
                f"image={image_name} "
                f"score={score_text} "
                f"delta={delta_text} "
                f"ref={ref_text} "
                f"screen={screen_text} "
                f"{extra_text}"
            )
            print(_colored(f"[runtime] ✓ wait_for_optional result image={image_name} ok=1", Color.GREEN + Color.BOLD), flush=True)
            return True
        sleep(step_delay_s)
        if time.monotonic() >= deadline:
            print(_colored(f"[runtime] ✗ wait_for_optional result image={image_name} ok=0", Color.RED + Color.BOLD), flush=True)
            return False


def wait_for_absent(
    image_name: str,
    timeout_s: float = DEFAULT_LOOP_TIMEOUT_S,
    step_delay_s: float = DEFAULT_STEP_DELAY_S,
    use_default_threshold: bool = True,
    require: bool = True,
    near: Tuple[int, int] | None = None,
    radius: int | None = None,
    roi_ref: Tuple[int, int, int, int] | None = None,
) -> bool:
    _log_action(
        "wait_for_absent "
        f"image={image_name} "
        f"timeout={float(timeout_s):.2f}s "
        f"step_delay={float(step_delay_s):.2f}s"
    )
    debug_context = _caller_context("wait_for_absent")
    deadline = time.monotonic() + max(0.0, timeout_s)
    last_match = None
    while True:
        if stop_requested():
            raise SystemExit(0)
        match = _locate_match(
            image_name,
            use_default_threshold=use_default_threshold,
            near=near,
            radius=radius,
            roi_ref=roi_ref,
            min_score_delta=None,
            debug_context=debug_context,
        )
        if match is None:
            print(_colored(f"[runtime] ✓ wait_for_absent result image={image_name} ok=1", Color.GREEN + Color.BOLD), flush=True)
            return True
        last_match = match
        sleep(step_delay_s)
        if time.monotonic() >= deadline:
            if require:
                print(_colored(f"[runtime] ✗ wait_for_absent result image={image_name} ok=0", Color.RED + Color.BOLD), flush=True)
                _fail(f"Timeout waiting for absence: {image_name}")
            if last_match is not None:
                score_text, delta_text, ref_text, screen_text, extra_text = _format_match(last_match)
                _log_action(
                    "wait_for_absent still_present "
                    f"image={image_name} "
                    f"score={score_text} "
                    f"delta={delta_text} "
                    f"ref={ref_text} "
                    f"screen={screen_text} "
                    f"{extra_text}"
                )
            print(_colored(f"[runtime] ✗ wait_for_absent result image={image_name} ok=0", Color.RED + Color.BOLD), flush=True)
            return False


def wait_for_absent_optional(
    image_name: str,
    timeout_s: float = DEFAULT_LOOP_TIMEOUT_S,
    step_delay_s: float = DEFAULT_STEP_DELAY_S,
    use_default_threshold: bool = True,
    near: Tuple[int, int] | None = None,
    radius: int | None = None,
    roi_ref: Tuple[int, int, int, int] | None = None,
) -> bool:
    _log_action(
        "wait_for_absent_optional "
        f"image={image_name} "
        f"timeout={float(timeout_s):.2f}s "
        f"step_delay={float(step_delay_s):.2f}s"
    )
    debug_context = _caller_context("wait_for_absent_optional")
    deadline = time.monotonic() + max(0.0, timeout_s)
    last_match = None
    while True:
        if stop_requested():
            raise SystemExit(0)
        match = _locate_match(
            image_name,
            use_default_threshold=use_default_threshold,
            near=near,
            radius=radius,
            roi_ref=roi_ref,
            min_score_delta=None,
            debug_context=debug_context,
        )
        if match is None:
            print(_colored(f"[runtime] ✓ wait_for_absent_optional result image={image_name} ok=1", Color.GREEN + Color.BOLD), flush=True)
            return True
        last_match = match
        sleep(step_delay_s)
        if time.monotonic() >= deadline:
            if last_match is not None:
                score_text, delta_text, ref_text, screen_text, extra_text = _format_match(last_match)
                _log_action(
                    "wait_for_absent_optional still_present "
                    f"image={image_name} "
                    f"score={score_text} "
                    f"delta={delta_text} "
                    f"ref={ref_text} "
                    f"screen={screen_text} "
                    f"{extra_text}"
                )
            print(_colored(f"[runtime] ✗ wait_for_absent_optional result image={image_name} ok=0", Color.RED + Color.BOLD), flush=True)
            return False


def _load_mask(mask_image: str, roi_size: Tuple[int, int]) -> Optional[np.ndarray]:
    """Load a PNG, extract its alpha channel, and resize to *roi_size*.

    Returns a 2-D uint8 array (>0 = active pixel) or ``None`` if the image
    has no alpha channel or cannot be read.
    """
    path = _resolve_path(mask_image)
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None or raw.size == 0:
        logging.warning("_load_mask: could not read %s", path)
        return None
    if raw.ndim != 3 or raw.shape[2] != 4:
        logging.warning("_load_mask: %s has no alpha channel; mask ignored", path)
        return None
    alpha = raw[:, :, 3]
    mask = (alpha > 0).astype(np.uint8) * 255
    h, w = roi_size  # (height, width)
    if mask.shape[0] != h or mask.shape[1] != w:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def _frames_differ(
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
    pixel_threshold: int,
    change_ratio: float,
    mask: Optional[np.ndarray] = None,
) -> bool:
    """Return True if enough pixels differ between two BGR frames."""
    diff = cv2.absdiff(prev_bgr, curr_bgr)
    max_diff = np.max(diff, axis=2)
    if mask is not None:
        max_diff = max_diff[mask > 0]
    total = max_diff.size
    if total == 0:
        return False
    changed = int(np.count_nonzero(max_diff > pixel_threshold))
    return (changed / total) >= change_ratio


def _capture_roi_bgr(roi_ref: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """Capture an ROI in reference coords and return the BGR array."""
    try:
        engine = _ensure_engine()
        return engine.capture_roi(roi_ref)
    except Exception as exc:
        logging.debug("_capture_roi_bgr failed: %s", exc)
        return None


def wait_for_stable(
    roi_ref: Tuple[int, int, int, int],
    timeout_s: float = 30.0,
    poll_interval_s: float = 0.2,
    stable_count: int = 3,
    pixel_threshold: int = 20,
    change_ratio: float = 0.01,
    wait_for_change: bool = False,
    mask_image: str | None = None,
    require: bool = True,
) -> bool:
    """Wait until an ROI stops changing between consecutive captures.

    Args:
        roi_ref: Region in reference coordinates (x, y, w, h).
        timeout_s: Maximum time to wait.
        poll_interval_s: Interval between captures.
        stable_count: Consecutive identical frames required.
        pixel_threshold: Per-channel diff threshold per pixel.
        change_ratio: Fraction of pixels that must change to count as different.
        wait_for_change: If True, first wait for the ROI to *change* from its
            initial state before waiting for stability.
        mask_image: Optional path to a PNG with alpha channel used as a mask.
        require: If True, raise SystemExit on timeout.

    Returns:
        True if the ROI stabilised within the timeout, False otherwise.
    """
    _log_action(
        f"wait_for_stable roi={roi_ref} timeout={timeout_s:.1f}s "
        f"stable_count={stable_count} wait_for_change={wait_for_change}"
    )
    if stop_requested():
        raise SystemExit(0)

    deadline = time.monotonic() + max(0.0, timeout_s)

    # Load mask (once) if provided
    mask: Optional[np.ndarray] = None
    if mask_image is not None:
        first_frame = _capture_roi_bgr(roi_ref)
        if first_frame is not None:
            mask = _load_mask(mask_image, (first_frame.shape[0], first_frame.shape[1]))

    # Phase 1: optionally wait for initial change
    if wait_for_change:
        baseline = _capture_roi_bgr(roi_ref)
        if baseline is None:
            baseline = None  # will retry
        while time.monotonic() < deadline:
            if stop_requested():
                raise SystemExit(0)
            sleep(poll_interval_s)
            current = _capture_roi_bgr(roi_ref)
            if current is None:
                baseline = None
                continue
            if baseline is None:
                baseline = current
                continue
            if baseline.shape != current.shape:
                baseline = current
                continue
            if _frames_differ(baseline, current, pixel_threshold, change_ratio, mask):
                _log_action("wait_for_stable: initial change detected")
                break
        else:
            _log_action("wait_for_stable: timeout waiting for initial change")
            if require:
                _fail("Timeout waiting for ROI change in wait_for_stable")
            print(_colored("[runtime] ✗ wait_for_stable timed out (no change)", Color.RED + Color.BOLD), flush=True)
            return False

    # Phase 2: wait for stability
    prev = _capture_roi_bgr(roi_ref)
    consecutive_stable = 0

    while time.monotonic() < deadline:
        if stop_requested():
            raise SystemExit(0)
        sleep(poll_interval_s)
        current = _capture_roi_bgr(roi_ref)
        if current is None:
            consecutive_stable = 0
            prev = None
            continue
        if prev is None or prev.shape != current.shape:
            prev = current
            consecutive_stable = 0
            continue
        if _frames_differ(prev, current, pixel_threshold, change_ratio, mask):
            consecutive_stable = 0
        else:
            consecutive_stable += 1
        prev = current
        if consecutive_stable >= stable_count:
            _log_action(f"wait_for_stable: stable after {stable_count} frames")
            print(_colored("[runtime] ✓ wait_for_stable ok=1", Color.GREEN + Color.BOLD), flush=True)
            return True

    _log_action("wait_for_stable: timeout")
    if require:
        _fail("Timeout waiting for stability in wait_for_stable")
    print(_colored("[runtime] ✗ wait_for_stable ok=0", Color.RED + Color.BOLD), flush=True)
    return False


def press_until_stable(
    keys: Iterable[str] | str,
    roi_ref: Tuple[int, int, int, int],
    timeout_s: float = 60.0,
    step_delay_s: float = 0.5,
    stable_count: int = 3,
    pixel_threshold: int = 20,
    change_ratio: float = 0.01,
    poll_interval_s: float = 0.2,
    attempt_timeout_s: float = 5.0,
    mask_image: str | None = None,
    press_while_waiting: bool = False,
    require: bool = True,
) -> bool:
    """Press *keys* repeatedly until an ROI stabilises.

    Each iteration presses every key in *keys*, then polls the ROI for
    ``attempt_timeout_s`` checking for stability.  If the ROI does not
    stabilise, keys are pressed again.

    If ``press_while_waiting`` is True, keys are spammed continuously in a
    background loop while stability is checked. In that mode, the
    ``attempt_timeout_s`` window is ignored and stability is checked across
    the full ``timeout_s`` duration.

    Args:
        keys: Key name(s) to press each attempt.
        roi_ref: Region in reference coordinates (x, y, w, h).
        timeout_s: Overall timeout.
        step_delay_s: Delay after each key press.
        stable_count: Consecutive identical frames required.
        pixel_threshold: Per-channel diff threshold per pixel.
        change_ratio: Fraction of changed pixels to count as different.
        poll_interval_s: Interval between captures within an attempt.
        attempt_timeout_s: Max time to wait per attempt before retrying keys.
        mask_image: Optional path to a PNG with alpha channel used as mask.
        press_while_waiting: If True, keep pressing keys while checking stability.
        require: If True, raise SystemExit on overall timeout.

    Returns:
        True if the ROI stabilised, False on timeout.
    """
    if isinstance(keys, str):
        key_list = [keys]
    else:
        key_list = list(keys)

    _log_action(
        f"press_until_stable keys={','.join(key_list)} roi={roi_ref} "
        f"timeout={timeout_s:.1f}s attempt_timeout={attempt_timeout_s:.1f}s "
        f"press_while_waiting={int(press_while_waiting)}"
    )
    if stop_requested():
        raise SystemExit(0)

    overall_deadline = time.monotonic() + max(0.0, timeout_s)

    # Load mask once
    mask: Optional[np.ndarray] = None
    if mask_image is not None:
        first_frame = _capture_roi_bgr(roi_ref)
        if first_frame is not None:
            mask = _load_mask(mask_image, (first_frame.shape[0], first_frame.shape[1]))

    if press_while_waiting:
        stop_spam = threading.Event()

        def _spam_keys() -> None:
            try:
                while not stop_spam.is_set():
                    for key in key_list:
                        if stop_spam.is_set():
                            break
                        press(key)
                        sleep(step_delay_s)
            except SystemExit:
                stop_spam.set()

        spam_thread = threading.Thread(target=_spam_keys, name="press-until-stable-spam", daemon=True)
        spam_thread.start()
        try:
            prev = _capture_roi_bgr(roi_ref)
            consecutive_stable = 0
            while time.monotonic() < overall_deadline:
                if stop_requested():
                    raise SystemExit(0)
                sleep(poll_interval_s)
                current = _capture_roi_bgr(roi_ref)
                if current is None:
                    consecutive_stable = 0
                    prev = None
                    continue
                if prev is None or prev.shape != current.shape:
                    prev = current
                    consecutive_stable = 0
                    continue
                if _frames_differ(prev, current, pixel_threshold, change_ratio, mask):
                    consecutive_stable = 0
                else:
                    consecutive_stable += 1
                prev = current
                if consecutive_stable >= stable_count:
                    _log_action(f"press_until_stable: stable after {stable_count} frames")
                    print(_colored("[runtime] ✓ press_until_stable ok=1", Color.GREEN + Color.BOLD), flush=True)
                    return True
        finally:
            stop_spam.set()
            if spam_thread.is_alive():
                spam_thread.join(timeout=0.5)
    else:
        while time.monotonic() < overall_deadline:
            if stop_requested():
                raise SystemExit(0)

            # Press keys
            for key in key_list:
                press(key)
                sleep(step_delay_s)

            # Observe for stability
            attempt_end = min(time.monotonic() + attempt_timeout_s, overall_deadline)
            prev = _capture_roi_bgr(roi_ref)
            consecutive_stable = 0

            while time.monotonic() < attempt_end:
                if stop_requested():
                    raise SystemExit(0)
                sleep(poll_interval_s)
                current = _capture_roi_bgr(roi_ref)
                if current is None:
                    consecutive_stable = 0
                    prev = None
                    continue
                if prev is None or prev.shape != current.shape:
                    prev = current
                    consecutive_stable = 0
                    continue
                if _frames_differ(prev, current, pixel_threshold, change_ratio, mask):
                    consecutive_stable = 0
                else:
                    consecutive_stable += 1
                prev = current
                if consecutive_stable >= stable_count:
                    _log_action(f"press_until_stable: stable after {stable_count} frames")
                    print(_colored("[runtime] ✓ press_until_stable ok=1", Color.GREEN + Color.BOLD), flush=True)
                    return True

            _log_action("press_until_stable: not stable, retrying keys")

    _log_action("press_until_stable: overall timeout")
    if require:
        _fail("Timeout in press_until_stable")
    print(_colored("[runtime] ✗ press_until_stable ok=0", Color.RED + Color.BOLD), flush=True)
    return False


def press_until(
    image_name: str,
    keys: Iterable[str],
    timeout_s: float = DEFAULT_LOOP_TIMEOUT_S,
    step_delay_s: float = DEFAULT_STEP_DELAY_S,
    use_default_threshold: bool = True,
    require: bool = True,
    near: Tuple[int, int] | None = None,
    radius: int | None = None,
    roi_ref: Tuple[int, int, int, int] | None = None,
    verify_pixels: list[tuple] | None = None,
) -> bool:
    key_list = list(keys)

    # Add pixel verification if provided
    if verify_pixels:
        add_pixel_verification(image_name, verify_pixels)
    
    _log_action(
        "press_until "
        f"image={image_name} "
        f"keys={','.join(key_list)} "
        f"timeout={float(timeout_s):.2f}s "
        f"step_delay={float(step_delay_s):.2f}s"
    )
    debug_context = _caller_context("press_until")
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        if stop_requested():
            raise SystemExit(0)
        match = _locate_match(
            image_name,
            use_default_threshold=use_default_threshold,
            near=near,
            radius=radius,
            roi_ref=roi_ref,
            min_score_delta=None,
            debug_context=debug_context,
        )
        if match is not None:
            score_text, delta_text, ref_text, screen_text, extra_text = _format_match(match)
            _log_action(
                "press_until match "
                f"image={image_name} "
                f"score={score_text} "
                f"delta={delta_text} "
                f"ref={ref_text} "
                f"screen={screen_text} "
                f"{extra_text}"
            )
            print(_colored(f"[runtime] ✓ press_until result image={image_name} ok=1", Color.GREEN + Color.BOLD), flush=True)
            return True
        for key in key_list:
            press(key)
            sleep(step_delay_s)
        sleep(step_delay_s)
        if time.monotonic() >= deadline:
            if require:
                print(_colored(f"[runtime] ✗ press_until result image={image_name} ok=0", Color.RED + Color.BOLD), flush=True)
                _fail(f"Timeout waiting for: {image_name}")
            print(_colored(f"[runtime] ✗ press_until result image={image_name} ok=0", Color.RED + Color.BOLD), flush=True)
            return False


def press_until_or_skip_on_stable(
    image_name: str,
    keys: Iterable[str],
    timeout_s: float = DEFAULT_LOOP_TIMEOUT_S,
    step_delay_s: float = DEFAULT_STEP_DELAY_S,
    poll_interval_s: float = 0.2,
    stable_count: int = 6,
    pixel_threshold: int = 20,
    change_ratio: float = 0.01,
    wait_for_change: bool = False,
    use_default_threshold: bool = True,
    require: bool = False,
    near: Tuple[int, int] | None = None,
    radius: int | None = None,
    roi_ref: Tuple[int, int, int, int] | None = None,
    stable_roi_ref: Tuple[int, int, int, int] | None = None,
    verify_pixels: list[tuple] | None = None,
    mask_image: str | None = None,
) -> bool:
    """Press keys repeatedly while checking for a target image or a stable ROI.

    Returns True if the image is found. Returns False if the ROI is stable
    (interpreted as "stuck") or the timeout is reached.
    """
    if stop_requested():
        raise SystemExit(0)

    key_list = list(keys)

    if verify_pixels:
        add_pixel_verification(image_name, verify_pixels)

    roi_ref = roi_ref if roi_ref is not None else _near_to_roi(near, radius)
    stable_roi = stable_roi_ref if stable_roi_ref is not None else roi_ref
    if stable_roi is None:
        stable_roi = (0, 0, REF_W, REF_H)

    _log_action(
        "press_until_or_skip_on_stable "
        f"image={image_name} "
        f"keys={','.join(key_list)} "
        f"timeout={float(timeout_s):.2f}s "
        f"step_delay={float(step_delay_s):.2f}s "
        f"poll_interval={float(poll_interval_s):.2f}s "
        f"stable_count={int(stable_count)} "
        f"wait_for_change={int(wait_for_change)}"
    )

    stop_spam = threading.Event()

    def _spam_keys() -> None:
        try:
            while not stop_spam.is_set():
                for key in key_list:
                    if stop_spam.is_set():
                        break
                    press(key)
                    sleep(step_delay_s)
        except SystemExit:
            stop_spam.set()

    spam_thread = threading.Thread(target=_spam_keys, name="press-until-skip-spam", daemon=True)
    spam_thread.start()

    deadline = time.monotonic() + max(0.0, timeout_s)
    debug_context = _caller_context("press_until_or_skip_on_stable")

    mask: Optional[np.ndarray] = None
    if mask_image is not None:
        first_frame = _capture_roi_bgr(stable_roi)
        if first_frame is not None:
            mask = _load_mask(mask_image, (first_frame.shape[0], first_frame.shape[1]))

    prev = _capture_roi_bgr(stable_roi)
    consecutive_stable = 0
    seen_change = False

    try:
        while time.monotonic() < deadline:
            if stop_requested():
                raise SystemExit(0)
            match = _locate_match(
                image_name,
                use_default_threshold=use_default_threshold,
                near=near,
                radius=radius,
                roi_ref=roi_ref,
                min_score_delta=None,
                debug_context=debug_context,
            )
            if match is not None:
                score_text, delta_text, ref_text, screen_text, extra_text = _format_match(match)
                _log_action(
                    "press_until_or_skip_on_stable match "
                    f"image={image_name} "
                    f"score={score_text} "
                    f"delta={delta_text} "
                    f"ref={ref_text} "
                    f"screen={screen_text} "
                    f"{extra_text}"
                )
                print(_colored(f"[runtime] ✓ press_until_or_skip_on_stable result image={image_name} ok=1", Color.GREEN + Color.BOLD), flush=True)
                return True

            sleep(poll_interval_s)
            current = _capture_roi_bgr(stable_roi)
            if current is None:
                consecutive_stable = 0
                prev = None
                continue
            if prev is None or prev.shape != current.shape:
                prev = current
                consecutive_stable = 0
                continue
            if _frames_differ(prev, current, pixel_threshold, change_ratio, mask):
                consecutive_stable = 0
                seen_change = True
            else:
                if not wait_for_change or seen_change:
                    consecutive_stable += 1
            prev = current
            if consecutive_stable >= stable_count:
                _log_action("press_until_or_skip_on_stable: stable, skipping")
                print(_colored(f"[runtime] ⚠ press_until_or_skip_on_stable result image={image_name} ok=0 stable=1", Color.YELLOW + Color.BOLD), flush=True)
                return False
    finally:
        stop_spam.set()
        if spam_thread.is_alive():
            spam_thread.join(timeout=0.5)

    _log_action("press_until_or_skip_on_stable: timeout")
    if require:
        print(_colored(f"[runtime] ✗ press_until_or_skip_on_stable result image={image_name} ok=0", Color.RED + Color.BOLD), flush=True)
        _fail(f"Timeout waiting for: {image_name}")
    print(_colored(f"[runtime] ✗ press_until_or_skip_on_stable result image={image_name} ok=0", Color.RED + Color.BOLD), flush=True)
    return False


def press_until_optional(
    image_name: str,
    keys: Iterable[str],
    timeout_s: float = DEFAULT_LOOP_TIMEOUT_S,
    step_delay_s: float = DEFAULT_STEP_DELAY_S,
    use_default_threshold: bool = True,
    near: Tuple[int, int] | None = None,
    radius: int | None = None,
    roi_ref: Tuple[int, int, int, int] | None = None,
) -> bool:
    key_list = list(keys)
    _log_action(
        "press_until_optional "
        f"image={image_name} "
        f"keys={','.join(key_list)} "
        f"timeout={float(timeout_s):.2f}s "
        f"step_delay={float(step_delay_s):.2f}s"
    )
    debug_context = _caller_context("press_until_optional")
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        if stop_requested():
            raise SystemExit(0)
        match = _locate_match(
            image_name,
            use_default_threshold=use_default_threshold,
            near=near,
            radius=radius,
            roi_ref=roi_ref,
            min_score_delta=None,
            debug_context=debug_context,
        )
        if match is not None:
            score_text, delta_text, ref_text, screen_text, extra_text = _format_match(match)
            _log_action(
                "press_until_optional match "
                f"image={image_name} "
                f"score={score_text} "
                f"delta={delta_text} "
                f"ref={ref_text} "
                f"screen={screen_text} "
                f"{extra_text}"
            )
            print(_colored(f"[runtime] ✓ press_until_optional result image={image_name} ok=1", Color.GREEN + Color.BOLD), flush=True)
            return True
        for key in key_list:
            press(key)
            sleep(step_delay_s)
        sleep(step_delay_s)
        if time.monotonic() >= deadline:
            print(_colored(f"[runtime] ✗ press_until_optional result image={image_name} ok=0", Color.RED + Color.BOLD), flush=True)
            return False
