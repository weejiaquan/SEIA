from __future__ import annotations

import ctypes
import inspect
import json
import os
import sys
import threading
import time
from typing import Callable, Iterable, Tuple

import keyboard
import pydirectinput

from .core import AutomationEngine
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
    keyboard.unhook_all_hotkeys()
    _global_hotkey_stop.set()
    if _global_hotkey_thread is not None and _global_hotkey_thread.is_alive():
        _global_hotkey_thread.join(timeout=0.5)


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
