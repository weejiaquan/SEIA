"""
Mouse Tracking Tool
Standalone mouse tracking hotkey (based on debug.py).
Press F8 to toggle tracking.
"""
from __future__ import annotations

import logging
import os
import time

import keyboard

from common import DEFAULT_CONFIG_PATH, get_capture_method, get_cursor_pos, get_render_context, load_config
from engine.capture import ScreenCapture
from engine.mapper import is_window_minimized, set_process_dpi_awareness
DEFAULT_TRACK_MOUSE_HOTKEY = "f8"
MOUSE_TRACK_INTERVAL_S = 0.2


def main():
    """Main entry point for mouse tracking tool."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    set_process_dpi_awareness()

    if not os.path.exists(DEFAULT_CONFIG_PATH):
        logging.error("Config file not found: %s", DEFAULT_CONFIG_PATH)
        return

    config = load_config(DEFAULT_CONFIG_PATH)
    capture_method = get_capture_method(config)
    hwnd, ref_size, render_rect, _scale_x, _scale_y = get_render_context(config)

    capture = ScreenCapture()
    mouse_state = {
        "track": True,
        "last_report": 0.0
    }

    def toggle_tracking():
        mouse_state["track"] = not mouse_state["track"]
        status = "ON" if mouse_state["track"] else "OFF"
        mouse_state["last_report"] = 0.0
        logging.info("Mouse tracking %s", status)

    keyboard.add_hotkey(DEFAULT_TRACK_MOUSE_HOTKEY, toggle_tracking)

    def log_mouse_position():
        x, y = get_cursor_pos()
        if is_window_minimized(hwnd):
            logging.warning("Window is minimized; mouse log skipped.")
            return
        render_x, render_y, render_w, render_h = render_rect
        scale_x = render_w / ref_size[0] if ref_size[0] else 1.0
        scale_y = render_h / ref_size[1] if ref_size[1] else 1.0
        ref_x = int(round((x - render_x) / scale_x)) if scale_x else 0
        ref_y = int(round((y - render_y) / scale_y)) if scale_y else 0

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
                    "Mouse screen %d,%d -> ref %d,%d | RGB%s %s %s (render origin %d,%d, scale %.3f/%.3f)",
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
                    "Mouse screen %d,%d -> ref %d,%d (render origin %d,%d, scale %.3f/%.3f)",
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
                "Mouse screen %d,%d -> ref %d,%d (render origin %d,%d, scale %.3f/%.3f)",
                x,
                y,
                ref_x,
                ref_y,
                render_x,
                render_y,
                scale_x,
                scale_y,
            )

    try:
        logging.info("Mouse tracking tool ready.")
        logging.info("Hotkey: %s toggle tracking", DEFAULT_TRACK_MOUSE_HOTKEY)
        logging.info("Tracking starts ON.")
        logging.info("Press Ctrl+C to exit.")
        while True:
            if mouse_state["track"]:
                now = time.monotonic()
                if now - mouse_state["last_report"] >= MOUSE_TRACK_INTERVAL_S:
                    log_mouse_position()
                    mouse_state["last_report"] = now
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        logging.info("Exiting...")
    finally:
        keyboard.unhook_all_hotkeys()


if __name__ == "__main__":
    main()
