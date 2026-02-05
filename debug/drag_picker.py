"""
Drag Picker Tool
Pick two points with F1 to set drag start/end for debug testing.
Outputs both reference-space and screen-space coordinates.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Callable, Dict, Tuple, Optional

import keyboard

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from common import get_cursor_pos
from engine.core import AutomationEngine
from engine.mapper import is_window_minimized


def pick_drag(
    engine: AutomationEngine,
    hold_s: Optional[float] = None,
    drag_duration_s: Optional[float] = None,
    begin_sleep_s: Optional[float] = None,
    end_sleep_s: Optional[float] = None,
    settle_s: Optional[float] = None,
    button: Optional[str] = None,
) -> Tuple[Callable[[], None], Callable[[], None], Callable[[], None]]:
    print("\n" + "=" * 60)
    print("DRAG PICKER")
    print("=" * 60)
    print("Instructions:")
    print("  1. Move mouse to START point, press F1")
    print("  2. Move mouse to END point, press F1")
    print("  3. Press F2 to enter custom coords, F3 to test")
    print("  4. Values will print for config.json")
    print("\nWaiting for first pick (F1)...")
    print("=" * 60)

    points: list[Tuple[int, int, int, int]] = []
    custom_mode: list[bool] = [False]
    hwnd = engine._ensure_window()

    def _screen_to_ref(x: int, y: int) -> Tuple[int, int]:
        _, _, render_rect, scale_x, scale_y = engine._get_window_metrics()
        render_x, render_y = render_rect[0], render_rect[1]
        ref_x = int(round((x - render_x) / scale_x)) if scale_x else 0
        ref_y = int(round((y - render_y) / scale_y)) if scale_y else 0
        return ref_x, ref_y

    def on_pick() -> None:
        if is_window_minimized(hwnd):
            print("ERROR: Window is minimized!")
            return
        if len(points) >= 2:
            points.clear()
        x, y = get_cursor_pos()
        ref_x, ref_y = _screen_to_ref(x, y)
        points.append((x, y, ref_x, ref_y))

        if len(points) == 1:
            print(f"\nOK Start: screen=({x}, {y}) ref=({ref_x}, {ref_y})")
            print("\nMove to END point, press F1...")
            return

        if len(points) == 2:
            custom_mode[0] = False
            start = points[0]
            end = points[1]
            print(f"OK End: screen=({x}, {y}) ref=({ref_x}, {ref_y})")
            print("\n" + "=" * 60)
            print("DRAG VALUES")
            print("=" * 60)
            print(f"Start ref: ({start[2]}, {start[3]})")
            print(f"End ref:   ({end[2]}, {end[3]})")
            print(f"Start screen: ({start[0]}, {start[1]})")
            print(f"End screen:   ({end[0]}, {end[1]})")
            print("\nConfig snippet (ref coords):")
            print(f'  "drag_start": {{"x": {start[2]}, "y": {start[3]}}},')
            print(f'  "drag_end": {{"x": {end[2]}, "y": {end[3]}}},')
            print("\nConfig snippet (screen coords):")
            print(f'  "drag_start": {{"x": {start[0]}, "y": {start[1]}}},')
            print(f'  "drag_end": {{"x": {end[0]}, "y": {end[1]}}},')
            print("=" * 60)
            print("\nPress F2 for custom input, F3 to test, or F1 to pick new points...")

    def on_custom_input() -> None:
        print("\n" + "=" * 60)
        print("CUSTOM INPUT MODE")
        print("=" * 60)
        try:
            use_ref = input("Use reference coordinates? (y/n, default y): ").strip().lower()
            use_ref = use_ref != 'n'
            
            coord_type = "ref" if use_ref else "screen"
            start_input = input(f"Start (x,y) [{coord_type}]: ").strip()
            end_input = input(f"End (x,y) [{coord_type}]: ").strip()
            
            # Parse (x,y) format
            start_input = start_input.strip('()')
            end_input = end_input.strip('()')
            start_x, start_y = map(int, start_input.split(','))
            end_x, end_y = map(int, end_input.split(','))
            
            if use_ref:
                _, _, render_rect, scale_x, scale_y = engine._get_window_metrics()
                render_x, render_y = render_rect[0], render_rect[1]
                screen_start_x = int(round(render_x + start_x * scale_x))
                screen_start_y = int(round(render_y + start_y * scale_y))
                screen_end_x = int(round(render_x + end_x * scale_x))
                screen_end_y = int(round(render_y + end_y * scale_y))
            else:
                screen_start_x, screen_start_y = start_x, start_y
                screen_end_x, screen_end_y = end_x, end_y
                start_x, start_y = _screen_to_ref(screen_start_x, screen_start_y)
                end_x, end_y = _screen_to_ref(screen_end_x, screen_end_y)
            
            points.clear()
            points.append((screen_start_x, screen_start_y, start_x, start_y))
            points.append((screen_end_x, screen_end_y, end_x, end_y))
            custom_mode[0] = True
            
            print(f"\nSet custom drag:")
            print(f"  Start: screen=({screen_start_x}, {screen_start_y}) ref=({start_x}, {start_y})")
            print(f"  End:   screen=({screen_end_x}, {screen_end_y}) ref=({end_x}, {end_y})")
            print("=" * 60)
            print("Press F3 to test, F1 to pick with mouse, or F2 for new custom input\n")
        except (ValueError, KeyboardInterrupt) as e:
            print(f"\nERROR: Invalid input - {e}")
            print("=" * 60 + "\n")

    def on_test_drag() -> None:
        if len(points) == 0:
            print("ERROR: No drag points set. Press F1 twice or F2 for custom input.")
            return
        if len(points) == 1:
            print("ERROR: Only start point set. Press F1 again for end point.")
            return
        if is_window_minimized(hwnd):
            print("ERROR: Window is minimized!")
            return
        
        start = points[0]
        end = points[1]
        test_cfg = engine._config.get("test_input", {})
        
        # Use parameters if provided, otherwise fall back to config
        final_hold_s = hold_s if hold_s is not None else max(0.0, float(test_cfg.get("drag_hold_s", 0.1)))
        final_duration_s = drag_duration_s if drag_duration_s is not None else max(0.0, float(test_cfg.get("drag_duration_s", 0.5)))
        final_button = button if button is not None else str(test_cfg.get("drag_button", "left") or "left")
        final_begin_sleep_s = begin_sleep_s if begin_sleep_s is not None else max(0.0, float(test_cfg.get("drag_begin_sleep_s", 0.0)))
        final_end_sleep_s = end_sleep_s if end_sleep_s is not None else max(0.0, float(test_cfg.get("drag_end_sleep_s", 1.0)))
        final_settle_s = settle_s if settle_s is not None else max(0.0, float(test_cfg.get("drag_settle_s", 0.1)))
        drag_use_ref = bool(test_cfg.get("drag_use_ref", True))
        
        mode_str = " (CUSTOM)" if custom_mode[0] else ""
        print(f"Dragging{mode_str} ref ({start[2]}, {start[3]}) -> ({end[2]}, {end[3]}) "
              f"/ screen ({start[0]}, {start[1]}) -> ({end[0]}, {end[1]}) "
              f"(hold {final_hold_s:.2f}s, duration {final_duration_s:.2f}s, "
              f"begin_sleep {final_begin_sleep_s:.2f}s, end_sleep {final_end_sleep_s:.2f}s, "
              f"settle {final_settle_s:.2f}s, button {final_button})")
        
        if drag_use_ref:
            engine.drag_ref((start[2], start[3]), (end[2], end[3]), 
                          final_hold_s, final_duration_s, final_button, 
                          final_settle_s, final_begin_sleep_s, final_end_sleep_s)
        else:
            engine.drag_screen((start[0], start[1]), (end[0], end[1]), 
                             final_hold_s, final_duration_s, final_button,
                             final_settle_s, final_begin_sleep_s, final_end_sleep_s)

    return on_pick, on_custom_input, on_test_drag


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    config_path = os.path.join(PARENT_DIR, "config.json")
    engine = AutomationEngine(config_path)
    
    ref_w, ref_h = engine._ref_size
    print("\n" + "=" * 60)
    print("DRAG PICKER - READY")
    print("=" * 60)
    print(f"Reference resolution: {ref_w}x{ref_h}")
    print(f"Window title substring: {engine._title_substring}")
    print("=" * 60)

    on_pick, on_custom_input, on_test_drag = pick_drag(
        engine,
        settle_s=1.0,  # Hold at end before release to kill velocity
        end_sleep_s=1.0  # Sleep after release
    )

    print("\nDrag picker is running...")
    print("Press F1 to pick start/end, F2 for custom input, F3 to test drag")
    print("Press Ctrl+C to exit\n")

    try:
        keyboard.add_hotkey("f1", on_pick)
        keyboard.add_hotkey("f2", on_custom_input)
        keyboard.add_hotkey("f3", on_test_drag)
        while True:
            keyboard.wait()
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
