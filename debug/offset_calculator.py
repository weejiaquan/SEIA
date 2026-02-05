"""
Offset Calculator Tool
Click two points to calculate the offset between them.
Useful for finding coords_offset parameters for action() calls.
"""
from __future__ import annotations

import logging
import os
from typing import Tuple

from common import DEFAULT_CONFIG_PATH, get_cursor_pos, get_render_context, load_config, start_mouse_poller
from engine.mapper import is_window_minimized, set_process_dpi_awareness


def calculate_offset(hwnd: int, ref_size: Tuple[int, int], render_rect: Tuple[int, int, int, int]) -> None:
    """
    Interactive offset calculation by clicking two points.
    """
    print("\n" + "="*60)
    print("OFFSET CALCULATOR")
    print("="*60)
    print("Instructions:")
    print("  1. Click on the FIRST point (e.g., template center)")
    print("  2. Click on the SECOND point (e.g., where you want to click)")
    print("  3. Offset will be calculated!")
    print("\nWaiting for first click...")
    print("="*60)
    
    points = []
    
    def on_click():
        """Capture click position."""
        if is_window_minimized(hwnd):
            print("ERROR: Window is minimized!")
            return
        
        x, y = get_cursor_pos()
        
        # Convert to reference coordinates
        render_x, render_y, render_w, render_h = render_rect
        scale_x = render_w / ref_size[0] if ref_size[0] else 1.0
        scale_y = render_h / ref_size[1] if ref_size[1] else 1.0
        ref_x = int(round((x - render_x) / scale_x)) if scale_x else 0
        ref_y = int(round((y - render_y) / scale_y)) if scale_y else 0
        
        points.append((ref_x, ref_y))
        
        if len(points) == 1:
            print(f"\n✓ Point 1 (initial): screen=({x}, {y}) ref=({ref_x}, {ref_y})")
            print("\nNow click the SECOND point...")
        elif len(points) == 2:
            print(f"✓ Point 2 (target): screen=({x}, {y}) ref=({ref_x}, {ref_y})")
            
            # Calculate offset
            x1, y1 = points[0]
            x2, y2 = points[1]
            
            offset_x = x2 - x1
            offset_y = y2 - y1
            
            print("\n" + "="*60)
            print("OFFSET CALCULATED!")
            print("="*60)
            print(f"\nPoint 1: ({x1}, {y1})")
            print(f"Point 2: ({x2}, {y2})")
            print(f"\nOffset: ({offset_x}, {offset_y})")
            print("\nAdd this to your code:")
            print("="*60)
            print(f"coords_offset=({offset_x}, {offset_y})")
            print("="*60)
            print("\nExample usage:")
            print(f'action("template.png", coords_offset=({offset_x}, {offset_y}))')
            print("="*60)
            
            points.clear()
            print("\n" + "="*60)
            print("Ready for next offset calculation...")
            print("Waiting for first click...")
            print("="*60)
    
    start_mouse_poller(on_click)


def main():
    """Main entry point."""
    set_process_dpi_awareness()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    # Load config
    if not os.path.exists(DEFAULT_CONFIG_PATH):
        print(f"ERROR: Config file not found: {DEFAULT_CONFIG_PATH}")
        return

    config = load_config(DEFAULT_CONFIG_PATH)
    hwnd, ref_size, render_rect, _scale_x, _scale_y = get_render_context(config)
    
    print("\n" + "="*60)
    print("OFFSET CALCULATOR - READY")
    print("="*60)
    print(f"Reference resolution: {ref_size[0]}x{ref_size[1]}")
    print(f"Window size: {render_rect[2]}x{render_rect[3]}")
    print("="*60)
    
    # Start offset calculation
    calculate_offset(hwnd, ref_size, render_rect)
    
    print("\nOffset calculator is running...")
    print("Click on the game window to select points")
    print("Press Ctrl+C to exit\n")
    
    try:
        import keyboard
        while True:
            keyboard.wait()
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
