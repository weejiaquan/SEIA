"""
ROI Rectangle Selector Tool
Click two points on screen to define a rectangular ROI region.
Press hotkey to start selection, click top-left then bottom-right.
"""
from __future__ import annotations

import os
from typing import Tuple

import keyboard

from common import DEFAULT_CONFIG_PATH, compute_render_rect, get_cursor_pos, get_ref_size, load_config, start_mouse_poller
from engine.mapper import find_window, get_client_origin_and_size, is_window_minimized, set_process_dpi_awareness


# Hotkey to start ROI selection
ROI_SELECT_HOTKEY = "ctrl+shift+r"
CONFIG_PATH = DEFAULT_CONFIG_PATH



def select_roi(hwnd: int, ref_size: Tuple[int, int], render_rect: Tuple[int, int, int, int]) -> None:
    """
    Interactive ROI selection by clicking two points.
    
    Args:
        hwnd: Window handle
        ref_size: Reference resolution (w, h)
        render_rect: Render area (x, y, w, h)
    """
    print("\n" + "="*60)
    print("ROI RECTANGLE SELECTOR")
    print("="*60)
    print("Instructions:")
    print("  1. Click on the TOP-LEFT corner of your ROI")
    print("  2. Click on the BOTTOM-RIGHT corner of your ROI")
    print("  3. ROI coordinates will be generated!")
    print("\nWaiting for first click (top-left)...")
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
            print(f"\n✓ Point 1 (top-left): screen=({x}, {y}) ref=({ref_x}, {ref_y})")
            print("\nNow click BOTTOM-RIGHT corner...")
        elif len(points) == 2:
            print(f"✓ Point 2 (bottom-right): screen=({x}, {y}) ref=({ref_x}, {ref_y})")
            
            # Calculate ROI
            x1, y1 = points[0]
            x2, y2 = points[1]
            
            # Ensure top-left and bottom-right are correct
            roi_x = min(x1, x2)
            roi_y = min(y1, y2)
            roi_w = abs(x2 - x1)
            roi_h = abs(y2 - y1)
            
            print("\n" + "="*60)
            print("ROI GENERATED!")
            print("="*60)
            print(f"\nTop-left:     ({roi_x}, {roi_y})")
            print(f"Bottom-right: ({roi_x + roi_w}, {roi_y + roi_h})")
            print(f"Size:         {roi_w}x{roi_h}")
            print("\nAdd this to your code:")
            print("="*60)
            print(f"roi_ref=({roi_x}, {roi_y}, {roi_w}, {roi_h})  # x, y, w, h")
            print("="*60)
            print("\nPress Ctrl+Shift+R to select another ROI")
            
            points.clear()
            print("\n" + "="*60)
            print("Ready for next ROI selection...")
            print("Waiting for first click (top-left)...")
            print("="*60)
    
    start_mouse_poller(on_click)


def main():
    set_process_dpi_awareness()
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Missing config: {CONFIG_PATH}")
    config = load_config(CONFIG_PATH)
    target = config.get("target", {})
    title_substring = target.get("window_title_substring", "")
    process_name = target.get("process_name", "") or None
    hwnd = find_window(title_substring, process_name)
    if hwnd is None:
        raise RuntimeError("Target window not found.")
    
    ref_size = get_ref_size(config)
    
    # Get render area
    try:
        origin_x, origin_y, client_w, client_h = get_client_origin_and_size(hwnd)
    except Exception as e:
        print(f"ERROR: Failed to get window metrics: {e}")
        return
    
    render_cfg = config.get("render_area", {})
    render_rect = compute_render_rect((origin_x, origin_y), (client_w, client_h), ref_size, render_cfg)
    
    print("\n" + "="*60)
    print("ROI RECTANGLE SELECTOR - READY")
    print("="*60)
    print(f"Hotkey: {ROI_SELECT_HOTKEY} - Start ROI selection")
    print(f"Reference resolution: {ref_size[0]}x{ref_size[1]}")
    print(f"Window size: {client_w}x{client_h}")
    print("="*60)
    
    # Start ROI selection mode
    select_roi(hwnd, ref_size, render_rect)
    
    print("\nROI selector is running...")
    print("Click on the game window to select points")
    print("Press Ctrl+C to exit\n")
    
    try:
        while True:
            keyboard.wait()
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
