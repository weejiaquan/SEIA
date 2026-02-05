"""
Debug helper: Visualize grid alignment on screenshot.

This creates a PNG with the grid cells drawn as rectangles so you can verify
that the ROI and grid_size are aligned properly with the character portraits.

NOTE: Imports SCAN_ROI and GRID_SIZE from script.py to stay in sync!
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

# Add parent directory to path for engine import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from engine.runtime import _ensure_engine

# Import configuration directly from script.py to keep in sync
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from script import SCAN_ROI, GRID_SIZE

print(f"[DEBUG] Using ROI from script.py: {SCAN_ROI}, Grid: {GRID_SIZE}")


def visualize_grid(screenshot_path: str = None):
    """Capture screen and draw grid overlay."""
    print("Visualizing grid alignment...")
    print(f"  ROI: {SCAN_ROI}")
    print(f"  Grid: {GRID_SIZE}")
    
    # Try to load from screenshot file first
    if screenshot_path and os.path.exists(screenshot_path):
        print(f"  Loading from: {screenshot_path}")
        capture_image = cv2.imread(screenshot_path, cv2.IMREAD_COLOR)
        if capture_image is None:
            print(f"ERROR: Failed to load {screenshot_path}")
            return
        
        img_h, img_w = capture_image.shape[:2]
        print(f"  Full image: {img_w}x{img_h}")
        
        # Crop to ROI if this is a full screen capture (1920x1080)
        # ROI is (x, y, w, h) format
        roi_x, roi_y, roi_w, roi_h = SCAN_ROI
        expected_roi_shape = (roi_h, roi_w)  # OpenCV shape is (height, width)
        
        if capture_image.shape[:2] != expected_roi_shape:
            # Crop to ROI
            print(f"  Cropping to ROI: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")
            capture_image = capture_image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    else:
        # Get engine and capture the ROI
        engine = _ensure_engine()
        capture_image = engine.capture_roi(SCAN_ROI)
        
        if capture_image is None:
            print("ERROR: Capture failed!")
            return
    
    h, w = capture_image.shape[:2]
    cols, rows = GRID_SIZE
    cell_w = w // cols
    cell_h = h // rows
    
    print(f"  Captured: {w}x{h}")
    print(f"  Cell size: {cell_w}x{cell_h}")
    
    # Convert to RGB for display (OpenCV uses BGR)
    vis_image = cv2.cvtColor(capture_image, cv2.COLOR_BGR2RGB)
    
    # Draw grid cells
    for row in range(rows):
        for col in range(cols):
            x1 = col * cell_w
            y1 = row * cell_h
            x2 = min(x1 + cell_w, w)
            y2 = min(y1 + cell_h, h)
            
            cell_idx = row * cols + col + 1
            
            # Draw rectangle border
            color = (0, 255, 0) if cell_idx <= 6 else (0, 255, 255)  # Green for row 1, yellow for row 2
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 3)
            
            # Draw cell number
            cv2.putText(
                vis_image,
                f"#{cell_idx}",
                (x1 + 10, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )
            
            # Draw center crosshair
            center_x = x1 + cell_w // 2
            center_y = y1 + cell_h // 2
            cv2.drawMarker(
                vis_image,
                (center_x, center_y),
                (255, 0, 0),
                cv2.MARKER_CROSS,
                20,
                2,
            )
    
    # Add ROI info text at top
    cv2.putText(
        vis_image,
        f"ROI: {SCAN_ROI} | Grid: {GRID_SIZE} | Cell: {cell_w}x{cell_h}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    
    # Save to file
    output_file = "grid_debug_visualization.png"
    cv2.imwrite(output_file, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    print(f"\nVisualization saved to: {output_file}")
    print("Open this file to check if grid cells align with character portraits.")


if __name__ == "__main__":
    import sys
    
    # Check for screenshot path argument or look for debug_out
    screenshot_path = None
    if len(sys.argv) > 1:
        screenshot_path = sys.argv[1]
    else:
        # Try to find latest screenshot in debug_out
        debug_out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "debug_out")
        if os.path.exists(debug_out):
            screenshots = [os.path.join(debug_out, f) for f in os.listdir(debug_out) if f.endswith('.png')]
            if screenshots:
                screenshots.sort(key=os.path.getmtime, reverse=True)
                screenshot_path = screenshots[0]
                print(f"Using latest screenshot: {screenshot_path}")
    
    visualize_grid(screenshot_path)
