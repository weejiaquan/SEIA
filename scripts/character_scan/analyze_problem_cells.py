"""
Analyze specific problem cells to understand why they're matching incorrectly.

This script will:
1. Extract the exact cell regions for boxes 3 and 4
2. Compare them against the expected templates (CH0069, CH0209)
3. Compare them against the wrongly matched template (CH0325)
4. Save side-by-side comparisons to help debug
"""

from __future__ import annotations

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.runtime import _ensure_engine
from script import SCAN_ROI, GRID_SIZE, TEMPLATE_FOLDER


def analyze_cells():
    """Analyze problematic cells."""
    print("Analyzing problem cells (3 and 4)...")
    
    # Capture the ROI
    engine = _ensure_engine()
    capture_image = engine.capture_roi(SCAN_ROI)
    
    if capture_image is None:
        print("ERROR: Capture failed!")
        return
    
    h, w = capture_image.shape[:2]
    cols, rows = GRID_SIZE
    cell_w = w // cols
    cell_h = h // rows
    
    print(f"ROI: {SCAN_ROI}")
    print(f"Capture: {w}x{h}")
    print(f"Cell size: {cell_w}x{cell_h}")
    print()
    
    # Extract cells 3 and 4
    problem_cells = {
        3: "CH0069",
        4: "CH0209",
    }
    
    for cell_idx, expected_char in problem_cells.items():
        row = (cell_idx - 1) // cols
        col = (cell_idx - 1) % cols
        
        x1 = col * cell_w
        y1 = row * cell_h
        x2 = x1 + cell_w
        y2 = y1 + cell_h
        
        cell_img = capture_image[y1:y2, x1:x2]
        
        print(f"Box {cell_idx} (row={row+1}, col={col+1}):")
        print(f"  Position: x={x1}, y={y1}, w={x2-x1}, h={y2-y1}")
        print(f"  Expected: {expected_char}")
        
        # Save the cell image
        cell_filename = f"debug_cell_{cell_idx}_captured.png"
        cv2.imwrite(cell_filename, cell_img)
        print(f"  Saved: {cell_filename}")
        
        # Load and compare templates
        templates_to_check = [
            f"Student_Portrait_{expected_char}_Collection.png",  # Expected
            "Student_Portrait_CH0325_Collection.png",  # Wrong match
        ]
        
        for template_name in templates_to_check:
            template_path = os.path.join(TEMPLATE_FOLDER, template_name)
            if not os.path.exists(template_path):
                print(f"  Template not found: {template_name}")
                continue
            
            template_img = cv2.imread(template_path)
            if template_img is None:
                print(f"  Failed to load: {template_name}")
                continue
            
            # Resize template to match cell size for visual comparison
            template_resized = cv2.resize(template_img, (cell_img.shape[1], cell_img.shape[0]))
            
            # Create side-by-side comparison
            comparison = np.hstack([cell_img, template_resized])
            comparison_filename = f"debug_cell_{cell_idx}_vs_{template_name.replace('.png', '')}.png"
            cv2.imwrite(comparison_filename, comparison)
            
            char_code = template_name.replace("Student_Portrait_", "").replace("_Collection.png", "")
            print(f"  Comparison with {char_code}: {comparison_filename}")
        
        print()
    
    print("\nAnalysis complete! Check the debug_cell_*.png files to see:")
    print("1. What the scanner actually captured for each cell")
    print("2. Side-by-side comparison with expected vs wrong templates")
    print("\nThis will help determine if:")
    print("- The grid is misaligned (captured image doesn't match portrait)")
    print("- The templates are too similar")
    print("- The expected templates are incorrect")


if __name__ == "__main__":
    analyze_cells()
