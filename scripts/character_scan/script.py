"""
Character Roster Scanner

Scans a game's character roster using CLIP-based template matching.
Scrolls through the list, detects stop markers, and exports results to CSV.

Usage:
    1. Place character portrait templates in templates/students/
    2. Configure config.json with your target window settings
    3. Run: python script.py
"""

from __future__ import annotations

import os
import sys
import time

# Add parent directory to path for engine import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from engine.runtime import (
    # Core automation
    present_clip,
    scan_templates_clip,
    hotkey_stop,
    hotkey_restart,
    wait_for_restart,
    stop_requested,
    sleep,
    scroll,
    # Utilities
    is_debug_mode,
    export_csv,
    clean_template_name,
    detect_stop_marker,
    filter_cells_by_stop_marker,
    collect_unique_matches,
    save_scan_results,
    scan_grid,
    stitch_debug_visualizations,
)

# ===== Domain Configuration =====

# CLIP cache sets - smart auto-rebuild
# Drop images in templates/students/ → auto-builds/updates cache/students.npz
# Cache only rebuilds if templates changed (add/remove images)
CACHE_SET = "students"  # Auto-managed: cache or rebuild as needed

# Character grid scan region (covers both rows)
SCAN_ROI = (60, 300, 1780, 650)  # x, y, w, h

# Grid layout
GRID_SIZE = (6, 2)  # 6 columns x 2 rows = 12 character slots
ROW_GAP = 180  # Pixels of gap between row 1 and row 2
COL_GAP = 30  # Pixels of gap between columns
THRESHOLD = 0.75  # CLIP similarity threshold

# Scroll configuration (reference coordinates 1920x1080)
SCROLL_START = (960, 1050)  # Start drag here (bottom center)
SCROLL_END = (960, 239.5)  # End drag here (top center)
SCROLL_HOLD_S = 0.0
SCROLL_DURATION_S = 1.0
SCROLL_SETTLE_S = 1.5
SCROLL_END_SLEEP_S = 0.5

# Stop marker detection - when this appears, characters below are unusable
STOP_MARKER_TEMPLATE = "templates/stop_marker.png"
STOP_MARKER_ROIS = [
    ((875, 295, 150, 60), "row1"),  # Row 1 position
    ((875, 710, 150, 45), "row2"),  # Row 2 position
]

# Output files
CSV_OUTPUT_FILE = "students.csv"
DEBUG_JSON_FILE = "scroll_scan_results.json"
DEBUG_STITCHED_FILE = "stitched_scan_visualization.png"

MAX_SCROLL_STEPS = 50  # Safety limit


# ===== Main Logic =====

def print_scan_summary(results: dict, step: int) -> None:
    """Print scan results summary for current view."""
    print(f"\n--- Scroll step {step} ---")
    print(f"Found {results['templates_matched']} characters in {results['scan_time_ms']:.0f}ms")

    if "cells" not in results:
        return

    cols = GRID_SIZE[0]
    for i, cell in enumerate(results["cells"]):
        candidates = cell.get("top_candidates", [])
        if candidates:
            name = clean_template_name(candidates[0]["template_name"])
            marker = "✓" if cell["matched"] else "✗"
            print(f"  Box {cell['cell_index']:2d} {marker}: {name}({candidates[0]['score']:.2f})")


def scroll_and_scan() -> None:
    """Scroll through character list and accumulate results."""
    hotkey_stop("F10")
    hotkey_restart("F5")

    debug_mode = is_debug_mode()

    print("Scroll Scanner Ready")
    print("  F5: Start scrolling scan")
    print("  F10: Stop")
    print()
    print(f"  Scroll: Y {SCROLL_START[1]} -> Y {SCROLL_END[1]}")
    print(f"  Duration: {SCROLL_DURATION_S}s, Settle: {SCROLL_SETTLE_S}s")
    print(f"  Debug mode: {'ON' if debug_mode else 'OFF'}")
    print()

    while True:
        wait_for_restart()
        if stop_requested():
            break

        all_detections = []
        visualization_images = []

        for step in range(MAX_SCROLL_STEPS):
            if stop_requested():
                break

            print(f"\n{'='*60}")
            print(f"SCAN STEP {step + 1}")
            print('='*60)

            # Scan current view (2 rows)
            # Smart: uses cache if valid, auto-rebuilds only if templates changed
            results = scan_grid(
                cache_set=CACHE_SET,
                threshold=THRESHOLD,
                roi_ref=SCAN_ROI,
                grid_size=GRID_SIZE,
                row_gap=ROW_GAP,
                col_gap=COL_GAP,
            )
            print_scan_summary(results, step + 1)

            # Check for stop marker
            stop_row = detect_stop_marker(
                STOP_MARKER_TEMPLATE,
                STOP_MARKER_ROIS,
                threshold=0.85,
                present_clip_fn=present_clip,
            )

            # Filter cells based on where stop marker was found
            cells = results.get("cells", [])
            filtered_cells = filter_cells_by_stop_marker(cells, stop_row, GRID_SIZE[0])

            # Collect debug visualization
            if debug_mode:
                debug_vis_path = "grid_debug_visualization.png"
                if os.path.exists(debug_vis_path):
                    visualization_images.append(debug_vis_path)

            # Accumulate results (with filtered cells)
            all_detections.append({
                "scroll_step": step,
                "timestamp": time.time(),
                "cells": filtered_cells,
                "templates_matched": len([c for c in filtered_cells if c.get("matched")]),
                "stop_marker": stop_row,
            })

            # Stop if marker found
            if stop_row:
                print(f"\n>>> Stop marker detected in {stop_row} - end of usable characters")
                break

            # Scroll to reveal more characters
            print()
            scroll(
                start_ref=SCROLL_START,
                end_ref=SCROLL_END,
                hold_s=SCROLL_HOLD_S,
                drag_duration_s=SCROLL_DURATION_S,
                settle_s=SCROLL_SETTLE_S,
                end_sleep_s=SCROLL_END_SLEEP_S,
            )
            sleep(0.3)

        # Process results using engine helpers
        students = collect_unique_matches(all_detections, clean_template_name)

        # Always export CSV
        export_csv(students, CSV_OUTPUT_FILE, fieldnames=["student_id"])

        # Debug mode: save detailed results and stitched visualization
        if debug_mode:
            save_scan_results(all_detections, DEBUG_JSON_FILE, students)

            # Stitch visualizations with labels
            if visualization_images:
                labels = [
                    f"Step {i+1}" + (f" [STOP@{d['stop_marker']}]" if d['stop_marker'] else "")
                    for i, d in enumerate(all_detections)
                ]
                highlight = [i for i, d in enumerate(all_detections) if d['stop_marker']]

                stitch_debug_visualizations(
                    visualization_images,
                    DEBUG_STITCHED_FILE,
                    labels=labels,
                    highlight_indices=highlight,
                )

        print(f"\n{'='*60}")
        print("SCAN COMPLETE")
        print('='*60)
        print(f"Total scans: {len(all_detections)}")
        print(f"Unique students found: {len(students)}")
        print()

        # Print summary in scan order
        for student_id in students:
            print(f"  {student_id}")

        print(f"\nResults exported to {CSV_OUTPUT_FILE}")
        print("\nPress F5 to scan again, F10 to stop")


if __name__ == "__main__":
    scroll_and_scan()
