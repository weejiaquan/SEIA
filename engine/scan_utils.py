"""
Scan utilities for grid-based template scanning.

Provides helpers for stop marker detection, cell filtering, result collection,
debug visualization stitching, and grid scanning.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Sequence

import cv2
import numpy as np


def scan_grid(
    cache_set: str | None = None,
    template_folder: str | None = None,
    threshold: float = 0.75,
    roi_ref: tuple[int, int, int, int] | None = None,
    grid_size: tuple[int, int] = (6, 2),
    row_gap: int = 0,
    col_gap: int = 0,
    cache_file: str | None = None,
    cache_only: bool | None = None,
    scan_fn: Callable | None = None,
) -> dict:
    """
    Scan a grid region for templates using CLIP matching.
    
    Smart auto-rebuild mode (default):
    - Uses cache if valid (templates unchanged)
    - Auto-rebuilds cache if templates added/removed
    - No rebuild if cache matches templates
    - Fastest workflow for development
    
    Args:
        cache_set: Name of cache set (e.g., "students", "class", "sports").
                  Auto-resolves to templates/{name}/ and cache/{name}.npz
        template_folder: Path to folder containing template images.
                        Auto-set from cache_set if not provided.
        threshold: CLIP similarity threshold (0.0-1.0)
        roi_ref: ROI in reference coordinates (x, y, w, h)
        grid_size: Grid dimensions (cols, rows)
        row_gap: Vertical gap between rows in pixels (default 0)
        col_gap: Horizontal gap between columns in pixels (default 0)
        cache_file: Path to embedding cache. Auto-set from cache_set if not provided.
        cache_only: If True, only uses cache (errors if missing).
                   If False, always rebuilds cache from templates.
                   If None (default), smart mode: auto-rebuilds only if changed.
        scan_fn: Optional scan function to use. If None, imports from runtime.
    
    Returns:
        Scan results dict with matched templates and cells
    
    Example (Recommended - smart auto-rebuild):
        # Just use cache_set - automatically handles everything!
        results = scan_grid(
            cache_set="students",  # Smart: uses cache or rebuilds if changed
            threshold=0.75,
            roi_ref=(60, 300, 1780, 650),
            grid_size=(6, 2),
        )
        
        # What happens:
        # - First run: builds cache/students.npz from templates/students/
        # - Next runs: uses cache (fast!)
        # - Add image: auto-detects and rebuilds cache
        # - Remove image: auto-detects and rebuilds cache
        # - No changes: uses cache (no rebuild!)
        
    Example (Multiple cache sets):
        scan_grid(cache_set="students", ...)  # Auto-managed
        scan_grid(cache_set="class", ...)     # Auto-managed
        scan_grid(cache_set="sports", ...)    # Auto-managed
    
    Example (Cache-only mode - production):
        # Ship only cache files, no templates
        results = scan_grid(
            cache_set="students",
            cache_only=True,  # Errors if cache missing
            ...
        )
    
    Workflow:
        1. Drop images in templates/students/
        2. Run script → builds cache/students.npz (first time)
        3. Run again → uses cache (instant!)
        4. Add CH0248.png → auto-rebuilds cache
        5. Run again → uses cache (instant!)
        6. Production → set cache_only=True, ship cache/*.npz only
    """
    if scan_fn is None:
        from .runtime import scan_templates_clip
        scan_fn = scan_templates_clip
    
    # Auto-resolve cache_set to paths
    if cache_set is not None:
        if template_folder is None:
            template_folder = f"templates/{cache_set}"
        if cache_file is None:
            cache_file = f"cache/{cache_set}.npz"
    
    # Load config defaults if not provided
    from .utils import load_script_config
    config = load_script_config()
    clip_config = config.get("clip", {})
    
    # Don't override cache_only if explicitly set
    # None = smart mode (auto-rebuild on changes)
    
    if cache_file is None:
        cache_file = clip_config.get("cache_file", "cache/templates.npz")
    
    if template_folder is None:
        template_folder = clip_config.get("template_folder", "templates")
    
    return scan_fn(
        template_folder=template_folder,
        threshold=threshold,
        roi_ref=roi_ref,
        grid_size=grid_size,
        row_gap=row_gap,
        col_gap=col_gap,
        cache_file=cache_file,
        cache_only=cache_only,
    )


def detect_stop_marker(
    template_path: str,
    rois: list[tuple[tuple[int, int, int, int], str]],
    threshold: float = 0.85,
    present_clip_fn: Callable | None = None,
) -> str | None:
    """
    Check if a stop marker is visible in any of the specified ROIs.

    Uses CLIP matching to detect the stop marker. Useful for detecting
    "end of list" markers in scrolling grids.

    Args:
        template_path: Path to stop marker template image
        rois: List of ((x, y, w, h), label) tuples defining ROIs and their labels
        threshold: CLIP similarity threshold (default: 0.85)
        present_clip_fn: Function to call for CLIP matching. If None, imports
                        from engine.runtime.

    Returns:
        Label of first matching ROI, or None if not found.

    Example:
        stop_row = detect_stop_marker(
            "templates/stop_marker.png",
            [
                ((875, 295, 150, 60), "row1"),
                ((875, 710, 150, 45), "row2"),
            ],
            threshold=0.85
        )
    """
    if present_clip_fn is None:
        from .runtime import present_clip
        present_clip_fn = present_clip

    # Check if template exists or if cache is available
    cache_file = _find_templates_cache()
    if not os.path.exists(template_path) and cache_file is None:
        print(f"  [stop marker] Template not found and no cache: {template_path}")
        return None

    # Check each ROI
    for roi, label in rois:
        found = present_clip_fn(template_path, threshold=threshold, roi_ref=roi, log=True)
        if found:
            print(f"  [stop marker] Found in {label}!")
            return label

    return None


def _find_templates_cache() -> str | None:
    """Find templates.npz cache file in current directory or parent."""
    candidates = [
        os.path.join(os.getcwd(), "cache", "templates.npz"),
        os.path.join(os.path.dirname(os.getcwd()), "cache", "templates.npz"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def filter_cells_by_stop_marker(
    cells: list[dict],
    stop_label: str | None,
    grid_cols: int,
    stop_behaviors: dict[str, str] | None = None,
) -> list[dict]:
    """
    Filter cells based on where a stop marker was found.

    Default behaviors:
    - "row1" or "all": discard ALL cells
    - "row2": discard only row 2 cells (indices >= grid_cols)
    - None: keep all cells

    Args:
        cells: List of cell dicts from scan results
        stop_label: Label returned by detect_stop_marker, or None
        grid_cols: Number of columns in the grid
        stop_behaviors: Optional dict mapping labels to behaviors:
                       "all" = discard all cells
                       "row2" = discard row 2+ cells
                       "none" = keep all cells

    Returns:
        Filtered list of cells

    Example:
        filtered = filter_cells_by_stop_marker(
            cells=results["cells"],
            stop_label="row2",
            grid_cols=6,
        )
    """
    if stop_label is None:
        return cells

    # Default behaviors
    default_behaviors = {
        "row1": "all",
        "all": "all",
        "row2": "row2",
    }

    behaviors = stop_behaviors or default_behaviors
    behavior = behaviors.get(stop_label, "none")

    if behavior == "all":
        print(f"  [filter] Stop marker in {stop_label} - discarding ALL cells")
        return []

    if behavior == "row2":
        # Keep only row 1 cells (indices 0 to grid_cols-1)
        # cell_index is 1-based in the data
        filtered = [c for c in cells if c.get("cell_index", 1) <= grid_cols]
        print(f"  [filter] Stop marker in {stop_label} - keeping {len(filtered)} row1 cells, discarding row2")
        return filtered

    return cells


def collect_unique_matches(
    detections: list[dict],
    name_cleaner: Callable[[str], str] | None = None,
    match_key: str = "template_name",
) -> list[str]:
    """
    Collect unique matched templates from scan results in scan order.

    Args:
        detections: List of detection dicts, each with a "cells" list
        name_cleaner: Optional function to clean template names
        match_key: Key in cell dict containing template name (default: "template_name")

    Returns:
        List of unique template names/IDs in the order they were first seen.

    Example:
        students = collect_unique_matches(
            detections=all_detections,
            name_cleaner=clean_template_name
        )
    """
    seen = set()
    result = []

    for detection in detections:
        for cell in detection.get("cells", []):
            if not cell.get("matched"):
                continue

            name = cell.get(match_key)
            if not name or name in seen:
                continue

            seen.add(name)
            if name_cleaner is not None:
                result.append(name_cleaner(name))
            else:
                result.append(name)

    return result


def save_scan_results(
    detections: list[dict],
    output_file: str,
    unique_items: list[str] | None = None,
    include_raw: bool = True,
) -> None:
    """
    Save scan results to JSON with summary stats.

    Args:
        detections: List of detection dicts from scanning
        output_file: Path to output JSON file
        unique_items: Optional list of unique items found (for summary)
        include_raw: If True, include raw_detections array in output

    Output JSON includes:
        - unique_count: Number of unique items
        - total_scans: Number of scan steps
        - total_cells_scanned: Total cells across all scans
        - students_in_order: List of unique items (if provided)
        - raw_detections: Full detection data (if include_raw=True)
    """
    total_cells = sum(len(d.get("cells", [])) for d in detections)

    output = {
        "unique_count": len(unique_items) if unique_items else 0,
        "total_scans": len(detections),
        "total_cells_scanned": total_cells,
    }

    if unique_items is not None:
        output["students_in_order"] = unique_items

    if include_raw:
        output["raw_detections"] = detections

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Scan results saved to {output_file}")


def stitch_debug_visualizations(
    images: Sequence[str | np.ndarray],
    output_file: str,
    labels: list[str] | None = None,
    highlight_indices: list[int] | None = None,
    label_color: tuple[int, int, int] = (0, 255, 255),  # Yellow BGR
    highlight_color: tuple[int, int, int] = (0, 0, 255),  # Red BGR
) -> bool:
    """
    Stitch multiple debug visualization images vertically.

    Args:
        images: List of image paths (str) or numpy arrays to stitch
        output_file: Path to save stitched image
        labels: Optional labels to add to each image (top-right corner)
        highlight_indices: Indices of images to highlight (e.g., where stop marker found)
        label_color: BGR color for normal labels (default: yellow)
        highlight_color: BGR color for highlighted labels (default: red)

    Returns:
        True if successful, False otherwise.

    Example:
        stitch_debug_visualizations(
            images=["step1.png", "step2.png", numpy_array],
            output_file="stitched.png",
            labels=["Step 1", "Step 2 [STOP@row2]"],
            highlight_indices=[1]
        )
    """
    if not images:
        print("No images to stitch")
        return False

    loaded_images = []
    highlight_set = set(highlight_indices or [])

    for i, img in enumerate(images):
        # Load image if it's a path
        if isinstance(img, str):
            if not os.path.exists(img):
                print(f"  [stitch] Skipping missing image: {img}")
                continue
            loaded = cv2.imread(img)
            if loaded is None:
                print(f"  [stitch] Failed to load: {img}")
                continue
        elif isinstance(img, np.ndarray):
            loaded = img.copy()
        else:
            continue

        # Add label if provided
        if labels and i < len(labels):
            label_text = labels[i]
            color = highlight_color if i in highlight_set else label_color

            # Position text at top-right
            text_x = loaded.shape[1] - 250
            text_y = 60

            cv2.putText(
                loaded,
                label_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                color,
                3,
            )

        loaded_images.append(loaded)

    if not loaded_images:
        print("No valid images to stitch")
        return False

    # Stitch vertically
    stitched = np.vstack(loaded_images)

    cv2.imwrite(output_file, stitched)
    print(f"Stitched {len(loaded_images)} images: {output_file} ({stitched.shape[1]}x{stitched.shape[0]})")

    return True
