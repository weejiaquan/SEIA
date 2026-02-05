from __future__ import annotations

from typing import Dict


TOOLS: Dict[str, Dict[str, str]] = {
    "debug": {
        "script": "debug.py",
        "desc": "Interactive debug harness (hotkeys, monitoring, click tests).",
    },
    "verify-tester": {
        "script": "verify_tester.py",
        "desc": "Interactive template + verify pixel tester.",
    },
    "roi-picker": {
        "script": "roi_picker.py",
        "desc": "Drag-select ROI + mark pixels (OpenCV window).",
    },
    "roi-selector": {
        "script": "roi_selector.py",
        "desc": "Click two points to compute ROI (mouse polling).",
    },
    "offset-calculator": {
        "script": "offset_calculator.py",
        "desc": "Click two points to compute coords_offset.",
    },
    "drag-picker": {
        "script": "drag_picker.py",
        "desc": "Pick drag start/end (F1) + optional test drag (F3).",
    },
    "mouse-tracker": {
        "script": "mouse_tracker.py",
        "desc": "Track mouse -> ref coords + pixel color.",
    },
    "pick-pixels": {
        "script": "pick_pixels.py",
        "desc": "Pick verification pixels from a template image.",
    },
    "debug-viewer": {
        "script": "debug_viewer.py",
        "desc": "Tile viewer for debug_out screenshots.",
    },
    "stability-tester": {
        "script": "stability_tester.py",
        "desc": "Calibrate ROI change detection thresholds for wait_for_stable().",
    },
    "ocr-tester": {
        "script": "ocr_tester.py",
        "desc": "Compare OCR engines side-by-side on a selected ROI.",
    },
}


def sorted_tools() -> list[tuple[str, Dict[str, str]]]:
    return sorted(TOOLS.items())
