"""
Interactive OCR comparison tool.

Two-phase workflow:
  Phase 1 - ROI selection (drag to pick sub-region, Enter to confirm)
  Phase 2 - Live OCR (Space to capture + OCR, Q to quit)

Usage:
    python debug/ocr_tester.py [--roi x,y,w,h] [--engine all|paddleocr|manga_ocr|easyocr]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from common import get_capture_method, get_render_context, load_config_with_fallback
from engine.capture import ScreenCapture
from engine.mapper import map_rect, set_process_dpi_awareness
from engine.ocr import (
    OCREngine,
    TextResult,
    available_engines,
    create_ocr_engine,
    parse_number,
    sort_by_position,
)

ROOT_DIR = PARENT_DIR
CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")
DEBUG_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

ENGINE_COLORS: list[Tuple[int, int, int]] = [
    (0, 255, 0),    # green
    (255, 165, 0),  # orange (BGR)
    (255, 0, 0),    # blue
    (0, 255, 255),  # yellow
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive OCR comparison tool")
    parser.add_argument("--roi", type=str, default=None, help="ROI as x,y,w,h (reference coords)")
    parser.add_argument("--engine", type=str, default="all", help="Engine: all, paddleocr, manga_ocr, easyocr")
    return parser.parse_args()


def _init_engines(
    engine_filter: str, languages: list[str], use_gpu: bool,
) -> list[OCREngine]:
    """Initialize OCR engines based on filter setting."""
    if engine_filter == "all":
        names = available_engines()
    else:
        names = [engine_filter]

    engines: list[OCREngine] = []
    for name in names:
        try:
            eng = create_ocr_engine(name)
            eng.initialize(languages, use_gpu)
            engines.append(eng)
            print(f"  Initialized: {name}")
        except ImportError:
            print(f"  Skipped (not installed): {name}")
        except Exception as exc:
            print(f"  Failed to init {name}: {exc}")
    return engines


def _phase_roi_selection(
    full_img: np.ndarray,
    ref_size: Tuple[int, int],
    scale_x: float,
    scale_y: float,
) -> Optional[Tuple[int, int, int, int]]:
    """Let user drag-select an ROI or accept the full image."""
    window_name = "OCR Tester - ROI Selection (drag, R=reset, Enter=confirm, Q=quit)"
    state: Dict[str, Any] = {"start": None, "end": None, "rect": None}
    img_h, img_w = full_img.shape[:2]

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state["start"] = (x, y)
            state["end"] = (x, y)
            state["rect"] = None
        elif event == cv2.EVENT_MOUSEMOVE and state["start"] is not None:
            state["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["start"] is not None:
            state["end"] = (x, y)
            x0, y0 = state["start"]
            x1, y1 = state["end"]
            left = max(0, min(x0, x1))
            top = max(0, min(y0, y1))
            right = min(img_w, max(x0, x1))
            bottom = min(img_h, max(y0, y1))
            w = right - left
            h = bottom - top
            if w > 4 and h > 4:
                state["rect"] = (left, top, w, h)
            state["start"] = None

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        frame = full_img.copy()
        if state["start"] is not None and state["end"] is not None:
            cv2.rectangle(frame, state["start"], state["end"], (0, 255, 0), 2)
        elif state["rect"] is not None:
            left, top, w, h = state["rect"]
            overlay = frame.copy()
            overlay[:, :] = (overlay[:, :] * 0.5).astype(np.uint8)
            overlay[top:top + h, left:left + w] = frame[top:top + h, left:left + w]
            frame = overlay
            cv2.rectangle(frame, (left, top), (left + w, top + h), (0, 255, 0), 2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q") or key == ord("Q"):
            cv2.destroyWindow(window_name)
            return None
        if key == ord("r") or key == ord("R"):
            state["rect"] = None
            state["start"] = None
            state["end"] = None
        if key == 13:  # Enter
            cv2.destroyWindow(window_name)
            if state["rect"] is not None:
                left, top, w, h = state["rect"]
                ref_x = int(round(left / scale_x)) if scale_x else 0
                ref_y = int(round(top / scale_y)) if scale_y else 0
                ref_w = int(round(w / scale_x)) if scale_x else 0
                ref_h = int(round(h / scale_y)) if scale_y else 0
                roi_ref = (ref_x, ref_y, ref_w, ref_h)
                print(f"ROI ref: ({ref_x}, {ref_y}, {ref_w}, {ref_h})")
                return roi_ref
            else:
                print(f"ROI ref: (0, 0, {ref_size[0]}, {ref_size[1]})  [full]")
                return (0, 0, ref_size[0], ref_size[1])

    return None  # pragma: no cover


def _print_results_table(
    engines: list[OCREngine],
    results_by_engine: dict[str, list[TextResult]],
) -> None:
    """Print a side-by-side terminal table of OCR results."""
    print("\n" + "=" * 80)
    print(f"{'Engine':<14} {'Text':<30} {'Confidence':>10} {'BBox':<20} {'Number':>8}")
    print("-" * 80)
    for eng in engines:
        results = results_by_engine.get(eng.name, [])
        if not results:
            print(f"{eng.name:<14} {'(no text detected)':<30}")
            continue
        for i, r in enumerate(results):
            bbox_str = f"({r.bbox[0]},{r.bbox[1]},{r.bbox[2]},{r.bbox[3]})" if r.bbox else "n/a"
            num = parse_number(r.text)
            num_str = f"{num}" if num is not None else "n/a"
            label = eng.name if i == 0 else ""
            # Truncate long text
            text_display = r.text[:28] + ".." if len(r.text) > 30 else r.text
            print(f"{label:<14} {text_display:<30} {r.confidence:>10.3f} {bbox_str:<20} {num_str:>8}")
    print("=" * 80)


def _draw_ocr_results(
    image: np.ndarray,
    engines: list[OCREngine],
    results_by_engine: dict[str, list[TextResult]],
) -> np.ndarray:
    """Draw bounding boxes color-coded per engine on the image."""
    display = image.copy()
    engine_colors: dict[str, Tuple[int, int, int]] = {}
    for i, eng in enumerate(engines):
        engine_colors[eng.name] = ENGINE_COLORS[i % len(ENGINE_COLORS)]

    for eng_name, results in results_by_engine.items():
        color = engine_colors.get(eng_name, (255, 255, 255))
        for r in results:
            if r.bbox is not None:
                bx, by, bw, bh = r.bbox
                cv2.rectangle(display, (bx, by), (bx + bw, by + bh), color, 2)
                label = f"{eng_name}: {r.text[:20]} ({r.confidence:.2f})"
                cv2.putText(
                    display, label,
                    (bx, max(12, by - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
                )

    # Legend
    y_offset = 20
    for eng_name, color in engine_colors.items():
        cv2.putText(display, eng_name, (5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        y_offset += 18

    return display


def _phase_live_ocr(
    capture: ScreenCapture,
    hwnd: int,
    capture_method: str,
    engines: list[OCREngine],
    roi_ref: Tuple[int, int, int, int],
    ref_size: Tuple[int, int],
    render_rect: Tuple[int, int, int, int],
    scale_x: float,
    scale_y: float,
) -> bool:
    """Live OCR phase: press Space to capture + OCR, Q to quit. Returns True to reselect ROI."""
    render_origin = (render_rect[0], render_rect[1])
    render_size = (render_rect[2], render_rect[3])
    roi_screen = map_rect(roi_ref, ref_size, render_origin, render_size)

    window_name = "OCR Tester - Live (Space=capture+OCR, R=back to ROI, Q=quit)"
    capture_count = 0
    last_display: Optional[np.ndarray] = None

    cv2.namedWindow(window_name)

    print("\n--- Live OCR Mode ---")
    print("Space=capture+OCR  R=back to ROI selection  Q=quit")
    print(f"Engines: {', '.join(e.name for e in engines)}")
    print(f"ROI ref: ({roi_ref[0]}, {roi_ref[1]}, {roi_ref[2]}, {roi_ref[3]})\n")

    while True:
        if last_display is not None:
            cv2.imshow(window_name, last_display)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break

        if key == ord("r") or key == ord("R"):
            cv2.destroyWindow(window_name)
            print("Returning to ROI selection...")
            return True

        if key == ord(" "):
            img = capture.grab_auto(roi_screen, hwnd, capture_method)
            if img is None:
                print("Capture failed.")
                continue
            capture_count += 1

            results_by_engine: dict[str, list[TextResult]] = {}
            for eng in engines:
                try:
                    results = eng.read(img)
                    results_by_engine[eng.name] = sort_by_position(results)
                except Exception as exc:
                    print(f"  {eng.name} error: {exc}")
                    results_by_engine[eng.name] = []

            print(f"\n[Capture #{capture_count}]")
            _print_results_table(engines, results_by_engine)

            # Combined text + number
            all_text = []
            for eng in engines:
                for r in results_by_engine.get(eng.name, []):
                    all_text.append(r.text)
            combined = " ".join(all_text)
            num = parse_number(combined)
            print(f"Combined text: {combined!r}")
            print(f"Parsed number: {num}")

            last_display = _draw_ocr_results(img, engines, results_by_engine)

    cv2.destroyWindow(window_name)
    print(f"\n--- Summary: {capture_count} captures ---")
    return False


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    set_process_dpi_awareness()
    config = load_config_with_fallback(DEBUG_CONFIG_PATH, CONFIG_PATH)

    try:
        hwnd, ref_size, render_rect, scale_x, scale_y = get_render_context(config)
    except Exception:
        print("Target window not found.")
        sys.exit(1)
    capture_method = get_capture_method(config)

    # Read OCR config
    ocr_cfg = config.get("ocr", {})
    languages = list(ocr_cfg.get("languages", ["ja", "en"]))
    use_gpu = bool(ocr_cfg.get("gpu", True))
    engine_filter = args.engine or str(ocr_cfg.get("engine", "all"))

    print("Initializing OCR engines...")
    engines = _init_engines(engine_filter, languages, use_gpu)
    if not engines:
        print("No OCR engines available. Install with: pip install -r requirements-ocr.txt")
        sys.exit(1)

    capture = ScreenCapture()

    # Parse --roi argument or go to ROI selection
    roi_ref: Optional[Tuple[int, int, int, int]] = None
    if args.roi:
        parts = [int(x.strip()) for x in args.roi.split(",")]
        if len(parts) == 4:
            roi_ref = (parts[0], parts[1], parts[2], parts[3])
            print(f"Using CLI ROI: ({roi_ref[0]}, {roi_ref[1]}, {roi_ref[2]}, {roi_ref[3]})")

    while True:
        if roi_ref is None:
            full_img = capture.grab_auto(render_rect, hwnd, capture_method)
            if full_img is None:
                print("Capture failed.")
                sys.exit(1)
            roi_ref = _phase_roi_selection(full_img, ref_size, scale_x, scale_y)
            if roi_ref is None:
                return

        reselect = _phase_live_ocr(
            capture=capture,
            hwnd=hwnd,
            capture_method=capture_method,
            engines=engines,
            roi_ref=roi_ref,
            ref_size=ref_size,
            render_rect=render_rect,
            scale_x=scale_x,
            scale_y=scale_y,
        )
        if not reselect:
            break
        roi_ref = None


if __name__ == "__main__":
    main()
