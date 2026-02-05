"""
Interactive verify tester tool.

Launches a two-phase workflow:
  Phase 1 — ROI selection (drag to pick sub-region, Enter to confirm)
  Phase 2 — Live match mode (Space to capture+match, A/G to add points, Q to quit)

Usage:
    python debug/verify_tester.py [template_path]

    If no template is given, scans templates/ and debug/templates/ and lets you pick.
    You can also drag-and-drop an image file onto the script.
"""
from __future__ import annotations

import glob
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
from engine.detector import TemplateDetector
from engine.mapper import map_rect, set_process_dpi_awareness
from engine.verify import VerifiedMatcher

ROOT_DIR = PARENT_DIR
CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")
DEBUG_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
DEFAULT_TOLERANCE = 30


# -- Data types for verify points -----------------------------------------

class VerifyPoint:
    """In-memory representation of a single verification point."""

    def __init__(
        self,
        x: int,
        y: int,
        rgb: Tuple[int, int, int],
        tolerance: int,
        is_global: bool = False,
        sample: int = 0,
    ) -> None:
        self.x = x
        self.y = y
        self.rgb = rgb
        self.tolerance = tolerance
        self.is_global = is_global
        self.sample = sample

    def to_tuple(self) -> tuple:
        opts: Dict[str, Any] = {}
        if self.is_global:
            opts["global"] = True
        if self.sample > 0:
            opts["sample"] = self.sample
        if opts:
            return (self.x, self.y, self.rgb, self.tolerance, opts)
        return (self.x, self.y, self.rgb, self.tolerance)


# -- Main ------------------------------------------------------------------

def _find_templates() -> List[str]:
    """Scan known template directories and return sorted list of .png files."""
    dirs = [
        os.path.join(os.path.dirname(__file__), "templates"),
        os.path.join(ROOT_DIR, "templates"),
        os.path.join(os.getcwd(), "templates"),
    ]
    seen: set[str] = set()
    results: List[str] = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(glob.glob(os.path.join(d, "*.png"))):
            abspath = os.path.abspath(f)
            if abspath not in seen:
                seen.add(abspath)
                results.append(abspath)
    return results


def _pick_template() -> str:
    """Interactive template picker when no argument given."""
    templates = _find_templates()
    if not templates:
        print("No .png templates found in templates/ or debug/templates/.")
        sys.exit(1)
    if len(templates) == 1:
        print(f"Using only template found: {templates[0]}")
        return templates[0]
    print("Available templates:")
    for i, t in enumerate(templates, 1):
        rel = os.path.relpath(t, ROOT_DIR)
        print(f"  {i}. {rel}")
    while True:
        choice = input(f"Pick template [1-{len(templates)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(templates):
            return templates[int(choice) - 1]
        print("Invalid choice.")


def main() -> None:
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    if len(sys.argv) >= 2:
        # Support drag-and-drop (Windows may wrap path in quotes)
        template_arg = sys.argv[1].strip('"').strip("'")
    else:
        template_arg = None

    set_process_dpi_awareness()
    config = load_config_with_fallback(DEBUG_CONFIG_PATH, CONFIG_PATH)
    try:
        hwnd, ref_size, render_rect, scale_x, scale_y = get_render_context(config)
    except Exception:
        print("Target window not found.")
        sys.exit(1)
    capture_method = get_capture_method(config)

    # Resolve template path
    if template_arg is not None:
        template_path = _resolve_template(template_arg)
        if not os.path.exists(template_path):
            print(f"Template not found: {template_path}")
            sys.exit(1)
    else:
        template_path = _pick_template()
    detector = TemplateDetector(template_path)
    template_name = os.path.basename(template_path)

    capture = ScreenCapture()

    # Capture full render area for ROI selection
    full_img = capture.grab_auto(render_rect, hwnd, capture_method)
    if full_img is None:
        print("Capture failed.")
        sys.exit(1)

    # ---- Phase 1: ROI Selection ----
    roi_ref = _phase_roi_selection(full_img, ref_size, scale_x, scale_y)
    if roi_ref is None:
        return

    # Read multi-scale config
    marker_cfg = config.get("marker_detection", {})
    multi_scale_range = float(marker_cfg.get("multi_scale_range", 0.03))
    multi_scale_steps = int(marker_cfg.get("multi_scale_steps", 3))

    # ---- Phase 2: Live Match Mode ----
    _phase_live_match(
        capture=capture,
        hwnd=hwnd,
        capture_method=capture_method,
        detector=detector,
        template_name=template_name,
        roi_ref=roi_ref,
        ref_size=ref_size,
        render_rect=render_rect,
        scale_x=scale_x,
        scale_y=scale_y,
        multi_scale_range=multi_scale_range,
        multi_scale_steps=multi_scale_steps,
    )


def _resolve_template(arg: str) -> str:
    if os.path.isabs(arg) or os.path.exists(arg):
        return arg
    # Try relative to cwd, then debug/templates, then root templates
    candidates = [
        arg,
        os.path.join(os.path.dirname(__file__), "templates", arg),
        os.path.join(ROOT_DIR, "templates", arg),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return arg


# --------------------------------------------------------------------------
# Phase 1: ROI Selection
# --------------------------------------------------------------------------

def _phase_roi_selection(
    full_img: np.ndarray,
    ref_size: Tuple[int, int],
    scale_x: float,
    scale_y: float,
) -> Optional[Tuple[int, int, int, int]]:
    """Let user drag-select an ROI or accept the full image. Returns roi_ref or None on quit."""

    window_name = "Verify Tester - ROI Selection (drag, R=reset, Enter=confirm, Q=quit)"
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
            # Active drag — draw rectangle
            cv2.rectangle(frame, state["start"], state["end"], (0, 255, 0), 2)
        elif state["rect"] is not None:
            # Confirmed selection — darken outside
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

    # Unreachable, but makes type-checker happy
    return None  # pragma: no cover


# --------------------------------------------------------------------------
# Phase 2: Live Match Mode
# --------------------------------------------------------------------------

def _phase_live_match(
    capture: ScreenCapture,
    hwnd: int,
    capture_method: str,
    detector: TemplateDetector,
    template_name: str,
    roi_ref: Tuple[int, int, int, int],
    ref_size: Tuple[int, int],
    render_rect: Tuple[int, int, int, int],
    scale_x: float,
    scale_y: float,
    multi_scale_range: float = 0.03,
    multi_scale_steps: int = 3,
) -> None:
    render_origin = (render_rect[0], render_rect[1])
    render_size = (render_rect[2], render_rect[3])

    window_name = (
        "Verify Tester - Live "
        "(Space=capture, A=rel point, G=global point, "
        "+/-=sample, [ ]=tolerance, T=set tol, E=export, D=del, R=roi, Q=quit)"
    )
    verify_points: List[VerifyPoint] = []
    capture_count = 0
    pass_count = 0
    fail_count = 0
    current_tolerance = DEFAULT_TOLERANCE
    last_img: Optional[np.ndarray] = None
    last_match_loc: Optional[Tuple[int, int]] = None
    last_match_size: Optional[Tuple[int, int]] = None
    last_score: float = 0.0
    last_second: Optional[float] = None
    last_match_debug: Optional[Dict[str, Any]] = None
    last_point_results: List[Dict[str, Any]] = []
    cursor_pos: List[Optional[Tuple[int, int]]] = [None]

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_LBUTTONDOWN:
            cursor_pos[0] = (x, y)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    roi_screen = map_rect(roi_ref, ref_size, render_origin, render_size)

    print("\n--- Live Match Mode ---")
    print("Space=capture+match  A=add relative point  G=add global point")
    print("+/-=adjust sample radius  [ ]=adjust tolerance  T=set tolerance")
    print("E=export setup  D=delete last point  R=back to ROI  Q=quit")
    print(f"Current tolerance: ±{current_tolerance}\n")

    while True:
        # Draw current state
        display = _draw_display(
            last_img, last_match_loc, last_match_size,
            last_score, last_second, verify_points,
            last_point_results, roi_screen, ref_size,
            render_origin, render_size, scale_x, scale_y,
        )
        if display is not None:
            cv2.imshow(window_name, display)
        elif last_img is not None:
            cv2.imshow(window_name, last_img)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break

        if key == ord("r") or key == ord("R"):
            cv2.destroyWindow(window_name)
            # Signal to go back to ROI selection — handled by returning
            print("Returning to ROI selection...")
            return

        if key == ord("["):
            current_tolerance = max(0, current_tolerance - 1)
            print(f"Current tolerance: ±{current_tolerance}")
        if key == ord("]"):
            current_tolerance = min(255, current_tolerance + 1)
            print(f"Current tolerance: ±{current_tolerance}")
        if key == ord("t") or key == ord("T"):
            try:
                raw = input(f"Set tolerance (0-255), current {current_tolerance}: ").strip()
            except EOFError:
                raw = ""
            if raw:
                try:
                    current_tolerance = int(raw)
                    if current_tolerance < 0:
                        current_tolerance = 0
                    if current_tolerance > 255:
                        current_tolerance = 255
                except ValueError:
                    print("Invalid tolerance value.")
            print(f"Current tolerance: ±{current_tolerance}")

        if key == ord(" "):
            # Capture + match + verify
            img = capture.grab_auto(roi_screen, hwnd, capture_method)
            if img is None:
                print("Capture failed.")
                continue
            last_img = img
            capture_count += 1

            score, loc, size, second_best = detector.match_with_location(
                img, scale_x, scale_y,
                multi_scale_range=multi_scale_range,
                multi_scale_steps=multi_scale_steps,
            )
            last_score = score
            last_second = second_best
            last_match_loc = loc
            last_match_size = size
            last_match_debug = detector.last_match_debug()

            # Run verification
            point_results = _run_verification(
                img, loc, verify_points, roi_screen, ref_size, render_origin, render_size, template_name,
            )
            last_point_results = point_results

            # Print terminal summary
            _print_capture_summary(
                capture_count, score, second_best, loc, verify_points, point_results, last_match_debug,
            )

            # Update stats
            if verify_points:
                all_pass = all(r["passed"] for r in point_results)
                if all_pass:
                    pass_count += 1
                else:
                    fail_count += 1
                total = pass_count + fail_count
                pct = (pass_count / total * 100) if total > 0 else 0
                print(f"Running: {pass_count} pass / {fail_count} fail ({pct:.0f}%)")

        if key == ord("a") or key == ord("A"):
            # Add relative verify point at cursor
            if last_img is None:
                print("No capture yet. Press Space first.")
                continue
            if last_match_loc is None:
                print("No match found. Cannot add relative point without a match.")
                continue
            cursor = cursor_pos[0]
            if cursor is None:
                continue
            cx, cy = cursor
            if cy < 0 or cx < 0 or cy >= last_img.shape[0] or cx >= last_img.shape[1]:
                continue
            rel_x = cx - last_match_loc[0]
            rel_y = cy - last_match_loc[1]
            b, g, r = last_img[cy, cx]
            rgb = (int(r), int(g), int(b))
            vp = VerifyPoint(rel_x, rel_y, rgb, current_tolerance, is_global=False)
            verify_points.append(vp)
            print(f"  Added REL point #{len(verify_points)}: ({rel_x}, {rel_y}) RGB{rgb} tol={current_tolerance}")

        if key == ord("g") or key == ord("G"):
            # Add global verify point at cursor
            if last_img is None:
                print("No capture yet. Press Space first.")
                continue
            cursor = cursor_pos[0]
            if cursor is None:
                continue
            cx, cy = cursor
            if cy < 0 or cx < 0 or cy >= last_img.shape[0] or cx >= last_img.shape[1]:
                continue
            # Convert image coords → screen coords → ref coords
            screen_x = roi_screen[0] + cx
            screen_y = roi_screen[1] + cy
            ref_x = int(round((screen_x - render_origin[0]) / scale_x)) if scale_x else 0
            ref_y = int(round((screen_y - render_origin[1]) / scale_y)) if scale_y else 0
            b, g, r = last_img[cy, cx]
            rgb = (int(r), int(g), int(b))
            vp = VerifyPoint(ref_x, ref_y, rgb, current_tolerance, is_global=True)
            verify_points.append(vp)
            print(f"  Added GBL point #{len(verify_points)}: ({ref_x}, {ref_y}) RGB{rgb} tol={current_tolerance}")

        if key == ord("+") or key == ord("="):
            if verify_points:
                verify_points[-1].sample += 1
                print(f"  Point #{len(verify_points)} sample radius -> {verify_points[-1].sample}")

        if key == ord("-") or key == ord("_"):
            if verify_points and verify_points[-1].sample > 0:
                verify_points[-1].sample -= 1
                print(f"  Point #{len(verify_points)} sample radius -> {verify_points[-1].sample}")

        if key == ord("d") or key == ord("D"):
            if verify_points:
                removed = verify_points.pop()
                kind = "GBL" if removed.is_global else "REL"
                print(f"  Deleted point #{len(verify_points) + 1} ({kind})")

        if key == ord("e") or key == ord("E"):
            _print_export(roi_ref, verify_points)

    cv2.destroyWindow(window_name)
    _print_exit_summary(verify_points, pass_count, fail_count, capture_count)


# --------------------------------------------------------------------------
# Verification runner
# --------------------------------------------------------------------------

def _run_verification(
    img: np.ndarray,
    match_loc: Optional[Tuple[int, int]],
    points: List[VerifyPoint],
    roi_screen: Tuple[int, int, int, int],
    ref_size: Tuple[int, int],
    render_origin: Tuple[int, int],
    render_size: Tuple[int, int],
    template_name: str,
) -> List[Dict[str, Any]]:
    """Run verification on all points and return per-point results."""
    results: List[Dict[str, Any]] = []
    if not points:
        return results

    verifier = VerifiedMatcher()
    img_h, img_w = img.shape[:2]

    for i, vp in enumerate(points):
        result: Dict[str, Any] = {
            "index": i,
            "point": vp,
            "passed": False,
            "actual_rgb": None,
            "drift": None,
        }

        # Calculate absolute image coords for this point
        if vp.is_global:
            sx = render_size[0] / ref_size[0] if ref_size[0] else 1.0
            sy = render_size[1] / ref_size[1] if ref_size[1] else 1.0
            screen_x = render_origin[0] + vp.x * sx
            screen_y = render_origin[1] + vp.y * sy
            abs_x = round(screen_x - roi_screen[0])
            abs_y = round(screen_y - roi_screen[1])
        else:
            if match_loc is None:
                results.append(result)
                continue
            abs_x = match_loc[0] + vp.x
            abs_y = match_loc[1] + vp.y

        if vp.sample > 0:
            passed, _ = verifier._check_sample_region(
                img, abs_x, abs_y, vp.sample, vp.rgb, vp.tolerance,
            )
            result["passed"] = passed
        else:
            if 0 <= abs_y < img_h and 0 <= abs_x < img_w:
                b, g, r = img[abs_y, abs_x]
                actual = (int(r), int(g), int(b))
                result["actual_rgb"] = actual
                result["drift"] = (
                    actual[0] - vp.rgb[0],
                    actual[1] - vp.rgb[1],
                    actual[2] - vp.rgb[2],
                )
                result["passed"] = verifier._colors_match(actual, vp.rgb, vp.tolerance)

        # For sample mode, also grab center pixel for display
        if vp.sample > 0 and 0 <= abs_y < img_h and 0 <= abs_x < img_w:
            b, g, r = img[abs_y, abs_x]
            actual = (int(r), int(g), int(b))
            result["actual_rgb"] = actual
            result["drift"] = (
                actual[0] - vp.rgb[0],
                actual[1] - vp.rgb[1],
                actual[2] - vp.rgb[2],
            )

        result["abs_pos"] = (abs_x, abs_y)
        results.append(result)

    return results


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

def _draw_display(
    img: Optional[np.ndarray],
    match_loc: Optional[Tuple[int, int]],
    match_size: Optional[Tuple[int, int]],
    score: float,
    second_best: Optional[float],
    points: List[VerifyPoint],
    point_results: List[Dict[str, Any]],
    roi_screen: Tuple[int, int, int, int],
    ref_size: Tuple[int, int],
    render_origin: Tuple[int, int],
    render_size: Tuple[int, int],
    scale_x: float,
    scale_y: float,
) -> Optional[np.ndarray]:
    if img is None:
        return None

    display = img.copy()

    if match_loc is not None and match_size is not None:
        # Determine match rectangle color based on verification
        if not points:
            rect_color = (255, 128, 0)  # Blue-ish (no verify points)
        elif point_results and all(r["passed"] for r in point_results):
            rect_color = (0, 255, 0)  # Green (all pass)
        else:
            rect_color = (0, 0, 255)  # Red (any fail)

        cv2.rectangle(
            display,
            match_loc,
            (match_loc[0] + match_size[0], match_loc[1] + match_size[1]),
            rect_color, 2,
        )

        # Draw score text
        delta = score - second_best if second_best is not None else None
        delta_str = f"  delta={delta:.3f}" if delta is not None else ""
        cv2.putText(
            display, f"score={score:.3f}{delta_str}",
            (match_loc[0], max(12, match_loc[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, rect_color, 1,
        )
    elif img is not None:
        cv2.putText(
            display, f"No match - score {score:.3f}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
        )

    # Draw verify points
    for i, vp in enumerate(points):
        res = point_results[i] if i < len(point_results) else None
        abs_pos = res.get("abs_pos") if res else None

        if abs_pos is None:
            # Calculate it for display
            if vp.is_global:
                sx = render_size[0] / ref_size[0] if ref_size[0] else 1.0
                sy = render_size[1] / ref_size[1] if ref_size[1] else 1.0
                screen_x = render_origin[0] + vp.x * sx
                screen_y = render_origin[1] + vp.y * sy
                abs_x = int(round(screen_x - roi_screen[0]))
                abs_y = int(round(screen_y - roi_screen[1]))
            else:
                if match_loc is None:
                    continue
                abs_x = match_loc[0] + vp.x
                abs_y = match_loc[1] + vp.y
            abs_pos = (abs_x, abs_y)

        passed = res["passed"] if res else False
        color = (0, 255, 0) if passed else (0, 0, 255)
        kind = "G" if vp.is_global else "R"

        if vp.sample > 0:
            # Draw box for sample region
            box_r = vp.sample
            cv2.rectangle(
                display,
                (abs_pos[0] - box_r, abs_pos[1] - box_r),
                (abs_pos[0] + box_r, abs_pos[1] + box_r),
                color, 1,
            )
        else:
            cv2.circle(display, abs_pos, 4, color, -1)

        # Label
        label_parts = [f"P{i + 1}{kind}"]
        if res and res.get("actual_rgb") is not None:
            ar = res["actual_rgb"]
            label_parts.append(f"({ar[0]},{ar[1]},{ar[2]})")
        if passed:
            label_parts.append("OK")
        else:
            label_parts.append("FAIL")
        if vp.sample > 0:
            label_parts.append(f"s={vp.sample}")

        label = " ".join(label_parts)
        cv2.putText(
            display, label,
            (abs_pos[0] + 8, abs_pos[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
        )

    return display


# --------------------------------------------------------------------------
# Terminal output
# --------------------------------------------------------------------------

def _print_capture_summary(
    capture_num: int,
    score: float,
    second_best: Optional[float],
    match_loc: Optional[Tuple[int, int]],
    points: List[VerifyPoint],
    results: List[Dict[str, Any]],
    match_debug: Optional[Dict[str, Any]] = None,
) -> None:
    delta = score - second_best if second_best is not None else None
    score_str = f"{score:.8f}"
    delta_str = f"{delta:.8f}" if delta is not None else "n/a"
    second_str = f"{second_best:.8f}" if second_best is not None else "n/a"

    if not points:
        verify_str = "no verify points"
    else:
        passed = sum(1 for r in results if r["passed"])
        total = len(results)
        status = "PASS" if passed == total else "FAIL"
        verify_str = f"{status} ({passed}/{total})"

    match_str = f"at ({match_loc[0]},{match_loc[1]})" if match_loc else "no match"
    print(
        f"[{capture_num}] Score: {score_str}  Second: {second_str}  "
        f"Delta: {delta_str}  {match_str}  Verify: {verify_str}"
    )

    if match_debug:
        use_bgr = match_debug.get("use_bgr")
        method = match_debug.get("method") or "unknown"
        mask = match_debug.get("mask")
        scales = match_debug.get("scales")
        range_val = match_debug.get("range")
        if use_bgr is True:
            mode = "BGR"
        elif use_bgr is False:
            mode = "GRAY"
        else:
            mode = "n/a"
        range_str = f"{range_val:.3f}" if isinstance(range_val, (int, float)) else "n/a"
        print(
            f"      Match: {mode}  Method: {method}  Mask: {mask}  "
            f"Scales: {scales}  Range: {range_str}"
        )

    for r in results:
        vp: VerifyPoint = r["point"]
        idx = r["index"] + 1
        kind = "gbl" if vp.is_global else "rel"
        expect_str = f"({vp.rgb[0]},{vp.rgb[1]},{vp.rgb[2]})"
        tol_str = f"±{vp.tolerance}"

        actual = r.get("actual_rgb")
        if actual is not None:
            actual_str = f"({actual[0]},{actual[1]},{actual[2]})"
        else:
            actual_str = "n/a"

        drift = r.get("drift")
        if drift is not None:
            drift_str = f"+{drift[0]:+d},{drift[1]:+d},{drift[2]:+d}".replace("+-", "-")
        else:
            drift_str = "n/a"

        status = "PASS" if r["passed"] else "FAIL"
        sample_str = f"  sample={vp.sample}" if vp.sample > 0 else ""
        print(
            f"  P{idx} {kind}({vp.x},{vp.y}): "
            f"expect {expect_str}{tol_str}  "
            f"actual {actual_str}  "
            f"drift {drift_str}  "
            f"  {status}{sample_str}"
        )


def _print_export(
    roi_ref: Tuple[int, int, int, int],
    points: List[VerifyPoint],
) -> None:
    print("\n--- Export ---")
    print(f"roi_ref=({roi_ref[0]}, {roi_ref[1]}, {roi_ref[2]}, {roi_ref[3]}),  # x, y, w, h")
    print("verify_pixels=[")
    for vp in points:
        t = vp.to_tuple()
        if len(t) == 5:
            x, y, rgb, tol, opts = t
            opts_str = repr(opts)
            print(f"    ({x}, {y}, ({rgb[0]}, {rgb[1]}, {rgb[2]}), {tol}, {opts_str}),")
        else:
            x, y, rgb, tol = t
            print(f"    ({x}, {y}, ({rgb[0]}, {rgb[1]}, {rgb[2]}), {tol}),")
    print("]")


def _print_exit_summary(
    points: List[VerifyPoint],
    pass_count: int,
    fail_count: int,
    capture_count: int,
) -> None:
    print("\n--- Summary ---")
    total = pass_count + fail_count
    if total > 0:
        pct = pass_count / total * 100
        print(f"Captures: {capture_count}  Verified: {total}  Pass: {pass_count}  Fail: {fail_count} ({pct:.0f}% pass)")

    if points:
        print("\nverify_pixels=[")
        for vp in points:
            t = vp.to_tuple()
            if len(t) == 5:
                x, y, rgb, tol, opts = t
                opts_str = repr(opts)
                print(f"    ({x}, {y}, ({rgb[0]}, {rgb[1]}, {rgb[2]}), {tol}, {opts_str}),")
            else:
                x, y, rgb, tol = t
                print(f"    ({x}, {y}, ({rgb[0]}, {rgb[1]}, {rgb[2]}), {tol}),")
        print("]")
    else:
        print("No verify points defined.")


if __name__ == "__main__":
    main()
