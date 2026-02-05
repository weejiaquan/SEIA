from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from common import DEFAULT_CONFIG_PATH, get_capture_method, get_render_context, load_config
from engine.capture import ScreenCapture
from engine.mapper import map_rect, set_process_dpi_awareness


WINDOW_NAME = "Stability Tester (drag ROI, q=quit, r=reset, +/-=pixel_thresh, [/]=change_ratio)"


def _parse_roi_arg(values: list[str]) -> Optional[Tuple[int, int, int, int]]:
    if not values:
        return None
    if len(values) == 1:
        raw = values[0].replace(",", " ").replace("x", " ").strip()
        parts = [p for p in raw.split() if p]
    else:
        parts = values
    if len(parts) != 4:
        raise ValueError("roi_ref must have 4 values: x y w h")
    return tuple(int(v) for v in parts)  # type: ignore[return-value]


def _parse_keys_arg(value: str) -> list[str]:
    if not value:
        return []
    cleaned = value.replace(",", " ").strip()
    return [k for k in cleaned.split() if k]


def _print_snippet(
    roi_ref: Optional[Tuple[int, int, int, int]],
    stable_count_target: int,
    pixel_threshold: int,
    change_ratio: float,
    args: argparse.Namespace,
    press_keys: list[str],
) -> None:
    if roi_ref is None:
        print("No ROI set; cannot export snippet.")
        return
    print()
    print("Copy-paste ready function call:")
    if args.emit == "press_until_stable":
        keys_literal = press_keys if press_keys else ["space"]
        print("press_until_stable(")
        print(f"    {keys_literal},")
        print(f"    roi_ref={roi_ref},")
        print(f"    timeout_s={args.timeout},")
        print(f"    poll_interval_s={args.poll_interval},")
        print(f"    stable_count={stable_count_target},")
        print(f"    pixel_threshold={pixel_threshold},")
        print(f"    change_ratio={change_ratio},")
        print(f"    require=False,")
        print(")")
    else:
        print("wait_for_stable(")
        print(f"    roi_ref={roi_ref},")
        print(f"    timeout_s={args.timeout},")
        print(f"    poll_interval_s={args.poll_interval},")
        print(f"    stable_count={stable_count_target},")
        print(f"    pixel_threshold={pixel_threshold},")
        print(f"    change_ratio={change_ratio},")
        if args.wait_for_change:
            print("    wait_for_change=True,")
        print("    require=False,")
        print(")")


def _frames_differ(
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
    pixel_threshold: int,
    change_ratio: float,
    mask: Optional[np.ndarray] = None,
) -> Tuple[bool, float, np.ndarray]:
    """Return (differs, ratio, heatmap) for two BGR frames."""
    diff = cv2.absdiff(prev_bgr, curr_bgr)
    max_diff = np.max(diff, axis=2)
    heatmap_source = max_diff.copy()
    if mask is not None:
        max_diff = max_diff[mask > 0]
    total = max_diff.size
    if total == 0:
        return False, 0.0, heatmap_source
    changed = int(np.count_nonzero(max_diff > pixel_threshold))
    ratio = changed / total
    return ratio >= change_ratio, ratio, heatmap_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive ROI stability tester.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.json")
    parser.add_argument("--roi-ref", nargs="+", help="ROI in ref coords: x y w h or 'x,y,w,h'")
    parser.add_argument("--pixel-threshold", type=int, default=20, help="Per-channel diff threshold")
    parser.add_argument("--change-ratio", type=float, default=0.01, help="Changed pixel ratio threshold")
    parser.add_argument("--stable-count", type=int, default=3, help="Consecutive stable frames required")
    parser.add_argument("--emit", choices=["wait_for_stable", "press_until_stable"], default="wait_for_stable")
    parser.add_argument("--keys", default="", help="Keys for press_until_stable (e.g. \"1,space\")")
    parser.add_argument("--timeout", type=float, default=10.0, help="Timeout for snippet output")
    parser.add_argument("--poll-interval", type=float, default=0.2, help="Poll interval for snippet output")
    parser.add_argument("--wait-for-change", action="store_true", help="Include wait_for_change=True in snippet")
    args = parser.parse_args()

    set_process_dpi_awareness()
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Missing config: {args.config}")
    config = load_config(args.config)
    capture_method = get_capture_method(config)
    hwnd, ref_size, render_rect, scale_x, scale_y = get_render_context(config)
    render_x, render_y, render_w, render_h = render_rect

    capture = ScreenCapture()

    # State
    pixel_threshold = int(args.pixel_threshold)
    change_ratio = float(args.change_ratio)
    stable_count_target = int(args.stable_count)
    roi_ref: Optional[Tuple[int, int, int, int]] = _parse_roi_arg(args.roi_ref or [])
    selecting = {"start": None, "end": None, "cursor": None}
    prev_frame: Optional[np.ndarray] = None
    consecutive_stable = 0
    mask: Optional[np.ndarray] = None
    press_keys = _parse_keys_arg(args.keys)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        selecting["cursor"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            selecting["start"] = (x, y)
            selecting["end"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and selecting["start"] is not None:
            selecting["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and selecting["start"] is not None:
            selecting["end"] = (x, y)
            x0, y0 = selecting["start"]
            x1, y1 = selecting["end"]
            left = min(x0, x1)
            top = min(y0, y1)
            width = abs(x1 - x0)
            height = abs(y1 - y0)
            if width > 4 and height > 4:
                # Convert from render-relative pixels to reference coords
                ref_x = int(round(left / scale_x))
                ref_y = int(round(top / scale_y))
                ref_w = int(round(width / scale_x))
                ref_h = int(round(height / scale_y))
                nonlocal roi_ref, prev_frame, consecutive_stable
                roi_ref = (ref_x, ref_y, ref_w, ref_h)
                prev_frame = None
                consecutive_stable = 0
                print(f"ROI set: roi_ref=({ref_x}, {ref_y}, {ref_w}, {ref_h})")
            selecting["start"] = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    print("Stability Tester")
    if roi_ref is None:
        print("  Drag to select ROI on the captured frame")
    else:
        print(f"  ROI preset: roi_ref={roi_ref}")
    print("  +/- : adjust pixel_threshold")
    print("  [/] : adjust change_ratio")
    print("  r   : reset baseline")
    print("  c   : print copy-paste snippet")
    print("  q   : quit")
    print()

    try:
        while True:
            # Capture full render area for display
            full_img = capture.grab_auto((render_x, render_y, render_w, render_h), hwnd, capture_method)
            if full_img is None:
                time.sleep(0.1)
                continue

            display = full_img.copy()

            # Draw current ROI selection in progress
            if selecting["start"] is not None and selecting["end"] is not None:
                x0, y0 = selecting["start"]
                x1, y1 = selecting["end"]
                cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 255), 1)

            heatmap_display: Optional[np.ndarray] = None
            ratio_text = "n/a"
            stable_text = "n/a"
            differs = False

            if roi_ref is not None:
                # Capture ROI
                roi_screen = map_rect(roi_ref, ref_size, (render_x, render_y), (render_w, render_h))
                roi_img = capture.grab_auto(roi_screen, hwnd, capture_method)

                if roi_img is not None:
                    if prev_frame is not None and prev_frame.shape == roi_img.shape:
                        differs, ratio, heatmap_raw = _frames_differ(
                            prev_frame, roi_img, pixel_threshold, change_ratio, mask
                        )
                        ratio_text = f"{ratio:.4f} ({ratio * 100:.2f}%)"

                        if differs:
                            consecutive_stable = 0
                        else:
                            consecutive_stable += 1

                        stable_text = f"{consecutive_stable}/{stable_count_target}"

                        # Build coloured heatmap
                        heat_norm = np.clip(heatmap_raw.astype(np.float32) / max(pixel_threshold, 1) * 255, 0, 255).astype(np.uint8)
                        heatmap_display = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)

                    prev_frame = roi_img

                    # Draw ROI on main display (screen-relative to render area)
                    rx = roi_screen[0] - render_x
                    ry = roi_screen[1] - render_y
                    rw, rh = roi_screen[2], roi_screen[3]
                    colour = (0, 255, 0) if consecutive_stable >= stable_count_target else (0, 0, 255)
                    cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), colour, 2)

            # HUD text
            lines = [
                f"pixel_thresh: {pixel_threshold}  change_ratio: {change_ratio:.4f}",
                f"change: {ratio_text}  stable: {stable_text}  differs: {int(differs)}",
            ]
            if roi_ref is not None:
                lines.append(f"roi_ref: {roi_ref}")
            for i, line in enumerate(lines):
                cv2.putText(display, line, (10, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(WINDOW_NAME, display)
            if heatmap_display is not None:
                cv2.imshow("Diff Heatmap", heatmap_display)

            key = cv2.waitKey(50) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                prev_frame = None
                consecutive_stable = 0
                print("Baseline reset")
            elif key == ord("+") or key == ord("="):
                pixel_threshold = min(255, pixel_threshold + 5)
                print(f"pixel_threshold = {pixel_threshold}")
            elif key == ord("-") or key == ord("_"):
                pixel_threshold = max(1, pixel_threshold - 5)
                print(f"pixel_threshold = {pixel_threshold}")
            elif key == ord("]"):
                change_ratio = min(1.0, change_ratio + 0.005)
                print(f"change_ratio = {change_ratio:.4f}")
            elif key == ord("["):
                change_ratio = max(0.0, change_ratio - 0.005)
                print(f"change_ratio = {change_ratio:.4f}")
            elif key == ord("c"):
                _print_snippet(roi_ref, stable_count_target, pixel_threshold, change_ratio, args, press_keys)

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

    _print_snippet(roi_ref, stable_count_target, pixel_threshold, change_ratio, args, press_keys)


if __name__ == "__main__":
    main()
