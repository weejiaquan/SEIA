from __future__ import annotations

import os
from typing import Any

import cv2

from common import DEFAULT_CONFIG_PATH, get_capture_method, get_render_context, load_config
from engine.capture import ScreenCapture
from engine.mapper import set_process_dpi_awareness


DEFAULT_TOLERANCE = 30


def main() -> None:
    set_process_dpi_awareness()
    if not os.path.exists(DEFAULT_CONFIG_PATH):
        raise FileNotFoundError(f"Missing config: {DEFAULT_CONFIG_PATH}")
    config = load_config(DEFAULT_CONFIG_PATH)
    capture_method = get_capture_method(config)
    hwnd, ref_size, render_rect, scale_x, scale_y = get_render_context(config)
    render_x, render_y, render_w, render_h = render_rect

    capture = ScreenCapture()
    img = capture.grab_auto((render_x, render_y, render_w, render_h), hwnd, capture_method)
    if img is None:
        raise RuntimeError("Capture failed.")

    window_name = "ROI Picker (drag to select, press A to mark pixel, q to quit)"
    selecting = {"start": None, "end": None, "rect": None, "cursor": None}
    marked_pixels: list[dict[str, Any]] = []

    def on_mouse(event, x, y, _flags, _param) -> None:
        selecting["cursor"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            selecting["start"] = (x, y)
            selecting["end"] = (x, y)
            selecting["rect"] = None
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
            selecting["rect"] = (left, top, width, height)
            selecting["start"] = None

            ref_x = int(round(left / scale_x)) if scale_x else 0
            ref_y = int(round(top / scale_y)) if scale_y else 0
            ref_w = int(round(width / scale_x)) if scale_x else 0
            ref_h = int(round(height / scale_y)) if scale_y else 0
            print(
                f"ROI ref: {{\"x\": {ref_x}, \"y\": {ref_y}, \"w\": {ref_w}, \"h\": {ref_h}}}"
            )

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        frame = img.copy()
        if selecting["start"] is not None and selecting["end"] is not None:
            x0, y0 = selecting["start"]
            x1, y1 = selecting["end"]
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        elif selecting["rect"] is not None:
            left, top, width, height = selecting["rect"]
            cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
        for pixel in marked_pixels:
            px, py = pixel["local"]
            cv2.circle(frame, (px, py), 3, (0, 255, 255), -1)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key in (ord("a"), ord("A")):
            cursor = selecting.get("cursor")
            if cursor is None:
                continue
            x, y = cursor
            if y < 0 or x < 0 or y >= img.shape[0] or x >= img.shape[1]:
                continue
            b, g, r = img[y, x]
            rgb = (int(r), int(g), int(b))
            ref_x = int(round(x / scale_x)) if scale_x else 0
            ref_y = int(round(y / scale_y)) if scale_y else 0
            marked_pixels.append(
                {"ref": (ref_x, ref_y), "rgb": rgb, "tol": DEFAULT_TOLERANCE, "local": (x, y)}
            )
            print(
                f"PIXEL ref: {{\"x\": {ref_x}, \"y\": {ref_y}, \"rgb\": [{rgb[0]}, {rgb[1]}, {rgb[2]}], "
                f"\"tolerance\": {DEFAULT_TOLERANCE}}}"
            )

    cv2.destroyAllWindows()
    if marked_pixels:
        print("\nverify_pixels=[")
        for pixel in marked_pixels:
            rx, ry = pixel["ref"]
            rgb = pixel["rgb"]
            tol = pixel["tol"]
            print(f"    ({rx}, {ry}, ({rgb[0]}, {rgb[1]}, {rgb[2]}), {tol}),")
        print("]")


if __name__ == "__main__":
    main()
