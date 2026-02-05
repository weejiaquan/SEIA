from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import logging
import os
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import pydirectinput

from .capture import ScreenCapture
from .detector import (
    FeatureDetector,
    TemplateDetector,
    parse_pixel_points,
    pixel_points_check,
    pixel_points_check_with_results,
)
from .mapper import (
    find_window,
    get_client_origin_and_size,
    is_window_minimized,
    map_rect,
    focus_window,
    set_process_dpi_awareness,
    set_window_topmost,
    set_window_client_size,
)
from .debug_draw import draw_match_rect, draw_verification_results
from .ocr import (
    OCREngine,
    TextResult,
    available_engines,
    create_ocr_engine,
    parse_number,
    sort_by_position,
)
from .verify import (
    verify_match,
    add_pixel_verification,
    get_last_failed_offset,
    get_last_verification_results,
)


def _get_cursor_pos() -> Tuple[int, int]:
    point = wintypes.POINT()
    if ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
        return int(point.x), int(point.y)
    return 0, 0


def _to_list(value: Optional[Tuple[int, int] | Tuple[int, int, int] | Tuple[int, int, int, int]]) -> Optional[list[int]]:
    if value is None:
        return None
    return [int(v) for v in value]


def _serialize_results(results: list[dict[str, object]]) -> list[dict[str, object]]:
    serialized: list[dict[str, object]] = []
    for result in results:
        serialized.append(
            {
                "abs_pos": _to_list(result.get("abs_pos")),
                "rel_pos": _to_list(result.get("rel_pos")),
                "ref_pos": _to_list(result.get("ref_pos")),
                "expected": _to_list(result.get("expected")),
                "actual": _to_list(result.get("actual")),
                "tolerance": int(result.get("tolerance") or 0),
                "passed": bool(result.get("passed")),
                "sample": int(result.get("sample") or 0),
                "match_pos": _to_list(result.get("match_pos")),
                "is_global": bool(result.get("is_global")) if "is_global" in result else None,
            }
        )
    return serialized


@dataclass(frozen=True)
class MatchResult:
    score: float
    top_left: Tuple[int, int]
    center: Tuple[int, int]
    template_size: Tuple[int, int]
    ref_center: Tuple[int, int]
    second_best: Optional[float] = None
    score_delta: Optional[float] = None
    method: str = "template"
    match_count: Optional[int] = None
    match_total: Optional[int] = None


class AutomationEngine:
    def __init__(self, config_path: str = "config.json") -> None:
        set_process_dpi_awareness()
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self._hwnd: Optional[int] = None
        self._capture = ScreenCapture()
        self._templates: Dict[str, TemplateDetector | FeatureDetector] = {}
        self._ocr_engines: Dict[str, OCREngine] = {}
        self._ocr_config: Dict[str, Any] = {}
        self._use_pixel_verification = False
        self._debug_screenshot = False
        self._debug_output_dir = "debug_out"
        self._debug_always_capture = False
        self._debug_keep_history = False
        self._capture_method = "screen"
        self._auto_resize = False
        self._resized_once = False
        self._force_topmost = False
        self._load_config()

    def _write_debug_meta(self, debug_path: str, meta: dict[str, object]) -> None:
        try:
            meta_path = f"{debug_path}.json"
            with open(meta_path, "w", encoding="utf-8") as handle:
                json.dump(meta, handle, indent=2, ensure_ascii=True)
        except Exception as exc:
            logging.debug("Failed to save debug metadata: %s", exc)

    def _load_config(self) -> None:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Missing config: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            self._config = json.load(f)
        target = self._config.get("target", {})
        self._title_substring = target.get("window_title_substring", "")
        self._process_name = target.get("process_name", "") or None
        ref = self._config.get("reference_resolution", {})
        self._ref_size = (int(ref.get("w", 1920)), int(ref.get("h", 1080)))
        self._render_cfg = self._config.get("render_area", {"mode": "stretch"})
        window_cfg = self._config.get("window", {})
        self._auto_resize = bool(window_cfg.get("auto_resize", False))
        self._force_topmost = bool(window_cfg.get("force_topmost", False))
        marker_cfg = self._config.get("marker_detection", {})
        self._default_threshold = float(marker_cfg.get("threshold", 0.85))
        self._default_min_score_delta = float(marker_cfg.get("min_score_delta", 0.0))
        self._use_pixel_verification = bool(marker_cfg.get("use_pixel_verification", False))
        self._multi_scale_range = float(marker_cfg.get("multi_scale_range", 0.03))
        self._multi_scale_steps = int(marker_cfg.get("multi_scale_steps", 3))
        self._detector_method = str(marker_cfg.get("method", "template")).lower()
        if self._detector_method not in {"template", "feature"}:
            self._detector_method = "template"
        input_cfg = self._config.get("input", {})
        self._mouse_motion = str(input_cfg.get("mouse_motion", "instant")).lower()
        self._mouse_motion_duration = max(0.0, float(input_cfg.get("mouse_motion_duration_ms", 0.0)) / 1000.0)
        self._mouse_motion_fps = max(1.0, float(input_cfg.get("mouse_motion_fps", 60.0)))
        self._mouse_jitter_px = max(0.0, float(input_cfg.get("mouse_jitter_px", 0.0)))
        if self._mouse_motion not in {"instant", "linear", "ease", "human"}:
            self._mouse_motion = "instant"
        debug_cfg = self._config.get("debug", {})
        self._debug_screenshot = bool(debug_cfg.get("screenshot", False))
        debug_output = debug_cfg.get("output_dir", "debug_out")
        self._debug_output_dir = str(debug_output) if debug_output else "debug_out"
        self._debug_always_capture = bool(debug_cfg.get("always_capture", False))
        self._debug_keep_history = bool(debug_cfg.get("keep_history", False))
        capture_cfg = self._config.get("capture", {})
        capture_method = str(capture_cfg.get("method", "auto")).lower()
        if capture_method not in {"auto", "screen", "window", "window_raise"}:
            capture_method = "auto"
        self._capture_method = capture_method
        poll_ms = self._config.get("poll_interval_ms", 50)
        self._poll_interval = max(0.01, float(poll_ms) / 1000.0)
        ocr_cfg = self._config.get("ocr", {})
        self._ocr_config = {
            "engine": str(ocr_cfg.get("engine", "paddleocr")),
            "languages": list(ocr_cfg.get("languages", ["ja", "en"])),
            "gpu": bool(ocr_cfg.get("gpu", True)),
            "confidence_threshold": float(ocr_cfg.get("confidence_threshold", 0.5)),
        }

    def _debug_path(self, debug_dir: str, prefix: str, image_filename: str) -> str:
        if self._debug_keep_history:
            stamp = int(time.time() * 1000)
            name = f"{prefix}_{stamp}_{image_filename}"
        else:
            name = f"{prefix}_{image_filename}"
        return os.path.join(debug_dir, name)

    def reload(self) -> None:
        self._load_config()
        self._templates.clear()
        self._ocr_engines.clear()
        self._hwnd = None
        self._resized_once = False

    def _ensure_window(self) -> int:
        if self._hwnd is None:
            self._hwnd = find_window(self._title_substring, self._process_name)
            if self._hwnd is None:
                raise RuntimeError("Target window not found.")
            if self._force_topmost:
                set_window_topmost(self._hwnd, True)
        if self._auto_resize and not self._resized_once:
            self._apply_startup_resize(self._hwnd)
        return self._hwnd

    def _apply_startup_resize(self, hwnd: int) -> None:
        if self._resized_once:
            return
        if is_window_minimized(hwnd):
            logging.info("Window is minimized; skipping auto-resize.")
            return
        try:
            _origin_x, _origin_y, client_w, client_h = get_client_origin_and_size(hwnd)
        except Exception as exc:
            logging.warning("Failed to read window metrics for resize: %s", exc)
            return
        ref_w, ref_h = self._ref_size
        if ref_w <= 0 or ref_h <= 0:
            return
        if client_w == ref_w and client_h == ref_h:
            self._resized_once = True
            return
        if set_window_client_size(hwnd, ref_w, ref_h):
            logging.info("Auto-resized window client to %dx%d", ref_w, ref_h)
            self._resized_once = True
        else:
            logging.warning("Auto-resize failed to set client size %dx%d", ref_w, ref_h)

    def focus_window(self) -> bool:
        hwnd = self._ensure_window()
        return focus_window(hwnd)

    @staticmethod
    def _compute_render_rect(
        client_origin: Tuple[int, int],
        client_size: Tuple[int, int],
        ref_size: Tuple[int, int],
        render_cfg: Dict[str, Any],
    ) -> Tuple[int, int, int, int]:
        mode = str(render_cfg.get("mode", "stretch")).lower()
        client_w, client_h = client_size
        ref_w, ref_h = ref_size
        if mode == "manual":
            offset = render_cfg.get("offset", {})
            size = render_cfg.get("size", {})
            width = int(size.get("w", 0)) or client_w
            height = int(size.get("h", 0)) or client_h
            off_x = int(offset.get("x", 0))
            off_y = int(offset.get("y", 0))
            return client_origin[0] + off_x, client_origin[1] + off_y, width, height
        if mode == "fit":
            if ref_w <= 0 or ref_h <= 0 or client_w <= 0 or client_h <= 0:
                return client_origin[0], client_origin[1], client_w, client_h
            scale = min(client_w / ref_w, client_h / ref_h)
            render_w = int(round(ref_w * scale))
            render_h = int(round(ref_h * scale))
            off_x = int(round((client_w - render_w) / 2))
            off_y = int(round((client_h - render_h) / 2))
            return client_origin[0] + off_x, client_origin[1] + off_y, render_w, render_h
        return client_origin[0], client_origin[1], client_w, client_h

    def _get_window_metrics(
        self,
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int, int, int], float, float]:
        hwnd = self._ensure_window()
        if is_window_minimized(hwnd):
            raise RuntimeError("Window is minimized.")
        origin_x, origin_y, client_w, client_h = get_client_origin_and_size(hwnd)
        if client_w == 0 or client_h == 0:
            raise RuntimeError("Client size is zero.")
        render_x, render_y, render_w, render_h = self._compute_render_rect(
            (origin_x, origin_y), (client_w, client_h), self._ref_size, self._render_cfg
        )
        scale_x = render_w / self._ref_size[0] if self._ref_size[0] else 1.0
        scale_y = render_h / self._ref_size[1] if self._ref_size[1] else 1.0
        return (origin_x, origin_y), (client_w, client_h), (render_x, render_y, render_w, render_h), scale_x, scale_y

    def _get_template(self, image_path: str) -> TemplateDetector | FeatureDetector:
        detector = self._templates.get(image_path)
        if detector is None:
            if self._detector_method == "feature":
                detector = FeatureDetector(image_path)
            else:
                detector = TemplateDetector(image_path)
            self._templates[image_path] = detector
        return detector

    def _capture_roi(
        self,
        roi_ref: Optional[Tuple[int, int, int, int]],
        render_rect: Tuple[int, int, int, int],
    ) -> Tuple[Tuple[int, int, int, int], Optional[Any]]:
        ref_w, ref_h = self._ref_size
        if roi_ref is None:
            roi_ref = (0, 0, ref_w, ref_h)
        roi_screen = map_rect(roi_ref, self._ref_size, (render_rect[0], render_rect[1]), (render_rect[2], render_rect[3]))
        hwnd = self._ensure_window()
        image = self._capture.grab_auto(roi_screen, hwnd, self._capture_method)
        return roi_screen, image

    def _resolve_threshold(self, threshold: Optional[float], use_default_threshold: bool) -> float:
        if threshold is not None:
            return float(threshold)
        if use_default_threshold:
            return self._default_threshold
        return 0.0

    def default_min_score_delta(self) -> float:
        return float(self._default_min_score_delta)

    def locate_image(
        self,
        image_path: str,
        roi_ref: Optional[Tuple[int, int, int, int]] = None,
        threshold: Optional[float] = None,
        use_default_threshold: bool = True,
        min_score_delta: Optional[float] = None,
        debug_context: Optional[dict[str, object]] = None,
    ) -> Optional[MatchResult]:
        _, _, render_rect, scale_x, scale_y = self._get_window_metrics()
        roi_screen, image = self._capture_roi(roi_ref, render_rect)
        if image is None:
            return None
        detector = self._get_template(image_path)
        method = "feature" if isinstance(detector, FeatureDetector) else "template"
        score, loc, size, _second_best = detector.match_with_location(
            image, scale_x, scale_y,
            multi_scale_range=self._multi_scale_range,
            multi_scale_steps=self._multi_scale_steps,
        )
        if loc is None or size is None or not math.isfinite(score):
            return None
        min_score = self._resolve_threshold(threshold, use_default_threshold)
        if score < min_score:
            logging.debug(f"Rejecting match: score={score:.3f} < threshold={min_score:.3f}")
            return None
        delta = None
        if _second_best is not None and math.isfinite(_second_best):
            delta = float(score - _second_best)
            logging.debug(f"Match delta check: score={score:.3f}, second={_second_best:.3f}, delta={delta:.3f}")
        min_delta = self._default_min_score_delta if min_score_delta is None else float(min_score_delta)
        logging.debug(f"min_delta={min_delta:.3f}, delta={delta}, condition: min_delta > 0.0 = {min_delta > 0.0}, delta is not None = {delta is not None}, delta < min_delta = {delta < min_delta if delta is not None else 'N/A'}")
        if min_delta > 0.0 and delta is not None and delta < min_delta:
            logging.warning(
                f"REJECTING MATCH: delta={delta:.3f} < min_delta={min_delta:.3f} "
                f"(score={score:.3f}, second={_second_best:.3f})"
            )
            return None
        top_left = (roi_screen[0] + loc[0], roi_screen[1] + loc[1])
        image_filename = os.path.basename(image_path)

        if self._debug_always_capture:
            try:
                import cv2

                debug_img = image.copy()
                debug_dir = self._debug_output_dir
                if not os.path.isabs(debug_dir):
                    debug_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), debug_dir)
                os.makedirs(debug_dir, exist_ok=True)
                draw_match_rect(debug_img, loc, size, (255, 0, 0), thickness=2)
                debug_path = self._debug_path(debug_dir, "match", image_filename)
                cv2.imwrite(debug_path, debug_img)
                self._write_debug_meta(
                    debug_path,
                    {
                        "type": "match",
                        "template": image_filename,
                        "score": float(score),
                        "second": float(_second_best) if _second_best is not None else None,
                        "delta": float(delta) if delta is not None else None,
                        "method": method,
                        "roi_ref": _to_list(roi_ref),
                        "roi_screen": _to_list(roi_screen),
                        "match_loc": _to_list(loc),
                        "match_size": _to_list(size),
                        "context": debug_context,
                        "timestamp": time.time(),
                    },
                )
                logging.info("Debug screenshot saved: %s", debug_path)
            except Exception as exc:
                logging.debug("Failed to save debug screenshot: %s", exc)

        # Pixel verification check (if enabled)
        if self._use_pixel_verification:
            verification_passed = verify_match(
                image, loc, image_filename,
                roi_screen=roi_screen,
                ref_size=self._ref_size,
                render_origin=(render_rect[0], render_rect[1]),
                render_size=(render_rect[2], render_rect[3]),
            )

            if self._debug_screenshot:
                try:
                    import cv2

                    debug_img = image.copy()
                    debug_dir = self._debug_output_dir
                    if not os.path.isabs(debug_dir):
                        debug_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), debug_dir)
                    os.makedirs(debug_dir, exist_ok=True)

                    results = get_last_verification_results()
                    if results:
                        draw_verification_results(debug_img, results)
                    else:
                        failed_offset = get_last_failed_offset()
                        if failed_offset is not None:
                            abs_x = loc[0] + failed_offset[0]
                            abs_y = loc[1] + failed_offset[1]
                            cv2.circle(debug_img, (abs_x, abs_y), 5, (0, 0, 255), -1)

                    rect_color = (0, 255, 0) if verification_passed else (0, 0, 255)
                    draw_match_rect(debug_img, loc, size, rect_color, thickness=2)
                    debug_suffix = "verify_pass" if verification_passed else "verify_fail"
                    debug_path = self._debug_path(debug_dir, debug_suffix, image_filename)
                    cv2.imwrite(debug_path, debug_img)
                    self._write_debug_meta(
                        debug_path,
                        {
                            "type": debug_suffix,
                            "template": image_filename,
                            "score": float(score),
                            "second": float(_second_best) if _second_best is not None else None,
                            "delta": float(delta) if delta is not None else None,
                            "method": method,
                            "roi_ref": _to_list(roi_ref),
                            "roi_screen": _to_list(roi_screen),
                            "match_loc": _to_list(loc),
                            "match_size": _to_list(size),
                            "verification": {
                                "passed": bool(verification_passed),
                                "results": _serialize_results(results),
                            },
                            "context": debug_context,
                            "timestamp": time.time(),
                        },
                    )
                    logging.info("Debug screenshot saved: %s", debug_path)
                except Exception as exc:
                    logging.debug("Failed to save debug screenshot: %s", exc)
            
            if not verification_passed:
                logging.warning(f"REJECTING MATCH: Pixel verification failed for {image_filename}")
                return None
        
        center = (top_left[0] + size[0] // 2, top_left[1] + size[1] // 2)
        ref_x = int(round((center[0] - render_rect[0]) / scale_x)) if scale_x else 0
        ref_y = int(round((center[1] - render_rect[1]) / scale_y)) if scale_y else 0
        match_count = None
        match_total = None
        if isinstance(detector, FeatureDetector):
            match_count = detector.last_good
            match_total = detector.last_total
        return MatchResult(
            score=score,
            top_left=top_left,
            center=center,
            template_size=size,
            ref_center=(ref_x, ref_y),
            second_best=_second_best,
            score_delta=delta,
            method=method,
            match_count=match_count,
            match_total=match_total,
        )

    def verify_pixels(
        self,
        points: Any,
        roi_ref: Optional[Tuple[int, int, int, int]] = None,
        debug_context: Optional[dict[str, object]] = None,
    ) -> Tuple[bool, float]:
        """
        Verify pixel colors at reference-space coordinates.

        Args:
            points: Iterable of pixel points. Each point can be:
                - PixelPoint
                - dict with x/y/rgb/tolerance
                - tuple (x, y, (r, g, b), tolerance)
            roi_ref: Optional ROI in reference coords to limit capture.
        Returns:
            (matched, score) where score is matched/total.
        """
        _, _, render_rect, _scale_x, _scale_y = self._get_window_metrics()
        roi_screen, image = self._capture_roi(roi_ref, render_rect)
        if image is None:
            return False, 0.0
        parsed_points = parse_pixel_points(points)
        if not parsed_points:
            return False, 0.0
        render_origin = (render_rect[0], render_rect[1])
        render_size = (render_rect[2], render_rect[3])
        ok, score, results = pixel_points_check_with_results(
            parsed_points,
            image,
            roi_screen,
            self._ref_size,
            render_origin,
            render_size,
            self._capture,
        )
        if self._debug_always_capture or self._debug_screenshot:
            try:
                import cv2

                debug_img = image.copy()
                draw_verification_results(debug_img, results)
                debug_dir = self._debug_output_dir
                if not os.path.isabs(debug_dir):
                    debug_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), debug_dir)
                os.makedirs(debug_dir, exist_ok=True)
                suffix = "pass" if ok else "fail"
                debug_path = os.path.join(debug_dir, f"verify_pixels_{suffix}_{int(time.time() * 1000)}.png")
                cv2.imwrite(debug_path, debug_img)
                self._write_debug_meta(
                    debug_path,
                    {
                        "type": f"verify_pixels_{suffix}",
                        "score": float(score),
                        "roi_ref": _to_list(roi_ref),
                        "roi_screen": _to_list(roi_screen),
                        "verification": {
                            "passed": bool(ok),
                            "results": _serialize_results(results),
                        },
                        "context": debug_context,
                        "timestamp": time.time(),
                    },
                )
                logging.info("Debug verify_pixels screenshot saved: %s", debug_path)
            except Exception as exc:
                logging.debug("Failed to save verify_pixels debug screenshot: %s", exc)
        return ok, score

    def capture_roi(
        self,
        roi_ref: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[Any]:
        """Capture a region of the screen and return the raw BGR numpy array.

        Args:
            roi_ref: Region in reference coordinates (x, y, w, h).
                     If None, captures the full reference area.
        Returns:
            BGR numpy array, or None on capture failure.
        """
        _, _, render_rect, _scale_x, _scale_y = self._get_window_metrics()
        _roi_screen, image = self._capture_roi(roi_ref, render_rect)
        return image

    def image_present(
        self,
        image_path: str,
        roi_ref: Optional[Tuple[int, int, int, int]] = None,
        threshold: Optional[float] = None,
        use_default_threshold: bool = True,
        debug_context: Optional[dict[str, object]] = None,
    ) -> bool:
        return (
            self.locate_image(
                image_path,
                roi_ref=roi_ref,
                threshold=threshold,
                use_default_threshold=use_default_threshold,
                debug_context=debug_context,
            )
            is not None
        )

    def image_present_clip(
        self,
        image_path: str,
        roi_ref: Optional[Tuple[int, int, int, int]] = None,
        threshold: float = 0.75,
    ) -> Tuple[bool, float]:
        """
        Check if image is present using CLIP matching.

        More robust than template matching for UI elements affected by
        anti-aliasing, compression, or rendering variations.

        Args:
            image_path: Path to template image
            roi_ref: Optional ROI in reference coordinates (x, y, w, h)
            threshold: CLIP similarity threshold (0.0-1.0), default 0.75

        Returns:
            (present, score) tuple:
            - present: True if CLIP similarity >= threshold
            - score: CLIP similarity score (0.0-1.0)
        """
        from .clip_scanner import match_single_clip

        _, _, render_rect, _scale_x, _scale_y = self._get_window_metrics()
        _roi_screen, image = self._capture_roi(roi_ref, render_rect)
        if image is None:
            return False, 0.0

        score, matched = match_single_clip(
            capture_image=image,
            template_path=image_path,
            threshold=threshold,
            roi=None,  # Already cropped by _capture_roi
        )
        return matched, score

    def wait_for_image_clip(
        self,
        image_path: str,
        roi_ref: Optional[Tuple[int, int, int, int]] = None,
        threshold: float = 0.75,
        timeout_s: float = 10.0,
        poll_interval_s: float = 0.1,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> Tuple[bool, float]:
        """
        Wait for image to appear using CLIP matching.

        Args:
            image_path: Path to template image
            roi_ref: Optional ROI in reference coordinates
            threshold: CLIP similarity threshold (0.0-1.0)
            timeout_s: Maximum wait time in seconds
            poll_interval_s: Time between checks
            stop_check: Optional callback to check for stop request

        Returns:
            (found, score) tuple:
            - found: True if image appeared within timeout
            - score: Final CLIP score (highest seen, or last check)
        """
        deadline = time.monotonic() + max(0.0, timeout_s)
        best_score = 0.0

        while True:
            if stop_check is not None and stop_check():
                return False, best_score

            matched, score = self.image_present_clip(image_path, roi_ref, threshold)
            best_score = max(best_score, score)

            if matched:
                return True, score

            if time.monotonic() >= deadline:
                return False, best_score

            time.sleep(poll_interval_s)

    def wait_for_image(
        self,
        image_path: str,
        roi_ref: Optional[Tuple[int, int, int, int]] = None,
        threshold: Optional[float] = None,
        use_default_threshold: bool = True,
        timeout_s: float = 5.0,
        stop_check: Optional[Callable[[], bool]] = None,
        debug_context: Optional[dict[str, object]] = None,
    ) -> Optional[MatchResult]:
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            if stop_check is not None and stop_check():
                return None
            try:
                result = self.locate_image(
                    image_path,
                    roi_ref=roi_ref,
                    threshold=threshold,
                    use_default_threshold=use_default_threshold,
                    debug_context=debug_context,
                )
            except RuntimeError as exc:
                logging.warning("Window not ready: %s", exc)
                result = None
            if result is not None:
                return result
            if timeout_s <= 0:
                return None
            if time.monotonic() >= deadline:
                return None
            if stop_check is not None and stop_check():
                return None
            time.sleep(self._poll_interval)

    def action(
        self,
        image_path: str,
        coords_offset: Tuple[int, int] = (0, 0),
        click_duration: float = 0.0,
        timeout_s: float = 5.0,
        roi_ref: Optional[Tuple[int, int, int, int]] = None,
        threshold: Optional[float] = None,
        use_default_threshold: bool = True,
        offset_in_ref: bool = True,
        stop_check: Optional[Callable[[], bool]] = None,
        debug_context: Optional[dict[str, object]] = None,
    ) -> bool:
        result = self.action_with_match(
            image_path,
            coords_offset=coords_offset,
            click_duration=click_duration,
            timeout_s=timeout_s,
            roi_ref=roi_ref,
            threshold=threshold,
            use_default_threshold=use_default_threshold,
            offset_in_ref=offset_in_ref,
            stop_check=stop_check,
            debug_context=debug_context,
        )
        return result is not None

    def action_with_match(
        self,
        image_path: str,
        coords_offset: Tuple[int, int] = (0, 0),
        click_duration: float = 0.0,
        timeout_s: float = 5.0,
        roi_ref: Optional[Tuple[int, int, int, int]] = None,
        threshold: Optional[float] = None,
        use_default_threshold: bool = True,
        offset_in_ref: bool = True,
        stop_check: Optional[Callable[[], bool]] = None,
        debug_context: Optional[dict[str, object]] = None,
    ) -> Optional[MatchResult]:
        result = self.wait_for_image(
            image_path,
            roi_ref=roi_ref,
            threshold=threshold,
            use_default_threshold=use_default_threshold,
            timeout_s=timeout_s,
            stop_check=stop_check,
            debug_context=debug_context,
        )
        if result is None:
            return None
        click_x, click_y = result.center
        if coords_offset != (0, 0):
            _, _, render_rect, scale_x, scale_y = self._get_window_metrics()
            dx, dy = coords_offset
            if offset_in_ref:
                click_x += int(round(dx * scale_x))
                click_y += int(round(dy * scale_y))
            else:
                click_x += int(dx)
                click_y += int(dy)
        if self._debug_always_capture or self._debug_screenshot:
            try:
                import cv2

                _, _, render_rect, _scale_x, _scale_y = self._get_window_metrics()
                roi_screen, image = self._capture_roi(roi_ref, render_rect)
                if image is not None:
                    debug_img = image.copy()
                    rel_click_x = int(click_x - roi_screen[0])
                    rel_click_y = int(click_y - roi_screen[1])
                    if 0 <= rel_click_x < debug_img.shape[1] and 0 <= rel_click_y < debug_img.shape[0]:
                        cv2.circle(debug_img, (rel_click_x, rel_click_y), 6, (0, 255, 255), -1)
                    rel_left = int(result.top_left[0] - roi_screen[0])
                    rel_top = int(result.top_left[1] - roi_screen[1])
                    t_w, t_h = result.template_size
                    cv2.rectangle(
                        debug_img,
                        (rel_left, rel_top),
                        (rel_left + t_w, rel_top + t_h),
                        (255, 0, 0),
                        2,
                    )
                    results = get_last_verification_results()
                    if results:
                        draw_verification_results(debug_img, results)
                    debug_dir = self._debug_output_dir
                    if not os.path.isabs(debug_dir):
                        debug_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), debug_dir)
                    os.makedirs(debug_dir, exist_ok=True)
                    image_filename = os.path.basename(image_path)
                    debug_path = self._debug_path(debug_dir, "click", image_filename)
                    cv2.imwrite(debug_path, debug_img)
                    self._write_debug_meta(
                        debug_path,
                        {
                            "type": "click",
                            "template": image_filename,
                            "click_pos": _to_list((click_x, click_y)),
                            "match_loc": _to_list((rel_left, rel_top)),
                            "match_size": _to_list((t_w, t_h)),
                            "roi_ref": _to_list(roi_ref),
                            "roi_screen": _to_list(roi_screen),
                            "verification": {
                                "results": _serialize_results(results) if results else [],
                            },
                            "context": debug_context,
                            "timestamp": time.time(),
                        },
                    )
                    logging.info("Debug click screenshot saved: %s", debug_path)
            except Exception as exc:
                logging.debug("Failed to save debug click screenshot: %s", exc)
        self._move_mouse(click_x, click_y)
        self._click_at(click_x, click_y, click_duration)
        return result

    def move_to_screen(self, x: int, y: int, duration: float = 0.0) -> None:
        if duration > 0:
            pydirectinput.moveTo(int(x), int(y), duration=max(0.0, float(duration)))
            return
        self._move_mouse(int(x), int(y))

    def move_to_ref(self, x: int, y: int, duration: float = 0.0) -> None:
        _, _, render_rect, scale_x, scale_y = self._get_window_metrics()
        screen_x = render_rect[0] + int(round(x * scale_x))
        screen_y = render_rect[1] + int(round(y * scale_y))
        if duration > 0:
            pydirectinput.moveTo(int(screen_x), int(screen_y), duration=max(0.0, float(duration)))
            return
        self._move_mouse(int(screen_x), int(screen_y))

    def _move_mouse(self, target_x: int, target_y: int) -> None:
        mode = self._mouse_motion
        if mode == "instant" or self._mouse_motion_duration <= 0:
            pydirectinput.moveTo(int(target_x), int(target_y), duration=0.0)
            return
        start_x, start_y = _get_cursor_pos()
        duration = self._mouse_motion_duration
        steps = max(1, int(round(duration * self._mouse_motion_fps)))
        for step in range(1, steps + 1):
            t = step / steps
            if mode in {"ease", "human"}:
                t = t * t * (3.0 - 2.0 * t)
            x = start_x + (target_x - start_x) * t
            y = start_y + (target_y - start_y) * t
            if mode == "human" and self._mouse_jitter_px > 0.0:
                jitter_scale = 1.0 - abs(0.5 - t) * 2.0
                jitter = self._mouse_jitter_px * jitter_scale
                x += random.uniform(-jitter, jitter)
                y += random.uniform(-jitter, jitter)
            pydirectinput.moveTo(int(round(x)), int(round(y)), duration=0.0)
            if step < steps:
                time.sleep(duration / steps)

    def drag_screen(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        hold_s: float = 0.0,
        drag_duration_s: float = 0.5,
        button: str = "left",
        settle_s: float = 1.0,
        begin_sleep_s: float = 0.0,
        end_sleep_s: float = 0.0,
    ) -> None:
        """
        Perform a drag operation from start to end position.
        
        Args:
            start: Starting (x, y) screen coordinates
            end: Ending (x, y) screen coordinates
            hold_s: Time to hold at START position after pressing mouse button (before drag motion)
            drag_duration_s: Duration of the drag motion itself
            button: Mouse button to use ("left", "right", "middle")
            settle_s: Time to hold at END position BEFORE releasing mouse button (kills velocity/momentum)
            begin_sleep_s: Time to sleep BEFORE pressing mouse button at start
            end_sleep_s: Time to sleep AFTER releasing mouse button at end
        """
        start_x, start_y = int(start[0]), int(start[1])
        end_x, end_y = int(end[0]), int(end[1])
        
        # Move to start position
        pydirectinput.moveTo(start_x, start_y, duration=0.0)
        
        # Optional: sleep before pressing button
        if begin_sleep_s > 0:
            time.sleep(max(0.0, float(begin_sleep_s)))
        
        # Press mouse button to start drag
        try:
            pydirectinput.mouseDown(button=button)
        except TypeError:
            pydirectinput.mouseDown(start_x, start_y)
        
        # Hold at start position after pressing button
        if hold_s > 0:
            time.sleep(max(0.0, float(hold_s)))
        
        # Perform drag motion with native SetCursorPos for maximum smoothness
        if drag_duration_s > 0:
            try:
                import ctypes
                user32 = ctypes.windll.user32
                use_native = True
            except:
                use_native = False
            
            drag_start_time = time.perf_counter()
            prev_x, prev_y = start_x, start_y
            last_move_time = drag_start_time
            
            while True:
                now = time.perf_counter()
                elapsed = now - drag_start_time
                t = min(1.0, elapsed / drag_duration_s)
                
                # Use ease for smoother drag
                ease_t = t * t * (3.0 - 2.0 * t)
                x = start_x + (end_x - start_x) * ease_t
                y = start_y + (end_y - start_y) * ease_t
                
                curr_x, curr_y = int(round(x)), int(round(y))
                
                # Only move when pixel position changes
                if curr_x != prev_x or curr_y != prev_y:
                    if use_native:
                        user32.SetCursorPos(curr_x, curr_y)
                    else:
                        pydirectinput.moveTo(curr_x, curr_y, duration=0.0)
                    prev_x, prev_y = curr_x, curr_y
                    last_move_time = time.perf_counter()
                
                if t >= 1.0:
                    # Ensure final position
                    if prev_x != end_x or prev_y != end_y:
                        if use_native:
                            user32.SetCursorPos(end_x, end_y)
                        else:
                            pydirectinput.moveTo(end_x, end_y, duration=0.0)
                        last_move_time = time.perf_counter()
                    break
                
                # Minimal sleep for smooth motion
                time.sleep(0.001)
        else:
            pydirectinput.moveTo(end_x, end_y, duration=0.0)
            last_move_time = time.perf_counter()
        
        # Hold at end position BEFORE releasing to kill velocity/momentum
        if settle_s > 0:
            time_since_last_move = time.perf_counter() - last_move_time
            remaining_settle = settle_s - time_since_last_move
            if remaining_settle > 0:
                time.sleep(remaining_settle)
        
        # Release mouse button to end drag
        try:
            pydirectinput.mouseUp(button=button)
        except TypeError:
            pydirectinput.mouseUp(end_x, end_y)
        
        # Optional: sleep after releasing button
        if end_sleep_s > 0:
            time.sleep(max(0.0, float(end_sleep_s)))

    def drag_ref(
        self,
        start_ref: Tuple[int, int],
        end_ref: Tuple[int, int],
        hold_s: float = 0.0,
        drag_duration_s: float = 0.5,
        button: str = "left",
        settle_s: float = 1.0,
        begin_sleep_s: float = 0.0,
        end_sleep_s: float = 0.0,
    ) -> None:
        """
        Perform a drag operation using reference coordinates.
        
        Args:
            start_ref: Starting (x, y) reference coordinates
            end_ref: Ending (x, y) reference coordinates
            hold_s: Time to hold at START position after pressing mouse button (before drag motion)
            drag_duration_s: Duration of the drag motion itself
            button: Mouse button to use ("left", "right", "middle")
            settle_s: Time to hold at END position BEFORE releasing mouse button (kills velocity/momentum)
            begin_sleep_s: Time to sleep BEFORE pressing mouse button at start
            end_sleep_s: Time to sleep AFTER releasing mouse button at end
        """
        _, _, render_rect, scale_x, scale_y = self._get_window_metrics()
        render_x, render_y = render_rect[0], render_rect[1]
        start_x = int(round(render_x + start_ref[0] * scale_x))
        start_y = int(round(render_y + start_ref[1] * scale_y))
        end_x = int(round(render_x + end_ref[0] * scale_x))
        end_y = int(round(render_y + end_ref[1] * scale_y))
        self.drag_screen((start_x, start_y), (end_x, end_y), hold_s, drag_duration_s, button, settle_s, begin_sleep_s, end_sleep_s)

    def _click_at(self, x: int, y: int, click_duration: float) -> None:
        if click_duration > 0:
            try:
                pydirectinput.mouseDown()
            except TypeError:
                pydirectinput.mouseDown(int(x), int(y))
            time.sleep(max(0.0, float(click_duration)))
            try:
                pydirectinput.mouseUp()
            except TypeError:
                pydirectinput.mouseUp(int(x), int(y))
            return
        try:
            pydirectinput.click()
        except TypeError:
            pydirectinput.click(int(x), int(y))

    # -- OCR -----------------------------------------------------------------

    def _get_ocr_engine(self, name: str) -> Optional[OCREngine]:
        """Get or lazily initialize a cached OCR engine."""
        engine = self._ocr_engines.get(name)
        if engine is not None:
            return engine
        try:
            engine = create_ocr_engine(name)
        except KeyError:
            logging.warning("Unknown OCR engine: %s", name)
            return None
        try:
            engine.initialize(
                languages=self._ocr_config.get("languages", ["ja", "en"]),
                use_gpu=self._ocr_config.get("gpu", True),
            )
        except ImportError:
            logging.warning("OCR engine %s not installed, skipping", name)
            return None
        except Exception as exc:
            logging.warning("OCR engine %s failed to initialize: %s", name, exc)
            return None
        self._ocr_engines[name] = engine
        return engine

    def _get_active_ocr_engines(self) -> list[OCREngine]:
        """Resolve which OCR engines to use from config."""
        engine_setting = self._ocr_config.get("engine", "paddleocr")
        if engine_setting == "all":
            engines: list[OCREngine] = []
            for name in available_engines():
                eng = self._get_ocr_engine(name)
                if eng is not None:
                    engines.append(eng)
            return engines
        eng = self._get_ocr_engine(engine_setting)
        return [eng] if eng is not None else []

    def read_text_roi(
        self,
        roi_ref: Optional[Tuple[int, int, int, int]] = None,
        confidence_threshold: Optional[float] = None,
        debug_context: Optional[dict[str, object]] = None,
    ) -> list[TextResult]:
        """Capture an ROI and run OCR, returning filtered TextResults."""
        _, _, render_rect, _scale_x, _scale_y = self._get_window_metrics()
        roi_screen, image = self._capture_roi(roi_ref, render_rect)
        if image is None:
            return []
        threshold = confidence_threshold if confidence_threshold is not None else self._ocr_config.get("confidence_threshold", 0.5)
        engines = self._get_active_ocr_engines()
        if not engines:
            logging.warning("No OCR engines available")
            return []
        all_results: list[TextResult] = []
        for engine in engines:
            try:
                results = engine.read(image)
                all_results.extend(results)
            except Exception as exc:
                logging.warning("OCR engine %s failed: %s", engine.name, exc)
        # Filter by confidence
        filtered = [r for r in all_results if r.confidence >= threshold]
        # Sort by position
        filtered = sort_by_position(filtered)
        if self._debug_always_capture or self._debug_screenshot:
            self._save_ocr_debug(image, filtered, roi_ref, roi_screen, debug_context)
        return filtered

    def read_text_rois(
        self,
        rois: Dict[str, Tuple[int, int, int, int]],
        confidence_threshold: Optional[float] = None,
        debug_context: Optional[dict[str, object]] = None,
        fallback_per_roi: bool = False,
    ) -> Dict[str, list[TextResult]]:
        """Read text from multiple ROIs using a single OCR pass when possible."""
        if not rois:
            return {}
        cleaned: Dict[str, Tuple[int, int, int, int]] = {}
        for key, roi in rois.items():
            if roi is None:
                continue
            try:
                x, y, w, h = (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
            except Exception:
                logging.warning("Invalid ROI for key %s: %s", key, roi)
                continue
            if w <= 0 or h <= 0:
                logging.warning("Skipping zero-size ROI for key %s: %s", key, roi)
                continue
            cleaned[str(key)] = (x, y, w, h)
        if not cleaned:
            return {}
        if len(cleaned) == 1:
            only_key = next(iter(cleaned))
            results = self.read_text_roi(
                roi_ref=cleaned[only_key],
                confidence_threshold=confidence_threshold,
                debug_context=debug_context,
            )
            return {only_key: results}

        min_x = min(r[0] for r in cleaned.values())
        min_y = min(r[1] for r in cleaned.values())
        max_x = max(r[0] + r[2] for r in cleaned.values())
        max_y = max(r[1] + r[3] for r in cleaned.values())
        union_ref = (min_x, min_y, max_x - min_x, max_y - min_y)

        _, _, render_rect, _scale_x, _scale_y = self._get_window_metrics()
        union_screen, image = self._capture_roi(union_ref, render_rect)
        if image is None:
            return {key: [] for key in cleaned}

        threshold = confidence_threshold if confidence_threshold is not None else self._ocr_config.get("confidence_threshold", 0.5)
        engines = self._get_active_ocr_engines()
        if not engines:
            logging.warning("No OCR engines available")
            return {key: [] for key in cleaned}

        all_results: list[TextResult] = []
        for engine in engines:
            try:
                results = engine.read(image)
                all_results.extend(results)
            except Exception as exc:
                logging.warning("OCR engine %s failed: %s", engine.name, exc)

        filtered = [r for r in all_results if r.confidence >= threshold]
        if not filtered:
            return {key: [] for key in cleaned}

        with_bbox = [r for r in filtered if r.bbox is not None]
        if not with_bbox:
            logging.warning("OCR multi-ROI requires bbox results (PaddleOCR/EasyOCR). No bbox available.")
            return {key: [] for key in cleaned}

        if self._debug_always_capture or self._debug_screenshot:
            self._save_ocr_debug(image, with_bbox, union_ref, union_screen, debug_context)

        render_origin = (render_rect[0], render_rect[1])
        render_size = (render_rect[2], render_rect[3])
        roi_rel_map: Dict[str, Tuple[int, int, int, int]] = {}
        for key, roi in cleaned.items():
            roi_screen = map_rect(roi, self._ref_size, render_origin, render_size)
            rel_x = roi_screen[0] - union_screen[0]
            rel_y = roi_screen[1] - union_screen[1]
            rel_w = roi_screen[2]
            rel_h = roi_screen[3]
            if rel_x < 0:
                rel_w += rel_x
                rel_x = 0
            if rel_y < 0:
                rel_h += rel_y
                rel_y = 0
            if rel_x + rel_w > union_screen[2]:
                rel_w = max(0, union_screen[2] - rel_x)
            if rel_y + rel_h > union_screen[3]:
                rel_h = max(0, union_screen[3] - rel_y)
            roi_rel_map[key] = (int(rel_x), int(rel_y), int(rel_w), int(rel_h))

        results_by_roi: Dict[str, list[TextResult]] = {key: [] for key in cleaned}
        for key, roi_rel in roi_rel_map.items():
            rx, ry, rw, rh = roi_rel
            if rw <= 0 or rh <= 0:
                continue
            for r in with_bbox:
                if r.bbox is None:
                    continue
                bx, by, bw, bh = r.bbox
                cx = bx + bw * 0.5
                cy = by + bh * 0.5
                if rx <= cx < rx + rw and ry <= cy < ry + rh:
                    adj_bbox = (int(bx - rx), int(by - ry), int(bw), int(bh))
                    results_by_roi[key].append(TextResult(
                        text=r.text,
                        confidence=r.confidence,
                        bbox=adj_bbox,
                        engine_name=r.engine_name,
                    ))
            results_by_roi[key] = sort_by_position(results_by_roi[key])

        return results_by_roi

    def read_number_roi(
        self,
        roi_ref: Optional[Tuple[int, int, int, int]] = None,
        confidence_threshold: Optional[float] = None,
        debug_context: Optional[dict[str, object]] = None,
    ) -> Optional[float]:
        """Capture an ROI, run OCR, and parse a number from the results.

        When multiple engines return results, each engine's text is parsed
        independently and the highest-confidence engine's number is used.
        This avoids cross-engine concatenation errors.
        """
        results = self.read_text_roi(
            roi_ref=roi_ref,
            confidence_threshold=confidence_threshold,
            debug_context=debug_context,
        )
        if not results:
            return None
        # Group results by engine, parse each independently
        by_engine: dict[str, list[TextResult]] = {}
        for r in results:
            by_engine.setdefault(r.engine_name, []).append(r)
        best_number: Optional[float] = None
        best_confidence: float = -1.0
        for engine_name, engine_results in by_engine.items():
            combined = " ".join(r.text for r in engine_results)
            number = parse_number(combined)
            if number is not None:
                avg_conf = sum(r.confidence for r in engine_results) / len(engine_results)
                if avg_conf > best_confidence:
                    best_confidence = avg_conf
                    best_number = number
        return best_number

    def read_number_rois(
        self,
        rois: Dict[str, Tuple[int, int, int, int]],
        confidence_threshold: Optional[float] = None,
        debug_context: Optional[dict[str, object]] = None,
        fallback_per_roi: bool = False,
    ) -> Dict[str, Optional[float]]:
        """Read numbers from multiple ROIs using a single OCR pass when possible."""
        results_by_roi = self.read_text_rois(
            rois=rois,
            confidence_threshold=confidence_threshold,
            debug_context=debug_context,
            fallback_per_roi=fallback_per_roi,
        )
        values: Dict[str, Optional[float]] = {}
        for key, results in results_by_roi.items():
            if not results:
                values[key] = None
                continue
            by_engine: dict[str, list[TextResult]] = {}
            for r in results:
                by_engine.setdefault(r.engine_name, []).append(r)
            best_number: Optional[float] = None
            best_confidence: float = -1.0
            for engine_name, engine_results in by_engine.items():
                combined = " ".join(r.text for r in engine_results)
                number = parse_number(combined)
                if number is not None:
                    avg_conf = sum(r.confidence for r in engine_results) / len(engine_results)
                    if avg_conf > best_confidence:
                        best_confidence = avg_conf
                        best_number = number
            values[key] = best_number
        return values

    def _save_ocr_debug(
        self,
        image: Any,
        results: list[TextResult],
        roi_ref: Optional[Tuple[int, int, int, int]],
        roi_screen: Tuple[int, int, int, int],
        debug_context: Optional[dict[str, object]],
    ) -> None:
        """Save annotated OCR debug screenshot and metadata JSON."""
        try:
            import cv2 as _cv2

            debug_img = image.copy()
            colors = [
                (0, 255, 0),    # green
                (255, 165, 0),  # orange (BGR)
                (255, 0, 0),    # blue
                (0, 255, 255),  # yellow
            ]
            engine_color_map: dict[str, Tuple[int, int, int]] = {}
            color_idx = 0
            for r in results:
                if r.engine_name not in engine_color_map:
                    engine_color_map[r.engine_name] = colors[color_idx % len(colors)]
                    color_idx += 1
                color = engine_color_map[r.engine_name]
                if r.bbox is not None:
                    bx, by, bw, bh = r.bbox
                    _cv2.rectangle(debug_img, (bx, by), (bx + bw, by + bh), color, 2)
                    label = f"{r.engine_name}: {r.text} ({r.confidence:.2f})"
                    _cv2.putText(
                        debug_img, label,
                        (bx, max(12, by - 4)),
                        _cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
                    )

            debug_dir = self._debug_output_dir
            if not os.path.isabs(debug_dir):
                debug_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), debug_dir)
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = self._debug_path(debug_dir, "ocr", "result.png")
            _cv2.imwrite(debug_path, debug_img)
            self._write_debug_meta(
                debug_path,
                {
                    "type": "ocr",
                    "roi_ref": _to_list(roi_ref),
                    "roi_screen": _to_list(roi_screen),
                    "results": [
                        {
                            "text": r.text,
                            "confidence": r.confidence,
                            "bbox": list(r.bbox) if r.bbox else None,
                            "engine": r.engine_name,
                        }
                        for r in results
                    ],
                    "context": debug_context,
                    "timestamp": time.time(),
                },
            )
            logging.info("Debug OCR screenshot saved: %s", debug_path)
        except Exception as exc:
            logging.debug("Failed to save OCR debug screenshot: %s", exc)
