"""
Multi-verification system for reliable image detection.
Combines template matching with pixel verification to eliminate false positives.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple, List

import cv2
import numpy as np


def _parse_point_options(point: tuple) -> Tuple[int, int, Tuple[int, int, int], int, Dict[str, Any]]:
    """Parse a verification point tuple, supporting both 4-tuple and 5-tuple formats.

    Returns:
        (x, y, rgb, tolerance, options) where options may contain
        ``global`` (bool) and ``sample`` (int).
    """
    if len(point) == 5:
        x, y, rgb, tolerance, options = point
        if not isinstance(options, dict):
            options = {}
    elif len(point) == 4:
        x, y, rgb, tolerance = point
        options = {}
    else:
        raise ValueError(f"Verification point must have 4 or 5 elements, got {len(point)}")
    return int(x), int(y), tuple(rgb), int(tolerance), options


class VerifiedMatcher:
    """
    Enhanced matcher that requires both template match AND pixel verification.
    This eliminates false positives by checking actual pixel colors at key positions.
    """

    def __init__(self):
        self.verification_points: dict[str, list[tuple]] = {}
        self.last_verified_points: list[Tuple[int, int]] = []
        self.last_verification_results: list[Dict[str, Any]] = []
        self.last_failed_offset: Optional[Tuple[int, int]] = None
        self.last_failed_expected: Optional[Tuple[int, int, int]] = None
        self.last_failed_actual: Optional[Tuple[int, int, int]] = None

    def add_verification_points(
        self,
        image_name: str,
        points: list[tuple],
    ) -> None:
        """
        Add pixel verification points for an image.

        Args:
            image_name: Name of the template image
            points: List of verification tuples. Each is either:
                (rel_x, rel_y, (r,g,b), tolerance)                     - relative to match
                (x, y, (r,g,b), tolerance, {"global": True})           - ref-space coords
                (x, y, (r,g,b), tolerance, {"sample": radius})         - sample NxN box
                (x, y, (r,g,b), tolerance, {"global": True, "sample": radius})
        """
        self.verification_points[image_name] = points

    def verify_match(
        self,
        screenshot: np.ndarray,
        match_location: Tuple[int, int],
        image_name: str,
        roi_screen: Optional[Tuple[int, int, int, int]] = None,
        ref_size: Optional[Tuple[int, int]] = None,
        render_origin: Optional[Tuple[int, int]] = None,
        render_size: Optional[Tuple[int, int]] = None,
    ) -> bool:
        """
        Verify a template match by checking pixel colors at specific locations.

        Args:
            screenshot: Captured ROI image (BGR).
            match_location: (x, y) of the template match top-left within *screenshot*.
            image_name: Key used in ``verification_points``.
            roi_screen: (left, top, w, h) of the captured ROI in screen coords.
                        Required for ``global`` mode points.
            ref_size: (w, h) of the reference coordinate space.
            render_origin: (x, y) screen origin of the render area.
            render_size: (w, h) screen size of the render area.

        Returns:
            True if all verification points match, False otherwise.
        """
        self.last_verified_points = []
        self.last_verification_results = []
        self.last_failed_offset = None
        self.last_failed_expected = None
        self.last_failed_actual = None
        if image_name not in self.verification_points:
            logging.warning(f"No verification points defined for {image_name}, accepting match")
            return True

        points = self.verification_points[image_name]
        top_x, top_y = match_location
        verified_points = []

        for point in points:
            px, py, expected_rgb, tolerance, options = _parse_point_options(point)
            is_global = bool(options.get("global", False))
            sample_radius = int(options.get("sample", 0))
            result: Dict[str, Any] = {
                "rel_pos": (px, py),
                "abs_pos": None,
                "passed": False,
                "sample": sample_radius,
                "is_global": is_global,
                "match_pos": None,
                "expected": expected_rgb,
                "tolerance": tolerance,
                "actual": None,
            }

            if is_global:
                # Convert ref coords → screen coords → image-relative coords
                if roi_screen is None or ref_size is None or render_origin is None or render_size is None:
                    logging.warning(
                        "Global verification point (%d, %d) requires mapping context "
                        "but none was provided; failing point", px, py
                    )
                    self.last_failed_offset = (px, py)
                    self.last_verification_results.append(result)
                    return False
                scale_x = render_size[0] / ref_size[0] if ref_size[0] else 1.0
                scale_y = render_size[1] / ref_size[1] if ref_size[1] else 1.0
                screen_x = render_origin[0] + px * scale_x
                screen_y = render_origin[1] + py * scale_y
                abs_x = round(screen_x - roi_screen[0])
                abs_y = round(screen_y - roi_screen[1])
            else:
                abs_x = top_x + px
                abs_y = top_y + py
            result["abs_pos"] = (abs_x, abs_y)

            if sample_radius > 0:
                passed, matched_coords = self._check_sample_region(
                    screenshot, abs_x, abs_y, sample_radius, expected_rgb, tolerance
                )
                result["passed"] = passed
                result["match_pos"] = matched_coords
                self.last_verification_results.append(result)
                if not passed:
                    logging.info(
                        f"Pixel verification FAILED at {'global' if is_global else 'rel'}"
                        f"({px}, {py}) sample={sample_radius}: "
                        f"expected RGB{expected_rgb} ± {tolerance}, no match in region"
                    )
                    self.last_failed_offset = (px, py)
                    self.last_failed_expected = expected_rgb
                    # Report center pixel as actual
                    if 0 <= abs_y < screenshot.shape[0] and 0 <= abs_x < screenshot.shape[1]:
                        b, g, r = screenshot[abs_y, abs_x]
                        self.last_failed_actual = (int(r), int(g), int(b))
                    return False
                verified_points.append(matched_coords)
            else:
                # Check bounds
                if abs_y < 0 or abs_x < 0 or abs_y >= screenshot.shape[0] or abs_x >= screenshot.shape[1]:
                    logging.debug(f"Verification point ({abs_x}, {abs_y}) out of bounds")
                    self.last_failed_offset = (px, py)
                    self.last_verification_results.append(result)
                    return False

                # Get actual pixel (BGR in OpenCV)
                b, g, r = screenshot[abs_y, abs_x]
                actual_rgb = (int(r), int(g), int(b))
                result["actual"] = actual_rgb

                # Check if within tolerance
                if not self._colors_match(actual_rgb, expected_rgb, tolerance):
                    logging.info(
                        f"Pixel verification FAILED at {'global' if is_global else 'rel'}"
                        f"({px}, {py}): "
                        f"expected RGB{expected_rgb} ± {tolerance}, got RGB{actual_rgb}"
                    )
                    self.last_failed_offset = (px, py)
                    self.last_failed_expected = expected_rgb
                    self.last_failed_actual = actual_rgb
                    self.last_verification_results.append(result)
                    return False

                result["passed"] = True
                result["match_pos"] = (abs_x, abs_y)
                self.last_verification_results.append(result)
                verified_points.append((px, py))

        logging.debug(f"All {len(points)} verification points matched for {image_name}")
        self.last_verified_points = verified_points
        return True

    def _check_sample_region(
        self,
        screenshot: np.ndarray,
        center_x: int,
        center_y: int,
        radius: int,
        expected_rgb: Tuple[int, int, int],
        tolerance: int,
    ) -> Tuple[bool, Tuple[int, int]]:
        """Check a (2*radius+1)^2 region around (center_x, center_y).

        Returns:
            (passed, (match_x, match_y)) — the coords of the first matching pixel.
        """
        img_h, img_w = screenshot.shape[:2]
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                sx = center_x + dx
                sy = center_y + dy
                if sx < 0 or sy < 0 or sx >= img_w or sy >= img_h:
                    continue
                b, g, r = screenshot[sy, sx]
                actual_rgb = (int(r), int(g), int(b))
                if self._colors_match(actual_rgb, expected_rgb, tolerance):
                    return True, (sx, sy)
        return False, (center_x, center_y)

    @staticmethod
    def _colors_match(
        actual: Tuple[int, int, int],
        expected: Tuple[int, int, int],
        tolerance: int
    ) -> bool:
        """Check if two RGB colors match within tolerance."""
        return all(abs(a - e) <= tolerance for a, e in zip(actual, expected))


# Global instance
_verifier = VerifiedMatcher()


def add_pixel_verification(
    image_name: str,
    points: list[tuple],
) -> None:
    """
    Add pixel verification points for an image.

    Example:
        # Verify that at offset (50, 20) from match, pixel is white (255,255,255) ± 10
        add_pixel_verification("statecheck4.png", [
            (50, 20, (255, 255, 255), 10),
            (100, 30, (0, 128, 255), 15),
        ])

        # Global coords + sample radius
        add_pixel_verification("statecheck4.png", [
            (960, 540, (200, 50, 50), 20, {"global": True, "sample": 3}),
        ])
    """
    key = os.path.basename(image_name)
    _verifier.add_verification_points(key, points)


def verify_match(
    screenshot: np.ndarray,
    match_location: Tuple[int, int],
    image_name: str,
    roi_screen: Optional[Tuple[int, int, int, int]] = None,
    ref_size: Optional[Tuple[int, int]] = None,
    render_origin: Optional[Tuple[int, int]] = None,
    render_size: Optional[Tuple[int, int]] = None,
) -> bool:
    """Verify a template match using pixel checks."""
    key = os.path.basename(image_name)
    return _verifier.verify_match(
        screenshot, match_location, key,
        roi_screen=roi_screen,
        ref_size=ref_size,
        render_origin=render_origin,
        render_size=render_size,
    )


def get_last_verified_points() -> list[Tuple[int, int]]:
    return list(_verifier.last_verified_points)


def get_last_verification_results() -> list[Dict[str, Any]]:
    return list(_verifier.last_verification_results)


def get_last_failed_offset() -> Optional[Tuple[int, int]]:
    return _verifier.last_failed_offset
