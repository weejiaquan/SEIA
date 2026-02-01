from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple

import cv2
import numpy as np


def draw_match_rect(
    image: np.ndarray,
    loc: Tuple[int, int],
    size: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    x, y = int(loc[0]), int(loc[1])
    w, h = int(size[0]), int(size[1])
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)


def draw_verification_results(
    image: np.ndarray,
    results: Iterable[dict[str, Any]],
    pass_color: Tuple[int, int, int] = (0, 255, 0),
    fail_color: Tuple[int, int, int] = (0, 0, 255),
    point_radius: int = 5,
    sample_thickness: int = 1,
) -> None:
    for result in results:
        abs_pos = result.get("abs_pos")
        if not abs_pos:
            continue
        abs_x = int(abs_pos[0])
        abs_y = int(abs_pos[1])
        passed = bool(result.get("passed"))
        color = pass_color if passed else fail_color
        sample = int(result.get("sample") or 0)
        if sample > 0:
            cv2.rectangle(
                image,
                (abs_x - sample, abs_y - sample),
                (abs_x + sample, abs_y + sample),
                color,
                sample_thickness,
            )
            match_pos = result.get("match_pos")
            if match_pos is not None:
                cv2.circle(
                    image,
                    (int(match_pos[0]), int(match_pos[1])),
                    max(1, point_radius - 2),
                    color,
                    -1,
                )
        else:
            cv2.circle(image, (abs_x, abs_y), point_radius, color, -1)
