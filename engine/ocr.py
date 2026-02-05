from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class TextResult:
    """Single OCR detection result."""

    text: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) in ROI space
    engine_name: str = ""


class OCREngine(ABC):
    """Abstract base for OCR engines."""

    @abstractmethod
    def initialize(self, languages: list[str], use_gpu: bool) -> None:
        """Initialize the engine with given languages and GPU preference."""

    @abstractmethod
    def read(self, image_bgr: np.ndarray) -> list[TextResult]:
        """Run OCR on a BGR image. Returns list of TextResult."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine identifier string."""

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Whether the engine has been initialized."""


class PaddleOCREngine(OCREngine):
    """Wraps PaddleOCR."""

    _LANG_MAP = {
        "ja": "japan",
        "en": "en",
        "zh": "ch",
        "ko": "korean",
        "fr": "french",
        "de": "german",
    }

    def __init__(self) -> None:
        self._ocr: object = None
        self._initialized = False

    @property
    def name(self) -> str:
        return "paddleocr"

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(self, languages: list[str], use_gpu: bool) -> None:
        from paddleocr import PaddleOCR  # type: ignore[import-untyped]

        gpu = use_gpu
        if use_gpu:
            try:
                import paddle  # type: ignore[import-untyped]
                if not paddle.device.is_compiled_with_cuda():
                    logging.info("PaddleOCR: CUDA not available, falling back to CPU")
                    gpu = False
            except Exception:
                gpu = False

        # PaddleOCR only supports one lang at a time; pick first mapped lang
        paddle_lang = "en"
        for lang in languages:
            mapped = self._LANG_MAP.get(lang)
            if mapped:
                paddle_lang = mapped
                break

        self._ocr = PaddleOCR(use_angle_cls=True, lang=paddle_lang, use_gpu=gpu, show_log=False)
        self._initialized = True
        logging.info("PaddleOCR initialized: lang=%s gpu=%s", paddle_lang, gpu)

    def read(self, image_bgr: np.ndarray) -> list[TextResult]:
        if not self._initialized or self._ocr is None:
            return []
        result = self._ocr.ocr(image_bgr, cls=True)
        results: list[TextResult] = []
        if not result:
            return results
        for line_group in result:
            if not line_group:
                continue
            for line in line_group:
                if not line or len(line) < 2:
                    continue
                box_pts, (text, confidence) = line[0], line[1]
                # box_pts is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                xs = [p[0] for p in box_pts]
                ys = [p[1] for p in box_pts]
                x = int(min(xs))
                y = int(min(ys))
                w = int(max(xs)) - x
                h = int(max(ys)) - y
                results.append(TextResult(
                    text=str(text),
                    confidence=float(confidence),
                    bbox=(x, y, w, h),
                    engine_name=self.name,
                ))
        return results


class MangaOCREngine(OCREngine):
    """Wraps manga-ocr (Japanese manga text recognition)."""

    def __init__(self) -> None:
        self._ocr: object = None
        self._initialized = False

    @property
    def name(self) -> str:
        return "manga_ocr"

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(self, languages: list[str], use_gpu: bool) -> None:
        from manga_ocr import MangaOcr  # type: ignore[import-untyped]

        self._ocr = MangaOcr()
        self._initialized = True
        logging.info("MangaOCR initialized (GPU auto-detected by transformers)")

    def read(self, image_bgr: np.ndarray) -> list[TextResult]:
        if not self._initialized or self._ocr is None:
            return []
        import cv2
        from PIL import Image  # type: ignore[import-untyped]

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        text = self._ocr(pil_img)
        if not text:
            return []
        return [TextResult(
            text=str(text),
            confidence=1.0,
            bbox=None,
            engine_name=self.name,
        )]


class EasyOCREngine(OCREngine):
    """Wraps EasyOCR."""

    def __init__(self) -> None:
        self._reader: object = None
        self._initialized = False

    @property
    def name(self) -> str:
        return "easyocr"

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(self, languages: list[str], use_gpu: bool) -> None:
        import easyocr  # type: ignore[import-untyped]

        gpu = use_gpu
        if use_gpu:
            try:
                import torch  # type: ignore[import-untyped]
                if not torch.cuda.is_available():
                    logging.info("EasyOCR: CUDA not available, falling back to CPU")
                    gpu = False
            except ImportError:
                gpu = False

        # EasyOCR uses language codes directly
        self._reader = easyocr.Reader(languages, gpu=gpu)
        self._initialized = True
        logging.info("EasyOCR initialized: languages=%s gpu=%s", languages, gpu)

    def read(self, image_bgr: np.ndarray) -> list[TextResult]:
        if not self._initialized or self._reader is None:
            return []
        detections = self._reader.readtext(image_bgr)
        results: list[TextResult] = []
        for det in detections:
            box, text, confidence = det[0], det[1], det[2]
            # box is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x = int(min(xs))
            y = int(min(ys))
            w = int(max(xs)) - x
            h = int(max(ys)) - y
            results.append(TextResult(
                text=str(text),
                confidence=float(confidence),
                bbox=(x, y, w, h),
                engine_name=self.name,
            ))
        return results


_ENGINE_CLASSES: dict[str, type[OCREngine]] = {
    "paddleocr": PaddleOCREngine,
    "manga_ocr": MangaOCREngine,
    "easyocr": EasyOCREngine,
}


def create_ocr_engine(name: str) -> OCREngine:
    """Create an OCR engine instance by name.

    Raises KeyError if name is not recognized.
    """
    cls = _ENGINE_CLASSES.get(name)
    if cls is None:
        raise KeyError(f"Unknown OCR engine: {name!r}. Available: {list(_ENGINE_CLASSES)}")
    return cls()


def available_engines() -> list[str]:
    """Probe which OCR packages are importable and return their names."""
    available: list[str] = []
    _import_checks = {
        "paddleocr": "paddleocr",
        "manga_ocr": "manga_ocr",
        "easyocr": "easyocr",
    }
    for engine_name, module_name in _import_checks.items():
        try:
            __import__(module_name)
            available.append(engine_name)
        except ImportError:
            pass
    return available


# -- Number parsing ----------------------------------------------------------

_NUMBER_PATTERN = re.compile(
    r"[+-]?\s*(?:\d[\d,.\s]*\d|\d)"
)

_OCR_DIGIT_FIXES: dict[str, str] = {
    "O": "0",
    "o": "0",
    "l": "1",
    "I": "1",
    "i": "1",
    "S": "5",
    "s": "5",
    "B": "8",
    "Z": "2",
    "z": "2",
    "G": "6",
    "g": "9",
    "q": "9",
    "b": "6",
    "D": "0",
}


def _normalize_fullwidth(text: str) -> str:
    """Convert full-width digits and symbols to ASCII equivalents."""
    out = []
    for ch in text:
        cp = ord(ch)
        # Full-width digits ０-９ (U+FF10 - U+FF19) → 0-9
        if 0xFF10 <= cp <= 0xFF19:
            out.append(chr(cp - 0xFF10 + ord("0")))
        # Full-width +- signs
        elif cp == 0xFF0B:  # ＋
            out.append("+")
        elif cp == 0xFF0D:  # −
            out.append("-")
        # Full-width comma/period
        elif cp == 0xFF0C:  # ，
            out.append(",")
        elif cp == 0xFF0E:  # ．
            out.append(".")
        else:
            out.append(ch)
    return "".join(out)


def parse_number(text: str) -> Optional[float]:
    """Extract a number from OCR text, handling common OCR errors.

    Normalizes full-width digits (１２３ → 123).
    Only applies OCR character fixes to characters adjacent to real digits.
    Returns None if no number can be extracted.
    """
    if not text:
        return None
    # Normalize full-width digits/symbols to ASCII
    text = _normalize_fullwidth(text)
    # Only process text that contains at least one real digit
    if not any(ch.isdigit() for ch in text):
        return None
    # First try plain extraction without OCR fixes
    match = _NUMBER_PATTERN.search(text)
    if match:
        num_str = match.group(0).replace(" ", "").replace(",", "")
        try:
            plain_result = float(num_str)
        except ValueError:
            plain_result = None
    else:
        plain_result = None
    # Apply OCR fixes only to characters adjacent to real digits
    chars = list(text)
    digit_positions = {i for i, ch in enumerate(chars) if ch.isdigit()}
    cleaned = []
    for i, ch in enumerate(chars):
        if ch.isdigit() or ch in "+-.,":
            cleaned.append(ch)
        elif ch in _OCR_DIGIT_FIXES and _near_digit(i, digit_positions):
            cleaned.append(_OCR_DIGIT_FIXES[ch])
        elif ch.isspace():
            cleaned.append(ch)
    cleaned_str = "".join(cleaned)
    match = _NUMBER_PATTERN.search(cleaned_str)
    if match:
        num_str = match.group(0).replace(" ", "").replace(",", "")
        try:
            fixed_result = float(num_str)
        except ValueError:
            fixed_result = None
    else:
        fixed_result = None
    # Prefer the OCR-fixed result if it produced a longer number string
    if fixed_result is not None and plain_result is not None:
        return fixed_result
    return fixed_result if fixed_result is not None else plain_result


def _near_digit(index: int, digit_positions: set[int], window: int = 2) -> bool:
    """Check if a character position is within *window* of a real digit."""
    for offset in range(1, window + 1):
        if (index - offset) in digit_positions or (index + offset) in digit_positions:
            return True
    return False


# -- Sorting utility ---------------------------------------------------------

def sort_by_position(results: list[TextResult]) -> list[TextResult]:
    """Sort TextResults top-to-bottom, left-to-right by bbox."""

    def _sort_key(r: TextResult) -> Tuple[int, int]:
        if r.bbox is None:
            return (0, 0)
        return (r.bbox[1], r.bbox[0])

    return sorted(results, key=_sort_key)
