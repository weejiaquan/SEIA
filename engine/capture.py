from __future__ import annotations

import ctypes
import logging
import time
from ctypes import wintypes
from typing import Optional, Tuple

import cv2
import mss
import numpy as np


_BI_RGB = 0
_DIB_RGB_COLORS = 0
_PW_RENDERFULLCONTENT = 0x00000002
_GWL_EXSTYLE = -20
_WS_EX_TOPMOST = 0x00000008
_HWND_TOPMOST = -1
_HWND_NOTOPMOST = -2
_SWP_NOMOVE = 0x0002
_SWP_NOSIZE = 0x0001
_SWP_NOACTIVATE = 0x0010
_SWP_SHOWWINDOW = 0x0040
_BLACK_WARNING_INTERVAL_S = 5.0


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", _BITMAPINFOHEADER),
        ("bmiColors", wintypes.DWORD * 3),
    ]


class ScreenCapture:
    def __init__(self) -> None:
        self._sct = mss.mss()
        self._last_capture_source: Optional[str] = None
        self._last_black_warning = 0.0

    def _warn_black(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_black_warning < _BLACK_WARNING_INTERVAL_S:
            return
        self._last_black_warning = now
        logging.warning(message)

    def grab(self, rect: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        return self.grab_screen(rect)

    def grab_screen(self, rect: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        left, top, width, height = rect
        if width <= 0 or height <= 0:
            return None
        monitor = {
            "left": int(left),
            "top": int(top),
            "width": int(width),
            "height": int(height),
        }
        try:
            raw = np.array(self._sct.grab(monitor), dtype=np.uint8)
        except Exception as exc:
            logging.warning("Screen grab failed: %s", exc)
            return None
        try:
            return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        except Exception as exc:
            logging.warning("Failed to convert capture to BGR: %s", exc)
            return None

    def grab_window(self, hwnd: int, rect: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        if not hwnd:
            return None
        left, top, width, height = rect
        if width <= 0 or height <= 0:
            return None
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        win_rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(win_rect)):
            return None
        win_left, win_top = int(win_rect.left), int(win_rect.top)
        win_width = int(win_rect.right - win_rect.left)
        win_height = int(win_rect.bottom - win_rect.top)
        if win_width <= 0 or win_height <= 0:
            return None

        hwindc = user32.GetWindowDC(hwnd)
        if not hwindc:
            return None
        memdc = gdi32.CreateCompatibleDC(hwindc)
        if not memdc:
            user32.ReleaseDC(hwnd, hwindc)
            return None
        bmp = gdi32.CreateCompatibleBitmap(hwindc, win_width, win_height)
        if not bmp:
            gdi32.DeleteDC(memdc)
            user32.ReleaseDC(hwnd, hwindc)
            return None
        gdi32.SelectObject(memdc, bmp)

        ok = user32.PrintWindow(hwnd, memdc, _PW_RENDERFULLCONTENT)
        if ok != 1:
            ok = user32.PrintWindow(hwnd, memdc, 0)
        if ok != 1:
            gdi32.DeleteObject(bmp)
            gdi32.DeleteDC(memdc)
            user32.ReleaseDC(hwnd, hwindc)
            return None

        bmi = _BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = win_width
        bmi.bmiHeader.biHeight = -win_height  # top-down
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = _BI_RGB

        buf_size = win_width * win_height * 4
        buffer = (ctypes.c_ubyte * buf_size)()
        bits = gdi32.GetDIBits(
            memdc,
            bmp,
            0,
            win_height,
            ctypes.byref(buffer),
            ctypes.byref(bmi),
            _DIB_RGB_COLORS,
        )

        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(memdc)
        user32.ReleaseDC(hwnd, hwindc)

        if bits == 0:
            return None

        img = np.frombuffer(buffer, dtype=np.uint8).reshape((win_height, win_width, 4))
        rel_x = int(left - win_left)
        rel_y = int(top - win_top)
        rel_x2 = rel_x + int(width)
        rel_y2 = rel_y + int(height)
        if rel_x2 <= 0 or rel_y2 <= 0 or rel_x >= win_width or rel_y >= win_height:
            return None
        x0 = max(0, rel_x)
        y0 = max(0, rel_y)
        x1 = min(win_width, rel_x2)
        y1 = min(win_height, rel_y2)
        if x1 <= x0 or y1 <= y0:
            return None
        cropped = img[y0:y1, x0:x1]
        try:
            return cv2.cvtColor(cropped, cv2.COLOR_BGRA2BGR)
        except Exception as exc:
            logging.warning("Failed to convert window capture to BGR: %s", exc)
            return None

    def grab_auto(self, rect: Tuple[int, int, int, int], hwnd: Optional[int], method: str) -> Optional[np.ndarray]:
        method = (method or "screen").lower()
        if method == "window_raise":
            img = self.grab_window(hwnd or 0, rect)
            if img is not None and not _looks_black(img):
                _log_capture_source(self, "window", method)
                return img
            if img is not None:
                self._warn_black("Window capture looks black; falling back to raised screen capture.")
            else:
                logging.debug("Window capture unavailable; falling back to raised screen capture.")
            screen_img = _grab_screen_with_raise(self, rect, hwnd)
            _log_capture_source(self, "screen", method)
            return screen_img
        if method == "window":
            img = self.grab_window(hwnd or 0, rect)
            if img is not None and _looks_black(img):
                self._warn_black("Window capture looks black; consider auto/screen capture.")
            _log_capture_source(self, "window", method)
            return img
        if method == "auto":
            img = self.grab_window(hwnd or 0, rect) if hwnd else None
            if img is not None and not _looks_black(img):
                _log_capture_source(self, "window", method)
                return img
            if img is not None:
                logging.warning("Window capture looks black; falling back to screen capture.")
            else:
                logging.debug("Window capture unavailable; falling back to screen capture.")
            screen_img = self.grab_screen(rect)
            _log_capture_source(self, "screen", method)
            return screen_img
        img = self.grab_screen(rect)
        _log_capture_source(self, "screen", method)
        return img


def _looks_black(img: np.ndarray) -> bool:
    if img is None or img.size == 0:
        return True
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
        gray = img[:, :, 0] if img.ndim == 3 else img
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    return mean < 5.0 and std < 2.0


def _log_capture_source(capture: "ScreenCapture", source: str, method: str) -> None:
    if capture._last_capture_source == source:
        return
    capture._last_capture_source = source
    logging.info("Capture source: %s (method=%s)", source, method)


def _grab_screen_with_raise(
    capture: ScreenCapture,
    rect: Tuple[int, int, int, int],
    hwnd: Optional[int],
) -> Optional[np.ndarray]:
    if not hwnd:
        return capture.grab_screen(rect)
    user32 = ctypes.windll.user32
    prev_hwnd = int(user32.GetForegroundWindow())
    prev_topmost = False
    was_topmost = True
    try:
        if prev_hwnd:
            prev_topmost = bool(user32.GetWindowLongW(prev_hwnd, _GWL_EXSTYLE) & _WS_EX_TOPMOST)
        was_topmost = bool(user32.GetWindowLongW(int(hwnd), _GWL_EXSTYLE) & _WS_EX_TOPMOST)
        user32.ShowWindow(int(hwnd), 9)  # SW_RESTORE
        user32.SetWindowPos(
            int(hwnd),
            _HWND_TOPMOST,
            0,
            0,
            0,
            0,
            _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOACTIVATE | _SWP_SHOWWINDOW,
        )
        user32.BringWindowToTop(int(hwnd))
        user32.SetForegroundWindow(int(hwnd))
        time.sleep(0.05)
        return capture.grab_screen(rect)
    finally:
        if not hwnd:
            return
        if not was_topmost:
            user32.SetWindowPos(
                int(hwnd),
                _HWND_NOTOPMOST,
                0,
                0,
                0,
                0,
                _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOACTIVATE | _SWP_SHOWWINDOW,
            )
        if prev_hwnd and prev_hwnd != int(hwnd) and user32.IsWindow(prev_hwnd):
            if prev_topmost:
                user32.SetWindowPos(
                    prev_hwnd,
                    _HWND_TOPMOST,
                    0,
                    0,
                    0,
                    0,
                    _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOACTIVATE | _SWP_SHOWWINDOW,
                )
            user32.BringWindowToTop(prev_hwnd)
            user32.SetForegroundWindow(prev_hwnd)
