from __future__ import annotations

import ctypes
import logging
import os
from typing import Optional, Tuple
from ctypes import wintypes

try:
    import win32api
    import win32con
    import win32gui
    import win32process

    _PYWIN32_AVAILABLE = True
    _PYWIN32_ERROR: Optional[Exception] = None
except Exception as exc:
    win32api = None  # type: ignore[assignment]
    win32con = None  # type: ignore[assignment]
    win32gui = None  # type: ignore[assignment]
    win32process = None  # type: ignore[assignment]
    _PYWIN32_AVAILABLE = False
    _PYWIN32_ERROR = exc

    from ctypes import wintypes

    _user32 = ctypes.windll.user32
    _kernel32 = ctypes.windll.kernel32

    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _PROCESS_QUERY_INFORMATION = 0x0400

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    _user32.EnumWindows.argtypes = [WNDENUMPROC, wintypes.LPARAM]
    _user32.EnumWindows.restype = wintypes.BOOL
    _user32.IsWindowVisible.argtypes = [wintypes.HWND]
    _user32.IsWindowVisible.restype = wintypes.BOOL
    _user32.IsIconic.argtypes = [wintypes.HWND]
    _user32.IsIconic.restype = wintypes.BOOL
    _user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    _user32.ShowWindow.restype = wintypes.BOOL
    _user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    _user32.SetForegroundWindow.restype = wintypes.BOOL
    _user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    _user32.GetWindowTextLengthW.restype = ctypes.c_int
    _user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    _user32.GetWindowTextW.restype = ctypes.c_int
    _user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    _user32.GetWindowRect.restype = wintypes.BOOL
    _user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    _user32.GetClientRect.restype = wintypes.BOOL
    _user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    _user32.ClientToScreen.restype = wintypes.BOOL
    _user32.MoveWindow.argtypes = [
        wintypes.HWND,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.BOOL,
    ]
    _user32.MoveWindow.restype = wintypes.BOOL
    _user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    _user32.GetWindowThreadProcessId.restype = wintypes.DWORD

    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    _kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL


def set_process_dpi_awareness() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_AWARE
        logging.info("DPI awareness set via shcore.SetProcessDpiAwareness")
        return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
        logging.info("DPI awareness set via user32.SetProcessDPIAware")
    except Exception as exc:
        logging.warning("Failed to set DPI awareness: %s", exc)


_warned_pywin32 = False


def _warn_pywin32_missing() -> None:
    global _warned_pywin32
    if _PYWIN32_AVAILABLE or _warned_pywin32:
        return
    _warned_pywin32 = True
    logging.warning("pywin32 import failed; using ctypes fallback: %s", _PYWIN32_ERROR)


def _process_name_matches(actual: Optional[str], expected: Optional[str]) -> bool:
    if not expected:
        return True
    if not actual:
        return False
    actual_l = actual.lower()
    expected_l = expected.lower()
    actual_base = actual_l[:-4] if actual_l.endswith(".exe") else actual_l
    expected_base = expected_l[:-4] if expected_l.endswith(".exe") else expected_l
    return actual_l == expected_l or actual_base == expected_base


def _get_process_name(hwnd: int) -> Optional[str]:
    if not _PYWIN32_AVAILABLE:
        return _get_process_name_ctypes(hwnd)
    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        access = win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ
        handle = win32api.OpenProcess(access, False, pid)
        try:
            path = win32process.GetModuleFileNameEx(handle, 0)
            return os.path.basename(path)
        finally:
            win32api.CloseHandle(handle)
    except Exception:
        return None


def _get_process_name_ctypes(hwnd: int) -> Optional[str]:
    if _PYWIN32_AVAILABLE:
        return None
    from ctypes import wintypes

    pid = wintypes.DWORD()
    _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if pid.value == 0:
        return None
    handle = _kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not handle:
        handle = _kernel32.OpenProcess(_PROCESS_QUERY_INFORMATION, False, pid.value)
        if not handle:
            return None
    try:
        size = wintypes.DWORD(1024)
        buf = ctypes.create_unicode_buffer(size.value)
        if _kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return os.path.basename(buf.value)
    finally:
        _kernel32.CloseHandle(handle)
    return None


def find_window(title_substring: str, process_name: Optional[str] = None) -> Optional[int]:
    if not title_substring:
        return None
    title_substring = title_substring.lower()
    process_name_l = process_name.lower() if process_name else None
    matches: list[int] = []
    if _PYWIN32_AVAILABLE:
        def enum_cb(hwnd: int, _param: object) -> bool:
            if not win32gui.IsWindowVisible(hwnd):
                return True
            title = win32gui.GetWindowText(hwnd)
            if not title:
                return True
            if title_substring not in title.lower():
                return True
            if process_name_l:
                name = _get_process_name(hwnd)
                if not _process_name_matches(name, process_name_l):
                    return True
            matches.append(hwnd)
            return True

        win32gui.EnumWindows(enum_cb, None)
        return matches[0] if matches else None

    _warn_pywin32_missing()
    from ctypes import wintypes

    @WNDENUMPROC
    def enum_cb(hwnd: int, _param: int) -> bool:
        if not _user32.IsWindowVisible(hwnd):
            return True
        length = _user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        _user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value
        if title_substring not in title.lower():
            return True
        if process_name_l:
            name = _get_process_name_ctypes(hwnd)
            if not _process_name_matches(name, process_name_l):
                return True
        matches.append(int(hwnd))
        return True

    _user32.EnumWindows(enum_cb, wintypes.LPARAM(0))
    return matches[0] if matches else None


def is_window_minimized(hwnd: int) -> bool:
    try:
        if _PYWIN32_AVAILABLE:
            return win32gui.IsIconic(hwnd)
        _warn_pywin32_missing()
        return bool(_user32.IsIconic(hwnd))
    except Exception:
        return False


def focus_window(hwnd: int) -> bool:
    try:
        if _PYWIN32_AVAILABLE:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
            return True
        _warn_pywin32_missing()
        _user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        return bool(_user32.SetForegroundWindow(hwnd))
    except Exception as exc:
        logging.warning("Failed to focus window: %s", exc)
        return False


def set_window_client_size(hwnd: int, client_w: int, client_h: int) -> bool:
    try:
        if client_w <= 0 or client_h <= 0:
            return False
        if _PYWIN32_AVAILABLE:
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            win_w = right - left
            win_h = bottom - top
            c_left, c_top, c_right, c_bottom = win32gui.GetClientRect(hwnd)
            cur_client_w = c_right - c_left
            cur_client_h = c_bottom - c_top
            extra_w = win_w - cur_client_w
            extra_h = win_h - cur_client_h
            target_w = max(1, int(client_w) + extra_w)
            target_h = max(1, int(client_h) + extra_h)
            return bool(win32gui.MoveWindow(hwnd, left, top, target_w, target_h, True))
        _warn_pywin32_missing()
        rect = wintypes.RECT()
        if not _user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return False
        client_rect = wintypes.RECT()
        if not _user32.GetClientRect(hwnd, ctypes.byref(client_rect)):
            return False
        win_w = rect.right - rect.left
        win_h = rect.bottom - rect.top
        cur_client_w = client_rect.right - client_rect.left
        cur_client_h = client_rect.bottom - client_rect.top
        extra_w = win_w - cur_client_w
        extra_h = win_h - cur_client_h
        target_w = max(1, int(client_w) + extra_w)
        target_h = max(1, int(client_h) + extra_h)
        return bool(_user32.MoveWindow(hwnd, rect.left, rect.top, target_w, target_h, True))
    except Exception as exc:
        logging.warning("Failed to resize window: %s", exc)
        return False


def get_client_origin_and_size(hwnd: int) -> Tuple[int, int, int, int]:
    if _PYWIN32_AVAILABLE:
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        width = max(0, right - left)
        height = max(0, bottom - top)
        origin_x, origin_y = win32gui.ClientToScreen(hwnd, (0, 0))
        return origin_x, origin_y, width, height

    _warn_pywin32_missing()
    from ctypes import wintypes

    rect = wintypes.RECT()
    if not _user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return 0, 0, 0, 0
    width = max(0, rect.right - rect.left)
    height = max(0, rect.bottom - rect.top)
    point = wintypes.POINT(0, 0)
    _user32.ClientToScreen(hwnd, ctypes.byref(point))
    return int(point.x), int(point.y), width, height


def map_point(
    ref_point: Tuple[int, int],
    ref_size: Tuple[int, int],
    client_origin: Tuple[int, int],
    client_size: Tuple[int, int],
) -> Tuple[int, int]:
    ref_w, ref_h = ref_size
    client_w, client_h = client_size
    if ref_w <= 0 or ref_h <= 0:
        return client_origin
    scale_x = client_w / ref_w
    scale_y = client_h / ref_h
    x = int(round(client_origin[0] + ref_point[0] * scale_x))
    y = int(round(client_origin[1] + ref_point[1] * scale_y))
    return x, y


def map_rect(
    ref_rect: Tuple[int, int, int, int],
    ref_size: Tuple[int, int],
    client_origin: Tuple[int, int],
    client_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    ref_x, ref_y, ref_w, ref_h = ref_rect
    ref_size_w, ref_size_h = ref_size
    client_w, client_h = client_size
    if ref_size_w <= 0 or ref_size_h <= 0:
        return client_origin[0], client_origin[1], 0, 0
    scale_x = client_w / ref_size_w
    scale_y = client_h / ref_size_h
    left = int(round(client_origin[0] + ref_x * scale_x))
    top = int(round(client_origin[1] + ref_y * scale_y))
    width = int(round(ref_w * scale_x))
    height = int(round(ref_h * scale_y))
    return left, top, width, height
