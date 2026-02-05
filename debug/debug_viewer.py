"""
Lightweight debug viewer for debug_out screenshots.

Usage:
    python debug/debug_viewer.py --dir debug_out
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _load_entries(folder: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not os.path.isdir(folder):
        return entries
    for name in os.listdir(folder):
        if os.path.splitext(name.lower())[1] not in IMAGE_EXTS:
            continue
        path = os.path.join(folder, name)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        meta = None
        meta_path = f"{path}.json"
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as handle:
                    meta = json.load(handle)
            except Exception:
                meta = None
        entries.append({"name": name, "path": path, "mtime": mtime, "meta": meta})
    entries.sort(key=lambda e: e["mtime"], reverse=True)
    return entries


def _status_from_meta(meta: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    if not meta:
        return "unknown", "n/a"
    typ = str(meta.get("type", "unknown"))
    status = typ
    score = meta.get("score")
    if isinstance(score, (int, float)):
        status = f"{typ} score={float(score):.3f}"
    result = "other"
    if "fail" in typ:
        result = "fail"
    elif "pass" in typ:
        result = "pass"
    return status, result


def _border_color(result: str, typ: str) -> Tuple[int, int, int]:
    if result == "pass":
        return (0, 180, 0)
    if result == "fail":
        return (0, 0, 200)
    if typ == "click":
        return (0, 200, 200)
    if typ == "match":
        return (200, 120, 0)
    return (120, 120, 120)


def _short_caller(meta: Optional[Dict[str, Any]]) -> str:
    if not meta:
        return ""
    ctx = meta.get("context") or {}
    filename = ctx.get("caller_file")
    line = ctx.get("caller_line")
    if not filename:
        return ""
    base = os.path.basename(str(filename))
    if line is None:
        return base
    return f"{base}:{int(line)}"


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _draw_text_block(img: np.ndarray, lines: List[str], origin: Tuple[int, int]) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thickness = 1
    line_h = 14
    block_h = line_h * len(lines) + 6
    block_w = max(1, max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines)) + 8
    cv2.rectangle(img, (x, y), (x + block_w, y + block_h), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x + 4, y + 14 + i * line_h), font, scale, (255, 255, 255), thickness)


def _make_tile(entry: Dict[str, Any], thumb_w: int, thumb_h: int) -> np.ndarray:
    tile = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
    img = cv2.imread(entry["path"])
    if img is None or img.size == 0:
        return tile
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return tile
    scale = min(thumb_w / w, thumb_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h))
    off_x = (thumb_w - new_w) // 2
    off_y = (thumb_h - new_h) // 2
    tile[off_y:off_y + new_h, off_x:off_x + new_w] = resized

    meta = entry.get("meta")
    status_text, result = _status_from_meta(meta)
    typ = str(meta.get("type", "unknown")) if meta else "unknown"
    border = _border_color(result, typ)
    cv2.rectangle(tile, (0, 0), (thumb_w - 1, thumb_h - 1), border, 2)

    caller = _short_caller(meta)
    name = _truncate(entry["name"], 28)
    status = _truncate(status_text, 36)
    lines = [name, status]
    if caller:
        lines.append(_truncate(caller, 36))
    _draw_text_block(tile, lines, (6, 6))
    return tile


def _build_canvas(
    entries: List[Dict[str, Any]],
    page: int,
    rows: int,
    cols: int,
    thumb_w: int,
    thumb_h: int,
    selected_idx: Optional[int],
    filter_mode: str,
) -> np.ndarray:
    page_size = rows * cols
    width = cols * thumb_w
    height = rows * thumb_h
    status_h = 40
    canvas = np.zeros((height + status_h, width, 3), dtype=np.uint8)

    for i in range(page_size):
        idx = page * page_size + i
        if idx >= len(entries):
            break
        entry = entries[idx]
        tile = _make_tile(entry, thumb_w, thumb_h)
        row = i // cols
        col = i % cols
        y0 = row * thumb_h
        x0 = col * thumb_w
        canvas[y0:y0 + thumb_h, x0:x0 + thumb_w] = tile

        if selected_idx == idx:
            cv2.rectangle(canvas, (x0 + 3, y0 + 3), (x0 + thumb_w - 4, y0 + thumb_h - 4), (255, 255, 255), 2)

    status_bar = canvas[height:height + status_h, :]
    status_bar[:, :] = (20, 20, 20)
    total_pages = max(1, (len(entries) + page_size - 1) // page_size)
    header = f"{len(entries)} items | page {page + 1}/{total_pages} | filter={filter_mode} | keys: q quit, r refresh, [/] page, f filter"

    detail = ""
    if selected_idx is not None and 0 <= selected_idx < len(entries):
        entry = entries[selected_idx]
        meta = entry.get("meta") or {}
        ctx = meta.get("context") or {}
        caller = _short_caller(meta)
        score = meta.get("score")
        score_text = f"{float(score):.3f}" if isinstance(score, (int, float)) else "n/a"
        typ = str(meta.get("type", "unknown"))
        detail = f"{entry['name']} | {caller or 'n/a'} | {typ} | score={score_text}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(status_bar, _truncate(header, 120), (8, 16), font, 0.45, (220, 220, 220), 1)
    if detail:
        cv2.putText(status_bar, _truncate(detail, 120), (8, 34), font, 0.45, (220, 220, 220), 1)
    return canvas


def _apply_filter(entries: List[Dict[str, Any]], filter_mode: str) -> List[Dict[str, Any]]:
    if filter_mode == "all":
        return entries
    filtered: List[Dict[str, Any]] = []
    for entry in entries:
        meta = entry.get("meta")
        _status, result = _status_from_meta(meta)
        if filter_mode == "pass" and result == "pass":
            filtered.append(entry)
        elif filter_mode == "fail" and result == "fail":
            filtered.append(entry)
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug screenshot viewer")
    parser.add_argument("--dir", default="debug_out", help="Debug output directory")
    parser.add_argument("--rows", type=int, default=3, help="Rows per page")
    parser.add_argument("--cols", type=int, default=4, help="Cols per page")
    parser.add_argument("--thumb-w", type=int, default=360, help="Thumbnail width")
    parser.add_argument("--thumb-h", type=int, default=220, help="Thumbnail height")
    parser.add_argument("--refresh", type=float, default=1.0, help="Refresh interval (seconds)")
    args = parser.parse_args()

    folder = os.path.abspath(args.dir)
    entries = _load_entries(folder)
    filter_mode = "all"
    page = 0
    selected_idx: Optional[int] = None
    last_refresh = 0.0

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        nonlocal selected_idx
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        tile_x = x // args.thumb_w
        tile_y = y // args.thumb_h
        if tile_x < 0 or tile_y < 0:
            return
        if tile_y >= args.rows or tile_x >= args.cols:
            return
        idx = page * (args.rows * args.cols) + (tile_y * args.cols + tile_x)
        if 0 <= idx < len(entries):
            selected_idx = idx

    cv2.namedWindow("Debug Viewer")
    cv2.setMouseCallback("Debug Viewer", on_mouse)

    while True:
        now = time.time()
        if now - last_refresh >= max(0.2, args.refresh):
            entries = _load_entries(folder)
            last_refresh = now
            selected_idx = None

        filtered = _apply_filter(entries, filter_mode)
        if page < 0:
            page = 0
        page_size = args.rows * args.cols
        max_page = max(0, (len(filtered) - 1) // page_size)
        if page > max_page:
            page = max_page

        canvas = _build_canvas(
            filtered,
            page,
            args.rows,
            args.cols,
            args.thumb_w,
            args.thumb_h,
            selected_idx,
            filter_mode,
        )
        cv2.imshow("Debug Viewer", canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == 27:
            break
        if key == ord("r"):
            entries = _load_entries(folder)
        if key == ord("["):
            page = max(0, page - 1)
        if key == ord("]"):
            page = min(max_page, page + 1)
        if key == ord("f"):
            filter_mode = {"all": "pass", "pass": "fail", "fail": "all"}[filter_mode]
            page = 0

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
