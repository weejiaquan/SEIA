"""
CLIP-based template scanner for fast image matching.

Uses CLIP embeddings for similarity matching instead of pixel-based template matching.
Much faster for large template sets (247+ images).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Lazy imports for optional dependencies
_clip_model = None
_clip_processor = None
_torch = None
_device = None


def _ensure_clip():
    """Lazy load CLIP model."""
    global _clip_model, _clip_processor, _torch, _device

    if _clip_model is not None:
        return

    import torch
    from transformers import CLIPProcessor, CLIPModel

    _torch = torch
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[CLIP] Loading model on {_device}...", flush=True)
    start = time.perf_counter()

    # Use smaller/faster CLIP model with safetensors (avoids torch.load vulnerability)
    model_name = "openai/clip-vit-base-patch32"
    try:
        _clip_model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(_device)
    except Exception as e:
        print(f"[CLIP] Safetensors failed ({e}), trying default...", flush=True)
        _clip_model = CLIPModel.from_pretrained(model_name).to(_device)

    _clip_processor = CLIPProcessor.from_pretrained(model_name)
    _clip_model.eval()

    elapsed = time.perf_counter() - start
    print(f"[CLIP] Model loaded in {elapsed:.1f}s", flush=True)


@dataclass
class EmbeddingCache:
    """Cache for template embeddings."""
    embeddings: np.ndarray  # Shape: (num_templates, embedding_dim)
    template_paths: list[str]
    template_names: list[str]
    cache_file: str

    def save(self):
        """Save cache to disk."""
        np.savez(
            self.cache_file,
            embeddings=self.embeddings,
            template_paths=self.template_paths,
            template_names=self.template_names,
        )
        print(f"[CLIP] Cache saved: {self.cache_file}", flush=True)

    @classmethod
    def load(cls, cache_file: str) -> Optional["EmbeddingCache"]:
        """Load cache from disk if exists."""
        if not os.path.exists(cache_file):
            return None
        try:
            data = np.load(cache_file, allow_pickle=True)
            return cls(
                embeddings=data["embeddings"],
                template_paths=list(data["template_paths"]),
                template_names=list(data["template_names"]),
                cache_file=cache_file,
            )
        except Exception as e:
            print(f"[CLIP] Cache load failed: {e}", flush=True)
            return None


def _find_templates(folder: str, recursive: bool = True) -> list[str]:
    """Find all PNG files in folder."""
    templates = []
    if not os.path.isdir(folder):
        return templates
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith('.png'):
                templates.append(os.path.join(root, file))
        if not recursive:
            break
    return sorted(templates)


def _compute_image_embedding(image: np.ndarray) -> np.ndarray:
    """Compute CLIP embedding for a BGR image."""
    _ensure_clip()

    from PIL import Image

    # Convert BGR to RGB PIL Image
    rgb = image[:, :, ::-1]
    pil_image = Image.fromarray(rgb)

    # Process and encode
    inputs = _clip_processor(images=pil_image, return_tensors="pt").to(_device)

    with _torch.no_grad():
        outputs = _clip_model.get_image_features(**inputs)
        # Handle both tensor and BaseModelOutput
        if hasattr(outputs, 'pooler_output'):
            embedding = outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state'):
            embedding = outputs.last_hidden_state[:, 0, :]
        else:
            embedding = outputs
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    return embedding.cpu().numpy().flatten()


def _compute_batch_embeddings(images: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
    """Compute embeddings for multiple images in batches."""
    _ensure_clip()

    from PIL import Image

    all_embeddings = []

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]

        # Convert to PIL
        pil_images = []
        for img in batch:
            rgb = img[:, :, ::-1]
            pil_images.append(Image.fromarray(rgb))

        # Process batch
        inputs = _clip_processor(images=pil_images, return_tensors="pt", padding=True).to(_device)

        with _torch.no_grad():
            outputs = _clip_model.get_image_features(**inputs)
            # Handle both tensor and BaseModelOutput
            if hasattr(outputs, 'pooler_output'):
                embeddings = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state[:, 0, :]
            else:
                embeddings = outputs
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


def load_embedding_cache(cache_file: str) -> Optional[EmbeddingCache]:
    """
    Load pre-computed embedding cache without requiring original images.

    Use this when you want to distribute only the cache file (2KB per template)
    instead of the full image library (100KB+ per template).

    Args:
        cache_file: Path to the .npz cache file

    Returns:
        EmbeddingCache if loaded successfully, None otherwise
    """
    cache = EmbeddingCache.load(cache_file)
    if cache is not None:
        print(f"[CLIP] Loaded pre-computed cache: {len(cache.template_names)} templates", flush=True)
    return cache


def build_embedding_cache(
    template_folder: str,
    cache_file: Optional[str] = None,
    recursive: bool = True,
    batch_size: int = 16,
) -> EmbeddingCache:
    """
    Build or load embedding cache for templates.

    Args:
        template_folder: Path to folder containing template images
        cache_file: Path to save/load cache. Default: template_folder/clip_cache.npz
        recursive: Search subfolders
        batch_size: Batch size for embedding computation

    Returns:
        EmbeddingCache with all template embeddings
    """
    import cv2

    if cache_file is None:
        cache_file = os.path.join(template_folder, "clip_cache.npz")

    # Try to load existing cache
    cache = EmbeddingCache.load(cache_file)

    # Find current templates
    template_paths = _find_templates(template_folder, recursive)
    if not template_paths:
        raise ValueError(f"No templates found in {template_folder}")

    template_names = [os.path.basename(p) for p in template_paths]

    # Check if cache is valid
    if cache is not None:
        if set(cache.template_names) == set(template_names):
            print(f"[CLIP] Using cached embeddings for {len(template_names)} templates", flush=True)
            return cache
        else:
            print(f"[CLIP] Cache outdated, rebuilding...", flush=True)

    # Build new cache
    print(f"[CLIP] Computing embeddings for {len(template_paths)} templates...", flush=True)
    start = time.perf_counter()

    # Load all template images
    images = []
    valid_paths = []
    valid_names = []

    for i, path in enumerate(template_paths):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
            valid_paths.append(path)
            valid_names.append(os.path.basename(path))

        if (i + 1) % 50 == 0:
            print(f"[CLIP] Loaded {i + 1}/{len(template_paths)} images...", flush=True)

    print(f"[CLIP] Computing embeddings in batches of {batch_size}...", flush=True)
    embeddings = _compute_batch_embeddings(images, batch_size)

    elapsed = time.perf_counter() - start
    print(f"[CLIP] Embeddings computed in {elapsed:.1f}s", flush=True)

    cache = EmbeddingCache(
        embeddings=embeddings,
        template_paths=valid_paths,
        template_names=valid_names,
        cache_file=cache_file,
    )
    cache.save()

    return cache


def scan_with_clip(
    capture_image: np.ndarray,
    cache: EmbeddingCache,
    grid_size: Tuple[int, int] = (6, 2),
    roi_ref: Optional[Tuple[int, int, int, int]] = None,
    threshold: float = 0.7,
    row_gap: int = 0,
    col_gap: int = 0,
    top_k: int = 20,
    unique_matches: bool = True,
) -> dict:
    """
    Scan screen capture against template embeddings.

    Args:
        capture_image: BGR numpy array of screen capture
        cache: Pre-computed template embeddings
        grid_size: (columns, rows) to divide the capture into cells
        roi_ref: Region of interest (x, y, w, h) - for metadata only
        threshold: Minimum similarity score (0.0-1.0)
        row_gap: Pixels of gap between rows to exclude from cells
        col_gap: Pixels of gap between columns to exclude from cells
        top_k: Maximum matches to return per cell
        unique_matches: If True, each template can only match one cell (highest score wins)

    Returns:
        Dict with matches and scan metadata
    """
    start = time.perf_counter()

    h, w = capture_image.shape[:2]
    cols, rows = grid_size
    
    # Calculate column widths - reduce each column by gap amount
    base_cell_w = w // cols
    if col_gap > 0:
        gap_per_col = col_gap // 2
        cell_w = base_cell_w - col_gap
        col_offset_increment = base_cell_w  # Total space per column (cell + gap)
    else:
        cell_w = base_cell_w
        col_offset_increment = base_cell_w
        gap_per_col = 0
    
    # Calculate row heights - for 2-row grids, adjust to exclude gap between rows
    if rows == 2 and row_gap > 0:
        # Distribute the gap reduction across both rows
        base_row_h = h // rows
        gap_per_row = row_gap // 2
        row_heights = [base_row_h - gap_per_row, base_row_h - gap_per_row]
        gap_offset = row_gap  # Total gap between rows
    else:
        row_heights = [h // rows] * rows
        gap_offset = 0

    print(f"[CLIP] Scanning {cols}x{rows} grid (cell={cell_w}x{row_heights}, row_gap={gap_offset}px, col_gap={col_gap}px)...", flush=True)

    # Extract grid cells
    cells = []
    cell_positions = []
    
    y_offset = 0
    for row in range(rows):
        row_h = row_heights[row]
        x_offset = gap_per_col  # Start first column with half-gap margin
        for col in range(cols):
            x1 = x_offset
            y1 = y_offset
            x2 = min(x1 + cell_w, w)
            y2 = min(y1 + row_h, h)

            cell = capture_image[y1:y2, x1:x2]
            cells.append(cell)
            cell_positions.append((x1, y1, x2 - x1, y2 - y1))
            
            x_offset += col_offset_increment
        
        y_offset += row_h
        # Add gap between rows
        if row < rows - 1:
            y_offset += gap_offset

    # Compute embeddings for all cells
    print(f"[CLIP] Computing embeddings for {len(cells)} cells...", flush=True)
    cell_embeddings = _compute_batch_embeddings(cells, batch_size=len(cells))

    # Compute similarity matrix: (num_cells, num_templates)
    similarity = np.dot(cell_embeddings, cache.embeddings.T)

    # Find best match for each cell
    matches = []
    cells_by_position = []  # List of {cell_idx, row, col, best_match}

    for cell_idx in range(len(cells)):
        row = cell_idx // cols
        col = cell_idx % cols
        cell_scores = similarity[cell_idx]

        # Get top 3 matches for this cell
        top_indices = np.argsort(cell_scores)[::-1][:3]
        best_idx = top_indices[0]
        best_score = float(cell_scores[best_idx])

        cell_x, cell_y, cell_w, cell_h = cell_positions[cell_idx]

        # Build top 3 candidates list
        top_candidates = []
        for idx in top_indices:
            top_candidates.append({
                "template_name": cache.template_names[idx],
                "score": float(cell_scores[idx]),
            })

        cell_result = {
            "cell_index": cell_idx + 1,  # 1-indexed for display
            "row": row + 1,
            "col": col + 1,
            "cell_position": [cell_x, cell_y, cell_w, cell_h],
            "center_ref": [cell_x + cell_w // 2, cell_y + cell_h // 2],
            "top_candidates": top_candidates,  # Top 3 for debugging
        }

        if best_score >= threshold:
            cell_result["template_name"] = cache.template_names[best_idx]
            cell_result["template_path"] = os.path.relpath(cache.template_paths[best_idx], os.path.dirname(cache.cache_file))
            cell_result["score"] = best_score
            cell_result["matched"] = True
        else:
            cell_result["template_name"] = None
            cell_result["score"] = best_score
            cell_result["matched"] = False

        cells_by_position.append(cell_result)

        # Also add to flat matches list if matched
        if best_score >= threshold:
            matches.append({
                "template_path": os.path.relpath(cache.template_paths[best_idx], os.path.dirname(cache.cache_file)),
                "template_name": cache.template_names[best_idx],
                "score": best_score,
                "cell_index": cell_idx + 1,
                "row": row + 1,
                "col": col + 1,
                "cell_position": [cell_x, cell_y, cell_w, cell_h],
                "center_ref": [cell_x + cell_w // 2, cell_y + cell_h // 2],
                "method": "clip",
            })

    # Sort matches by score
    matches.sort(key=lambda m: m["score"], reverse=True)

    elapsed_ms = (time.perf_counter() - start) * 1000

    result = {
        "matches": matches,
        "cells": cells_by_position,  # Results organized by cell
        "scan_time_ms": round(elapsed_ms, 2),
        "templates_checked": len(cache.template_names),
        "templates_matched": len(matches),
        "grid_size": list(grid_size),
        "scan_roi_ref": list(roi_ref) if roi_ref else None,
    }

    print(f"[CLIP] Scan complete: {len(matches)} matches in {elapsed_ms:.0f}ms", flush=True)

    # Save debug visualization if debug mode enabled
    # Pass average cell dimensions for info display
    avg_cell_h = sum(row_heights) // len(row_heights) if row_heights else h // rows
    _save_grid_debug_visualization(capture_image, cells_by_position, grid_size, cell_w, avg_cell_h, cache)

    # Print cell-by-cell results
    for cell in cells_by_position:
        cell_label = f"[{cell['row']},{cell['col']}]"
        if cell["matched"]:
            print(f"[CLIP]   Cell {cell['cell_index']:2d} {cell_label}: {cell['template_name']} ({cell['score']:.3f})", flush=True)
        else:
            print(f"[CLIP]   Cell {cell['cell_index']:2d} {cell_label}: (no match, best={cell['score']:.3f})", flush=True)

    return result


def scan_fullscreen(
    capture_image: np.ndarray,
    cache: EmbeddingCache,
    threshold: float = 0.7,
) -> dict:
    """
    Simple full-screen scan - compares entire capture against all templates.

    Args:
        capture_image: BGR numpy array of screen capture
        cache: Pre-computed template embeddings
        threshold: Minimum similarity score (0.0-1.0)

    Returns:
        Dict with matches sorted by score
    """
    start = time.perf_counter()

    h, w = capture_image.shape[:2]
    print(f"[CLIP] Scanning full screen ({w}x{h})...", flush=True)

    # Compute single embedding for entire screen
    screen_embedding = _compute_image_embedding(capture_image)

    # Compare against all templates
    similarities = np.dot(cache.embeddings, screen_embedding)

    # Get all matches above threshold
    matches = []
    for idx, score in enumerate(similarities):
        if score >= threshold:
            matches.append({
                "template_path": os.path.relpath(cache.template_paths[idx], os.path.dirname(cache.cache_file)),
                "template_name": cache.template_names[idx],
                "score": float(score),
                "method": "clip_fullscreen",
            })

    # Sort by score descending
    matches.sort(key=lambda m: m["score"], reverse=True)

    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"[CLIP] Full scan complete: {len(matches)} matches in {elapsed_ms:.0f}ms", flush=True)

    return {
        "matches": matches,
        "scan_time_ms": round(elapsed_ms, 2),
        "templates_checked": len(cache.template_names),
        "templates_matched": len(matches),
    }


def scan_sliding_window(
    capture_image: np.ndarray,
    cache: EmbeddingCache,
    window_size: Tuple[int, int] = (200, 250),
    stride: Tuple[int, int] = (100, 125),
    roi_ref: Optional[Tuple[int, int, int, int]] = None,
    threshold: float = 0.7,
    nms_threshold: float = 0.3,
) -> dict:
    """
    Scan using sliding window approach for better localization.

    Args:
        capture_image: BGR numpy array
        cache: Pre-computed template embeddings
        window_size: (width, height) of sliding window
        stride: (x_stride, y_stride) between windows
        roi_ref: Region of interest for metadata
        threshold: Minimum similarity score
        nms_threshold: IoU threshold for non-max suppression

    Returns:
        Dict with matches
    """
    start = time.perf_counter()

    h, w = capture_image.shape[:2]
    win_w, win_h = window_size
    stride_x, stride_y = stride

    # Generate windows
    windows = []
    positions = []

    for y in range(0, h - win_h + 1, stride_y):
        for x in range(0, w - win_w + 1, stride_x):
            window = capture_image[y:y + win_h, x:x + win_w]
            windows.append(window)
            positions.append((x, y, win_w, win_h))

    print(f"[CLIP] Scanning {len(windows)} windows ({win_w}x{win_h}, stride {stride_x}x{stride_y})...", flush=True)

    # Compute embeddings
    window_embeddings = _compute_batch_embeddings(windows, batch_size=32)

    # Compute similarities
    similarity = np.dot(window_embeddings, cache.embeddings.T)

    # Collect all detections
    detections = []  # (score, template_idx, window_idx)

    for win_idx in range(len(windows)):
        for template_idx in range(len(cache.template_names)):
            score = similarity[win_idx, template_idx]
            if score >= threshold:
                detections.append((score, template_idx, win_idx))

    # Sort by score descending
    detections.sort(key=lambda x: x[0], reverse=True)

    # Simple NMS: keep best detection per template
    matches = []
    seen_templates = set()

    for score, template_idx, win_idx in detections:
        template_name = cache.template_names[template_idx]

        if template_name in seen_templates:
            continue

        seen_templates.add(template_name)
        x, y, ww, wh = positions[win_idx]

        matches.append({
            "template_path": os.path.relpath(cache.template_paths[template_idx], os.path.dirname(cache.cache_file)),
            "template_name": template_name,
            "score": float(score),
            "roi_ref": [x, y, ww, wh],
            "center_ref": [x + ww // 2, y + wh // 2],
            "method": "clip_sliding",
        })

    elapsed_ms = (time.perf_counter() - start) * 1000

    result = {
        "matches": matches,
        "scan_time_ms": round(elapsed_ms, 2),
        "templates_checked": len(cache.template_names),
        "templates_matched": len(matches),
        "window_size": list(window_size),
        "stride": list(stride),
        "windows_scanned": len(windows),
        "scan_roi_ref": list(roi_ref) if roi_ref else None,
    }

    print(f"[CLIP] Sliding window complete: {len(matches)} matches in {elapsed_ms:.0f}ms", flush=True)

    return result


def _save_grid_debug_visualization(
    capture_image: np.ndarray,
    cells_by_position: list,
    grid_size: Tuple[int, int],
    cell_w: int,
    cell_h: int,
    cache: EmbeddingCache = None,
) -> None:
    """Save debug visualization showing grid overlay on capture with matched templates."""
    import cv2
    import os
    
    # Check if debug mode is enabled - only save if config has debug enabled
    # For now, always save to help with tuning
    try:
        h, w = capture_image.shape[:2]
        cols, rows = grid_size
        
        # Create visualization image (convert to RGB for drawing)
        vis_image = capture_image.copy()
        
        # Prepare space below for matched templates (add extra height)
        # Need space for 2 rows of template previews
        template_preview_height = 300  # Height for template previews (2 rows)
        vis_with_previews = np.zeros((h + template_preview_height, w, 3), dtype=np.uint8)
        vis_with_previews[:h, :w] = vis_image
        
        # Draw grid cells
        for cell in cells_by_position:
            cell_x, cell_y, cell_width, cell_height = cell["cell_position"]
            cell_idx = cell["cell_index"]
            row = cell["row"]
            col = cell["col"]
            
            # Draw rectangle border
            color = (0, 255, 0) if row == 1 else (0, 255, 255)  # Green for row 1, yellow for row 2
            cv2.rectangle(vis_with_previews, (cell_x, cell_y), (cell_x + cell_width, cell_y + cell_height), color, 3)
            
            # Draw cell number
            cv2.putText(
                vis_with_previews,
                f"#{cell_idx}",
                (cell_x + 10, cell_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )
            
            # Draw center crosshair
            center_x = cell_x + cell_width // 2
            center_y = cell_y + cell_height // 2
            cv2.drawMarker(
                vis_with_previews,
                (center_x, center_y),
                (255, 0, 0),
                cv2.MARKER_CROSS,
                20,
                2,
            )
            
            # Draw top candidate match (if any)
            if cell["matched"] and "template_name" in cell:
                match_text = cell["template_name"].replace("Student_Portrait_", "").replace("_Collection.png", "")
                score_text = f"{cell['score']:.2f}"
                cv2.putText(
                    vis_with_previews,
                    f"{match_text[:12]}",
                    (cell_x + 5, cell_y + cell_height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
                cv2.putText(
                    vis_with_previews,
                    score_text,
                    (cell_x + 5, cell_y + cell_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
                
                # Load and display matched template preview at bottom
                if cache and "template_path" in cell:
                    template_full_path = os.path.join(os.path.dirname(cache.cache_file), cell["template_path"])
                    if os.path.exists(template_full_path):
                        template_img = cv2.imread(template_full_path)
                        if template_img is not None:
                            # Resize template to fit in preview area (smaller)
                            preview_w = cell_width
                            preview_h = 140  # Height per preview row
                            
                            # Maintain aspect ratio
                            t_h, t_w = template_img.shape[:2]
                            scale = min(preview_w / t_w, preview_h / t_h)
                            new_w = int(t_w * scale)
                            new_h = int(t_h * scale)
                            
                            template_resized = cv2.resize(template_img, (new_w, new_h))
                            
                            # Place template preview below the grid, aligned with cell
                            # Row 1 templates go in first preview row, Row 2 templates in second preview row
                            preview_x = cell_x + (cell_width - new_w) // 2  # Center in cell column
                            if row == 1:
                                preview_y = h + 10  # First preview row
                            else:
                                preview_y = h + 150  # Second preview row
                            
                            # Make sure it fits
                            if preview_x + new_w <= w and preview_y + new_h <= h + template_preview_height:
                                vis_with_previews[preview_y:preview_y + new_h, preview_x:preview_x + new_w] = template_resized
                                
                                # Draw border around template preview
                                cv2.rectangle(vis_with_previews, (preview_x, preview_y), 
                                            (preview_x + new_w, preview_y + new_h), color, 2)
        
        # Add info text at top
        cv2.putText(
            vis_with_previews,
            f"Grid: {cols}x{rows} | Cell: {cell_w}x{cell_h} | Capture: {w}x{h}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        
        # Save to file in scripts/character_scan directory
        output_file = "grid_debug_visualization.png"
        cv2.imwrite(output_file, vis_with_previews)
        print(f"[CLIP] Grid debug visualization saved: {output_file}", flush=True)
        
    except Exception as e:
        print(f"[CLIP] Failed to save debug visualization: {e}", flush=True)


# -- Single Template CLIP Matching (UI Automation) ---------------------------

# Cache for single template embeddings (path -> embedding)
_single_template_cache: dict[str, np.ndarray] = {}


def _get_template_embedding(template_path: str) -> Optional[np.ndarray]:
    """Get or compute CLIP embedding for a single template image."""
    global _single_template_cache

    # Check memory cache first
    abs_path = os.path.abspath(template_path)
    if abs_path in _single_template_cache:
        return _single_template_cache[abs_path]

    # Check for pre-trained cache file
    # Root templates (templates/*.png) -> cache/templates.npz
    template_name = os.path.basename(template_path)
    template_dir = os.path.dirname(abs_path)
    script_dir = os.path.dirname(template_dir)
    cache_file = os.path.join(script_dir, "cache", "templates.npz")

    if os.path.exists(cache_file):
        try:
            data = np.load(cache_file, allow_pickle=True)
            template_names = list(data["template_names"])
            if template_name in template_names:
                idx = template_names.index(template_name)
                embedding = data["embeddings"][idx]
                _single_template_cache[abs_path] = embedding
                print(f"[CLIP] Loaded from templates cache: {template_name}", flush=True)
                return embedding
        except Exception as e:
            print(f"[CLIP] Failed to load cache {cache_file}: {e}", flush=True)

    # Fall back to computing from image
    if not os.path.exists(template_path):
        print(f"[CLIP] Template not found: {template_path}", flush=True)
        return None

    import cv2
    template_img = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if template_img is None:
        print(f"[CLIP] Failed to load template: {template_path}", flush=True)
        return None

    embedding = _compute_image_embedding(template_img)
    _single_template_cache[abs_path] = embedding

    return embedding


def match_single_clip(
    capture_image: np.ndarray,
    template_path: str,
    threshold: float = 0.75,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[float, bool]:
    """
    Match a single template against a capture using CLIP embeddings.

    Args:
        capture_image: BGR numpy array of screen capture
        template_path: Path to template image file
        threshold: Minimum similarity score (0.0-1.0)
        roi: Optional (x, y, w, h) region to crop from capture before matching

    Returns:
        (score, matched) tuple:
        - score: CLIP similarity score (0.0-1.0)
        - matched: True if score >= threshold

    Note: CLIP gives overall similarity, not pixel coordinates.
          Use ROI to constrain the search region for more accurate matching.
    """
    _ensure_clip()

    # Crop to ROI if specified
    if roi is not None:
        x, y, w, h = roi
        capture_image = capture_image[y:y+h, x:x+w]

    # Get template embedding (cached)
    template_embedding = _get_template_embedding(template_path)
    if template_embedding is None:
        return 0.0, False

    # Compute capture embedding
    capture_embedding = _compute_image_embedding(capture_image)

    # Compute cosine similarity
    score = float(np.dot(template_embedding, capture_embedding))
    matched = score >= threshold

    return score, matched


def clear_single_template_cache() -> None:
    """Clear the single template embedding cache."""
    global _single_template_cache
    _single_template_cache.clear()
    print("[CLIP] Single template cache cleared", flush=True)
