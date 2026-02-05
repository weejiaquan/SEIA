"""
Utility functions for SEIA automation scripts.

Provides common utilities for config loading, debug detection, CSV export,
and template name processing.
"""

from __future__ import annotations

import csv
import inspect
import json
import os
import re
from typing import Callable, Sequence


def _get_caller_dir() -> str:
    """Get the directory of the calling script (skipping engine frames)."""
    engine_dir = os.path.dirname(os.path.abspath(__file__))

    for frame_info in inspect.stack()[2:]:  # Skip this function and immediate caller
        filename = os.path.abspath(frame_info.filename)
        if not filename.startswith(engine_dir):
            return os.path.dirname(filename)

    return os.getcwd()


def load_script_config(script_file: str | None = None) -> dict:
    """
    Load config.json from the script's directory.

    Args:
        script_file: Optional explicit path to script. If None, auto-detects
                    from call stack.

    Returns:
        Config dict, or empty dict if not found.
    """
    if script_file is not None:
        config_dir = os.path.dirname(os.path.abspath(script_file))
    else:
        config_dir = _get_caller_dir()

    config_path = os.path.join(config_dir, "config.json")

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    return {}


def is_debug_mode(config: dict | None = None) -> bool:
    """
    Check if debug mode is enabled.

    Debug mode is on if either config.debug.log or config.debug.screenshot is True.

    Args:
        config: Optional config dict. If None, loads from calling script's config.json.

    Returns:
        True if debug mode is enabled.
    """
    if config is None:
        config = load_script_config()

    debug_config = config.get("debug", {})
    return bool(debug_config.get("log", False) or debug_config.get("screenshot", False))


def clean_template_name(
    template_name: str,
    prefix: str = "Student_Portrait_",
    suffix: str = "_Collection.png",
) -> str:
    """
    Clean up a template filename by removing prefix and suffix.

    Args:
        template_name: Raw template filename
        prefix: Prefix to remove (default: "Student_Portrait_")
        suffix: Suffix to remove (default: "_Collection.png")

    Returns:
        Cleaned name, e.g., "CH0001"
    """
    if not template_name:
        return ""

    result = template_name
    if prefix and result.startswith(prefix):
        result = result[len(prefix):]
    if suffix and result.endswith(suffix):
        result = result[:-len(suffix)]

    return result


def export_csv(
    data: Sequence[str] | Sequence[dict],
    output_file: str,
    fieldnames: list[str] | None = None,
) -> None:
    """
    Export data to CSV file.

    Supports two modes:
    - Single column: list of strings -> each string is a row
    - Multi-column: list of dicts -> each dict is a row with fieldnames as columns

    Args:
        data: List of strings (single column) or list of dicts (multi-column)
        output_file: Path to output CSV file
        fieldnames: Column names. For strings, defaults to ["value"].
                   For dicts, required.

    Example:
        # Single column
        export_csv(["CH0001", "CH0002"], "students.csv")

        # Multi-column
        export_csv(
            [{"id": "CH0001", "name": "Alice"}],
            "data.csv",
            fieldnames=["id", "name"]
        )
    """
    if not data:
        # Write empty file with headers
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            if fieldnames:
                writer = csv.writer(f)
                writer.writerow(fieldnames)
        print(f"Exported 0 items to {output_file}")
        return

    # Auto-detect mode from first element
    first = data[0]

    if isinstance(first, dict):
        # Multi-column mode
        if fieldnames is None:
            raise ValueError("fieldnames required for dict data")

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
    else:
        # Single column mode
        if fieldnames is None:
            fieldnames = ["value"]

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)
            for item in data:
                writer.writerow([item])

    print(f"Exported {len(data)} items to {output_file}")
