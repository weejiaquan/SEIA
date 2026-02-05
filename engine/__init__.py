from .core import AutomationEngine
from .capture import ScreenCapture
from .debug_draw import draw_match_rect, draw_verification_results
from .detector import FeatureDetector, TemplateDetector, RoiDeltaDetector, parse_pixel_points, pixel_points_check
from .mapper import (
    find_window,
    get_client_origin_and_size,
    is_window_minimized,
    map_point,
    map_rect,
    set_process_dpi_awareness,
    set_window_topmost,
)
from .ocr import TextResult, available_engines, parse_number
from .runtime import scan_templates_clip, scan_screen_clip, present_clip
from .utils import (
    clean_template_name,
    export_csv,
    is_debug_mode,
    load_script_config,
)
from .scan_utils import (
    collect_unique_matches,
    detect_stop_marker,
    filter_cells_by_stop_marker,
    save_scan_results,
    stitch_debug_visualizations,
)

__all__ = [
    "AutomationEngine",
    "ScreenCapture",
    "draw_match_rect",
    "draw_verification_results",
    "TemplateDetector",
    "FeatureDetector",
    "RoiDeltaDetector",
    "parse_pixel_points",
    "pixel_points_check",
    "find_window",
    "get_client_origin_and_size",
    "is_window_minimized",
    "map_point",
    "map_rect",
    "set_process_dpi_awareness",
    "set_window_topmost",
    "TextResult",
    "available_engines",
    "parse_number",
    "scan_templates_clip",
    "scan_screen_clip",
    "present_clip",
    # Utils
    "clean_template_name",
    "export_csv",
    "is_debug_mode",
    "load_script_config",
    # Scan utils
    "collect_unique_matches",
    "detect_stop_marker",
    "filter_cells_by_stop_marker",
    "save_scan_results",
    "stitch_debug_visualizations",
]
