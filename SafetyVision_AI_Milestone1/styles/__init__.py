"""
Detecto AI - Styling & Dynamic Theme Package
Provides theme management, custom CSS injection, and Plotly theme formatters.
"""

from styles.theme_manager import (
    apply_theme_css,
    get_chart_theme,
    apply_chart_theme,
    get_theme_mode,
    set_theme_mode,
    render_theme_switcher,
    DARK_PALETTE,
    LIGHT_PALETTE
)

__all__ = [
    "apply_theme_css",
    "get_chart_theme",
    "apply_chart_theme",
    "get_theme_mode",
    "set_theme_mode",
    "render_theme_switcher",
    "DARK_PALETTE",
    "LIGHT_PALETTE"
]

