"""
Utility functions for handling model and plot settings.
Supports both predefined models (from JSON) and dynamically registered custom models.
"""

import json
import os
from typing import Dict, Any, List, Optional

# Runtime registries for custom models added via the UI
_custom_models: Dict[str, Dict[str, Any]] = {}
_custom_plot_settings: Dict[str, Dict[str, Any]] = {}

# Module-level caches for JSON files (avoid re-reading from disk every call)
_cache_model_settings: Optional[Dict[str, Any]] = None
_cache_plot_settings: Optional[Dict[str, Any]] = None

# Default colour cycle used when the user does not pick a colour
_DEFAULT_CUSTOM_COLORS = [
    "#e377c2", "#bcbd22", "#17becf", "#ff7f0e",
    "#8c564b", "#9467bd", "#d62728", "#2ca02c",
]
_color_idx = 0


def load_settings(file_name: str) -> Dict[str, Any]:
    """Load settings from a JSON file, with module-level caching."""
    global _cache_model_settings, _cache_plot_settings

    # Use cache when available
    if file_name == 'model_settings.json' and _cache_model_settings is not None:
        import copy
        return copy.deepcopy(_cache_model_settings)
    if file_name == 'plot_settings.json' and _cache_plot_settings is not None:
        import copy
        return copy.deepcopy(_cache_plot_settings)

    settings_path = os.path.join(os.path.dirname(__file__), 'config', file_name)
    try:
        with open(settings_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"{file_name} file not found at {settings_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {file_name}")

    # Populate cache
    if file_name == 'model_settings.json':
        _cache_model_settings = data
    elif file_name == 'plot_settings.json':
        _cache_plot_settings = data

    import copy
    return copy.deepcopy(data)


# ---------------------------------------------------------------------------
# Model retrieval settings
# ---------------------------------------------------------------------------

def get_model_settings() -> Dict[str, Any]:
    """Get model retrieval settings (predefined + custom)."""
    settings = load_settings('model_settings.json')
    # Merge in any custom models registered at runtime
    settings['models'].update(_custom_models)
    return settings


def get_plot_settings() -> Dict[str, Any]:
    """Get plot appearance settings (predefined + custom)."""
    settings = load_settings('plot_settings.json')
    settings['models'].update(_custom_plot_settings)
    return settings


def get_available_models() -> List[str]:
    """Get list of available models (predefined + custom)."""
    settings = get_model_settings()
    return list(settings['models'].keys())


def get_predefined_models() -> List[str]:
    """Get list of predefined models only (from JSON, no custom)."""
    settings = load_settings('model_settings.json')
    return list(settings['models'].keys())


def get_custom_models() -> Dict[str, Dict[str, Any]]:
    """Return the current dictionary of custom models."""
    return dict(_custom_models)


def get_model_retrieval_settings(model_name: str) -> Dict[str, Any]:
    """Get retrieval settings for a specific model (predefined or custom)."""
    # Check custom models first (they may shadow predefined ones)
    if model_name in _custom_models:
        return dict(_custom_models[model_name])
    settings = load_settings('model_settings.json')
    if model_name not in settings['models']:
        raise ValueError(f"Model '{model_name}' not found in settings")
    return settings['models'][model_name]


def get_analysis_settings() -> Dict[str, Any]:
    """Get analysis retrieval settings."""
    settings = load_settings('model_settings.json')
    return settings['analysis']


def get_climatology_settings() -> Dict[str, Any]:
    """Get climatology retrieval settings."""
    settings = load_settings('model_settings.json')
    return settings['climatology']


# ---------------------------------------------------------------------------
# Plot settings helpers
# ---------------------------------------------------------------------------

def _next_default_color() -> str:
    """Return the next colour from the default cycle."""
    global _color_idx
    color = _DEFAULT_CUSTOM_COLORS[_color_idx % len(_DEFAULT_CUSTOM_COLORS)]
    _color_idx += 1
    return color


def _hex_to_rgb(color: str):
    """Parse a hex colour string (#RRGGBB) into (r, g, b) ints.

    Returns ``None`` if the colour is not in ``#RRGGBB`` format (e.g. a named
    colour like ``'red'``).
    """
    if (isinstance(color, str) and color.startswith('#')
            and len(color) == 7
            and all(c in '0123456789abcdefABCDEF' for c in color[1:])):
        return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    return None


def _generate_default_plot_settings(model_name: str, is_ensemble: bool,
                                     color: Optional[str] = None) -> Dict[str, Any]:
    """Generate sensible default plot settings for a custom model."""
    if color is None:
        color = _next_default_color()

    rgb = _hex_to_rgb(color)

    if is_ensemble:
        fillcolor = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.3)" if rgb else "rgba(128, 128, 128, 0.3)"
        return {
            "color": color,
            "box": {
                "fillcolor": fillcolor,
                "marker_color": color,
                "width": 0.35
            },
            "marker": {
                "jitter": 0.3,
                "pointpos": 0
            }
        }
    else:
        return {
            "color": color,
            "marker": {
                "symbol": "diamond",
                "size": 12,
                "line": {
                    "color": color,
                    "width": 2
                }
            },
            "line": None
        }


def get_model_plot_settings(model_name: str) -> Dict[str, Any]:
    """Get plot settings for a specific model (predefined or custom)."""
    if model_name in _custom_plot_settings:
        return dict(_custom_plot_settings[model_name])
    settings = load_settings('plot_settings.json')
    if model_name not in settings['models']:
        # Auto-generate settings for unknown models (e.g. custom mean names)
        is_ens = "Mean" in model_name or "ENS" in model_name
        ps = _generate_default_plot_settings(model_name, is_ens)
        _custom_plot_settings[model_name] = ps
        return ps
    return settings['models'][model_name]


def get_reference_plot_settings(reference_name: str) -> Dict[str, Any]:
    """Get plot settings for a reference (observations, analysis, climatology)."""
    settings = load_settings('plot_settings.json')
    if reference_name not in settings['reference']:
        raise ValueError(f"Reference '{reference_name}' not found in plot settings")
    return settings['reference'][reference_name]


def get_layout_settings() -> Dict[str, Any]:
    """Get general layout settings."""
    settings = load_settings('plot_settings.json')
    return settings['layout']


# ---------------------------------------------------------------------------
# Custom model registration / removal
# ---------------------------------------------------------------------------

def register_custom_model(name: str, retrieval_args: Dict[str, Any],
                          is_ensemble: bool = False,
                          n_members: int = 50,
                          color: Optional[str] = None) -> None:
    """
    Register a custom model at runtime.

    Parameters
    ----------
    name : str
        Display name for the model (must be unique).
    retrieval_args : dict
        MARS retrieval keyword arguments.  Must contain at least
        ``class``, ``type``, ``stream``.  May contain any additional
        MARS keys (``expver``, ``model``, ``database``, ``grid``, …).
    is_ensemble : bool
        Whether this model produces ensemble (perturbed) forecasts.
    n_members : int
        Number of ensemble members (only relevant if *is_ensemble* is True).
    color : str or None
        Hex colour for plotting.  If *None* a colour is picked automatically.
    """
    model_entry = dict(retrieval_args)
    model_entry['ensemble'] = is_ensemble
    # Only add 'number' when type=pf (perturbed forecast)
    if is_ensemble and 'number' not in model_entry:
        if str(model_entry.get('type', '')).lower() == 'pf':
            model_entry['number'] = [1, "TO", n_members]
    _custom_models[name] = model_entry

    # Also generate plot settings
    ps = _generate_default_plot_settings(name, is_ensemble, color=color)
    _custom_plot_settings[name] = ps

    # If ensemble, also create mean plot settings
    if is_ensemble:
        mean_name = f"{name} Mean"
        mean_color = color or ps["color"]
        rgb = _hex_to_rgb(mean_color)
        marker_rgba = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.7)" if rgb else mean_color
        _custom_plot_settings[mean_name] = {
            "color": mean_color,
            "marker": {
                "symbol": "star",
                "size": 12,
                "color": marker_rgba,
                "line": {
                    "color": mean_color,
                    "width": 2
                }
            },
            "line": {
                "color": mean_color,
                "width": 2
            }
        }


def unregister_custom_model(name: str) -> None:
    """Remove a previously registered custom model."""
    _custom_models.pop(name, None)
    _custom_plot_settings.pop(name, None)
    _custom_plot_settings.pop(f"{name} Mean", None)


def clear_custom_models() -> None:
    """Remove all custom models."""
    _custom_models.clear()
    _custom_plot_settings.clear()
    global _color_idx
    _color_idx = 0
