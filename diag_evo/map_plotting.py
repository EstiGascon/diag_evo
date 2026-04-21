"""
Map plotting utilities for diag_evo.

Renders Metview contour/shading maps of GRIB fields already
retrieved and cached by ``retrieve_and_store_data``.
"""

import os
import numpy as np
import metview as mv
from datetime import datetime, timedelta

from .variables import get_base_var, get_variable_settings, get_grib_units
from .settings import get_model_retrieval_settings, get_analysis_settings

# ---------------------------------------------------------------------------
# Contour presets keyed by variable "category"
# ---------------------------------------------------------------------------

_PRECIP_LEVELS = [0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100, 200, 300]
_PRECIP_COLOURS = [
    "rgb(0.95,0.65,0.95)", "rgb(0.85,0.55,0.85)", "rgb(0.75,0.5,0.85)",
    "rgb(0.6,0.6,1)", "rgb(0,0.6,1)", "rgb(0,0.8,1.0)",
    "blue_green", "bluish_green", "yellow_green", "greenish_yellow",
    "rgb(1,1,0.5)", "yellow", "orange_yellow", "yellowish_orange",
    "rgb(1,0.45,0)", "red", "rgb(0.8,0,0)", "burgundy",
]

_TEMP_LEVELS = list(range(-30, 50, 2))
_TEMP_COLOURS = [
    "rgb(89,0,153)", "rgb(128,0,230)", "rgb(153,51,255)", "rgb(192,102,255)", "rgb(217,153,255)",
    "rgb(255,192,255)", "rgb(255,151,255)", "rgb(225,51,225)", "rgb(174,51,174)", "rgb(122,51,122)",
    "rgb(0,0,192)", "rgb(0,0,255)", "rgb(51,102,255)", "rgb(102,179,255)", "rgb(153,230,255)",
    "rgb(0,140,48)", "rgb(38,192,25)", "rgb(128,217,0)", "rgb(166,243,0)", "rgb(204,255,51)",
    "rgb(255,255,153)", "rgb(255,255,0)", "rgb(255,217,0)", "rgb(255,189,0)",
    "rgb(255,153,0)", "rgb(255,128,0)", "rgb(255,96,0)", "rgb(255,0,0)", "rgb(204,0,0)",
    "rgb(204,61,110)", "rgb(255,0,255)", "rgb(255,151,255)", "rgb(215,121,255)",
    "rgb(174,0,249)", "rgb(125,0,179)", "rgb(151,95,64)", "rgb(182,117,82)",
    "rgb(192,137,107)", "rgb(204,158,134)",
]

_GEOPOT_COLOURS = [
    "rgb(76,0,92)", "rgb(89,0,153)", "rgb(102,0,204)",
    "rgb(128,0,230)", "rgb(153,51,255)", "rgb(179,102,255)",
    "rgb(0,0,192)", "rgb(0,0,255)", "rgb(51,102,255)",
    "rgb(102,179,255)", "rgb(153,230,255)", "rgb(204,255,255)",
    "rgb(0,140,48)", "rgb(38,192,25)", "rgb(128,217,0)",
    "rgb(166,243,0)", "rgb(204,255,51)",
    "rgb(255,255,153)", "rgb(255,255,0)", "rgb(255,217,0)",
    "rgb(255,189,0)", "rgb(255,153,0)",
    "rgb(255,128,0)", "rgb(255,96,0)", "rgb(255,0,0)",
    "rgb(204,0,0)", "rgb(153,0,0)",
    "rgb(204,61,110)", "rgb(255,0,255)", "rgb(255,151,255)",
    "rgb(215,121,255)", "rgb(174,0,249)",
]

_GEOPOT_LEVEL_RANGES = {
    1000: {"min": 0,     "max": 400,   "interval": 20},
    925:  {"min": 200,   "max": 800,   "interval": 20},
    850:  {"min": 1000,  "max": 1600,  "interval": 20},
    700:  {"min": 2400,  "max": 3200,  "interval": 20},
    500:  {"min": 4800,  "max": 6000,  "interval": 40},
    400:  {"min": 6400,  "max": 7600,  "interval": 40},
    300:  {"min": 8400,  "max": 9600,  "interval": 40},
    250:  {"min": 9600,  "max": 11200, "interval": 40},
    200:  {"min": 11200, "max": 12800, "interval": 40},
    100:  {"min": 15200, "max": 16800, "interval": 40},
    50:   {"min": 19200, "max": 20800, "interval": 40},
}

_CLOUD_LEVELS = list(range(0, 105, 5))
_CLOUD_COLOURS = [
    "rgb(255,255,255)", "rgb(240,240,255)", "rgb(220,220,255)",
    "rgb(200,200,255)", "rgb(180,180,255)", "rgb(160,160,255)",
    "rgb(140,140,255)", "rgb(120,120,240)", "rgb(100,100,220)",
    "rgb(80,80,200)", "rgb(60,60,180)", "rgb(50,50,170)",
    "rgb(40,40,160)", "rgb(30,30,150)", "rgb(20,20,140)",
    "rgb(15,15,130)", "rgb(10,10,120)", "rgb(5,5,110)",
    "rgb(0,0,100)", "rgb(0,0,80)",
]

_PRESSURE_LEVELS = list(range(960, 1060, 2))
_PRESSURE_COLOURS = [
    "rgb(76,0,92)", "rgb(89,0,153)", "rgb(102,0,204)",
    "rgb(128,0,230)", "rgb(153,51,255)", "rgb(179,102,255)",
    "rgb(0,0,192)", "rgb(0,0,255)", "rgb(51,102,255)",
    "rgb(102,179,255)", "rgb(153,230,255)", "rgb(204,255,255)",
    "rgb(0,140,48)", "rgb(38,192,25)", "rgb(128,217,0)",
    "rgb(166,243,0)", "rgb(204,255,51)",
    "rgb(255,255,153)", "rgb(255,255,0)", "rgb(255,217,0)",
    "rgb(255,189,0)", "rgb(255,153,0)",
    "rgb(255,128,0)", "rgb(255,96,0)", "rgb(255,0,0)",
    "rgb(204,0,0)", "rgb(153,0,0)",
    "rgb(204,61,110)", "rgb(255,0,255)", "rgb(255,151,255)",
    "rgb(215,121,255)", "rgb(174,0,249)",
    "rgb(125,0,179)", "rgb(151,95,64)", "rgb(182,117,82)",
    "rgb(192,137,107)", "rgb(204,158,134)", "rgb(204,204,204)",
    "rgb(179,179,179)", "rgb(153,153,153)", "rgb(128,128,128)",
    "rgb(76,76,76)", "rgb(50,50,50)", "rgb(30,30,30)",
    "rgb(20,20,20)", "rgb(10,10,10)", "rgb(5,5,5)",
    "rgb(0,0,0)",
]

# Wind speed levels & colours (based on ecCharts sh_mc_wind_f0t80)
_WIND_LEVELS = [0, 2, 4, 6, 8, 10, 12, 15, 18, 21, 25, 30, 35, 40, 45, 50, 80]
_WIND_COLOURS = [
    "RGB(0.85,0.99,0.93)", "RGB(0.45,0.98,0.96)", "RGB(0,0.73,1)",
    "RGB(0,0.18,1)", "RGB(0.03,0.01,0.64)", "RGB(0.16,0.99,0.14)",
    "RGB(0.57,1,0)", "RGB(0.83,1,0)", "RGB(1,0.98,0)",
    "RGB(1,0.85,0)", "RGB(1,0.63,0)", "RGB(1,0.4,0)",
    "RGB(1,0.12,0)", "RGB(1,0,0.67)", "RGB(0.85,0,1)",
    "RGB(0.47,0.01,0.55)",
]


def _get_var_category(param):
    """Map a param shortName to a contour category."""
    base = get_base_var(param)
    if base == 'tp':
        return 'precipitation'
    if base in ('2t', '2d', 't'):
        return 'temperature'
    if base == 'z':
        return 'geopotential'
    if base in ('tcc', 'lcc', 'mcc', 'hcc'):
        return 'cloud_cover'
    if base == 'msl':
        return 'pressure'
    if base in ('10u', '10v', '10si', '10fg', '10fg6', '100u', '100v', '100si', 'u', 'v', 'ws'):
        return 'wind'
    # Fallback: infer from units / description in variable_settings.json
    try:
        vs = get_variable_settings(param)
    except (ValueError, KeyError):
        return 'default'
    units = str(vs.get('units', '')).lower()
    desc = str(vs.get('description', '')).lower()
    if 'm/s' in units or 'wind' in desc or 'gust' in desc:
        return 'wind'
    if units in ('°c', 'c', 'k') or 'temperature' in desc:
        return 'temperature'
    if units in ('mm', 'm') and ('precip' in desc or 'rain' in desc or 'snow' in desc):
        return 'precipitation'
    if units in ('hpa', 'pa', 'mb') or 'pressure' in desc:
        return 'pressure'
    if units == '%' and ('cloud' in desc or 'cover' in desc):
        return 'cloud_cover'
    return 'default'


def _build_contour(param, level=None):
    """Return an ``mv.mcont`` object appropriate for *param*."""
    cat = _get_var_category(param)

    if cat == 'precipitation':
        return mv.mcont(
            legend="on", contour="off",
            contour_level_selection_type="level_list",
            contour_level_list=_PRECIP_LEVELS,
            contour_label="off", contour_shade="on",
            contour_shade_colour_method="list",
            contour_shade_technique="grid_shading",
            contour_shade_colour_list=_PRECIP_COLOURS,
        )

    if cat == 'temperature':
        return mv.mcont(
            legend="on", contour="off",
            contour_level_selection_type="level_list",
            contour_level_list=_TEMP_LEVELS,
            contour_label="off", contour_shade="on",
            contour_shade_colour_method="list",
            contour_shade_technique="grid_shading",
            contour_shade_colour_list=_TEMP_COLOURS,
        )

    if cat == 'geopotential':
        lr = _GEOPOT_LEVEL_RANGES.get(level, {"min": 0, "max": 20800, "interval": 40})
        return mv.mcont(
            legend="on", contour="on",
            contour_line_colour="charcoal", contour_line_thickness=2,
            contour_highlight_colour="charcoal",
            contour_highlight_thickness=4,
            contour_level_selection_type="interval",
            contour_interval=lr["interval"],
            contour_label="on", contour_label_height=0.4,
            contour_label_colour="black", contour_label_frequency=2,
            contour_shade="on", contour_shade_method="area_fill",
            contour_shade_max_level=lr["max"],
            contour_shade_min_level=lr["min"],
            contour_shade_colour_method="list",
            contour_shade_technique="grid_shading",
            contour_shade_colour_list=_GEOPOT_COLOURS,
        )

    if cat == 'cloud_cover':
        return mv.mcont(
            legend="on", contour="off",
            contour_level_selection_type="level_list",
            contour_level_list=_CLOUD_LEVELS,
            contour_label="off", contour_shade="on",
            contour_shade_colour_method="list",
            contour_shade_technique="grid_shading",
            contour_shade_colour_list=_CLOUD_COLOURS,
        )

    if cat == 'pressure':
        return mv.mcont(
            legend="off", contour="on",
            contour_line_colour="navy", contour_line_thickness=1.7,
            contour_highlight_colour="navy",
            contour_highlight_thickness=3,
            contour_level_selection_type="interval",
            contour_interval=4,
            contour_label="on", contour_label_height=0.4,
            contour_shade="off",
        )

    if cat == 'wind':
        return mv.mcont(
            legend="on", contour="off",
            contour_level_selection_type="level_list",
            contour_level_list=_WIND_LEVELS,
            contour_label="off", contour_shade="on",
            contour_shade_colour_method="list",
            contour_shade_method="area_fill",
            contour_shade_technique="grid_shading",
            contour_shade_colour_list=_WIND_COLOURS,
        )

    # Default fallback
    return mv.mcont(
        legend="on", contour="off",
        contour_level_selection_type="count",
        contour_level_count=20,
        contour_label="off", contour_shade="on",
        contour_shade_colour_method="palette",
        contour_shade_palette_name="eccharts_rainbow_blue_red_20",
        contour_shade_technique="grid_shading",
    )


def _build_coastlines():
    """Return a styled ``mv.mcoast`` object."""
    return mv.mcoast(
        map_coastline_colour="charcoal",
        map_coastline_thickness=3.5,
        map_coastline_resolution="medium",
        map_coastline_land_shade="on",
        map_coastline_land_shade_colour="cream",
        map_coastline_sea_shade="off",
        map_boundaries="on",
        map_rivers="on",
        map_boundaries_colour="charcoal",
        map_boundaries_thickness=2.5,
        map_grid_colour="tan",
        map_label_colour="RGB(0,0,0)",
    )


def _build_geoview(area, plot_radius=2):
    """Return a ``mv.geoview`` with padded area."""
    n, w, s, e = area
    padded = [n + plot_radius, w - plot_radius * 1.5,
              s - plot_radius, e + plot_radius * 1.5]
    center_lon = (w + e) / 2.0
    return mv.geoview(
        map_projection="polar_stereographic",
        map_area_definition="corners",
        area=padded,
        map_vertical_longitude=center_lon,
        coastlines=_build_coastlines(),
    )


def _build_area_overlay(area_sub, point=None, nearest_info=None):
    """Return plot objects for an area box or point markers overlay.

    Parameters
    ----------
    area_sub : list
        [N, W, S, E] bounding box.
    point : list or None
        [lat, lon] when a single point was selected. Rendered as a red
        crosshair at the exact user-selected coordinates.
    nearest_info : dict or None
        Nearest-gridpoint / station info (with ``'latitude'`` and
        ``'longitude'``) used as the observation / extraction location.
        When provided and distinct from *point*, a second blue diagonal
        cross ("X") marker is drawn at its coordinates.

    Returns
    -------
    list
        Metview plot objects to unpack into ``mv.plot()``.
    """
    objs = []

    if point is None:
        # Area mode — draw the area box
        style = mv.mgraph(graph_line_colour="black", graph_line_thickness=4)
        n, w, s, e = area_sub
        lats = [s, n, n, s, s]
        lons = [w, w, e, e, w]
        objs.extend([mv.mvl_geopolyline(lats, lons, 0.1), style])
        return objs

    # --- Point mode ---
    # 1) User-selected point — red "+" crosshair
    lat_u, lon_u = point[0], point[1]
    size = 0.15  # degrees
    user_style = mv.mgraph(graph_line_colour="red", graph_line_thickness=5)
    objs.extend([
        mv.mvl_geopolyline([lat_u, lat_u], [lon_u - size, lon_u + size], 0.05),
        user_style,
        mv.mvl_geopolyline([lat_u - size, lat_u + size], [lon_u, lon_u], 0.05),
        user_style,
    ])

    # 2) Nearest station / gridpoint — blue "X" diagonal marker
    if (nearest_info is not None
            and 'latitude' in nearest_info
            and 'longitude' in nearest_info):
        lat_n = float(nearest_info['latitude'])
        lon_n = float(nearest_info['longitude'])
        if abs(lat_n - lat_u) > 1e-4 or abs(lon_n - lon_u) > 1e-4:
            station_style = mv.mgraph(graph_line_colour="blue", graph_line_thickness=4)
            d = size * 0.8
            objs.extend([
                mv.mvl_geopolyline(
                    [lat_n - d, lat_n + d], [lon_n - d, lon_n + d], 0.05),
                station_style,
                mv.mvl_geopolyline(
                    [lat_n - d, lat_n + d], [lon_n + d, lon_n - d], 0.05),
                station_style,
            ])

    return objs


def _build_gridpoint_markers():
    """Return an ``mv.mcont`` that renders grid-point location markers."""
    return mv.mcont(
        contour="off",
        contour_grid_value_plot="on",
        contour_grid_value_plot_type="marker",
        contour_grid_value_marker_height=0.2,
        contour_grid_value_marker_index=15,
    )


def _build_legend(param, units):
    """Return a styled legend for shaded fields."""
    if _get_var_category(param) == 'pressure':
        return None
    return mv.mlegend(
        legend_text_colour="black",
        legend_units_text=units,
        legend_text_font_size=0.31,
    )


def _format_mars_keywords(model_name):
    """Return a compact MARS keyword string for the title (values only, pipe-separated)."""
    settings = get_model_retrieval_settings(model_name)
    exclude = {'ensemble', 'number', 'grid'}
    preferred_order = ['class', 'type', 'stream', 'expver', 'model', 'database']

    parts = []
    seen = set()

    for key in preferred_order:
        if key in settings and settings[key] not in (None, ''):
            parts.append(str(settings[key]))
            seen.add(key)

    for key in sorted(settings):
        if key in seen or key in exclude:
            continue
        value = settings[key]
        if value in (None, ''):
            continue
        parts.append(str(value))

    return ' | '.join(parts)


def _format_settings_keywords(settings):
    """Return a compact keyword string from a raw settings dict (values only, pipe-separated)."""
    exclude = {'grid'}
    preferred_order = ['class', 'type', 'stream', 'expver']

    parts = []
    seen = set()

    for key in preferred_order:
        if key in settings and settings[key] not in (None, ''):
            parts.append(str(settings[key]))
            seen.add(key)

    for key in sorted(settings):
        if key in seen or key in exclude:
            continue
        value = settings[key]
        if value in (None, ''):
            continue
        parts.append(str(value))

    return '|'.join(parts)


def _build_title(param, model_name, fc_date, step, valid_date, member=None,
                 units='', level=None, acc_period=None, is_ensemble=False):
    """Return an ``mv.mtext`` title block."""
    var_desc = param
    try:
        vs = get_variable_settings(param)
        var_desc = vs.get('description', param)
    except (ValueError, KeyError):
        pass

    level_str = f" at {level} hPa" if level else ""
    acc_str = f" ({acc_period}h)" if acc_period else ""
    line1 = f"{var_desc}{level_str}{acc_str} ({units})"
    member_str = ""
    if member is not None:
        member_str = f" member {member}" if isinstance(member, int) else f" {member}"
    elif is_ensemble:
        member_str = " member 1"
    line2 = (f"{model_name}{member_str}  |  "
             f"Init: {fc_date.strftime('%Y-%m-%d %H:%M')} "
             f"T+{step}h  Valid: {valid_date.strftime('%Y-%m-%d %H:%M')}")
    line3 = _format_mars_keywords(model_name)
    return mv.mtext(
        text_line_count=3,
        text_line_1=line1,
        text_line_2=line2,
        text_line_3=line3,
        text_font_size=0.6,
    )


# ---------------------------------------------------------------------------
# Observation overlay
# ---------------------------------------------------------------------------

# Obs bins must match the model contour intervals so colours align.
_OBS_TEMP_MIN = list(range(-30, 48, 2))   # same as _TEMP_LEVELS[:-1]
_OBS_TEMP_MAX = list(range(-28, 50, 2))   # same as _TEMP_LEVELS[1:]

_OBS_PRECIP_MIN = [0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100, 200]
_OBS_PRECIP_MAX = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100, 200, 300]
_OBS_PRECIP_HEIGHTS = [0.5, 0.6, 0.6, 0.6, 0.6, 0.7, 0.7, 0.7, 0.7,
                       0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.8, 0.9, 1.0]

# Wind speed obs bins (one fewer than _WIND_LEVELS boundaries)
_OBS_WIND_MIN = [0, 2, 4, 6, 8, 10, 12, 15, 18, 21, 25, 30, 35, 40, 45, 50]
_OBS_WIND_MAX = [2, 4, 6, 8, 10, 12, 15, 18, 21, 25, 30, 35, 40, 45, 50, 80]


def _build_obs_marker(param, legend="off"):
    """Return an ``mv.msymb`` for observation overlay (filled circles with outline), or None."""
    cat = _get_var_category(param)
    if cat == 'temperature':
        return mv.msymb(
            symbol_type="marker",
            symbol_table_mode="advanced",
            legend=legend,
            symbol_advanced_table_selection_type="list",
            symbol_advanced_table_level_list=_TEMP_LEVELS,
            symbol_advanced_table_height_list=[0.4]*len(_TEMP_LEVELS),
            symbol_advanced_table_colour_method="list",
            symbol_advanced_table_colour_list=_TEMP_COLOURS,
            symbol_advanced_table_marker_list=15,
            symbol_outline="on",
            symbol_outline_colour="charcoal",
            symbol_outline_thickness=2.2,
        )
    if cat == 'precipitation':
        return mv.msymb(
            symbol_type="marker",
            symbol_table_mode="advanced",
            legend=legend,
            symbol_advanced_table_selection_type="list",
            symbol_advanced_table_level_list=_PRECIP_LEVELS,
            symbol_advanced_table_height_list=[0.4]*len(_PRECIP_LEVELS),
            symbol_advanced_table_colour_method="list",
            symbol_advanced_table_colour_list=_PRECIP_COLOURS,
            symbol_advanced_table_marker_list=15,
            symbol_outline="on",
            symbol_outline_colour="charcoal",
            symbol_outline_thickness=3.2,
        )
    if cat == 'wind':
        return mv.msymb(
            symbol_type="marker",
            symbol_table_mode="advanced",
            legend=legend,
            symbol_advanced_table_selection_type="list",
            symbol_advanced_table_level_list=_WIND_LEVELS,
            symbol_advanced_table_height_list=[0.4]*len(_WIND_LEVELS),
            symbol_advanced_table_colour_method="list",
            symbol_advanced_table_colour_list=_WIND_COLOURS,
            symbol_advanced_table_marker_list=15,
            symbol_outline="on",
            symbol_outline_colour="charcoal",
            symbol_outline_thickness=3.2,
        )
    return None


def _build_model_marker(param, legend="off"):
    """Return an ``mv.msymb`` for model geopoint overlay (filled squares with thick outline), or None."""
    cat = _get_var_category(param)
    if cat == 'temperature':
        return mv.msymb(
            symbol_type="marker",
            symbol_table_mode="advanced",
            legend=legend,
            symbol_advanced_table_selection_type="list",
            symbol_advanced_table_level_list=_TEMP_LEVELS,
            symbol_advanced_table_colour_method="list",
            symbol_advanced_table_colour_list=_TEMP_COLOURS,
            symbol_advanced_table_marker_list=18,
            symbol_advanced_table_height_list=[0.55]*len(_TEMP_LEVELS),
            symbol_outline="on",
            symbol_outline_colour="charcoal",
            symbol_outline_thickness=3.2,
        )
    if cat == 'precipitation':
        return mv.msymb(
            symbol_type="marker",
            symbol_table_mode="advanced",
            legend=legend,
            symbol_advanced_table_selection_type="list",
            symbol_advanced_table_level_list=_PRECIP_LEVELS,
            symbol_advanced_table_colour_method="list",
            symbol_advanced_table_colour_list=_PRECIP_COLOURS,
            symbol_advanced_table_marker_list=18,
            symbol_advanced_table_height_list=[0.55]*len(_PRECIP_LEVELS),
            symbol_outline="on",
            symbol_outline_colour="charcoal",
            symbol_outline_thickness=3.2,
        )
    if cat == 'wind':
        return mv.msymb(
            symbol_type="marker",
            symbol_table_mode="advanced",
            legend=legend,
            symbol_advanced_table_selection_type="list",
            symbol_advanced_table_level_list=_WIND_LEVELS,
            symbol_advanced_table_colour_method="list",
            symbol_advanced_table_colour_list=_WIND_COLOURS,
            symbol_advanced_table_marker_list=18,
            symbol_advanced_table_height_list=[0.55]*len(_WIND_LEVELS),
            symbol_outline="on",
            symbol_outline_colour="charcoal",
            symbol_outline_thickness=3.2,
        )
    return None


# ---------------------------------------------------------------------------
# Internal helpers for directory / file resolution
# ---------------------------------------------------------------------------

def _resolve_data_dir(plot_data, widgets_dict):
    """Resolve the data directory name for the current config.

    Returns (dir_name, base_path, area_sub) — *area_sub* may be updated if
    a point was matched against an existing directory.
    """
    from .core import get_area_string

    config = widgets_dict['config']
    param = config['param']
    valid_date = config['valid_date']
    area_sub = list(config['area_sub'])
    base_path = plot_data.get('base_path', '')
    date_str = valid_date.strftime("%Y%m%d")
    time_str = f"{valid_date.hour:02d}00"
    area_str = get_area_string(area_sub)

    dir_name_new = f"{param}_{area_str}_{date_str}_{time_str}"
    dir_name_old = f"{param}_{area_str}_{date_str}"
    if os.path.isdir(os.path.join(base_path, dir_name_new)):
        dir_name = dir_name_new
    elif os.path.isdir(os.path.join(base_path, dir_name_old)):
        dir_name = dir_name_old
    else:
        dir_name = dir_name_new

    return dir_name, base_path, area_sub


def _load_obs_geopoints(base_path, dir_name, param, date_str, time_str):
    """Load observation geopoints from disk, returning (geopoints, path) or (None, None)."""
    obs_dir = os.path.join(base_path, dir_name, "obs_files")
    obs_file = os.path.join(obs_dir, f"STVL_{param}_{date_str}_{time_str}.grib")
    if os.path.exists(obs_file):
        return mv.read(obs_file), obs_file
    return None, None


# ---------------------------------------------------------------------------
# Public API — observation-only map
# ---------------------------------------------------------------------------

def plot_obs_map(plot_data, widgets_dict, plot_radius=0, export_png=True,
                 add_markers=False):
    """Plot observation geopoints on a map (no model field).

    Parameters
    ----------
    plot_data : dict
        The dict returned by ``retrieve_and_store_data``.
    widgets_dict : dict
        The widgets dictionary with ``config`` sub-dict.
    plot_radius : float
        Degrees to pad around the data area for the map view.
    export_png : bool
        If True, save a PNG to the plot directory.
    """
    config = widgets_dict['config']
    param = config['param']
    valid_date = config['valid_date']
    date_str = valid_date.strftime("%Y%m%d")
    time_str = f"{valid_date.hour:02d}00"

    dir_name, base_path, area_sub = _resolve_data_dir(plot_data, widgets_dict)

    obs_data, obs_file = _load_obs_geopoints(base_path, dir_name, param, date_str, time_str)
    if obs_data is None:
        print(f"No observation file found for {param} at {date_str}_{time_str}")
        return

    # Unit conversion for temperature obs (STVL stores in K)
    base = get_base_var(param)
    if base in ('2t', '2d', 't'):
        from .variables import convert_to_display
        obs_data = convert_to_display(obs_data, param, grib_units='K')

    obs_marker = _build_obs_marker(param, legend="on")
    if obs_marker is None:
        # Fallback generic marker
        obs_marker = mv.msymb(
            symbol_type="marker", symbol_table_mode="off",
            legend="on",
            symbol_marker_index=15,
            symbol_colour="black",
            symbol_height=0.4,
            symbol_outline="on",
            symbol_outline_colour="black",
            symbol_outline_thickness=1,
        )

    units_str = ''
    try:
        vs = get_variable_settings(param)
        units_str = vs.get('units', '')
    except (ValueError, KeyError):
        pass

    geoview = _build_geoview(area_sub, plot_radius)
    legend = _build_legend(param, units_str)
    legend_objects = [] if legend is None else [legend]
    nginfo = plot_data.get('nearest_gridinfo_dict', {})
    area_overlay = _build_area_overlay(
        area_sub, point=config.get('point'),
        nearest_info=nginfo.get('nearest_station')) if add_markers else []

    title = mv.mtext(
        text_line_count=2,
        text_line_1=f"Observations — {param} ({units_str})",
        text_line_2=f"Valid: {valid_date.strftime('%Y-%m-%d %H:%M')}",
        text_font_size=0.6,
    )

    if export_png:
        from .core import _safe_label
        plot_dir = os.path.join(base_path, dir_name, "plot_files")
        os.makedirs(plot_dir, exist_ok=True)
        png_name = os.path.join(plot_dir, f"map_obs_{param}_{date_str}_{time_str}")
        mv.setoutput(mv.png_output(output_name=png_name, output_name_first_page_number='off', output_font_scale=1.6, output_width=2200))
        mv.plot(obs_data, obs_marker, geoview, *legend_objects, title, *area_overlay)
        print(f"Obs map exported to: {png_name}.png")
    else:
        mv.plot(obs_data, obs_marker, geoview, *legend_objects, title, *area_overlay)


# ---------------------------------------------------------------------------
# Public API — analysis map
# ---------------------------------------------------------------------------

def plot_analysis_map(plot_data, widgets_dict, plot_radius=0, export_png=True,
                      overlay_obs=False, show_gridpoints=False,
                      add_markers=False):
    """Plot the analysis field on a map.

    Parameters
    ----------
    plot_data : dict
        The dict returned by ``retrieve_and_store_data``.
    widgets_dict : dict
        The widgets dictionary with ``config`` sub-dict.
    plot_radius : float
        Degrees to pad around the data area for the map view.
    export_png : bool
        If True, save a PNG to the plot directory.
    overlay_obs : bool
        If True, overlay STVL observations on the map (when available).
    show_gridpoints : bool
        If True, overlay grid-point markers on the field.
    """
    from .core import _reference_grib_filename, _get_var_settings_safe

    config = widgets_dict['config']
    param = config['param']
    valid_date = config['valid_date']
    levtype = config['levtype']
    level = config.get('level')
    date_str = valid_date.strftime("%Y%m%d")
    time_str = f"{valid_date.hour:02d}00"

    dir_name, base_path, area_sub = _resolve_data_dir(plot_data, widgets_dict)
    grib_dir = os.path.join(base_path, dir_name, "grib_files")

    # Locate analysis GRIB file
    analysis_settings = get_analysis_settings()
    analysis_file = os.path.join(grib_dir, _reference_grib_filename(
        'Analysis', param, analysis_settings, levtype, level, date_str, time_str))

    if not os.path.exists(analysis_file):
        raise FileNotFoundError(
            f"Analysis GRIB file not found: {analysis_file}\n"
            "Run retrieve_and_store_data first."
        )

    print(f"Reading {analysis_file}")
    data = mv.read(analysis_file)

    # Unit conversion
    units_str = ''
    try:
        vs = get_variable_settings(param)
        units_str = vs.get('units', '')
        grib_units = get_grib_units(data)
        if grib_units:
            from .variables import convert_to_display
            data = convert_to_display(data, param, grib_units)
    except Exception as e:
        print(f"Warning: unit conversion skipped — {e}")

    contour = _build_contour(param, level)
    geoview = _build_geoview(area_sub, plot_radius)
    legend = _build_legend(param, units_str)
    legend_objects = [] if legend is None else [legend]
    gp_objects = [data, _build_gridpoint_markers()] if show_gridpoints else []
    nginfo = plot_data.get('nearest_gridinfo_dict', {})
    area_overlay = _build_area_overlay(
        area_sub, point=config.get('point'),
        nearest_info=nginfo.get('nearest_analysis')) if add_markers else []

    level_str = f" at {level} hPa" if level else ""
    mars_kw = _format_settings_keywords(analysis_settings)
    title = mv.mtext(
        text_line_count=2,
        text_line_1=f"Analysis — {param}{level_str} ({units_str})",
        text_line_2=f"Valid: {valid_date.strftime('%Y-%m-%d %H:%M')}  |  {mars_kw}",
        text_font_size=0.6,
    )

    # Observation overlay
    obs_objects = []
    if overlay_obs:
        obs_data, obs_file = _load_obs_geopoints(base_path, dir_name, param, date_str, time_str)
        if obs_data is not None:
            base = get_base_var(param)
            if base in ('2t', '2d', 't'):
                from .variables import convert_to_display
                obs_data = convert_to_display(obs_data, param, grib_units='K')
            obs_marker = _build_obs_marker(param)
            if obs_marker is not None:
                obs_objects = [obs_data, obs_marker]
                print(f"Overlaying observations from {obs_file}")
        else:
            print("Observation file not found — skipping overlay")

    if export_png:
        from .core import _safe_label
        plot_dir = os.path.join(base_path, dir_name, "plot_files")
        os.makedirs(plot_dir, exist_ok=True)
        mars_parts = [
            str(analysis_settings.get('class', '')),
            str(analysis_settings.get('stream', '')),
            str(analysis_settings.get('type', '')),
        ]
        mars_tag = '_'.join(p for p in mars_parts if p)
        name_parts = ["map_Analysis", param, mars_tag, levtype]
        if level is not None:
            name_parts.append(f"L{level}hPa")
        name_parts.append(f"{date_str}_{time_str}")
        png_name = os.path.join(plot_dir, '_'.join(str(p) for p in name_parts))
        mv.setoutput(mv.png_output(output_name=png_name, output_name_first_page_number='off', output_font_scale=1.6, output_width=2200))
        mv.plot(data, contour, geoview, *legend_objects, title, *obs_objects, *area_overlay, *gp_objects)
        print(f"Analysis map exported to: {png_name}.png")
    else:
        mv.plot(data, contour, geoview, *legend_objects, title, *obs_objects, *area_overlay, *gp_objects)


# ---------------------------------------------------------------------------
# Public API — model field map
# ---------------------------------------------------------------------------

def plot_field_map(plot_data, widgets_dict, model_name, step,
                   member=None, plot_radius=0, export_png=True,
                   overlay_obs=False, plot_mode='field',
                   show_gridpoints=False, add_markers=False):
    """Plot a GRIB field on a map using Metview.

    Parameters
    ----------
    plot_data : dict
        The dict returned by ``retrieve_and_store_data``.
    widgets_dict : dict
        The widgets dictionary with ``config`` sub-dict.
    model_name : str
        Model to plot (e.g. ``'IFS Control'``, ``'AIFS ENS'``).
    step : int
        Forecast step in hours.
    member : int | ``'mean'`` | None
        For ensemble models: member number (1-based) or ``'mean'``.
        Ignored for deterministic models.
    plot_radius : float
        Degrees to pad around the data area for the map view.
    export_png : bool
        If True, save a PNG to the plot directory.
    overlay_obs : bool
        If True, overlay STVL observations on the map (when available).
    plot_mode : str
        ``'field'`` (default) — plot the full gridded field with contour shading.
        ``'station_nearest'`` — extract model values at the nearest grid points
        to observation stations and plot as coloured markers (requires obs data).
    show_gridpoints : bool
        If True, overlay grid-point markers on the field.

    Returns
    -------
    None
        Displays the plot inline (Jupyter) and optionally saves a PNG.
    """
    config = widgets_dict['config']
    param = config['param']
    valid_date = config['valid_date']
    area_sub = config['area_sub']
    levtype = config['levtype']
    level = config.get('level')

    # Determine forecast init date from valid_date and step
    fc_date = valid_date - timedelta(hours=step)

    # Locate the grib file
    from .core import (_model_grib_filename,
                       _safe_label, _get_var_settings_safe)

    date_str = valid_date.strftime("%Y%m%d")
    time_str = f"{valid_date.hour:02d}00"

    dir_name, base_path, area_sub = _resolve_data_dir(plot_data, widgets_dict)
    grib_dir = os.path.join(base_path, dir_name, "grib_files")
    plot_dir = os.path.join(base_path, dir_name, "plot_files")
    os.makedirs(plot_dir, exist_ok=True)

    # Get variable settings
    try:
        var_settings = get_variable_settings(param)
        is_accumulated = var_settings.get('is_accumulated', False)
    except (ValueError, KeyError):
        var_settings = _get_var_settings_safe(param)
        is_accumulated = False

    acc_period = None
    if is_accumulated:
        acc_widget = widgets_dict.get('acc_period_widget')
        acc_period = acc_widget.value if acc_widget is not None else 24

    # Build model filename — try new naming, fall back to old
    new_name = _model_grib_filename(model_name, param, levtype, level, fc_date,
                                     step, acc_period=acc_period)
    old_name = f"{model_name}_{param}_{fc_date.strftime('%Y%m%d')}_{fc_date.hour:02d}00_step{step}.grib"
    model_file = os.path.join(grib_dir, new_name)
    if not os.path.exists(model_file):
        model_file = os.path.join(grib_dir, old_name)
    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"GRIB file not found: {os.path.join(grib_dir, new_name)}\n"
            f"Run retrieve_and_store_data first, or check that this model/step combination was retrieved."
        )

    print(f"Reading {model_file}")
    data = mv.read(model_file)

    # --- Select the right field / member ---
    model_settings = get_model_retrieval_settings(model_name)
    is_ensemble = model_settings.get('ensemble', False)

    # Handle accumulated variables (deaccumulate)
    if is_accumulated:
        if is_ensemble:
            if member == 'mean':
                # Deaccumulate all members, then average
                n_fields = len(data)
                n_members = n_fields // 2
                deacc = []
                for i in range(n_members):
                    m_data = data.select(shortName=get_base_var(param), number=i + 1)
                    if len(m_data) >= 2:
                        deacc.append(m_data[1] - m_data[0])
                if deacc:
                    data = mv.mean(mv.merge(*deacc))
                else:
                    raise ValueError("No valid ensemble members found for mean computation")
            elif isinstance(member, int):
                m_data = data.select(shortName=get_base_var(param), number=member)
                if len(m_data) >= 2:
                    data = m_data[1] - m_data[0]
                else:
                    raise ValueError(f"Member {member}: expected 2 fields for deaccumulation, found {len(m_data)}")
            else:
                # No member specified for ensemble — default to member 1
                m_data = data.select(shortName=get_base_var(param), number=1)
                if len(m_data) >= 2:
                    data = m_data[1] - m_data[0]
                elif len(data) >= 2:
                    data = data[1] - data[0]
        else:
            if len(data) >= 2:
                data = data[1] - data[0]
            else:
                data = data[0]
    else:
        # Non-accumulated
        if is_ensemble:
            if member == 'mean':
                selected = data.select(shortName=get_base_var(param))
                data = mv.mean(selected) if selected else data
            elif isinstance(member, int):
                selected = data.select(shortName=get_base_var(param), number=member)
                if selected:
                    data = selected
                else:
                    raise ValueError(f"Member {member} not found in data")
            else:
                # No member specified — default to member 1
                selected = data.select(shortName=get_base_var(param), number=1)
                data = selected if selected else data[0]
        else:
            # Deterministic — select by shortName if multiple fields
            selected = data.select(shortName=get_base_var(param))
            if selected:
                data = selected

    # --- Unit conversion ---
    units_str = ''
    try:
        vs = get_variable_settings(param)
        units_str = vs.get('units', '')
        grib_units = get_grib_units(data)
        if grib_units:
            from .variables import convert_to_display
            data = convert_to_display(data, param, grib_units)
    except Exception as e:
        print(f"Warning: unit conversion skipped — {e}")

    # --- Build plot objects ---
    contour = _build_contour(param, level)
    geoview = _build_geoview(area_sub, plot_radius)
    legend = _build_legend(param, units_str)
    legend_objects = [] if legend is None else [legend]
    gp_objects = [data, _build_gridpoint_markers()] if show_gridpoints else []
    nginfo = plot_data.get('nearest_gridinfo_dict', {})
    area_overlay = _build_area_overlay(
        area_sub, point=config.get('point'),
        nearest_info=nginfo.get(model_name + '_nearest')) if add_markers else []
    title = _build_title(param, model_name, fc_date, step, valid_date,
                         member=member, units=units_str, level=level,
                         acc_period=acc_period, is_ensemble=is_ensemble)

    # --- Station-nearest mode: replace gridded field with geopoints ---
    if plot_mode == 'station_nearest':
        obs_gpt, obs_file = _load_obs_geopoints(base_path, dir_name, param, date_str, time_str)
        if obs_gpt is None:
            print(f"No observation file found — cannot use station_nearest mode. "
                  f"Falling back to full field.")
            plot_mode = 'field'
        else:
            model_gpt = mv.nearest_gridpoint(data, obs_gpt)
            model_marker = _build_model_marker(param, legend="on")
            if model_marker is None:
                model_marker = mv.msymb(
                    symbol_type="marker", symbol_table_mode="off",
                    legend="on",
                    symbol_marker_index=18,
                    symbol_colour="navy",
                    symbol_height=0.5,
                    symbol_outline="on",
                    symbol_outline_colour="charcoal",
                    symbol_outline_thickness=2,
                )
            n_pts = len(obs_gpt)
            print(f"Station-nearest mode: model values extracted at {n_pts} station locations")
            # In station_nearest mode, also overlay obs if requested
            obs_objects = []
            if overlay_obs:
                base = get_base_var(param)
                obs_display = obs_gpt
                if base in ('2t', '2d', 't'):
                    from .variables import convert_to_display
                    obs_display = convert_to_display(obs_gpt, param, grib_units='K')
                obs_marker_style = _build_obs_marker(param)
                if obs_marker_style is not None:
                    obs_objects = [obs_display, obs_marker_style]

    # --- Observation overlay (field mode only) ---
    if plot_mode == 'field':
        obs_objects = []
        if overlay_obs:
            obs_data, obs_file = _load_obs_geopoints(base_path, dir_name, param, date_str, time_str)
            if obs_data is not None:
                base = get_base_var(param)
                if base in ('2t', '2d', 't'):
                    from .variables import convert_to_display
                    obs_data = convert_to_display(obs_data, param, grib_units='K')
                obs_marker = _build_obs_marker(param)
                if obs_marker is not None:
                    obs_objects = [obs_data, obs_marker]
                    print(f"Overlaying observations from {obs_file}")
                else:
                    print(f"No observation marker style defined for {param} — skipping overlay")
            else:
                print(f"Observation file not found — skipping overlay "
                      f"(not all parameters are available in STVL)")

    # --- Render ---
    if export_png:
        member_tag = ""
        if isinstance(member, int):
            member_tag = f"_mem{member}"
        elif member == 'mean':
            member_tag = "_mean"
        safe_model = _safe_label(model_name)
        from .core import _mars_file_tag
        mars_tag = _mars_file_tag(model_name)
        mode_tag = "_stn" if plot_mode == 'station_nearest' else ""
        name_parts = [f"map_{safe_model}", param, mars_tag, levtype]
        if level is not None:
            name_parts.append(f"L{level}hPa")
        if acc_period is not None:
            name_parts.append(f"acc{acc_period}h")
        name_parts.extend([
            fc_date.strftime('%Y%m%d_%H%M'),
            f"step{step}{member_tag}{mode_tag}",
        ])
        png_name = os.path.join(plot_dir, '_'.join(str(p) for p in name_parts))
        mv.setoutput(mv.png_output(output_name=png_name, output_name_first_page_number='off',output_font_scale=2.5, output_width=2200))

    if plot_mode == 'station_nearest':
        if export_png:
            mv.plot(model_gpt, model_marker, geoview, *legend_objects, title, *obs_objects, *area_overlay)
            print(f"Map exported to: {png_name}.png")
        else:
            mv.plot(model_gpt, model_marker, geoview, *legend_objects, title, *obs_objects, *area_overlay)
    else:
        if export_png:
            mv.plot(data, contour, geoview, *legend_objects, title, *obs_objects, *area_overlay, *gp_objects)
            print(f"Map exported to: {png_name}.png")
        else:
            mv.plot(data, contour, geoview, *legend_objects, title, *obs_objects, *area_overlay, *gp_objects)
