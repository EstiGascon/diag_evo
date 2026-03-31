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
from .settings import get_model_retrieval_settings

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

_TEMP_LEVELS = list(range(-50, 60, 2))
_TEMP_COLOURS = [
    "rgb(76,76,76)", "rgb(128,128,128)", "rgb(153,153,153)", "rgb(179,179,179)", "rgb(204,204,204)",
    "rgb(204,158,134)", "rgb(192,137,107)", "rgb(182,117,82)", "rgb(151,95,64)", "rgb(124,78,52)",
    "rgb(89,0,153)", "rgb(128,0,230)", "rgb(153,51,255)", "rgb(192,102,255)", "rgb(217,153,255)",
    "rgb(255,192,255)", "rgb(255,151,255)", "rgb(225,51,225)", "rgb(174,51,174)", "rgb(122,51,122)",
    "rgb(0,0,192)", "rgb(0,0,255)", "rgb(51,102,255)", "rgb(102,179,255)", "rgb(153,230,255)",
    "rgb(0,140,48)", "rgb(38,192,25)", "rgb(128,217,0)", "rgb(166,243,0)", "rgb(204,255,51)",
    "rgb(255,255,153)", "rgb(255,255,0)", "rgb(255,217,0)", "rgb(255,189,0)",
    "rgb(255,153,0)", "rgb(255,128,0)", "rgb(255,96,0)", "rgb(255,0,0)", "rgb(204,0,0)",
    "rgb(204,61,110)", "rgb(255,0,255)", "rgb(255,151,255)", "rgb(215,121,255)",
    "rgb(174,0,249)", "rgb(125,0,179)", "rgb(151,95,64)", "rgb(182,117,82)",
    "rgb(192,137,107)", "rgb(204,158,134)", "rgb(204,204,204)", "rgb(179,179,179)",
    "rgb(153,153,153)", "rgb(128,128,128)", "rgb(76,76,76)",
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
    if base in ('10u', '10v', '10si', '10fg'):
        return 'wind'
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
            contour_line_colour="black", contour_line_thickness=1,
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
            legend="on", contour="on",
            contour_line_colour="black", contour_line_thickness=1,
            contour_level_selection_type="level_list",
            contour_level_list=_PRESSURE_LEVELS,
            contour_label="on", contour_label_height=0.4,
            contour_shade="on",
            contour_shade_colour_method="list",
            contour_shade_technique="grid_shading",
            contour_shade_colour_list=_PRESSURE_COLOURS,
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
    return mv.geoview(
        map_projection="cylindrical",
        map_area_definition="corners",
        area=padded,
        coastlines=_build_coastlines(),
    )


def _build_legend(units):
    """Return a styled ``mv.mlegend``."""
    return mv.mlegend(
        legend_text_colour="black",
        legend_automatic_position="right",
        legend_units_text=units,
        legend_text_font_style="bold",
        legend_text_font_size=0.35,
        legend_entry_text_width=50,
    )


def _build_title(param, model_name, fc_date, step, valid_date, member=None,
                 units='', level=None):
    """Return an ``mv.mtext`` title block."""
    var_desc = param
    try:
        vs = get_variable_settings(param)
        var_desc = vs.get('description', param)
    except (ValueError, KeyError):
        pass

    level_str = f" at {level} hPa" if level else ""
    line1 = f"{var_desc}{level_str} ({units})"
    line2 = (f"Init: {fc_date.strftime('%Y-%m-%d %H:%M')} "
             f"T+{step}h  Valid: {valid_date.strftime('%Y-%m-%d %H:%M')}")
    member_str = ""
    if member is not None:
        member_str = f" member {member}" if isinstance(member, int) else f" {member}"
    line3 = f"{model_name}{member_str}"
    return mv.mtext(
        text_line_count=4,
        text_line_1=line1,
        text_line_2=line2,
        text_line_3=line3,
        text_line_4=" ",
        text_font_size=0.6,
    )


# ---------------------------------------------------------------------------
# Observation overlay
# ---------------------------------------------------------------------------

_OBS_TEMP_MIN = list(range(-50, 58, 2))
_OBS_TEMP_MAX = list(range(-48, 60, 2))

_OBS_PRECIP_MIN = [0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100, 200]
_OBS_PRECIP_MAX = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100, 200, 300]
_OBS_PRECIP_HEIGHTS = [0.5, 0.6, 0.6, 0.6, 0.6, 0.7, 0.7, 0.7, 0.7,
                       0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.8, 0.9, 1.0]


def _build_obs_marker(param):
    """Return an ``mv.psymb`` for observation overlay, or None."""
    cat = _get_var_category(param)
    if cat == 'temperature':
        return mv.psymb(
            symbol_type="marker", symbol_table_mode="on",
            legend="off", symbol_quality="high",
            symbol_min_table=_OBS_TEMP_MIN,
            symbol_max_table=_OBS_TEMP_MAX,
            symbol_marker_table=[15],
            symbol_colour_table=_TEMP_COLOURS,
            symbol_height_table=[0.4] * len(_OBS_TEMP_MIN),
        )
    if cat == 'precipitation':
        return mv.psymb(
            symbol_type="marker", symbol_table_mode="on",
            legend="off", symbol_quality="high",
            symbol_min_table=_OBS_PRECIP_MIN,
            symbol_max_table=_OBS_PRECIP_MAX,
            symbol_marker_table=[15],
            symbol_colour_table=_PRECIP_COLOURS,
            symbol_height_table=_OBS_PRECIP_HEIGHTS,
        )
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_field_map(plot_data, widgets_dict, model_name, step,
                   member=None, plot_radius=0, export_png=True,
                   overlay_obs=False):
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
        If True, overlay STVL observations on the map.

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
    from .core import setup_data_directories, get_area_string, _find_existing_directory_for_point
    base_path = plot_data.get('base_path', '')
    area_str = get_area_string(area_sub)
    date_str = valid_date.strftime("%Y%m%d")

    # Try to find existing directory (same logic as retrieve_and_store_data)
    point = config.get('point')
    if point is not None:
        reuse_area_str, reuse_area = _find_existing_directory_for_point(
            base_path, param, date_str, point,
        )
        if reuse_area_str is not None:
            area_str = reuse_area_str
            area_sub = reuse_area

    # Build grib_dir path
    grib_dir = os.path.join(base_path, f"{param}_{area_str}_{date_str}", "grib_files")
    plot_dir = os.path.join(base_path, f"{param}_{area_str}_{date_str}", "plot_files")
    os.makedirs(plot_dir, exist_ok=True)

    model_file = os.path.join(
        grib_dir,
        f"{model_name}_{param}_{fc_date.strftime('%Y%m%d')}_{fc_date.hour:02d}00_step{step}.grib"
    )
    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"GRIB file not found: {model_file}\n"
            f"Run retrieve_and_store_data first, or check that this model/step combination was retrieved."
        )

    print(f"Reading {model_file}")
    data = mv.read(model_file)

    # --- Select the right field / member ---
    model_settings = get_model_retrieval_settings(model_name)
    is_ensemble = model_settings.get('ensemble', False)

    # Get variable settings for unit conversion
    try:
        var_settings = get_variable_settings(param)
        is_accumulated = var_settings.get('is_accumulated', False)
    except (ValueError, KeyError):
        is_accumulated = False

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
                # No member specified for ensemble — plot control/first available
                if len(data) >= 2:
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
                # No member specified — use first field
                data = data[0]
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
    legend = _build_legend(units_str)
    title = _build_title(param, model_name, fc_date, step, valid_date,
                         member=member, units=units_str, level=level)

    # --- Observation overlay ---
    obs_objects = []
    if overlay_obs:
        obs_dir = os.path.join(base_path, f"{param}_{area_str}_{date_str}", "obs_files")
        obs_file = os.path.join(obs_dir, f"STVL_{param}_{date_str}_{valid_date.hour:02d}00.grib")
        if os.path.exists(obs_file):
            obs_data = mv.read(obs_file)
            obs_marker = _build_obs_marker(param)
            if obs_marker is not None:
                obs_objects = [obs_data, obs_marker]
                print(f"Overlaying observations from {obs_file}")
            else:
                print(f"No observation marker style defined for {param} — skipping overlay")
        else:
            print(f"Observation file not found: {obs_file} — skipping overlay")

    # --- Render ---
    if export_png:
        member_tag = ""
        if isinstance(member, int):
            member_tag = f"_mem{member}"
        elif member == 'mean':
            member_tag = "_mean"
        safe_model = model_name.replace(' ', '_')
        png_name = os.path.join(
            plot_dir,
            f"map_{safe_model}_{param}_{fc_date.strftime('%Y%m%d_%H%M')}_step{step}{member_tag}"
        )
        mv.setoutput(mv.png_output(output_name=png_name))
        mv.plot(data, contour, geoview, legend, title, *obs_objects)
        print(f"Map exported to: {png_name}.png")
    else:
        mv.plot(data, contour, geoview, legend, title, *obs_objects)
