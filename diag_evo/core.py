"""
Forecast Evolution Analysis - core module.

Provides the interactive UI (ipywidgets + ipyleaflet), MARS retrieval,
and both interactive (Plotly) and static (Matplotlib) plotting.
"""

# Imports

from datetime import datetime, timedelta
import plotly.graph_objs as go
import numpy as np
import ipywidgets as widgets
import pandas as pd
from IPython.display import display
from ipyleaflet import Map, DrawControl, basemaps, basemap_to_tiles
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import metview as mv
import earthkit.data
import os
import traceback
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Default timeout (seconds) for a single MARS retrieval.
# Set to 0 or None to disable.
MARS_RETRIEVAL_TIMEOUT = 120


def _retrieve_with_timeout(timeout=MARS_RETRIEVAL_TIMEOUT, *args, **kwargs):
    """Call *earthkit.data.from_source* with an optional timeout.

    Parameters
    ----------
    timeout : int | None
        Maximum seconds to wait.  ``0`` or ``None`` disables the timeout.
    *args, **kwargs
        Forwarded to ``earthkit.data.from_source``.

    Raises
    ------
    TimeoutError
        If the retrieval exceeds *timeout* seconds.
    """
    if not timeout:
        return earthkit.data.from_source(*args, **kwargs)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(earthkit.data.from_source, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            raise TimeoutError(
                f"MARS retrieval timed out after {timeout} seconds. "
                "The data source may be temporarily unavailable."
            )


from .variables import (
    load_variable_settings, get_base_var, get_level, get_variable_settings,
    convert_to_display, convert_from_display, get_grib_units,
    get_retrieval_settings, process_accumulated_data, get_variable_display_name
)
from .settings import (
    get_model_settings,
    get_plot_settings,
    get_available_models,
    get_predefined_models,
    get_custom_models,
    get_model_retrieval_settings,
    get_analysis_settings,
    get_climatology_settings,
    get_model_plot_settings,
    get_reference_plot_settings,
    get_layout_settings,
    register_custom_model,
    unregister_custom_model,
    clear_custom_models,
)

# Global variables
point = None  # Store point coordinates when a point is selected
valid_date = None
param = None
area_sub = None
forecast_dates = []
forecast_steps = []
max_days = None
step_interval = None
data_dir = None  # Will store the base data directory path
#base_path = "./data_files"
# Users can override this in their Jupyter notebook by setting: base_path = "/your/custom/path/"

def sanitize_mars_request(request):
    """Enforce MARS keyword interdependencies.

    Rules applied:
    - ``number`` is only valid when ``type`` is ``pf``.
    - ``levelist`` is only valid when ``levtype`` is ``pl``.
    - ``dataset`` is only valid for DestinE-class requests (class d1).
    - ``address`` is only valid for DestinE-class requests.
    - Remove internal bookkeeping keys that must never reach MARS.
    """
    req = dict(request)

    # --- number only with type=pf ---
    req_type = str(req.get('type', '')).lower()
    if req_type != 'pf' and 'number' in req:
        del req['number']

    # --- levelist only with levtype=pl ---
    req_levtype = str(req.get('levtype', '')).lower()
    if req_levtype != 'pl' and 'levelist' in req:
        del req['levelist']

    # --- dataset only for DestinE (class d1) ---
    req_class = str(req.get('class', '')).lower()
    if req_class != 'd1':
        req.pop('dataset', None)
        req.pop('address', None)

    # --- Remove internal keys ---
    req.pop('ensemble', None)

    return req


import re

_AREA_DIR_RE = re.compile(
    r'^(?P<var>.+)_N(?P<north>[^_]+)_W(?P<west>[^_]+)_S(?P<south>[^_]+)_E(?P<east>[^_]+)_(?P<date>\d{8})$'
)


def _parse_area_from_dirname(dirname):
    """Parse [N, W, S, E] from a data directory name.

    Returns (var, [N, W, S, E], date_str) or None if the name doesn't match.
    """
    m = _AREA_DIR_RE.match(dirname)
    if not m:
        return None
    try:
        area = [float(m.group('north')), float(m.group('west')),
                float(m.group('south')), float(m.group('east'))]
        return m.group('var'), area, m.group('date')
    except ValueError:
        return None


def _find_existing_directory_for_point(base_path, var, date_str, point):
    """Look for an existing data directory whose area contains *point*.

    Scans ``base_path`` for directories matching ``{var}_*_{date_str}`` and
    checks whether *point* ``[lat, lon]`` falls inside the stored area.

    Returns ``(area_string, [N, W, S, E])`` if a match is found, or
    ``(None, None)`` otherwise.
    """
    if point is None or not os.path.isdir(base_path):
        return None, None

    lat, lon = point
    for entry in os.listdir(base_path):
        parsed = _parse_area_from_dirname(entry)
        if parsed is None:
            continue
        d_var, d_area, d_date = parsed
        if d_var != var or d_date != date_str:
            continue
        n, w, s, e = d_area
        if s <= lat <= n and w <= lon <= e:
            # Verify the directory actually contains grib files
            grib_dir = os.path.join(base_path, entry, "grib_files")
            if os.path.isdir(grib_dir) and os.listdir(grib_dir):
                print(f"Reusing existing data directory (point {lat},{lon} is inside "
                      f"area N{n}/W{w}/S{s}/E{e}): {entry}")
                return get_area_string(d_area), d_area
    return None, None


def setup_data_directories(base_path, var, area_str, date_str):
    """Create and return paths for data directories"""
    # Create main data directory structure
    grib_dir = os.path.join(base_path, f"{var}_{area_str}_{date_str}", "grib_files")
    obs_dir = os.path.join(base_path, f"{var}_{area_str}_{date_str}", "obs_files")
    plot_dir = os.path.join(base_path, f"{var}_{area_str}_{date_str}", "plot_files")
    
    # Create directories if they don't exist
    os.makedirs(grib_dir, exist_ok=True)
    os.makedirs(obs_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    return grib_dir, obs_dir, plot_dir

def get_area_string(area):
    """Convert area coordinates to string for directory naming"""
    return f"N{area[0]}_W{area[1]}_S{area[2]}_E{area[3]}"

def _get_var_settings_safe(param):
    """Try to look up variable settings from JSON; return a minimal default if not found."""
    try:
        return get_variable_settings(param)
    except (ValueError, KeyError):
        pass
    # Fallback: try interpreting as base_var
    try:
        base = get_base_var(param)
        return get_variable_settings(base)
    except (ValueError, KeyError):
        pass
    # Unknown param — return a safe default (no conversion, not accumulated)
    return {
        'levtype': 'sfc',
        'description': param,
        'is_accumulated': False,
        'units': '',
        'grib_units': [],
        'conversion_rules': {},
    }


def _safe_convert_to_display(data, param, grib_units=None):
    """Convert to display units if param is known; otherwise return data unchanged."""
    try:
        return convert_to_display(data, param, grib_units)
    except (ValueError, KeyError):
        return data


def _safe_display_name(param):
    """Return a human-readable display name for param, falling back to the raw string."""
    try:
        return get_variable_display_name(param)
    except (ValueError, KeyError):
        return str(param)


def _model_display_label(model_name):
    """Return 'ModelName (type_stream_class_expver)' for legend labels."""
    try:
        ms = get_model_retrieval_settings(model_name)
        parts = [
            str(ms.get('type', '')),
            str(ms.get('stream', '')),
            str(ms.get('class', '')),
            str(ms.get('expver', '')),
        ]
        suffix = '_'.join(p for p in parts if p)
        if suffix:
            return f"{model_name} ({suffix})"
    except (ValueError, KeyError):
        pass
    return model_name


def _build_ylabel(param, widgets_dict):
    """Build a y-axis label including display name, units, and accumulation period."""
    # If the user entered a bare variable (e.g. 'z') with a separate level widget,
    # build a combined name so the label reads e.g. 'z500' instead of just 'z'.
    effective_param = param
    level_from_name = get_level(param)
    if level_from_name is None:
        levtype = widgets_dict.get('levtype_widget')
        level_w = widgets_dict.get('level_widget')
        if levtype is not None and getattr(levtype, 'value', '') == 'pl' and level_w is not None:
            effective_param = f"{param}{level_w.value}"

    display_name = _safe_display_name(effective_param)
    if _get_var_settings_safe(param).get('is_accumulated', False):
        acc_widget = widgets_dict.get('acc_period_widget', None)
        acc_val = acc_widget.value if acc_widget is not None else ''
        return f"{acc_val}h {display_name}"
    return display_name


def _build_base_request(param, levtype, level, date, time, area):
    """Build a base MARS request from explicit parameter/levtype/level values."""
    request = {
        "param": get_base_var(param),
        "levtype": levtype,
        "date": date.strftime("%Y%m%d"),
        "time": f"{time:02d}00",
        "area": area,
    }
    if levtype == 'pl' and level is not None:
        request["levelist"] = level
    return request


def create_widgets():
    """Create and return all widgets needed for the interface"""
    # Create widgets for parameter selection — free text input
    param_widget = widgets.Text(
        value='2t',
        placeholder='shortname or param number (e.g. 2t, tp, t, 130)',
        description='Parameter:',
        style={'description_width': 'initial'},
        layout={'width': '280px'}
    )

    # Levtype widget — user specifies sfc or pl
    levtype_widget = widgets.Dropdown(
        options=['sfc', 'pl'],
        value='sfc',
        description='Levtype:',
        style={'description_width': 'initial'},
        layout={'width': '140px'}
    )

    # Pressure level widget — visible only when levtype == 'pl'
    level_widget = widgets.IntText(
        value=850,
        description='Level (hPa):',
        style={'description_width': 'initial'},
        layout={'width': '180px', 'display': 'none'}
    )

    def _on_levtype_change(change):
        level_widget.layout.display = 'flex' if change['new'] == 'pl' else 'none'
    levtype_widget.observe(_on_levtype_change, names='value')

    # Create accumulation period widget (initially hidden)
    acc_period_widget = widgets.Dropdown(
        options=[(f"{h}h", h) for h in [6, 12, 24]],  # 6h, 12h, 24h intervals
        value=24,
        description='Accumulation Period:',
        style={'description_width': 'initial'},
        layout={'display': 'none'}  # Initially hidden
    )

    # Create ensemble members widget
    n_members_widget = widgets.IntSlider(
        value=50,
        min=10,
        max=50,
        step=1,
        description='Ensemble Members:',
        style={'description_width': 'initial'},
        allow_none=False  # Add this line to prevent None values
    )

    # Function to show/hide accumulation period based on parameter selection
    def on_param_change(change):
        vs = _get_var_settings_safe(change['new'])
        if vs.get('is_accumulated', False):
            acc_period_widget.layout.display = 'flex'
        else:
            acc_period_widget.layout.display = 'none'
    
    # Register the callback
    param_widget.observe(on_param_change, names='value')

    # Create date picker widget
    date_widget = widgets.DatePicker(
        description='Valid Date:',
        value=datetime.now() - timedelta(days=2),
        style={'description_width': 'initial'}
    )

    # Create time picker widget
    time_widget = widgets.Dropdown(
        options=[(f"{h:02d}:00", h) for h in [0, 6, 12, 18]],  # 6-hour intervals
        value=12,
        description='Valid Time:',
        style={'description_width': 'initial'}
    )

    # ECMWF standard domains
    standard_regions = {
        'Custom': None,
        'Europe': [72, -25, 35, 45],
        'North America': [85, -170, 7, -50],
        'Asia': [80, 30, 5, 180],
        'Tropics': [30, -180, -30, 180],
        'Northern Hemisphere': [90, -180, 0, 180],
        'Southern Hemisphere': [0, -180, -90, 180],
        'Global': [90, -180, -90, 180],
        'Mediterranean': [48, -10, 30, 45],
        'North Atlantic': [75, -80, 25, 20],
        'North Pacific': [75, 120, 15, -120],
        'South America': [15, -90, -60, -30],
        'Africa': [40, -20, -40, 60],
        'Australia': [-10, 110, -50, 180]
    }
    
    # Create standard regions dropdown
    region_widget = widgets.Dropdown(
        options=[(name, name) for name in standard_regions.keys()],
        value='Custom',
        description='Standard Region:',
        style={'description_width': 'initial'}
    )

    # Create widgets for area selection
    area_widgets = {
        'north': widgets.FloatText(value=50, description='North:', style={'description_width': 'initial'}),
        'west': widgets.FloatText(value=8, description='West:', style={'description_width': 'initial'}),
        'south': widgets.FloatText(value=48, description='South:', style={'description_width': 'initial'}),
        'east': widgets.FloatText(value=13, description='East:', style={'description_width': 'initial'})
    }

    # Create widgets for forecast settings
    max_days_widget = widgets.IntSlider(
        value=3,
        min=1,
        max=15,
        step=1,
        description='Max Days:',
        style={'description_width': 'initial'}
    )

    step_interval_widget = widgets.Dropdown(
        options=[6, 12, 24],
        value=12,
        description='Step Interval (hours):',
        style={'description_width': 'initial'}
    )

    # Create model selection widget as a dropdown with multiple selection
    model_settings = get_model_settings()
    model_widgets = widgets.SelectMultiple(
        options=tuple(model_settings['models'].keys()),
        value=tuple(model_settings['models'].keys()),  # All selected by default
        description='Models:',
        style={'description_width': 'initial'},
        layout={'width': 'auto', 'height': '120px'}
    )

    # ---- Custom MARS model entry widgets ----
    custom_model_name_w = widgets.Text(
        value='',
        placeholder='e.g. IFS 49r1',
        description='Name:',
        style={'description_width': 'initial'},
        layout={'width': '280px'}
    )
    custom_class_w = widgets.Text(
        value='',
        placeholder='e.g. od, ai, rd',
        description='class:',
        style={'description_width': 'initial'},
        layout={'width': '220px'}
    )
    custom_type_w = widgets.Dropdown(
        options=['fc', 'cf', 'pf', 'an'],
        value='fc',
        description='type:',
        style={'description_width': 'initial'},
        layout={'width': '160px'}
    )
    custom_stream_w = widgets.Dropdown(
        options=['oper', 'enfo', 'dacl'],
        value='oper',
        description='stream:',
        style={'description_width': 'initial'},
        layout={'width': '180px'}
    )
    custom_expver_w = widgets.Text(
        value='',
        placeholder='e.g. 1, iekm',
        description='expver:',
        style={'description_width': 'initial'},
        layout={'width': '220px'}
    )
    custom_ensemble_w = widgets.Checkbox(
        value=False,
        description='Ensemble',
        style={'description_width': 'initial'},
        indent=False,
        layout={'width': '120px'}
    )
    custom_n_members_w = widgets.IntText(
        value=50,
        description='Members:',
        style={'description_width': 'initial'},
        layout={'width': '160px', 'display': 'none'}
    )

    def _toggle_members_vis(change):
        custom_n_members_w.layout.display = 'flex' if change['new'] else 'none'
        # Warn if ensemble is checked but type is not 'pf'
        if change['new'] and custom_type_w.value != 'pf':
            custom_status_w.value = (
                '<span style="color:orange">⚠ Ensemble checked but type is '
                f'<b>{custom_type_w.value}</b>. '
                'MARS keyword <code>number</code> will only be sent when type=pf.</span>'
            )
        else:
            custom_status_w.value = ''
    custom_ensemble_w.observe(_toggle_members_vis, names='value')

    def _toggle_type_warn(change):
        # If ensemble is checked but type changes away from 'pf', warn
        if custom_ensemble_w.value and change['new'] != 'pf':
            custom_status_w.value = (
                '<span style="color:orange">⚠ Ensemble checked but type is '
                f'<b>{change["new"]}</b>. '
                'MARS keyword <code>number</code> will only be sent when type=pf.</span>'
            )
        else:
            custom_status_w.value = ''
    custom_type_w.observe(_toggle_type_warn, names='value')

    # Extra MARS keyword arguments (dynamic key-value rows)
    extra_mars_label = widgets.HTML(
        value='<b style="font-size:12px">Extra MARS arguments (optional):</b>',
        layout={'width': '280px'}
    )
    extra_mars_container = widgets.VBox([], layout={'width': '100%'})

    def _make_extra_row(key='', val=''):
        k = widgets.Text(value=key, placeholder='key (e.g. model)', layout={'width': '130px'})
        v = widgets.Text(value=val, placeholder='value', layout={'width': '170px'})
        remove_btn = widgets.Button(description='×', layout={'width': '30px'}, button_style='danger')
        row = widgets.HBox([k, v, remove_btn])
        def _remove(_):
            rows = list(extra_mars_container.children)
            if row in rows:
                rows.remove(row)
                extra_mars_container.children = rows
        remove_btn.on_click(_remove)
        return row

    add_extra_btn = widgets.Button(description='+ Add arg', button_style='', layout={'width': '100px'})
    def _on_add_extra(_):
        extra_mars_container.children = list(extra_mars_container.children) + [_make_extra_row()]
    add_extra_btn.on_click(_on_add_extra)

    # Color picker (optional)
    custom_color_w = widgets.ColorPicker(
        value='#e377c2',
        description='Color:',
        concise=True,
        style={'description_width': 'initial'},
        layout={'width': '160px'}
    )

    add_model_btn = widgets.Button(
        description='Add Custom Model',
        button_style='success',
        icon='plus',
        layout={'width': '200px'}
    )

    custom_models_list_w = widgets.HTML(
        value='<i style="color:gray">No custom models added yet</i>',
        layout={'width': '100%'}
    )
    custom_status_w = widgets.HTML(value='', layout={'width': '100%'})

    # Store removable custom model names
    _added_custom_names = []  # mutable list shared via closure

    def _refresh_custom_models_list():
        """Refresh the HTML list of custom models and update the model_widgets options."""
        if not _added_custom_names:
            custom_models_list_w.value = '<i style="color:gray">No custom models added yet</i>'
        else:
            items = []
            for cname in _added_custom_names:
                cinfo = get_custom_models().get(cname, {})
                ens_label = ' (ensemble)' if cinfo.get('ensemble') else ''
                summary = ', '.join(f'{k}={v}' for k, v in cinfo.items()
                                    if k not in ('ensemble', 'number'))
                items.append(
                    f'<div style="margin:2px 0">'
                    f'<b>{cname}</b>{ens_label} — <span style="color:#555">{summary}</span>'
                    f'</div>'
                )
            custom_models_list_w.value = ''.join(items)

        # Update the SelectMultiple widget to include custom models
        predefined = list(get_model_settings()['models'].keys())
        # Keep current selection, add any new models
        prev_selected = set(model_widgets.value)
        new_options = list(dict.fromkeys(predefined))  # preserves order, removes dupes
        model_widgets.options = tuple(new_options)
        # Re-select previously selected + any newly added custom model
        model_widgets.value = tuple(n for n in new_options if n in prev_selected or n in _added_custom_names)

    def _on_add_model(_):
        name = custom_model_name_w.value.strip()
        if not name:
            custom_status_w.value = '<span style="color:red">⚠ Please enter a model name</span>'
            return
        cls = custom_class_w.value.strip()
        if not cls:
            custom_status_w.value = '<span style="color:red">⚠ Please enter a class</span>'
            return

        retrieval_args = {
            'class': cls,
            'type': custom_type_w.value,
            'stream': custom_stream_w.value,
        }
        exp = custom_expver_w.value.strip()
        if exp:
            retrieval_args['expver'] = exp

        # Collect extra MARS args
        for row in extra_mars_container.children:
            children = row.children
            k = children[0].value.strip()
            v = children[1].value.strip()
            if k and v:
                # Try to parse as number/list
                try:
                    import ast
                    v = ast.literal_eval(v)
                except Exception:
                    pass
                retrieval_args[k] = v

        is_ens = custom_ensemble_w.value
        n_mem = custom_n_members_w.value if is_ens else 50

        register_custom_model(
            name=name,
            retrieval_args=retrieval_args,
            is_ensemble=is_ens,
            n_members=n_mem,
            color=custom_color_w.value
        )
        if name not in _added_custom_names:
            _added_custom_names.append(name)

        _refresh_custom_models_list()
        custom_status_w.value = f'<span style="color:green">✓ Added "{name}"</span>'

        # Reset input fields
        custom_model_name_w.value = ''
        custom_class_w.value = ''
        custom_expver_w.value = ''
        custom_ensemble_w.value = False
        extra_mars_container.children = []

    add_model_btn.on_click(_on_add_model)

    # Build the custom model entry panel
    custom_model_panel = widgets.VBox([
        widgets.HTML('<b>Custom MARS Model</b>'),
        custom_model_name_w,
        widgets.HBox([custom_class_w, custom_type_w]),
        widgets.HBox([custom_stream_w, custom_expver_w]),
        widgets.HBox([custom_ensemble_w, custom_n_members_w, custom_color_w]),
        extra_mars_label,
        extra_mars_container,
        add_extra_btn,
        add_model_btn,
        custom_status_w,
        widgets.HTML('<hr style="margin:4px 0"><b>Added Custom Models:</b>'),
        custom_models_list_w,
    ], layout={'border': '1px solid #ccc', 'padding': '8px', 'width': '50%'})

    predefined_panel = widgets.VBox([
        widgets.HTML('<b>Predefined Models</b>'),
        model_widgets,
    ], layout={'padding': '8px', 'width': '50%'})

    # Create a button to generate the forecast
    generate_button = widgets.Button(
        description='Forecast Setup',
        button_style='primary'
    )

    # Create output widget to display results
    output = widgets.Output()

    # Create the map (use CartoDB Positron — no Referer header required)
    m = Map(
        center=(49, 10.5),
        zoom=6,
        basemap=basemaps.CartoDB.Positron,
    )
    draw_control = DrawControl()
    m.add_control(draw_control)

    return {
        'param_widget': param_widget,
        'levtype_widget': levtype_widget,
        'level_widget': level_widget,
        'acc_period_widget': acc_period_widget,
        'n_members_widget': n_members_widget,
        'date_widget': date_widget,
        'time_widget': time_widget,
        'region_widget': region_widget,
        'area_widgets': area_widgets,
        'max_days_widget': max_days_widget,
        'step_interval_widget': step_interval_widget,
        'model_widgets': model_widgets,
        'predefined_panel': predefined_panel,
        'custom_model_panel': custom_model_panel,
        'generate_button': generate_button,
        'output': output,
        'map': m,
        'draw_control': draw_control,
        'standard_regions': standard_regions
    }

def update_map_rectangle(widgets_dict):
    """Update the map rectangle when coordinates are manually changed"""
    # Clear existing rectangle layers
    for layer in list(widgets_dict['map'].layers)[1:]:  # Skip the base layer
        if hasattr(layer, 'bounds'):  # Check if it's a rectangle
            widgets_dict['map'].remove_layer(layer)
    
    # Get current coordinates
    north = widgets_dict['area_widgets']['north'].value
    west = widgets_dict['area_widgets']['west'].value
    south = widgets_dict['area_widgets']['south'].value
    east = widgets_dict['area_widgets']['east'].value
    
    # Create new rectangle
    from ipyleaflet import Rectangle
    bounds = [[south, west], [north, east]]
    rectangle = Rectangle(
        bounds=bounds,
        color='red',
        fill_color='red',
        fill_opacity=0.2
    )
    widgets_dict['map'].add_layer(rectangle)
    
    # Update map center and zoom to fit the area
    center_lat = (north + south) / 2
    center_lon = (east + west) / 2
    widgets_dict['map'].center = (center_lat, center_lon)
    
    # Calculate appropriate zoom level based on area size
    lat_span = abs(north - south)
    lon_span = abs(east - west)
    max_span = max(lat_span, lon_span)
    
    if max_span > 100:
        zoom = 2
    elif max_span > 50:
        zoom = 3
    elif max_span > 20:
        zoom = 4
    elif max_span > 10:
        zoom = 5
    elif max_span > 5:
        zoom = 6
    elif max_span > 2:
        zoom = 7
    else:
        zoom = 8
    
    widgets_dict['map'].zoom = zoom

def on_coordinate_change(change, widgets_dict):
    """Handle coordinate widget changes"""
    update_map_rectangle(widgets_dict)

def on_region_change(change, widgets_dict):
    """Handle standard region selection"""
    selected_region = change['new']
    if selected_region != 'Custom':
        coords = widgets_dict['standard_regions'][selected_region]
        if coords:
            # Update coordinate widgets
            widgets_dict['area_widgets']['north'].value = coords[0]
            widgets_dict['area_widgets']['west'].value = coords[1]
            widgets_dict['area_widgets']['south'].value = coords[2]
            widgets_dict['area_widgets']['east'].value = coords[3]
            
            # Update map rectangle
            update_map_rectangle(widgets_dict)

def handle_draw(target, action, geo_json):
    """Handle drawing on the map"""
    global point
    if action == 'created':
        # Clear any existing layers
        for layer in list(target.widgets_dict['map'].layers)[1:]:  # Skip the base layer
            target.widgets_dict['map'].remove_layer(layer)
            
        # Get the coordinates from the geojson
        coords = geo_json['geometry']['coordinates']
        
        if geo_json['geometry']['type'] == 'Point':
            # For a point, create a small box around it
            point = coords  # Store point coordinates globally
            west = point[0] - 3
            east = point[0] + 3
            south = point[1] - 3
            north = point[1] + 3
            point.reverse()
            
            # Create a rectangle layer to highlight the area
            from ipyleaflet import Rectangle
            bounds = [[south, west], [north, east]]
            rectangle = Rectangle(
                bounds=bounds,
                color='red',
                fill_color='red',
                fill_opacity=0.2
            )
            target.widgets_dict['map'].add_layer(rectangle)
            
        else:
            # For a polygon/rectangle, get the bounds
            point = None  # Reset point if rectangle is drawn
            bounds = coords[0]
            west = min(coord[0] for coord in bounds)
            east = max(coord[0] for coord in bounds)
            south = min(coord[1] for coord in bounds)
            north = max(coord[1] for coord in bounds)
            
            # Create a rectangle layer to highlight the bounding box
            from ipyleaflet import Rectangle
            bounds = [[south, west], [north, east]]
            rectangle = Rectangle(
                bounds=bounds,
                color='red',
                fill_color='red',
                fill_opacity=0.2
            )
            target.widgets_dict['map'].add_layer(rectangle)
        
        # Update the area widgets through the widgets_dict
        target.widgets_dict['area_widgets']['north'].value = round(north, 5)
        target.widgets_dict['area_widgets']['west'].value = round(west, 5)
        target.widgets_dict['area_widgets']['south'].value = round(south, 5)
        target.widgets_dict['area_widgets']['east'].value = round(east, 5)
        
        # Reset region widget to Custom since user drew a custom area
        target.widgets_dict['region_widget'].value = 'Custom'

def on_button_clicked(b, widgets_dict):
    """Handle button click event"""
    global valid_date, param, area_sub, forecast_dates, forecast_steps, max_days, step_interval
    
    with widgets_dict['output']:
        widgets_dict['output'].clear_output()
        
        # Get values from widgets
        param = widgets_dict['param_widget'].value.strip()
        levtype = widgets_dict['levtype_widget'].value
        level = widgets_dict['level_widget'].value if levtype == 'pl' else None
        valid_date = datetime.combine(widgets_dict['date_widget'].value, 
                                    datetime.min.time().replace(hour=widgets_dict['time_widget'].value))
        area_sub = [widgets_dict['area_widgets']['north'].value, 
                   widgets_dict['area_widgets']['west'].value, 
                   widgets_dict['area_widgets']['south'].value, 
                   widgets_dict['area_widgets']['east'].value]
        max_days = widgets_dict['max_days_widget'].value
        step_interval = widgets_dict['step_interval_widget'].value
        selected_models = widgets_dict['model_widgets'].value
        #n_members = widgets_dict['n_members_widget'].value
        
        level_str = f" at {level} hPa" if level else ""
        print(f"Setting up forecast dates for {param} (levtype={levtype}{level_str}) with validity date {valid_date}")
        print(f"Area coordinates [N,W,S,E]: {area_sub}")
        print(f"Selected models: {', '.join(selected_models)}")
        print(f"Number of ensemble members: {widgets_dict['n_members_widget'].value}")
        if point:
            print(f"Selected point coordinates: {point}")
        
        # Calculate forecast dates and steps (exclude step 0)
        steps = list(range(step_interval, max_days*24 + step_interval, step_interval))
        forecast_dates = []
        forecast_steps = []
        
        for step in steps:
            forecast_date = valid_date - timedelta(hours=step)
            forecast_dates.append(forecast_date)
            forecast_steps.append(step)
        
        # Create dataframe to display the information
        df = pd.DataFrame({
            'Forecast Date': forecast_dates,
            'Step (hours)': forecast_steps
        })
        
        print(f"\nValidity/Observation/Analysis date: {valid_date}")
        print("\nForecast initialization dates and their steps:")
        display(df)
        
        # Store the configuration in the widgets dictionary for later use
        widgets_dict['config'] = {
            'param': param,
            'levtype': levtype,
            'level': level,
            'valid_date': valid_date,
            'area_sub': area_sub,
            'point': point,
            'max_days': max_days,
            'step_interval': step_interval,
            'forecast_dates': forecast_dates,
            'forecast_steps': forecast_steps,
            'selected_models': selected_models,
            'n_members': widgets_dict['n_members_widget'].value
        }
        print(widgets_dict['config'])
        
def setup_interface():
    """Set up the complete interface"""
    widgets_dict = create_widgets()
    
    # Add widgets_dict to the draw_control for the draw handler to access
    widgets_dict['draw_control'].widgets_dict = widgets_dict
    
    # Register the callback for drawing
    widgets_dict['draw_control'].on_draw(handle_draw)
    
    # Register callbacks for coordinate changes
    for coord_widget in widgets_dict['area_widgets'].values():
        coord_widget.observe(lambda change: on_coordinate_change(change, widgets_dict), names='value')
    
    # Register callback for region selection
    widgets_dict['region_widget'].observe(lambda change: on_region_change(change, widgets_dict), names='value')
    
    # Attach the button click event
    widgets_dict['generate_button'].on_click(lambda b: on_button_clicked(b, widgets_dict))
    
    # Display all widgets
    print("Please configure your forecast settings:")
    display(widgets.VBox([
        widgets.HBox([widgets_dict['param_widget'],
                     widgets_dict['levtype_widget'],
                     widgets_dict['level_widget'],
                     widgets_dict['acc_period_widget'],
                     widgets_dict['n_members_widget']]),
        widgets.HBox([widgets_dict['date_widget'], 
                     widgets_dict['time_widget']]),
        widgets.HBox([widgets_dict['max_days_widget'], 
                     widgets_dict['step_interval_widget']]),
        widgets.HBox([widgets_dict['region_widget']]),  # Standard regions dropdown
        widgets.HBox([widgets_dict['area_widgets']['north'], 
                     widgets_dict['area_widgets']['west'],
                     widgets_dict['area_widgets']['south'],
                     widgets_dict['area_widgets']['east']]),
        widgets.HTML('<hr><b style="font-size:14px">Model Selection</b>'),
        widgets.HBox([widgets_dict['predefined_panel'],
                      widgets_dict['custom_model_panel']]),
        widgets_dict['map'],
        widgets_dict['generate_button'],
        widgets_dict['output']
    ]))
    
    return widgets_dict

def retrieve_and_store_data(widgets_dict, base_path):
    """Retrieve data and create the plot"""
    global valid_date, param, area_sub, point, forecast_dates, forecast_steps, max_days, step_interval, data_dir
    
    # Get variable settings (safe — works for unknown params too)
    var_settings = _get_var_settings_safe(param)
    levtype = widgets_dict['config']['levtype']
    level = widgets_dict['config']['level']
    area_sub = widgets_dict['config']['area_sub']

    # Setup data directories
    area_str = get_area_string(area_sub)
    date_str = valid_date.strftime("%Y%m%d")

    # If a point is selected, check whether it falls inside an already-retrieved area
    point = widgets_dict['config']['point']
    if point is not None:
        reuse_area_str, reuse_area = _find_existing_directory_for_point(base_path, param, date_str, point)
        if reuse_area_str is not None:
            area_str = reuse_area_str
            area_sub = reuse_area  # use the existing directory's area for MARS requests

    grib_dir, obs_dir, plot_dir = setup_data_directories(base_path, param, area_str, date_str)
    data_dir = grib_dir  # Store for later use

    n_members = widgets_dict['config']['n_members']
    # Calculate distances using haversine formula
    def haversine_distance(lat1, lon1, lat2, lon2):
        from math import radians, sin, cos, sqrt, atan2
        R = 6371  # Earth's radius in km
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c
    nearest_gridinfo_dict={}
    try:
        print("Retrieving observations...")
        # Only retrieve observations for surface variables
        if levtype == 'sfc':
            try:
                # Define file paths
                obs_file = os.path.join(obs_dir, f"STVL_{param}_{date_str}_{valid_date.hour:02d}00.grib")
                obs_csv = os.path.join(obs_dir, f"STVL_{param}_{date_str}_{valid_date.hour:02d}00.csv")
                
                # Check if files already exist
                if os.path.exists(obs_file) and os.path.exists(obs_csv):
                    print(f"Loading existing observation files from {obs_file}")
                    obs = mv.read(obs_file)
                    obs_df = pd.read_csv(obs_csv)
                else:
                    print(f"Retrieving new observation data...")
                    obs1 = mv.stvl(
                        parameter=get_base_var(param),
                        dates=valid_date.strftime("%Y%m%d"),
                        times=valid_date.hour,
                        sources="synop",
                        area=area_sub,
                        period=str(widgets_dict['acc_period_widget'].value) if var_settings.get('is_accumulated', False) else None
                    )

                    # Retrieve HDOBS observations
                    obs2 = mv.stvl(
                        parameter=get_base_var(param),
                        dates=valid_date.strftime("%Y%m%d"),
                        times=valid_date.hour,
                        sources="hdobs",
                        area=area_sub,
                        period=str(widgets_dict['acc_period_widget'].value) if var_settings.get('is_accumulated', False) else None
                    )
                    
                    # Merge and remove duplicates
                    obs = mv.merge(obs1, obs2)
                    obs = mv.remove_duplicates(obs)
                    
                    # Save observations to file
                    mv.write(obs_file, obs)
                    
                    # Convert to dataframe and save
                    obs_df = obs.to_dataframe()
                    obs_df.to_csv(obs_csv, index=False)

                # Process observations data
                if point:
                    # Add distance column
                    target_lat, target_lon = point[0], point[1]
                    obs_df['distance'] = obs_df.apply(lambda row: haversine_distance(row['latitude'], row['longitude'], 
                                                                            target_lat, target_lon), axis=1)
                    # Get nearest station
                    nearest_station = obs_df.loc[obs_df['distance'].idxmin()]
                    print(f"Nearest station (stnid: {nearest_station['stnid']}, elevation: {nearest_station['elevation']},lat: {nearest_station['latitude']}, lon: {nearest_station['longitude']}, distance: {nearest_station['distance']:.2f} km, value: {nearest_station['value_0']:.2f}): {nearest_station}")
                    obs_fs_area=nearest_station['value_0']
                else:
                    # Calculate center of original area
                    center_lat = (area_sub[0] + area_sub[2]) / 2
                    center_lon = (area_sub[1] + area_sub[3]) / 2
                    
                    # Calculate distance to center for all stations
                    obs_df['distance'] = obs_df.apply(lambda row: haversine_distance(row['latitude'], row['longitude'],
                                                                            center_lat, center_lon), axis=1)
                    
                    expansion = 0
                    stations_found = False
                    
                    while not stations_found and expansion <= 2:
                        # Expand boundaries
                        expanded_area = [
                            area_sub[0] + expansion,  # North
                            area_sub[1] - expansion,  # West 
                            area_sub[2] - expansion,  # South
                            area_sub[3] + expansion   # East
                        ]
                        
                        # Filter stations within expanded boundaries
                        filtered_df = obs_df[
                            (obs_df['latitude'] <= expanded_area[0]) &  # North
                            (obs_df['longitude'] >= expanded_area[1]) &  # West
                            (obs_df['latitude'] >= expanded_area[2]) &  # South
                            (obs_df['longitude'] <= expanded_area[3])    # East
                        ]
                        
                        if len(filtered_df) > 0:
                            stations_found = True
                            obs_df = filtered_df.reset_index(drop=True)
                            
                            # Get nearest station to center
                            nearest_station = obs_df.loc[obs_df['distance'].idxmin()]
                            
                            print(f"Number of stations found: {len(filtered_df)}")
                            print(f"Area expanded by {expansion:.3f} degrees")
                            print(f"Nearest station to original area center:")
                            print(f"Station ID: {nearest_station['stnid']}")
                            print(f"Distance from center: {nearest_station['distance']:.2f} km")
                            print('...')
                        else:
                            expansion += 0.25
                        
                        obs_fs_area=(obs_df['value_0'].values).mean()

                    # If no stations found within 2 degrees, use nearest overall station
                    if not stations_found:
                        nearest_station = obs_df.loc[obs_df['distance'].idxmin()]
                        print(f"No stations found within expanded area. Using nearest station overall:")
                        print(f"Station ID: {nearest_station['stnid']}")
                        print(f"Distance from center: {nearest_station['distance']:.2f} km")
                        obs_fs_area=(nearest_station['value_0']).mean()

                # Convert to display units using the conversion function
                print(f'OBS value: {obs_fs_area:.2f}')
                obs_fs_area = _safe_convert_to_display(obs_fs_area, param)
                print(f"Station ID: {nearest_station['stnid']}")
                print(f"Station elevation: {nearest_station['elevation']} m")
                print(f"Station latitude: {nearest_station['latitude']}°")
                print(f"Station longitude: {nearest_station['longitude']}°")
                nearest_gridinfo_dict['nearest_station'] = nearest_station
            except Exception as e:
                print(f"Error retrieving observations: {str(e)}")
                obs_fs_area = None
        else:
            print(f"No observations available for pressure level variable {param}")
            obs_fs_area = None
        if param == '2t' or param == '2d':
            obs_fs_area = obs_fs_area - 273.15

    except Exception as e:
        print("Error in observation processing:", e)
        obs_fs_area = None

    # Data Retrieval and Processing
    analysis_area = None
    if param != 'tp':  # Skip analysis for total precipitation
        try:
            print("Retrieving analysis...")
            analysis_file = os.path.join(grib_dir, f"Analysis_{param}_{date_str}_{valid_date.hour:02d}00.grib")
            
            if os.path.exists(analysis_file):
                print("Analysis file exists, reading with metview...")
                analysis = mv.read(analysis_file)
            else:
                print("Retrieving analysis from MARS...")
                request = _build_base_request(param, levtype, level, valid_date, valid_date.hour, area_sub)
                request.update(get_analysis_settings())
                
                # Grid from settings: only keep for pressure level variables
                if levtype != 'pl':
                    request.pop('grid', None)
                
                analysis = _retrieve_with_timeout(MARS_RETRIEVAL_TIMEOUT, "mars", request)
                analysis.save(analysis_file)
                analysis = mv.read(analysis_file)

            # Get units from analysis data
            units_analysis = get_grib_units(analysis)
            print(f"Analysis units: {units_analysis}")
            
            # Convert to display units using the new system
            analysis = _safe_convert_to_display(analysis, param, units_analysis)

            if point:
                print(f"The nearest grid point is at {mv.nearest_gridpoint_info(analysis, point)[0]['distance']}km from the selected point")
                nearest_gridinfo_dict['nearest_analysis'] = mv.nearest_gridpoint_info(analysis, point)[0]
                analysis_area = mv.nearest_gridpoint(analysis,point)
            else:
                analysis_area = mv.integrate(analysis, area_sub)
        except Exception as e:
            print(f"Error retrieving analysis: {str(e)}")
            analysis_area = None

    # Set reference value
    if obs_fs_area is not None:
        reference = obs_fs_area
    elif analysis_area is not None:
        reference = analysis_area
    else:
        reference = None

    # Calculate climatology
    clim_em_area = None
    try:
        clim_file = os.path.join(grib_dir, f"Climatology_{param}_{date_str}_{valid_date.hour:02d}00.grib")
        
        if os.path.exists(clim_file):
            print("Climatology file exists, reading with metview...")
            data_clim_em = mv.read(clim_file)
        else:
            print("Retrieving climatology from MARS...")
            request = _build_base_request(param, levtype, level, valid_date, valid_date.hour, area_sub)
            request.update(get_climatology_settings())
            
            # Grid from settings: only keep for pressure level variables
            if levtype != 'pl':
                request.pop('grid', None)
            
            data_clim_em = _retrieve_with_timeout(MARS_RETRIEVAL_TIMEOUT, "mars", request)
            data_clim_em.save(clim_file)
            data_clim_em = mv.read(clim_file)

        # Convert to display units using the new system
        data_clim_em = _safe_convert_to_display(data_clim_em, param)
        if point:
            clim_em_area = mv.nearest_gridpoint(data_clim_em,point)
            nearest_gridinfo_dict['nearest_climatology'] = mv.nearest_gridpoint_info(data_clim_em,point)[0]
        else:
            clim_em_area = mv.integrate(data_clim_em, area_sub)
    except Exception as e:
        print(f"Error retrieving climatology: {str(e)}")
        clim_em_area = None

    # Initialize DataFrame to store all data with time information
    data_df = pd.DataFrame({
        'forecast_date': forecast_dates,
        'forecast_step': forecast_steps,
        'valid_date': [valid_date] * len(forecast_dates)
    })

    # Add columns for each model, including all fields that may be used later
    for model_name in widgets_dict['model_widgets'].value:
        model_settings = get_model_retrieval_settings(model_name)
        if model_settings['ensemble']:
            # For ensemble models: always initialize all possible ensemble output columns for later usage
            data_df[f'{model_name}_ensemble'] = None
            data_df[f'{model_name}_mean'] = None
            data_df[f'{model_name}_mean_area'] = None
            data_df[f'{model_name}_ens_area'] = None
            data_df[f'{model_name}_ENS_mem'] = None
            data_df[f'{model_name}_ENS_mean'] = None
            data_df[f'{model_name}_ENS_mem_area'] = None
        else:
            # For deterministic models
            data_df[f'{model_name}_area'] = None
            data_df[f'{model_name}_field'] = None

    # Add columns for reference/verification data
    data_df['observations'] = None
    data_df['analysis'] = None
    data_df['climatology'] = None

    # Retrieve data for each model, forecast date, and step
    timed_out_models = set()  # models that hit a timeout — skip for remaining steps
    for idx, (fc_date, step) in enumerate(zip(forecast_dates, forecast_steps)):
        print(f"\nRetrieving data for forecast date {fc_date} at step {step}...")

        for model_name in widgets_dict['model_widgets'].value:
            if model_name in timed_out_models:
                continue
            print(f"Retrieving {model_name}...")
            if model_name in ['DE-LUMI', 'DE-ATOS']:
                if fc_date.hour != 0:
                    print(f"Skipping {model_name} - only available for 00:00:00 initialization times")
                    continue
                if step > 121:
                    print(f"Skipping {model_name} - not available for steps > 120h")
                    continue

            try:
                model_file = os.path.join(grib_dir, f"{model_name}_{param}_{fc_date.strftime('%Y%m%d')}_{fc_date.hour:02d}00_step{step}.grib")

                if os.path.exists(model_file):
                    print(f"Model file exists for {model_name} at step {step}, reading with metview...")
                    data = mv.read(model_file)
                else:
                    print(f"Retrieving {model_name} from MARS for step {step}...")
                    request = _build_base_request(param, levtype, level, fc_date, fc_date.hour, area_sub)
                    model_ret = get_model_retrieval_settings(model_name)
                    request.update(model_ret)

                    # Grid from model_settings: keep for all levtypes for DE-ATOS,
                    # only for pl otherwise.
                    if levtype != 'pl' and model_name != 'DE-ATOS':
                        request.pop('grid', None)

                    if model_name == "DE-LUMI":
                        if var_settings.get('is_accumulated', False):
                            acc_period = widgets_dict['acc_period_widget'].value
                            step_start = max(0, step - acc_period)
                            if step_start == 0:
                                request["step"] = f'{int(step)-1}-{int(step)}'
                            else:
                                request["step"] = f'{step_start-1}-{step_start}/{int(step)-1}-{step}'
                        else:
                            request["step"] = step
                        lumi_address = request.pop("address", None)
                        request = sanitize_mars_request(request)
                        print(f"[DE-LUMI request] {request}")
                        data = _retrieve_with_timeout(
                            MARS_RETRIEVAL_TIMEOUT,
                            "polytope",
                            "ecmwf-destination-earth",
                            request,
                            address=lumi_address,
                            stream=False,
                        )
                        data.save(model_file)
                        data = mv.read(model_file)
                    else:
                        # --- Build step/number/expver into request ---
                        is_ensemble = model_ret.get('ensemble', False)
                        if var_settings.get('is_accumulated', False):
                            acc_period = widgets_dict['acc_period_widget'].value
                            step_start = max(0, step - acc_period)
                            request["step"] = f"{step_start}/{step}"
                        else:
                            request["step"] = step

                        # number: only add when type=pf (sanitize_mars_request will
                        # strip it otherwise, but be explicit here)
                        if is_ensemble and str(request.get('type', '')).lower() == 'pf':
                            request["number"] = [1, "TO", widgets_dict['config']['n_members']]

                        # Special expver override only for predefined AIFS ENS
                        if model_name == "AIFS ENS":
                            if fc_date <= datetime(2025, 7, 1, 0, 0):
                                request['expver'] = '103'
                            else:
                                request['expver'] = '1'

                        # Sanitize: enforce MARS keyword interdependencies
                        request = sanitize_mars_request(request)
                        print(f"[{model_name} request] {request}")
                        data = _retrieve_with_timeout(MARS_RETRIEVAL_TIMEOUT, "mars", request)
                        data.save(model_file)
                        data = mv.read(model_file)

                # Handle accumulated variables
                if var_settings.get('is_accumulated', False):
                    model_settings = get_model_retrieval_settings(model_name)
                    print(f"Processing accumulated data for {model_name}, step {step}")

                    if model_settings['ensemble']:
                        # n_members from widgets_dict['config'], use reliable source
                        n_members = widgets_dict['config'].get('n_members', 0)
                        deacc = []
                        for i in range(n_members):
                            member_data = data.select(shortName=get_base_var(param), number=i+1)
                            print(f"Member {i+1}: Found {len(member_data)} fields")
                            if len(member_data) >= 2:
                                diff = member_data[1] - member_data[0]
                                deacc.append(diff)
                        if deacc:
                            data = mv.merge(*deacc)
                            print(f"Successfully processed {len(deacc)} ensemble members")
                        else:
                            data = None
                            print("Warning: No valid ensemble members found")
                    else:
                        print(f"Data fields: {len(data)}")
                        if len(data) >= 2:
                            data = data[1] - data[0]
                            print("Calculated accumulation difference")
                        else:
                            data = data[0]
                            print("Warning: Only one field found, using as is")
                    if data is not None:
                        data = _safe_convert_to_display(data, param)
                else:
                    # Non-accumulated variables
                    print(f"Processing non-accumulated data for {model_name}, step {step}")
                    model_settings = get_model_retrieval_settings(model_name)
                    if model_settings['ensemble']:
                        if levtype == 'pl' and level is not None:
                            data = data.select(shortName=get_base_var(param), levelist=level)
                        else:
                            data = data.select(shortName=get_base_var(param))
                        if data is not None:
                            data = _safe_convert_to_display(data, param)
                            print(f"Successfully processed ensemble data with {len(data)} members")
                        else:
                            print("Warning: No valid ensemble data found")
                    else:
                        print(f"Converting {model_name} to display units")
                        if data is not None:
                            data = _safe_convert_to_display(data, param)
                            print(f"Converted {model_name} to display units")
                        else:
                            print("Warning: No data found for deterministic model")

                # Store the processed data in the DataFrame
                model_settings = get_model_retrieval_settings(model_name)
                print(f"Storing data for {model_name} (ensemble: {model_settings['ensemble']})")
                print(data)

                if model_settings['ensemble']:
                    if data is not None:
                        data_df.at[idx, f'{model_name}_ensemble'] = data

                        # Points and areas
                        if point:
                            data_point = mv.nearest_gridpoint(data, point)
                            # Filter out None values (point outside grid)
                            valid_vals = [v for v in data_point if v is not None]
                            if not valid_vals:
                                print(f"Warning: nearest_gridpoint returned no valid values for {model_name} — point may be outside the data grid")
                                continue
                            nearest_gridinfo_dict[model_name+'_nearest']= mv.nearest_gridpoint_info(data[0], point)[0]
                            data_df.at[idx, f'{model_name}_ENS_mem'] = np.array(valid_vals)
                            mean_val = np.mean(valid_vals)
                        else:
                            data_df.at[idx, f'{model_name}_ENS_mem'] = data
                            data_df.at[idx, f'{model_name}_ENS_mem_area'] = mv.integrate(data, area_sub)
                            mean_val = mv.mean(data)
                        data_df.at[idx, f'{model_name}_ENS_mean'] = mean_val

                        if point:
                            area_val = mean_val
                        else:
                            area_val = mv.integrate(mean_val, area_sub)
                        data_df.at[idx, f'{model_name}_mean_area'] = area_val
                        print(f"Area mean value: {area_val}")

                        if point:
                            ens_area = list(valid_vals)
                        else:
                            ens_area = [mv.integrate(member, area_sub) for member in data]
                        data_df.at[idx, f'{model_name}_ens_area'] = ens_area
                        print(f"Ensemble area values: min={min(ens_area) if ens_area else None}, max={max(ens_area) if ens_area else None}, mean={np.mean(ens_area) if ens_area else None}")
                else:
                    if data is not None:
                        data_df.at[idx, f'{model_name}_field'] = data
                        if point:
                            area_val = mv.nearest_gridpoint(data, point)
                            if area_val is None:
                                print(f"Warning: nearest_gridpoint returned None for {model_name} — point may be outside the data grid")
                                continue
                            nearest_gridinfo_dict[model_name+'_nearest']= mv.nearest_gridpoint_info(data[0], point)[0]
                        else:
                            area_val = mv.integrate(data, area_sub)
                        data_df.at[idx, f'{model_name}_area'] = area_val
                        print(f"value: {area_val}")

            except TimeoutError as e:
                print(
                    f"\n{'='*60}\n"
                    f"TIMEOUT: {model_name} for step {step} — skipping this model for all remaining steps.\n"
                    f"{str(e)}\n"
                    f"{'='*60}"
                )
                timed_out_models.add(model_name)

            except Exception as e:
                error_msg = (
                    f"\n{'='*60}\n"
                    f"ERROR retrieving {model_name} for step {step}\n"
                    f"{'='*60}\n"
                    f"Exception type : {type(e).__name__}\n"
                    f"Message        : {str(e)}\n"
                )
                # Show the MARS request that failed (if available)
                try:
                    error_msg += f"Last request   : {request}\n"
                except NameError:
                    pass
                error_msg += f"{'='*60}\n{traceback.format_exc()}"
                print(error_msg)

    # Prepare reference row
    reference_row = pd.DataFrame({
        'forecast_date': [valid_date],
        'forecast_step': [0],
        'valid_date': [valid_date]
    })

    # Add all possible model columns for the reference row, set to np.nan as placeholder
    for model_name in widgets_dict['model_widgets'].value:
        model_settings = get_model_retrieval_settings(model_name)
        if model_settings['ensemble']:
            reference_row[f'{model_name}_ensemble'] = np.nan
            reference_row[f'{model_name}_mean'] = np.nan
            reference_row[f'{model_name}_mean_area'] = np.nan
            reference_row[f'{model_name}_ens_area'] = np.nan
            reference_row[f'{model_name}_ENS_mem_area'] = np.nan
            reference_row[f'{model_name}_ENS_mem'] = np.nan
            reference_row[f'{model_name}_ENS_mean'] = np.nan
        else:
            reference_row[f'{model_name}_area'] = np.nan
            reference_row[f'{model_name}_field'] = np.nan

    # Reference columns
    reference_row['observations'] = obs_fs_area
    reference_row['analysis'] = analysis_area
    reference_row['climatology'] = clim_em_area
    print(f"Reference data - obs: {obs_fs_area}, analysis: {analysis_area}, climatology: {clim_em_area}")

    # Concatenate reference row with forecast data
    data_df = pd.concat([reference_row, data_df], ignore_index=True)
    print(f"Final DataFrame shape: {data_df.shape}")
    
    # Create plotting data structure
    plot_data = {
        'data_df': data_df,
        'steps': list(range(len(forecast_dates))),
        'x_labels': [fc_date.strftime('%d/%m %H:%M') for fc_date in forecast_dates],
        'forecast_steps': forecast_steps,
        'clim_em_area': clim_em_area,
        'obs_fs_area': obs_fs_area,
        'analysis_area': analysis_area,
        'reference': reference,
        'var_settings': var_settings,
        'nearest_gridinfo_dict': nearest_gridinfo_dict,
        'base_path': base_path,
    }
    
    return plot_data 
def plot_forecast_evolution(plot_data, widgets_dict, plot_dir, export_html=False, html_filename=None):
    """Create and display the forecast evolution plot"""
    global valid_date, param, area_sub, point

    # --- Color override helper ------------------------------------------------
    # Users may set  widgets_dict['model_colors'] = {'ModelName': '#aabbcc', ...}
    # or             widgets_dict['config']['model_colors'] = {…}
    # before calling this function to override default/registered colours.
    _color_overrides = widgets_dict.get('model_colors', {})
    if not _color_overrides:
        _color_overrides = widgets_dict.get('config', {}).get('model_colors', {})

    def _resolve_color(model_name):
        """Return the colour for *model_name*, checking user overrides first."""
        # 1. Exact match in user overrides
        if model_name in _color_overrides:
            return _color_overrides[model_name]
        # 2. Case-insensitive / partial match (e.g. 'ifs' matches 'IFS Control')
        mn_lower = model_name.lower()
        for key, color in _color_overrides.items():
            if key.lower() in mn_lower or mn_lower in key.lower():
                return color
        # 3. Fall back to plot_settings
        ps = get_model_plot_settings(model_name)
        return ps.get('color', '#333333')

    def _apply_color_override(plot_settings, model_name):
        """Return a *copy* of plot_settings with colour fields replaced by
        the resolved colour for *model_name* (if an override exists)."""
        color = _resolve_color(model_name)
        ps = dict(plot_settings)  # shallow copy

        ps['color'] = color

        # Ensemble box colours
        if 'box' in ps:
            ps['box'] = dict(ps['box'])
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            ps['box']['fillcolor'] = f"rgba({r}, {g}, {b}, 0.3)"
            ps['box']['marker_color'] = color

        # Marker border/line colour
        if 'marker' in ps and isinstance(ps['marker'], dict):
            ps['marker'] = dict(ps['marker'])
            if 'line' in ps['marker'] and isinstance(ps['marker']['line'], dict):
                ps['marker']['line'] = dict(ps['marker']['line'])
                ps['marker']['line']['color'] = color
            if 'color' in ps['marker']:
                r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                ps['marker']['color'] = f"rgba({r}, {g}, {b}, 0.7)"

        # Line colour
        if 'line' in ps and ps['line'] is not None:
            ps['line'] = dict(ps['line'])
            ps['line']['color'] = color

        return ps
    # --------------------------------------------------------------------------
    
    # Create the main figure
    fig = go.Figure()

    data_df = plot_data['data_df'].iloc[::-1].reset_index(drop=True).copy()  # Make a copy to avoid modifying original

    # Determine ensemble models for dynamic x-offset spacing
    _ensemble_models = [
        m for m in widgets_dict['model_widgets'].value
        if get_model_retrieval_settings(m)['ensemble']
    ]
    _n_ens = len(_ensemble_models)
    # Build a mapping: ensemble_model_name -> x-offset
    # Spread offsets symmetrically around 0, e.g. 2 models -> -0.2, +0.2;
    # 3 models -> -0.25, 0.0, +0.25; etc.
    _ens_offsets = {}
    if _n_ens == 1:
        _ens_offsets[_ensemble_models[0]] = 0.0
    elif _n_ens > 1:
        total_span = min(0.6, 0.2 * _n_ens)  # cap at 0.6 to avoid overlap
        for idx, m in enumerate(_ensemble_models):
            _ens_offsets[m] = -total_span / 2 + idx * total_span / (_n_ens - 1)

    # Add traces for each model
    for model_name in widgets_dict['model_widgets'].value:
        model_settings = get_model_retrieval_settings(model_name)
        plot_settings = _apply_color_override(get_model_plot_settings(model_name), model_name)
        
        if model_settings['ensemble']:
            # Filter data for this model (only rows where ensemble data exists)
            model_data = data_df[data_df[f'{model_name}_ens_area'].notna()].copy()
            
            if not model_data.empty:
                # Add boxplot for ensemble
                valid_data = []
                customdata_list = []
                text_list = []
                
                for _, row in model_data.iterrows():
                    ens_area_data = row[f'{model_name}_ens_area']
                    if ens_area_data is not None and len(ens_area_data) > 0:
                        # Use the data directly
                        member_values = ens_area_data
                        
                        # Add data points with correct x-position
                        for member_idx, member_val in enumerate(member_values):
                            if member_val is not None:
                                valid_data.append((row.name, member_val))
                                
                                # Calculate bias if reference exists
                                bias = member_val - (plot_data['reference'] if plot_data['reference'] is not None else 0)
                                
                                customdata_list.append([
                                    row['forecast_step'], 
                                    member_idx + 1, 
                                    row['forecast_date'].strftime('%d/%m %H:%M'),
                                    bias
                                ])
                                
                                text_list.append(f"Initialisation Date: {row['forecast_date'].strftime('%d/%m %H:%M')}")
                
                if valid_data:
                    _dlabel = _model_display_label(model_name)
                    fig.add_trace(
                        go.Box(
                            x=[i + _ens_offsets.get(model_name, 0.0) for i, _ in valid_data],
                            y=[val for _, val in valid_data],
                            name=_dlabel,
                            fillcolor=plot_settings['box']['fillcolor'],
                            marker_color=plot_settings['box']['marker_color'],
                            boxpoints='all',
                            jitter=plot_settings['marker']['jitter'],
                            pointpos=plot_settings['marker']['pointpos'],
                            customdata=customdata_list,
                            hovertemplate=f"Model: {_dlabel}<br>Date: %{{customdata[2]}}<br>Lead Time: %{{customdata[0]}}h<br>Member: %{{customdata[1]}}<br>Value: %{{y:.2f}}{plot_data['var_settings']['units']}" + 
                                        (f"<br>Bias: %{{customdata[3]:.2f}}{plot_data['var_settings']['units']}" if plot_data['reference'] is not None else "") +
                                        "<br><extra></extra>",
                            hoverinfo='y+name+text',
                            text=text_list,
                            showlegend=True,
                            width=plot_settings['box']['width']
                        )
                    )
                
                # Add ensemble means with lines connecting points
                mean_data = data_df[data_df[f'{model_name}_mean_area'].notna()].copy()
                if not mean_data.empty:
                    # Get mean plot settings
                    mean_model_name = f"{model_name} Mean"
                    try:
                        mean_plot_settings = _apply_color_override(
                            get_model_plot_settings(mean_model_name), model_name
                        )
                    except:
                        # Fallback to base model settings if mean-specific settings don't exist
                        mean_plot_settings = plot_settings
                    
                    # Calculate bias for means
                    mean_customdata = []
                    for _, row in mean_data.iterrows():
                        bias = row[f'{model_name}_mean_area'] - (plot_data['reference'] if plot_data['reference'] is not None else 0)
                        mean_customdata.append([
                            row['forecast_step'], 
                            row['forecast_date'].strftime('%d/%m %H:%M'), 
                            bias
                        ])
                    
                    _dlabel = _model_display_label(model_name)
                    fig.add_trace(
                        go.Scatter(
                            x=mean_data.index.tolist(),
                            y=mean_data[f'{model_name}_mean_area'].tolist(),
                            mode='lines+markers',
                            name=f'{_dlabel} Mean',
                            line=dict(
                                color=mean_plot_settings['line']['color'],
                                width=mean_plot_settings['line']['width']
                            ),
                            marker=dict(
                                color=mean_plot_settings['marker']['color'],
                                size=mean_plot_settings['marker']['size'],
                                symbol=mean_plot_settings['marker']['symbol'],
                                line=dict(
                                    color=mean_plot_settings['marker']['line']['color'],
                                    width=mean_plot_settings['marker']['line']['width']
                                )
                            ),
                            customdata=mean_customdata,
                            hovertemplate=f"Initialisation Date: %{{customdata[1]}}<br>Lead Time: %{{customdata[0]}}h<br>Value: %{{y:.2f}}{plot_data['var_settings']['units']}" + 
                                        (f"<br>Bias: %{{customdata[2]:.2f}}{plot_data['var_settings']['units']}" if plot_data['reference'] is not None else "") +
                                        f"<extra>Model: {_dlabel} Mean</extra>"
                        )
                    )
        else:
            # Filter data for deterministic models
            model_data = data_df[data_df[f'{model_name}_area'].notna()].copy()
            
            if not model_data.empty:
                # Calculate bias for deterministic models
                det_customdata = []
                for _, row in model_data.iterrows():
                    bias = row[f'{model_name}_area'] - (plot_data['reference'] if plot_data['reference'] is not None else 0)
                    det_customdata.append([
                        row['forecast_step'], 
                        row['forecast_date'].strftime('%d/%m %H:%M'), 
                        bias
                    ])
                
                # Set mode based on plot_settings: use lines+markers if 'line' config exists
                mode = 'lines+markers' if plot_settings.get('line') is not None else 'markers'
                
                # Create trace dictionary
                _dlabel = _model_display_label(model_name)
                trace_dict = dict(
                    x=model_data.index.tolist(),
                    y=model_data[f'{model_name}_area'].tolist(),
                    mode=mode,
                    name=_dlabel,
                    marker=dict(
                        color=plot_settings['color'],
                        size=plot_settings['marker']['size'],
                        symbol=plot_settings['marker']['symbol'],
                        line=dict(
                            color=plot_settings['marker']['line']['color'],
                            width=plot_settings['marker']['line']['width']
                        )
                    ),
                    customdata=det_customdata,
                    hovertemplate=f"Initialisation Date: %{{customdata[1]}}<br>Lead Time: %{{customdata[0]}}h<br>Value: %{{y:.2f}}{plot_data['var_settings']['units']}" + 
                                (f"<br>Bias: %{{customdata[2]:.2f}}{plot_data['var_settings']['units']}" if plot_data['reference'] is not None else "") +
                                f"<extra>Model: {_dlabel}</extra>"
                )
                
                # Add line settings if mode includes lines
                if 'lines' in mode and plot_settings.get('line') is not None:
                    trace_dict['line'] = dict(
                        color=plot_settings['line']['color'],
                        width=plot_settings['line']['width']
                    )
                
                fig.add_trace(go.Scatter(**trace_dict))

    # Add reference data from DataFrame
    # Observations
    obs_data = data_df[data_df['observations'].notna()].copy()
    if not obs_data.empty:
        obs_settings = get_reference_plot_settings('Observations')
        fig.add_trace(
            go.Scatter(
                x=obs_data.index.tolist(),
                y=obs_data['observations'].tolist(),
                mode='markers',
                name='Observations',
                marker=dict(
                    color=obs_settings['color'],
                    size=obs_settings['marker']['size'],
                    symbol=obs_settings['marker']['symbol'],
                    line=dict(
                        color=obs_settings['marker']['line']['color'],
                        width=obs_settings['marker']['line']['width']
                    )
                ),
                hovertemplate=f"Observation: %{{y:.2f}}{plot_data['var_settings']['units']}<extra></extra>",
                customdata=[[row['forecast_step']] for _, row in obs_data.iterrows()],
            )
        )

    # Analysis
    analysis_data = data_df[data_df['analysis'].notna()].copy()
    if not analysis_data.empty:
        analysis_settings = get_reference_plot_settings('Analysis')
        fig.add_trace(
            go.Scatter(
                x=analysis_data.index.tolist(),
                y=analysis_data['analysis'].tolist(),
                mode='markers',
                name='Analysis',
                marker=dict(
                    color=analysis_settings['color'],
                    size=analysis_settings['marker']['size'],
                    symbol=analysis_settings['marker']['symbol'],
                    line=dict(
                        color=analysis_settings['marker']['line']['color'],
                        width=analysis_settings['marker']['line']['width']
                    )
                ),
                hovertemplate=f"Analysis: %{{y:.2f}}{plot_data['var_settings']['units']}<extra></extra>",
                customdata=[[row['forecast_step']] for _, row in analysis_data.iterrows()],
            )
        )

    # Climatology
    clim_data = data_df[data_df['climatology'].notna()].copy()
    if not clim_data.empty:
        clim_settings = get_reference_plot_settings('Climatology Mean')
        fig.add_trace(
            go.Scatter(
                x=clim_data.index.tolist(),
                y=clim_data['climatology'].tolist(),
                mode='markers',
                name='Climatology Mean',
                marker=dict(
                    color=clim_settings['color'],
                    size=clim_settings['marker']['size'],
                    symbol=clim_settings['marker']['symbol'],
                    line=dict(
                        color=clim_settings['marker']['line']['color'],
                        width=clim_settings['marker']['line']['width']
                    )
                ),
                hovertemplate=f"Climatology Mean: %{{y:.2f}}{plot_data['var_settings']['units']}<extra></extra>",
                customdata=[[row['forecast_step']] for _, row in clim_data.iterrows()],
            )
        )

    if point:
        titre = (f"Forecast Evolution (Valid: {valid_date.strftime('%Y-%m-%d %H:%M')}) "
                f"at {point[0]:.3f}\u00b0N, {point[1]:.3f}\u00b0E")
        ns = plot_data.get('nearest_gridinfo_dict', {}).get('nearest_station')
        if ns is not None:
            titre += (f"<br>Nearest station (stnid: {ns['stnid']}, "
                     f"elev: {ns['elevation']}, "
                     f"lat: {ns['latitude']:.2f}, lon: {ns['longitude']:.2f}, "
                     f"dist: {ns['distance']:.2f} km, "
                     f"value: {ns['value_0']:.2f})")
    else:
        titre = (f"Forecast Evolution (Valid: {valid_date.strftime('%Y-%m-%d %H:%M')}) "
                f"for {area_sub[0]:.4f}°N to {area_sub[2]:.4f}°N, "
                f"{area_sub[1]:.4f}°E to {area_sub[3]:.4f}°E")

    # Get layout settings
    layout_settings = get_layout_settings()

    # Update layout with correct x-axis (largest lead time on the left)
    fig.update_layout(
        title_text=titre,
        showlegend=True,
        height=layout_settings['height'] * 0.8,
        width=layout_settings['width'] * 0.8,
        xaxis=dict(
            ticktext=plot_data['x_labels'][::-1][::2],  # Show every 2nd tick label instead of every 6th
            tickvals=list(range(len(plot_data['steps'])))[::2],  # Show every 2nd tick value instead of every 6th
            **layout_settings['xaxis'],
            title="Forecast Initialization Date/Time"
        ),
        yaxis_title=_build_ylabel(param, widgets_dict)
    )

    # Generate base filename using plot_dir
    if point:
        base_filename = f"{plot_dir}/forecast_evolution_point_{point[0]:.4f}N_{point[1]:.4f}E_{valid_date.strftime('%Y%m%d_%H%M')}_{param}"
    else:
        base_filename = f"{plot_dir}/forecast_evolution_area_{area_sub[0]:.4f}N_{area_sub[1]:.4f}E_{area_sub[2]:.4f}N_{area_sub[3]:.4f}E_{valid_date.strftime('%Y%m%d_%H%M')}_{param}"

    # Export to HTML if requested
    if export_html:
        if html_filename is None:
            # Generate default filename based on current parameters
            if point:
                filename = f"{plot_dir}/forecast_evolution_point_{point[0]:.4f}N_{point[1]:.4f}E_{valid_date.strftime('%Y%m%d_%H%M')}_{param}.html"
            else:
                filename = f"{plot_dir}/forecast_evolution_area_{area_sub[0]:.4f}N_{area_sub[1]:.4f}E_{area_sub[2]:.4f}N_{area_sub[3]:.4f}E_{valid_date.strftime('%Y%m%d_%H%M')}_{param}.html"
        else:
            filename = html_filename
        
        # Export to HTML with include_plotlyjs='cdn' for smaller file size and to ensure it's not blank
        fig.write_html(filename, include_plotlyjs='cdn')
        print(f"Plot exported to: {filename}")

    # Display the plot
    display(fig)


def plot_forecast_evolution_static(plot_data, widgets_dict, plot_dir,
                                    figsize=(22, 9), export_png=False,
                                    png_filename=None):
    """Create a static matplotlib forecast evolution plot using percentile-based
    boxplots (99/90/75/50/25/10/1) for ensemble models and scatter markers for
    deterministic models and reference data.

    Parameters
    ----------
    plot_data : dict
        Output of ``retrieve_and_store_data``.
    widgets_dict : dict
        Widget dictionary (must contain 'config' with model list, etc.).
    plot_dir : str
        Directory for saving plots.
    figsize : tuple
        Figure size ``(width, height)`` in inches.
    export_png : bool
        If True, save the figure as PNG.
    png_filename : str or None
        Override filename for the PNG export.
    """
    global valid_date, param, area_sub, point

    # --- Colour override resolution (same logic as interactive plot) -----------
    _color_overrides = widgets_dict.get('model_colors', {})
    if not _color_overrides:
        _color_overrides = widgets_dict.get('config', {}).get('model_colors', {})

    def _resolve_mpl_color(model_name):
        """Return a matplotlib-compatible colour for *model_name*."""
        def _norm(c):
            # Convert 8-char hex (#RRGGBBAA) → (r,g,b,a) tuple for matplotlib
            if isinstance(c, str) and c.startswith('#') and len(c) == 9:
                r = int(c[1:3], 16) / 255.0
                g = int(c[3:5], 16) / 255.0
                b = int(c[5:7], 16) / 255.0
                a = int(c[7:9], 16) / 255.0
                return (r, g, b, a)
            return c

        if model_name in _color_overrides:
            return _norm(_color_overrides[model_name])
        mn_lower = model_name.lower()
        for key, color in _color_overrides.items():
            if key.lower() in mn_lower or mn_lower in key.lower():
                return _norm(color)
        ps = get_model_plot_settings(model_name)
        return _norm(ps.get('color', '#333333'))

    # --- Prepare DataFrame (sorted earliest first → left) ---------------------
    data_df = plot_data['data_df'].iloc[::-1].reset_index(drop=True).copy()

    date_labels = data_df['forecast_date'].apply(
        lambda d: d.strftime('%Y-%m-%d %H:%M') if hasattr(d, 'strftime') else str(d)
    ).tolist()
    xticks = np.arange(len(data_df))

    # --- Identify ensemble and deterministic models --------------------------
    selected_models = list(widgets_dict['model_widgets'].value)
    ensemble_models = []
    deterministic_models = []
    for mn in selected_models:
        ms = get_model_retrieval_settings(mn)
        if ms.get('ensemble', False):
            ensemble_models.append(mn)
        else:
            deterministic_models.append(mn)

    # --- Slot offsets (one per ensemble model, deterministic share middle) ----
    n_ens = len(ensemble_models)
    if n_ens == 0:
        slot_offsets_ens = []
    elif n_ens == 1:
        slot_offsets_ens = [0.0]
    else:
        # Space ensemble boxes symmetrically
        slot_offsets_ens = np.linspace(-0.18, 0.18, n_ens).tolist()

    # Map ensemble model name → slot index
    ens_slot = {mn: i for i, mn in enumerate(ensemble_models)}

    # --- Percentile box-plot constants ----------------------------------------
    pctl_levels = [99, 90, 75, 50, 25, 10, 1]
    pctl_to_idx = {p: i for i, p in enumerate(pctl_levels)}
    box_width = 0.20
    thin_width = 0.10
    median_bar_width = 0.18

    def _safe_percentiles(vals):
        arr = np.array([float(v) for v in vals if v is not None
                        and not (isinstance(v, float) and np.isnan(v))])
        if len(arr) < 2:
            return [np.nan] * 7
        return list(np.percentile(arr, pctl_levels))

    # --- Create figure --------------------------------------------------------
    fig_mpl, ax = plt.subplots(figsize=figsize)

    # --- Draw ensemble percentile boxes ---------------------------------------
    ens_handles = []
    for model_name in ensemble_models:
        color = _resolve_mpl_color(model_name)
        si = ens_slot[model_name]
        offset = slot_offsets_ens[si]

        pctl_matrix = []
        for _, row in data_df.iterrows():
            vals = row.get(f'{model_name}_ens_area')
            if isinstance(vals, (list, np.ndarray)) and len(vals) > 0:
                pctl_matrix.append(_safe_percentiles(vals))
            else:
                pctl_matrix.append([np.nan] * 7)
        pctl_matrix = np.array(pctl_matrix)
        xpos = xticks + offset

        for idx, x in enumerate(xpos):
            p = pctl_matrix[idx]
            if np.all(np.isnan(p)):
                continue
            # Large box: 25-75
            lo25, hi75 = p[pctl_to_idx[25]], p[pctl_to_idx[75]]
            ax.add_patch(plt.Rectangle(
                (x - box_width / 2, lo25), box_width, hi75 - lo25,
                color=color, alpha=1, lw=0, zorder=5
            ))
            # Thin box: 10-90
            lo10, hi90 = p[pctl_to_idx[10]], p[pctl_to_idx[90]]
            ax.add_patch(plt.Rectangle(
                (x - thin_width / 2, lo10), thin_width, hi90 - lo10,
                color=color, alpha=1, lw=0, zorder=6
            ))
            # Line: 1-99
            ax.plot([x, x], [p[pctl_to_idx[1]], p[pctl_to_idx[99]]],
                    color=color, lw=2.2, solid_capstyle='round', zorder=7)
            # Median horizontal bar
            ax.plot([x - median_bar_width / 2, x + median_bar_width / 2],
                    [p[pctl_to_idx[50]], p[pctl_to_idx[50]]],
                    color='black', lw=3, zorder=8)

        # Legend entries
        _dlabel = _model_display_label(model_name)
        ens_handles.append(plt.Line2D([0], [0], color=color, lw=10, alpha=0.90,
                                      label=f'{_dlabel} (IQR 25-75)'))
        ens_handles.append(plt.Line2D([0], [0], color=color, lw=5, alpha=0.90,
                                      label=f'{_dlabel} (10-90)'))
        ens_handles.append(plt.Line2D([0], [0], color=color, lw=2.2,
                                      label=f'{_dlabel} (1-99)'))

    # --- Map for deterministic marker symbols --------------------------------
    _mpl_symbol_map = {
        'triangle-down': 'v', 'triangle-up': '^', 'diamond': 'D',
        'circle': 'o', 'square': 's', 'star': '*', 'cross': 'x',
        'x': 'X', 'pentagon': 'p', 'hexagon': 'h',
    }

    # --- Draw ensemble mean lines + deterministic scatter --------------------
    scatter_handles = []

    # Ensemble means
    for model_name in ensemble_models:
        color = _resolve_mpl_color(model_name)
        si = ens_slot[model_name]
        offset = slot_offsets_ens[si]

        mean_col = f'{model_name}_mean_area'
        if mean_col in data_df.columns:
            ys = pd.to_numeric(data_df[mean_col], errors='coerce').values
            mask = ~np.isnan(ys)
            if mask.any():
                xs = xticks[mask] + offset
                _dlabel = _model_display_label(model_name)
                h = ax.scatter(xs, ys[mask], color=color, marker='*',
                               edgecolor='black', s=220, zorder=6,
                               label=f'{_dlabel} Mean')
                scatter_handles.append(h)

    # Deterministic models
    for model_name in deterministic_models:
        color = _resolve_mpl_color(model_name)
        ps = get_model_plot_settings(model_name)
        plotly_sym = ps.get('marker', {}).get('symbol', 'circle')
        mpl_marker = _mpl_symbol_map.get(plotly_sym, 'o')
        ms = 220

        col = f'{model_name}_area'
        if col in data_df.columns:
            ys = pd.to_numeric(data_df[col], errors='coerce').values
            mask = ~np.isnan(ys)
            if mask.any():
                _dlabel = _model_display_label(model_name)
                h = ax.scatter(xticks[mask], ys[mask], color=color,
                               marker=mpl_marker, edgecolor='black', s=ms,
                               zorder=10, label=_dlabel)
                scatter_handles.append(h)

    # --- Reference data (observations / analysis / climatology) ---------------
    ref_specs = [
        ('observations', 'Observations', '#000000', 'o', 160),
        ('analysis', 'Analysis', '#ff7f0e', '^', 160),
        ('climatology', 'Climatology Mean', '#7f7f7f', '*', 160),
    ]
    for col, label, color, marker, ms in ref_specs:
        if col in data_df.columns:
            ys = pd.to_numeric(data_df[col], errors='coerce').values
            mask = ~np.isnan(ys)
            if mask.any():
                h = ax.scatter(xticks[mask], ys[mask], color=color, marker=marker,
                               edgecolor='black', s=ms, zorder=10, label=label)
                scatter_handles.append(h)

    # --- Vertical separators --------------------------------------------------
    sep_half = 0.5
    for i in range(len(data_df)):
        ax.axvline(x=xticks[i] - sep_half, color='gray', alpha=0.3,
                   linewidth=0.8, zorder=1)
        ax.axvline(x=xticks[i] + sep_half, color='gray', alpha=0.3,
                   linewidth=0.8, zorder=1)

    # --- Axes formatting ------------------------------------------------------
    ax.set_xticks(xticks)
    ax.set_xticklabels(date_labels, rotation=70, ha='right', fontsize=15)
    ax.set_xlabel('Forecast Initialization Date/Time', fontsize=16)

    ax.set_ylabel(_build_ylabel(param, widgets_dict), fontsize=22)
    ax.tick_params(axis='y', labelsize=15)

    if point:
        titre = (f"Forecast Evolution (Valid: {valid_date.strftime('%Y-%m-%d %H:%M')}) "
                 f"at {point[0]:.3f}\u00b0N, {point[1]:.3f}\u00b0E")
        ns = plot_data.get('nearest_gridinfo_dict', {}).get('nearest_station')
        if ns is not None:
            titre += (f"\nNearest station (stnid: {ns['stnid']}, "
                     f"elev: {ns['elevation']}, "
                     f"lat: {ns['latitude']:.2f}, lon: {ns['longitude']:.2f}, "
                     f"dist: {ns['distance']:.2f} km, "
                     f"value: {ns['value_0']:.2f})")
    else:
        titre = (f"Forecast Evolution (Valid: {valid_date.strftime('%Y-%m-%d %H:%M')}) "
                 f"for {area_sub[0]:.4f}°N to {area_sub[2]:.4f}°N, "
                 f"{area_sub[1]:.4f}°E to {area_sub[3]:.4f}°E")
    ax.set_title(titre, fontsize=18)

    # --- Legend ---------------------------------------------------------------
    legend_entries = list(ens_handles)
    seen = set()
    for h in scatter_handles:
        lbl = h.get_label()
        if lbl not in seen and lbl != '_nolegend_':
            legend_entries.append(h)
            seen.add(lbl)
    # Place legend and map in a right-hand column with matching width
    _right_x = 0.83          # left edge of legend / map column
    _right_w = 0.16          # common width for both

    fig_mpl.subplots_adjust(left=0.06, right=_right_x - 0.01,
                            top=0.93, bottom=0.22)

    # Legend — place it in the upper portion of the right column
    ax.legend(legend_entries,
              [e.get_label() for e in legend_entries],
              loc='upper left',
              bbox_to_anchor=(_right_x, 0.93),
              bbox_transform=fig_mpl.transFigure,
              fontsize=11, frameon=False)

    # Inset map — directly below the legend, same column width
    _map_h = 0.28
    _map_y = 0.12
    map_n, map_w, map_s, map_e = area_sub
    center_lat = (map_n + map_s) / 2.0
    center_lon = (map_w + map_e) / 2.0
    lat_span = max(abs(map_n - map_s), 2.0) * 2.5
    lon_span = max(abs(map_e - map_w), 2.0) * 2.5
    ext_s = center_lat - lat_span / 2
    ext_n = center_lat + lat_span / 2
    ext_w = center_lon - lon_span / 2
    ext_e = center_lon + lon_span / 2

    axins = fig_mpl.add_axes([_right_x, _map_y, _right_w, _map_h],
                             projection=ccrs.Mercator())
    axins.set_extent([ext_w, ext_e, ext_s, ext_n], crs=ccrs.PlateCarree())
    axins.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='black')
    axins.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
    axins.add_feature(cfeature.LAND, facecolor='whitesmoke', zorder=0)

    gl = axins.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                         linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}

    if point:
        axins.plot(point[1], point[0], marker='*', color='red', markersize=14,
                   markeredgecolor='black', markeredgewidth=0.8,
                   transform=ccrs.PlateCarree(), zorder=10)
    else:
        import matplotlib.patches as mpatches
        rect = mpatches.Rectangle(
            (map_w, map_s), map_e - map_w, map_n - map_s,
            linewidth=2.0, edgecolor='red', facecolor='none',
            transform=ccrs.PlateCarree(), zorder=10
        )
        axins.add_patch(rect)

    # --- Export ---------------------------------------------------------------
    if export_png:
        if png_filename is None:
            if point:
                png_filename = (f"{plot_dir}/forecast_evolution_static_point_"
                                f"{point[0]:.4f}N_{point[1]:.4f}E_"
                                f"{valid_date.strftime('%Y%m%d_%H%M')}_{param}.png")
            else:
                png_filename = (f"{plot_dir}/forecast_evolution_static_area_"
                                f"{area_sub[0]:.4f}N_{area_sub[1]:.4f}E_"
                                f"{area_sub[2]:.4f}N_{area_sub[3]:.4f}E_"
                                f"{valid_date.strftime('%Y%m%d_%H%M')}_{param}.png")
        fig_mpl.savefig(png_filename, dpi=150, bbox_inches='tight')
        print(f"Static plot exported to: {png_filename}")

    plt.show()
    plt.close(fig_mpl)
    return fig_mpl


# Only run the interface setup if this file is run directly
if __name__ == '__main__':
    widgets_dict = setup_interface() 