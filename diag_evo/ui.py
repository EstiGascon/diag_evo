"""
Forecast Evolution Analysis — UI module.

Interactive widget creation (ipywidgets + ipyleaflet) and event handlers.
"""

from datetime import datetime, timedelta
import ipywidgets as widgets
import pandas as pd
from IPython.display import display
from ipyleaflet import Map, DrawControl, basemaps, CircleMarker, LayerGroup, Popup

from .settings import (
    get_model_settings,
    get_custom_models,
    register_custom_model,
)
from .core import (
    _get_var_settings_safe,
    save_run_config,
    normalize_config,
    fetch_observations_only,
    STVL_AVAILABLE_PARAMS,
    get_base_var,
)


def create_widgets():
    """Create and return all widgets needed for the interface."""
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
        options=[(f"{h}h", h) for h in [6, 12, 24]],
        value=6,
        description='Accumulation Period:',
        style={'description_width': 'initial'},
        layout={'display': 'none'}
    )

    # Create ensemble members widget
    n_members_widget = widgets.IntSlider(
        value=50,
        min=10,
        max=50,
        step=1,
        description='Ensemble Members:',
        style={'description_width': 'initial'},
        allow_none=False
    )

    # Area averaging mode widget
    area_avg_mode_widget = widgets.Dropdown(
        options=[
            ('Area integral (weighted)', 'integrate'),
            ('Station nearest gridpoints', 'station_nearest'),
        ],
        value='station_nearest',
        description='Area averaging:',
        style={'description_width': 'initial'},
        layout={'width': '320px'}
    )

    # Function to show/hide accumulation period and the STVL "show obs" button
    # based on parameter selection.
    def on_param_change(change):
        vs = _get_var_settings_safe(change['new'])
        if vs.get('is_accumulated', False):
            acc_period_widget.layout.display = 'flex'
        else:
            acc_period_widget.layout.display = 'none'
        # Show "Show station observations" only for STVL-available parameters.
        param = change['new'].strip()
        base = get_base_var(param) if param else ''
        stvl_param = vs.get('obs_param', base)
        if stvl_param in STVL_AVAILABLE_PARAMS or base in STVL_AVAILABLE_PARAMS:
            show_obs_btn.layout.display = 'inline-flex'
        else:
            show_obs_btn.layout.display = 'none'

    param_widget.observe(on_param_change, names='value')

    # "Show station observations" button — visible only for STVL parameters.
    show_obs_btn = widgets.Button(
        description='Show station observations',
        button_style='info',
        icon='map-marker',
        layout={'width': '260px', 'display': 'inline-flex'},
        tooltip='Retrieve STVL observations for the current valid date/time and overlay them on the map.',
    )
    obs_status_w = widgets.HTML(value='')
    obs_colorbar_w = widgets.HTML(value='')

    # Create date picker widget
    date_widget = widgets.DatePicker(
        description='Valid Date:',
        value=datetime.now() - timedelta(days=2),
        style={'description_width': 'initial'}
    )

    # Create time picker widget
    time_widget = widgets.Dropdown(
        options=[(f"{h:02d}:00", h) for h in [0, 6, 12, 18]],
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
        value='Europe',
        description='Standard Region:',
        style={'description_width': 'initial'}
    )

    # Create widgets for area selection (defaults match the 'Europe' region)
    _eu = standard_regions['Europe']
    area_widgets = {
        'north': widgets.FloatText(value=_eu[0], description='North:', style={'description_width': 'initial'}),
        'west': widgets.FloatText(value=_eu[1], description='West:', style={'description_width': 'initial'}),
        'south': widgets.FloatText(value=_eu[2], description='South:', style={'description_width': 'initial'}),
        'east': widgets.FloatText(value=_eu[3], description='East:', style={'description_width': 'initial'})
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

    # Create model selection widget as a dropdown with multiple selection.
    # Labels show MARS keywords so users can identify each model at a
    # glance; values remain plain model names for internal use.
    model_settings = get_model_settings()

    def _model_option_label(name, info):
        """Build a display label like 'IFS Control  (od/fc/oper)'."""
        cls   = info.get('class', '?')
        typ   = info.get('type', '?')
        strm  = info.get('stream', '?')
        parts = [f"{cls}/{typ}/{strm}"]
        if info.get('expver') and str(info['expver']) != '1':
            parts.append(f"e={info['expver']}")
        if info.get('model'):
            parts.append(info['model'])
        if info.get('ensemble'):
            parts.append('ens')
        return f"{name}  ({', '.join(parts)})"

    _model_options = tuple(
        (_model_option_label(n, m), n)
        for n, m in model_settings['models'].items()
    )
    model_widgets = widgets.SelectMultiple(
        options=_model_options,
        value=tuple(n for _, n in _model_options),
        description='Models:',
        style={'description_width': 'initial'},
        layout={'width': 'auto', 'height': '320px'}
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

    custom_status_w = widgets.HTML(value='', layout={'width': '100%'})

    def _toggle_members_vis(change):
        custom_n_members_w.layout.display = 'flex' if change['new'] else 'none'
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

    custom_color_w = widgets.ColorPicker(
        value="#ff00f2",
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

    _added_custom_names = []  # mutable list shared via closure

    def _refresh_custom_models_list():
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

        predefined_settings = get_model_settings()['models']
        prev_selected = set(model_widgets.value)
        new_options = []
        seen = set()
        for n, m in predefined_settings.items():
            if n not in seen:
                new_options.append((_model_option_label(n, m), n))
                seen.add(n)
        model_widgets.options = tuple(new_options)
        model_widgets.value = tuple(n for _, n in new_options if n in prev_selected or n in _added_custom_names)

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

        for row in extra_mars_container.children:
            children = row.children
            k = children[0].value.strip()
            v = children[1].value.strip()
            if k and v:
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

        custom_model_name_w.value = ''
        custom_class_w.value = ''
        custom_expver_w.value = ''
        custom_ensemble_w.value = False
        extra_mars_container.children = []

    add_model_btn.on_click(_on_add_model)

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

    generate_button = widgets.Button(
        description='Forecast Setup',
        button_style='primary'
    )

    output = widgets.Output()

    m = Map(
        center=(51.45, -0.95), #Reading coords
        zoom=4,
        basemap=basemaps.CartoDB.Positron,
    )
    # Rectangle + point only (click-and-drag for area, marker for point).
    # Polygon / polyline / circle / circlemarker are disabled.
    draw_control = DrawControl(
        rectangle={'shapeOptions': {'color': "#ff0000a1", 'weight': 2,
                                     'fill': False, 'fillOpacity': 0}},
        marker={'shapeOptions': {'color': '#ff0000'}},
        polygon={},
        polyline={},
        circle={},
        circlemarker={},
    )
    m.add_control(draw_control)

    # Layer group used for STVL observation markers (cleared/replaced on each refresh).
    obs_layer_group = LayerGroup(layers=[])
    m.add_layer(obs_layer_group)

    return {
        'param_widget': param_widget,
        'levtype_widget': levtype_widget,
        'level_widget': level_widget,
        'acc_period_widget': acc_period_widget,
        'n_members_widget': n_members_widget,
        'area_avg_mode_widget': area_avg_mode_widget,
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
        'show_obs_btn': show_obs_btn,
        'obs_status_w': obs_status_w,
        'obs_colorbar_w': obs_colorbar_w,
        'obs_layer_group': obs_layer_group,
        'output': output,
        'map': m,
        'draw_control': draw_control,
        'standard_regions': standard_regions
    }


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

def update_map_rectangle(widgets_dict):
    """Update the map rectangle when coordinates are manually changed."""
    from ipyleaflet import Rectangle
    for layer in list(widgets_dict['map'].layers)[1:]:
        if hasattr(layer, 'bounds'):
            widgets_dict['map'].remove_layer(layer)

    north = widgets_dict['area_widgets']['north'].value
    west = widgets_dict['area_widgets']['west'].value
    south = widgets_dict['area_widgets']['south'].value
    east = widgets_dict['area_widgets']['east'].value

    bounds = [[south, west], [north, east]]
    rectangle = Rectangle(
        bounds=bounds,
        color='red',
        fill=False,
        weight=2,
    )
    widgets_dict['map'].add_layer(rectangle)

    center_lat = (north + south) / 2
    center_lon = (east + west) / 2
    widgets_dict['map'].center = (center_lat, center_lon)

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
    """Handle coordinate widget changes."""
    update_map_rectangle(widgets_dict)


def on_region_change(change, widgets_dict):
    """Handle standard region selection."""
    selected_region = change['new']
    if selected_region != 'Custom':
        coords = widgets_dict['standard_regions'][selected_region]
        if coords:
            widgets_dict['area_widgets']['north'].value = coords[0]
            widgets_dict['area_widgets']['west'].value = coords[1]
            widgets_dict['area_widgets']['south'].value = coords[2]
            widgets_dict['area_widgets']['east'].value = coords[3]
            update_map_rectangle(widgets_dict)


def handle_draw(target, action, geo_json):
    """Handle drawing on the map."""
    from ipyleaflet import Rectangle
    if action == 'created':
        for layer in list(target.widgets_dict['map'].layers)[1:]:
            target.widgets_dict['map'].remove_layer(layer)

        coords = geo_json['geometry']['coordinates']

        if geo_json['geometry']['type'] == 'Point':
            point = coords
            west = point[0] - 3
            east = point[0] + 3
            south = point[1] - 3
            north = point[1] + 3
            point.reverse()
            # Store point in widgets_dict so config picks it up
            target.widgets_dict['_point'] = list(point)

            bounds = [[south, west], [north, east]]
            rectangle = Rectangle(
                bounds=bounds,
                color='red',
                fill=False,
                weight=2,
            )
            target.widgets_dict['map'].add_layer(rectangle)
        else:
            target.widgets_dict['_point'] = None
            bounds = coords[0]
            west = min(coord[0] for coord in bounds)
            east = max(coord[0] for coord in bounds)
            south = min(coord[1] for coord in bounds)
            north = max(coord[1] for coord in bounds)

            rect_bounds = [[south, west], [north, east]]
            rectangle = Rectangle(
                bounds=rect_bounds,
                color='red',
                fill=False,
                weight=2,
            )
            target.widgets_dict['map'].add_layer(rectangle)

        target.widgets_dict['area_widgets']['north'].value = round(north, 3)
        target.widgets_dict['area_widgets']['west'].value = round(west, 3)
        target.widgets_dict['area_widgets']['south'].value = round(south, 3)
        target.widgets_dict['area_widgets']['east'].value = round(east, 3)
        target.widgets_dict['region_widget'].value = 'Custom'


def on_button_clicked(b, widgets_dict):
    """Handle button click event."""
    # Clear previous output BEFORE re-entering the context so the widget
    # is empty when we start writing to it (avoids duplicated renders
    # when the button is clicked repeatedly in quick succession).
    widgets_dict['output'].clear_output(wait=True)
    with widgets_dict['output']:

        param = widgets_dict['param_widget'].value.strip()
        levtype = widgets_dict['levtype_widget'].value
        level = widgets_dict['level_widget'].value if levtype == 'pl' else None
        valid_date = datetime.combine(
            widgets_dict['date_widget'].value,
            datetime.min.time().replace(hour=widgets_dict['time_widget'].value))
        area_sub = [
            widgets_dict['area_widgets']['north'].value,
            widgets_dict['area_widgets']['west'].value,
            widgets_dict['area_widgets']['south'].value,
            widgets_dict['area_widgets']['east'].value,
        ]
        # Normalise longitudes to [-180, 180] up-front so the printed log
        # and the stored config both show the canonical coordinates.
        from .core import _normalize_lon as _norm_lon
        area_sub = [
            round(float(area_sub[0]), 3), round(_norm_lon(area_sub[1]), 3),
            round(float(area_sub[2]), 3), round(_norm_lon(area_sub[3]), 3),
        ]
        max_days = widgets_dict['max_days_widget'].value
        step_interval = widgets_dict['step_interval_widget'].value
        selected_models = widgets_dict['model_widgets'].value
        point = widgets_dict.get('_point')

        level_str = f" at {level} hPa" if level else ""
        print(f"Setting up forecast dates for {param} (levtype={levtype}{level_str}) with validity date {valid_date}")
        print(f"Area coordinates [N,W,S,E]: {area_sub}")
        print(f"Selected models: {', '.join(selected_models)}")
        print(f"Number of ensemble members: {widgets_dict['n_members_widget'].value}")
        if point:
            print(f"Selected point coordinates: {point}")

        steps = list(range(step_interval, max_days * 24 + step_interval, step_interval))
        forecast_dates = []
        forecast_steps = []
        for step in steps:
            forecast_date = valid_date - timedelta(hours=step)
            forecast_dates.append(forecast_date)
            forecast_steps.append(step)

        df = pd.DataFrame({
            'Forecast Date': forecast_dates,
            'Step (hours)': forecast_steps
        })

        print(f"\nValidity/Observation/Analysis date: {valid_date}")
        print("\nForecast initialization dates and their steps:")
        print(df.to_string(index=False))

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
            'n_members': widgets_dict['n_members_widget'].value,
            'area_avg_mode': widgets_dict['area_avg_mode_widget'].value,
            'acc_period': (
                widgets_dict['acc_period_widget'].value
                if _get_var_settings_safe(param).get('is_accumulated', False)
                else None
            ),
        }
        # Normalise (wraps longitudes, auto-derives area_sub/dates if needed)
        normalize_config(widgets_dict['config'])

        # Save run_config.json under <data_files>/run_configs/ so the user
        # can re-run directly via `python examples/run_from_config.py ...`
        try:
            import os as _os
            repo_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
            base_path = widgets_dict.get('base_path') or _os.path.join(repo_root, 'data_files')
            config_dir = _os.path.join(base_path, 'run_configs')
            area_tag = (
                f"point_{point[0]:.2f}_{point[1]:.2f}" if point
                else f"area_{area_sub[0]:.2f}_{area_sub[1]:.2f}_{area_sub[2]:.2f}_{area_sub[3]:.2f}"
            )
            config_subdir = _os.path.join(
                config_dir,
                f"{param}_{valid_date.strftime('%Y%m%d_%H%M')}_{area_tag}",
            )
            save_run_config(widgets_dict['config'], config_subdir)
            config_path = _os.path.join(config_subdir, 'run_config.json')
            try:
                rel_path = _os.path.relpath(config_path, repo_root)
                # Fall back to absolute path if base_path lies outside the repo
                display_path = rel_path if not rel_path.startswith('..') else config_path
            except ValueError:
                display_path = config_path
            print(
                "\nTo re-run this configuration from the command line:\n"
                f"  cd {repo_root}\n"
                f"  python3 examples/run_from_config.py {display_path}"
            )
        except Exception as e:
            print(f"Warning: could not save run_config.json: {e}")


def _format_obs_value(value, units):
    """Pretty-format a station observation value for popup display."""
    try:
        return f"{float(value):.2f}{(' ' + units) if units else ''}"
    except (TypeError, ValueError):
        return f"{value}{(' ' + units) if units else ''}"


def _cmap_for_param(param):
    """Pick a sensible matplotlib colormap name for a given parameter."""
    base = (get_base_var(param) or param).lower()
    if base in ('2t', '2d', 't'):
        return 'RdYlBu_r'
    if base in ('tp', 'sd'):
        return 'YlGnBu'
    if base in ('msl', 'pres'):
        return 'viridis'
    if base in ('10ff', '10si', '10fg', '100ff'):
        return 'plasma'
    if base in ('tcc', 'lcc', 'mcc', 'hcc'):
        return 'Greys'
    if base == 'vis':
        return 'cividis'
    return 'viridis'


def _build_colorbar_html(values, cmap_name, units, width_px=420, height_px=110):
    """Render a horizontal matplotlib colorbar as an inline base64 PNG."""
    import base64
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib import cm

    vmin = float(min(values))
    vmax = float(max(values))
    if vmin == vmax:
        vmax = vmin + 1.0

    fig, ax = plt.subplots(figsize=(width_px / 100, height_px / 100), dpi=100)
    # Reserve room for tick labels (top of bar) and axis label (bottom).
    fig.subplots_adjust(left=0.06, right=0.97, top=0.58, bottom=0.32)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cb = matplotlib.colorbar.ColorbarBase(
        ax, cmap=cm.get_cmap(cmap_name), norm=norm, orientation='horizontal',
    )
    cb.set_label(units or '', fontsize=11)
    cb.ax.tick_params(labelsize=10)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='white', transparent=False)
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode('ascii')
    return (vmin, vmax,
            f'<img src="data:image/png;base64,{data}" style="vertical-align:middle"/>')


def _refresh_obs_layer(widgets_dict, base_path, obs_df=None):
    """Clear and repopulate the STVL observation marker layer on the map.

    Stations are filtered to the current area bounding box ``[N, W, S, E]``
    read from the UI coordinate widgets, because STVL currently ignores the
    ``area`` retrieval argument and returns the full global station set.
    Markers are coloured by ``value_0`` using a parameter-specific colormap,
    and a horizontal colorbar PNG is shown next to the map.
    """
    import time
    import numpy as np
    from matplotlib import cm
    from matplotlib.colors import Normalize, to_hex

    layer_group = widgets_dict['obs_layer_group']
    # Ensure the layer group is still attached to the map (handle_draw clears
    # all non-basemap layers when the user draws a new shape).
    m = widgets_dict['map']
    if layer_group not in m.layers:
        m.add_layer(layer_group)
    # Clear any existing markers and colorbar
    layer_group.layers = []
    if 'obs_colorbar_w' in widgets_dict:
        widgets_dict['obs_colorbar_w'].value = ''

    if obs_df is None or len(obs_df) == 0:
        widgets_dict['obs_status_w'].value = '<span style="color:#a00">No observations to display.</span>'
        return

    n_total = len(obs_df)
    t0 = time.perf_counter()

    # Filter to the current area bbox (N, W, S, E) from the UI.
    aw = widgets_dict['area_widgets']
    north = aw['north'].value
    west = aw['west'].value
    south = aw['south'].value
    east = aw['east'].value
    if {'latitude', 'longitude'}.issubset(obs_df.columns):
        # Normalise longitudes to [-180, 180] so a panned/wrapped map area
        # (e.g. west=-252.6, east=-201.4 after scrolling left) still matches
        # station coordinates that are stored in [-180, 180].
        def _wrap_lon(x):
            return ((x + 180.0) % 360.0) - 180.0

        lat_col = obs_df['latitude']
        mask = (lat_col <= north) & (lat_col >= south)

        # Only filter by longitude if the bbox does NOT cover the full globe.
        # Wrapping a 360°-wide bbox makes west_w == east_w which would otherwise
        # collapse the mask to a single meridian.
        if abs(east - west) < 360.0:
            west_w = _wrap_lon(west)
            east_w = _wrap_lon(east)
            crosses_antimeridian = (west_w > east_w)
            lon_col = obs_df['longitude'].apply(_wrap_lon)
            if not crosses_antimeridian:
                mask &= (lon_col >= west_w) & (lon_col <= east_w)
            else:
                mask &= (lon_col >= west_w) | (lon_col <= east_w)
        obs_df = obs_df[mask]

    n_in_area = len(obs_df)
    filtered = n_in_area < n_total

    if n_in_area == 0:
        widgets_dict['obs_status_w'].value = (
            '<span style="color:#a00">No observations within the selected area.</span>'
        )
        return

    param = widgets_dict['param_widget'].value.strip()
    vs = _get_var_settings_safe(param)
    units = vs.get('units', '')
    cmap_name = _cmap_for_param(param)

    # Build the value→colour mapping (vectorised) using percentile clipping
    # to avoid extreme outliers compressing the visible colour range.
    values = pd.to_numeric(obs_df.get('value_0'), errors='coerce').to_numpy()
    finite = np.isfinite(values)
    if finite.any():
        vmin = float(np.nanpercentile(values[finite], 2))
        vmax = float(np.nanpercentile(values[finite], 98))
        data_min = float(np.nanmin(values[finite]))
        data_max = float(np.nanmax(values[finite]))
        if vmin == vmax:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0
        data_min, data_max = float('nan'), float('nan')
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(norm(values))
    hex_colors = [to_hex(c) for c in rgba]

    lats = obs_df['latitude'].to_numpy()
    lons = obs_df['longitude'].to_numpy()
    # Shift station longitudes into the same map "copy" as the displayed area
    # so markers appear inside the user's current viewport even when they have
    # scrolled past the antimeridian (e.g. west=-230.801).
    # Each station's lon is in [-180,180]; we add an integer multiple of 360
    # that brings it closest to the bbox centre.
    bbox_lon_centre = (west + east) / 2.0
    shift = np.round((bbox_lon_centre - lons) / 360.0) * 360.0
    lons_disp = lons + shift
    stnids = obs_df.get('stnid', pd.Series([''] * n_in_area)).astype(str).to_numpy()
    elevations = obs_df.get('elevation', pd.Series([''] * n_in_area)).to_numpy()

    markers = []
    for i in range(n_in_area):
        try:
            lat = float(lats[i]); lon = float(lons_disp[i])
        except (TypeError, ValueError):
            continue
        val = values[i]
        val_str = _format_obs_value(val, units)
        color = hex_colors[i] if np.isfinite(val) else '#888888'
        marker = CircleMarker(
            location=(lat, lon),
            radius=5,
            color='#222222',
            fill_color=color,
            fill_opacity=0.9,
            weight=1,
        )
        marker.popup = widgets.HTML(
            value=(
                f"<b>Station {stnids[i]}</b><br>"
                f"Value: <b>{val_str}</b><br>"
                f"Lat: {lat:.3f}, Lon: {lon:.3f}<br>"
                f"Elevation: {elevations[i]} m"
            )
        )
        markers.append(marker)

    layer_group.layers = tuple(markers)

    # Build colorbar HTML.
    if 'obs_colorbar_w' in widgets_dict and finite.any():
        try:
            _, _, html = _build_colorbar_html(
                [vmin, vmax], cmap_name, units,
            )
            widgets_dict['obs_colorbar_w'].value = (
                f'<div style="margin-top:4px">'
                f'<b>{param}</b> &nbsp; '
                f'<span style="font-size:11px;color:#666">'
                f'min/max: {data_min:.2f} / {data_max:.2f} {units} &nbsp;|&nbsp; '
                f'colour-clipped at 2/98 percentile: {vmin:.2f}–{vmax:.2f} {units}'
                f'</span><br>{html}</div>'
            )
        except Exception as e:
            widgets_dict['obs_colorbar_w'].value = (
                f'<span style="color:#a00">Could not render colorbar: {e}</span>'
            )

    elapsed = time.perf_counter() - t0
    if filtered:
        widgets_dict['obs_status_w'].value = (
            f'<span style="color:#060">Showing {len(markers):,} of {n_total:,} stations '
            f'(filtered to [N{north}, W{west}, S{south}, E{east}]). '
            f'Render: {elapsed:.2f}s</span>'
        )
    else:
        widgets_dict['obs_status_w'].value = (
            f'<span style="color:#060">Showing {len(markers):,} stations '
            f'(click a marker for details). Render: {elapsed:.2f}s</span>'
        )


def on_show_obs_clicked(b, widgets_dict, base_path):
    """Handler for the 'Show station observations' button."""
    widgets_dict['obs_status_w'].value = '<span style="color:#666">Retrieving observations…</span>'
    param = widgets_dict['param_widget'].value.strip()
    valid_date = datetime.combine(
        widgets_dict['date_widget'].value,
        datetime.min.time().replace(hour=widgets_dict['time_widget'].value),
    )
    area = [
        widgets_dict['area_widgets']['north'].value,
        widgets_dict['area_widgets']['west'].value,
        widgets_dict['area_widgets']['south'].value,
        widgets_dict['area_widgets']['east'].value,
    ]
    vs = _get_var_settings_safe(param)
    acc_period = (
        widgets_dict['acc_period_widget'].value
        if vs.get('is_accumulated', False) else None
    )

    with widgets_dict['output']:
        obs_df = fetch_observations_only(
            param=param,
            valid_date=valid_date,
            area=area,
            base_path=base_path,
            acc_period=acc_period,
        )

    widgets_dict['obs_df'] = obs_df
    try:
        _refresh_obs_layer(widgets_dict, base_path, obs_df)
    except Exception as e:
        widgets_dict['obs_status_w'].value = (
            f'<span style="color:#a00">Failed to overlay obs: {e}</span>'
        )
        with widgets_dict['output']:
            import traceback as _tb
            _tb.print_exc()


def setup_interface(base_path=None):
    """Set up the complete interface.

    Parameters
    ----------
    base_path : str, optional
        Root directory under which observation/GRIB caches are written.
        Defaults to ``<repo>/data_files`` (one level above the package).
    """
    if base_path is None:
        import os as _os
        base_path = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            'data_files',
        )

    widgets_dict = create_widgets()
    widgets_dict['base_path'] = base_path

    widgets_dict['draw_control'].widgets_dict = widgets_dict

    widgets_dict['draw_control'].on_draw(handle_draw)

    for coord_widget in widgets_dict['area_widgets'].values():
        coord_widget.observe(lambda change: on_coordinate_change(change, widgets_dict), names='value')

    widgets_dict['region_widget'].observe(lambda change: on_region_change(change, widgets_dict), names='value')

    widgets_dict['generate_button'].on_click(lambda b: on_button_clicked(b, widgets_dict))
    widgets_dict['show_obs_btn'].on_click(
        lambda b: on_show_obs_clicked(b, widgets_dict, widgets_dict['base_path'])
    )
    # Initialise the show-obs button visibility based on the default param.
    _initial_param = widgets_dict['param_widget'].value.strip()
    _initial_vs = _get_var_settings_safe(_initial_param)
    _initial_base = get_base_var(_initial_param) if _initial_param else ''
    _initial_stvl = _initial_vs.get('obs_param', _initial_base)
    if _initial_stvl in STVL_AVAILABLE_PARAMS or _initial_base in STVL_AVAILABLE_PARAMS:
        widgets_dict['show_obs_btn'].layout.display = 'inline-flex'
    else:
        widgets_dict['show_obs_btn'].layout.display = 'none'

    print("Please configure your forecast settings:")
    display(widgets.VBox([
        widgets.HBox([widgets_dict['param_widget'],
                     widgets_dict['levtype_widget'],
                     widgets_dict['level_widget'],
                     widgets_dict['acc_period_widget'],
                     widgets_dict['n_members_widget']]),
        widgets.HBox([widgets_dict['date_widget'],
                     widgets_dict['time_widget'],
                     widgets_dict['area_avg_mode_widget']]),
        widgets.HBox([widgets_dict['max_days_widget'],
                     widgets_dict['step_interval_widget']]),
        widgets.HBox([widgets_dict['region_widget']]),
        widgets.HBox([widgets_dict['area_widgets']['north'],
                     widgets_dict['area_widgets']['west'],
                     widgets_dict['area_widgets']['south'],
                     widgets_dict['area_widgets']['east']]),
        widgets.HTML('<hr><b style="font-size:14px">Model Selection</b>'),
        widgets.HBox([widgets_dict['predefined_panel'],
                      widgets_dict['custom_model_panel']]),
        widgets.HBox([widgets_dict['show_obs_btn'],
                      widgets_dict['obs_status_w']]),
        widgets_dict['obs_colorbar_w'],
        widgets_dict['map'],
        widgets_dict['generate_button'],
        widgets_dict['output']
    ]))

    return widgets_dict
