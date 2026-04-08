"""
Forecast Evolution Analysis — UI module.

Interactive widget creation (ipywidgets + ipyleaflet) and event handlers.
"""

from datetime import datetime, timedelta
import ipywidgets as widgets
import pandas as pd
from IPython.display import display
from ipyleaflet import Map, DrawControl, basemaps

from .settings import (
    get_model_settings,
    get_custom_models,
    register_custom_model,
)
from .core import _get_var_settings_safe


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
        value=24,
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
        value='integrate',
        description='Area averaging:',
        style={'description_width': 'initial'},
        layout={'width': '320px'}
    )

    # Function to show/hide accumulation period based on parameter selection
    def on_param_change(change):
        vs = _get_var_settings_safe(change['new'])
        if vs.get('is_accumulated', False):
            acc_period_widget.layout.display = 'flex'
        else:
            acc_period_widget.layout.display = 'none'

    param_widget.observe(on_param_change, names='value')

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
        value=tuple(model_settings['models'].keys()),
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

        predefined = list(get_model_settings()['models'].keys())
        prev_selected = set(model_widgets.value)
        new_options = list(dict.fromkeys(predefined))
        model_widgets.options = tuple(new_options)
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
        fill_color='red',
        fill_opacity=0.2
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
                fill_color='red',
                fill_opacity=0.2
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
                fill_color='red',
                fill_opacity=0.2
            )
            target.widgets_dict['map'].add_layer(rectangle)

        target.widgets_dict['area_widgets']['north'].value = round(north, 3)
        target.widgets_dict['area_widgets']['west'].value = round(west, 3)
        target.widgets_dict['area_widgets']['south'].value = round(south, 3)
        target.widgets_dict['area_widgets']['east'].value = round(east, 3)
        target.widgets_dict['region_widget'].value = 'Custom'


def on_button_clicked(b, widgets_dict):
    """Handle button click event."""
    with widgets_dict['output']:
        widgets_dict['output'].clear_output()

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
        display(df)

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
        }
        print(widgets_dict['config'])


def setup_interface():
    """Set up the complete interface."""
    widgets_dict = create_widgets()

    widgets_dict['draw_control'].widgets_dict = widgets_dict

    widgets_dict['draw_control'].on_draw(handle_draw)

    for coord_widget in widgets_dict['area_widgets'].values():
        coord_widget.observe(lambda change: on_coordinate_change(change, widgets_dict), names='value')

    widgets_dict['region_widget'].observe(lambda change: on_region_change(change, widgets_dict), names='value')

    widgets_dict['generate_button'].on_click(lambda b: on_button_clicked(b, widgets_dict))

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
        widgets_dict['map'],
        widgets_dict['generate_button'],
        widgets_dict['output']
    ]))

    return widgets_dict
