"""
Forecast Evolution Analysis — core module.

Data retrieval (MARS / STVL / Polytope), GRIB file caching,
and domain helpers (area strings, file naming, request sanitisation).
"""

from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2
import numpy as np
import pandas as pd
import metview as mv
import earthkit.data
import json
import os
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from .variables import (
    load_variable_settings, get_base_var, get_level, get_variable_settings,
    convert_to_display, convert_from_display, get_grib_units,
    get_retrieval_settings, process_accumulated_data, get_variable_display_name,
    is_derived, get_derivation_info,
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default timeout (seconds) for a single MARS retrieval.
# Set to 0 or None to disable.
MARS_RETRIEVAL_TIMEOUT = 180


# ---------------------------------------------------------------------------
# Derived variable computation
# ---------------------------------------------------------------------------

def _compute_derived(method, component_fields):
    """Apply a derivation method to component fieldsets.

    Parameters
    ----------
    method : str
        Derivation method name (e.g. ``'wind_speed'``).
    component_fields : list of Metview Fieldsets
        One fieldset per component, in the order defined in variable_settings.

    Returns
    -------
    Metview Fieldset with the derived result.
    """
    if method == 'wind_speed':
        u, v = component_fields
        return mv.sqrt(u * u + v * v)
    raise ValueError(f"Unknown derivation method: {method}")


def _fix_derived_metadata(data, derivation_info):
    """Set the correct paramId on derived fields so GRIB headers are accurate."""
    output_pid = derivation_info.get('output_paramId')
    if output_pid is not None:
        data = mv.grib_set_long(data, ['paramId', output_pid])
    return data


# ---------------------------------------------------------------------------
# MARS retrieval with timeout
# ---------------------------------------------------------------------------

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

    # NOTE: we deliberately do NOT use ``with ThreadPoolExecutor(...)`` here.
    # The context manager calls ``shutdown(wait=True)`` on exit, which would
    # block until the underlying MARS call returns — defeating the timeout.
    # Instead, shut the pool down without waiting so ``TimeoutError`` can
    # actually propagate to the caller. The orphaned worker thread will
    # finish in the background; the MARS subprocess itself cannot be
    # cancelled from Python, but the user regains control immediately.
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(earthkit.data.from_source, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except FuturesTimeoutError:
        raise TimeoutError(
            f"MARS retrieval timed out after {timeout} seconds. "
            "The data source may be temporarily unavailable."
        )
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------------
# Pure-utility helpers
# ---------------------------------------------------------------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    """Return the great-circle distance in km between two points."""
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def _extract_value(data, point, area_sub):
    """Return nearest gridpoint value (point mode) or area integral."""
    if point:
        return mv.nearest_gridpoint(data, point)
    return mv.integrate(data, area_sub)


def _expand_area_for_plotting(area, radius=4.0):
    """Return *area* expanded by *radius* degrees, clamped to valid bounds."""
    north, west, south, east = area
    return [
        round(min(90.0, north + radius), 3),
        round(max(-180.0, west - radius), 3),
        round(max(-90.0, south - radius), 3),
        round(min(180.0, east + radius), 3),
    ]


def _normalize_lon(lon):
    """Wrap a longitude to the canonical [-180, 180] range.

    Handles values produced by ipyleaflet when the user scrolls the map
    horizontally past the dateline (e.g. 486.869 → 126.869).
    """
    return ((float(lon) + 180.0) % 360.0) - 180.0


def normalize_config(config):
    """Normalise / auto-fill a user-provided config dict in-place.

    - Wrap out-of-range longitudes in ``area_sub`` and ``point`` to [-180, 180].
    - If ``point`` is given and ``area_sub`` is missing/empty/None, auto-derive
      ``area_sub`` as a ±3° bounding box around the point.
    - If ``forecast_dates``/``forecast_steps`` are missing, derive them from
      ``valid_date``, ``max_days`` and ``step_interval``.
    - Set ``acc_period`` to ``None`` for non-accumulated variables so it is
      never consulted downstream for such variables.

    Returns the same ``config`` dict (mutated) for convenience.
    """
    # --- Longitude wrapping ---
    area_sub = config.get('area_sub')
    if area_sub:
        n, w, s, e = area_sub
        config['area_sub'] = [
            round(float(n), 3), round(_normalize_lon(w), 3),
            round(float(s), 3), round(_normalize_lon(e), 3),
        ]

    point = config.get('point')
    if point:
        lat, lon = point[0], point[1]
        config['point'] = [round(float(lat), 3), round(_normalize_lon(lon), 3)]

    # --- Auto-derive area_sub from point if missing ---
    if config.get('point') and not config.get('area_sub'):
        lat, lon = config['point']
        config['area_sub'] = [
            round(lat + 3.0, 3), round(_normalize_lon(lon - 3.0), 3),
            round(lat - 3.0, 3), round(_normalize_lon(lon + 3.0), 3),
        ]

    # --- Auto-derive forecast_dates / forecast_steps ---
    need_dates = not config.get('forecast_dates') or not config.get('forecast_steps')
    if need_dates and config.get('valid_date') is not None \
            and config.get('max_days') is not None \
            and config.get('step_interval') is not None:
        valid_date = config['valid_date']
        max_days = int(config['max_days'])
        step_interval = int(config['step_interval'])
        steps = list(range(step_interval, max_days * 24 + step_interval, step_interval))
        config['forecast_steps'] = steps
        config['forecast_dates'] = [valid_date - timedelta(hours=s) for s in steps]

    # --- acc_period only for accumulated variables ---
    param = config.get('param')
    if param is not None:
        vs = _get_var_settings_safe(param)
        if not vs.get('is_accumulated', False):
            config['acc_period'] = None

    return config


def _cached_retrieve(filepath, build_request_fn):
    """Return data from cache or retrieve, save, and re-read.

    Parameters
    ----------
    filepath : str
        Path to the GRIB file on disk.
    build_request_fn : callable
        A zero-argument callable that performs the retrieval and returns data.

    Returns
    -------
    Metview Fieldset read from *filepath*.
    """
    if os.path.exists(filepath):
        print(f"File exists, reading: {os.path.basename(filepath)}")
        return mv.read(filepath)
    data = build_request_fn()
    data.save(filepath)
    return mv.read(filepath)


def _col(model_name, suffix):
    """Build a DataFrame column name for *model_name*."""
    return f'{model_name}_{suffix}'


# ---------------------------------------------------------------------------
# MARS request helpers
# ---------------------------------------------------------------------------

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

    req_type = str(req.get('type', '')).lower()
    if req_type != 'pf' and 'number' in req:
        del req['number']

    req_levtype = str(req.get('levtype', '')).lower()
    if req_levtype != 'pl' and 'levelist' in req:
        del req['levelist']

    req_class = str(req.get('class', '')).lower()
    if req_class != 'd1':
        req.pop('dataset', None)
        req.pop('address', None)

    req.pop('ensemble', None)

    return req


# ---------------------------------------------------------------------------
# Directory / filename helpers
# ---------------------------------------------------------------------------

_AREA_DIR_RE = re.compile(
    r'^(?P<var>.+)_N(?P<north>[^_]+)_W(?P<west>[^_]+)_S(?P<south>[^_]+)_E(?P<east>[^_]+)_(?P<date>\d{8})_(?P<time>\d{4})$'
)


def _parse_area_from_dirname(dirname):
    """Parse [N, W, S, E] from a data directory name."""
    m = _AREA_DIR_RE.match(dirname)
    if m:
        try:
            area = [float(m.group('north')), float(m.group('west')),
                    float(m.group('south')), float(m.group('east'))]
            return m.group('var'), area, m.group('date'), m.group('time')
        except ValueError:
            return None
    old_re = re.match(
        r'^(?P<var>.+)_N(?P<north>[^_]+)_W(?P<west>[^_]+)_S(?P<south>[^_]+)_E(?P<east>[^_]+)_(?P<date>\d{8})$',
        dirname
    )
    if old_re:
        try:
            area = [float(old_re.group('north')), float(old_re.group('west')),
                    float(old_re.group('south')), float(old_re.group('east'))]
            return old_re.group('var'), area, old_re.group('date'), '0000'
        except ValueError:
            return None
    return None


def setup_data_directories(base_path, var, area_str, date_str, time_str='0000'):
    """Create and return paths for data directories."""
    dir_name = f"{var}_{area_str}_{date_str}_{time_str}"
    grib_dir = os.path.join(base_path, dir_name, "grib_files")
    obs_dir = os.path.join(base_path, dir_name, "obs_files")
    plot_dir = os.path.join(base_path, dir_name, "plot_files")

    os.makedirs(grib_dir, exist_ok=True)
    os.makedirs(obs_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    return grib_dir, obs_dir, plot_dir


def get_area_string(area):
    """Convert area coordinates to string for directory naming."""
    return f"N{area[0]}_W{area[1]}_S{area[2]}_E{area[3]}"


def _safe_label(name):
    """Replace spaces with underscores for file/directory names."""
    return str(name).replace(' ', '_')


def _mars_file_tag(model_name):
    """Build a MARS-detail file tag for *model_name*."""
    try:
        ms = get_model_retrieval_settings(model_name)
        parts = [
            str(ms.get('class', '')),
            str(ms.get('stream', '')),
            str(ms.get('type', '')),
            str(ms.get('expver', '')),
        ]
        tag = '_'.join(p for p in parts if p)
        if tag:
            return tag
    except (ValueError, KeyError):
        pass
    return _safe_label(model_name)


def _model_grib_filename(model_name, param, levtype, level, fc_date, step,
                         acc_period=None):
    """Build a standardised GRIB filename for a model forecast."""
    safe = _safe_label(model_name)
    tag = _mars_file_tag(model_name)
    date_part = fc_date.strftime('%Y%m%d')
    hour_part = f"{fc_date.hour:02d}00"
    parts = [safe, param, tag, levtype]
    if level is not None:
        parts.append(f"L{level}")
    parts.extend([date_part, hour_part, f"step{step}"])
    if acc_period is not None:
        parts.append(f"acc{acc_period}h")
    return '_'.join(str(p) for p in parts) + '.grib'


def _reference_grib_filename(ref_type, param, ref_settings, levtype, level,
                              date_str, hour_str):
    """Build a standardised GRIB filename for analysis/climatology."""
    parts_list = [
        str(ref_settings.get('class', '')),
        str(ref_settings.get('stream', '')),
        str(ref_settings.get('type', '')),
        str(ref_settings.get('expver', '')),
    ]
    tag = '_'.join(p for p in parts_list if p)
    parts = [ref_type, param, tag, levtype]
    if level is not None:
        parts.append(f"L{level}")
    parts.extend([date_str, hour_str])
    return '_'.join(str(p) for p in parts) + '.grib'


# ---------------------------------------------------------------------------
# Variable / display helpers
# ---------------------------------------------------------------------------

def _get_var_settings_safe(param):
    """Try to look up variable settings from JSON; return a minimal default if not found."""
    try:
        return get_variable_settings(param)
    except (ValueError, KeyError):
        pass
    try:
        base = get_base_var(param)
        return get_variable_settings(base)
    except (ValueError, KeyError):
        pass
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
    """Return 'ModelName (type|stream|class|expver)' for legend labels."""
    try:
        ms = get_model_retrieval_settings(model_name)
        parts = [
            str(ms.get('type', '')),
            str(ms.get('stream', '')),
            str(ms.get('class', '')),
            str(ms.get('expver', '')),
        ]
        suffix = '|'.join(p for p in parts if p)
        if suffix:
            return f"{model_name} ({suffix})"
    except (ValueError, KeyError):
        pass
    return model_name


def _build_ylabel(param, widgets_dict):
    """Build a y-axis label including display name, units, and accumulation period."""
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


def _should_keep_grid(levtype, model_name=None):
    """Return True if the ``grid`` key should be kept in a MARS request."""
    if levtype == 'pl':
        return True
    if model_name is not None and model_name.upper().startswith('DE-'):
        return True
    return False


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


# ---------------------------------------------------------------------------
# Standalone STVL retrieval (used by the UI "Show station observations" button)
# ---------------------------------------------------------------------------

# STVL parameters known to be available as station observations.
STVL_AVAILABLE_PARAMS = ['10ff', '2d', '2t', 'msl', 'pres', 'sd', 'tcc', 'vis', 'tp', '10fg', '10fg6']


def fetch_observations_only(param, valid_date, area, base_path, acc_period=None):
    """Retrieve STVL station observations for a single valid date/time.

    Lightweight helper used by the UI to overlay station observations on the
    map *before* the user has finalised their area / point selection.

    Parameters
    ----------
    param : str
        Variable shortname (must be in :data:`STVL_AVAILABLE_PARAMS` or have an
        ``obs_param`` mapping in ``variable_settings.json``).
    valid_date : datetime
        Observation valid date and time.
    area : list of float
        ``[N, W, S, E]`` retrieval bounding box.
    base_path : str
        Root directory for the local ``data_files`` cache.
    acc_period : int or None
        Accumulation period in hours (only used for accumulated variables
        such as ``tp``).

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with columns including ``stnid``, ``latitude``, ``longitude``,
        ``elevation`` and ``value_0`` (in display units). ``None`` if retrieval
        fails or the variable is not available in STVL.
    """
    var_settings = _get_var_settings_safe(param)
    if var_settings.get('levtype', 'sfc') != 'sfc':
        print(f"No observations available for pressure-level variable '{param}'")
        return None

    base = get_base_var(param)
    stvl_param = var_settings.get('obs_param', base)
    if stvl_param not in STVL_AVAILABLE_PARAMS and base not in STVL_AVAILABLE_PARAMS:
        print(f"Parameter '{param}' is not in the STVL available list "
              f"({', '.join(STVL_AVAILABLE_PARAMS)}).")
        return None

    area_str = get_area_string(area)
    date_str = valid_date.strftime("%Y%m%d")
    time_str = f"{valid_date.hour:02d}00"
    _, obs_dir, _ = setup_data_directories(base_path, param, area_str, date_str, time_str)

    obs_file = os.path.join(obs_dir, f"STVL_{param}_{date_str}_{time_str}.grib")
    obs_csv = os.path.join(obs_dir, f"STVL_{param}_{date_str}_{time_str}.csv")

    try:
        if os.path.exists(obs_file) and os.path.exists(obs_csv):
            print(f"Loading existing observation files from {obs_file}")
            obs_df = pd.read_csv(obs_csv)
        else:
            print(f"Retrieving STVL observations for {param} "
                  f"(valid {date_str} {time_str}, area {area})...")
            # Period: explicit `obs_period` in settings wins; otherwise use
            # acc_period for accumulated vars, else None.
            if 'obs_period' in var_settings:
                period = str(var_settings['obs_period'])
            elif var_settings.get('is_accumulated', False) and acc_period is not None:
                period = str(acc_period)
            else:
                period = None
            # Custom source (e.g. 'wgust' for 10fg) overrides synop/hdobs merge.
            custom_source = var_settings.get('obs_source')
            if custom_source:
                obs = mv.stvl(
                    parameter=stvl_param,
                    dates=date_str,
                    times=valid_date.hour,
                    sources=custom_source,
                    area=area,
                    period=period,
                )
            else:
                obs1 = mv.stvl(
                    parameter=stvl_param,
                    dates=date_str,
                    times=valid_date.hour,
                    sources="synop",
                    area=area,
                    period=period,
                )
                obs2 = mv.stvl(
                    parameter=stvl_param,
                    dates=date_str,
                    times=valid_date.hour,
                    sources="hdobs",
                    area=area,
                    period=period,
                )
                obs = mv.merge(obs1, obs2)
            obs = mv.remove_duplicates(obs)
            mv.write(obs_file, obs)
            obs_df = obs.to_dataframe()
            obs_df.to_csv(obs_csv, index=False)

        # Convert temperature obs from K to display units (°C).
        if base in ('2t', '2d', 't') and 'value_0' in obs_df.columns:
            obs_df['value_0'] = _safe_convert_to_display(
                obs_df['value_0'].to_numpy(), param, grib_units='K'
            )
        print(f"Retrieved {len(obs_df)} stations.")
        return obs_df

    except Exception as e:
        print(f"Warning: could not retrieve STVL observations for '{param}': {e}")
        return None


# ---------------------------------------------------------------------------
# Sub-functions extracted from retrieve_and_store_data
# ---------------------------------------------------------------------------

def _retrieve_observations(param, var_settings, levtype, valid_date, area_sub,
                           retrieval_area, point, obs_dir, acc_period_value,
                           nearest_gridinfo_dict):
    """Retrieve and process STVL observations.

    Returns
    -------
    tuple of (obs_fs_area, obs_geopoints)
        obs_fs_area : float or None
            Observation value (converted to display units).
        obs_geopoints : Metview Geopoints or None
            The raw observation geopoints (for station-nearest-gridpoint extraction).
    """
    if levtype != 'sfc':
        print(f"No observations available for pressure level variable {param}")
        return None, None

    try:
        date_str = valid_date.strftime("%Y%m%d")
        time_str = f"{valid_date.hour:02d}00"
        obs_file = os.path.join(obs_dir, f"STVL_{param}_{date_str}_{time_str}.grib")
        obs_csv = os.path.join(obs_dir, f"STVL_{param}_{date_str}_{time_str}.csv")

        if os.path.exists(obs_file) and os.path.exists(obs_csv):
            print(f"Loading existing observation files from {obs_file}")
            obs_df = pd.read_csv(obs_csv)
            obs_geopoints = mv.read(obs_file)
        else:
            print("Retrieving new observation data...")
            # Period: explicit `obs_period` in settings wins; otherwise use
            # acc_period for accumulated vars, else None.
            if 'obs_period' in var_settings:
                period = str(var_settings['obs_period'])
            elif var_settings.get('is_accumulated', False):
                period = str(acc_period_value)
            else:
                period = None
            stvl_param = var_settings.get('obs_param', get_base_var(param))
            custom_source = var_settings.get('obs_source')
            if custom_source:
                obs = mv.stvl(
                    parameter=stvl_param,
                    dates=date_str,
                    times=valid_date.hour,
                    sources=custom_source,
                    area=retrieval_area,
                    period=period,
                )
            else:
                obs1 = mv.stvl(
                    parameter=stvl_param,
                    dates=date_str,
                    times=valid_date.hour,
                    sources="synop",
                    area=retrieval_area,
                    period=period,
                )
                obs2 = mv.stvl(
                    parameter=stvl_param,
                    dates=date_str,
                    times=valid_date.hour,
                    sources="hdobs",
                    area=retrieval_area,
                    period=period,
                )
                obs = mv.merge(obs1, obs2)
            obs = mv.remove_duplicates(obs)
            mv.write(obs_file, obs)
            obs_df = obs.to_dataframe()
            obs_df.to_csv(obs_csv, index=False)
            obs_geopoints = obs

        # Process observations
        if point:
            target_lat, target_lon = point[0], point[1]
            obs_df['distance'] = obs_df.apply(
                lambda row: haversine_distance(row['latitude'], row['longitude'],
                                               target_lat, target_lon), axis=1)
            nearest_station = obs_df.loc[obs_df['distance'].idxmin()]
            print(f"Nearest station (stnid: {nearest_station['stnid']}, "
                  f"elevation: {nearest_station['elevation']}, "
                  f"lat: {nearest_station['latitude']}, "
                  f"lon: {nearest_station['longitude']}, "
                  f"distance: {nearest_station['distance']:.2f} km, "
                  f"value: {nearest_station['value_0']:.2f})")
            obs_fs_area = nearest_station['value_0']
        else:
            center_lat = (area_sub[0] + area_sub[2]) / 2
            center_lon = (area_sub[1] + area_sub[3]) / 2
            obs_df['distance'] = obs_df.apply(
                lambda row: haversine_distance(row['latitude'], row['longitude'],
                                               center_lat, center_lon), axis=1)
            expansion = 0
            stations_found = False
            while not stations_found and expansion <= 2:
                expanded_area = [
                    area_sub[0] + expansion, area_sub[1] - expansion,
                    area_sub[2] - expansion, area_sub[3] + expansion,
                ]
                filtered_df = obs_df[
                    (obs_df['latitude'] <= expanded_area[0]) &
                    (obs_df['longitude'] >= expanded_area[1]) &
                    (obs_df['latitude'] >= expanded_area[2]) &
                    (obs_df['longitude'] <= expanded_area[3])
                ]
                if len(filtered_df) > 0:
                    stations_found = True
                    obs_df = filtered_df.reset_index(drop=True)
                    nearest_station = obs_df.loc[obs_df['distance'].idxmin()]
                    print(f"Number of stations found: {len(filtered_df)}")
                    print(f"Area expanded by {expansion:.3f} degrees")
                    print(f"Nearest station to original area center:")
                    print(f"Station ID: {nearest_station['stnid']}")
                    print(f"Distance from center: {nearest_station['distance']:.2f} km")
                else:
                    expansion += 0.25
                obs_fs_area = (obs_df['value_0'].values).mean()

            if not stations_found:
                nearest_station = obs_df.loc[obs_df['distance'].idxmin()]
                print(f"No stations found within expanded area. Using nearest station overall:")
                print(f"Station ID: {nearest_station['stnid']}")
                print(f"Distance from center: {nearest_station['distance']:.2f} km")
                obs_fs_area = (nearest_station['value_0']).mean()

        # STVL returns temperature obs in Kelvin — convert to display units.
        # Other variables (tp, msl, …) are already in display-compatible units.
        base = get_base_var(param)
        if base in ('2t', '2d', 't'):
            obs_fs_area = _safe_convert_to_display(obs_fs_area, param, grib_units='K')
        if point:
            print(f'OBS value: {obs_fs_area:.2f} (nearest station at {nearest_station["distance"]:.2f} km)')
        else:
            print(f'OBS value: {obs_fs_area:.2f} ({len(obs_df)} stations)')
        print(f"Station ID: {nearest_station['stnid']}")
        print(f"Station elevation: {nearest_station['elevation']} m")
        print(f"Station latitude: {nearest_station['latitude']}°")
        print(f"Station longitude: {nearest_station['longitude']}°")
        nearest_gridinfo_dict['nearest_station'] = nearest_station
        return obs_fs_area, obs_geopoints

    except Exception as e:
        print(f"Warning: Could not retrieve STVL observations for '{param}': {str(e)}\n"
              f"(Not all parameters are available in STVL — this is expected for some variables.)")
        return None, None


def _retrieve_reference_field(ref_type, ref_settings_fn, param, levtype, level,
                               valid_date, area_sub, retrieval_area, point,
                               grib_dir, nearest_gridinfo_dict,
                               obs_geopoints=None):
    """Retrieve a reference field (analysis or climatology).

    Parameters
    ----------
    obs_geopoints : Metview Geopoints or None
        When provided (station-nearest mode, area only), the field value is
        extracted at observation station locations and averaged.

    Returns the scalar value or None on failure.
    """
    try:
        date_str = valid_date.strftime("%Y%m%d")
        time_str = f"{valid_date.hour:02d}00"
        ref_settings = ref_settings_fn()
        filepath = os.path.join(grib_dir, _reference_grib_filename(
            ref_type, param, ref_settings, levtype, level, date_str, time_str))

        def _do_retrieve():
            print(f"Retrieving {ref_type.lower()} from MARS...")
            request = _build_base_request(
                param, levtype, level, valid_date, valid_date.hour,
                retrieval_area)
            request.update(ref_settings)
            if not _should_keep_grid(levtype):
                request.pop('grid', None)
            return _retrieve_with_timeout(MARS_RETRIEVAL_TIMEOUT, "mars", request)

        if is_derived(param):
            deriv = get_derivation_info(param)
            if os.path.exists(filepath):
                print(f"File exists, reading: {os.path.basename(filepath)}")
                data = mv.read(filepath)
            else:
                component_data = []
                for comp_param in deriv['components']:
                    comp_filepath = os.path.join(grib_dir, _reference_grib_filename(
                        ref_type, comp_param, ref_settings, levtype, level, date_str, time_str))

                    def _do_retrieve_comp(cp=comp_param):
                        print(f"Retrieving {ref_type.lower()} component '{cp}' from MARS...")
                        request = _build_base_request(
                            cp, levtype, level, valid_date, valid_date.hour,
                            retrieval_area)
                        request.update(ref_settings)
                        if not _should_keep_grid(levtype):
                            request.pop('grid', None)
                        return _retrieve_with_timeout(MARS_RETRIEVAL_TIMEOUT, "mars", request)

                    comp_data = _cached_retrieve(comp_filepath, _do_retrieve_comp)
                    component_data.append(comp_data)
                data = _compute_derived(deriv['method'], component_data)
                data = _fix_derived_metadata(data, deriv)
                mv.write(filepath, data)
                data = mv.read(filepath)
        else:
            data = _cached_retrieve(filepath, _do_retrieve)

        data = _safe_convert_to_display(data, param, get_grib_units(data))

        if point:
            info = mv.nearest_gridpoint_info(data, point)[0]
            print(f"The nearest grid point is at {info['distance']}km from the selected point")
            nearest_gridinfo_dict[f'nearest_{ref_type.lower()}'] = info
            return mv.nearest_gridpoint(data, point)
        elif obs_geopoints is not None:
            gpt_vals = mv.nearest_gridpoint(data, obs_geopoints)
            vals = mv.values(gpt_vals)
            valid = vals[~np.isnan(vals)] if hasattr(vals, '__len__') else vals
            return float(np.mean(valid)) if len(valid) > 0 else None
        else:
            return mv.integrate(data, area_sub)

    except Exception as e:
        print(f"Error retrieving {ref_type.lower()}: {str(e)}")
        return None


def _retrieve_single_model_step(model_name, param, var_settings, levtype, level,
                                 fc_date, step, area_sub, retrieval_area, point,
                                 grib_dir, acc_period_value, n_members,
                                 nearest_gridinfo_dict, obs_geopoints=None):
    """Retrieve and process one model × one step.

    Returns
    -------
    dict with keys matching DataFrame columns, or None on failure.
    """
    if model_name in ['DE-LUMI', 'DE-ATOS']:
        if fc_date.hour != 0:
            print(f"Skipping {model_name} - only available for 00:00:00 initialization times")
            return None
        if step > 121:
            print(f"Skipping {model_name} - not available for steps > 120h")
            return None

    acc_period = acc_period_value if var_settings.get('is_accumulated', False) else None
    model_file = os.path.join(grib_dir, _model_grib_filename(
        model_name, param, levtype, level, fc_date, step, acc_period=acc_period))

    model_ret = get_model_retrieval_settings(model_name)
    is_ensemble = model_ret.get('ensemble', False)

    def _do_retrieve(retrieve_param=param):
        """Build and execute a MARS request for *retrieve_param*."""
        print(f"Retrieving {model_name} '{retrieve_param}' from MARS for step {step}...")
        request = _build_base_request(
            retrieve_param, levtype, level, fc_date, fc_date.hour,
            retrieval_area)
        request.update(model_ret)

        if not _should_keep_grid(levtype, model_name):
            request.pop('grid', None)

        if model_name == "DE-LUMI":
            return _retrieve_de_lumi(request, var_settings, acc_period_value, step)

        # --- Build step/number/expver ---
        if var_settings.get('is_accumulated', False):
            step_start = max(0, step - acc_period_value)
            request["step"] = f"{step_start}/{step}"
        else:
            request["step"] = step

        if is_ensemble and str(request.get('type', '')).lower() == 'pf':
            request["number"] = [1, "TO", n_members]

        if model_name == "AIFS ENS":
            if fc_date <= datetime(2025, 7, 1, 0, 0):
                request['expver'] = '103'
            else:
                request['expver'] = '1'

        request = sanitize_mars_request(request)
        print(f"[{model_name} request] {request}")
        return _retrieve_with_timeout(MARS_RETRIEVAL_TIMEOUT, "mars", request)

    if is_derived(param):
        deriv = get_derivation_info(param)
        # Check if derived result already cached
        if os.path.exists(model_file):
            print(f"File exists, reading: {os.path.basename(model_file)}")
            data = mv.read(model_file)
        else:
            # Retrieve each component (each cached separately)
            component_data = []
            for comp_param in deriv['components']:
                comp_file = os.path.join(grib_dir, _model_grib_filename(
                    model_name, comp_param, levtype, level, fc_date, step,
                    acc_period=acc_period))
                comp_data = _cached_retrieve(
                    comp_file, lambda cp=comp_param: _do_retrieve(cp))
                component_data.append(comp_data)
            # Compute derived field (element-wise: works on all ensemble members at once)
            data = _compute_derived(deriv['method'], component_data)
            data = _fix_derived_metadata(data, deriv)
            mv.write(model_file, data)
            data = mv.read(model_file)
    else:
        data = _cached_retrieve(model_file, _do_retrieve)

    # --- Post-processing: de-accumulation / field selection / unit conversion ---
    data = _process_model_data(data, model_name, param, var_settings, levtype, level,
                                is_ensemble, n_members)
    if data is None:
        return None

    # --- Extract scalar values for the DataFrame ---
    return _store_model_data(data, model_name, is_ensemble, point, area_sub,
                             nearest_gridinfo_dict, obs_geopoints=obs_geopoints)


def _retrieve_de_lumi(request, var_settings, acc_period_value, step):
    """Handle DE-LUMI specific retrieval via Polytope."""
    if var_settings.get('is_accumulated', False):
        step_start = max(0, step - acc_period_value)
        if step_start == 0:
            request["step"] = f'{int(step)-1}-{int(step)}'
        else:
            request["step"] = f'{step_start-1}-{step_start}/{int(step)-1}-{step}'
    else:
        request["step"] = step
    lumi_address = request.pop("address", None)
    request = sanitize_mars_request(request)
    print(f"[DE-LUMI request] {request}")
    return _retrieve_with_timeout(
        MARS_RETRIEVAL_TIMEOUT,
        "polytope", "ecmwf-destination-earth",
        request, address=lumi_address, stream=False,
    )


def _process_model_data(data, model_name, param, var_settings, levtype, level,
                         is_ensemble, n_members):
    """De-accumulate / select fields / convert units. Returns processed data or None."""
    if var_settings.get('is_accumulated', False):
        print(f"Processing accumulated data for {model_name}")
        if is_ensemble:
            deacc = []
            for i in range(n_members):
                member_data = data.select(shortName=get_base_var(param), number=i + 1)
                if len(member_data) >= 2:
                    deacc.append(member_data[1] - member_data[0])
            if deacc:
                data = mv.merge(*deacc)
                print(f"Successfully processed {len(deacc)} ensemble members")
            else:
                print("Warning: No valid ensemble members found")
                return None
        else:
            print(f"Data fields: {len(data)}")
            if len(data) >= 2:
                data = data[1] - data[0]
            else:
                data = data[0]
                print("Warning: Only one field found, using as is")
        return _safe_convert_to_display(data, param)
    else:
        print(f"Processing non-accumulated data for {model_name}")
        if is_ensemble:
            if levtype == 'pl' and level is not None:
                data = data.select(shortName=get_base_var(param), levelist=level)
            else:
                data = data.select(shortName=get_base_var(param))
            if data is not None:
                data = _safe_convert_to_display(data, param)
                print(f"Successfully processed ensemble data with {len(data)} members")
            else:
                print("Warning: No valid ensemble data found")
                return None
        else:
            if data is not None:
                data = _safe_convert_to_display(data, param)
            else:
                print("Warning: No data found for deterministic model")
                return None
        return data


def _store_model_data(data, model_name, is_ensemble, point, area_sub,
                      nearest_gridinfo_dict, obs_geopoints=None):
    """Extract scalar values from processed data for DataFrame storage.

    Parameters
    ----------
    obs_geopoints : Metview Geopoints or None
        When provided (station-nearest mode, area only), model values are
        extracted at observation station locations via
        ``mv.nearest_gridpoint(field, obs_geopoints)`` and then averaged,
        instead of using ``mv.integrate(field, area_sub)``.

    Returns a dict of column_suffix → value pairs.
    """
    result = {}

    if is_ensemble:
        result['ensemble'] = data
        if point:
            data_point = mv.nearest_gridpoint(data, point)
            valid_vals = [v for v in data_point if v is not None]
            if not valid_vals:
                print(f"Warning: nearest_gridpoint returned no valid values for {model_name}")
                return None
            nearest_gridinfo_dict[model_name + '_nearest'] = mv.nearest_gridpoint_info(data[0], point)[0]
            result['ENS_mem'] = np.array(valid_vals)
            mean_val = np.mean(valid_vals)
            result['ens_area'] = list(valid_vals)
        elif obs_geopoints is not None:
            result['ENS_mem'] = data
            member_means = []
            all_member_vals = []
            for member in data:
                gpt_vals = mv.nearest_gridpoint(member, obs_geopoints)
                vals = mv.values(gpt_vals)
                valid = vals[~np.isnan(vals)] if hasattr(vals, '__len__') else vals
                member_mean = float(np.mean(valid)) if len(valid) > 0 else np.nan
                member_means.append(member_mean)
                all_member_vals.append(member_mean)
            result['ENS_mem_area'] = member_means
            mean_val = mv.mean(data)
            result['ens_area'] = all_member_vals
        else:
            result['ENS_mem'] = data
            result['ENS_mem_area'] = mv.integrate(data, area_sub)
            mean_val = mv.mean(data)
            result['ens_area'] = [mv.integrate(member, area_sub) for member in data]

        result['ENS_mean'] = mean_val
        if point:
            area_val = mean_val
        elif obs_geopoints is not None:
            gpt_vals = mv.nearest_gridpoint(mean_val, obs_geopoints)
            vals = mv.values(gpt_vals)
            valid = vals[~np.isnan(vals)] if hasattr(vals, '__len__') else vals
            area_val = float(np.mean(valid)) if len(valid) > 0 else np.nan
        else:
            area_val = mv.integrate(mean_val, area_sub)
        result['mean_area'] = area_val
        print(f"Area mean value: {area_val}")
        ens = result['ens_area']
        print(f"Ensemble area values: min={min(ens) if ens else None}, "
              f"max={max(ens) if ens else None}, mean={np.mean(ens) if ens else None}")
    else:
        result['field'] = data
        if point:
            area_val = mv.nearest_gridpoint(data, point)
            if area_val is None:
                print(f"Warning: nearest_gridpoint returned None for {model_name}")
                return None
            nearest_gridinfo_dict[model_name + '_nearest'] = mv.nearest_gridpoint_info(data[0], point)[0]
        elif obs_geopoints is not None:
            gpt_vals = mv.nearest_gridpoint(data, obs_geopoints)
            vals = mv.values(gpt_vals)
            valid = vals[~np.isnan(vals)] if hasattr(vals, '__len__') else vals
            area_val = float(np.mean(valid)) if len(valid) > 0 else np.nan
        else:
            area_val = mv.integrate(data, area_sub)
        result['area'] = area_val
        print(f"value: {area_val}")

    return result


def _init_model_columns(data_df, model_name, is_ensemble):
    """Add empty columns for a model to the DataFrame."""
    if is_ensemble:
        for suffix in ('ensemble', 'mean', 'mean_area', 'ens_area',
                        'ENS_mem', 'ENS_mean', 'ENS_mem_area'):
            data_df[_col(model_name, suffix)] = None
    else:
        for suffix in ('area', 'field'):
            data_df[_col(model_name, suffix)] = None


def _init_reference_row(model_names, valid_date, obs_fs_area, analysis_area, clim_em_area):
    """Build the step-0 reference row for the DataFrame."""
    reference_row = pd.DataFrame({
        'forecast_date': [valid_date],
        'forecast_step': [0],
        'valid_date': [valid_date],
    })
    for model_name in model_names:
        ms = get_model_retrieval_settings(model_name)
        if ms['ensemble']:
            for suffix in ('ensemble', 'mean', 'mean_area', 'ens_area',
                            'ENS_mem_area', 'ENS_mem', 'ENS_mean'):
                reference_row[_col(model_name, suffix)] = np.nan
        else:
            for suffix in ('area', 'field'):
                reference_row[_col(model_name, suffix)] = np.nan

    reference_row['observations'] = obs_fs_area
    reference_row['analysis'] = analysis_area
    reference_row['climatology'] = clim_em_area
    return reference_row


# ---------------------------------------------------------------------------
# Main retrieval orchestrator
# ---------------------------------------------------------------------------

def _to_python_scalar(value):
    """Return a JSON-serialisable scalar for caching.

    Converts numpy scalars and 1-element fieldsets/geopoints to floats where
    possible; leaves ``None`` untouched.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return float(np.asarray(value).item())
        except Exception:
            return None


def _serialise_nearest_info(info):
    """Return a JSON-serialisable copy of a nearest-gridpoint info dict."""
    if info is None:
        return None
    out = {}
    for k, v in dict(info).items():
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = str(v)
    return out


def _load_reference_cache(path):
    """Return a dict of cached reference values, or {} if no cache exists."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_reference_cache(path, data):
    """Persist reference scalars to *path* (creating parent dir if needed)."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    except OSError as e:
        print(f"Warning: could not save reference cache: {e}")


def retrieve_and_store_data(widgets_dict, base_path):
    """Retrieve observations, analysis, climatology and model forecasts.

    Reads configuration from ``widgets_dict['config']`` (no global state).
    Returns a ``plot_data`` dict ready for the plotting functions.
    """
    config = widgets_dict['config']
    normalize_config(config)
    param = config['param']
    valid_date = config['valid_date']
    area_sub = list(config['area_sub'])
    point = config.get('point')
    forecast_dates = list(config['forecast_dates'])
    forecast_steps = list(config['forecast_steps'])
    step_interval = config['step_interval']
    n_members = config['n_members']
    selected_models = list(widgets_dict['model_widgets'].value)

    var_settings = _get_var_settings_safe(param)
    levtype = config['levtype']
    level = config['level']

    # Setup data directories
    area_str = get_area_string(area_sub)
    date_str = valid_date.strftime("%Y%m%d")
    time_str = f"{valid_date.hour:02d}00"

    retrieval_area = list(area_sub)
    if point is None:
        retrieval_area = _expand_area_for_plotting(area_sub, radius=4.0)
        print(f"Using expanded retrieval area for plotting: {retrieval_area}")

    grib_dir, obs_dir, plot_dir = setup_data_directories(
        base_path, param, area_str, date_str, time_str)

    nearest_gridinfo_dict = {}
    # Read acc_period from config when available (authoritative); fall back
    # to the widget only for legacy code paths. For non-accumulated variables
    # normalize_config() already sets this to None.
    acc_period_value = config.get('acc_period')
    if acc_period_value is None:
        acc_widget = widgets_dict.get('acc_period_widget')
        acc_period_value = acc_widget.value if acc_widget is not None else None
    area_avg_mode = config.get('area_avg_mode', 'integrate')

    # Scalar cache for reference values (obs/analysis/climatology). Avoids
    # re-querying MARS / recomputing the area-average on every re-run.
    ref_cache_path = os.path.join(plot_dir, 'reference_values.json')
    ref_cache = _load_reference_cache(ref_cache_path)

    # 1. Observations
    print("Retrieving observations...")
    obs_fs_area, obs_geopoints_raw = _retrieve_observations(
        param, var_settings, levtype, valid_date, area_sub,
        retrieval_area, point, obs_dir, acc_period_value,
        nearest_gridinfo_dict)
    # Determine whether to use station-nearest-gridpoint extraction.
    # Only applicable in area mode (not point) when the user chose
    # 'station_nearest' and observations were successfully retrieved.
    obs_geopoints = None
    if area_avg_mode == 'station_nearest' and point is None:
        if obs_geopoints_raw is not None:
            n_stations = len(obs_geopoints_raw)
            if n_stations > 0:
                obs_geopoints = obs_geopoints_raw
                print(f"Station-nearest mode: using {n_stations} station locations "
                      "for gridpoint extraction")
            else:
                print("Warning: No stations found in obs geopoints — "
                      "falling back to area integral")
        else:
            print("Warning: No observation geopoints available — "
                  "falling back to area integral")

    # 2. Analysis
    analysis_area = None
    if param != 'tp':
        if 'analysis_area' in ref_cache:
            analysis_area = ref_cache['analysis_area']
            print(f"Analysis value loaded from cache: {analysis_area}")
            if ref_cache.get('nearest_analysis'):
                nearest_gridinfo_dict['nearest_analysis'] = ref_cache['nearest_analysis']
        else:
            print("Retrieving analysis...")
            analysis_area = _retrieve_reference_field(
                'Analysis', get_analysis_settings, param, levtype, level,
                valid_date, area_sub, retrieval_area, point, grib_dir,
                nearest_gridinfo_dict,
                obs_geopoints=obs_geopoints)

    # 3. Reference value
    if obs_fs_area is not None:
        reference = obs_fs_area
    elif analysis_area is not None:
        reference = analysis_area
    else:
        reference = None

    # 4. Climatology
    if 'clim_em_area' in ref_cache:
        clim_em_area = ref_cache['clim_em_area']
        print(f"Climatology value loaded from cache: {clim_em_area}")
    else:
        print("Retrieving climatology...")
        clim_em_area = _retrieve_reference_field(
            'Climatology', get_climatology_settings, param, levtype, level,
            valid_date, area_sub, retrieval_area, point, grib_dir,
            nearest_gridinfo_dict,
            obs_geopoints=obs_geopoints)

    # Persist reference scalars so subsequent runs skip MARS retrieval.
    _save_reference_cache(ref_cache_path, {
        'analysis_area': _to_python_scalar(analysis_area),
        'clim_em_area': _to_python_scalar(clim_em_area),
        'nearest_analysis': _serialise_nearest_info(
            nearest_gridinfo_dict.get('nearest_analysis')),
    })

    # 5. Initialize DataFrame
    data_df = pd.DataFrame({
        'forecast_date': forecast_dates,
        'forecast_step': forecast_steps,
        'valid_date': [valid_date] * len(forecast_dates),
    })
    for model_name in selected_models:
        ms = get_model_retrieval_settings(model_name)
        _init_model_columns(data_df, model_name, ms['ensemble'])
    data_df['observations'] = None
    data_df['analysis'] = None
    data_df['climatology'] = None

    # 6. Retrieve model data
    timed_out_models = set()
    for idx, (fc_date, step) in enumerate(zip(forecast_dates, forecast_steps)):
        print(f"\nRetrieving data for forecast date {fc_date} at step {step}...")

        for model_name in selected_models:
            if model_name in timed_out_models:
                continue
            print(f"Retrieving {model_name}...")

            try:
                result = _retrieve_single_model_step(
                    model_name, param, var_settings, levtype, level,
                    fc_date, step, area_sub, retrieval_area, point, grib_dir,
                    acc_period_value, n_members, nearest_gridinfo_dict,
                    obs_geopoints=obs_geopoints)

                if result is not None:
                    ms = get_model_retrieval_settings(model_name)
                    print(f"Storing data for {model_name} (ensemble: {ms['ensemble']})")
                    for suffix, value in result.items():
                        data_df.at[idx, _col(model_name, suffix)] = value

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
                    f"{'='*60}\n{traceback.format_exc()}"
                )
                print(error_msg)

    # 7. Build reference row and assemble final DataFrame
    reference_row = _init_reference_row(
        selected_models, valid_date, obs_fs_area, analysis_area, clim_em_area)
    print(f"Reference data - obs: {obs_fs_area}, analysis: {analysis_area}, climatology: {clim_em_area}")

    data_df = pd.concat([reference_row, data_df], ignore_index=True)
    print(f"Final DataFrame shape: {data_df.shape}")

    # 8. Filter incomplete accumulation rows
    if var_settings.get('is_accumulated', False):
        acc_period = acc_period_value
        if acc_period and acc_period > step_interval:
            before = len(data_df)
            data_df = data_df[
                (data_df['forecast_step'] == 0) | (data_df['forecast_step'] >= acc_period)
            ].reset_index(drop=True)
            dropped = before - len(data_df)
            if dropped:
                print(f"Dropped {dropped} forecast row(s) with step < {acc_period}h "
                      f"(incomplete accumulation period)")

    # Rebuild forecast lists from (possibly filtered) DataFrame
    fc_rows = data_df[data_df['forecast_step'] > 0]
    forecast_dates = fc_rows['forecast_date'].tolist()
    forecast_steps = fc_rows['forecast_step'].tolist()

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
        'retrieval_area': retrieval_area,
        'base_path': base_path,
    }

    return plot_data


# ---------------------------------------------------------------------------
# Run configuration persistence
# ---------------------------------------------------------------------------

def save_run_config(config, output_dir):
    """Save the run configuration to a JSON file.

    Serialises ``config`` (the ``widgets_dict['config']`` dict) to
    ``<output_dir>/run_config.json``.  Datetime objects are converted to
    ISO-8601 strings so the file is plain JSON.

    Parameters
    ----------
    config : dict
        The configuration dictionary (``widgets_dict['config']``).
    output_dir : str
        Directory where ``run_config.json`` will be written.

    Returns
    -------
    str
        Path to the saved JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)

    serialisable = {}
    for key, value in config.items():
        if isinstance(value, datetime):
            serialisable[key] = value.isoformat()
        elif isinstance(value, (list, tuple)):
            serialisable[key] = [
                v.isoformat() if isinstance(v, datetime) else v
                for v in value
            ]
        else:
            serialisable[key] = value

    path = os.path.join(output_dir, 'run_config.json')
    with open(path, 'w') as f:
        json.dump(serialisable, f, indent=2)
    print(f"Run configuration saved to {path}")
    return path


def build_widgets_dict(config):
    """Build a minimal widgets_dict-compatible dict from a plain *config*.

    Equivalent to pressing the "Forecast Setup" button in the UI:
    the config is normalised (longitudes wrapped, area_sub / forecast
    dates auto-derived from point, acc_period cleared for non-accumulated
    variables) and wrapped into the shape expected by
    :func:`retrieve_and_store_data` and the plotting functions.

    Parameters
    ----------
    config : dict
        User-provided configuration. Any missing optional keys
        (``area_sub``, ``forecast_dates``, ``forecast_steps``) are filled
        by :func:`normalize_config`.

    Returns
    -------
    dict
        A ``widgets_dict`` with keys ``'config'``, ``'model_widgets'`` and
        ``'acc_period_widget'``.
    """
    class _StubWidget:
        def __init__(self, value):
            self.value = value

    normalize_config(config)
    selected = config.get('selected_models', ())
    acc = config.get('acc_period')
    return {
        'config': config,
        'model_widgets': _StubWidget(tuple(selected)),
        'acc_period_widget': _StubWidget(acc),
    }


def load_run_config(config_path):
    """Load a previously saved run configuration and build a widgets_dict.

    Reads ``run_config.json`` and returns a minimal ``widgets_dict``
    structure (with a ``'config'`` key) that can be passed directly to
    ``retrieve_and_store_data`` and the plotting functions.

    Parameters
    ----------
    config_path : str
        Path to a ``run_config.json`` file.

    Returns
    -------
    dict
        A ``widgets_dict``-compatible dictionary.
    """
    with open(config_path, 'r') as f:
        raw = json.load(f)

    # Restore datetime objects
    if 'valid_date' in raw and isinstance(raw['valid_date'], str):
        raw['valid_date'] = datetime.fromisoformat(raw['valid_date'])
    if 'forecast_dates' in raw:
        raw['forecast_dates'] = [
            datetime.fromisoformat(d) if isinstance(d, str) else d
            for d in raw['forecast_dates']
        ]

    return build_widgets_dict(raw)
