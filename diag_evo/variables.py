"""
Utility functions for handling different variable types in forecast evolution analysis.
"""

import json
import os
import numpy as np

# Module-level cache for variable settings
_cache_variable_settings = None

def load_variable_settings():
    """Load variable settings from JSON file, with module-level caching."""
    global _cache_variable_settings
    if _cache_variable_settings is not None:
        return _cache_variable_settings
    settings_path = os.path.join(os.path.dirname(__file__), 'config', 'variable_settings.json')
    try:
        with open(settings_path, 'r') as f:
            data = json.load(f)['variable_settings']
    except FileNotFoundError:
        raise FileNotFoundError(f"variable_settings.json file not found at {settings_path}")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in variable_settings.json")
    _cache_variable_settings = data
    return data

def _split_var_level(var):
    """Split a variable string into (shortName, level_int_or_None).

    Uses the JSON settings to decide whether trailing digits are a pressure
    level rather than hardcoding specific variable names.

    Examples:
        'z500'  -> ('z',  500)
        'T850'  -> ('t',  850)
        'q700'  -> ('q',  700)
        '2t'    -> ('2t', None)
        'tp'    -> ('tp', None)
        ''      -> ('',   None)
    """
    if not var:
        return '', None

    settings = load_variable_settings()
    low = var.lower()

    # If the whole string is already a known key, no splitting needed
    if low in settings:
        return low, None

    # Try progressively shorter prefixes to find a known pl variable
    for i in range(1, len(low)):
        prefix = low[:i]
        suffix = low[i:]
        if suffix.isdigit() and prefix in settings:
            if settings[prefix].get('levtype') == 'pl':
                return prefix, int(suffix)

    # No match — return lowered whole string, no level
    return low, None


def get_base_var(var):
    """Extract base variable name from input (e.g., 'T850' -> 't', '2t' -> '2t')."""
    base, _ = _split_var_level(var)
    return base


def get_level(var):
    """Extract level from variable name (e.g., 'T850' -> 850, '2t' -> None)."""
    _, level = _split_var_level(var)
    return level

def _resolve_param_id(var, settings):
    """If *var* is a numeric string (e.g. '167'), look it up by paramId.

    Returns the (shortName, var_settings) tuple if found, otherwise
    ``(None, {})``.
    """
    try:
        pid = int(var)
    except (ValueError, TypeError):
        return None, {}

    for short_name, entry in settings.items():
        if entry.get("paramId") == pid:
            return short_name, entry
    return None, {}


def get_variable_settings(var):
    """Get settings for a specific variable.

    Accepts:
      - combined names like ``'z500'`` or ``'T850'``
      - bare shortNames like ``'z'``, ``'2t'``
      - numeric paramId strings like ``'167'`` (→ 2t)

    When a level is embedded in the name it is validated against the
    known levels, but a missing level is tolerated so that callers
    can still look up units / conversion rules for the base variable.
    """
    settings = load_variable_settings()

    # ── Try paramId lookup first ──
    resolved_name, var_settings = _resolve_param_id(var, settings)
    if var_settings:
        return var_settings

    # ── Standard shortName lookup ──
    base_var = get_base_var(var)
    var_settings = settings.get(base_var, {})

    if not var_settings:
        raise ValueError(f"Settings for variable '{var}' not found in variable_settings.json")

    # If this is a pressure level variable *and* a level was given, validate it
    if var_settings['levtype'] == 'pl':
        level = get_level(var)
        if level is not None:
            valid_levels = list(var_settings['levels'].values())
            if level not in valid_levels:
                raise ValueError(f"Invalid level {level} for variable '{var}'. Valid levels are {valid_levels}")

    return var_settings

def get_grib_units(data):
    """Extract units from GRIB data"""
    try:
        units = data.grib_get(['units'])[0]
        # Handle cases where units might be a list or None
        if units is None:
            return None
        if isinstance(units, list) and len(units) > 0:
            return str(units[0]).strip()
        return str(units).strip()
    except Exception as e:
        print(f"Warning: Could not extract units from GRIB data: {e}")
        return None

def apply_conversion(value, conversion_type):
    """Apply a specific conversion to a value"""
    if conversion_type == 'none':
        return value
    elif conversion_type == 'subtract_273.15':
        return value - 273.15
    elif conversion_type == 'add_273.15':
        return value + 273.15
    elif conversion_type == 'multiply_1000':
        return value * 1000
    elif conversion_type == 'divide_1000':
        return value / 1000
    elif conversion_type == 'multiply_100':
        return value * 100
    elif conversion_type == 'divide_100':
        return value / 100
    elif conversion_type == 'multiply_3600':
        return value * 3600
    elif conversion_type == 'divide_3600':
        return value / 3600
    elif conversion_type == 'divide_10':
        return value / 10
    elif conversion_type == 'multiply_10':
        return value * 10
    elif conversion_type == 'divide_by_g':
        return value / 9.80665  # Standard gravitational acceleration
    elif conversion_type == 'multiply_by_g':
        return value * 9.80665
    else:
        raise ValueError(f"Unknown conversion type: {conversion_type}")

def _convert(value, var, direction, grib_units=None):
    """Convert a value to or from display units.

    Parameters
    ----------
    value :
        Numeric value or metview fieldset.
    var : str
        Variable name (e.g. ``'2t'``, ``'z500'``).
    direction : str
        ``'to_display'`` or ``'from_display'``.
    grib_units : str or None
        GRIB unit string.  If *None*, extracted from *value* when possible.
    """
    settings = get_variable_settings(var)
    target_units = settings['units']

    # Try to get GRIB units from metview object if not provided
    if grib_units is None and hasattr(value, 'grib_get'):
        grib_units = get_grib_units(value)

    # Fallback: old conversion system
    if grib_units is None:
        print(f"Warning: No GRIB units found for {var}, using fallback conversion")
        if 'conversion' in settings:
            conversion = settings['conversion'][direction]
            return apply_conversion(value, conversion)
        # No old-style 'conversion' key — infer from the first entry in
        # grib_units (the canonical GRIB source unit, e.g. 'K' for temperature).
        if 'grib_units' in settings and settings['grib_units']:
            grib_units = settings['grib_units'][0]
            print(f"Assuming default GRIB unit '{grib_units}' for {var}")
        else:
            return value

    # Normalise and look up conversion rule
    grib_units_upper = grib_units.upper()
    if grib_units_upper in [u.upper() for u in settings['grib_units']]:
        for unit, rules in settings['conversion_rules'].items():
            if unit.upper() == grib_units_upper:
                conversion = rules[direction]
                src = grib_units if direction == 'to_display' else target_units
                dst = target_units if direction == 'to_display' else grib_units
                print(f"Converting {var} from {src} to {dst} using {conversion}")
                return apply_conversion(value, conversion)

    # Already in target units?
    compare_unit = target_units if direction == 'to_display' else grib_units
    if grib_units_upper == compare_unit.upper():
        print(f"No conversion needed for {var}: already in {compare_unit}")
        return value

    src = grib_units if direction == 'to_display' else target_units
    dst = target_units if direction == 'to_display' else grib_units
    print(f"Warning: No conversion rule found for {var} from {src} to {dst}")
    return value


def convert_to_display(value, var, grib_units=None):
    """Convert value to display units based on variable settings and actual GRIB units"""
    return _convert(value, var, 'to_display', grib_units)


def convert_from_display(value, var, grib_units=None):
    """Convert value from display units to model units based on variable settings and actual GRIB units"""
    return _convert(value, var, 'from_display', grib_units)

def get_retrieval_settings(var, date, time, area, step=None, step_start=None):
    """Get retrieval settings for a variable"""
    settings = get_variable_settings(var)
    base_var = get_base_var(var)
    
    request = {
        "param": base_var,
        "levtype": settings['levtype'],
        "date": date.strftime("%Y%m%d"),
        "time": f"{time:02d}00",
        "area": area
    }
    
    # Add level for pressure level variables
    if settings['levtype'] == 'pl':
        level = get_level(var)
        if level is not None:
            request["levelist"] = level
    
    # Add step information if provided
    if step is not None:
        if settings['is_accumulated'] and step_start is not None:
            request["step"] = f"{step_start}/{step}"
        else:
            request["step"] = step
    
    return request

def process_accumulated_data(data, step_start, step):
    """Process accumulated data by taking the difference between steps"""
    import metview as mv
    if step_start == 0:
        return data
    else:
        # Get data for both steps
        data_start = mv.read(data, step=step_start)
        data_end = mv.read(data, step=step)
        # Calculate difference
        return data_end - data_start

def get_variable_display_name(var):
    """Get the display name for a variable.

    For pressure-level variables the level is included when available
    (e.g. 'z500' -> 'Geopotential Height at 500hPa (dam)').
    If only the bare name is given (e.g. 'z') the level is omitted.
    """
    settings = get_variable_settings(var)

    if settings.get('levtype') == 'pl':
        level = get_level(var)
        if level is not None:
            return f"{settings['description']} at {level}hPa ({settings['units']})"

    return f"{settings['description']} ({settings['units']})"


def is_derived(param):
    """Check if param is a derived variable requiring component retrieval."""
    try:
        vs = get_variable_settings(param)
        return vs.get('is_derived', False)
    except (ValueError, KeyError):
        return False


def get_derivation_info(param):
    """Return the derivation dict for a derived variable, or empty dict.

    The dict has keys: ``method``, ``components``, ``output_paramId``.
    """
    try:
        vs = get_variable_settings(param)
        return vs.get('derivation', {})
    except (ValueError, KeyError):
        return {}
