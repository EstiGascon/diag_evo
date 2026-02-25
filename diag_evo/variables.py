"""
Utility functions for handling different variable types in forecast evolution analysis.
"""

import json
import os
import metview as mv
import numpy as np
from datetime import datetime, timedelta

def load_variable_settings():
    """Load variable settings from JSON file"""
    settings_path = os.path.join(os.path.dirname(__file__), 'config', 'variable_settings.json')
    try:
        with open(settings_path, 'r') as f:
            return json.load(f)['variable_settings']
    except FileNotFoundError:
        raise FileNotFoundError(f"variable_settings.json file not found at {settings_path}")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in variable_settings.json")

def get_base_var(var):
    """Extract base variable name from input (e.g., 'T850' -> 't')"""
    if var[0].upper() in ['T', 'Z'] and len(var) > 1 and var[1:].isdigit():
        return var[0].lower()
    else:
        return var.lower()
    
def get_level(var):
    """Extract level from variable name (e.g., 'T850' -> 850)"""
    if var[0].upper() in ['T', 'Z'] and len(var) > 1 and var[1:].isdigit():
        return int(var[1:])
    return None

def get_variable_settings(var):
    """Get settings for a specific variable.

    Accepts both combined names like 'z500' and bare names like 'z'.
    When a level is embedded in the name it is validated against the
    known levels, but a missing level is tolerated so that callers
    can still look up units / conversion rules for the base variable.
    """
    settings = load_variable_settings()
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

def convert_to_display(value, var, grib_units=None):
    """Convert value to display units based on variable settings and actual GRIB units"""
    settings = get_variable_settings(var)
    target_units = settings['units']
    
    # If no GRIB units provided, try to get them from the value if it's a metview object
    if grib_units is None and hasattr(value, 'grib_get'):
        grib_units = get_grib_units(value)
    
    # If still no units, use the old conversion system as fallback
    if grib_units is None:
        print(f"Warning: No GRIB units found for {var}, using fallback conversion")
        # Check if old conversion system exists
        if 'conversion' in settings:
            conversion = settings['conversion']['to_display']
            return apply_conversion(value, conversion)
        else:
            return value
    
    # Normalize units for comparison
    grib_units_upper = grib_units.upper()
    
    # Check if the GRIB units are in our known units list
    if grib_units_upper in [u.upper() for u in settings['grib_units']]:
        # Find the matching conversion rule
        for unit, rules in settings['conversion_rules'].items():
            if unit.upper() == grib_units_upper:
                conversion = rules['to_display']
                print(f"Converting {var} from {grib_units} to {target_units} using {conversion}")
                return apply_conversion(value, conversion)
    
    # If no matching conversion rule found, check if units are already correct
    if grib_units_upper == target_units.upper():
        print(f"No conversion needed for {var}: already in {target_units}")
        return value
    
    # If we get here, we don't know how to convert
    print(f"Warning: No conversion rule found for {var} from {grib_units} to {target_units}")
    return value

def convert_from_display(value, var, grib_units=None):
    """Convert value from display units to model units based on variable settings and actual GRIB units"""
    settings = get_variable_settings(var)
    target_units = settings['units']
    
    # If no GRIB units provided, try to get them from the value if it's a metview object
    if grib_units is None and hasattr(value, 'grib_get'):
        grib_units = get_grib_units(value)
    
    # If still no units, use the old conversion system as fallback
    if grib_units is None:
        print(f"Warning: No GRIB units found for {var}, using fallback conversion")
        # Check if old conversion system exists
        if 'conversion' in settings:
            conversion = settings['conversion']['from_display']
            return apply_conversion(value, conversion)
        else:
            return value
    
    # Normalize units for comparison
    grib_units_upper = grib_units.upper()
    
    # Check if the GRIB units are in our known units list
    if grib_units_upper in [u.upper() for u in settings['grib_units']]:
        # Find the matching conversion rule
        for unit, rules in settings['conversion_rules'].items():
            if unit.upper() == grib_units_upper:
                conversion = rules['from_display']
                print(f"Converting {var} from {target_units} to {grib_units} using {conversion}")
                return apply_conversion(value, conversion)
    
    # If no matching conversion rule found, check if units are already correct
    if grib_units_upper == target_units.upper():
        print(f"No conversion needed for {var}: already in {grib_units}")
        return value
    
    # If we get here, we don't know how to convert
    print(f"Warning: No conversion rule found for {var} from {target_units} to {grib_units}")
    return value

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
    base_var = get_base_var(var)

    if base_var in ['t', 'z']:
        level = get_level(var)
        if level is not None:
            return f"{settings['description']} at {level}hPa ({settings['units']})"
        # bare name — no level embedded
        return f"{settings['description']} ({settings['units']})"
    else:
        return f"{settings['description']} ({settings['units']})" 
