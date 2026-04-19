#!/usr/bin/env python3
"""
Example 3 — Pressure-level parameter, area averaging, multi-model maps.

Showcases:
  * A pressure-level variable: temperature at 850 hPa
    (``levtype='pl'``, ``level=850``).
  * Area selection with ``area_avg_mode='station_nearest'``: model
    gridpoint values are extracted at STVL station locations inside the
    box and then averaged, rather than integrating the full grid.
  * Analysis overlay on both the Plotly and Matplotlib plots.
  * Map plots for all selected models across multiple lead times,
    including analysis overlay (``overlay_analysis=True``).
  * Default (``nearest_neighbour``) and linear percentile methods
    compared side-by-side by exporting both PNGs.

Parameter : Temperature at 850 hPa
Area      : Central Europe  [N=54, W=5, S=44, E=22]

Defaults for predefined models, plot styles, and variable metadata
are loaded from the JSON files in ``diag_evo/config/``
(``model_settings.json``, ``plot_settings.json``,
``variable_settings.json``).

Usage
-----
    python run_area_t850_central_europe.py
"""

import os
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR   = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_DIR)

from diag_evo import (
    build_widgets_dict,
    retrieve_and_store_data,
    setup_data_directories,
    get_area_string,
    plot_forecast_evolution,
    plot_forecast_evolution_static,
    plot_field_map,
    plot_analysis_map,
)

# ── Configuration ──────────────────────────────────────────────────────
VALID_DATE    = datetime(2026, 4, 7, 12)
AREA_SUB      = [54.0, 5.0, 44.0, 22.0]   # [N, W, S, E]
MAX_DAYS      = 5
STEP_INTERVAL = 12

SELECTED_MODELS = [
    "IFS Control",
    "AIFS Single",
    "IFS ENS",
    "AIFS ENS",
]

config = {
    "param":   "t",
    "levtype": "pl",
    "level":   850,
    "valid_date": VALID_DATE,
    "point":   None,                    # area mode
    "area_sub": AREA_SUB,
    "max_days":       MAX_DAYS,
    "step_interval":  STEP_INTERVAL,
    "selected_models": SELECTED_MODELS,
    "n_members": 50,
    # station_nearest: extract model value at each synop station inside
    # the box, then average — more physically meaningful than area
    # integration when stations cluster in populated areas.
    "area_avg_mode": "station_nearest",
}

widgets_dict = build_widgets_dict(config)

# ── Retrieve ───────────────────────────────────────────────────────────
BASE_PATH = os.path.join(REPO_DIR, "data_files")
print("Retrieving data ...")
plot_data = retrieve_and_store_data(widgets_dict, BASE_PATH)

area_str = get_area_string(config["area_sub"])
date_str = VALID_DATE.strftime("%Y%m%d")
time_str = f"{VALID_DATE.hour:02d}00"
_, _, plot_dir = setup_data_directories(
    BASE_PATH, config["param"], area_str, date_str, time_str
)

# ── Interactive Plotly plot ────────────────────────────────────────────
print("Interactive plot ...")
plot_forecast_evolution(plot_data, widgets_dict, plot_dir, export_html=True)

# ── Static plots — compare both percentile methods side by side ────────
for method in ["nearest_neighbour", "linear"]:
    print(f"Static plot (pctl_method='{method}') ...")
    plot_forecast_evolution_static(
        plot_data, widgets_dict, plot_dir,
        pctl_method=method,
        png_filename=os.path.join(
            plot_dir,
            f"forecast_evolution_static_{method}_t850"
            f"_{VALID_DATE.strftime('%Y%m%d_%H%M')}.png",
        ),
        export_png=True,
    )

# ── Map plots ──────────────────────────────────────────────────────────
# For each lead time: gridded field with analysis contour overlay.
print("Map plots ...")
for step in [24, 48, 72]:
    for model in SELECTED_MODELS:
        try:
            plot_field_map(
                plot_data, widgets_dict, model, step=step,
                plot_mode="field",
                overlay_obs=False,
                overlay_analysis=True,   # analysis contour for context
                plot_radius=4,
                export_png=True,
            )
        except Exception as e:
            print(f"  skip {model} T+{step}h: {e}")

# Analysis-only map with observation overlay
try:
    plot_analysis_map(plot_data, widgets_dict,
                      overlay_obs=True, plot_radius=4, export_png=True)
except Exception as e:
    print(f"  skip analysis map: {e}")

print(f"\nDone — outputs written to {plot_dir}")
