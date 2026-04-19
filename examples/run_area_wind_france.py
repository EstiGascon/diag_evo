#!/usr/bin/env python3
"""
Example 1 — Area selection, custom MARS model, colour overrides.

Showcases:
  * Bounding-box (area) selection — no single point;
    ``area_avg_mode='integrate'`` takes a Gaussian-weighted spatial
    average over the entire box.
  * Registering a custom MARS experiment at runtime with
    ``register_custom_model``.
  * Per-model colour overrides via ``config['model_colors']``.
  * Switching the static plot to ECMWF-linear percentiles
    (``pctl_method='linear'``).
  * Gridded-field map plots and station-nearest extraction side by side.

Parameter : 10 m wind speed (10si — derived from U/V components)
Area      : France + English Channel  [N=52, W=-5, S=43, E=8]

Defaults for predefined models, plot styles, and variable metadata
are loaded from the JSON files in ``diag_evo/config/``
(``model_settings.json``, ``plot_settings.json``,
``variable_settings.json``).  Edit them to add models or change
appearance without touching Python code.

Usage
-----
    python run_area_wind_france.py
"""

import os
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR   = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_DIR)

from diag_evo import (
    build_widgets_dict,
    register_custom_model,
    retrieve_and_store_data,
    setup_data_directories,
    get_area_string,
    plot_forecast_evolution,
    plot_forecast_evolution_static,
    plot_field_map,
    plot_obs_map,
    plot_analysis_map,
)

# ── 1. Register a custom MARS experiment ──────────────────────────────
# Any model with known MARS keys can be added at runtime without editing
# JSON config files.  Use color= to control its appearance in plots.
register_custom_model(
    name="IFS Exp",
    retrieval_args={
        "class":  "rd",
        "type":   "fc",
        "stream": "oper",
        "expver": "xxxx",   # <-- replace with your actual expver
    },
    is_ensemble=False,
    color="#e07b00",
)

# ── 2. Configuration ───────────────────────────────────────────────────
VALID_DATE    = datetime(2026, 4, 7, 12)
AREA_SUB      = [52.0, -5.0, 43.0, 8.0]   # [N, W, S, E]
MAX_DAYS      = 5
STEP_INTERVAL = 12

SELECTED_MODELS = [
    "IFS Control",
    "AIFS Single",
    "IFS ENS",
    "IFS Exp",
]

config = {
    "param":    "10si",
    "levtype":  "sfc",
    "level":    None,
    "valid_date": VALID_DATE,
    "point":    None,              # area mode — no single-point extraction
    "area_sub": AREA_SUB,
    "max_days":       MAX_DAYS,
    "step_interval":  STEP_INTERVAL,
    "selected_models": SELECTED_MODELS,
    "n_members": 50,
    "area_avg_mode": "integrate",  # Gaussian-weighted spatial average
}

widgets_dict = build_widgets_dict(config)

# ── 3. Colour overrides ────────────────────────────────────────────────
# Set after build_widgets_dict; picked up by both Plotly and Matplotlib
# plot functions without touching any JSON config file.
widgets_dict["config"]["model_colors"] = {
    "IFS Control": "#1a56a0",
    "AIFS Single": "#c0392b",
    "IFS ENS":     "#1a8c4e",
    "IFS Exp":     "#e07b00",
}

# ── 4. Retrieve ────────────────────────────────────────────────────────
BASE_PATH = os.path.join(REPO_DIR, "data_files")
print("Retrieving data ...")
plot_data = retrieve_and_store_data(widgets_dict, BASE_PATH)

area_str = get_area_string(config["area_sub"])
date_str = VALID_DATE.strftime("%Y%m%d")
time_str = f"{VALID_DATE.hour:02d}00"
_, _, plot_dir = setup_data_directories(
    BASE_PATH, config["param"], area_str, date_str, time_str
)

# ── 5. Interactive Plotly plot ─────────────────────────────────────────
print("Interactive plot ...")
plot_forecast_evolution(plot_data, widgets_dict, plot_dir, export_html=True)

# ── 6. Static plot — ECMWF-linear percentile method ───────────────────
# 'linear' applies R = P/100 * (N+1) with fractional interpolation
# (Hyndman & Fan method 6 / Weibull).
# The default 'nearest_neighbour' matches Metview's convention.
print("Static plot (pctl_method='linear') ...")
plot_forecast_evolution_static(
    plot_data, widgets_dict, plot_dir,
    pctl_method="linear",
    export_png=True,
)

# ── 7. Map plots ───────────────────────────────────────────────────────
print("Map plots ...")
for step in [24, 48, 72]:
    for model in ["IFS Control", "IFS ENS"]:
        # Gridded shading with obs overlay
        try:
            plot_field_map(plot_data, widgets_dict, model, step=step,
                           plot_mode="field", overlay_obs=True,
                           plot_radius=3, export_png=True)
        except Exception as e:
            print(f"  skip {model} T+{step}h field: {e}")

        # Nearest-gridpoint values extracted at station locations
        try:
            plot_field_map(plot_data, widgets_dict, model, step=step,
                           plot_mode="station_nearest", overlay_obs=False,
                           plot_radius=3, export_png=True)
        except Exception as e:
            print(f"  skip {model} T+{step}h station_nearest: {e}")

try:
    plot_obs_map(plot_data, widgets_dict, plot_radius=3, export_png=True)
except Exception as e:
    print(f"  skip obs map: {e}")

try:
    plot_analysis_map(plot_data, widgets_dict,
                      overlay_obs=True, plot_radius=3, export_png=True)
except Exception as e:
    print(f"  skip analysis map: {e}")

print(f"\nDone — outputs written to {plot_dir}")
