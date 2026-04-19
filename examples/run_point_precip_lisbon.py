#!/usr/bin/env python3
"""
Example 2 — Accumulated precipitation with NB ENS and config replay.

Showcases:
  * 6-hourly accumulated total precipitation (``acc_period=6``).
  * Injecting the Neighbourhood Ensemble (NB ENS) via
    ``add_nb_ens_to_plot_data`` — the static plot uses the originally
    retrieved 7 percentiles without recomputing from synthetic members.
  * Default ECMWF ``nearest_neighbour`` percentile method (explicit).
  * Reproducing an existing run from its saved ``run_config.json`` with
    ``load_run_config`` — demonstrating the full replay workflow.

Parameter : Total precipitation (tp), 6 h accumulation
Region    : Western Iberia — point selection at Lisbon

Defaults for predefined models, plot styles, and variable metadata
are loaded from the JSON files in ``diag_evo/config/``
(``model_settings.json``, ``plot_settings.json``,
``variable_settings.json``).

Usage
-----
    python run_point_precip_lisbon.py
    python run_point_precip_lisbon.py --replay  # replay from saved config
"""

import argparse
import os
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR   = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_DIR)

from diag_evo import (
    build_widgets_dict,
    load_run_config,
    retrieve_and_store_data,
    setup_data_directories,
    get_area_string,
    plot_forecast_evolution,
    plot_forecast_evolution_static,
    plot_field_map,
    plot_obs_map,
    add_nb_ens_to_plot_data,
)

NB_PATH = "/ec/vol/destine/neighbourhood/percentile"


def _run(widgets_dict, base_path):
    config = widgets_dict["config"]

    print("Retrieving data ...")
    plot_data = retrieve_and_store_data(widgets_dict, base_path)

    area_str = get_area_string(config["area_sub"])
    date_str = config["valid_date"].strftime("%Y%m%d")
    time_str = f"{config['valid_date'].hour:02d}00"
    _, _, plot_dir = setup_data_directories(
        base_path, config["param"], area_str, date_str, time_str
    )

    # ── Neighbourhood Ensemble (NB ENS) ───────────────────────────────
    # Only supported for tp; skips gracefully if files are not present.
    # The static plot will use the originally retrieved 7 percentiles
    # directly, not recomputed from the 51-member synthetic expansion.
    try:
        print("Adding NB ENS ...")
        plot_data_nb, widgets_dict = add_nb_ens_to_plot_data(
            plot_data, widgets_dict, NB_PATH
        )
    except Exception as e:
        print(f"  NB ENS unavailable: {e}; continuing without it.")
        plot_data_nb = plot_data

    # ── Plots ─────────────────────────────────────────────────────────
    print("Interactive plot ...")
    plot_forecast_evolution(plot_data_nb, widgets_dict, plot_dir,
                            export_html=True)

    # nearest_neighbour is the default and matches the Metview convention;
    # shown explicitly here for clarity.
    print("Static plot (pctl_method='nearest_neighbour') ...")
    plot_forecast_evolution_static(
        plot_data_nb, widgets_dict, plot_dir,
        pctl_method="nearest_neighbour",
        export_png=True,
    )

    print("Map plots ...")
    for step in [24, 48]:
        for model in config.get("selected_models", []):
            if model == "NB ENS":
                continue   # no GRIB field available for NB ENS
            try:
                plot_field_map(plot_data_nb, widgets_dict, model, step=step,
                               plot_mode="field", overlay_obs=True,
                               plot_radius=4, export_png=True)
            except Exception as e:
                print(f"  skip {model} T+{step}h: {e}")

    try:
        plot_obs_map(plot_data_nb, widgets_dict, plot_radius=3, export_png=True)
    except Exception as e:
        print(f"  skip obs map: {e}")

    print(f"\nDone — outputs written to {plot_dir}")
    return plot_dir


def _fresh_run(base_path):
    """Build config from scratch and run."""
    config = {
        "param":    "tp",
        "levtype":  "sfc",
        "level":    None,
        "valid_date": datetime(2026, 4, 8, 0),
        "point":  [38.72, -9.14],          # Lisbon
        # area_sub is auto-derived as point ±3°
        "max_days":      5,
        "step_interval": 6,                # 6 h between each init
        "selected_models": [
            "IFS Control",
            "AIFS Single",
            "IFS ENS",
            "AIFS ENS",
        ],
        "n_members":  50,
        "area_avg_mode": "station_nearest",
        "acc_period":    6,                # 6 h accumulation
    }
    widgets_dict = build_widgets_dict(config)
    plot_dir = _run(widgets_dict, base_path)
    return plot_dir


def _replay_run(config_path, base_path):
    """Re-run an existing saved configuration."""
    print(f"Loading config from {config_path} ...")
    widgets_dict = load_run_config(config_path)
    _run(widgets_dict, base_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Forecast evolution — tp with NB ENS."
    )
    parser.add_argument(
        "--replay", metavar="CONFIG_JSON", default=None,
        help="Path to a run_config.json to replay (skips fresh retrieval).",
    )
    parser.add_argument(
        "--base-path",
        default=os.path.join(REPO_DIR, "data_files"),
        help="Base directory for cached data.",
    )
    args = parser.parse_args()

    if args.replay:
        _replay_run(args.replay, args.base_path)
    else:
        _fresh_run(args.base_path)
