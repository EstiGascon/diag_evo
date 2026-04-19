#!/usr/bin/env python3
"""
Run the forecast evolution tool from a saved run_config.json.

Usage
-----
    python run_from_config.py /path/to/run_config.json

This script demonstrates how to reproduce a notebook run entirely from
the command line, using the JSON configuration that is automatically
saved by ``retrieve_and_store_data``.

You can also build a ``widgets_dict`` from scratch (without the UI)
to create new runs programmatically — see the "From scratch" section
at the bottom of this file.
"""

import argparse
import os
import sys

# ── Ensure the package is importable ──────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_DIR)

from diag_evo import (
    load_run_config,
    retrieve_and_store_data,
    setup_data_directories,
    get_area_string,
    plot_forecast_evolution,
    plot_forecast_evolution_static,
    plot_field_map,
    plot_obs_map,
    plot_analysis_map,
    get_model_retrieval_settings,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run forecast evolution diagnostics from a saved config."
    )
    parser.add_argument(
        "config", help="Path to a run_config.json file."
    )
    parser.add_argument(
        "--base-path", default=os.path.join(REPO_DIR, "data_files"),
        help="Base directory for cached data (default: <repo>/data_files).",
    )
    parser.add_argument(
        "--no-interactive", action="store_true",
        help="Skip the interactive Plotly plot.",
    )
    parser.add_argument(
        "--no-static", action="store_true",
        help="Skip the static Matplotlib plot.",
    )
    parser.add_argument(
        "--no-maps", action="store_true",
        help="Skip map plots.",
    )
    parser.add_argument(
        "--map-steps", nargs="+", type=int, default=None,
        help="Lead times (hours) for map plots. Default: all forecast_steps from the config.",
    )
    args = parser.parse_args()

    # 1. Load configuration
    print(f"Loading config from {args.config}")
    widgets_dict = load_run_config(args.config)
    config = widgets_dict['config']

    # 2. Retrieve data (uses cache when available)
    print("Retrieving data...")
    plot_data = retrieve_and_store_data(widgets_dict, args.base_path)

    # 3. Build output directory paths
    area_str = get_area_string(config['area_sub'])
    date_str = config['valid_date'].strftime("%Y%m%d")
    time_str = f"{config['valid_date'].hour:02d}00"
    _, _, plot_dir = setup_data_directories(
        args.base_path, config['param'], area_str, date_str, time_str
    )

    # 4. Forecast evolution plots
    if not args.no_interactive:
        print("Creating interactive plot...")
        plot_forecast_evolution(plot_data, widgets_dict, plot_dir, export_html=True)

    if not args.no_static:
        print("Creating static plot...")
        plot_forecast_evolution_static(plot_data, widgets_dict, plot_dir, export_png=True)

    # 5. Map plots
    if not args.no_maps:
        map_steps = args.map_steps if args.map_steps else list(config.get('forecast_steps', []))
        for step in map_steps:
            for model in config['selected_models']:
                is_ens = get_model_retrieval_settings(model).get('ensemble', False)
                col = f'{model}_mean_area' if is_ens else f'{model}_area'
                if col not in plot_data['data_df'].columns:
                    continue
                has_data = plot_data['data_df'].loc[
                    plot_data['data_df']['forecast_step'] == step, col
                ].notna().any()
                if not has_data:
                    continue
                print(f"Map: {model} T+{step}h ...")
                plot_field_map(plot_data, widgets_dict, model, step=step,
                               plot_mode='field', overlay_obs=True,
                               plot_radius=2, export_png=True)
                plot_field_map(plot_data, widgets_dict, model, step=step,
                               plot_mode='station_nearest', overlay_obs=True,
                               plot_radius=2, export_png=True)

        print("Observation map...")
        try:
            plot_obs_map(plot_data, widgets_dict, plot_radius=3, export_png=True)
        except Exception as e:
            print(f"  skip obs map: {e}")

        print("Analysis map...")
        try:
            plot_analysis_map(plot_data, widgets_dict,
                              overlay_obs=True, plot_radius=3, export_png=True)
        except Exception as e:
            print(f"  skip analysis map: {e}")

    print(f"\nDone. Outputs saved to {plot_dir}")


if __name__ == "__main__":
    main()


# ======================================================================
# From scratch (without a saved config)
# ======================================================================
# Uncomment the block below to build a widgets_dict manually via
# build_widgets_dict, which mimics pressing "Forecast Setup" in the UI.
#
# Either `point` (with area_sub auto-derived as a ±3° box) or a full
# `area_sub` bounding box can be supplied. Custom models can be
# registered at runtime via `register_custom_model` before the call.
#
# from datetime import datetime
# from diag_evo import build_widgets_dict, register_custom_model
#
# # Optional: register a custom MARS model
# register_custom_model(
#     name="My_Experiment",
#     retrieval_args={"class": "rd", "type": "fc", "stream": "oper",
#                     "expver": "0080"},
#     is_ensemble=False,
#     color="#ff6600",
# )
#
# config = {
#     "param": "2t",
#     "levtype": "sfc",
#     "level": None,
#     "valid_date": datetime(2026, 4, 7, 12),
#     "point": [50.748, 6.405],          # omit "area_sub" when using a point
#     "max_days": 5,
#     "step_interval": 12,
#     "selected_models": ["IFS Control", "AIFS Single", "My_Experiment"],
#     "n_members": 50,
#     "area_avg_mode": "station_nearest",
# }
# widgets_dict = build_widgets_dict(config)
# plot_data = retrieve_and_store_data(widgets_dict, "data_files")
