"""
diag_evo — Forecast Evolution Diagnostics
==========================================

Interactive and static visualisation of forecast evolution
("Linus plot") for multiple NWP / ML models.

Quick start
-----------
>>> from diag_evo import setup_interface, retrieve_and_store_data
>>> from diag_evo import plot_forecast_evolution, plot_forecast_evolution_static
>>> widgets_dict = setup_interface()
>>> plot_data = retrieve_and_store_data(widgets_dict, base_path)
>>> plot_forecast_evolution(plot_data, widgets_dict, plot_dir)
"""

from .ui import setup_interface

from .core import (
    retrieve_and_store_data,
    setup_data_directories,
    get_area_string,
    sanitize_mars_request,
    save_run_config,
    load_run_config,
    normalize_config,
    build_widgets_dict,
    fetch_observations_only,
    STVL_AVAILABLE_PARAMS,
)

from .nb_ensemble import add_nb_ens_to_plot_data

from .plotting import (
    plot_forecast_evolution,
    plot_forecast_evolution_static,
)

from .map_plotting import plot_field_map, plot_obs_map, plot_analysis_map

from .settings import (
    register_custom_model,
    unregister_custom_model,
    clear_custom_models,
    get_model_retrieval_settings,
    get_model_plot_settings,
    get_available_models,
    get_predefined_models,
    get_custom_models,
)

__all__ = [
    # UI
    "setup_interface",
    # Core workflow
    "retrieve_and_store_data",
    "setup_data_directories",
    "get_area_string",
    "sanitize_mars_request",
    "save_run_config",
    "load_run_config",
    "normalize_config",
    "build_widgets_dict",
    "add_nb_ens_to_plot_data",
    "fetch_observations_only",
    "STVL_AVAILABLE_PARAMS",
    # Plotting
    "plot_forecast_evolution",
    "plot_forecast_evolution_static",
    # Map plotting
    "plot_field_map",
    "plot_obs_map",
    "plot_analysis_map",
    # Model management
    "register_custom_model",
    "unregister_custom_model",
    "clear_custom_models",
    "get_model_retrieval_settings",
    "get_model_plot_settings",
    "get_available_models",
    "get_predefined_models",
    "get_custom_models",
]
