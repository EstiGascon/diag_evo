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

from .core import (
    setup_interface,
    retrieve_and_store_data,
    plot_forecast_evolution,
    plot_forecast_evolution_static,
    setup_data_directories,
    get_area_string,
    sanitize_mars_request,
)

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
    # Core workflow
    "setup_interface",
    "retrieve_and_store_data",
    "plot_forecast_evolution",
    "plot_forecast_evolution_static",
    "setup_data_directories",
    "get_area_string",
    "sanitize_mars_request",
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
