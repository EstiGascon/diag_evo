# diag_evo — Forecast Evolution Diagnostics

Interactive and static visualisation of forecast evolution ("Linus plot")
for multiple NWP and ML models.  Built on **MARS**, **earthkit-data**,
**Metview**, **ipywidgets** and **Plotly / Matplotlib**.

---

## Features

| Feature | Details |
|---------|---------|
| **Predefined models** | IFS Control, AIFS Single, AIFS ENS Control, IFS ENS, AIFS ENS, DE-ATOS |
| **Custom MARS models** | Add any model at runtime by specifying `class`, `type`, `stream`, `expver` and optional extra MARS keys |
| **Interactive plot** | Plotly box-plots with hover info (error, lead time, member index) |
| **Static plot** | Matplotlib percentile boxes (1/10/25/50/75/90/99) with inset map. Percentiles use the ECMWF / Metview convention (`R = P/100 * (N+1)`), default `nearest_neighbour`, switchable to `linear` |
| **Reference data** | Analysis, ERA5 climatology, STVL station observations |
| **Station observations overlay** | Press **Show station observations** in the UI to colour-code STVL stations on the interactive map for the selected param + valid date |
| **Existing-case browser** | `display_case_selector(base_path, widgets_dict)` to reload a previously cached case from `data_files/` |
| **Neighbourhood Ensemble (NB ENS)** | Inject pre-computed 6 h tp percentiles via `add_nb_ens_to_plot_data`. The static plot uses the raw percentiles directly (no recomputation from synthetic members) |
| **Run config persistence** | Pressing **Forecast Setup** writes `run_config.json` under `<base_path>/run_configs/<param>_<date>_<area>/`, ready to be replayed via `examples/run_from_config.py` |
| **Accumulated variables** | Automatic step differencing with configurable accumulation period |
| **Colour overrides** | Per-model colour control via `widgets_dict['config']['model_colors']` |
| **Export** | HTML (interactive) and PNG (static) |

---

## Area & Point Selection

The interactive map widget supports two selection modes:

### Area selection
Draw a polygon on the map to define a bounding box.

<img src="./diag_evo/Area_select.gif" width="400" align="center">

### Point selection
Click a single point on the map — the system creates a ±3° bounding box around it and
reports the nearest observation station.

<img src="./diag_evo/Point_select.gif" width="400" align="center">

---

## Repository layout

```
diag_evo/
├── diag_evo/                  # Python package
│   ├── __init__.py            # Public API
│   ├── core.py                # Data retrieval, caching, config I/O
│   ├── plotting.py            # Plotly & Matplotlib forecast evolution plots
│   ├── map_plotting.py        # Metview geographic map plots
│   ├── settings.py            # Model & plot settings helpers
│   ├── variables.py           # Variable metadata & unit conversion
│   ├── ui.py                  # ipywidgets / ipyleaflet interface
│   └── config/                # Default JSON configuration
│       ├── model_settings.json
│       ├── plot_settings.json
│       └── variable_settings.json
├── examples/                  # Scripted usage examples
│   ├── run_from_config.py             # Replay any saved run_config.json
│   ├── run_area_wind_france.py        # Ex 1: area mode, custom model, colour overrides, pctl_method='linear'
│   ├── run_point_precip_lisbon.py     # Ex 2: tp accumulation, NB ENS, config replay with load_run_config
│   └── run_area_t850_central_europe.py  # Ex 3: pressure-level param, station_nearest, dual pctl-method
├── data_files/                # Auto-created at runtime (.gitignored)
├── tests/                     # Unit tests
├── forecast_evolution.ipynb   # Main notebook (interactive UI)
├── programmatic_workflow.ipynb # Scripted / reproduce-from-config notebook
├── requirements.txt
├── .gitignore
└── README.md                  # ← you are here
```

---

## Installation

The package is designed to run on systems where **Metview** and
**earthkit-data** are already available (e.g. Atos HPC).  
Clone the repository and install the remaining Python dependencies (if needed):

```bash
git clone https://git.ecmwf.int/scm/~ecm3468/diag_evo.git diag_evo
cd diag_evo
```
All required packages are already pre-installed on ATOS.

Start a JupyterHub session on ATOS.

Open the notebook:
[`forecast_evolution.ipynb`](forecast_evolution.ipynb)

Select an appropriate kernel (e.g. Python 3.12.11).

Run the notebook cells to start the diagnostic workflow.

If you encounter errors related to missing Python packages, install the required dependencies by running:

```bash
pip install -r requirements.txt
```
---

## Quick start

```python
import sys, os
sys.path.insert(0, "/path/to/diag_evo")

from diag_evo import (
    setup_interface,
    retrieve_and_store_data,
    plot_forecast_evolution,
    plot_forecast_evolution_static,
    setup_data_directories,
    get_area_string,
)

# 1. Launch the interactive widget UI
widgets_dict = setup_interface()

# 2. (Optional) override settings programmatically. Important Note: When manually selecting a 'point', make sure it is within 'area_sub'
# widgets_dict['config']['area_sub'] = [51.4, 5.7, 50.1, 9.4]
# widgets_dict['config']['point'] = [50.88, 7.01]

# 3. Retrieve data
base_path = "data_files"
plot_data = retrieve_and_store_data(widgets_dict, base_path)

# 4. Build output directories
config = widgets_dict['config']
area_str = get_area_string(config['area_sub'])
date_str = config['valid_date'].strftime("%Y%m%d")
time_str = f"{config['valid_date'].hour:02d}00"
_, _, plot_dir = setup_data_directories(base_path, config['param'], area_str, date_str, time_str)

# 5. Plot
plot_forecast_evolution(plot_data, widgets_dict, plot_dir, export_html=True)
plot_forecast_evolution_static(plot_data, widgets_dict, plot_dir, export_png=True)
```

See [`forecast_evolution.ipynb`](forecast_evolution.ipynb)
for the interactive workflow, or
[`programmatic_workflow.ipynb`](programmatic_workflow.ipynb)
for a UI-free / reproduce-from-config example.
---

## Adding custom models

### Via the UI

Use the **Custom MARS Model** panel on the right-hand side of the widget
interface.  Fill in `class`, `type`, `stream`, `expver`, tick
**Ensemble** if applicable, and click **Add Model**.

### Programmatically

```python
from diag_evo import register_custom_model

register_custom_model(
    name="My Experiment",
    retrieval_args={
        "class": "rd",
        "type": "fc",
        "stream": "oper",
        "expver": "xxxx",
    },
    is_ensemble=False,
    color="#ff6600",
)
```

---

## Colour overrides

Override plot colours for any model *before* calling the plot functions:

```python
widgets_dict['config']['model_colors'] = {
    'IFS Control': '#dd00ff',
    'AIFS ENS': '#2c32a0',
}
```

---

## How retrieval works

`retrieve_and_store_data(widgets_dict, base_path)` orchestrates the full data
pipeline.  Understanding the steps helps when debugging or tuning behaviour.

### 1. Directory naming

Data is cached under `base_path` in directories named:

```
{param}_{area_str}_{date}_{time}
```

Example: `2t_N51.228_W5.691_S50.268_E7.119_20260407_1200/`

Each directory contains three sub-folders: `grib_files/`, `obs_files/`,
`plot_files/`.  Re-running the retrieval with the same settings reuses
cached files.

### 2. Area expansion for retrieval

When the user selects an **area** (not a point), the MARS retrieval area
is expanded by **4°** in every direction so that map plots have a margin
around the selected box.  
See `_expand_area_for_plotting(area, radius=4.0)` in
[`core.py`](diag_evo/core.py).

For **point** mode, no expansion is applied for retrieval — the area is the
±3° bounding box created by the UI.

### 3. Observation search

STVL station observations are fetched for the selected area.  If no stations
are found, the search area is expanded by **0.25°** increments up to a maximum
of **2°** until stations are found (or the limit is reached).

### 4. Reference fields

- **Analysis** — retrieved for the validity date (not available for `tp`)
- **ERA5 climatology** — retrieved for comparison

### 5. Model forecasts

Each selected model × forecast step is retrieved individually.  A MARS
request timeout of **180 seconds** (`MARS_RETRIEVAL_TIMEOUT` in
[`core.py`](diag_evo/core.py)) prevents a single slow request from
blocking the pipeline.  If a model times out, it is skipped for all
remaining steps.

### 6. Configuration export

Pressing **Forecast Setup** in the notebook saves a `run_config.json`
under `<base_path>/run_configs/<param>_<date>_<area>/`.  It captures
all settings and can be loaded later with `load_run_config()`.

---

## Map plotting

After retrieving data, you can visualise forecast fields and observations on
geographic maps using the Metview-based map functions.

### Available functions

| Function | Description |
|----------|-------------|
| `plot_field_map` | Model forecast field for a given lead time |
| `plot_obs_map` | Observation station locations and values |
| `plot_analysis_map` | Analysis field, optionally with observation overlay |

### Example

```python
from diag_evo import plot_field_map, plot_obs_map, plot_analysis_map

# Forecast field at T+24h
plot_field_map(plot_data, widgets_dict, 'IFS Control', step=24,
               plot_mode='field', overlay_obs=False,
               plot_radius=4, export_png=True)

# Same field, but extract nearest gridpoint at each station location
plot_field_map(plot_data, widgets_dict, 'IFS Control', step=24,
               plot_mode='station_nearest', overlay_obs=True,
               plot_radius=4, export_png=True)

# Observation and analysis maps
plot_obs_map(plot_data, widgets_dict, plot_radius=3, export_png=True)
plot_analysis_map(plot_data, widgets_dict,
                  overlay_obs=True, plot_radius=3, export_png=True)
```

### Key parameters

- **`step`** — Forecast lead time in hours (e.g. `24`, `48`)
- **`plot_mode`** — `'field'` (gridded shading) or `'station_nearest'`
  (nearest-gridpoint extraction at station locations)
- **`overlay_obs`** — Overlay observation markers on the map
- **`plot_radius`** — Additional degrees around the data area for the map
  view.  Inside `_build_geoview()` in [`map_plotting.py`](diag_evo/map_plotting.py),
  latitude is padded by `plot_radius` and longitude by `plot_radius × 1.5`
  (to compensate for map aspect ratio)
- **`export_png`** — Save as PNG to the plot directory

---

## Run configuration (JSON)

Pressing **Forecast Setup** in the notebook saves a `run_config.json`
under `<base_path>/run_configs/<param>_<date>_<area>/`.  This file
captures all settings (parameter, area, date, models, etc.) and enables:

- **Reproducibility** — re-run the exact same retrieval later
- **Scripted usage** — drive the tool from a Python script instead of the
  notebook UI (see [`examples/run_from_config.py`](examples/run_from_config.py)
  and [`programmatic_workflow.ipynb`](programmatic_workflow.ipynb))
- **Documentation** — keep a record of what was plotted

### Loading a saved configuration

```python
from diag_evo import load_run_config, retrieve_and_store_data

widgets_dict = load_run_config("<base_path>/run_configs/2t_20260407_1200_area_50.91_-1.14_44.91_4.86/run_config.json")
plot_data = retrieve_and_store_data(widgets_dict, base_path="data_files")
```

---

## Static plot — percentile method

The ensemble boxes in `plot_forecast_evolution_static` use the ECMWF /
Metview percentile convention rather than NumPy's default. The rank of
the `P`-th percentile in a sample of size `N` is

```
R = P / 100 * (N + 1)
```

Two interpolation methods are exposed via the `pctl_method` argument:

| `pctl_method` | Behaviour | Equivalent |
|---------------|-----------|------------|
| `'nearest_neighbour'` *(default)* | `P_th = V[round(R)]` | Metview default |
| `'linear'` | `P_th = FR * (V[IR+1] - V[IR]) + V[IR]` | `np.percentile(..., method='weibull')` (Hyndman & Fan method 6) |

```python
plot_forecast_evolution_static(
    plot_data, widgets_dict, plot_dir,
    pctl_method='linear',          # or 'nearest_neighbour' (default)
    export_png=True,
)
```

Note: the Plotly box plot uses Plotly's own (linear) percentile
implementation and is not affected by this argument.

---

## Configuration files

All defaults are driven by three JSON files under
[`diag_evo/config/`](diag_evo/config/).
Edit them to add predefined models, change plot styles, or register new
variables — no Python code changes required.

| File | Purpose | What you can customise |
|------|---------|-----------------------|
| [`model_settings.json`](diag_evo/config/model_settings.json) | MARS retrieval arguments per predefined model | Add / remove models, change `class`, `type`, `stream`, `expver`, `grid`, toggle `ensemble`, set `number` range |
| [`plot_settings.json`](diag_evo/config/plot_settings.json) | Marker styles, colours, box-plot layout for each model | Colours, marker symbols/sizes, line styles, figure dimensions |
| [`variable_settings.json`](diag_evo/config/variable_settings.json) | Variable metadata, units, conversion rules | Display names, unit strings, K→°C offsets, accumulation flags, MARS `param` aliases |

> **Tip:** The model list shown in the UI widget is built directly from
> `model_settings.json` — any model you add there will appear
> automatically (with its MARS keywords shown in the selection list).
> To add models at *runtime* without editing JSON, use
> `register_custom_model()` (see Examples).

---

## Dependencies

See [`requirements.txt`](requirements.txt).  Key dependencies:
All should be preinstalled for ECMWF users.

- Python ≥ 3.9
- `metview`
- `earthkit-data` 
- `ipywidgets`, `ipyleaflet`
- `plotly`, `matplotlib`, `cartopy`
- `numpy`, `pandas`

---

## License

Internal use — ECMWF.

## Contact
Do you want to report issues, suggest improvements or do you have any questions? Please contact [soufiane.karmouche@ecmwf.int](mailto:soufiane.karmouche@ecmwf.int)
