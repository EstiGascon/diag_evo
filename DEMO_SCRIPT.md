# Forecast Evolution Tool — Demo Script

---

## Part 1 — Introduction to Jupyter Notebooks 

> Open `intro_to_jupyter.ipynb`

##### 1.1. Log in to JupyterHub (use your ECMWF credentials)
##### 1.2. Start Server (select memory, CPU, temporary storage)
##### 1.3. Open a Terminal (File → New → Terminal)
##### 1.4. Clone the repo
```bash
git clone https://git.ecmwf.int/scm/~ecm3468/diag_evo.git diag_evo
```
##### 1.5. Open `intro_to_jupyter.ipynb` — walk through:
- Code cells vs Markdown cells
- Running cells (`Shift+Enter`)
- Variables persist between cells
- Interactive widgets (slider example)
- Execution order matters

##### 1.6. Select kernel (Python ≥ 3.9, e.g. Python 3.12.11)

##### Quick tips to mention:
- **Restart kernel:** restart button, or stop + start kernel
- **Restart server:** File → Hub Control Panel → Stop Server → refresh → start again
- **Storage:** keep repo on `$PERM` (or `$HOME`), set data output to `$SCRATCH`

---

## Part 2 — Forecast Evolution Notebook 

> Open `forecast_evolution.ipynb`

> **Refer to `README.md` for full documentation**

---

### 2.1. Setup Cell (Cell 2)

- Replace the dummy path with your path to the repo:
  ```python
  notebook_dir = "/perm/<username>/diag_evo"
  sys.path.insert(0, notebook_dir)
  ```
- Set `base_path` to your scratch directory:
  ```python
  base_path = "/scratch/<username>/data_files"
  ```
- Run the cell → the **interactive UI** appears

---

### 2.2. Configure the Retrieval Using the UI

#### Parameter Selection
- Free-text input: type a shortName (e.g. `2t`, `tp`, `10fg`, `z500`) or a paramId number (e.g. `167`)
- If **pressure-level** variable: levtype switches to `pl`, level field appears
- If **accumulated** variable (e.g. `tp`): accumulation period selector appears automatically (6h, 12h, 24h)
- If **surface variable available on STVL** (e.g. `2t`, `10fg`, `tp`): a **"Show station observations"** button appears automatically — click it to overlay colour-coded STVL stations on the map for the selected parameter and valid date. The colorbar is truncated (1st–99th percentile to avoid outliers); min and max are shown as larger scatter markers.

#### Forecast Settings
- **Valid date** — calendar picker
- **Valid time** — dropdown (00, 06, 12, 18 UTC)
- **Max forecast days** — slider (1 to 15)
- **Step interval** — frequency of initialisation (6h, 12h, or 24h)

#### Model Selection

**Predefined models** (left panel):
- Select one or more from the list (IFS Control, AIFS Single, IFS ENS, AIFS ENS, DE-ATOS, etc.)
- Each entry shows its MARS keywords (`class/type/stream`) for identification
- List is driven by `diag_evo/config/model_settings.json` — edit to add/remove models permanently

**Custom MARS models** (right panel):
- Provide a name + MARS keywords: `class`, `type`, `stream`, `expver`
- Tick **Ensemble** if perturbed forecast
- Add extra key-value pairs (e.g. `model=aifs-single`, `database=...`, `grid=[0.25,0.25]`)
- Click **Add Model**

#### Area Averaging Mode
- **Area integral (weighted)** — uses `mv.integrate()` over the bounding box
- **Station nearest gridpoints** — extracts model values at STVL station locations inside the box, then averages

#### Region / Point Selection (interactive map)

**Area selection:**
- Draw a polygon/rectangle on the map → bounding box is computed from northernmost/westernmost/southernmost/easternmost points
- Or pick from **Standard Regions** dropdown (Europe, Global, Mediterranean, etc.)

**Point selection:**
- Click a single point on the map → system creates a ±3° bounding box around it
- Nearest observation station is identified

#### Forecast Setup Button
- Click **Forecast Setup** to finalise the configuration
- This saves a `run_config.json` to `<base_path>/run_configs/<param>_<date>_<area>/`
- The config can later be used from command line (see `examples/run_from_config.py`)

---

### 2.3. Retrieve Data (Cell 8)

```python
plot_data = retrieve_and_store_data(widgets_dict, base_path)
```

**What happens:**
- Retrieves observations (STVL), analysis, ERA5 climatology, and model forecasts via MARS
- Data cached locally as `.grib` (forecasts) and `.csv` (obs) — re-running is instant
- Each model × step is retrieved individually
- **Timeout:** 180 seconds per MARS request (`MARS_RETRIEVAL_TIMEOUT` in `core.py`). If a model times out, remaining steps for that model are skipped.
- Area is automatically expanded by 4° for retrieval (to allow map plot margins)

---

### 2.4. Interactive Plot — Plotly (Cell 10)

```python
plot_forecast_evolution(plot_data, widgets_dict, plot_dir, export_html=True)
```

- Box-and-whiskers plot showing forecast evolution
- Hover info: bias/error, lead time, member index
- Exported as standalone HTML file
- **Note:** Plotly uses standard IQR (25th–median–75th percentile). Outliers follow Plotly's definition (>1.5×IQR beyond Q1/Q3)

---

### 2.5. Static Plot — Matplotlib (Cell 12)

```python
plot_forecast_evolution_static(plot_data, widgets_dict, plot_dir, pctl_method='nearest_neighbour', export_png=True)
```

- Percentile-box view: **1/10/25/50/75/90/99** percentiles
- Inset map showing the selected area or point
- Exported as PNG

**Percentile method** (`pctl_method` argument):
- `'nearest_neighbour'` *(default)* — matches the ECMWF/Metview convention: `R = P/100 × (N+1)`, rounded to nearest rank
- `'linear'` — linear interpolation, equivalent to `np.percentile(..., method='weibull')`

**Note:** The static plot percentiles differ from the Plotly box plot (which uses Plotly's own linear implementation).

---

### 2.6. Map Plots — Metview (Cells 13–17)

Three map functions available:

| Function | Description |
|----------|-------------|
| `plot_field_map` | Model forecast field at a given lead time |
| `plot_obs_map` | Observation station locations and values |
| `plot_analysis_map` | Analysis field with optional observation overlay |

**Key arguments for `plot_field_map`:**

| Argument | Options | Description |
|----------|---------|-------------|
| `plot_mode` | `'field'` / `'station_nearest'` | Gridded shading vs. nearest-gridpoint at stations |
| `overlay_obs` | `True` / `False` | Overlay observation markers |
| `add_markers` | `True` / `False` | Add point markers |
| `plot_radius` | float (degrees) | Extra padding around the area for map extent |
| `model` | string | Which model to plot |
| `step` | int (hours) | Lead time |
| `member` | int | Specific ensemble member (for ENS models) |
| `export_png` | `True` / `False` | Save to file |

**Examples to show:**
```python
# Gridded field with obs overlay
plot_field_map(plot_data, widgets_dict, "IFS Control", step=36,
               plot_mode='field', overlay_obs=True, plot_radius=2, export_png=True)

# Station-nearest mode
plot_field_map(plot_data, widgets_dict, "IFS Control", step=36,
               plot_mode='station_nearest', overlay_obs=False, plot_radius=2, export_png=True)

# Specific ensemble member
plot_field_map(plot_data, widgets_dict, "AIFS ENS", member=31, step=48,
               plot_mode='field', overlay_obs=True, plot_radius=0, export_png=True)

# Observation and analysis maps
plot_obs_map(plot_data, widgets_dict, plot_radius=3, export_png=True)
plot_analysis_map(plot_data, widgets_dict, overlay_obs=True, plot_radius=3, export_png=True)
```

---

### 2.7. OPTIONAL — Override Default Colours (Cells 18–21)

```python
widgets_dict['config']['model_colors'] = {
    'IFS ENS': '#199fff',
    'DE-ATOS': '#9467bd',
    'AIFSv2 Single': '#e6194b',
    'AIFSv2 ENS': '#008b27',
}
# Then re-run the plot cells
plot_forecast_evolution_static(plot_data, widgets_dict, plot_dir, export_png=True)
```

---

### 2.8. OPTIONAL — Neighbourhood Ensemble / NB ENS (Cells 22–23)

> **Only available for 6-hourly accumulated precipitation (`tp`).**

```python
from diag_evo import add_nb_ens_to_plot_data

nb_path = "/ec/vol/destine/neighbourhood/percentile"
plot_data_nb, widgets_dict = add_nb_ens_to_plot_data(plot_data, widgets_dict, nb_path)

plot_forecast_evolution(plot_data_nb, widgets_dict, plot_dir, export_html=True)
plot_forecast_evolution_static(plot_data_nb, widgets_dict, plot_dir, export_png=True)
```

**How it works:**
- Reads precomputed percentile GRIB files (`NB_tp6h_p{pctl}.0_rVR_ST0_{YYYYMMDD}_step_6-120.grib2`)
- Extracts values at the configured point/area
- Expands 7 percentiles to synthetic 51-member distribution for the Plotly plot
- The **static plot** uses the original 7 percentiles directly (no recomputation)
- Only 00 UTC model initialisations are processed; 06/12/18 UTC rows are NaN

---

## Part 3 — Programmatic / Command-Line Usage 

> Can be explored independently via `programmatic_workflow.ipynb` and `examples/`

- **Reproduce from saved config:**
  ```bash
  python examples/run_from_config.py /path/to/run_config.json
  ```
- **From scratch (no UI):** use `build_widgets_dict(config)` — see examples:
  - `examples/run_area_t850_central_europe.py` — pressure-level, area mode
  - `examples/run_point_precip_lisbon.py` — tp accumulation, NB ENS, point mode
  - `examples/run_area_wind_france.py` — custom model, colour overrides

---

## Configuration Files (mention)

All defaults are in `diag_evo/config/`:

| File | Controls |
|------|----------|
| `model_settings.json` | Predefined models and MARS retrieval args |
| `plot_settings.json` | Marker styles, colours, box-plot layout per model |
| `variable_settings.json` | Variable metadata: display names, units, conversion, accumulation flags |

Edit these to customise without changing Python code.

---

## Key Notes

- **Retrieval timeout:** 180 sec per MARS request (configurable in `core.py` via `MARS_RETRIEVAL_TIMEOUT`). If a model fails, remaining steps for that model are skipped.
- **Plotting settings** can be manually edited in `plot_settings.json`
- **Plotly vs static box plots differ:** Plotly uses standard IQR; static uses ECMWF/Metview percentile convention
- **Projection:** Map plots use `POLAR_STEREOGRAPHIC` projection
- **Longitude normalisation:** Out-of-range longitudes (from map scrolling) are automatically wrapped to [-180, 180]

---

## Troubleshooting Tips

| Issue | Solution |
|-------|----------|
| Kernel stuck | Restart button, or Kernel → Restart |
| Server unresponsive | File → Hub Control Panel → Stop Server → refresh → start |
| `NameError` | Run cells in order from the top |
| Missing packages | `pip install -r requirements.txt` |
| Retrieval timeout | Data source may be temporarily unavailable; retry later |
| Play button = run cell | `Shift+Enter` also works |
