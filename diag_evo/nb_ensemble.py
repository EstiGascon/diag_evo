"""
Neighbourhood Ensemble (NB ENS) integration for diag_evo.

The NB ensemble provides precomputed percentile forecasts
(p1, p10, p25, p50, p75, p90, p99) for 6-hourly accumulated total
precipitation. This module extracts those percentiles at the
configured point (or area integral) and expands them into a synthetic
member distribution so the existing plotting functions recover the
correct quantiles.
"""

import warnings

import metview as mv
import numpy as np
import pandas as pd

from .settings import register_custom_model


# Percentile files are stored highest → lowest: [99, 90, 75, 50, 25, 10, 1]
_NB_PCTLS = ["99", "90", "75", "50", "25", "10", "1"]
_NB_ENS_NAME = "NB ENS"
_NB_ENS_COLOR = "#7f7f7f"


def _expand_nb_pctls(pctls_desc):
    """Expand 7 precomputed percentiles to 51 synthetic members.

    ``pctls_desc`` is ordered highest → lowest (as stored in the files).
    The output is interpolated so ``np.percentile`` recovers the original
    quantile values.
    """
    p = sorted(float(v) for v in pctls_desc)  # ascending
    q_knots = [0,    1,    10,   25,   50,   75,   90,   99,   100]
    v_knots = [p[0], p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[6]]
    return list(np.interp(np.linspace(0, 100, 51), q_knots, v_knots))


def add_nb_ens_to_plot_data(plot_data, widgets_dict, nb_path):
    """Retrieve NB ENS percentile data and inject it into ``plot_data``.

    Only supported for parameters whose name contains ``'tp'`` (6-hourly
    accumulated precipitation). Rows from non-00 UTC initialisations are
    left as NaN.

    Parameters
    ----------
    plot_data : dict
        Output from :func:`retrieve_and_store_data`.
    widgets_dict : dict
        Widget / config dict (requires ``config['param']`` and either
        ``config['point']`` or ``config['area_sub']``).
    nb_path : str
        Directory containing files named
        ``NB_tp6h_p{pctl}.0_rVR_ST0_{YYYYMMDD}_step_6-120.grib2``.

    Returns
    -------
    plot_data_nb : dict
        Copy of ``plot_data`` with ``'NB ENS_ens_area'`` column added.
    widgets_dict : dict
        Same dict, with ``'NB ENS'`` appended to ``config['selected_models']``.
    """
    config = widgets_dict['config']
    param = config.get('param', '')
    if 'tp' not in param.lower():
        warnings.warn(
            f"NB ENS is only supported for 6h accumulated TP, not '{param}'. Skipping.",
            UserWarning,
        )
        return plot_data, widgets_dict

    point = config.get('point')
    area_sub = config.get('area_sub')

    def _compute_row(row):
        dt = pd.to_datetime(row['forecast_date'])
        if dt.strftime('%H:%M') != '00:00':
            return pd.Series({'NB ENS_ens_area': np.nan, 'NB ENS_ens_pctls': np.nan})
        step = int(row['forecast_step'])
        if step <= 0:
            return pd.Series({'NB ENS_ens_area': np.nan, 'NB ENS_ens_pctls': np.nan})

        ymd = dt.strftime('%Y%m%d')
        step_range = f'{step - 6}-{step}'
        values = []

        for perc in _NB_PCTLS:
            try:
                path = f"{nb_path}/NB_tp6h_p{perc}.0_rVR_ST0_{ymd}_step_6-120.grib2"
                datanb = mv.read(path)
                datanb = datanb.select(stepRange=step_range)
                if point:
                    val = float(mv.nearest_gridpoint(datanb, point))
                else:
                    val = float(mv.integrate(datanb, area_sub))
                values.append(val)
            except Exception:
                return pd.Series({'NB ENS_ens_area': np.nan, 'NB ENS_ens_pctls': np.nan})

        # Synthetic 51-member expansion drives the Plotly box plot; the raw
        # 7 percentiles (in [99, 90, 75, 50, 25, 10, 1] order — same as
        # PCTL_LEVELS in plotting.py) are kept so the static plot can use
        # them directly without recomputing from the synthetic distribution.
        return pd.Series({
            'NB ENS_ens_area': _expand_nb_pctls(values),
            'NB ENS_ens_pctls': [float(v) for v in values],
        })

    df = plot_data['data_df'].copy()
    print(f"Computing NB ENS for {len(df)} rows (00 UTC initialisations only)…")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        nb_cols = df.apply(_compute_row, axis=1)
    df['NB ENS_ens_area'] = nb_cols['NB ENS_ens_area']
    df['NB ENS_ens_pctls'] = nb_cols['NB ENS_ens_pctls']

    n_ok = df['NB ENS_ens_area'].apply(lambda x: isinstance(x, list)).sum()
    print(f"Done — {n_ok}/{len(df)} rows populated.")

    # Register NB ENS as a custom ensemble model (idempotent; dummy MARS args).
    register_custom_model(
        name=_NB_ENS_NAME,
        retrieval_args={'class': 'rd', 'type': 'pf', 'stream': 'oper', 'expver': 'iekm'},
        is_ensemble=True,
        color=_NB_ENS_COLOR,
    )

    # Update config['selected_models'] so plotting functions include NB ENS.
    current_sel = list(config.get('selected_models', ()))
    if _NB_ENS_NAME not in current_sel:
        current_sel.append(_NB_ENS_NAME)
    config['selected_models'] = tuple(current_sel)

    return {**plot_data, 'data_df': df}, widgets_dict
