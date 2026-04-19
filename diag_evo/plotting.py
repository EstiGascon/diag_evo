"""
Forecast Evolution Analysis — plotting module.

Interactive (Plotly) and static (Matplotlib) forecast evolution plots.
"""

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from IPython.display import display

from .settings import (
    get_model_retrieval_settings,
    get_model_plot_settings,
    get_reference_plot_settings,
    get_layout_settings,
    _hex_to_rgb,
)
from .core import _model_display_label, _build_ylabel


# ---------------------------------------------------------------------------
# Percentile helpers (ECMWF / Metview convention)
# ---------------------------------------------------------------------------

# Standard percentile levels used in the static plot, ordered as the
# downstream code expects (highest -> lowest).
PCTL_LEVELS = [99, 90, 75, 50, 25, 10, 1]


def _ecmwf_percentiles(values, levels=PCTL_LEVELS, method='nearest_neighbour'):
    """Compute percentiles using the ECMWF / Metview convention.

    Rank formula::

        R = P / 100 * (N + 1)

    with one of two interpolation methods:

    * ``'nearest_neighbour'`` (default, matches Metview): round R to the
      nearest integer rank, ``P_th = V[int(R + 0.5)]``.
    * ``'linear'``: linear interpolation between adjacent ranks,
      ``P_th = FR * (V[IR+1] - V[IR]) + V[IR]``.

    Note this differs from NumPy's default percentile (Hyndman & Fan
    method 7), which uses ``R = P/100 * (N - 1)`` and linear
    interpolation. The ECMWF linear option is equivalent to NumPy's
    ``method='weibull'``.
    """
    arr = np.array([float(v) for v in values
                    if v is not None
                    and not (isinstance(v, float) and np.isnan(v))])
    n = len(arr)
    if n < 2:
        return [np.nan] * len(levels)
    arr.sort()  # ascending; ranks are 1-based

    out = []
    for p in levels:
        r = (p / 100.0) * (n + 1)
        if method == 'nearest_neighbour':
            idx = int(r + 0.5) - 1            # round, convert to 0-based
            idx = max(0, min(n - 1, idx))
            out.append(float(arr[idx]))
        elif method == 'linear':
            ir = int(np.floor(r))
            fr = r - ir
            i_lo = max(0, min(n - 1, ir - 1))      # V[IR] (0-based)
            i_hi = max(0, min(n - 1, ir))          # V[IR+1] (0-based)
            out.append(float(arr[i_lo] + fr * (arr[i_hi] - arr[i_lo])))
        else:
            raise ValueError(
                f"Unknown pctl_method {method!r}. Use 'nearest_neighbour' or 'linear'.")
    return out


# ---------------------------------------------------------------------------
# Colour override helpers (shared by both plot functions)
# ---------------------------------------------------------------------------

def _get_color_overrides(widgets_dict):
    overrides = widgets_dict.get('model_colors', {})
    if not overrides:
        overrides = widgets_dict.get('config', {}).get('model_colors', {})
    return overrides


def _resolve_color(model_name, color_overrides):
    if model_name in color_overrides:
        return color_overrides[model_name]
    mn_lower = model_name.lower()
    for key, color in color_overrides.items():
        if key.lower() in mn_lower or mn_lower in key.lower():
            return color
    ps = get_model_plot_settings(model_name)
    return ps.get('color', '#333333')


def _apply_color_override(plot_settings, model_name, color_overrides):
    """Return a copy of *plot_settings* with colours replaced."""
    color = _resolve_color(model_name, color_overrides)
    ps = dict(plot_settings)
    ps['color'] = color
    rgb = _hex_to_rgb(color)

    if 'box' in ps:
        ps['box'] = dict(ps['box'])
        if rgb:
            r, g, b = rgb
            ps['box']['fillcolor'] = f"rgba({r}, {g}, {b}, 0.3)"
        ps['box']['marker_color'] = color

    if 'marker' in ps and isinstance(ps['marker'], dict):
        ps['marker'] = dict(ps['marker'])
        if 'line' in ps['marker'] and isinstance(ps['marker']['line'], dict):
            ps['marker']['line'] = dict(ps['marker']['line'])
            ps['marker']['line']['color'] = color
        if 'color' in ps['marker'] and rgb:
            r, g, b = rgb
            ps['marker']['color'] = f"rgba({r}, {g}, {b}, 0.7)"

    if 'line' in ps and ps['line'] is not None:
        ps['line'] = dict(ps['line'])
        ps['line']['color'] = color

    return ps


def _resolve_mpl_color(model_name, color_overrides):
    """Return a matplotlib-compatible colour for *model_name*."""
    def _norm(c):
        if isinstance(c, str) and c.startswith('#') and len(c) == 9:
            r = int(c[1:3], 16) / 255.0
            g = int(c[3:5], 16) / 255.0
            b = int(c[5:7], 16) / 255.0
            a = int(c[7:9], 16) / 255.0
            return (r, g, b, a)
        return c

    if model_name in color_overrides:
        return _norm(color_overrides[model_name])
    mn_lower = model_name.lower()
    for key, color in color_overrides.items():
        if key.lower() in mn_lower or mn_lower in key.lower():
            return _norm(color)
    ps = get_model_plot_settings(model_name)
    return _norm(ps.get('color', '#333333'))


# ---------------------------------------------------------------------------
# Interactive Plotly plot
# ---------------------------------------------------------------------------

def plot_forecast_evolution(plot_data, widgets_dict, plot_dir,
                            export_html=False, html_filename=None):
    """Create and display the interactive forecast evolution plot."""
    config = widgets_dict['config']
    valid_date = config['valid_date']
    param = config['param']
    area_sub = config['area_sub']
    point = config.get('point')

    color_overrides = _get_color_overrides(widgets_dict)

    fig = go.Figure()
    data_df = plot_data['data_df'].iloc[::-1].reset_index(drop=True).copy()

    # Determine ensemble models for dynamic x-offset spacing
    _selected = config.get('selected_models') or list(widgets_dict['model_widgets'].value)
    _ensemble_models = [
        m for m in _selected
        if get_model_retrieval_settings(m)['ensemble']
    ]
    _n_ens = len(_ensemble_models)
    _ens_offsets = {}
    if _n_ens == 1:
        _ens_offsets[_ensemble_models[0]] = 0.0
    elif _n_ens > 1:
        total_span = min(0.6, 0.2 * _n_ens)
        for idx, m in enumerate(_ensemble_models):
            _ens_offsets[m] = -total_span / 2 + idx * total_span / (_n_ens - 1)

    # Add traces for each model
    for model_name in _selected:
        model_settings = get_model_retrieval_settings(model_name)
        plot_settings = _apply_color_override(
            get_model_plot_settings(model_name), model_name, color_overrides)

        if model_settings['ensemble']:
            model_data = data_df[data_df[f'{model_name}_ens_area'].notna()].copy()

            if not model_data.empty:
                valid_data = []
                customdata_list = []
                text_list = []

                for _, row in model_data.iterrows():
                    ens_area_data = row[f'{model_name}_ens_area']
                    if ens_area_data is not None and len(ens_area_data) > 0:
                        for member_idx, member_val in enumerate(ens_area_data):
                            if member_val is not None:
                                valid_data.append((row.name, member_val))
                                bias = member_val - (plot_data['reference'] if plot_data['reference'] is not None else 0)
                                customdata_list.append([
                                    row['forecast_step'],
                                    member_idx + 1,
                                    row['forecast_date'].strftime('%d/%m %H:%M'),
                                    bias
                                ])
                                text_list.append(f"Initialisation Date: {row['forecast_date'].strftime('%d/%m %H:%M')}")

                if valid_data:
                    _dlabel = _model_display_label(model_name)
                    fig.add_trace(
                        go.Box(
                            x=[i + _ens_offsets.get(model_name, 0.0) for i, _ in valid_data],
                            y=[val for _, val in valid_data],
                            name=_dlabel,
                            fillcolor=plot_settings['box']['fillcolor'],
                            marker_color=plot_settings['box']['marker_color'],
                            boxpoints='all',
                            jitter=plot_settings['marker']['jitter'],
                            pointpos=plot_settings['marker']['pointpos'],
                            customdata=customdata_list,
                            hovertemplate=(
                                f"Model: {_dlabel}<br>Date: %{{customdata[2]}}<br>"
                                f"Lead Time: %{{customdata[0]}}h<br>Member: %{{customdata[1]}}<br>"
                                f"Value: %{{y:.2f}}{plot_data['var_settings']['units']}"
                                + (f"<br>Fcst-REF: %{{customdata[3]:.2f}}{plot_data['var_settings']['units']}"
                                   if plot_data['reference'] is not None else "")
                                + "<br><extra></extra>"
                            ),
                            hoverinfo='y+name+text',
                            text=text_list,
                            showlegend=True,
                            width=plot_settings['box']['width']
                        )
                    )

                # Ensemble means
                mean_col = f'{model_name}_mean_area'
                if mean_col not in data_df.columns:
                    continue
                mean_data = data_df[data_df[mean_col].notna()].copy()
                if not mean_data.empty:
                    mean_model_name = f"{model_name} Mean"
                    try:
                        mean_plot_settings = _apply_color_override(
                            get_model_plot_settings(mean_model_name), model_name, color_overrides)
                    except Exception:
                        mean_plot_settings = plot_settings

                    mean_customdata = []
                    for _, row in mean_data.iterrows():
                        bias = row[f'{model_name}_mean_area'] - (
                            plot_data['reference'] if plot_data['reference'] is not None else 0)
                        mean_customdata.append([
                            row['forecast_step'],
                            row['forecast_date'].strftime('%d/%m %H:%M'),
                            bias
                        ])

                    _dlabel = _model_display_label(model_name)
                    fig.add_trace(
                        go.Scatter(
                            x=mean_data.index.tolist(),
                            y=mean_data[f'{model_name}_mean_area'].tolist(),
                            mode='lines+markers',
                            name=f'{_dlabel} Mean',
                            line=dict(
                                color=mean_plot_settings['line']['color'],
                                width=mean_plot_settings['line']['width']
                            ),
                            marker=dict(
                                color=mean_plot_settings['marker']['color'],
                                size=mean_plot_settings['marker']['size'],
                                symbol=mean_plot_settings['marker']['symbol'],
                                line=dict(
                                    color=mean_plot_settings['marker']['line']['color'],
                                    width=mean_plot_settings['marker']['line']['width']
                                )
                            ),
                            customdata=mean_customdata,
                            hovertemplate=(
                                f"Initialisation Date: %{{customdata[1]}}<br>"
                                f"Lead Time: %{{customdata[0]}}h<br>"
                                f"Value: %{{y:.2f}}{plot_data['var_settings']['units']}"
                                + (f"<br>Fcst-REF: %{{customdata[2]:.2f}}{plot_data['var_settings']['units']}"
                                   if plot_data['reference'] is not None else "")
                                + f"<extra>Model: {_dlabel} Mean</extra>"
                            ),
                        )
                    )
        else:
            model_data = data_df[data_df[f'{model_name}_area'].notna()].copy()

            if not model_data.empty:
                det_customdata = []
                for _, row in model_data.iterrows():
                    bias = row[f'{model_name}_area'] - (
                        plot_data['reference'] if plot_data['reference'] is not None else 0)
                    det_customdata.append([
                        row['forecast_step'],
                        row['forecast_date'].strftime('%d/%m %H:%M'),
                        bias
                    ])

                mode = 'lines+markers' if plot_settings.get('line') is not None else 'markers'

                _dlabel = _model_display_label(model_name)
                trace_dict = dict(
                    x=model_data.index.tolist(),
                    y=model_data[f'{model_name}_area'].tolist(),
                    mode=mode,
                    name=_dlabel,
                    marker=dict(
                        color=plot_settings['color'],
                        size=plot_settings['marker']['size'],
                        symbol=plot_settings['marker']['symbol'],
                        line=dict(
                            color=plot_settings['marker']['line']['color'],
                            width=plot_settings['marker']['line']['width']
                        )
                    ),
                    customdata=det_customdata,
                    hovertemplate=(
                        f"Initialisation Date: %{{customdata[1]}}<br>"
                        f"Lead Time: %{{customdata[0]}}h<br>"
                        f"Value: %{{y:.2f}}{plot_data['var_settings']['units']}"
                        + (f"<br>Fcst-REF: %{{customdata[2]:.2f}}{plot_data['var_settings']['units']}"
                           if plot_data['reference'] is not None else "")
                        + f"<extra>Model: {_dlabel}</extra>"
                    ),
                )

                if 'lines' in mode and plot_settings.get('line') is not None:
                    trace_dict['line'] = dict(
                        color=plot_settings['line']['color'],
                        width=plot_settings['line']['width']
                    )

                fig.add_trace(go.Scatter(**trace_dict))

    # --- Reference data -------------------------------------------------------
    _add_reference_traces(fig, data_df, plot_data)

    # --- Title ----------------------------------------------------------------
    titre = _build_title_text(valid_date, point, area_sub, plot_data)

    layout_settings = get_layout_settings()
    fig.update_layout(
        title_text=titre,
        showlegend=True,
        height=layout_settings['height'] * 0.8,
        width=layout_settings['width'] * 0.8,
        xaxis=dict(
            ticktext=plot_data['x_labels'][::-1][::2],
            tickvals=list(range(len(plot_data['steps'])))[::2],
            **layout_settings['xaxis'],
            title="Forecast Initialization Date/Time"
        ),
        yaxis_title=_build_ylabel(param, widgets_dict)
    )

    # --- Export ---------------------------------------------------------------
    if export_html:
        if html_filename is None:
            html_filename = _build_export_filename(plot_dir, point, area_sub,
                                                    valid_date, param, '.html')
        fig.write_html(html_filename, include_plotlyjs='cdn')
        print(f"Plot exported to: {html_filename}")

    display(fig)


# ---------------------------------------------------------------------------
# Static Matplotlib plot
# ---------------------------------------------------------------------------

def plot_forecast_evolution_static(plot_data, widgets_dict, plot_dir,
                                    figsize=(22, 9), export_png=False,
                                    png_filename=None,
                                    pctl_method='nearest_neighbour'):
    """Create a static matplotlib forecast evolution plot.

    Parameters
    ----------
    pctl_method : {'nearest_neighbour', 'linear'}, default 'nearest_neighbour'
        Interpolation method used by :func:`_ecmwf_percentiles` to compute
        ensemble percentiles. Matches the Metview / ECMWF convention
        (``R = P/100 * (N + 1)``). Ignored for models that already provide
        precomputed percentiles via the ``<model>_ens_pctls`` column
        (e.g. NB ENS).
    """
    config = widgets_dict['config']
    valid_date = config['valid_date']
    param = config['param']
    area_sub = config['area_sub']
    point = config.get('point')

    color_overrides = _get_color_overrides(widgets_dict)

    data_df = plot_data['data_df'].iloc[::-1].reset_index(drop=True).copy()

    date_labels = data_df['forecast_date'].apply(
        lambda d: (d.strftime('%b ') + str(d.day) + d.strftime(' %Hz'))
        if hasattr(d, 'strftime') else str(d)
    ).tolist()
    xticks = np.arange(len(data_df))

    # Classify models
    selected_models = list(config.get('selected_models') or widgets_dict['model_widgets'].value)
    ensemble_models = []
    deterministic_models = []
    for mn in selected_models:
        ms = get_model_retrieval_settings(mn)
        if ms.get('ensemble', False):
            ensemble_models.append(mn)
        else:
            deterministic_models.append(mn)

    # Slot offsets
    n_ens = len(ensemble_models)
    if n_ens == 0:
        slot_offsets_ens = []
    elif n_ens == 1:
        slot_offsets_ens = [0.0]
    else:
        slot_offsets_ens = np.linspace(-0.24, 0.24, n_ens).tolist()

    ens_slot = {mn: i for i, mn in enumerate(ensemble_models)}

    # Percentile constants
    pctl_levels = PCTL_LEVELS
    pctl_to_idx = {p: i for i, p in enumerate(pctl_levels)}
    box_width = 0.20
    thin_width = 0.10
    median_bar_width = 0.18

    sns.set_theme(style='whitegrid', rc={
        'axes.edgecolor': '.3',
        'grid.color': '.85',
        'axes.grid.axis': 'y',
    })
    fig_mpl, ax = plt.subplots(figsize=figsize)

    # --- Ensemble percentile boxes ---
    ens_handles = []
    for model_name in ensemble_models:
        color = _resolve_mpl_color(model_name, color_overrides)
        si = ens_slot[model_name]
        offset = slot_offsets_ens[si]

        # Models such as NB ENS provide precomputed percentiles directly;
        # use them as-is rather than recomputing from a synthetic member
        # expansion.
        pctl_col = f'{model_name}_ens_pctls'
        ens_col = f'{model_name}_ens_area'
        pctl_matrix = []
        for _, row in data_df.iterrows():
            raw_pctls = row.get(pctl_col) if pctl_col in data_df.columns else None
            if isinstance(raw_pctls, (list, np.ndarray)) and len(raw_pctls) == len(pctl_levels):
                pctl_matrix.append([float(v) for v in raw_pctls])
                continue
            vals = row.get(ens_col)
            if isinstance(vals, (list, np.ndarray)) and len(vals) > 0:
                pctl_matrix.append(_ecmwf_percentiles(vals, pctl_levels, pctl_method))
            else:
                pctl_matrix.append([np.nan] * len(pctl_levels))
        pctl_matrix = np.array(pctl_matrix)
        xpos = xticks + offset

        for idx, x in enumerate(xpos):
            p = pctl_matrix[idx]
            if np.all(np.isnan(p)):
                continue
            lo25, hi75 = p[pctl_to_idx[25]], p[pctl_to_idx[75]]
            ax.add_patch(plt.Rectangle(
                (x - box_width / 2, lo25), box_width, hi75 - lo25,
                facecolor=color, alpha=1, lw=0.8, edgecolor='black', zorder=7))
            lo10, hi90 = p[pctl_to_idx[10]], p[pctl_to_idx[90]]
            ax.add_patch(plt.Rectangle(
                (x - thin_width / 2, lo10), thin_width, hi90 - lo10,
                facecolor=color, alpha=1, lw=0.8, edgecolor='black', zorder=6))
            ax.plot([x, x], [p[pctl_to_idx[1]], p[pctl_to_idx[99]]],
                    color=color, lw=2.2, solid_capstyle='round', zorder=5)
            ax.plot([x - median_bar_width / 2, x + median_bar_width / 2],
                    [p[pctl_to_idx[50]], p[pctl_to_idx[50]]],
                    color='black', lw=3, zorder=8)

        _dlabel = _model_display_label(model_name)
        ens_handles.append(plt.Line2D([0], [0], color=color, lw=10, alpha=1,
                                      label=f'{_dlabel} \n(1/10/25/50/75/90/99th)'))

    _mpl_symbol_map = {
        'triangle-down': 'v', 'triangle-up': '^', 'diamond': 'D',
        'circle': 'o', 'square': 's', 'star': '*', 'cross': 'x',
        'x': 'X', 'pentagon': 'p', 'hexagon': 'h',
    }

    scatter_handles = []

    # Ensemble means
    for model_name in ensemble_models:
        color = _resolve_mpl_color(model_name, color_overrides)
        si = ens_slot[model_name]
        offset = slot_offsets_ens[si]
        mean_col = f'{model_name}_mean_area'
        if mean_col in data_df.columns:
            ys = pd.to_numeric(data_df[mean_col], errors='coerce').values
            mask = ~np.isnan(ys)
            if mask.any():
                xs = xticks[mask] + offset
                _dlabel = _model_display_label(model_name)
                h = ax.scatter(xs, ys[mask], facecolor=color, marker='*',
                               edgecolor='black', alpha=1, s=190, zorder=8,
                               label=f'{_dlabel} Mean')
                scatter_handles.append(h)

    # Deterministic models
    for model_name in deterministic_models:
        color = _resolve_mpl_color(model_name, color_overrides)
        ps = get_model_plot_settings(model_name)
        plotly_sym = ps.get('marker', {}).get('symbol', 'circle')
        mpl_marker = _mpl_symbol_map.get(plotly_sym, 'o')

        col = f'{model_name}_area'
        if col in data_df.columns:
            ys = pd.to_numeric(data_df[col], errors='coerce').values
            mask = ~np.isnan(ys)
            if mask.any():
                _dlabel = _model_display_label(model_name)
                
                h = ax.scatter(xticks[mask], ys[mask], facecolor=color,
                               marker=mpl_marker, edgecolor='black', alpha=0.85,
                               s=160 if mpl_marker != 's' else 130, zorder=10, label=_dlabel)
                scatter_handles.append(h)

    # Reference data
    ref_specs = [
        ('observations', 'Observations', '#000000', 'o', 160),
        ('analysis', 'Analysis', '#7f7f7f', '^', 160),
        ('climatology', 'Climatology Mean', '#7f7f7f', '*', 160),
    ]
    for col, label, color, marker, ms in ref_specs:
        if col in data_df.columns:
            ys = pd.to_numeric(data_df[col], errors='coerce').values
            mask = ~np.isnan(ys)
            if mask.any():
                h = ax.scatter(xticks[mask], ys[mask], facecolor=color, marker=marker,
                               edgecolor='black', alpha=0.85, s=ms, zorder=10, label=label)
                scatter_handles.append(h)

    # Vertical separators
    for i in range(len(data_df)):
        ax.axvline(x=xticks[i] - 0.5, color='gray', alpha=0.3, linewidth=0.8, zorder=1)
        ax.axvline(x=xticks[i] + 0.5, color='gray', alpha=0.3, linewidth=0.8, zorder=1)

    # Axes formatting
    ax.set_xticks(xticks)
    ax.set_xticklabels(date_labels, rotation=70, ha='right', fontsize=15)
    ax.set_xlabel('Forecast Initialization Date/Time', fontsize=16)
    ax.set_ylabel(_build_ylabel(param, widgets_dict), fontsize=22)
    ax.tick_params(axis='y', labelsize=15)

    titre = _build_title_text(valid_date, point, area_sub, plot_data, html=False)
    ax.set_title(titre, fontsize=18)

    # Legend
    legend_entries = list(ens_handles)
    seen = set()
    for h in scatter_handles:
        lbl = h.get_label()
        if lbl not in seen and lbl != '_nolegend_':
            legend_entries.append(h)
            seen.add(lbl)

    _right_x = 0.83
    _right_w = 0.16
    fig_mpl.subplots_adjust(left=0.06, right=_right_x - 0.01, top=0.93, bottom=0.22)

    # Legend at top-right
    leg = ax.legend(legend_entries,
                    [e.get_label() for e in legend_entries],
                    loc='upper left',
                    bbox_to_anchor=(_right_x, 0.93),
                    bbox_transform=fig_mpl.transFigure,
                    fontsize=11, frameon=False)

    # Determine where the legend ends (in figure coords) so the inset
    # map is placed below it without overlap.
    fig_mpl.canvas.draw()  # force layout so legend bbox is computed
    leg_bbox_fig = leg.get_window_extent().transformed(
        fig_mpl.transFigure.inverted())
    _map_h = 0.20
    _map_top = max(leg_bbox_fig.y0 - 0.02, _map_h + 0.02)

    # Inset map placed below the legend
    _draw_inset_map(fig_mpl, area_sub, point, _right_x, _right_w,
                    map_y=_map_top - _map_h, map_h=_map_h)

    # Export
    if export_png:
        if png_filename is None:
            png_filename = _build_export_filename(
                plot_dir, point, area_sub, valid_date, param, '.png',
                prefix='forecast_evolution_static')
        fig_mpl.savefig(png_filename, dpi=150, bbox_inches='tight')
        print(f"Static plot exported to: {png_filename}")

    plt.close(fig_mpl)
    return fig_mpl


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_reference_traces(fig, data_df, plot_data):
    """Add observation / analysis / climatology traces to a Plotly figure."""
    units = plot_data['var_settings']['units']
    ref_configs = [
        ('observations', 'Observations'),
        ('analysis', 'Analysis'),
        ('climatology', 'Climatology Mean'),
    ]
    for col, label in ref_configs:
        ref_data = data_df[data_df[col].notna()].copy()
        if ref_data.empty:
            continue
        ref_settings = get_reference_plot_settings(label)
        fig.add_trace(
            go.Scatter(
                x=ref_data.index.tolist(),
                y=ref_data[col].tolist(),
                mode='markers',
                name=label,
                marker=dict(
                    color=ref_settings['color'],
                    size=ref_settings['marker']['size'],
                    symbol=ref_settings['marker']['symbol'],
                    line=dict(
                        color=ref_settings['marker']['line']['color'],
                        width=ref_settings['marker']['line']['width']
                    )
                ),
                hovertemplate=f"{label}: %{{y:.2f}}{units}<extra></extra>",
                customdata=[[row['forecast_step']] for _, row in ref_data.iterrows()],
            )
        )


def _build_title_text(valid_date, point, area_sub, plot_data, html=True):
    """Build the plot title string."""
    sep = "<br>" if html else "\n"
    date_str = valid_date.strftime('%b ') + str(valid_date.day) + valid_date.strftime(' %Hz %Y')
    if point:
        titre = (f"Forecast Evolution (Valid: {date_str}) "
                 f"at {point[0]:.2f}\u00b0N, {point[1]:.2f}\u00b0E")
        ns = plot_data.get('nearest_gridinfo_dict', {}).get('nearest_station')
        if ns is not None:
            titre += (f"{sep}Nearest station {ns['stnid']} "
                      f"(elev {ns['elevation']} m, "
                      f"{ns['latitude']:.2f}\u00b0N, {ns['longitude']:.2f}\u00b0E, "
                      f"{ns['distance']:.1f} km, val {ns['value_0']:.2f})")
    else:
        titre = (f"Forecast Evolution (Valid: {date_str}) "
                 f"for [{area_sub[0]:.2f}\u00b0N, {area_sub[1]:.2f}\u00b0E] – "
                 f"[{area_sub[2]:.2f}\u00b0N, {area_sub[3]:.2f}\u00b0E]")
    return titre


def _build_export_filename(plot_dir, point, area_sub, valid_date, param,
                            ext, prefix='forecast_evolution'):
    """Build a default export filename."""
    if point:
        return (f"{plot_dir}/{prefix}_point_"
                f"{point[0]:.4f}N_{point[1]:.4f}E_"
                f"{valid_date.strftime('%Y%m%d_%H%M')}_{param}{ext}")
    return (f"{plot_dir}/{prefix}_area_"
            f"{area_sub[0]:.4f}N_{area_sub[1]:.4f}E_"
            f"{area_sub[2]:.4f}N_{area_sub[3]:.4f}E_"
            f"{valid_date.strftime('%Y%m%d_%H%M')}_{param}{ext}")


def _draw_inset_map(fig_mpl, area_sub, point, right_x, right_w,
                    map_y=None, map_h=None):
    """Draw the cartopy inset map on the right side of the figure."""
    _map_h = map_h if map_h is not None else 0.28
    _map_y = map_y if map_y is not None else 0.12
    map_n, map_w, map_s, map_e = area_sub
    center_lat = (map_n + map_s) / 2.0
    center_lon = (map_w + map_e) / 2.0
    lat_span = max(abs(map_n - map_s), 2.0) * 2.5
    lon_span = max(abs(map_e - map_w), 2.0) * 2.5

    axins = fig_mpl.add_axes(
        [right_x, _map_y, right_w, _map_h],
        projection=ccrs.Mercator())
    axins.set_extent([
        center_lon - lon_span / 2, center_lon + lon_span / 2,
        center_lat - lat_span / 2, center_lat + lat_span / 2,
    ], crs=ccrs.PlateCarree())
    axins.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='black')
    axins.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
    axins.add_feature(cfeature.LAND, facecolor='whitesmoke', zorder=0)

    gl = axins.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                         linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}

    if point:
        axins.plot(point[1], point[0], marker='*', color='red', markersize=14,
                   markeredgecolor='black', markeredgewidth=0.8,
                   transform=ccrs.PlateCarree(), zorder=10)
    else:
        import matplotlib.patches as mpatches
        rect = mpatches.Rectangle(
            (map_w, map_s), map_e - map_w, map_n - map_s,
            linewidth=2.0, edgecolor='red', facecolor='none',
            transform=ccrs.PlateCarree(), zorder=10)
        axins.add_patch(rect)
