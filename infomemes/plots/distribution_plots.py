from infomemes.plots.styles import palettes
from infomemes.plots.functions import AutoDictionary
from infomemes.plots.layouts import lay_base

import plotly
import plotly.graph_objs as go
from sklearn.neighbors import KernelDensity
import numpy as np
import itertools


def distribution_plots(df_sims, df_media, df_clusters, x_var='survival_time',
                       group_by='cp_category', hist_type='kde', kde_width=None,
                       xlim=None, palette=None):
    """
    Makes the 9-tiles distribution plots.

    Parameters
    ----------
    df_sims: DataFrame
        Pandas DataFrame with results by simulation
    df_media: DataFrame
        Pandas DataFrame with results by media
    df_clusters: DataFrame
        Pandas DataFrame with results by cluster
    x_var: string
        Name of the simulation/media variable to be plotted
    group_by: string
        Name of the simulation variable used to group the distributions
    hist_type: string
        'kde' or 'hist'
    """
    # List of colors for multiple groups, if group_by!=None
    if palette is None:
        palette = palettes['palette_0']

    figs = plotly.subplots.make_subplots(rows=3, cols=3, print_grid=False,
                                         horizontal_spacing=.02, vertical_spacing=.03)

    # Figure initial layout
    for xa in figs.select_xaxes():
        xa = xa.update(lay_base['xaxis'])
    for ya in figs.select_yaxes():
        ya = ya.update(lay_base['yaxis'])
    figs.layout['plot_bgcolor'] = "#ffffff"
    fig_title = ''
    if x_var == 'n_clusters':
        fig_title = 'number of clusters'
    elif x_var == 'n_members':
        fig_title = 'number of members'
    elif x_var == 'survival_time':
        fig_title = 'survival time'
    elif x_var == 'meme_production_rate':
        fig_title = 'meme production rate'
    elif x_var == 'cov_x':
        fig_title = 'autocovariance x'
    elif x_var == 'cov_y':
        fig_title = 'autocovariance y'
    elif x_var == 'cov_xy':
        fig_title = 'covariance xy'
    title = {
        'text': fig_title,
        'font': {"family": "Arial", "size": 23},
        'x': 0.5,
        'xanchor': 'center',
        'xref': 'paper',
        'y': 1.1,
        'yanchor': 'bottom',
        'yref': 'paper',
        'showarrow': False,
    }
    figs.add_annotation(title)

    # If countable variable
    if x_var in ['n_clusters']:
        hist_type = 'hist'
    # elif x_var in ['n_members']:
    #     hist_type = 'hist'

    # If using KDE smoothing
    if hist_type == 'kde' and kde_width is None:
        if x_var in ['n_members']:
            kde_width = 5 * df_clusters[x_var].std() / np.abs(df_clusters[x_var].mean())
        else:
            kde_width = 0.1 * df_media[x_var].std() / np.abs(df_media[x_var].mean())

    # X limits
    if xlim is None:
        if x_var in ['n_clusters']:
            xx_min = 0
            xx_max = df_sims[x_var].max() + 1
        elif x_var in ['n_members']:
            xx_min = 0
            xx_max = df_clusters[x_var].max() + 1
        else:
            xx_min = df_media[x_var].min()
            xx_max = df_media[x_var].max() + 1
    else:
        xx_min = xlim[0]
        xx_max = xlim[1]
    xx = np.linspace(xx_min, xx_max, 100)

    # Groupby groups names
    groups_names = ['NA']
    if group_by is not None:
        groups_names = list(df_sims[group_by].unique())

    # Categories to separate subplots
    yy = AutoDictionary()
    yy_max = 0
    m_dyn = df_sims['media_dynamics'].unique()
    i_dyn = df_sims['individual_dynamics'].unique()
    for md, id in itertools.product(m_dyn, i_dyn):
        for jj, grp in enumerate(groups_names):
            if len(groups_names) == 1:
                mask_sims = (df_sims['media_dynamics'] == md) & (df_sims['individual_dynamics'] == id)
                group_key = 'NA'
            else:
                mask_sims = (df_sims['media_dynamics'] == md) & (df_sims['individual_dynamics'] == id) & (df_sims[group_by] == grp)
                group_key = grp

            # Variable from simulations, clusters or media dataframe
            sims_indexes = df_sims[mask_sims].index
            if x_var in ['n_clusters']:
                y = df_sims[mask_sims][x_var].to_numpy()
            elif x_var in ['n_members']:
                mask_clusters = df_clusters['simulation'].isin(sims_indexes)
                y = df_clusters[mask_clusters][x_var].to_numpy()
            else:
                mask_media = (df_media['simulation'].isin(sims_indexes)) & (df_media['deactivated'] < 999)
                y = df_media[mask_media][x_var].to_numpy()

            # KDE or Hist
            if hist_type == 'kde':
                kde = KernelDensity(kernel='gaussian', bandwidth=kde_width).fit(y.reshape(-1, 1))
                log_dens = kde.score_samples(xx.reshape(-1, 1))
                yy[md][id][group_key] = np.exp(log_dens)
                # Plots
                trace_pdf = {
                    "fill": "tonexty",
                    "line": {
                        "color": "#000000",
                        "shape": "spline",
                        "width": 0.5
                    },
                    "mode": "lines",
                    "name": grp,
                    "type": "scatter",
                    "x": xx.tolist(),
                    "y": yy[md][id][group_key].tolist(),
                    "fillcolor": palette[jj],
                    "legendgroup": grp,
                    "showlegend": [True if (md == 0) and (id == 0) and (len(groups_names) > 1)
                                   else False][0],
                }
                figs.add_trace(go.Scatter(trace_pdf), row=3 - id, col=md + 1)
            elif hist_type == 'hist':
                yy_aux, xx = np.histogram(y, bins=np.linspace(xx_min, xx_max, xx_max + 1))
                yy[md][id][group_key] = yy_aux / sum(yy_aux)
                # Plots
                trace_pdf = {
                    "name": grp,
                    "x": xx.tolist(),
                    "y": yy[md][id][group_key].tolist(),
                    "marker_color": palette[jj],
                    "legendgroup": grp,
                    "showlegend": [True if (md == 0) and (id == 0) and (len(groups_names) > 1)
                                   else False][0],
                }
                figs.add_trace(go.Bar(trace_pdf), row=3 - id, col=md + 1)
            yy_max = max(yy_max, max(yy[md][id][group_key]))

    # Update layouts
    for xa in figs.select_xaxes():
        x_label = ''
        showticklabels = False
        if xa.plotly_name == 'xaxis7':
            x_label = "low<br><a> </a>"
            showticklabels = True
        elif xa.plotly_name == 'xaxis8':
            x_label = "medium<br>media dynamics"
            showticklabels = True
        elif xa.plotly_name == 'xaxis9':
            x_label = "high<br><a> </a>"
            showticklabels = True
        xa = xa.update({
            "title": {'text': x_label, 'font': {"family": "Balto", "size": 20}},
            "type": "linear",
            "range": [xx_min, xx_max],
            "autorange": False,
            "showgrid": True,
            "gridcolor": "#d4d4d4",
            "gridwidth": .2,
            "showticklabels": showticklabels,
            "nticks": 5,
        })
    for ya in figs.select_yaxes():
        y_label = ''
        showticklabels = False
        if ya.plotly_name == 'yaxis':
            y_label = "<a> </a><br>high"
            showticklabels = True
        elif ya.plotly_name == 'yaxis4':
            y_label = "individual dynamics<br>medium"
            showticklabels = True
        elif ya.plotly_name == 'yaxis7':
            y_label = "<a> </a><br>low"
            showticklabels = True
        ya = ya.update({
            "title": {'text': y_label, 'font': {"family": "Balto", "size": 20}},
            "type": "linear",
            "visible": True,
            "color": "#000000",
            "showgrid": True,
            "gridcolor": "#d4d4d4",
            "gridwidth": .2,
            "range": [0, yy_max],
            "autorange": False,
            "showticklabels": showticklabels,
            "nticks": 3,
        })

    return figs
