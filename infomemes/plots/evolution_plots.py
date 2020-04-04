import plotly.graph_objs as go
import plotly
import numpy as np
import json


def make_evolution_plots(sim_file, output_html='output.html'):
    # make data
    fname = sim_file
    with open(fname, 'r') as f:
        file_dict = json.load(f)

    all_data = file_dict['data']

    # Figure basics
    # vertical/horizontal lines
    vline = {
        "mode": "lines",
        "x": [0, 0],
        "y": [-1, 1],
        "showlegend": False,
        "line": {"color": "#000000", "width": 0.8},
        "hoverinfo": 'skip',
    }
    hline = {
        "mode": "lines",
        "x": [-1, 1],
        "y": [0, 0],
        "showlegend": False,
        "line": {"color": "#000000", "width": 0.8},
        "hoverinfo": 'skip',
    }

    # Annotations
    auth = {
        'text': '<b>authoritarian</b>',
        'font': {"family": "Arial", "size": 20},
        'x': 0.5,
        'xanchor': 'center',
        'xref': 'paper',
        'y': 1.02,
        'yanchor': 'bottom',
        'yref': 'paper',
        'showarrow': False,
    }
    lib = {
        'text': '<b>libertarian</b>',
        'font': {"family": "Arial", "size": 20},
        'x': 0.5,
        'xanchor': 'center',
        'xref': 'paper',
        'y': -0.08,
        'yanchor': 'bottom',
        'yref': 'paper',
        'showarrow': False,
    }
    left = {
        'text': '<b>left</b>',
        'font': {"family": "Arial", "size": 20},
        'x': -0.1,
        'xanchor': 'center',
        'xref': 'paper',
        'y': 0.5,
        'yanchor': 'bottom',
        'yref': 'paper',
        'showarrow': False,
    }
    right={
        'text': '<b>right</b>',
        'font': {"family": "Arial", "size": 20},
        'x': 1.1,
        'xanchor': 'center',
        'xref': 'paper',
        'y': 0.5,
        'yanchor': 'bottom',
        'yref': 'paper',
        'showarrow': False,
    }

    layout = dict(
        xaxis={
            "tickmode": 'array',
            "tickvals": [],
            "range": [-1, 1],
            "autorange": False,
            "zeroline": True
        },
        yaxis={
            "tickmode": 'array',
            "tickvals": [],
            "range": [-1, 1],
            "autorange": False,
            "zeroline": True
        },
        updatemenus=[dict(
            type="buttons",
            direction="left",
            buttons=list([
                dict(
                    args=[None, {"frame": {"duration": 100, "redraw": False},
                                 "fromcurrent": True,
                                 "transition": {"duration": 1000, "easing": "linear"}}],
                    label="Play",
                    method="animate"
                ),
                dict(
                    args=[[None], {"frame": {"duration": 0, "redraw": False},
                                   "mode": "immediate",
                                   "transition": {"duration": 0}}],
                    label="Pause",
                    method="animate"
                )
            ]),
            pad={"r": 10, "t": 10},
            active=0,
            showactive=True,
            x=-0.2,
            xanchor="left",
            y=-0.1,
            yanchor="bottom"),
        ]
    )

    fig = go.Figure(
        data=[],
        layout=layout,
        frames=[]
    )

    frames = []
    for step in range(len(all_data)):
        curr_data = all_data[str(step)]
        individuals_dict = {
            "name": 'individuals',
            "mode": "markers",
            "type": "scatter",
            "x": [ind['x'] for ind in curr_data['individuals']],
            "y": [ind['y'] for ind in curr_data['individuals']],
            "showlegend": False,
            "marker": {"size": 2, "color": '#000000'},
            "hoverinfo": 'skip',
        }

        xx = []
        yy = []
        cc = []
        for m in curr_data['media']:
            xx.append(m['x'])
            yy.append(m['y'])
            if not m['active']:
                cc.append('rgba(0,0,0,0)')
            elif m['x'] < 0 and m['y'] < 0:
                cc.append('rgba(81, 184, 108, .4)')
            elif m['x'] > 0 and m['y'] < 0:
                cc.append('rgba(175, 81, 184, .4)')
            elif m['x'] < 0 and m['y'] > 0:
                cc.append('rgba(184, 81, 81, .4)')
            elif m['x'] > 0 and m['y'] > 0:
                cc.append('rgba(81, 138, 184, .4)')
        media_dict = {
            "name": 'media',
            "mode": "markers",
            "type": "scatter",
            "x": xx,
            "y": yy,
            "showlegend": False,
            "marker": {"size": 10, "color": cc},
            "hoverinfo": 'skip',
        }

        if step == 0:
            fig.add_trace(go.Scatter(individuals_dict))
            fig.add_trace(go.Scatter(media_dict))

        # make frames
        frame = {"data": [go.Scatter(individuals_dict), go.Scatter(media_dict)], "name": str(step), "traces": [0, 1]}
        frames.append(frame)

    fig.add_trace(go.Scatter(vline))
    fig.add_trace(go.Scatter(hline))
    fig["frames"] = [go.Frame(fr) for fr in frames]

    fig.layout['plot_bgcolor'] = "rgba(0,0,0,0)"
    fig.layout['paper_bgcolor'] = "rgba(0,0,0,0)"
    fig.layout['width'] = 600
    fig.layout['height'] = 500
    fig.layout['margin']['l'] = 80
    fig.layout['margin']['r'] = 120
    fig.layout['margin']['t'] = 50
    fig.layout['margin']['b'] = 80
    fig.layout['margin']['autoexpand'] = False

    # Write to html
    fig.write_html(output_html, config={'showLink': False, 'displayModeBar': False})
