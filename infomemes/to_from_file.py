import pandas as pd
import numpy as np
import json


def save_stepwise_data(sim, fname):
    # Simulation metadata
    metadata = {
        'duration': sim.current_step,
    }
    metadata.update(sim.config)

    file_dict = {}
    file_dict['metadata'] = metadata
    file_dict['data'] = sim.stepwise_values

    with open(fname, 'w') as f:
        json.dump(file_dict, f)


def save_light_data(sim, fname):
    activated = []
    deactivated = []
    position_x = []
    position_y = []
    cov_x = []
    cov_y = []
    cov_xy = []
    meme_production_rate = []
    for m in sim.all_media:
        if m.active:
            deactivated.append(sim.current_step)
        else:
            deactivated.append(m.deactivated)
        activated.append(m.activated)
        position_x.append(m.x)
        position_y.append(m.y)
        cov_x.append(m.cov[0, 0])
        cov_y.append(m.cov[1, 1])
        cov_xy.append(m.cov[0, 1])
        meme_production_rate.append(m.meme_production_rate)

    # Arrange data in a DataFrame
    data = np.array([activated, deactivated, position_x, position_y, cov_x, cov_y, cov_xy, meme_production_rate]).T
    colnames = ['activated', 'deactivated', 'position_x', 'position_y', 'cov_x', 'cov_y', 'cov_xy', 'meme_production_rate']
    df = pd.DataFrame(data=data, columns=colnames)

    # Simulation metadata
    metadata = {
        'duration': sim.current_step,
        'individuals_xy': [(i.x, i.y) for i in sim.all_individuals],
    }
    metadata.update(sim.config)

    file_dict = {}
    file_dict['metadata'] = metadata
    file_dict['data'] = df.to_dict()

    with open(fname, 'w') as f:
        json.dump(file_dict, f)


def load_light_data(fname):
    with open(fname, 'r') as f:
        file_dict = json.load(f)

    df_as_dict = file_dict.pop('data', {})
    df = pd.DataFrame().from_dict(df_as_dict)

    metadata = file_dict['metadata']

    return metadata, df
