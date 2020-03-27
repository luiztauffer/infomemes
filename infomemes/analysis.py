import matplotlib.pyplot as plt
import numpy as np
import json

# from infomemes.analysis import survival_statistics
# survival_statistics('sim_0.json')


def survival_statistics(sim):
    """
    Basic analysis of a simulation.

    sim: simulation object or string with path to json file.
    """
    survival_times = []
    position_x = []
    position_y = []
    cov_diagonal = []
    cov_xy = []
    mpr = []
    if isinstance(sim, str):
        with open(sim, 'r') as f:
            sim = json.loads(f.read())
        # data
        activated = list(sim['data']['activated'].values())
        deactivated = list(sim['data']['deactivated'].values())
        n_steps = max(deactivated)
        active = np.array([t == n_steps for t in deactivated])
        survival_times = np.array(deactivated) - np.array(activated)
        position_x = np.array(list(sim['data']['position_x'].values()))
        position_y = np.array(list(sim['data']['position_y'].values()))
        cov_x = list(sim['data']['cov_x'].values())
        cov_y = list(sim['data']['cov_y'].values())
        cov_diagonal = np.array(cov_x) + np.array(cov_y)
        cov_xy = list(sim['data']['cov_xy'].values())
        mpr = list(sim['data']['meme_production_rate'].values())
        # metadata
        media_reproduction_rate = sim['metadata']['media_reproduction_rate']
        covariance_punishment = sim['metadata']['covariance_punishment']
    else:
        for m in sim.all_media:
            if m.active:
                survival_times.append(sim.current_step - m.activated)
            else:
                survival_times.append(m.deactivated - m.activated)
            position_x.append(m.x)
            position_y.append(m.y)
            cov_diagonal.append(m.cov[0, 0] + m.cov[1, 1])
            cov_xy.append(m.cov[0, 1])
            mpr.append(m.meme_production_rate)

    max_time = max(survival_times)
    min_time = min(survival_times)

    # Plots
    fig = plt.figure()
    fig.suptitle(f'MRR: {media_reproduction_rate}, CP:{covariance_punishment}', fontsize=16)
    gs = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(survival_times)
    ax1.set_xlabel('Survival time')

    ax2 = fig.add_subplot(gs[1:, 0])
    ax2.plot([-1, 1], [0, 0], 'k', linewidth=0.8)
    ax2.plot([0, 0], [-1, 1], 'k', linewidth=0.8)
    ax2.scatter(position_x, position_y, s=10 * np.array(survival_times) / max_time, alpha=0.2)
    ax2.scatter(position_x[active], position_y[active], s=20, c='r')

    ax3 = fig.add_subplot(gs[0, 1])
    ax3.scatter(survival_times, cov_diagonal, c='k', s=1)
    ax3.set_xlabel('Survival time')
    ax3.set_ylabel('Cov diagonal')

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(survival_times, cov_xy, c='k', s=3)
    ax4.set_xlabel('Survival time')
    ax4.set_ylabel('Cov xy')

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.scatter(survival_times, mpr, c='k', s=5)
    ax5.set_xlabel('Survival time')
    ax5.set_ylabel('MPR')

    plt.show()

    return active, deactivated


def plot_current_state(sim, media_ids=None):
    # if media_ids is None:
    #     media_ids = [m.id for m in simulation.all_media if m.active]
    # budgets = []

    plt.figure()
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot([-1, 1], [0, 0], 'k', linewidth=0.8)
    ax1.plot([0, 0], [-1, 1], 'k', linewidth=0.8)
    for i in sim.all_individuals:
        ax1.plot(i.x, i.y, '.k', alpha=0.2)
    for media in sim.all_media:
        # media = simulation.get_media(id)
        if media.active:
            print('id: ', media.id, '   |  x,y: ', media.x, ', ', media.y)
            plot_media_contours(media=media, ax=ax1)
            ax1.plot(media.x, media.y, 'ok')
    plt.show()


def plot_media_contours(media, ax):
    ds = 0.02
    X = np.arange(-1, 1 + ds, ds)
    Y = np.arange(-1, 1 + ds, ds)
    Z = np.zeros((len(X), len(Y)))
    for i, y in enumerate(X):
        for j, x in enumerate(Y):
            Z[i, j] = media.mvg.pdf([x, y])
    ax.contourf(X, Y, Z, levels=3, cmap=media.cmap)  # , alpha=0.3)
    ax.plot(media.x, media.y, 'o', color=media.cmap(1), alpha=1, markersize=5)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Simulates the media evolution game')

    parser.add_argument(
        "--file",
        default=None,
        help="Path to JSON file containing simulation results."
    )
    parser.add_argument(
        "--analysis",
        default='survival_statistics',
        help="Function to run simulation results analysis."
    )

    args = parser.parse_args()

    fname = args.file
    func_name = args.analysis
    if func_name == 'survival_statistics':
        survival_statistics(fname)
