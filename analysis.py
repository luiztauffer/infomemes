import matplotlib.pyplot as plt
import numpy as np


def survival_statistics(sim):
    survival_times = []
    position_x = []
    position_y = []
    cov_diagonal = []
    cov_xy = []
    mpr = []
    for m in sim.all_media:
        if m.active:
            survival_times.append(sim.current_step - m.activated)
        else:
            survival_times.append(m.deactivated - m.activated)
        position_x.append(m.x)
        position_y.append(m.y)
        cov_diagonal.append(m.cov[0, 0] + m.cov[1, 1])
        cov_xy.append(m.cov[0, 1])
        mpr.append(m.mpr)

    max_time = max(survival_times)
    min_time = min(survival_times)

    # Plots
    fig = plt.figure()
    gs = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(survival_times)
    ax1.set_xlabel('Survival time')

    ax2 = fig.add_subplot(gs[1:, 0])
    ax2.plot([-1, 1], [0, 0], 'k', linewidth=0.8)
    ax2.plot([0, 0], [-1, 1], 'k', linewidth=0.8)
    ax2.scatter(position_x, position_y, s=10 * np.array(survival_times) / max_time)

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
