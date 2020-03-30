from infomemes.utils import media_color_schema

from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json


def read_sim_results(sim, step_filtered=0):
    """
    Basic analysis of a simulation.

    sim: simulation object or string with path to json file.
    step_filtered: int
        Produces a boolan array 'filtered' with True for all media that were
        active at any step > step_filtered.
    """
    if isinstance(sim, str):
        with open(sim, 'r') as f:
            sim = json.loads(f.read())
        # metadata
        duration = sim['metadata']['duration']
        media_reproduction_rate = sim['metadata']['media_reproduction_rate']
        media_deactivation_rate = sim['metadata']['media_deactivation_rate']
        covariance_punishment = sim['metadata']['covariance_punishment']
        individuals_xy = np.array(sim['metadata']['individuals_xy'])
        individual_renewal_rate = sim['metadata']['individual_renewal_rate']
        individual_mui = sim['metadata']['individual_mui']
        individual_mcr = sim['metadata']['individual_mcr']
        max_reward = sim['metadata']['max_reward']
        # data
        activated = np.array(list(sim['data']['activated'].values()))
        deactivated = np.array(list(sim['data']['deactivated'].values()))
        active = np.array([t == duration for t in deactivated])
        survival_times = np.array(deactivated) - np.array(activated)
        position_x = np.array(list(sim['data']['position_x'].values()))
        position_y = np.array(list(sim['data']['position_y'].values()))
        cov_x = np.array(list(sim['data']['cov_x'].values()))
        cov_y = np.array(list(sim['data']['cov_y'].values()))
        cov_diagonal = cov_x + cov_y
        cov_xy = np.array(list(sim['data']['cov_xy'].values()))
        mpr = np.array(list(sim['data']['meme_production_rate'].values()))
        filtered = np.array([deact > step_filtered for deact in deactivated])
    else:
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
            mpr.append(m.meme_production_rate)

    results = {
        'duration': duration,
        'media_reproduction_rate': media_reproduction_rate,
        'media_deactivation_rate': media_deactivation_rate,
        'max_reward': max_reward,
        'covariance_punishment': covariance_punishment,
        'individuals_xy': individuals_xy,
        'individual_renewal_rate': individual_renewal_rate,
        'individual_mui': individual_mui,
        'individual_mcr': individual_mcr,
        'activated': activated[filtered],
        'deactivated': deactivated[filtered],
        'active': active[filtered],
        'survival_times': survival_times[filtered].astype('int'),
        'position_x': position_x[filtered],
        'position_y': position_y[filtered],
        'cov_x': cov_x[filtered],
        'cov_y': cov_y[filtered],
        'cov_diagonal': cov_diagonal[filtered],
        'cov_xy': cov_xy[filtered],
        'meme_production_rate': mpr[filtered],
        'step_filtered': filtered
    }
    return results


def all_sims_summary(sims_list, step_filtered=0):
    """

    Parameters
    ----------
    sims_list: list
        List of json files with simulation results
    step_filtered: int
        Produces a boolan array 'filtered' with True for all media that were
        active at any step > step_filtered

    Returns
    -------
    df_sims: Pandas DataFrame with results by simulation
    df_media: Pandas DataFrame with results by media
    df_clusters: Pandas DataFrame with results by cluster
    """
    # Organize Simulations DataFrame
    df_sims = pd.DataFrame({
        'covariance_punishment': pd.Series([], dtype='float'),
        'media_reproduction_rate': pd.Series([], dtype='float'),
        'media_deactivation_rate': pd.Series([], dtype='float'),
        'individual_mui': pd.Series([], dtype='float'),
        'individual_renewal_rate': pd.Series([], dtype='float'),
        'individual_mcr': pd.Series([], dtype='float'),
        'max_reward': pd.Series([], dtype='float'),
        'clusters_distances': pd.Series([], dtype='O'),
        'media_dynamics': pd.Series([], dtype='category'),
        'individual_dynamics': pd.Series([], dtype='category'),
    })

    # Organize Media DataFrame
    df_media = pd.DataFrame({
        'simulation': pd.Series([], dtype='int'),
        'activated': pd.Series([], dtype='int'),
        'deactivated': pd.Series([], dtype='int'),
        'position_x': pd.Series([], dtype='float'),
        'position_y': pd.Series([], dtype='float'),
        'cov_x': pd.Series([], dtype='float'),
        'cov_y': pd.Series([], dtype='float'),
        'cov_xy': pd.Series([], dtype='float'),
        'meme_production_rate': pd.Series([], dtype='float'),
        'survival_time': pd.Series([], dtype='Int64'),
        'cluster_index': pd.Series([], dtype='Int64'),
    })

    # Organize Clusters DataFrame
    colnames_clusters = ['simulation', 'n_members', 'center_of_mass']
    df_clusters = pd.DataFrame({
        'simulation': pd.Series([], dtype='int'),
        'n_members': pd.Series([], dtype='int'),
        'center_of_mass': pd.Series([], dtype='O'),
    })

    # Iterate over all simulations in list and populate DataFrames
    last_cluster_index = 0
    for sim_id, sim in enumerate(sims_list):
        sim_results = read_sim_results(sim, step_filtered=step_filtered)

        # Cluster analysis
        position_x = sim_results['position_x']
        position_y = sim_results['position_y']
        active = sim_results['active']
        X = np.array([position_x[active], position_y[active]]).T
        dbc, center_of_mass, n_members, clusters_distances = cluster_analysis(data_points=X)
        n_clusters = n_members.shape[0]
        cluster_indexes = np.arange(last_cluster_index, last_cluster_index + n_clusters)
        last_cluster_index = cluster_indexes[-1] + 1

        # Update Clusters DataFrame
        sim_array = [sim_id] * n_clusters
        df = pd.DataFrame(columns=colnames_clusters)
        for i in range(n_clusters):
            aux = pd.DataFrame(data=[[sim_array[i], n_members[i], [center_of_mass[i]]]],
                               columns=df_clusters.columns)
            df = pd.concat([df, aux])
        df_clusters = pd.concat([df_clusters, df], ignore_index=True)

        # Update Media DataFrame
        n_media = sim_results['activated'].shape[0]
        sim_array = np.array([sim_id] * n_media, dtype=int)
        media_cluster_index = np.array([np.nan] * n_media)
        media_cluster_index[np.where(sim_results['active'])[0]] = cluster_indexes[dbc.labels_]
        df = pd.DataFrame(data=np.array([sim_array, sim_results['activated'], sim_results['deactivated'],
                                         sim_results['position_x'], sim_results['position_y'],
                                         sim_results['cov_x'], sim_results['cov_y'], sim_results['cov_xy'],
                                         sim_results['meme_production_rate'], sim_results['survival_times'],
                                         media_cluster_index]).T,
                          columns=df_media.columns)
        df_media = pd.concat([df_media, df], ignore_index=True)

        # Update Simulations DataFrame
        media_dynamics = [.5, 1, 2].index(sim_results['media_reproduction_rate'])
        individual_dynamics = [0.01, 0.05, 0.1].index(sim_results['individual_renewal_rate'])
        df = pd.DataFrame(data=[[sim_results['covariance_punishment'], sim_results['media_reproduction_rate'],
                                 sim_results['media_deactivation_rate'], sim_results['individual_mui'],
                                 sim_results['individual_renewal_rate'], sim_results['individual_mcr'],
                                 sim_results['max_reward'], clusters_distances,
                                 media_dynamics, individual_dynamics]],
                          columns=df_sims.columns, index=[sim_id])
        df_sims = pd.concat([df_sims, df])

    return df_sims, df_media, df_clusters, dbc


def cluster_analysis(data_points):
    """
    Automatic cluster analysis.

    Parameters:
    data_points: 2D numpy array
        2D numpy array of dimmensions [n_points, 2] with (x, y) positions of n_points.
    """
    dbc = DBSCAN(eps=0.2, min_samples=3).fit(data_points)
    cluster_labels = dbc.labels_
    n_clusters = np.unique(cluster_labels).shape[0]
    # Clusters center of mass and sizes
    center_of_mass = np.zeros(shape=(n_clusters, 2))
    clusters_sizes = np.zeros(n_clusters, dtype=int)
    for cln in np.unique(cluster_labels):
        ind = cluster_labels == cln
        points = data_points[ind]
        center_of_mass[cln, :] = np.mean(points, axis=0)
        clusters_sizes[cln] = int(points.shape[0])
    # Clusters distances
    clusters_distances = []
    for i, cm_i in enumerate(center_of_mass):
        for j, cm_j in enumerate(center_of_mass[i + 1:, :]):
            clusters_distances.append(np.sqrt((cm_i[0] - cm_j[0])**2 + (cm_i[1] - cm_j[1])**2))
    clusters_distances = np.array(clusters_distances)

    return (dbc, center_of_mass, clusters_sizes, clusters_distances)


def survival_statistics(sim):
    sim_results = read_sim_results(sim)

    individual_renewal_rate = sim_results['individual_renewal_rate']
    individual_mui = sim_results['individual_mui']
    individuals_xy = sim_results['individuals_xy']
    covariance_punishment = sim_results['covariance_punishment']
    media_reproduction_rate = sim_results['media_reproduction_rate']
    position_x = sim_results['position_x']
    position_y = sim_results['position_y']
    active = sim_results['active']
    survival_times = sim_results['survival_times']
    cov_xy = sim_results['cov_xy']
    cov_diagonal = sim_results['cov_diagonal']
    mpr = sim_results['mpr']

    max_time = max(survival_times)
    min_time = min(survival_times)

    # Cluster analysis
    X = np.array([position_x[active], position_y[active]]).T
    dbc, center_of_mass, clusters_sizes, clusters_distances = cluster_analysis(data_points=X)
    cluster_labels = dbc.labels_
    cluster_colors = [media_color_schema[i % len(media_color_schema)](1) for i in cluster_labels]

    # Plots
    fig = plt.figure()
    fig.suptitle(f'M_RR: {media_reproduction_rate}, CP:{covariance_punishment} \n'
                 f'I_RR: {individual_renewal_rate}, I_MUI: {individual_mui}', fontsize=16)
    gs = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(survival_times)
    ax1.set_xlabel('Survival time')

    ax2 = fig.add_subplot(gs[1:, 0])
    ax2.plot([-1, 1], [0, 0], 'k', linewidth=0.8)
    ax2.plot([0, 0], [-1, 1], 'k', linewidth=0.8)
    ax2.scatter(individuals_xy[:, 0], individuals_xy[:, 1], c='k', marker='.', s=1)
    ax2.scatter(position_x, position_y, s=10 * np.array(survival_times) / max_time, alpha=0.2)
    ax2.scatter(position_x[active], position_y[active], s=20, c=cluster_colors, alpha=1)
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])

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
