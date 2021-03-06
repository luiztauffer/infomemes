from infomemes.classes import Simulation
from infomemes.to_from_file import save_light_data, save_stepwise_data

import numpy as np
from pathlib import Path
import concurrent.futures
import multiprocessing
import json


# Default simulation configurations
default_config = {
    # media
    'n_media': 100,
    'meme_production_rate': 5,
    'media_reproduction_rate': 0.05,
    'media_deactivation_rate': 0.001,
    'covariance_punishment': 3,
    # individuals
    'individual_renewal_rate': 0.03,
    'n_individuals': 200,
    'individual_mui': 0.001,
    'individual_mcr': 5,
    'max_reward': 0.4,
}


# Simulation routine
def sim_routine(sim_config, n_steps=100, n_sims=1, proc_id=0, verbose=0, save_dir=None,
                store_stepwise_values=False):
    try:
        np.random.seed()
        for i in range(n_sims):
            # Set up simulation
            sim = Simulation(sim_config=sim_config)

            # Run simulation
            n_steps = n_steps
            sim.run_simulation(n_steps=n_steps, proc_id=proc_id, verbose=verbose,
                               store_stepwise_values=store_stepwise_values)

            # Save simulation to file
            if save_dir is None:
                save_dir = Path.cwd()
            fname = str(save_dir / ('sim_' + str(proc_id * n_sims + i) + '.json'))
            save_light_data(sim, fname)

            # Save stepwise data to file
            if store_stepwise_values:
                fname = str(save_dir / ('sim_' + str(proc_id * n_sims + i) + '_stepwise.json'))
                save_stepwise_data(sim, fname)

        return 'Process ' + str(proc_id) + ' finished all simulations'
    except BaseException as e:
        print(e)
        return e


# Multiprocessing organizer
def multiprocessing_organizer(sim_config, n_steps=10, n_sims=1, n_procs=2, verbose=0,
                              save_dir=None, store_stepwise_values=False):
    # Maximum available processors
    max_procs = multiprocessing.cpu_count()
    n_procs = min(n_procs, max_procs)

    # Multiprocessing pool
    if isinstance(sim_config, list):
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_procs) as executor:
            procs_list = []
            for i, sim in enumerate(sim_config):
                kwargs = {
                    'sim_config': sim,
                    'n_steps': n_steps,
                    'n_sims': n_sims,
                    'proc_id': i,
                    'verbose': verbose,
                    'save_dir': save_dir,
                    'store_stepwise_values': store_stepwise_values,
                }
                p = executor.submit(sim_routine, **kwargs)
                procs_list.append(p)

            for p in concurrent.futures.as_completed(procs_list):
                print(p.result())
    # else:
    #     with concurrent.futures.ProcessPoolExecutor() as executor:
    #         procs_list = []
    #         for i in range(n_procs):
    #             kwargs = {
    #                 'sim_config': sim_config,
    #                 'n_steps': n_steps,
    #                 'n_sims': n_sims,
    #                 'proc_id': i,
    #                 'verbose': verbose,
    #                 'save_dir': save_dir,
    #                 'store_stepwise_values': store_stepwise_values,
    #             }
    #             p = executor.submit(sim_routine, **kwargs)
    #             procs_list.append(p)
    #
    #         for p in concurrent.futures.as_completed(procs_list):
    #             print(p.result())


# Parse arguments and call routines
def main():
    import argparse

    parser = argparse.ArgumentParser(description='Simulates the media evolution game')

    parser.add_argument(
        "--sim_config",
        default=None,
        help="Path to JSON file containing simulation configuration."
    )
    parser.add_argument(
        "--n_steps",
        default=100,
        help="Number of simulated steps."
    )
    parser.add_argument(
        "--n_sims",
        default=1,
        help="Number of simulations to run."
    )
    parser.add_argument(
        "--n_procs",
        default=1,
        help="Number of processes to use."
    )
    parser.add_argument(
        "--verbose",
        default=0,
        help="0: silent, 1: prints summary info, 2: prints detailed info."
    )
    parser.add_argument(
        "--stepwise",
        default=False,
        help="Whether to store or not stepwise values."
    )

    args = parser.parse_args()
    if args.sim_config is None:
        sim_config = default_config
    else:
        with open(args.sim_config, 'r') as f:
            sim_config = json.loads(f.read())
    n_steps = int(args.n_steps)
    n_sims = int(args.n_sims)
    n_procs = int(args.n_procs)
    verbose = int(args.verbose)
    stepwise = args.stepwise

    if n_procs == 1:
        sim_routine(sim_config=sim_config, n_steps=n_steps, n_sims=n_sims,
                    verbose=verbose, store_stepwise_values=stepwise)
    elif n_procs > 1:
        multiprocessing_organizer(sim_config=[sim_config], n_steps=n_steps, n_sims=n_sims,
                                  n_procs=n_procs, verbose=verbose, store_stepwise_values=stepwise)


# Called from command line
if __name__ == '__main__':
    """
    Example:
    infomemes --n_steps=1000 --n_sims=10 --n_procs=2 --stepwise=True
    """
    main()
