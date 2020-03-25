from infomemes.classes import Simulation
from infomemes.to_from_file import save_light_data

import concurrent.futures
import multiprocessing


# Default simulation configurations
default_config = {
    # media
    'n_media': 100,
    'media_mpr': 10,
    'covariance_punishment': 0.5,
    # individuals
    'n_individuals': 1000,
    'individual_mui': 0.01,
    'individual_mcr': 5,
}


# Simulation routine
def sim_routine(sim_config, n_steps=100, n_sims=1, proc_id=0, verbose=0):
    try:
        for i in range(n_sims):
            # Set up simulation
            sim = Simulation(sim_config=sim_config)

            # Run simulation
            n_steps = n_steps
            sim.run_simulation(n_steps=n_steps, proc_id=proc_id, verbose=verbose)

            # Save simulation to file
            fname = 'sim_' + str(proc_id * n_sims + i) + '.json'
            save_light_data(sim, fname)
        return 'Process ' + str(proc_id) + ' finished all simulations'
    except BaseException as e:
        print(e)
        return e

    # sim.plot_current_state()
    # sim.plot_history()


# Multiprocessing organizer
def multiprocessing_organizer(sim_config, n_steps=100, n_sims=1, n_procs=2, verbose=0):
    # Maximum available processors
    max_procs = multiprocessing.cpu_count()
    n_procs = min(n_procs, max_procs)

    # Multiprocessing pool
    with concurrent.futures.ProcessPoolExecutor() as executor:
        procs_list = []
        for i in range(n_procs):
            kwargs = {
                'sim_config': sim_config,
                'n_steps': n_steps,
                'n_sims': n_sims,
                'proc_id': i,
                'verbose': verbose
            }
            p = executor.submit(sim_routine, **kwargs)
            procs_list.append(p)

        for p in concurrent.futures.as_completed(procs_list):
            print(p.result())


# Parse arguments and call routines
def main():
    import argparse

    parser = argparse.ArgumentParser(description='Simulates the media evolution game')

    parser.add_argument(
        "--sim_config",
        default=None,
        help="JSON file containing simulation configuration."
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

    args = parser.parse_args()
    if args.sim_config is None:
        sim_config = default_config
    else:
        sim_config = args.sim_config
    n_steps = int(args.n_steps)
    n_sims = int(args.n_sims)
    n_procs = int(args.n_procs)
    verbose = int(args.verbose)

    if n_procs == 1:
        sim_routine(sim_config=sim_config, n_steps=n_steps, n_sims=n_sims, verbose=verbose)
    elif n_procs > 1:
        multiprocessing_organizer(sim_config=sim_config, n_steps=n_steps, n_sims=n_sims, n_procs=n_procs, verbose=verbose)


# Called from command line
if __name__ == '__main__':
    main()
