from infomemes.classes import Simulation
import cProfile
import pstats


# Default simulation configurations
default_config = {
    # media
    'n_media': 80,
    'meme_production_rate': 10,
    'media_reproduction_rate': 1,
    'covariance_punishment': 0.5,
    # individuals
    'n_individuals': 500,
    'individual_mui': 0.01,
    'individual_mcr': 5,
    'max_reward': 0.2,
}


# Simulation routine
def sim_routine(sim_config=default_config, n_steps=100, n_sims=1, proc_id=0, verbose=0):
    # Set up simulation
    sim = Simulation(sim_config)

    # Run simulation
    n_steps = n_steps
    sim.run_simulation(n_steps=n_steps, proc_id=proc_id, verbose=verbose)


profile = cProfile.Profile()
profile.runcall(sim_routine)
ps = pstats.Stats(profile)
ps.print_stats()
