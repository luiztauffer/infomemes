from infomemes.utils import media_color_schema
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps
from alive_progress import alive_bar
import sys


class Media():
    def __init__(self, simulation, id, step, quadrant):
        self.id = id
        self.activated = step
        self.deactivated = None
        self.cmap = media_color_schema[id % len(media_color_schema)]
        # True for active, False for inactive
        self.active = True
        # Parent simulation
        self.simulation = simulation
        # Position in XY space
        if quadrant == 1:
            self.x = np.random.rand() - 1
            self.y = np.random.rand()
        elif quadrant == 2:
            self.x = np.random.rand()
            self.y = np.random.rand()
        elif quadrant == 3:
            self.x = np.random.rand()
            self.y = np.random.rand() - 1
        elif quadrant == 4:
            self.x = np.random.rand() - 1
            self.y = np.random.rand() - 1
        # Memes production covariance - has to be positive semi-definite
        # self.cov = np.random.rand(2, 2) * 0.7 - 0.35
        # self.cov = np.dot(self.cov, self.cov.T)
        # self.cov[0, 1] = self.cov[1, 0]
        self.cov = np.array([[.2, 0], [0, .2]])
        # Media distribution
        self.mvg = sps.multivariate_normal([self.x, self.y], self.cov)
        # Initial budget
        self.budget = 100
        # Meme Producion Rate: memes / time step
        self.mpr = simulation.media_mpr  # 5
        # Reward variable
        self.reward = 0

    def produce_memes(self):
        # Produces a number of memes
        n_memes = np.random.poisson(lam=self.mpr)
        memes_pos = np.random.multivariate_normal(
            mean=[self.x, self.y],
            cov=self.cov,
            size=n_memes,
        )
        self.memes = [Meme(self, pos) for pos in memes_pos]
        # Discount from budget
        self.budget -= n_memes
        return self.memes

    def get_reward(self, distance):
        # In a [-1, 1] 2D space, the average random distance is 1
        # Reward as a function of distance to meme minus covariance punishment
        # reward = 1 / (100 * distance) - self.simulation.covariance_punishment * self.cov
        reward = 1 / (self.simulation.n_individuals * distance) - \
                 self.simulation.covariance_punishment * (self.cov[0, 0] + self.cov[1, 1])
        self.reward += max(0, min(1, reward))
        # print(reward)

    def reset_states(self):
        if self.active:
            self.reward = 0
            self.memes = []
        else:
            self.budget = np.nan
            self.reward = np.nan
            self.memes = []

    def to_dict(self):
        output = dict()
        for k, v in self.__dict__.items():
            if k == 'memes':
                output[k] = [meme.to_dict() for meme in v]
            elif k == 'simulation':
                pass
            else:
                output[k] = v
        return output


class Meme():
    def __init__(self, parent, position):
        self.parent = parent
        self.x = position[0]
        self.y = position[1]

    def to_dict(self):
        output = dict()
        for k, v in self.__dict__.items():
            if k == 'parent':
                output[k] = v.id
            else:
                output[k] = v
        return output


class Individual():
    def __init__(self, simulation, id):
        self.id = id
        # Parent simulation
        self.simulation = simulation
        # Position in XY space
        self.x = np.random.rand() * 2 - 1
        self.y = np.random.rand() * 2 - 1
        # Mind Updating Index
        self.mui = simulation.individual_mui
        # Memes Consumption Rate
        self.mcr = simulation.individual_mcr

    def consume_memes(self):
        # Calculate distances/memes list: [(distance, meme.parent)...]
        dist_mp = [self.calculate_distance_to_meme(meme) for meme in self.simulation.all_memes]
        # Sort in ascending order
        distances = [d[0] for d in dist_mp]
        ind_order = np.array(distances).argsort()
        ordered_memes = [dist_mp[ind] for ind in ind_order]
        # Consumes a number of memes: updates (x,y) position influenced by them
        n_memes = np.random.poisson(lam=self.mcr)
        consumed_memes = ordered_memes[0:n_memes]
        self.update_position(consumed_memes)
        # Reward Media that produced closest memes
        self.reward_media(consumed_memes)

    def calculate_distance_to_meme(self, meme):
        # Geometric distance
        dist = np.sqrt((self.x - meme.x)**2 + (self.x - meme.x)**2)
        return (dist, meme.parent)

    def update_position(self, consumed_memes):
        # Iterate over consumed_memes list: [(distance, meme.parent)...]
        # Moves towards (mui>0) or away from (mui<0) consumed memes positions
        for dist, meme in consumed_memes:
            self.x += (meme.x - self.x) * self.mui
            self.y += (meme.y - self.y) * self.mui

    def reward_media(self, consumed_memes):
        # Iterate over consumed_memes list: [(distance, meme.parent)...]
        for meme in consumed_memes:
            distance = meme[0]
            media = meme[1]
            media.get_reward(distance=distance)

    def to_dict(self):
        output = dict()
        for k, v in self.__dict__.items():
            if k == 'simulation':
                pass
            else:
                output[k] = v
        return output


class Simulation():
    def __init__(self, sim_config):
        self.config = sim_config
        self.n_individuals = sim_config['n_individuals']
        self.n_media = sim_config['n_media']
        # Media Memes Production Rate
        self.media_mpr = sim_config['media_mpr']
        # Covariance punishment
        self.covariance_punishment = sim_config['covariance_punishment']
        # Individual Mind Updating Index
        self.individual_mui = sim_config['individual_mui']
        # Individual Memes Consumption Rate
        self.individual_mcr = sim_config['individual_mcr']
        # Generate initial individuals and media
        self.all_individuals = [Individual(self, id=i) for i in range(self.n_individuals)]
        self.all_media = [Media(self, id=i, step=0, quadrant=i % 4 + 1) for i in range(self.n_media)]
        self.all_memes = []
        # Dictionary to store time-evolving values
        self.simulation_values = dict()
        self.current_step = -1

    def run_simulation(self, n_steps=1, proc_id=0, verbose=0):
        percent = 0.2
        with alive_bar(n_steps, spinner='waves') as bar:
            for step in np.arange(n_steps):
                self.current_step += 1
                n_active = np.sum([m.active for m in self.all_media])

                # Verbose control
                if verbose == 0:   # loading bar
                    msg = 'Process ' + str(proc_id) + ', Sim ' + str(0) + '  |  Active Media: ' + str(n_active)
                    bar(text=msg)
                if verbose == 1:  # multiprocessing
                    if step / n_steps >= percent:
                        print('Process ' + str(proc_id) + ' in ' + str(100 * percent) + '%  |  Active Media: ' + str(n_active))
                        percent += .2
                elif verbose == 2:  # every step
                    print('Step: ', self.current_step, '  |  Active Media: ', n_active)

                # Populate world with memes
                active_media = []
                for m in self.all_media:
                    m.reset_states()
                    if m.active:
                        active_media.append(m)
                        self.all_memes.extend(m.produce_memes())

                # Individuals routines
                for i in self.all_individuals:
                    # Chance of reborn (position randomly changed)
                    if np.random.rand() > 0.96:
                        i.x = np.random.rand() * 2 - 1
                        i.y = np.random.rand() * 2 - 1
                    # Individuals consume memes | media collect rewards
                    i.consume_memes()

                # Control media budget: remove (<0) or constrain (>100)
                for m in self.all_media:
                    if m.active:
                        m.budget += m.reward
                        if m.budget <= 0 or np.random.rand() > 0.99:
                            m.active = False
                            m.deactivated = self.current_step
                        elif m.budget > 100:
                            m.budget = 100

                # Random chance of "reproduction"
                n_new = np.random.poisson(1)
                for new in np.arange(n_new):
                    new_media = Media(
                        simulation=self,
                        id=len(self.all_media),
                        step=self.current_step,
                        quadrant=np.random.randint(4) + 1
                    )
                    # Randomly choose parents and inherited attributes
                    parents = np.random.choice(active_media, size=2, replace=False)
                    attributes = np.array([0, 0, 1, 1])
                    np.random.shuffle(attributes)
                    new_media.mpr = parents[attributes[0]].mpr + (np.random.rand() - 0.5)
                    new_media.x = parents[attributes[1]].x + (np.random.rand() - 0.5) * 0.1
                    new_media.y = parents[attributes[2]].y + (np.random.rand() - 0.5) * 0.1
                    # Estimate Covariance of parent from random sampling. This has two advantages:
                    # 1- guarantees it will be positive semi-definite
                    # 2- introduces random noise (finite sample), while being close to the parent
                    memes_xy = np.random.multivariate_normal(
                        mean=[parents[attributes[3]].x, parents[attributes[3]].y],
                        cov=parents[attributes[3]].cov,
                        size=20,
                    )  # np.array([[meme.x, meme.y] for meme in parents[attributes[3]].memes])
                    try:
                        new_media.cov = np.cov(memes_xy, rowvar=False)
                        # new_media.cov = check_covariance(cov)
                        new_media.mvg = sps.multivariate_normal([new_media.x, new_media.y], new_media.cov)
                    except:
                        print('Singular matrix, skipping new media')
                    # Add new media to simulation
                    self.all_media.append(new_media)

                    # Store instantaneous values
                    # self.store_simulation_values(self.current_step)

                    # Clear memes
                    self.all_memes = []

    def store_simulation_values(self, step):
        self.simulation_values[str(step)] = {
            'individuals': [i.to_dict() for i in self.all_individuals],
            'media': [m.to_dict() for m in self.all_media]
        }

    def get_media(self, id):
        for m in self.all_media:
            if m.id == id:
                return m
        return None

    def get_history_media(self, id):
        history = {
            'budgets': [],
            'rewards': [],
            'x': [],
            'y': [],
            'cov': []
        }
        for step, val in self.simulation_values.items():
            media = self.search(val['media'], 'id', id)
            if media is not None:
                history['budgets'].append(media['budget'])
                history['rewards'].append(media['reward'])
                history['x'].append(media['x'])
                history['y'].append(media['y'])
                history['cov'].append(media['cov'])
            else:
                history['budgets'].append(np.nan)
                history['rewards'].append(np.nan)
                history['x'].append(np.nan)
                history['y'].append(np.nan)
                history['cov'].append(np.nan)
        return history

    def get_history_individual(self, id):
        history = {
            'x': [],
            'y': []
        }
        for step, val in self.simulation_values.items():
            individual = self.search(val['individuals'], 'id', id)
            history['x'].append(individual['x'])
            history['y'].append(individual['y'])
        return history

    def search(self, list_of_dict, key, value):
        for i in list_of_dict:
            if i[str(key)] == value:
                return i
        return None

    def plot_current_state(self, media_ids=None):
        if media_ids is None:
            media_ids = [m.id for m in self.all_media if m.active]
        budgets = []

        plt.figure()
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot([-1, 1], [0, 0], 'k', linewidth=0.8)
        ax1.plot([0, 0], [-1, 1], 'k', linewidth=0.8)
        for i in self.all_individuals:
            ax1.plot(i.x, i.y, '.k', alpha=0.2)
        for id in media_ids:
            media = self.get_media(id)
            if media.active:
                self.plot_media_contours(media=media, ax=ax1)
                budgets.append(media.budget)
        plt.show()

    def plot_history(self, media_ids=None):
        # Media history
        if media_ids is None:
            media_ids = [m.id for m in self.all_media]
        media_budgets = np.zeros(shape=(self.current_step + 1, len(media_ids)))
        media_rewards = np.zeros(shape=(self.current_step + 1, len(media_ids)))
        media_positions_x = np.zeros(shape=(self.current_step + 1, len(media_ids)))
        media_positions_y = np.zeros(shape=(self.current_step + 1, len(media_ids)))
        # media_covs = np.zeros(shape=(self.current_step + 1, len(media_ids)))
        for ind, id in enumerate(media_ids):
            history = self.get_history_media(id)
            media_budgets[:, ind] = history['budgets']
            media_rewards[:, ind] = history['rewards']
            media_positions_x[:, ind] = history['x']
            media_positions_y[:, ind] = history['y']
            # media_covs[:, ind] = history['cov']

        # Individuals history
        individuals_ids = [i.id for i in self.all_individuals]
        individuals_positions_x = np.zeros(shape=(self.current_step + 1, len(individuals_ids)))
        individuals_positions_y = np.zeros(shape=(self.current_step + 1, len(individuals_ids)))
        for ind, id in enumerate(individuals_ids):
            history = self.get_history_individual(id)
            individuals_positions_x[:, ind] = history['x']
            individuals_positions_y[:, ind] = history['y']

        plt.figure()
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot([-1, 1], [0, 0], 'k', linewidth=0.8)
        ax1.plot([0, 0], [-1, 1], 'k', linewidth=0.8)
        ax1.plot(individuals_positions_x, individuals_positions_y, '.k', alpha=0.02)
        # ax1.plot(media_positions_x, media_positions_y, 'o', alpha=0.2)
        for id in media_ids:
            media = self.get_media(id)
            self.plot_media_contours(media=media, ax=ax1)

        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(media_budgets)
        ax2.set_ylim([0, 101])
        ax2.set_ylabel('Media budget')
        ax2.set_xlabel('Time')
        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(media_rewards)
        ax4.set_ylabel('Media rewards')
        ax4.set_xlabel('Time')
        plt.show()

    def plot_media_contours(self, media, ax):
        ds = 0.02
        X = np.arange(-1, 1 + ds, ds)
        Y = np.arange(-1, 1 + ds, ds)
        Z = np.zeros((len(X), len(Y)))
        for i, y in enumerate(X):
            for j, x in enumerate(Y):
                Z[i, j] = media.mvg.pdf([x, y])
        ax.contourf(X, Y, Z, levels=3, cmap=media.cmap)  # , alpha=0.3)
        ax.plot(media.x, media.y, 'o', color=media.cmap(1), alpha=1, markersize=5)


def check_covariance(cov):
    if hasattr(cov, 'shape'):
        if cov.shape == (2, 2):
            # Diagonal minimum value
            cov[0, 0] = max(cov[0, 0], 0.01)
            cov[1, 1] = max(cov[1, 1], 0.01)
        else:
            cov = np.array([[1, 0], [0, 1]])
    else:
        cov = np.array([[1, 0], [0, 1]])
    return cov


def up():
    sys.stdout.write('\x1b[1A')
    sys.stdout.flush()


def down():
    sys.stdout.write('\n')
    sys.stdout.flush()
