from matplotlib.colors import ListedColormap
from matplotlib import cm
import numpy as np


cmap = cm.get_cmap('tab10')
colors_absolute = cmap(np.linspace(0, 1, 10))
colors_absolute = np.delete(colors_absolute, 9, axis=0)
colors_absolute = np.delete(colors_absolute, 4, axis=0)


def make_listed_cmap(color_absolute):
    newcolors = []
    alphas = [0., .1, .3, .6]
    for i in range(4):
        color = color_absolute
        newcolors.append(color)
    newcolors = np.array(newcolors)
    newcolors[0, :] = [1, 1, 1, 0]
    newcolors[:, -1] = alphas
    newcmp = ListedColormap(newcolors)
    return newcmp


media_color_schema = [make_listed_cmap(color_absolute=ca) for ca in colors_absolute]
