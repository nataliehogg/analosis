import numpy as np
import pandas as pd
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import copy

# Use TeX
rc('text', usetex=True)
rc('font', family='serif')
matplotlib.rcParams.update({'font.size': 18})

class Plots:

    def __init__(self):
        rings_to_bind_them_all = 1

    def lens_mass_plot(self, path):

        return plot

    def image_plot(self, path, number_of_images, save=True, show=True):

        # Define the quality of images (the criterion is very empirical here)
        kwargs = pd.read_csv(str(path) + '/datasets/input_kwargs.csv')
        R_s = kwargs['R_sersic_sl'].to_numpy()
        beta = np.sqrt(kwargs['x_sl']**2. + kwargs['y_sl']**2.).to_numpy()
        quality = 1 / (1 + (beta/3/R_s)**2)
        
        # make this user-defined
        quality_cut = 0

        filename = str(path) + '/datasets/image_list.pickle'
        infile = open(filename,'rb')
        image_list = pickle.load(infile)
        infile.close()

        cmap_string = 'bone'
        cmap = copy.copy(matplotlib.cm.get_cmap(cmap_string))
        cmap.set_bad(color='k', alpha=1.)
        cmap.set_under('k')

        v_min = -4
        v_max = 3


        if number_of_images == 1:
            f, ax = plt.subplots(1, 1, figsize = (5,5), sharex=False, sharey=False)
            im = ax.matshow(np.log10(image_list[0]), origin='lower', vmin=v_min, vmax=v_max, cmap=cmap, extent=[0, 1, 0, 1])
            if quality < quality_cut:
                ax.set_title('{:.2f}'.format(quality[0]), color='red', fontsize=12)
                ax.plot([0,1],[0,1], color='red')
                ax.plot([0,1],[1,0], color='red')
            else:
                ax.set_title('{:.2f}'.format(quality[0]), fontsize=12)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.autoscale(False)
        else:
            if number_of_images >= 4:
                f, ax = plt.subplots(int(np.sqrt(number_of_images)), int(np.sqrt(number_of_images)),
                                 figsize=(10, 10), sharex=False, sharey=False)
            else:
                f, ax = plt.subplots(1, number_of_images, figsize = (10, 10), sharex=False, sharey=False)

            for a, i in zip(ax.flat, range(number_of_images)):
                im = a.matshow(np.log10(image_list[i]), origin='lower', vmin=v_min, vmax=v_max, cmap=cmap, extent=[0, 1, 0, 1])
                if quality[i] < quality_cut:
                    a.set_title('{:.2f}'.format(quality[i]), color='red', fontsize=12)
                    a.plot([0,1],[0,1], color='red')
                    a.plot([0,1],[1,0], color='red')
                else:
                    a.set_title('{:.2f}'.format(quality[i]), fontsize=12)
                a.get_xaxis().set_visible(False)
                a.get_yaxis().set_visible(False)
                a.autoscale(False)

        if save:
            plt.savefig(str(path) + '/plots/image.pdf', dpi=300, bbox_inches='tight')
        if show:
            plt.show()

        return None


    def contour_plot(self, path):

        return plot
