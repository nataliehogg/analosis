import numpy as np
import pandas as pd
import matplotlib
from matplotlib import rc, gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import copy
import math
import emcee
from chainconsumer import ChainConsumer
c = ChainConsumer()

# common thinning setting (takes every nth sample in chain to ensure independence of samples)
thin = 10

# colours for plots
LOS         = ['#a6dba0','#5aae61','#1b7837']
LOS_minimal = ['#c2a5cf', '#9970ab', '#762a83']

# Use TeX
rc('text', usetex=True)
rc('font', family='serif')
matplotlib.rcParams.update({'font.size': 18})

class Plots:

    def __init__(self):
        rings_to_bind_them_all = 1

    def lens_mass_plot(self, path):

        return plot

    def image_plot(self, path, settings, number_of_columns=5, quality_cut=0, save=True, show=True):
        print('Preparing image plot...')
        number_of_images = settings['number_of_images']
        if number_of_images > 10:
            print('The plotter is slow but the result looks soooo good. Patience, my young padawan!')

        # Define the quality of images (the criterion is very empirical here)
        kwargs = pd.read_csv(str(path) + '/datasets/input_kwargs.csv')
        R_s = kwargs['R_sersic_sl'].to_numpy()
        beta = np.sqrt(kwargs['x_sl']**2. + kwargs['y_sl']**2.).to_numpy()
        quality = 1 / (1 + (beta/3/R_s)**2)

        filename = str(path) + '/datasets/image_list.pickle'
        infile = open(filename,'rb')
        image_list = pickle.load(infile)
        infile.close()

        cmap_string = 'bone'
        cmap = copy.copy(matplotlib.cm.get_cmap(cmap_string))
        cmap.set_bad(color='k', alpha=1.)
        cmap.set_under('k')

        v_min = -3
        v_max = 2

        rows = int(math.ceil(number_of_images/number_of_columns))

        plt.rcParams['figure.constrained_layout.use'] = True

        fig = plt.figure(figsize=(12,12))
        gs = gridspec.GridSpec(rows, number_of_columns, figure=fig)
        for n in range(number_of_images):
            ax = fig.add_subplot(gs[n])
            im = ax.matshow(np.log10(image_list[n]), origin='lower', vmin=v_min, vmax=v_max, cmap=cmap, extent=[0, 1, 0, 1])
            if quality[n] < quality_cut:
                ax.set_title('{:.2f}'.format(quality[n]), color='red', fontsize=8)
                ax.plot([0,1],[0,1], color='red')
                ax.plot([0,1],[1,0], color='red')
            else:
                ax.set_title('{:.2f}'.format(quality[n]), fontsize=8)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.autoscale(False)

        # fig.tight_layout()

        if save:
            plt.savefig(str(path) + '/plots/image.pdf', dpi=300, bbox_inches='tight')
        if show:
            plt.show()

        return None

    def input_output_plot(self, path, settings, quality_cut=0, show_not_converged=True, save=True, show=True):

        in_kwargs = pd.read_csv(path + '/datasets/input_kwargs.csv')

        # define the quality
        R_s = in_kwargs['R_sersic_sl'].to_numpy()
        beta = np.sqrt(in_kwargs['x_sl']**2. + in_kwargs['y_sl']**2.).to_numpy()
        quality = 1 / (1 + (beta/3/R_s)**2)

        in_gamma1 = in_kwargs['gamma1_los']
        in_gamma2 = in_kwargs['gamma2_los']

        # # flatten the lists
        # in_gamma1 = [item for sublist in in_gammas1 for item in sublist]
        # in_gamma2 = [item for sublist in in_gammas2 for item in sublist]

        for i in range(settings['number_of_images']):
            chain = path + '/chains/' + str(settings['job_name']) + '_' + str(i) + '.h5'
            reader = emcee.backends.HDFBackend(filename = chain, name = 'lenstronomy_mcmc_emcee')
            samples = reader.get_chain(discard = settings['n_burn'], flat = True, thin = thin)
            c.add_chain(samples[:,2:4], walkers=np.shape(samples)[0], parameters = ['gamma1_los', 'gamma2_los'])
        summary = c.analysis.get_summary()

        # Remove the images under a certain quality
        if quality_cut is not None:
            summary = [s for i,s in enumerate(summary) if quality[i]>quality_cut]
            in_gamma1 = [g for i, g in enumerate(in_gamma1) if quality[i]>quality_cut]
            in_gamma2 = [g for i, g in enumerate(in_gamma2) if quality[i]>quality_cut]

        # Isolate the cases where the MCMC did not converge
        #summary_converged = [s for s in summary if s['gamma1_los'][0] is not None]
        #summary_not_converged = [s for s in summary if s['gamma1_los'][0] is None]
        summary_converged = []
        indices_converged = []
        summary_not_converged = []
        indices_not_converged = []
        for i in range(len(summary)):
            s = summary[i]
            if (s['gamma1_los'][0] is None
                or s['gamma1_los'][2] is None
                or s['gamma2_los'][0] is None
                or s['gamma2_los'][2] is None):
                indices_not_converged.append(i)
                summary_not_converged.append(s)
            else:
                indices_converged.append(i)
                summary_converged.append(s)

        # plot the converged ones with error bars
        in_gamma1_converged = [in_gamma1[i] for i in indices_converged]
        in_gamma2_converged = [in_gamma2[i] for i in indices_converged]

        out_gamma1 = [s['gamma1_los'][1] for s in summary_converged]
        out_gamma2 = [s['gamma2_los'][1] for s in summary_converged]
        gamma1_lower = [np.abs(s['gamma1_los'][0] - s['gamma1_los'][1]) for s in summary_converged]
        gamma1_upper = [np.abs(s['gamma1_los'][2] - s['gamma1_los'][1]) for s in summary_converged]
        gamma2_lower = [np.abs(s['gamma2_los'][0] - s['gamma2_los'][1]) for s in summary_converged]
        gamma2_upper = [np.abs(s['gamma2_los'][2] - s['gamma2_los'][1]) for s in summary_converged]

        fig, ax = plt.subplots(1, 1, figsize = (7,7), sharex=True, sharey=True)

        plt.errorbar(in_gamma1_converged, out_gamma1, yerr = [gamma1_lower, gamma1_upper],
                     ls = ' ', marker = '.', color = LOS[1], label = r'$\gamma_1^{\rm LOS}$')

        plt.errorbar(in_gamma2_converged, out_gamma2, yerr = [gamma1_lower, gamma1_upper],
                     ls = ' ', marker = '.', color = LOS_minimal[1], label = r'$\gamma_2^{\rm LOS}$')


        if show_not_converged and len(indices_not_converged) > 0:
            # plot the non-converged ones with crosses and without error bars
            in_gamma1_not_converged = [in_gamma1[i] for i in indices_not_converged]
            in_gamma2_not_converged = [in_gamma2[i] for i in indices_not_converged]

            out_gamma1 = [s['gamma1_los'][1] for s in summary_not_converged]
            out_gamma2 = [s['gamma2_los'][1] for s in summary_not_converged]

            plt.plot(in_gamma1_not_converged, out_gamma1,
                     ls = ' ', marker = 'x', markersize=10, color = LOS[1], label = r'non-converged chains')

            plt.plot(in_gamma2_not_converged, out_gamma2,
                     ls = ' ', marker = 'x', markersize=10, color = LOS_minimal[1])

        ax.set_xlabel('Input $\gamma_{\\rm LOS}$')
        ax.set_ylabel('Output $\gamma_{\\rm LOS}$')

        # make an x = y line for the range of our plot
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            ]

        ax.plot(lims, lims, color = 'black', ls = '--', alpha=0.3, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        if quality_cut is not None:
            plt.title(r"$Q > {}$".format(quality_cut))

        plt.legend(frameon=False)

        if save:
            plt.savefig(str(path) + '/plots/input_output.pdf', dpi=300, bbox_inches='tight')
        if show:
            plt.show()

        return None


    def contour_plot(self, path):

        return plot
