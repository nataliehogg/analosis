import numpy as np
import pandas as pd
import matplotlib
from matplotlib import rc, gridspec, cm
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
import seaborn as sns
import pickle
import copy
import math
import emcee
import h5py
import os
from chainconsumer import ChainConsumer

from analosis.utilities.useful_functions import Utilities

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

    def __init__(self, cosmo, path):

        self.util = Utilities(cosmo, path)


    def lens_mass_plot(self, path):

        return plot


    def image_plot(self, path, settings, number_of_columns=5, u_max=1, save=True, show=True):
        print('Preparing image plot...')
        number_of_images = settings['number_of_images']
        if number_of_images > 10:
            print('The plotter is slow for this many images but the result looks soooo good. Patience, my young padawan!')

        # Define the quality of images (the criterion is very empirical here)
        kwargs = pd.read_csv(str(path) + '/datasets/'+ str(settings['job_name'])+ '_input_kwargs.csv')
        beta = np.sqrt(kwargs['x_sl']**2. + kwargs['y_sl']**2.).to_numpy()
        theta_E = kwargs['theta_E'].to_numpy()
        u = beta / theta_E # reduced impact parameter

        filename = str(path) + '/datasets/' + str(settings['job_name']) + '_image_list.pickle'
        infile = open(filename,'rb')
        image_list = pickle.load(infile)
        infile.close()

        cmap_string = 'bone'
        cmap = copy.copy(matplotlib.cm.get_cmap(cmap_string))
        cmap.set_bad(color='k', alpha=1.)
        cmap.set_under('k')

        v_min = -3
        v_max = 0

        rows = int(math.ceil(number_of_images/number_of_columns))

        plt.rcParams['figure.constrained_layout.use'] = True

        fig = plt.figure(figsize=(12,12))
        gs = gridspec.GridSpec(rows, number_of_columns, figure=fig)
        for n in range(number_of_images):
            ax = fig.add_subplot(gs[n])
            im = ax.matshow(np.log10(image_list[n]), origin='lower', vmin=v_min, vmax=v_max, cmap=cmap, extent=[0, 1, 0, 1])
            if u[n] > u_max:
                ax.set_title(r'$u = {:.2f}$'.format(u[n]), fontsize=8)
                ax.plot([0,1],[0,1], color='red')
                ax.plot([0,1],[1,0], color='red')
            else:
                ax.set_title(r'$u = {:.2f}$'.format(u[n]), fontsize=8)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.autoscale(False)

        # fig.tight_layout()

        if save:
            plt.savefig(str(path) + '/plots/' + str(settings['job_name']) + '_image.pdf', dpi=300, bbox_inches='tight')
        if show:
            plt.show()

        return None


    def input_output_plot(self, path, settings, u_max=1, show_not_converged=True, use_colourmap=True, save=True, show=True):

        in_kwargs = pd.read_csv(path + '/datasets/' +str(settings['job_name']) + '_input_kwargs.csv')

        # define the quality
        beta = np.sqrt(in_kwargs['x_sl']**2. + in_kwargs['y_sl']**2.).to_numpy()
        theta_E = in_kwargs['theta_E'].to_numpy()
        u = beta / theta_E

        in_gamma1 = in_kwargs['gamma1_los']
        in_gamma2 = in_kwargs['gamma2_los']

        c = ChainConsumer()
        for i in range(len(in_kwargs)):
            chain = path + '/chains/' + str(settings['job_name']) + '_' + str(i) + '.h5'
            reader = emcee.backends.HDFBackend(filename = chain, name = 'lenstronomy_mcmc_emcee')
            samples = reader.get_chain(discard = settings['n_burn'], flat = True, thin = thin)
            c.add_chain(samples[:,2:4], walkers=np.shape(samples)[0], parameters = ['gamma1_los', 'gamma2_los'])
        summary = c.analysis.get_summary()

        # Remove the images under a certain quality
        if u_max is not None:
            summary   = [s for i, s in enumerate(summary)   if u[i] < u_max]
            in_gamma1 = [g for i, g in enumerate(in_gamma1) if u[i] < u_max]
            in_gamma2 = [g for i, g in enumerate(in_gamma2) if u[i] < u_max]

        # Isolate the cases where the MCMC did not converge
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

        if use_colourmap == True:

            fig, ax = plt.subplots(1, 2, figsize = (11,5), sharex=True, sharey=True)

            # maybe this should be user-definable but I personally think this cmap is best for our use case
            cmap = 'copper'

            # make the main plot and color bars
            g1 = ax[0].scatter(in_gamma1_converged, out_gamma1, c = u, marker='.', vmin = min(u), vmax = max(u), cmap = cmap)
            ax[0].text(-0.05, 0.05, '$\gamma_1^{\\rm LOS}$')
            self.util.colorbar(g1, None, 'vertical')

            g2 = ax[1].scatter(in_gamma2_converged, out_gamma2, c = u, marker='.', vmin = min(u), vmax = max(u), cmap = cmap)
            ax[1].text(-0.05, 0.05, '$\gamma_2^{\\rm LOS}$')
            self.util.colorbar(g2, '$u$', 'vertical')

            # now get the cbar colours for the error bars
            norm = matplotlib.colors.Normalize(vmin = min(u), vmax = max(u))
            mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
            u_colour = np.array([(mapper.to_rgba(v)) for v in u])

            # loop over each point to get the right colour for each error bar
            for x, y, e1, e2, color in zip(in_gamma1_converged, out_gamma1, gamma1_lower, gamma1_upper, u_colour):
                ax[0].errorbar(x, y, yerr=np.array(e1,e2), color=color)

            for x, y, e1, e2, color in zip(in_gamma2_converged, out_gamma2, gamma2_lower, gamma2_upper, u_colour):
                ax[1].errorbar(x, y, yerr=np.array(e1,e2), color=color)

            if show_not_converged and len(indices_not_converged) > 0:
                # plot the non-converged ones with crosses and without error bars
                in_gamma1_not_converged = [in_gamma1[i] for i in indices_not_converged]
                in_gamma2_not_converged = [in_gamma2[i] for i in indices_not_converged]

                out_gamma1_not_converged = [s['gamma1_los'][1] for s in summary_not_converged]
                out_gamma2_not_converged = [s['gamma2_los'][1] for s in summary_not_converged]

                # I need some unconverged chains to test this but it should work
                # maybe an ax.text or some kind of artist to say what the x's represent? fiddly...

                ax[0].scatter(in_gamma1_not_converged, out_gamma1_not_converged,
                              c = u, marker = 'x', vmin = min(u), vmax = max(u), cmap = cmap)

                ax[1].scatter(in_gamma2_not_converged, out_gamma2_not_converged,
                              c = u, marker = 'x', vmin = min(u), vmax = max(u), cmap = cmap)

            fig.supxlabel('Input $\gamma_{\\rm LOS}$')
            fig.supylabel('Output $\gamma_{\\rm LOS}$')

            # make an x = y line for the range of our plot
            # min/max should be the same for gamma1 and gamma2
            # for full generality we could generate separate lines for each subplot...
            lims = [
                np.min([ax[0].get_xlim(), ax[0].get_ylim()]),  # min of both axes
                np.max([ax[0].get_xlim(), ax[0].get_ylim()]),  # max of both axes
                ]

            for a in ax:
                a.plot(lims, lims, color = 'black', ls = '--', alpha=0.3, zorder=0)
                a.set_aspect('equal')
                a.set_xlim(lims)
                a.set_ylim(lims)

            if u_max is not None:
                fig.suptitle(r"$u < {}$".format(u_max))

        if save:
            plt.savefig(str(path) + '/plots/' + str(settings['job_name'])+'_input_output_cmap.pdf', dpi=300, bbox_inches='tight')
        if show:
            plt.show()

        else:

            fig, ax = plt.subplots(1, 1, figsize = (7,7), sharex=True, sharey=True)

            plt.errorbar(in_gamma1_converged, out_gamma1, yerr = [gamma1_lower, gamma1_upper],
                         ls = ' ', marker = '.', color = LOS[1], label = r'$\gamma_1^{\rm LOS}$')

            plt.errorbar(in_gamma2_converged, out_gamma2, yerr = [gamma2_lower, gamma2_upper],
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

            if u_max is not None:
                plt.title(r"$u < {}$".format(u_max))

            plt.legend(frameon=False)

        if save:
            plt.savefig(str(path) + '/plots/' + str(settings['job_name'])+'_input_output.pdf', dpi=300, bbox_inches='tight')
        if show:
            plt.show()

        return None


    def contour_plot(self, path, settings, chain_number, plot_params, size, draft=True, save=True, show=True):

        # get the chain name
        filename = str(settings['job_name']) + '_' + str(chain_number) +'.h5'

        # look at the raw chain file to get the number of walkers
        raw_chain = h5py.File(str(path) + '/chains/'+ filename, 'r')
        group = raw_chain['lenstronomy_mcmc_emcee']
        nwalkers = group.attrs['nwalkers']

        # read in the parameters which were sampled in the mcmc
        # (expected_values contains ALL params, including ones which were kept fixed)
        sampled_parameters = np.genfromtxt(str(path) + '/datasets/' + str(settings['job_name'])+ '_sampled_params.csv', dtype='str')

        # rename the columns of sampled parameters to match plot_params
        # this could be condensed into a loop but whatever
        # unnested loops preserve interpretability!
        renamed_lens0 = [name.replace('_lens0', '') for name in sampled_parameters]
        renamed_lens1 = [name.replace('lens1', 'bar') for name in renamed_lens0]
        renamed_lens2 = [name.replace('lens2', 'nfw') for name in renamed_lens1]
        renamed_sl = [name.replace('source_light0', 'sl') for name in renamed_lens2]

        if any('lens_light0' in s for s in renamed_sl):
            # rename the lens light if it's present
            renamed = [name.replace('lens_light0', 'll') for name in renamed_sl]
        else:
            renamed = renamed_sl

        # get the indices corresponding to the params of interest
        param_inds = [renamed.index(p) for p in plot_params]

        reader = emcee.backends.HDFBackend(str(path) + '/chains/' + filename, name='lenstronomy_mcmc_emcee')

        samples = reader.get_chain(discard=settings['n_burn'], flat=True, thin=thin)

        # get the list of LaTeX strings for our params
        labels = self.get_labels(plot_params)

        c = ChainConsumer()

        c.add_chain([samples[:,ind] for ind in param_inds],
                     walkers = nwalkers,
                     parameters = labels)

        # read in the input kwargs for this set of jobs
        input_kwargs = pd.read_csv(str(path) + '/datasets/' + str(settings['job_name'])+ '_input_kwargs.csv')

        # get the expected values for the chain and parameters of interest as a list
        expected_values = input_kwargs.iloc[chain_number][param_inds].to_list()

        if settings['complexity'] == 'perfect':
            color = '#d7301f'
        elif settings['complexity'] == 'perfect minimal':
            color = '#253494'
        else:
            # use default mpl colours; to be updated
            color = None

        c.configure(smooth = True, flip = False, summary = True,
                    spacing = 1.0, max_ticks = 4,
                    colors = color, shade = True, shade_gradient = 0.4,
                    bar_shade = True, linewidths = [3.0],
                    tick_font_size=18, label_font_size=18,
                    usetex = True, serif = True)

        fig = c.plotter.plot(truth=expected_values, figsize = size)

        fig.patch.set_facecolor('white')

        if draft:
            # add a plot title with the job name
            fig.suptitle(settings['job_name'].replace('_', '\_'), fontsize=18)

        if save:
            plt.savefig(str(path) + '/plots/' + str(settings['job_name']) + '_contours.pdf', dpi=300, bbox_inches='tight')
        if show:
            plt.show()

        return None

    def get_labels(self, plot_params):
        '''
        provides a dict to convert the kwarg strings to LaTeX for plotting
        '''

        param_dict = {# LOS
                      'kappa_od': r'$\kappa^{\rm od}$',
                      'omega_od': r'$\omega^{\rm od}$',
                      'gamma1_od': r'$\gamma_1^{\rm od}$',
                      'gamma2_od': r'$\gamma_2^{\rm od}$',
                      'kappa_os': r'$\kappa^{\rm os}$',
                      'omega_os': r'$\omega^{\rm os}$',
                      'gamma1_os': r'$\gamma_1^{\rm os}$',
                      'gamma2_os': r'$\gamma_2^{\rm os}$',
                      'kappa_ds': r'$\kappa^{\rm ds}$',
                      'omega_ds': r'$\omega^{\rm ds}$',
                      'gamma1_ds': r'$\gamma_1^{\rm ds}$',
                      'gamma2_ds': r'$\gamma_2^{\rm ds}$',
                      'kappa_los': r'$\kappa^{\rm LOS}$',
                      'omega_los': r'$\omega^{\rm LOS}$',
                      'gamma1_los': r'$\gamma_1^{\rm LOS}$',
                      'gamma2_los': r'$\gamma_2^{\rm LOS}$',
                      # baryons
                      'k_eff_bar': r'$k_{\rm eff}$',
                      'R_sersic_bar': r'$R_{\rm S\acute{e}rsic}^{\rm bar}$',
                      'n_sersic_bar': r'$n_{\rm S\acute{e}rsic}^{\rm bar}$',
                      'e1_bar': r'$e_1^{\rm bar}$',
                      'e2_bar': r'$e_2{\rm bar}$',
                      # DM
                      'R_s': r'$R_s$',
                      'alpha_Rs': r'$\alpha_{R_s}$',
                      'x_nfw': r'$x^{\rm DM}$',
                      'y_nfw': r'$y^{\rm DM}$',
                      'e1_nfw': r'$e_1^{\rm DM}$',
                      'e2_nfw': r'$e_2^{\rm DM}$',
                      # source
                      'R_sersic_sl': r'$R_{\rm S\acute{e}rsic}^{\rm source}$',
                      'n_sersic_sl': r'$n_{\rm S\acute{e}rsic}^{\rm source}$',
                      'e1_sl': r'$e_1^{\rm source}$',
                      'e2_sl': r'$e_2^{\rm source}$',
                      'x_sl': r'$x^{\rm source}$',
                      'y_sl': r'$y^{\rm source}$',
                      # lens light
                      'R_sersic_ll': r'$R_{\rm S\acute{e}rsic}^{\rm lens\ light}$',
                      'n_sersic_ll': r'$n_{\rm S\acute{e}rsic}^{\rm lens\ light}$',
                      'e1_ll': r'$e_1^{\rm lens\ light}$',
                      'e2_ll': r'$e_2^{\rm lens\ light}$',
                      'x_ll': r'$x^{\rm lens\ light}$',
                      'y_ll': r'$y^{\rm lens\ light}$'}

        tex_list = [param_dict[p] for p in plot_params]

        return tex_list
