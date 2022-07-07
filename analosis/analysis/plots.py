import numpy as np
import pandas as pd
import matplotlib
from matplotlib import rc, gridspec, cm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
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


    def image_plot(self, path, settings, number_of_columns=5, b_max=None, save=True, show=True):
        print('Preparing image plot...')
        kwargs = pd.read_csv(str(path) + '/datasets/'+ str(settings['job_name']) + '_input_kwargs.csv')

        number_of_images = len(kwargs)

        if number_of_images > 10:
            print('The plotter is slow for this many images but the result looks soooo good. Patience, my young padawan!')

        # Check if a starting index has been specified
        try:
            i_start = settings['starting_index']
            assert i_start > 0
            filename = str(path) + '/datasets/' + str(settings['job_name']) + '_image_list_' + str(i_start) + '.pickle'
        except KeyError:
            i_start = 0
            filename = str(path) + '/datasets/' + str(settings['job_name']) + '_image_list.pickle'
        except AssertionError:
            filename = str(path) + '/datasets/' + str(settings['job_name']) + '_image_list.pickle'

        # Define the quality of images from the impact parameter
        # normalised with the source half-light radius
        # position of the lens centre of mass
        try:
            m_bar  = kwargs['mass_bar'].loc[i_start:,].to_numpy()
            m_halo = kwargs['virial_mass_nfw'].loc[i_start:,].to_numpy()
            x_bar  = kwargs['x_bar'].loc[i_start:,].to_numpy()
            y_bar  = kwargs['y_bar'].loc[i_start:,].to_numpy()
            x_nfw  = kwargs['x_nfw'].loc[i_start:,].to_numpy()
            y_nfw  = kwargs['y_nfw'].loc[i_start:,].to_numpy()
            x_cm = (m_bar * x_bar + m_halo * x_nfw) / (m_bar + m_halo)
            y_cm = (m_bar * y_bar + m_halo * y_nfw) / (m_bar + m_halo)
        except KeyError:
            # this is for retrocompatibility before we had the masses in kwargs
            # in this case we fix the centre of mass on the optical axis
            x_cm = y_cm = 0
        # position of the source
        x_s = kwargs['x_sl'].loc[i_start:,].to_numpy()
        y_s = kwargs['y_sl'].loc[i_start:,].to_numpy()
        # normalised impact parameter to the centre of mass
        R_s = kwargs['R_sersic_sl'].loc[i_start:,].to_numpy()
        b   = np.sqrt((x_s - x_cm)**2 + (y_s - y_cm)**2) / R_s

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

            if b_max is not None and b[n] > b_max:
                ax.set_title(r'$b = {:.2f}$'.format(b[n]), fontsize=8)
                ax.plot([0,1],[0,1], color='red')
                ax.plot([0,1],[1,0], color='red')
            else:
                ax.set_title(r'$b = {:.2f}$'.format(b[n]), fontsize=8)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.autoscale(False)

        # fig.tight_layout()

        if save:
            plt.savefig(str(path) + '/plots/' + str(settings['job_name']) + '_image.pdf', dpi=300, bbox_inches='tight')
        if show:
            plt.show()

        return None


    def input_output_plot(self, path, settings, b_max=None, show_not_converged=True, use_colourmap=True, save=True, show=True):

        in_kwargs = pd.read_csv(path + '/datasets/' + str(settings['job_name']) + '_input_kwargs.csv')

        # define the impact parameter normalised by the source half-light radius
        beta = np.sqrt(in_kwargs['x_sl']**2. + in_kwargs['y_sl']**2.).to_numpy()
        R_s = in_kwargs['R_sersic_sl'].to_numpy()
        b = beta / R_s

        in_gamma1 = in_kwargs['gamma1_los']
        in_gamma2 = in_kwargs['gamma2_los']

        c = ChainConsumer()

        for i in range(len(in_kwargs)):
            chain = path + '/chains/' + str(settings['job_name']) + '_' + str(settings['complexity']) +'_' + str(i) + '.h5'
            reader = emcee.backends.HDFBackend(filename = chain, name = 'lenstronomy_mcmc_emcee')
            samples = reader.get_chain(discard = settings['n_burn'], flat = True, thin = thin)
            c.add_chain(samples[:,2:4], walkers=np.shape(samples)[0], parameters = ['gamma1_los', 'gamma2_los'])
        summary = c.analysis.get_summary()

        # Remove the images under a certain quality
        if b_max is not None:
            summary   = [s for i, s in enumerate(summary)   if b[i] < b_max]
            in_gamma1 = [g for i, g in enumerate(in_gamma1) if b[i] < b_max]
            in_gamma2 = [g for i, g in enumerate(in_gamma2) if b[i] < b_max]

        # Isolate the cases where the MCMC did not converge
        g1_summary_converged = []
        g1_indices_converged = []
        g1_summary_not_converged = []
        g1_indices_not_converged = []

        g2_summary_converged = []
        g2_indices_converged = []
        g2_summary_not_converged = []
        g2_indices_not_converged = []
        for i in range(len(summary)):
            s = summary[i]
            if (s['gamma1_los'][0] is None
                or s['gamma1_los'][2] is None):
                g1_indices_not_converged.append(i)
                g1_summary_not_converged.append(s)
            else:
                g1_indices_converged.append(i)
                g1_summary_converged.append(s)

        for i in range(len(summary)):
            s = summary[i]
            if (s['gamma2_los'][0] is None
                or s['gamma2_los'][2] is None):
                g2_indices_not_converged.append(i)
                g2_summary_not_converged.append(s)
            else:
                g2_indices_converged.append(i)
                g2_summary_converged.append(s)


        # plot the converged ones with error bars
        in_gamma1_converged = [in_gamma1[i] for i in g1_indices_converged]
        in_gamma2_converged = [in_gamma2[i] for i in g2_indices_converged]
        b_g1                = [b[i] for i in range(len(in_gamma1_converged))]
        b_g2                = [b[i] for i in range(len(in_gamma2_converged))]


        out_gamma1 = [s['gamma1_los'][1] for s in g1_summary_converged]
        out_gamma2 = [s['gamma2_los'][1] for s in g2_summary_converged]
        gamma1_lower = [np.abs(s['gamma1_los'][0] - s['gamma1_los'][1]) for s in g1_summary_converged]
        gamma1_upper = [np.abs(s['gamma1_los'][2] - s['gamma1_los'][1]) for s in g1_summary_converged]
        gamma2_lower = [np.abs(s['gamma2_los'][0] - s['gamma2_los'][1]) for s in g2_summary_converged]
        gamma2_upper = [np.abs(s['gamma2_los'][2] - s['gamma2_los'][1]) for s in g2_summary_converged]

        fig, ax = plt.subplots(1, 2, figsize = (11,5), sharex=True, sharey=True)

        cmap = 'copper'

        # make the main plot and color bars
        b_max_cb = max(max(b_g1), max(b_g2))
        g1 = ax[0].scatter(in_gamma1_converged, out_gamma1, c=b_g1, marker='.', vmin=0, vmax=b_max_cb, cmap=cmap)
        g1_text = AnchoredText('$\gamma_1^{\\rm LOS}$', loc=2, frameon=False)
        ax[0].add_artist(g1_text)
        self.util.colorbar(g1, None, 'vertical')

        g2 = ax[1].scatter(in_gamma2_converged, out_gamma2, c=b_g2, marker='.', vmin=0, vmax=b_max_cb, cmap=cmap)
        g2_text = AnchoredText('$\gamma_2^{\\rm LOS}$', loc=2, frameon=False)
        ax[1].add_artist(g2_text)
        self.util.colorbar(g2, r'$b=\beta/R_{\rm source}$', 'vertical')

        # now get the cbar colours for the error bars
        norm_g1 = matplotlib.colors.Normalize(vmin=0, vmax=b_max_cb)
        mapper_g1 = cm.ScalarMappable(norm=norm_g1, cmap=cmap)
        b_colour_g1 = np.array([(mapper_g1.to_rgba(b)) for b in b_g1])

        norm_g2 = matplotlib.colors.Normalize(vmin=0, vmax=b_max_cb)
        mapper_g2 = cm.ScalarMappable(norm=norm_g2, cmap=cmap)
        b_colour_g2 = np.array([(mapper_g2.to_rgba(b)) for b in b_g2])

        # loop over each point to get the right colour for each error bar
        for x, y, e1, e2, color in zip(in_gamma1_converged, out_gamma1, gamma1_lower, gamma1_upper, b_colour_g1):
            ax[0].errorbar(x, y, yerr=np.array(e1,e2), color=color)

        for x, y, e1, e2, color in zip(in_gamma2_converged, out_gamma2, gamma2_lower, gamma2_upper, b_colour_g2):
            ax[1].errorbar(x, y, yerr=np.array(e1,e2), color=color)

        if show_not_converged and len(g1_indices_not_converged) > 0:
            # plot the non-converged ones with crosses and without error bars
            in_gamma1_not_converged = [in_gamma1[i] for i in g1_indices_not_converged]
            out_gamma1_not_converged = [s['gamma1_los'][1] for s in g1_summary_not_converged]

            ax[0].plot(in_gamma1_not_converged, out_gamma1_not_converged, marker = 'o', ls = '',
                       markeredgecolor = 'black', markerfacecolor='None')

        if show_not_converged and len(g2_indices_not_converged) > 0:

            in_gamma2_not_converged = [in_gamma2[i] for i in g2_indices_not_converged]

            out_gamma2_not_converged = [s['gamma2_los'][1] for s in g2_summary_not_converged]

            ax[1].plot(in_gamma2_not_converged, out_gamma2_not_converged,  marker = 'o', ls = '',
                       markeredgecolor = 'black', markerfacecolor='None')

        fig.supxlabel('Input $\gamma^{\\rm LOS}$')
        fig.supylabel('Output $\gamma^{\\rm LOS}$')

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

        if b_max is not None:
            fig.suptitle(r"$b < {}$".format(b_max))

        if save:
            plt.savefig(str(path) + '/plots/' + str(settings['job_name']) + '_' +str(settings['complexity']) + '_input_output.pdf', dpi=300, bbox_inches='tight')
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

            ax.set_xlabel('Input $\gamma^{\\rm LOS}$')
            ax.set_ylabel('Output $\gamma^{\\rm LOS}$')

            # make an x = y line for the range of our plot
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
                ]

            ax.plot(lims, lims, color = 'black', ls = '--', alpha=0.3, zorder=0)
            ax.set_aspect('equal')
            ax.set_xlim(lims)
            ax.set_ylim(lims)

            if b_max is not None:
                plt.title(r"$b < {}$".format(b_max))

            plt.legend(frameon=False)

            if save:
                plt.savefig(str(path) + '/plots/' + str(settings['job_name'])+ '_' +str(settings['complexity']) + '_input_output.pdf', dpi=300, bbox_inches='tight')
            if show:
                plt.show()

        return None


    def emcee_contour_plot(self, path, settings, chain_number, plot_params, size, draft=True, save=True, show=True):

        # get the chain name
        filename = str(settings['job_name']) + '_' + str(settings['complexity']) +'_' + str(chain_number) +'.h5'

        # look at the raw chain file to get the number of walkers
        raw_chain = h5py.File(str(path) + '/chains/'+ filename, 'r')
        group = raw_chain['lenstronomy_mcmc_emcee']
        nwalkers = group.attrs['nwalkers']

        # read in the parameters which were sampled in the mcmc
        # (expected_values contains ALL params, including ones which were kept fixed)
        sampled_parameters = np.genfromtxt(str(path) + '/datasets/' + str(settings['job_name']) + '_' + str(settings['complexity']) + '_sampled_params.csv', dtype='str')

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
        input_kwargs = pd.read_csv(str(path) + '/datasets/' + str(settings['job_name']) + '_input_kwargs.csv')

        # get the expected values for the chain and parameters of interest as a list
        expected_values = input_kwargs.iloc[chain_number][plot_params].to_list()

        if settings['complexity'] == 'perfect':
            color = '#d7301f'
        elif settings['complexity'] == 'perfect_minimal':
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
            plt.savefig(str(path) + '/plots/' + str(settings['job_name']) + '_' + str(settings['complexity']) + '_contours_' +str(chain_number)+'.pdf', dpi=300, bbox_inches='tight')
        if show:
            plt.show()

        return None

    def zeus_contour_plot(self, path, settings, chain_number, plot_params, size, draft=True, save=True, show=True):
        '''
        plot contours from zeus chains
        '''

        filename = str(settings['job_name']) + '_' + str(settings['complexity']) +'_' + str(chain_number) +'.h5'

        with h5py.File(str(path) + '/chains/'+ filename, 'r') as hf:
            samples = np.copy(hf['samples'])
            logprob_samples = np.copy(hf['logprob'])

        nwalkers = samples.shape[1]
        ndim = samples.shape[2] # number of parameters sampled in the chain

        flat_samples = samples.reshape((-1, ndim)) # flatten the chain

        # read in the parameters which were sampled in the mcmc
        # (expected_values contains ALL params, including ones which were kept fixed)
        sampled_parameters = np.genfromtxt(str(path) + '/datasets/' + str(settings['job_name']) + '_' + str(settings['complexity']) + '_sampled_params.csv', dtype='str')

        # rename the columns of sampled parameters to match plot_params
        # this could be condensed into a loop but whatever
        # unnested loops preserve interpretability!
        renamed_lens0 = [name.replace('_lens0', '') for name in sampled_parameters]
        renamed_lens1 = [name.replace('lens1', 'bar') for name in renamed_lens0]
        renamed_lens2 = [name.replace('lens2', 'nfw') for name in renamed_lens1]
        renamed_sl    = [name.replace('source_light0', 'sl') for name in renamed_lens2]

        if any('lens_light0' in s for s in renamed_sl):
            # rename the lens light if it's present
            renamed = [name.replace('lens_light0', 'll') for name in renamed_sl]
        else:
            renamed = renamed_sl

        # get the indices corresponding to the params of interest
        param_inds = [renamed.index(p) for p in plot_params]

        # get the list of LaTeX strings for our params
        labels = self.get_labels(plot_params)

        c = ChainConsumer()

        c.add_chain([flat_samples[:,ind] for ind in param_inds],
                     walkers = nwalkers,
                     parameters = labels)

        # read in the input kwargs for this set of jobs
        input_kwargs = pd.read_csv(str(path) + '/datasets/' + str(settings['job_name']) + '_input_kwargs.csv')

        # get the expected values for the chain and parameters of interest as a list
        expected_values = input_kwargs.iloc[chain_number][plot_params].to_list()

        if settings['complexity'] == 'perfect':
            color = '#d7301f'
        elif settings['complexity'] == 'perfect_minimal':
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
            plt.savefig(str(path) + '/plots/' + str(settings['job_name']) + '_' + str(settings['complexity']) + '_contours_' +str(chain_number)+'.pdf', dpi=300, bbox_inches='tight')
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
                      'e2_bar': r'$e_2^{\rm bar}$',
                      # DM
                      # all of these need to be fixed!
                      'Rs': r'$R_s$',
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
