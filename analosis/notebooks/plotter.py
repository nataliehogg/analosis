import numpy as np
import pandas as pd
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import seaborn as sns
import emcee
from chainconsumer import ChainConsumer
# from lenstronomy.Cosmo.lens_cosmo import LensCosmo
# from astropy.cosmology import FlatLambdaCDM
# from lenstronomy.ImSim.image_model import ImageModel
# import lenstronomy.Util.simulation_util as sim_util
# import lenstronomy.Util.image_util as image_util
# from lenstronomy.Data.imaging_data import ImageData
# from lenstronomy.Data.psf import PSF
# from lenstronomy.LensModel.lens_model import LensModel
# from lenstronomy.LightModel.light_model import LightModel
# from lenstronomy.Workflow.fitting_sequence import FittingSequence
import pickle
import corner

c = ChainConsumer()

# Use TeX
rc('text', usetex=True)
rc('font', family='serif')
matplotlib.rcParams.update({'font.size': 18})

# colours for plots
LOS         = ['#a6dba0','#5aae61','#1b7837']
LOS_minimal = ['#c2a5cf', '#9970ab', '#762a83']

user = 'Pierre' # this is to avoid changing paths all the time

if user == 'Natalie':
    inpath = r'/home/natalie/Documents/Projects/los_effects/figures/composite_lens/hundred_runs/'
    path_list = [inpath + 'Matteo/100522/', inpath + 'Pierre/100522/', inpath + 'Natalie/100522/']
elif user == 'Pierre':
    inpath = '/Users/pierrefleury/GitHub/analosis/analosis/results/'
    path_list = [inpath]
else:
    print("I don't know this user, please specify a path for them.")


plot_image = True
plot_chains = True
show_not_converged = False # show the results for non-converged chains on plots
quality_cut = 0.85 # criterion for the quality of the image, use 0 if you don't want a cut
number_of_images = 60 # should be the number of chains you have inside the directory specified as outpath
number_of_walkers = 220 # 10*number of parameters (23 for incomplete model, 25 for full model)

# Define the quality of images (the criterion is very empirical here)
kwargs = pd.read_csv(path_list[0] + 'datasets/input_kwargs.csv')
R_s = kwargs['R_sersic_sl'].to_numpy()
beta = np.sqrt(kwargs['x_sl']**2. + kwargs['y_sl']**2.).to_numpy()
quality = 1 / (1 + (beta/3/R_s)**2)


if plot_image:

    # to do: add a loop over items in path_list
    filename = path_list[0] + 'datasets/image_list.pickle'
    infile = open(filename,'rb')
    image_list = pickle.load(infile)
    infile.close()

    cmap_string = 'bone'
    cmap = plt.get_cmap(cmap_string)
    cmap.set_bad(color='k', alpha=1.)
    cmap.set_under('k')

    v_min = -4
    v_max = 1

    f, ax = plt.subplots(int(np.sqrt(number_of_images)), int(np.sqrt(number_of_images)),
                         figsize=(10, 10), sharex=False, sharey=False)

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

    #plt.savefig(path_list[0] + 'images_' + data_set + '.pdf', dpi=300, bbox_inches='tight')


if plot_chains:
    fig, ax = plt.subplots(1, 1, figsize = (7,7), sharex=True, sharey=True)
    method = 'chains'

    in_kwargs = [pd.read_csv(p + 'datasets/input_kwargs.csv') for p in path_list]

    in_gammas1 = [in_kwargs[i]['gamma1_los'] for i in range(len(path_list))]
    in_gammas2 = [in_kwargs[i]['gamma2_los'] for i in range(len(path_list))]

    # flatten the lists
    in_gamma1 = [item for sublist in in_gammas1 for item in sublist]
    in_gamma2 = [item for sublist in in_gammas2 for item in sublist]

    n_burn = 1000 # this should be the same as when the run was done
    #number_of_images = 25

    for p in range(len(path_list)):
        for i in range(number_of_images):
            chain = path_list[p] + 'chains/' + 'fit_image_' + str(i) + '.h5'
            reader = emcee.backends.HDFBackend(filename = chain, name = 'lenstronomy_mcmc_emcee')
            samples = reader.get_chain(discard = n_burn, flat = True, thin = 10)
            c.add_chain(samples[:,2:4], walkers=number_of_walkers, parameters = ['gamma1_los', 'gamma2_los'])
    summary = c.analysis.get_summary()
    
    # Remove the images under a certain quality
    if quality_cut is not None:
        summary = [s for i,s in enumerate(summary) if quality[i]>quality_cut]
        in_gamma1 = [g for i, g in enumerate(in_gamma1) if quality[i]>quality_cut]
        in_gamma2 = [g for i, g in enumerate(in_gamma2) if quality[i]>quality_cut]
        print(len(summary))
        
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
    plt.savefig(
        path_list[0] + '/plots/inference_gamma_Q>{}.pdf'.format(quality_cut),
        dpi=300, bbox_inches='tight')
    plt.show()
