import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc, gridspec, cm, rcParams, colors, offsetbox
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmasher as cmr
from useful_functions import estimate_quality

def colorbar(mappable, lab, ori):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size = '5%', pad = 0.15)
    cbar = fig.colorbar(mappable, cax = cax, label=lab, orientation = ori)
    plt.sca(last_axes)
    return cbar

rc('text', usetex=True)
rc('font', family='serif')
rcParams.update({'font.size': 18})

path = '/Users/pierrefleury/ownCloud/Seminars and presentations/LOS/input-output/'

job = 'pm'

# Estimate quality of the images
in_kwargs = pd.read_csv(path + 'golden_sample_input_kwargs.csv')
qualities = estimate_quality(in_kwargs, snr_cut=5)
qualities = np.array(qualities)
qualities = np.log10(qualities/1000)

# Reduce sample to have only converged indices
indices_converged_g1 = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18,
                        20, 21, 22, 23, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36,
                        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 51, 53,
                        54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
q_g1 = [q for i, q in enumerate(qualities) if i in indices_converged_g1]

indices_converged_g2 = [0, 1, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                        18, 19, 20, 21, 22, 24, 25, 26, 28, 29, 30, 31, 33, 34,
                        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 48, 53, 55,
                        56, 57, 58, 59, 60, 61, 62, 63]
q_g2 = [q for i, q in enumerate(qualities) if i in indices_converged_g2]

summary_g1_kwargs = pd.read_csv(path + 'summary_gamma1_'+job+'.csv')
summary_g2_kwargs = pd.read_csv(path + 'summary_gamma2_'+job+'.csv')

if job == 'moe':
    cmap = cmr.get_sub_cmap('cmr.nuclear', 0.0, 0.85) # removing the white at the top of the cmap
    title = 'Missing offset and ellipticity model'
elif job == 'me':
    cmap = cmr.get_sub_cmap('cmr.flamingo', 0.0, 0.85)
    title = 'Missing ellipticity model'
elif job =='pl':
    cmap =  cmr.get_sub_cmap('cmr.arctic', 0.0, 0.85)
    title = 'Elliptical power law model'
elif job == 'pm':
    cmap = 'copper'
    title = 'Perfect minimal model'
else:
    cmap = 'viridis'

in_gamma1_converged = summary_g1_kwargs['in_gamma1'].to_list()
out_gamma1          = summary_g1_kwargs['out_gamma1'].to_list()
g1_lower_error      = summary_g1_kwargs['g1_lower'].to_numpy()
g1_upper_error      = summary_g1_kwargs['g1_upper'].to_numpy()
#b_g1                = summary_g1_kwargs['b'].to_list()

# estimate the average error
#errg1 = sum((g1_upper_error + g1_lower_error)) / 2 / len(g1_upper_error)
#print(errg1)

in_gamma2_converged = summary_g2_kwargs['in_gamma2'].to_list()
out_gamma2          = summary_g2_kwargs['out_gamma2'].to_list()
g2_lower_error      = summary_g2_kwargs['g2_lower'].to_numpy()
g2_upper_error      = summary_g2_kwargs['g2_upper'].to_numpy()
#b_g2                = summary_g2_kwargs['b'].to_list()


# estimate the average error
#errg2 = sum((g2_upper_error + g2_lower_error)) / 2 / len(g2_upper_error)
#print(errg2)

#b_max = 1.0
q_min = min(qualities)
q_max = max(qualities)

fig, ax = plt.subplots(1, 2, figsize = (11,5), sharex=True, sharey=True)

# make the main plot and color bars
g1 = ax[0].scatter(in_gamma1_converged, out_gamma1, c=q_g1, marker='.', vmin=q_min, vmax=q_max, cmap=cmap)
g1_text = offsetbox.AnchoredText('$\gamma_1^{\\rm LOS}$', loc=2, frameon=False)
ax[0].add_artist(g1_text)
colorbar(g1, None, 'vertical')

g2 = ax[1].scatter(in_gamma2_converged, out_gamma2, c=q_g2, marker='.', vmin=q_min, vmax=q_max, cmap=cmap)
g2_text = offsetbox.AnchoredText('$\gamma_2^{\\rm LOS}$', loc=2, frameon=False)
ax[1].add_artist(g2_text)
colorbar(g2, r'$\log_{10}(Q/1000)$', 'vertical')

# now get the cbar colours for the error bars
norm_g1 = colors.Normalize(vmin=q_min, vmax=q_max)
mapper_g1 = cm.ScalarMappable(norm=norm_g1, cmap=cmap)
q_colour_g1 = np.array([(mapper_g1.to_rgba(q)) for q in q_g1])

norm_g2 = colors.Normalize(vmin=q_min, vmax=q_max)
mapper_g2 = cm.ScalarMappable(norm=norm_g2, cmap=cmap)
q_colour_g2 = np.array([(mapper_g2.to_rgba(q)) for q in q_g2])

# loop over each point to get the right colour for each error bar
for x, y, e1, e2, color in zip(in_gamma1_converged, out_gamma1, g1_lower_error, g1_upper_error, q_colour_g1):
    ax[0].errorbar(x, y, yerr=[[e1], [e2]], color=color)

for x, y, e1, e2, color in zip(in_gamma2_converged, out_gamma2, g2_lower_error, g2_upper_error, q_colour_g2):
    ax[1].errorbar(x, y, yerr=[[e1], [e2]], color=color)

fig.supxlabel('Input $\gamma_{\\rm LOS}$')
fig.supylabel('Output $\gamma_{\\rm LOS}$')

# make an x = y line for the range of our plot
# min/max should be the same for gamma1 and gamma2
# for full generality we could generate separate lines for each subplot...
# make an x = y line for the range of our plot
# min/max should be the same for gamma1 and gamma2
# for full generality we could generate separate lines for each subplot...
#lims = [
#    np.min([ax[0].get_xlim(), ax[0].get_ylim()]),  # min of both axes
#    np.max([ax[0].get_xlim(), ax[0].get_ylim()]),  # max of both axes
#    ]
lims = [-0.1, 0.1]

for a in ax:
    a.plot(lims, lims, color = 'black', ls = '--', alpha=0.3, zorder=0)
    a.set_aspect('equal')
    a.set_xlim(lims)
    a.set_ylim(lims)

plt.savefig(path+'input_output.pdf', dpi=300, bbox_inches='tight')
plt.show()
