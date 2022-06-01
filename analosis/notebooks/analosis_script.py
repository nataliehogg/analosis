# set the path to analosis relative to this script
import sys
sys.path.append('../..')

# import the Run class which allows you to run the analysis of your choice
# and the Plots class for plotting
from analosis.run import Run
from analosis.analysis.plots import Plots

cosmology = {'id': 'planck18', 'H0': 67.4, 'Om': 0.315}

settings = {'scenario': 'composite lens',
            'complexity': 'perfect minimal',
            'lens_light': False,
            'number_of_images': 1,
            'MCMC': False,
            'n_burn': 1,
            'n_run': 1}

parameters = {'maximum_shear': 0.03,
              'Einstein_radius_min': 0.5, # arcsec
              'maximum_source_offset_factor': 1,
              'sigma_halo_offset': 300} # pc

result = Run(cosmology, settings, parameters)

p = Plots()

path = result.pathfinder()

p.image_plot(path, settings['number_of_images'], number_of_columns=10, save=True, show=False)
