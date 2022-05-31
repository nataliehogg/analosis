"""
everything gets run from here
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from colossus.cosmology import cosmology as colcos
from colossus.lss import mass_function
#import random as r
#from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.Util.simulation_util as sim_util
import lenstronomy.Util.image_util as image_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Workflow.fitting_sequence import FittingSequence
#from lenstronomy.LensModel.Profiles.sersic_utils import SersicUtil
#import pickle

#from analosis.image.composite_lens.composite_lens_generator import CompositeLens
#from analosis.image.distributed_haloes.distributed_haloes_generator import DistributedHaloes

from analosis.utilities.useful_functions import Utilities
from analosis.image.mock_generator import Mocks
from analosis.image.image_generator import Image
from analosis.analysis.mcmc import MCMC

class Run:

    def __init__(self, cpars, settings, parameters):

        path = (Path(__file__).parent/'results/').resolve()

        print('Running the {} case with the following settings:\n\nModel: {}\nNumber of runs: {}\nLens light: {}'
              .format(settings['scenario'], settings['complexity'],
                      settings['number_of_images'], settings['lens_light']))

        colcos.setCosmology(cpars['id'])
        cosmo = FlatLambdaCDM(H0 = cpars['H0'], Om0 = cpars['Om'])
        util = Utilities(cosmo, path)

        if settings['scenario'] == 'composite lens':
            self.mocks = Mocks(util=util,
                                 scenario=settings['scenario'],
                                 path=path,
                                 number_of_images=settings['number_of_images'],
                                 Einstein_radius_min=parameters['Einstein_radius_min'],
                                 gamma_max=parameters['maximum_shear'],
                                 sigma_halo_offset=parameters['sigma_halo_offset'],
                                 maximum_source_offset_factor=parameters['maximum_source_offset_factor'])

            # get the dictionary of kwargs from the mock generator
            kwargs_dict = self.mocks.draw_kwargs()
            
            # get the list of Einstein radii (useful for pictures)
            Einstein_radii = self.mocks.Einstein_radii

            # convert these into individual dataframes
            # these are what will get passed around in the code
            baryons = util.get_dataframe(kwargs_dict['baryons'])
            halo = util.get_dataframe(kwargs_dict['halo'])
            los = util.get_dataframe(kwargs_dict['los'])
            lens_light = util.get_dataframe(kwargs_dict['lens_light'])
            source = util.get_dataframe(kwargs_dict['source'])

            # combine the dataframes for saving to file
            complete_data = util.combine_dataframes([baryons, halo, los, lens_light, source])
            util.save_input_kwargs(complete_data)

            # rename the dataframe columns for lenstronomy
            # we want them to have distinguishable names when we save the dataset above
            # otherwise we will not know which R_sersic or e1 or x or whatever
            # corresponds to which component
            # possibly we can make this step less clunky
            baryons = baryons.rename(index = str, columns = {'k_eff_bar': 'k_eff',
                                                             'R_sersic_bar': 'R_sersic',
                                                             'n_sersic_bar': 'n_sersic',
                                                             'x_bar': 'center_x',
                                                             'y_bar': 'center_y',
                                                             'e1_bar': 'e1',
                                                             'e2_bar': 'e2'})

            halo = halo.rename(index = str, columns = {'x_nfw': 'center_x',
                                                      'y_nfw': 'center_y',
                                                      'e1_nfw': 'e1',
                                                      'e2_nfw': 'e2'})

            lens_light = lens_light.rename(index = str, columns = {'amp_ll': 'amp',
                                                                  'R_sersic_ll': 'R_sersic',
                                                                  'n_sersic_ll': 'n_sersic',
                                                                  'x_ll': 'center_x',
                                                                  'y_ll': 'center_y',
                                                                  'e1_ll': 'e1',
                                                                  'e2_ll': 'e2'})

            source = source.rename(index = str, columns = {'amp_sl': 'amp',
                                                          'R_sersic_sl': 'R_sersic',
                                                          'n_sersic_sl': 'n_sersic',
                                                          'x_sl': 'center_x',
                                                          'y_sl': 'center_y',
                                                          'e1_sl': 'e1',
                                                          'e2_sl': 'e2'})

            # generate the image and the associated data kwargs for either plotting or fitting
            im = Image()
            kwargs_data, kwargs_psf, kwargs_numerics = im.generate_image(settings,
                                                                         baryons,
                                                                         halo,
                                                                         los,
                                                                         lens_light,
                                                                         source,
                                                                         Einstein_radii,
                                                                         path)

        if settings['MCMC'] == True:
            chain = MCMC(settings, baryons, halo, los, lens_light, source, kwargs_data, kwargs_psf, kwargs_numerics, path)
        elif settings['MCMC'] == False:
            print('MCMC will not be run.')
        else:
            print('MCMC must be True or False.')

        print('\nAnalysis complete and results saved at {}.'.format(path))

    def pathfinder(self):
        path = (Path(__file__).parent/'results/').resolve()
        return str(path)
