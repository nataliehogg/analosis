"""
everything gets run from here
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# cosmology
from colossus.cosmology import cosmology as colcos
from colossus.lss import mass_function
from astropy.cosmology import FlatLambdaCDM

# lenstronomy
from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.Util.simulation_util as sim_util
import lenstronomy.Util.image_util as image_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Workflow.fitting_sequence import FittingSequence

# analosis
from analosis.utilities.useful_functions import Utilities
from analosis.image.mock_generator import Mocks
from analosis.image.image_generator import Image
from analosis.analysis.mcmc import MCMC

class Run:

    def __init__(self, cpars, settings, parameters):

        path = (Path(__file__).parent/'results/').resolve()

        # print('Running the {} case with the following settings:\n\nModel: {}\nNumber of images: {}\nLens light: {}'
        #       .format(settings['scenario'], settings['complexity'],
        #               settings['number_of_images'], settings['lens_light']))

        colcos.setCosmology(cpars['id'])
        cosmo = FlatLambdaCDM(H0 = cpars['H0'], Om0 = cpars['Om'])
        util = Utilities(cosmo, path)
        self.settings = settings
        #todo: save the settings in file

        # set the starting index to zero if not specified
        try:
            assert 'starting_index' in self.settings.keys()
        except AssertionError:
            self.settings['starting_index'] = 0

        # set the source perturbation to zero if not specified
        try:
            source_perturbations = parameters['source_perturbations']
        except KeyError:
            source_perturbations = []

        if settings['scenario'] == 'composite lens':
            self.mocks = Mocks(
                util=util,
                scenario=self.settings['scenario'],
                path=path,
                number_of_images=self.settings['number_of_images'],
                Einstein_radius_min=parameters['Einstein_radius_min'],
                min_aspect_ratio_source=parameters['min_aspect_ratio_source'],
                min_aspect_ratio_baryons=parameters['min_aspect_ratio_baryons'],
                min_aspect_ratio_nfw=parameters['min_aspect_ratio_nfw'],
                gamma_max=parameters['maximum_shear'],
                sigma_halo_offset=parameters['sigma_halo_offset'],
                maximum_source_offset_factor=parameters['maximum_source_offset_factor'])

            if self.settings['generate_image'] == True:
                # get the dictionary of kwargs from the mock generator
                kwargs_dict = self.mocks.draw_kwargs()

                # extract Einstein radii and other useful parameters
                Einstein_radii = self.mocks.Einstein_radii
                Einstein_radii_dataframe = util.get_dataframe({'theta_E': Einstein_radii})
                mass_bar_dataframe = util.get_dataframe({'mass_bar': self.mocks.masses_baryons})
                mass_nfw_dataframe = util.get_dataframe({'virial_mass_nfw': self.mocks.masses_haloes})

                # convert these into individual dataframes
                # these are what will get passed around in the code
                baryons = util.get_dataframe(kwargs_dict['baryons'])
                halo = util.get_dataframe(kwargs_dict['halo'])
                los = util.get_dataframe(kwargs_dict['los'])
                lens_light = util.get_dataframe(kwargs_dict['lens_light'])
                source = util.get_dataframe(kwargs_dict['source'])

                # combine the dataframes for saving to file
                # in the same order as the params are put into the MCMC for future ease of plotting
                complete_data = util.combine_dataframes(
                    [los, baryons, mass_bar_dataframe, halo, mass_nfw_dataframe,
                     source, lens_light, Einstein_radii_dataframe])

                if self.settings['starting_index'] == 0:
                    util.save_input_kwargs(self.settings, complete_data)
                else:
                    starting_index_dataframe = util.append_from_starting_index(path, self.settings, complete_data)
                    util.save_input_kwargs(self.settings, starting_index_dataframe)

                # rename the dataframes for lenstronomy
                baryons, halo, lens_light, source = util.rename_kwargs(baryons, halo, lens_light, source)

                # generate the image and the associated data kwargs for either plotting or fitting
                im = Image()
                im.generate_image(self.settings, baryons, halo, los, lens_light, source, Einstein_radii, path)
            else:
                print('New images will not be generated.')
                pass

        if self.settings['MCMC'] == True:

            input_kwargs = pd.read_csv(str(path) + '/datasets/' + str(settings['job_name']) + '_input_kwargs.csv')

            los_cols = ['kappa_os', 'gamma1_os', 'gamma2_os', 'omega_os',
                        'kappa_od', 'gamma1_od', 'gamma2_od', 'omega_od',
                        'kappa_ds', 'gamma1_ds', 'gamma2_ds', 'omega_ds',
                        'kappa_los', 'gamma1_los', 'gamma2_los', 'omega_los']

            bar_cols = ['R_sersic_bar', 'n_sersic_bar', 'k_eff_bar', 'e1_bar', 'e2_bar', 'x_bar', 'y_bar', 'mass_bar']

            nfw_cols = ['Rs', 'alpha_Rs', 'x_nfw', 'y_nfw', 'e1_nfw', 'e2_nfw', 'virial_mass_nfw']

            ll_cols = ['R_sersic_ll', 'n_sersic_ll', 'e1_ll', 'e2_ll', 'x_ll', 'y_ll', 'magnitude_ll']

            sl_cols = ['magnitude_sl', 'R_sersic_sl', 'n_sersic_sl', 'x_sl', 'y_sl', 'e1_sl', 'e2_sl']

            los        = input_kwargs.loc[:, los_cols]
            baryons    = input_kwargs.loc[:, bar_cols]
            halo       = input_kwargs.loc[:, nfw_cols]
            lens_light = input_kwargs.loc[:, ll_cols]
            source     = input_kwargs.loc[:, sl_cols]
            Einstein_radii = input_kwargs.loc[:, 'theta_E']

            baryons, halo, lens_light, source = util.rename_kwargs(baryons, halo, lens_light, source)

            chain = MCMC(self.settings, baryons, halo, los, lens_light, Einstein_radii, source, path)

        elif self.settings['MCMC'] == False:
            print('MCMC will not be run.')

        else:
            raise ValueError('MCMC must be True or False.')

        print('\nAnalysis complete and results saved at {}.'.format(path))

    def pathfinder(self):
        path = (Path(__file__).parent/'results/').resolve()
        return str(path)
