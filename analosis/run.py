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
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Workflow.fitting_sequence import FittingSequence

# analosis
from analosis.utilities.useful_functions import Utilities
from analosis.image.mock_generator import Mocks
from analosis.image.image_generator import Image
from analosis.analysis.mcmc import MCMC
from analosis.analysis.split_mcmc import SplitMCMC


class Run:

    def __init__(self, cpars, image_settings, mcmc_settings):

        path = (Path(__file__).parent/'results/').resolve()

        colcos.setCosmology(cpars['id'])
        cosmo = FlatLambdaCDM(H0 = cpars['H0'], Om0 = cpars['Om'])
        util = Utilities(cosmo, path)
        self.image_settings = image_settings
        self.mcmc_settings = mcmc_settings

        self.mocks = Mocks(
            util=util,
            path=path,
            number_of_images=self.image_settings['number_of_images'],
            Einstein_radius_min=self.image_settings['Einstein_radius_min'],
            min_aspect_ratio_source=self.image_settings['min_aspect_ratio_source'],
            min_aspect_ratio_baryons=self.image_settings['min_aspect_ratio_baryons'],
            min_aspect_ratio_nfw=self.image_settings['min_aspect_ratio_nfw'],
            gamma_max=self.image_settings['maximum_shear'],
            sigma_halo_offset=self.image_settings['sigma_halo_offset'],
            maximum_source_offset_factor=self.image_settings['maximum_source_offset_factor'],
            telescope = self.image_settings['telescope'],
            band = self.image_settings['band'])

        if self.image_settings['generate_image'] == True:
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
            if image_settings['lens_light'] == True:
                complete_data = util.combine_dataframes([los, baryons, mass_bar_dataframe,
                                                         halo, mass_nfw_dataframe, source,
                                                         lens_light, Einstein_radii_dataframe])
            else:
                complete_data = util.combine_dataframes([los, baryons, mass_bar_dataframe,
                                                         halo, mass_nfw_dataframe, source,
                                                         Einstein_radii_dataframe])

            util.save_input_kwargs(self.image_settings, complete_data)

            # rename the dataframes for lenstronomy
            baryons, halo, lens_light, source = util.rename_kwargs(baryons, halo, lens_light, source)

            # generate the image and the associated data kwargs for either plotting or fitting
            im = Image()
            im.generate_image(self.image_settings, baryons, halo, los, lens_light, source, Einstein_radii, path)
        else:
            print('New images will not be generated.')
            pass

        if self.mcmc_settings['MCMC'] == True:

            if self.mcmc_settings['split'] == True:
                index = self.mcmc_settings['split_index']
                input_kwargs = pd.read_csv(str(path) + '/datasets/' + str(image_settings['image_name']) + '_input_kwargs_{}.csv'.format(index))
                print('using split kwargs index', index)
            else:
                input_kwargs = pd.read_csv(str(path) + '/datasets/' + str(image_settings['image_name']) + '_input_kwargs.csv')

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
            source     = input_kwargs.loc[:, sl_cols]
            Einstein_radii = input_kwargs.loc[:, 'theta_E']

            if set(ll_cols).issubset(input_kwargs.columns):
                lens_light = input_kwargs.loc[:, ll_cols]
            else:
                lens_light = None

            baryons, halo, lens_light, source = util.rename_kwargs(baryons, halo, lens_light, source)

            if self.mcmc_settings['split'] == True:
                chain = SplitMCMC(self.image_settings, self.mcmc_settings, baryons, halo, los, lens_light, Einstein_radii, source, path)
            else:
                chain = MCMC(self.image_settings, self.mcmc_settings, baryons, halo, los, lens_light, Einstein_radii, source, path)

        elif self.mcmc_settings['MCMC'] == False:
            print('MCMC will not be run.')

        else:
            raise ValueError('MCMC must be True or False.')

        print('\nAnalysis complete and results saved at {}.'.format(path))

    def pathfinder(self):
        path = (Path(__file__).parent/'results/').resolve()
        return str(path)
