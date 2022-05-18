import numpy as np
import pandas as pd

from analosis.utilities.useful_functions import Utilities
from analosis.image.source import Source
from analosis.image.los import LOS
from analosis.image.composite_lens.composite_baryons import CompositeBaryons
from analosis.image.composite_lens.composite_nfw_halo import CompositeNFWHalo

class CompositeLens:

    def __init__(self):#, cpars, cosmo, lens_cosmo, settings, parameters, path):
        rings_for_mortal_men = 9

        # self.kwargs(cpars, cosmo, lens_cosmo, settings, parameters, path)

    def kwargs(self, cpars, cosmo, lens_cosmo, settings, parameters, path):

        '''
        set up the kwargs
        '''

        util = Utilities()
        source = Source()
        los = LOS()
        bar = CompositeBaryons()
        nfw = CompositeNFWHalo()

        d_od = util.dA(cosmo, cpars['z_observer'], cpars['z_lens'])
        d_os = util.dA(cosmo, cpars['z_observer'], cpars['z_source'])
        d_ds = util.dA(cosmo, cpars['z_lens'], cpars['z_source'])

        number_of_images = settings['number_of_images']
        max_gamma = parameters['maximum_shear']

        # get the dataframes: main lens (*_ml), lens light (*_ll), NFW halo (*_nfw), source light (*_sl)
        kwargs_sl_dataframe = source.kwargs(number_of_images)
        kwargs_los_dataframe = los.kwargs(number_of_images, max_gamma)
        # lens light kwargs are fixed to be the same as the main lens kwargs
        kwargs_bar_dataframe, kwargs_ll_dataframe = bar.kwargs(number_of_images, d_os, d_od, d_ds)
        kwargs_nfw_dataframe = nfw.kwargs(number_of_images, cpars, lens_cosmo)

        # check this part works as wanted
        amplitude_source = kwargs_sl_dataframe['amp_sl'][0] # provided every source is the same this is ok
        amplitude_lens = amplitude_source * (1 + cpars['z_source'])**4 / (1 + cpars['z_lens'])**4
        kwargs_ll_dataframe['amp_ll'] = [amplitude_lens]*number_of_images

        # combine and save final dataframe to file
        kwargs_dataframe = pd.concat([kwargs_bar_dataframe, kwargs_ll_dataframe,
                                      kwargs_nfw_dataframe, kwargs_sl_dataframe,
                                      kwargs_los_dataframe], axis=1)

        kwargs_dataframe.to_csv(str(path) + '/datasets/input_kwargs.csv', index = False)

        # rename the columns for lenstronomy

        kwargs_bar_dataframe = kwargs_bar_dataframe.rename(index = str, columns = {'k_eff_ml': 'k_eff',
                                                                                   'R_sersic_ml': 'R_sersic',
                                                                                   'n_sersic_ml': 'n_sersic',
                                                                                   'x_ml': 'center_x',
                                                                                   'y_ml': 'center_y',
                                                                                   'e1_ml': 'e1',
                                                                                   'e2_ml': 'e2'})

        kwargs_nfw_dataframe = kwargs_nfw_dataframe.rename(index = str, columns = {'x_nfw': 'center_x',
                                                                                   'y_nfw': 'center_y',
                                                                                   'e1_nfw': 'e1',
                                                                                   'e2_nfw': 'e2'})

        kwargs_ll_dataframe = kwargs_ll_dataframe.rename(index = str, columns = {'amp_ll': 'amp',
                                                                                 'R_sersic_ll': 'R_sersic',
                                                                                 'n_sersic_ll': 'n_sersic',
                                                                                 'x_ll': 'center_x',
                                                                                 'y_ll': 'center_y',
                                                                                 'e1_ll': 'e1',
                                                                                 'e2_ll': 'e2'})

        kwargs_sl_dataframe = kwargs_sl_dataframe.rename(index = str, columns = {'amp_sl': 'amp',
                                                                                 'R_sersic_sl': 'R_sersic',
                                                                                 'n_sersic_sl': 'n_sersic',
                                                                                 'x_sl': 'center_x',
                                                                                 'y_sl': 'center_y',
                                                                                 'e1_sl': 'e1',
                                                                                 'e2_sl': 'e2'})


        kwargs_dataframe_dict = {'source_kwargs': kwargs_sl_dataframe,
                                 'los_kwargs': kwargs_los_dataframe,
                                 'baryonic_kwargs': kwargs_bar_dataframe,
                                 'nfw_kwargs': kwargs_nfw_dataframe,
                                 'lens_light_kwargs': kwargs_ll_dataframe}


        return kwargs_dataframe_dict
