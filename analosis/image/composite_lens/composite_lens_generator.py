import numpy as np
import pandas as pd

from analosis.utilities.useful_functions import Utilities
from analosis.image.source import Source
from analosis.image.los import LOS
from analosis.image.composite_lens.composite_main_lens import CompositeMainLens
from analosis.image.composite_lens.composite_nfw_halo import CompositeNFWHalo

class CompositeLens:

    def __init__(self, cpars, cosmo, lens_cosmo, settings, parameters, path):
        rings_for_mortal_men = 9

        self.kwargs(cpars, cosmo, lens_cosmo, settings, parameters, path)

    def kwargs(self, cpars, cosmo, lens_cosmo, settings, parameters, path):

        '''
        set up the kwargs
        '''

        util = Utilities()
        source = Source()
        los = LOS()
        main = CompositeMainLens()
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
        kwargs_main_dataframe, kwargs_ll_dataframe = main.kwargs(number_of_images, d_os, d_od, d_ds)
        kwargs_nfw_dataframe = nfw.kwargs(number_of_images, cpars, lens_cosmo)

        # check this part works as wanted
        amplitude_source = kwargs_sl_dataframe['amp_sl'][0] # provided every source is the same this is ok
        amplitude_lens = amplitude_source * (1 + cpars['z_source'])**4 / (1 + cpars['z_lens'])**4
        kwargs_ll_dataframe['amp_ll'] = [amplitude_lens]*number_of_images


        # combine and save final dataframe to file
        kwargs_dataframe = pd.concat([kwargs_main_dataframe, kwargs_ll_dataframe,
                                      kwargs_nfw_dataframe, kwargs_sl_dataframe,
                                      kwargs_los_dataframe], axis=1)

        kwargs_dataframe.to_csv(str(path) + '/datasets/input_kwargs.csv', index = False)
