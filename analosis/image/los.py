'''
los kwargs
'''

import numpy as np
import pandas as pd
from analosis.utilities.useful_functions import Utilities


class LOS:

    def __init__(self):
        rings_for_the_elven_kings = 3

    def kwargs(self, number_of_images, max_gamma):

        util = Utilities()

        kwargs_los_dataframe = pd.DataFrame(columns = ['gamma1_os', 'gamma2_os', 'kappa_os', 'omega_os',
                                                       'gamma1_od', 'gamma2_od', 'kappa_od', 'omega_od',
                                                       'gamma1_ds', 'gamma2_ds', 'kappa_ds', 'omega_ds',
                                                       'gamma1_los', 'gamma2_los', 'kappa_los', 'omega_los'])

        kwargs_los_dataframe['gamma1_os']  = [util.gamma(max_gamma, 'one') for i in range(number_of_images)]
        kwargs_los_dataframe['gamma2_os']  = [util.gamma(max_gamma, 'two') for i in range(number_of_images)]
        kwargs_los_dataframe['kappa_os']   = [0.0]*number_of_images
        kwargs_los_dataframe['omega_os']   = [0.0]*number_of_images

        kwargs_los_dataframe['gamma1_od']  = [util.gamma(max_gamma, 'one') for i in range(number_of_images)]
        kwargs_los_dataframe['gamma2_od']  = [util.gamma(max_gamma, 'two') for i in range(number_of_images)]
        kwargs_los_dataframe['kappa_od']   = [0.0]*number_of_images
        kwargs_los_dataframe['omega_od']   = [0.0]*number_of_images

        kwargs_los_dataframe['gamma1_ds']  = [util.gamma(max_gamma, 'one') for i in range(number_of_images)]
        kwargs_los_dataframe['gamma2_ds']  = [util.gamma(max_gamma, 'two') for i in range(number_of_images)]
        kwargs_los_dataframe['kappa_ds']   = [0.0]*number_of_images
        kwargs_los_dataframe['omega_ds']   = [0.0]*number_of_images

        kwargs_los_dataframe['gamma1_los']  = kwargs_los_dataframe['gamma1_os'] + kwargs_los_dataframe['gamma1_od'] - kwargs_los_dataframe['gamma1_ds']
        kwargs_los_dataframe['gamma2_los']  = kwargs_los_dataframe['gamma2_os'] + kwargs_los_dataframe['gamma2_od'] - kwargs_los_dataframe['gamma2_ds']
        kwargs_los_dataframe['kappa_los']   = [0.0]*number_of_images
        kwargs_los_dataframe['omega_los']   = [util.compute_omega_LOS(kwargs_los_dataframe['gamma1_od'][i],
                                                                 kwargs_los_dataframe['gamma2_od'][i],
                                                                 kwargs_los_dataframe['gamma1_os'][i],
                                                                 kwargs_los_dataframe['gamma2_os'][i],
                                                                 kwargs_los_dataframe['gamma1_ds'][i],
                                                                 kwargs_los_dataframe['gamma2_ds'][i]) for i in range(number_of_images)]

        return kwargs_los_dataframe
