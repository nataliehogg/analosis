import numpy as np
import pandas as pd

from analosis.utilities.useful_functions import Utilities

class CompositeMainLens():

    def __init__(self):
        rings_for_mortal_men = 9

    def kwargs(self, number_of_images, d_os, d_od, d_ds):

        util = Utilities()

        kwargs_main_dataframe = pd.DataFrame(columns = ['k_eff_ml', 'R_sersic_ml', 'n_sersic_ml', 'x_ml', 'y_ml', 'e1_ml', 'e2_ml'])
        kwargs_ll_dataframe = pd.DataFrame(columns = ['k_eff_ll', 'R_sersic_ll', 'n_sersic_ll', 'x_ll', 'y_ll', 'e1_ll', 'e2_ll'])


        # get the main lens kwargs
        # the shape and position of the lens light matches that of the lens itself

        mass_main_lens = np.random.normal(5e10, 1e10, number_of_images) # [MSun] # idk how to use the lognormal distribution properly here
        R_sersic_phys = [2.0]*number_of_images # random here too?
        R_sersic_rad = R_sersic_phys/(d_od*1000) # d_od is in Mpc -> multiply by 1000 to get kpc
        R_sersic = util.angle_conversion(R_sersic_rad, 'to arcsecs')

        # note that we impose that the lens light parameters are identical to the lens params themselves now we have switched profiles
        kwargs_main_dataframe['R_sersic_ml'] = kwargs_ll_dataframe['R_sersic_ll'] = R_sersic
        kwargs_main_dataframe['n_sersic_ml'] = kwargs_ll_dataframe['n_sersic_ll'] = [2.0]*number_of_images # randomise?
        kwargs_main_dataframe['x_ml']        = kwargs_ll_dataframe['x_ll']        = [0.0]*number_of_images
        kwargs_main_dataframe['y_ml']        = kwargs_ll_dataframe['y_ll']        = [0.0]*number_of_images
        kwargs_main_dataframe['e1_ml']       = kwargs_ll_dataframe['e1_ll']       = np.random.normal(0.0, 0.1, number_of_images)
        kwargs_main_dataframe['e2_ml']       = kwargs_ll_dataframe['e2_ll']       = np.random.normal(0.0, 0.1, number_of_images)
        kwargs_main_dataframe['k_eff_ml']    = kwargs_ll_dataframe['k_eff_ll']    = [util.main_lens_convergence(mass_main_lens[i],
                                                                                                                  kwargs_main_dataframe['R_sersic_ml'][i],
                                                                                                                  kwargs_main_dataframe['n_sersic_ml'][i],
                                                                                                                  kwargs_main_dataframe['e1_ml'][i],
                                                                                                                  kwargs_main_dataframe['e2_ml'][i],
                                                                                                                  d_os, d_od, d_ds) for i in range(len(mass_main_lens))]

        return kwargs_main_dataframe, kwargs_ll_dataframe
