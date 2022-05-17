import numpy as np
import pandas as pd
from colossus.lss import mass_function
import random as r
from analosis.utilities.useful_functions import Utilities


class CompositeNFWHalo():

    def __init__(self):
        rings_for_elves = 3

    def kwargs(self, number_of_images, cpars, lens_cosmo):

        kwargs_nfw_dataframe = pd.DataFrame(columns = ['Rs', 'alpha_Rs', 'x_nfw', 'y_nfw', 'e1_nfw', 'e2_nfw'])

        # get the NFW halo kwargs
        # must be updated once Pierre is done
        mass_range = np.logspace(11, 15) # [MSun].
        N = mass_function.massFunction(mass_range, cpars['z_lens'], mdef = '200m', model = 'tinker08', q_out = 'dndlnM')
        virial_mass = r.choices(mass_range, N, k = number_of_images)

        concentration_parameter = 19.5 # should this also be random?

        kwargs_nfw_dataframe['Rs'], kwargs_nfw_dataframe['alpha_Rs'] = lens_cosmo.nfw_physical2angle(M = np.array(virial_mass),
                                                                                                     c = np.full((len(virial_mass),), concentration_parameter))

        kwargs_nfw_dataframe['x_nfw'] = np.random.normal(0.0, 0.16, number_of_images)
        kwargs_nfw_dataframe['y_nfw'] = np.random.normal(0.0, 0.16, number_of_images)
        kwargs_nfw_dataframe['e1_nfw'] = np.random.normal(0.0, 0.2, number_of_images)
        kwargs_nfw_dataframe['e2_nfw'] = np.random.normal(0.0, 0.2, number_of_images)

        return kwargs_nfw_dataframe
