import numpy as np
import pandas as pd
import copy

class Baryons():
    """
    This class randomly draws and handles the properties of the baryonic
    component of the main lens.
    """

    def __init__(self,
                 redshifts,
                 distances, # in Mpc
                 util,
                 model_mass='SERSIC_ELLIPSE_POTENTIAL',
                 model_light='SERSIC_ELLIPSE',
                 amplitude_reference=100, # amplitude for a source with mean mass at z=1
                 ):
        """
        Create the baryonic component of the main lens with Sérsic profile.
        This could later be generalised to other profiles.
        """

        self.model_mass = model_mass
        self.model_light = model_light
        self.kwargs = {}

        # Define the kwargs

        # mass and size
        # orders of magnitude freely inspired from https://arxiv.org/abs/1904.10992
        mean_mass = 1e11 # mean total baryonic mass [solar masses]
        # yes, we consider quite large masses here
        self.mass = np.random.lognormal(np.log(mean_mass), np.log(2)/2)
        # this ensures that 95% of the events have a mass that is at most
        # a factor two larger or smaller than the mean mass.
        R_sersic = (self.mass / mean_mass) * 2e-3 # Sérsic half-light radius [Mpc]
        R_sersic /= distances['od'] # [rad]
        R_sersic = util.angle_conversion(R_sersic, 'to arcsecs') # [arcsec]

        # Sérsic index
        mean_sersic_index = 4
        n_sersic = np.random.lognormal(np.log(mean_sersic_index), np.log(1.5)/2)

        # ellipticity
        e1 = np.random.normal(0, 0.2)
        e2 = np.random.normal(0, 0.2)

        # convergence at half-light radius
        effective_convergence = util.get_effective_convergence(
            mass=self.mass, R_sersic=R_sersic, n_sersic=n_sersic, e1=e1, e2=e2,
            d_os=distances['os'], d_od=distances['od'], d_ds=distances['ds'])

        # amplitude of the lens light: taken to be proportional to the mass
        amplitude = (amplitude_reference
                     * self.mass / mean_mass
                     * (2 / (1 + redshifts['lens']))**4)

        # Save the kwargs as attributes
        self.kwargs['R_sersic'] = R_sersic
        self.kwargs['n_sersic'] = n_sersic
        self.kwargs['k_eff'] = effective_convergence
        self.kwargs['e1'] = e1
        self.kwargs['e2'] = e2
        self.kwargs['x'] = 0
        self.kwargs['y'] = 0
        self.kwargs['amp'] = amplitude

    def return_kwargs(self, data_type):
        if data_type == 'mass':
            bk = copy.copy(self.kwargs)
            bk.pop('amp')
            kwargs = {k + '_bar': v for k, v in bk.items()}
        elif data_type == 'light':
            lk = copy.copy(self.kwargs)
            lk.pop('k_eff')
            kwargs = {k + '_ll': v for k, v in lk.items()}
        else:
            print('bad data type')
        return kwargs


    def make_dataframe(self, data_type):
        """
        Transforms the kwargs into a dataframe. Currently we assume that mass
        traces light exactly.
        """

        dataframe = pd.DataFrame()

        if data_type == 'mass':

            for key, value in self.kwargs.items():
                if key != 'amp':
                    key_mass = key + '_bar'
                    dataframe[key_mass] = [value]

        elif data_type == 'light':

            for key, value in self.kwargs.items():
                if key != 'k_eff':
                    key_light = key + '_ll'
                    dataframe[key_light] = [value]

        else:
            raise ValueError("Unknown data type.")

        return dataframe
