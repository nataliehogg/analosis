import numpy as np
import pandas as pd
from lenstronomy.Cosmo.lens_cosmo import LensCosmo

class Halo():
    """
    This class randomly draws and handles the properties of the dark matter
    component of the main lens.
    """

    def __init__(self,
                 mass_baryons, # in solar masses
                 redshifts,
                 distances,
                 util,
                 min_aspect_ratio_nfw = 0.9,
                 sigma_offset=300, # in pc
                 model_mass='NFW_ELLIPSE',
                 ):
        """
        Creates the dark-matter component of the main lens with NFW profile.
        This could later be generalised to other profiles.
        """

        self.model_mass = model_mass
        self.kwargs = {}
        self.lens_cosmo = LensCosmo(z_lens=redshifts['lens'],
                                    z_source=redshifts['source'],
                                    cosmo=util.cosmo)

        # Determine the parameters

        ratio_DM_baryons = np.random.uniform(low=30, high=100)
        # Reasonable but could be more realistic
        # Freely inspired from:
        # https://www.aanda.org/articles/aa/pdf/2020/02/aa36329-19.pdf
        self.virial_mass = mass_baryons * ratio_DM_baryons

        # concentration freely inspired from https://arxiv.org/pdf/1312.0945.pdf
        mean_concentration = 10**0.9
        concentration = np.random.lognormal(np.log(mean_concentration), 0.1)

        # scale radius and displacement angle [arcsec]
        Rs, alpha_Rs = self.lens_cosmo.nfw_physical2angle(M=self.virial_mass,
                                                          c=concentration)

        # ellipticity
        orientation_angle = np.random.uniform(0.0, 2*np.pi)
        aspect_ratio      = np.random.uniform(min_aspect_ratio_nfw, 1.0)
        e1, e2    = util.ellipticity(orientation_angle, aspect_ratio)

        # offset with respect to light centre (careful with pc -> arcsec)
        d_od = distances['od'] # in Mpc
        offset = {'x': 0, 'y': 0}
        for key in offset:
            component = np.random.normal(0.0, sigma_offset) # pc
            component *= 1e-6 # Mpc
            component /= d_od # rad
            component = util.angle_conversion(component, 'to arcsecs') # arcsec
            offset[key] = component

        # Save kwargs
        self.kwargs['Rs'] = Rs
        self.kwargs['alpha_Rs'] = alpha_Rs
        self.kwargs['x_nfw'] = offset['x']
        self.kwargs['y_nfw'] = offset['y']
        self.kwargs['e1_nfw'] = e1
        self.kwargs['e2_nfw'] = e2

    def make_dataframe(self):
        """
        Transforms the kwargs into a dataframe.
        """

        dataframe = pd.DataFrame()

        for key, value in self.kwargs.items():
            dataframe[key] = [value]

        return dataframe
