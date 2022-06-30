import numpy as np
import pandas as pd

# import packages for each component
from analosis.image.source import Source
from analosis.image.los import LOS
from analosis.image.baryons import Baryons
from analosis.image.dark_matter import Halo
from lenstronomy.LensModel.lens_model import LensModel

class Mocks:
    """
    This class handles the creation of a certain number of images, and allows
    to draw their parameters randomly.
    """

    def __init__(self,
                 util,
                 # scenario='composite lens',
                 path='',
                 number_of_images=1,
                 Einstein_radius_min=0.5, # arcsec
                 min_aspect_ratio_source = 0.9,
                 min_aspect_ratio_baryons = 0.9,
                 min_aspect_ratio_nfw = 0.9,
                 gamma_max=0.03,
                 sigma_halo_offset=300, # pc
                 maximum_source_offset_factor=1, # in units of source half-light radius
                 ):

        self.util = util
        # self.scenario = scenario
        self.path = path
        self.number_of_images = number_of_images
        self.Einstein_radius_min = Einstein_radius_min
        self.min_aspect_ratio_source = min_aspect_ratio_source
        self.min_aspect_ratio_baryons = min_aspect_ratio_baryons
        self.min_aspect_ratio_nfw = min_aspect_ratio_nfw
        self.Einstein_radii = []
        self.masses_baryons = []
        self.masses_haloes  = []
        self.gamma_max = gamma_max
        self.sigma_halo_offset = sigma_halo_offset
        self.maximum_source_offset_factor = maximum_source_offset_factor


    def draw_kwargs(self):
        """
        Draws the parameters for the lensing system and returns them as the
        row of a dataframe.
        """

        # if self.scenario == 'composite lens':

        kwargs = {'baryons': [], 'halo':[], 'los':[], 'lens_light':[], 'source': []}

        for i in range(self.number_of_images):

            # redshifts
            redshifts = {}
            redshifts['lens'] = np.random.uniform(low=0.4, high=0.6)
            redshifts['source'] = np.random.uniform(low=1.5, high=2.5)
            # TODO: save the redshifts in dataframe?

            # distances
            distances = {}
            distances['os'] = self.util.dA(0, redshifts['source'])
            distances['od'] = self.util.dA(0, redshifts['lens'])
            distances['ds'] = self.util.dA(redshifts['lens'], redshifts['source'])

            # main lens
            Einstein_radius = 0
            attempt = 0
            while Einstein_radius < self.Einstein_radius_min:

                baryons = Baryons(redshifts, distances, self.util,
                                  min_aspect_ratio_baryons=self.min_aspect_ratio_baryons)
                halo = Halo(mass_baryons=baryons.mass,
                            redshifts=redshifts,
                            distances=distances,
                            util=self.util,
                            min_aspect_ratio_nfw=self.min_aspect_ratio_nfw,
                            sigma_offset=self.sigma_halo_offset)

                # Estimate the Einstein radius in arcsec
                # start with the baryons as if they were a point lens
                theta_E_bar = self.util.Einstein_radius_point_lens(
                    mass=baryons.mass,
                    distances=distances)
                # displacement angle of NFW at that position
                spherical_halo_class = LensModel(lens_model_list=['NFW'])
                kwargs_spherical_halo = [{'Rs': halo.kwargs['Rs'],
                                          'alpha_Rs': halo.kwargs['alpha_Rs']}]
                alpha_x, alpha_y = spherical_halo_class.alpha(x=theta_E_bar,
                                                              y=0,
                                                              kwargs=kwargs_spherical_halo)
                # sum the contribution of baryons and NFW
                Einstein_radius = theta_E_bar + alpha_x

                attempt += 1
                if attempt > 100:
                    raise RuntimeWarning("I seem to have difficulties to\
                                         reach the required Einstein radius.")
            self.Einstein_radii.append(Einstein_radius)
            self.masses_baryons.append(baryons.mass)
            self.masses_haloes.append(halo.virial_mass)

            halo_kwargs = halo.kwargs #for i in range(self.number_of_images)]
            baryon_kwargs = baryons.return_kwargs(data_type='mass') #for i in range(self.number_of_images)]
            lens_light_kwargs = baryons.return_kwargs(data_type='light') #for i in range(self.number_of_images)]

            # centre of mass of the main lens
            lens_mass_centre = {}
            mass_tot = halo.virial_mass + baryons.mass
            lens_mass_centre['x'] = (halo.virial_mass * halo.kwargs['x_nfw']
                                     + baryons.mass * baryons.kwargs['x']) / mass_tot
            lens_mass_centre['y'] = (halo.virial_mass * halo.kwargs['y_nfw']
                                     + baryons.mass * baryons.kwargs['y']) / mass_tot

            # LOS effects
            los = LOS(util=self.util, gamma_max=self.gamma_max)
            los_kwargs = los.kwargs # for i in range(self.number_of_images)]

            # source
            source = Source(redshifts, distances, self.util,
                            maximum_source_offset_factor=self.maximum_source_offset_factor,
                            min_aspect_ratio_source = self.min_aspect_ratio_source,
                            Einstein_radius=Einstein_radius,
                            lens_mass_centre=lens_mass_centre)
            source_kwargs = source.kwargs

            kwargs['baryons'].append(baryon_kwargs)
            kwargs['halo'].append(halo_kwargs)
            kwargs['los'].append(los_kwargs)
            kwargs['lens_light'].append(lens_light_kwargs)
            kwargs['source'].append(source_kwargs)

        # elif self.scenario == 'distributed haloes':
        #     raise ValueError("The distributed halo case is not implemented yet.")
        # else:
        #     raise ValueError("Unknown scenario.")

        return kwargs
