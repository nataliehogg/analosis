import numpy as np
import pandas as pd

class Source:
    """
    This class randomly draws and handles the properties of the source.

    Perturbations to the source are added in the Image class.

    """

    def __init__(self,
                 redshifts,
                 distances,
                 util,
                 maximum_source_offset_factor=1,
                 min_aspect_ratio_source = 0.9,
                 telescope = 'JWST',
                 band = 'F115W',
                 index = 0,
                 Einstein_radius=1,
                 lens_mass_centre={'x':0, 'y':0},
                 model='SERSIC_ELLIPSE'
                 ):
      """
      Creates the source light with an elliptical Sérsic profile.
      This could later be generalised to other profiles.
      """

      self.model = model
      self.kwargs = {}

      # Define the parameters

      if telescope == 'JWST':
        # with JWST settings we ignore the max source offset factor, just passing in the catalogue source positions
        R_sersic, magnitude, x, y = util.source_from_catalogue(band, index)
      else:
        # half-light radius size
        # (freely inspired from https://arxiv.org/abs/1904.10992)
        mean_radius = 3e-3 #[Mpc] #PFmod
        radius = np.random.lognormal(np.log(mean_radius), np.log(2)/2)
        # this ensures that 95% of the events have a radius that is at most
        # a factor two larger or smaller than the mean radius.
        radius = max(radius, mean_radius/2) #PFmod: ensures large enough source
        R_sersic = radius / distances['os'] # [rad]
        R_sersic = util.angle_conversion(R_sersic, 'to arcsecs') # [arcsec]

        # absolute magnitude: assume that the luminosity is proportional to the
        # galaxy's area; for r = mean_radius we have M = reference_magnitude
        reference_magnitude = -22 #PFmod
        absolute_magnitude = reference_magnitude - 5 * np.log10(radius / mean_radius)

        # apparent magnitude
        D = (1 + redshifts['source'])**2 * distances['os'] # luminosity distance to s [Mpc]
        magnitude = absolute_magnitude + 5 * np.log10(D) + 25 # 25 = log10(Mpc/10pc)

        # position
        r_max = min(R_sersic, maximum_source_offset_factor * Einstein_radius) #PFmod
        r_sq_max = r_max**2 #[arcsec^2]
        r_sq = np.random.uniform(0, r_sq_max)
        r = np.sqrt(r_sq)
        phi = np.random.uniform(0, 2*np.pi)
        x = lens_mass_centre['x'] + r * np.cos(phi)
        y = lens_mass_centre['y'] + r * np.sin(phi)

      # Sérsic index
      mean_sersic_index = 4
      n_sersic = np.random.lognormal(np.log(mean_sersic_index), np.log(1.5)/2)

      # ellipticity
      orientation_angle = np.random.uniform(0.0, 2*np.pi)
      aspect_ratio      = np.random.uniform(min_aspect_ratio_source, 1.0)
      e1, e2    = util.ellipticity(orientation_angle, aspect_ratio)


      # Save kwargs
      self.kwargs['magnitude_sl'] = magnitude
      self.kwargs['R_sersic_sl'] = R_sersic
      self.kwargs['n_sersic_sl'] = n_sersic
      self.kwargs['x_sl'] = x
      self.kwargs['y_sl'] = y
      self.kwargs['e1_sl'] = e1
      self.kwargs['e2_sl'] = e2


    def make_dataframe(self):
        """
        Transforms the kwargs into a dataframe.
        """

        dataframe = pd.DataFrame()

        for key, value in self.kwargs.items():
            dataframe[key] = [value]

        return dataframe
