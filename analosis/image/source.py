import numpy as np
import pandas as pd

class Source:
    """
    This class randomly draws and handles the properties of source.
    """

    def __init__(self,
                 redshifts,
                 distances,
                 util,
                 maximum_source_offset_factor=1,
                 Einstein_radius=10,
                 lens_mass_centre={'x':0, 'y':0},
                 model='SERSIC_ELLIPSE',
                 amplitude_reference=100):
      """
      Creates the source light with Sérsic profile.
      This could later be generalised to other profiles.
      """

      self.model = model
      self.kwargs = {}

      # Define the parameters
      # half-light radius size
      # (freely inspired from https://arxiv.org/abs/1904.10992)
      mean_radius = 2e-3 #[Mpc]
      R_sersic = np.random.lognormal(np.log(mean_radius), np.log(2)/2)
      # this ensures that 95% of the events have a mass that is at most
      # a factor two larger or smaller than the mean mass.
      R_sersic /= distances['os'] # [rad]
      R_sersic = util.angle_conversion(R_sersic, 'to arcsecs') # [arcsec]

      # Sérsic index
      mean_sersic_index = 4
      n_sersic = np.random.lognormal(np.log(mean_sersic_index), np.log(1.5)/2)

      # position
      r_max = min(maximum_source_offset_factor * R_sersic, Einstein_radius)
      r_sq_max = r_max**2 #[arcsec^2]
      r_sq = np.random.uniform(0, r_sq_max)
      r = np.sqrt(r_sq)
      phi = np.random.uniform(0, 2*np.pi)
      x = lens_mass_centre['x'] + r * np.cos(phi)
      y = lens_mass_centre['y'] + r * np.sin(phi)

      # ellipticity
      e1 = np.random.normal(0, 0.2)
      e2 = np.random.normal(0, 0.2)

      # amplitude of light
      amplitude = amplitude_reference * (2 / (1 + redshifts['source']))**4

      # Save kwargs
      self.kwargs['amp_sl'] = amplitude
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
