import numpy as np
import pandas as pd

# import packages for each component
from analosis.image.source import Source
from analosis.image.los import LOS
from analosis.image.baryons import Baryons
from analosis.image.dark_matter import Halo

class Mocks:
    """
    This class handles the creation of a certain number of images by drawing
    randomly their parameters, it saves the parameters in a dataframe.
    """
    
    def __init__(self,
                 util,
                 scenario='composite lens',
                 path='',
                 number_of_images=1,
                 Einstein_radius_min=0.5,
                 gamma_max=0.03,
                 sigma_halo_offset=0.16,
                 maximum_source_offset_factor=2
                 ):
        
        self.util = util
        self.scenario = scenario
        self.path = path
        self.number_of_images = number_of_images
        self.Einstein_radius_min = Einstein_radius_min
        self.gamma_max = gamma_max
        self.sigma_halo_offset = sigma_halo_offset
        self.maximum_source_offset_factor = maximum_source_offset_factor
      
        
    def draw_kwargs(self):
        """
        Draws the parameters for the lensing system and returns them as the
        row of a dataframe.
        """
        
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
        
        
        if self.scenario == 'composite lens':
            
            # main lens
            Einstein_radius = 0
            attempt = 0
            while Einstein_radius < self.Einstein_radius_min:
                
                baryons = Baryons(redshifts, distances, self.util)
                halo = Halo(mass_baryons=baryons.mass,
                            redshifts=redshifts,
                            util=self.util,
                            sigma_offset=self.sigma_halo_offset)
                
                # estimate the Einstein radius in arcsec
                theta_E_bar = self.util.Einstein_radius_point_lens(
                    mass=baryons.mass, distances=distances)
                Einstein_radius = theta_E_bar + halo.kwargs['alpha_Rs']
                
                attempt += 1
                if attempt > 100:
                    raise RuntimeWarning("I seem to have difficulties to\
                                         reach the required Einstein radius.")
            
            data_baryons = baryons.make_dataframe(data_type='mass')
            data_lens_light = baryons.make_dataframe(data_type='light')
            data_halo = halo.make_dataframe()

            # LOS effects
            los = LOS(util=self.util, gamma_max=self.gamma_max)
            data_los = los.make_dataframe()
            
            # source
            source = Source(redshifts, distances, self.util,
                            maximum_source_offset_factor=self.maximum_source_offset_factor)
            data_source = source.make_dataframe()
            
            # make dataframe row
            data_row = pd.concat([data_baryons,
                                  data_lens_light,
                                  data_halo,
                                  data_source,
                                  data_los],
                                 axis=1)
            
        elif self.scenario == 'distributed haloes':
            raise ValueError("The distributed halo case is not implemented yet.")
        else:
            raise ValueError("Unknown scenario.")
            
        return data_row
            
       
    def get_kwargs(self):
        """
        This function makes all the images and gathers their parameters in
        a single dataframe.
        """
            
        data_rows = [self.draw_kwargs() for i in range(self.number_of_images)]
        self.dataframe = pd.concat(data_rows, axis=0)
        
        
    def save_dataframe(self):
        """
        Saves the dataframe on a file.
        """
        
        self.dataframe.to_csv(str(self.path) + '/datasets/input_kwargs.csv',
                              index = False)


    def show_images(self):
        raise ValueError("not implemented yet")
            
            
            
            
        
        
        
        
        
        