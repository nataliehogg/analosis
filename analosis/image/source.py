'''
we use a common source for everything
'''
import numpy as np
import pandas as pd

class Source:

    def __init__(self):

        rings_for_mortal_men = 9

    def kwargs(self, number_of_images):

        kwargs_sl_dataframe = pd.DataFrame(columns = ['amp_sl','R_sersic_sl', 'n_sersic_sl', 'x_sl', 'y_sl', 'e1_sl', 'e2_sl'])

        # draw source position from a uniform disk of radius 4*R_s (source radius)
        R_sl = 0.03
        r2_sl = np.random.uniform(0, (6*R_sl)**2, number_of_images)
        r_sl = np.sqrt(r2_sl)
        phi_sl = np.random.uniform(0, 2*np.pi, number_of_images)
        x_sl = r_sl * np.cos(phi_sl)
        y_sl = r_sl * np.sin(phi_sl)

        amplitude_source = 100

        kwargs_sl_dataframe['amp_sl'] = [amplitude_source]*number_of_images
        kwargs_sl_dataframe['R_sersic_sl'] = [R_sl]*number_of_images
        kwargs_sl_dataframe['n_sersic_sl'] = [1.0]*number_of_images
        kwargs_sl_dataframe['x_sl'] = x_sl
        kwargs_sl_dataframe['y_sl'] = y_sl
        kwargs_sl_dataframe['e1_sl'] = np.random.normal(0.0, 0.2, number_of_images)
        kwargs_sl_dataframe['e2_sl'] = np.random.normal(0.0, 0.2, number_of_images)

        return kwargs_sl_dataframe
