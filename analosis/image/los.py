import numpy as np
import pandas as pd

class LOS:
    """
    This class randomly draws and handles the line-of-sight parameters.
    """

    def __init__(self, util, gamma_max=0.03, model='LOS'):
        """
        Creates the LOS structure, with upper limit for the value of the shears.
        """

        self.model = model
        self.kwargs = {}

        # standard parameters
        for interval in ['_os', '_od', '_ds', '_los']:
            for distortion in ['kappa', 'gamma', 'omega']:

                if interval != '_los':
                    if distortion == 'gamma':
                        gamma_sq = np.random.uniform(0, gamma_max**2)
                        gamma = np.sqrt(gamma_sq)
                        phi = np.random.uniform(0, np.pi)
                        self.kwargs['gamma1'+interval] = gamma * np.cos(phi)
                        self.kwargs['gamma2'+interval] = gamma * np.sin(phi)
                    else:
                        self.kwargs[distortion + interval] = 0

                else: # this syntax works because _los is last

                    if distortion == 'omega':
                        self.kwargs[distortion + interval] = util.compute_omega_LOS(self.kwargs)
                    else:
                        if distortion == 'gamma':
                            components = ['1', '2']
                        else:
                            components = ['']

                        for component in components:
                            self.kwargs[distortion + component + interval] = (
                                self.kwargs[distortion + component + '_od']
                                + self.kwargs[distortion + component + '_os']
                                - self.kwargs[distortion + component + '_ds'])

    def make_dataframe(self):
        """
        Transforms the kwargs into a dataframe.
        """

        dataframe = pd.DataFrame()

        for key, value in self.kwargs.items():
            dataframe[key] = [value]

        return dataframe
