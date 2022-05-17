"""
everything gets run from here
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology as colcos
from colossus.lss import mass_function
import random as r
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.Util.simulation_util as sim_util
import lenstronomy.Util.image_util as image_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.LensModel.Profiles.sersic_utils import SersicUtil
sersic_util = SersicUtil()
import pickle

from analosis.image.composite_lens.composite_lens_generator import CompositeLens
from analosis.image.distributed_haloes.distributed_haloes_generator import DistributedHaloes

class Run:

    def __init__(self, cpars, settings, parameters):

        print('Running the {} case with the following settings:\n\nModel: {}\nNumber of runs: {}'
              .format(settings['scenario'], settings['complexity'], settings['number_of_images']))

        colcos.setCosmology(cpars['id'])
        cosmo = FlatLambdaCDM(H0 = cpars['H0'], Om0 = cpars['Om'])
        lens_cosmo = LensCosmo(z_lens = cpars['z_lens'], z_source = cpars['z_source'], cosmo = cosmo)

        path = (Path(__file__).parent/'results/').resolve()

        if settings['scenario'] == 'composite lens':
            self.result = CompositeLens(cpars, cosmo, lens_cosmo, settings, parameters, path)
        elif settings['scenario'] == 'distributed haloes':
            self.result = DistributedHaloes(cpars, cosmo, lens_cosmo, settings, parameters, path)
        else:
            print('Scenario options are `composite lens` or `distributed haloes`.')

        print('\nAnalysis complete and results saved at {}.'.format(path))
