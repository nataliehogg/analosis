import numpy as np
import pandas as pd
import pickle

from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.Util.simulation_util as sim_util
import lenstronomy.Util.image_util as image_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.SimulationAPI.ObservationConfig.HST import HST

class Image:

    def __init__(self):
        rings_for_dwarf_lords = 7

    def generate_image(self, settings, baryons, halo, los, lens_light, source, Einstein_radii, path):

        image_list = []
        kwargs_data_list = []

        # convert the dfs to dicts for use with lenstronomy
        kwargs_los = los.to_dict('records')
        kwargs_bar = baryons.to_dict('records')
        kwargs_nfw = halo.to_dict('records')
        kwargs_sl  = source.to_dict('records')
        kwargs_ll  = lens_light.to_dict('records')

        # work out how to deal with this
        lens_model_list = ['LOS', 'SERSIC_ELLIPSE_POTENTIAL', 'NFW_ELLIPSE']
        if settings['lens_light']:
            lens_light_model_list = ['SERSIC_ELLIPSE']
        else:
            lens_light_model_list = []
        source_model_list = ['SERSIC_ELLIPSE']

        # lens model
        kwargs_model = {'lens_model_list': lens_model_list,
                        'lens_light_model_list': lens_light_model_list,
                        'source_light_model_list': source_model_list}

        # telescope settings (HST)
        psf = 'GAUSSIAN'
        band = HST(band='WFC3_F160W', psf_type=psf)
        kwargs_band = band.kwargs_single_band()
        pixel_size = band.camera['pixel_scale'] # in arcsec
        kwargs_psf = {'psf_type': psf,
                      'fwhm': kwargs_band['seeing'],
                      'pixel_size': pixel_size,
                      'truncation': 3}

        # numerics
        kwargs_numerics = {'supersampling_factor': 1,
                           'supersampling_convolution': False}


        for i in range(settings['number_of_images']):

            # define kwargs for the lens, source, image
            kwargs_lens = [kwargs_los[i], kwargs_bar[i], kwargs_nfw[i]]
            kwargs_source = [kwargs_sl[i]]
            if settings['lens_light']:
                kwargs_lens_light = [kwargs_ll[i]]
            else:
                kwargs_lens_light = None

            # compute the size of the image from the Einstein radius
            theta_E = Einstein_radii[i] # in arcsec
            beta = np.sqrt(kwargs_sl[i]['center_x']**2
                           + kwargs_sl[i]['center_y']**2) # source offset
            image_size = 3 * (theta_E + beta)
            numpix = int(image_size / pixel_size)

            # simulation API
            sim = SimAPI(numpix=numpix,
                         kwargs_single_band=kwargs_band,
                         kwargs_model=kwargs_model)

            # convert magnitudes into amplitudes
            kwargs_lens_light, kwargs_source, ps = sim.magnitude2amplitude(
                kwargs_lens_light_mag=kwargs_lens_light,
                kwargs_source_mag=kwargs_source)

            # generate image with noise
            imSim = sim.image_model_class(kwargs_numerics)
            image = imSim.image(kwargs_lens=kwargs_lens,
                                kwargs_source=kwargs_source,
                                kwargs_lens_light=kwargs_lens_light)
            image += sim.noise_for_model(model=image)
            image_list.append(image)

            # save the image data (list of arrays) to file for plotting
            filename = str(path)+'/datasets/'+str(settings['job_name'])+'_image_list_'+str(settings['starting_index'])+'.pickle'
            outfile = open(filename,'wb')
            pickle.dump(image_list, outfile)
            outfile.close()

            # extract data kwargs
            kwargs_data = sim.kwargs_data
            kwargs_data_list.append(kwargs_data)

        return kwargs_data_list, kwargs_psf, kwargs_numerics
