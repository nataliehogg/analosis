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

class Image:

    def __init__(self):
        rings_for_dwarf_lords = 7

    def generate_image(self, baryons, halo, los, lens_light, source, number_of_images, path):

        image_list = []
        data = []
        psf = []

        kwargs_los = los.to_dict('records')
        kwargs_bar = baryons.to_dict('records')
        kwargs_nfw = halo.to_dict('records')
        kwargs_sl  = source.to_dict('records')
        kwargs_ll  = lens_light.to_dict('records')

        # work out how to deal with this
        lens_model_list = ['LOS', 'SERSIC_ELLIPSE_POTENTIAL', 'NFW_ELLIPSE']

        for i in range(number_of_images):

            # telescope settings (HST)
            kwargs_data = sim_util.data_configure_simple(50, 0.08, 5400, 0.005)
            data_class = ImageData(**kwargs_data)
            kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': 0.15, 'pixel_size': 0.08, 'truncation': 3}
            psf_class = PSF(**kwargs_psf)
            kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}

            imageModel = ImageModel(data_class, psf_class,
                                    lens_model_class = LensModel(lens_model_list),
                                    source_model_class = LightModel(['SERSIC_ELLIPSE']),
                                    lens_light_model_class = LightModel(['SERSIC_ELLIPSE']),
                                    kwargs_numerics = kwargs_numerics)

            image_model = imageModel.image([kwargs_los[i],
                                           kwargs_bar[i],
                                           kwargs_nfw[i]],
                                           [kwargs_sl[i]],
                                           [kwargs_ll[i]],
                                           kwargs_ps = None)

            poisson = image_util.add_poisson(image_model, exp_time = 5400)
            bkg = image_util.add_background(image_model, sigma_bkd = 0.005)
            image_real = image_model + poisson + bkg

            data_class.update_data(image_real)
            kwargs_data['image_data'] = image_real

            image_list.append(image_real)

            # save the image data (list of arrays) to file for plotting
            filename = str(path) +'/datasets/image_list.pickle'
            outfile = open(filename,'wb')
            pickle.dump(image_list, outfile)
            outfile.close()

        return kwargs_data, kwargs_psf, kwargs_numerics
