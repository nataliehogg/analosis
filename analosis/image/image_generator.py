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
        data = []
        psf = []

        kwargs_lens_light = []
        kwargs_source = []

        # convert the dfs to dicts for use with lenstronomy
        kwargs_los = los.to_dict('records')
        kwargs_bar = baryons.to_dict('records')
        kwargs_nfw = halo.to_dict('records')
        kwargs_sl  = source.to_dict('records')
        kwargs_ll  = lens_light.to_dict('records')

        # work out how to deal with this
        lens_model_list = ['LOS', 'SERSIC_ELLIPSE_POTENTIAL', 'NFW_ELLIPSE']
        lens_light_model_list = ['SERSIC_ELLIPSE']
        source_model_list = ['SERSIC_ELLIPSE']

        for i in range(settings['number_of_images']):

            # # tune the size of the image
            # theta_E = Einstein_radii[i]
            # beta = np.sqrt(kwargs_sl[i]['center_x']**2
            #                + kwargs_sl[i]['center_y']**2) # source offset
            # Rs = kwargs_sl[i]['R_sersic']# source half-light radius
            # size_image = max(2 * (theta_E + 5 * Rs), 2 * (beta + 5 * Rs)) # in arcsec
            # numPix = int(size_image / deltaPix) # total number of pixels is numPix**2

            numPix = 50
            psf = 'GAUSSIAN'
            background_rms = 0.01 # background noise rms # NH: where did this come from?

            kwargs_model = {'lens_model_list': lens_model_list,
                            'lens_light_model_list': lens_light_model_list,
                            'source_light_model_list': source_model_list}

            kwargs_numerics = {'supersampling_factor': 1, 'supersampling_convolution': False}

            # use the SimulationAPI module of lenstronomy to get the HST settings
            WFC3_F160W = HST(band = 'WFC3_F160W', psf_type = psf)
            kwargs_WFC3_F160W = WFC3_F160W.kwargs_single_band()
            sim = SimAPI(numpix = numPix, kwargs_single_band = kwargs_WFC3_F160W, kwargs_model = kwargs_model)
            imSim = sim.image_model_class(kwargs_numerics)

            # convert apparent magnitudes to amplitudes
            # this also necessarily returns the point source kwargs, though we don't need them...
            # also the mag2amp function doesn't take lists of dicts so we necessarily have to iterate
            ll, sl, ps = sim.magnitude2amplitude([kwargs_ll[i]], [kwargs_sl[i]])

            # ll and sl look like this [{}]
            # but if we append that directly to the kwargs lists we get [[{}, {}, ..., {}]]
            # when we want [{}, {}, ..., {}]
            # so ll[0] yields {} from ll = [{}]
            # I am aware this is disgusting but it's because mag2amp doesn't take lists as arguments
            # a far future goal would be to rewrite lots of lenstronomy functions so that they do
            # maybe there's a better way that I'm not aware of also
            kwargs_lens_light.append(ll[0])
            kwargs_source.append(sl[0])

            # telescope settings (HST)
            deltaPix = kwargs_WFC3_F160W['pixel_scale'] # size of a pixel in arcsec
            exp_time = kwargs_WFC3_F160W['exposure_time'] # exposition time in sec

            # copying the comment from HST.py here:
            # the fwhm is set equal to the approx pixel size for drizzled PSF.
            # Note that undrizzled PSF FWHM ~ 0.15" (Windhorst et al 2011)
            fwhm = kwargs_WFC3_F160W['seeing'] # width of the PSF in pixels

            kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exp_time, background_rms)
            data_class = ImageData(**kwargs_data)
            kwargs_psf = {'psf_type': psf, 'fwhm': fwhm, 'pixel_size': deltaPix, 'truncation': 3}
            psf_class = PSF(**kwargs_psf)

            if settings['lens_light'] == True:
                imageModel = ImageModel(data_class, psf_class,
                                        lens_model_class = LensModel(lens_model_list),
                                        source_model_class = LightModel(source_model_list),
                                        lens_light_model_class = LightModel(lens_light_model_list),
                                        kwargs_numerics = kwargs_numerics)

                image_model = imageModel.image([kwargs_los[i],
                                               kwargs_bar[i],
                                               kwargs_nfw[i]],
                                               [kwargs_source[i]],
                                               [kwargs_lens_light[i]],
                                               kwargs_ps = None)

            elif settings['lens_light'] == False:
                imageModel = ImageModel(data_class, psf_class,
                                        lens_model_class = LensModel(lens_model_list),
                                        source_model_class = LightModel(source_model_list),
                                        kwargs_numerics = kwargs_numerics)

                image_model = imageModel.image([kwargs_los[i],
                                               kwargs_bar[i],
                                               kwargs_nfw[i]],
                                               [kwargs_source[i]],
                                               kwargs_ps = None)
            else:
                print('You need to select lens light True or False.')

            poisson = image_util.add_poisson(image_model, exp_time=exp_time)
            bkg = image_util.add_background(image_model, sigma_bkd=background_rms)
            image_real = image_model + poisson + bkg

            data_class.update_data(image_real)
            kwargs_data['image_data'] = image_real

            print('kwargs_data in image gen', kwargs_data)

            image_list.append(image_real)
            kwargs_data_list.append(kwargs_data)

            # save the image data (list of arrays) to file for plotting
            filename = str(path) +'/datasets/' + str(settings['job_name']) + '_image_list.pickle'
            outfile = open(filename,'wb')
            pickle.dump(image_list, outfile)
            outfile.close()

        return kwargs_data_list, kwargs_psf, kwargs_numerics
