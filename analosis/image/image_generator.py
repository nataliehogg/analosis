import numpy as np
import pandas as pd
import pickle
import random

from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.SimulationAPI.ObservationConfig.HST import HST

class Image:

    def __init__(self):
        rings_for_dwarf_lords = 7

    def generate_image(self, image_settings,
                       baryons, halo, los, lens_light, source,
                       Einstein_radii, path
                       ):

        image_list = []
        hyper_list = []

        # convert the dfs to dicts for use with lenstronomy
        kwargs_los = los.to_dict('records')
        kwargs_bar = baryons.to_dict('records')
        kwargs_nfw = halo.to_dict('records')
        kwargs_sl  = source.to_dict('records')
        kwargs_ll  = lens_light.to_dict('records')

        # work out how to deal with this
        lens_model_list = ['LOS', 'SERSIC_ELLIPSE_POTENTIAL', 'NFW_ELLIPSE']
        lens_light_model_list = ['SERSIC_ELLIPSE']

        # source and its potential perturbations
        # source_model_list = ['SERSIC_ELLIPSE']

        max_source_perturbations = image_settings['max_source_perturbations']

        if max_source_perturbations == 0:
            print('Adding no perturbations to the source.')
        elif max_source_perturbations == 1:
            print('Adding 1 source perturbation per image.')
        elif max_source_perturbations > 1:
            print('Adding between 1 and {} source perturbations per image.'.format(max_source_perturbations))
        else:
            raise Warning("image_settings['max_source_perturbations'] should be an integer.")

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


        for i in range(image_settings['number_of_images']):

            # define kwargs for the lens
            kwargs_lens = [kwargs_los[i], kwargs_bar[i], kwargs_nfw[i]]

            # define kwargs for the main source
            kwargs_source = [kwargs_sl[i]]

            # add perturbations to the source
            mag = kwargs_sl[i]['magnitude']
            R_s = kwargs_sl[i]['R_sersic']
            x_s = kwargs_sl[i]['center_x']
            y_s = kwargs_sl[i]['center_y']

            if max_source_perturbations == 0:
                 source_model_list = ['SERSIC_ELLIPSE']
            else:
                source_perturbations_list = list(np.random.uniform(0.0, 0.06, random.randint(1, max_source_perturbations)))
                source_model_list = ['SERSIC_ELLIPSE'] + ['SERSIC']*len(source_perturbations_list)
                for pert in source_perturbations_list:
                    mag_pert = mag - 2.5 * np.log10(pert)
                    r2       = np.random.uniform(0, R_s**2)
                    r        = np.sqrt(r2)
                    phi      = np.random.uniform(0, 2*np.pi)
                    x        = x_s + r * np.cos(phi)
                    y        = y_s + r * np.sin(phi)
                    R        = np.random.uniform(R_s/2, R_s) 
                    n        = np.random.uniform(2, 6)
                    kwargs_pert = {'magnitude': mag_pert,
                               'R_sersic' : R,
                               'n_sersic' : n,
                               'center_x' : x,
                               'center_y' : y
                               }
                    kwargs_source.append(kwargs_pert)

            if image_settings['lens_light'] == True:
                kwargs_model = {'lens_model_list': lens_model_list, 
                                'lens_light_model_list': lens_light_model_list,
                                'source_light_model_list': source_model_list}
            else:
                kwargs_model = {'lens_model_list': lens_model_list,
                                'source_light_model_list': source_model_list}

            # always having lens light now
            kwargs_lens_light = [kwargs_ll[i]]


            if image_settings['fixed_numpix'] == True:
                numpix = 100
            elif image_settings['fixed_numpix'] == False:
                # compute the size of the image from the Einstein radius
                 theta_E = Einstein_radii[i] # in arcsec
                 beta = np.sqrt(kwargs_sl[i]['center_x']**2 + kwargs_sl[i]['center_y']**2) # source offset
                 image_size = 4 * (theta_E + beta)
                 numpix = int(image_size / pixel_size)
            else:
                raise Warning("image_settings['fixed_numpix'] should be True or False.")

            # simulation API
            sim = SimAPI(numpix=numpix,
                         kwargs_single_band=kwargs_band,
                         kwargs_model=kwargs_model)

            # extract data kwargs
            kwargs_data = sim.kwargs_data

            # convert magnitudes into amplitudes
            kwargs_lens_light, kwargs_source, ps = sim.magnitude2amplitude(kwargs_lens_light_mag=kwargs_lens_light,
                                                                           kwargs_source_mag=kwargs_source)

            # generate image
            imSim = sim.image_model_class(kwargs_numerics)
            image = imSim.image(kwargs_lens=kwargs_lens,
                                kwargs_source=kwargs_source,
                                kwargs_lens_light=kwargs_lens_light)
            # add noise
            image_noisy = image + sim.noise_for_model(model=image)

            # update kwargs_data with noisy image
            kwargs_data['image_data'] = image_noisy

            # append noisy image to list to be saved
            image_list.append(image_noisy)

            # append hyper-data to list to be saved
            hyper_list.append([kwargs_data, kwargs_psf, kwargs_numerics])

            # save the image data (list of arrays) to file for plotting
            image_filename = str(path)+'/datasets/'+str(image_settings['image_name'])+'_image_list.pickle'
            image_outfile = open(image_filename,'wb')
            pickle.dump(image_list, image_outfile)
            image_outfile.close()

        # saving the hyper-data for each image so we can MCMC any previously generated image
        hyper_filename = str(path)+'/datasets/'+str(image_settings['image_name'])+'_hyperdata.pickle'
        hyper_outfile = open(hyper_filename,'wb')
        pickle.dump(hyper_list, hyper_outfile)
        hyper_outfile.close()


        return None
