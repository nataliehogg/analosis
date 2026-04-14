import numpy as np
import pandas as pd
import pickle
import random
from copy import deepcopy

from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.SimulationAPI.ObservationConfig.HST import HST
from lenstronomy.SimulationAPI.ObservationConfig.JWST import JWST

class Image:

    def __init__(self):
        rings_for_dwarf_lords = 7

    @staticmethod
    def _serialise_value(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def _source_component_rows(self, image_index, source_model_list, kwargs_source_mag, kwargs_source_amp):
        rows = []

        for component_index, model_name in enumerate(source_model_list):
            row = {
                'image_index': image_index,
                'component_index': component_index,
                'model_name': model_name,
                'is_main_source': component_index == 0,
            }

            kwargs_mag = kwargs_source_mag[component_index]
            kwargs_amp = kwargs_source_amp[component_index]

            for key, value in kwargs_mag.items():
                row[f'mag_{key}'] = self._serialise_value(value)
            for key, value in kwargs_amp.items():
                row[f'amp_{key}'] = self._serialise_value(value)

            rows.append(row)

        return rows

    @staticmethod
    def _ellipticity(phi, q):
        e = (1 - q) / (1 + q)
        return e * np.cos(2 * phi), e * np.sin(2 * phi)

    def _build_sersic_perturbation(self, mag, R_s, x_s, y_s):
        pert = np.random.uniform(0.0, 0.06)
        mag_pert = mag - 2.5 * np.log10(pert)
        r2 = np.random.uniform(0, R_s**2)
        r = np.sqrt(r2)
        phi = np.random.uniform(0, 2 * np.pi)
        x = x_s + r * np.cos(phi)
        y = y_s + r * np.sin(phi)
        R = np.random.uniform(R_s / 2, R_s)
        n = np.random.uniform(2, 6)
        kwargs_pert = {
            'magnitude': mag_pert,
            'R_sersic': R,
            'n_sersic': n,
            'center_x': x,
            'center_y': y,
        }
        return 'SERSIC', kwargs_pert

    def _build_gaussian_clump(self, image_settings, mag, R_s, x_s, y_s):
        flux_min = image_settings.get('gaussian_clump_flux_min', 0.03)
        flux_max = image_settings.get('gaussian_clump_flux_max', 0.20)
        radius_factor = image_settings.get('gaussian_clump_radius_factor', 1.5)
        sigma_min_factor = image_settings.get('gaussian_clump_sigma_min_factor', 0.06)
        sigma_max_factor = image_settings.get('gaussian_clump_sigma_max_factor', 0.25)
        min_aspect_ratio = image_settings.get('gaussian_clump_min_aspect_ratio', 0.35)

        flux_fraction = np.random.uniform(flux_min, flux_max)
        mag_pert = mag - 2.5 * np.log10(flux_fraction)

        r2 = np.random.uniform(0, (radius_factor * R_s) ** 2)
        r = np.sqrt(r2)
        phi = np.random.uniform(0, 2 * np.pi)
        x = x_s + r * np.cos(phi)
        y = y_s + r * np.sin(phi)

        sigma_min = max(0.005, sigma_min_factor * R_s)
        sigma_max = max(sigma_min * 1.1, sigma_max_factor * R_s)
        sigma = np.random.uniform(sigma_min, sigma_max)

        axis_ratio = np.random.uniform(min_aspect_ratio, 1.0)
        phi_ell = np.random.uniform(0, np.pi)
        e1, e2 = self._ellipticity(phi_ell, axis_ratio)

        kwargs_pert = {
            'magnitude': mag_pert,
            'sigma': sigma,
            'e1': e1,
            'e2': e2,
            'center_x': x,
            'center_y': y,
        }
        return 'GAUSSIAN_ELLIPSE', kwargs_pert

    def _build_source_components(self, image_settings, source_kwargs):
        source_model_list = ['SERSIC_ELLIPSE']
        kwargs_source = [deepcopy(source_kwargs)]

        max_source_perturbations = image_settings['max_source_perturbations']
        if max_source_perturbations == 0:
            return source_model_list, kwargs_source

        source_perturbation_model = image_settings.get('source_perturbation_model', 'sersic')
        mag = source_kwargs['magnitude']
        R_s = source_kwargs['R_sersic']
        x_s = source_kwargs['center_x']
        y_s = source_kwargs['center_y']

        number_of_components = random.randint(1, max_source_perturbations)
        for _ in range(number_of_components):
            if source_perturbation_model == 'sersic':
                model_name, kwargs_pert = self._build_sersic_perturbation(mag, R_s, x_s, y_s)
            elif source_perturbation_model == 'gaussian_clumps':
                model_name, kwargs_pert = self._build_gaussian_clump(
                    image_settings, mag, R_s, x_s, y_s
                )
            else:
                raise ValueError('Unknown source_perturbation_model setting.')

            source_model_list.append(model_name)
            kwargs_source.append(kwargs_pert)

        return source_model_list, kwargs_source

    def generate_image(self, image_settings,
                       baryons, halo, los, lens_light, source,
                       boxydisky,
                       Einstein_radii, path
                       ):

        image_list = []
        hyper_list = []
        source_truth_list = []
        source_truth_rows = []

        # convert the dfs to dicts for use with lenstronomy
        kwargs_los = los.to_dict('records')
        kwargs_bar = baryons.to_dict('records')
        kwargs_nfw = halo.to_dict('records')
        kwargs_sl  = source.to_dict('records')
        kwargs_ll  = lens_light.to_dict('records')
        kwargs_bd = None if boxydisky is None else boxydisky.to_dict('records')

        truth_model = image_settings.get('truth_model', 'composite')
        if truth_model == 'composite':
            lens_model_list = ['LOS', 'SERSIC_ELLIPSE_POTENTIAL', 'NFW_ELLIPSE_POTENTIAL']
        elif truth_model == 'boxydisky':
            lens_model_list = ['LOS_MINIMAL', 'EPL_BOXYDISKY']
        else:
            raise ValueError('Unknown truth model.')
        lens_light_model_list = ['SERSIC_ELLIPSE']
        los_minimal_keys = [
            'kappa_od',
            'gamma1_od',
            'gamma2_od',
            'omega_od',
            'kappa_los',
            'gamma1_los',
            'gamma2_los',
            'omega_los',
        ]

        # source and its potential perturbations
        # source_model_list = ['SERSIC_ELLIPSE']

        max_source_perturbations = image_settings['max_source_perturbations']
        source_perturbation_model = image_settings.get('source_perturbation_model', 'sersic')

        if max_source_perturbations == 0:
            print('Adding no perturbations to the source.')
        elif max_source_perturbations == 1:
            print(f'Adding 1 {source_perturbation_model} source perturbation per image.')
        elif max_source_perturbations > 1:
            print(
                'Adding between 1 and {} {} source perturbations per image.'.format(
                    max_source_perturbations, source_perturbation_model
                )
            )
        else:
            raise Warning("image_settings['max_source_perturbations'] should be an integer.")

        psf = 'GAUSSIAN'

        if image_settings['telescope'] == 'HST':
            # telescope settings (HST)
            band = HST(band='WFC3_F160W', psf_type=psf)
            kwargs_band = band.kwargs_single_band()
            pixel_size = band.camera['pixel_scale'] # in arcsec
        elif image_settings['telescope'] == 'JWST':
            band = JWST(band='F115W', psf_type=psf)
            kwargs_band = band.kwargs_single_band()
            pixel_size = band.camera['pixel_scale']
        else:
            raise ValueError('Unknown telescope.')

        kwargs_psf = {'psf_type': psf,
                          'fwhm': kwargs_band['seeing'],
                          'pixel_size': pixel_size,
                          'truncation': 3}

        # numerics
        kwargs_numerics = {'supersampling_factor': 1,
                           'supersampling_convolution': False}


        for i in range(image_settings['number_of_images']):

            # define kwargs for the lens
            if truth_model == 'composite':
                kwargs_lens = [kwargs_los[i], kwargs_bar[i], kwargs_nfw[i]]
            else:
                kwargs_lens = [
                    {key: kwargs_los[i][key] for key in los_minimal_keys},
                    kwargs_bd[i],
                ]

            source_model_list, kwargs_source = self._build_source_components(
                image_settings, kwargs_sl[i]
            )

            kwargs_source_mag = deepcopy(kwargs_source)
            source_model_list_truth = list(source_model_list)

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
            kwargs_source_amp = deepcopy(kwargs_source)

            source_truth_list.append({
                'image_index': i,
                'source_light_model_list': source_model_list_truth,
                'kwargs_source_mag': kwargs_source_mag,
                'kwargs_source_amp': kwargs_source_amp,
            })
            source_truth_rows.extend(
                self._source_component_rows(i, source_model_list_truth, kwargs_source_mag, kwargs_source_amp)
            )

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

        source_truth_filename = str(path)+'/datasets/'+str(image_settings['image_name'])+'_source_truth.pickle'
        source_truth_outfile = open(source_truth_filename, 'wb')
        pickle.dump(source_truth_list, source_truth_outfile)
        source_truth_outfile.close()

        source_truth_dataframe = pd.DataFrame(source_truth_rows)
        source_truth_dataframe.to_csv(
            str(path) + '/datasets/' + str(image_settings['image_name']) + '_source_truth.csv',
            index=False,
        )


        return None
