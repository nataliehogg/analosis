import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from lenstronomy.LensModel.Profiles.sersic_utils import SersicUtil
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.SimulationAPI.ObservationConfig.HST import HST
from lenstronomy.LensModel.lens_model import LensModel
from scipy.optimize import fsolve


class Utilities:
    """
    This class contains useful functions.
    """

    def __init__(self, cosmo, path):
        self.cosmo = cosmo
        self.path = path
        self.sersic_util = SersicUtil()


    def dA(self, z1, z2):
        """
        returns angular diameter distances in Mpc
        """

        distance = self.cosmo.angular_diameter_distance_z1z2(z1, z2).value

        return distance


    def ellipticity(self, phi, q):
        # transforms orientation angle phi and aspect ratio q into complex ellipticity modulii e1, e2
        e1 = (1 - q)/(1 + q)*np.cos(2*phi)
        e2 = (1 - q)/(1 + q)*np.sin(2*phi)
        return e1, e2


    def colorbar(self, mappable, lab, ori):
        # thanks to Joseph Long! https://joseph-long.com/writing/colorbars/
        last_axes = plt.gca()
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size = '5%', pad = 0.05)
        cbar = fig.colorbar(mappable, cax = cax, label=lab, orientation=ori)
        plt.sca(last_axes)
        return cbar


    def distance_conversion(self, distance, conversion_type):
        # converts a distance in Mpc to Gpc, kpc, pc or m
        # careful! it doesn't sanity check your input
        if conversion_type == 'to Gpc':
            new_distance = distance/(10**3)
        elif conversion_type == 'to kpc':
            new_distance = distance*(10**3)
        elif conversion_type == 'to pc':
            new_distance = distance*(10**6)
        elif conversion_type == 'to m':
            new_distance = distance*(3.086*10**22)
        else:
            print('Unknown conversion type')
        return new_distance


    def angle_conversion(self, angle, conversion_type):
        # converts an angle in arcsec to rad or rad to arcsec
        # careful! it doesn't sanity check your input
        conversion_factor = np.pi/(180*3600)
        if conversion_type == 'to arcsecs':
            new_angle = angle/conversion_factor
        elif conversion_type == 'to radians':
            new_angle = angle*conversion_factor
        else:
            raise ValueError('Unknown conversion type')
        return new_angle


    def get_effective_convergence(self, mass, R_sersic, n_sersic, e1, e2,
                                  d_os, d_od, d_ds):
        """
        Computes convergence at half-light radius for a Sérsic profile,
        assuming light traces baryonic mass.
        """

        # Get critical density in lens plane
        rS_sun = 2.95e3 / (3.086e22) # Schwarszchild radius of the sun in Mpc
        Sigma_crit_rad = d_os * d_od / 2 / rS_sun / d_ds # [M_sun / rad^2]
        Sigma_crit = Sigma_crit_rad / (3600 * 180 / np.pi)**2 # [M_sun / arcsec^2]

        # Compute integrated convergence if kappa_eff = 1

        integral_unity = self.sersic_util.total_flux(amp=1,
                                                R_sersic=R_sersic,
                                                n_sersic=n_sersic,
                                                e1=e1, e2=e2) # [arcsec^2]
        # Deduce the value of kappa_eff
        kappa = mass / Sigma_crit / integral_unity

        return kappa


    def gamma(self, max_value, which):
        '''
        gives you a random line of sight shear value according to the max value you set
        '''
        gamma_sq = np.random.uniform(0.0, max_value**2.)
        gamma_sqrt = np.sqrt(gamma_sq)
        phi = np.random.uniform(0.0, np.pi)
        if which == 'one':
            gamma = gamma_sqrt*np.cos(2*phi)
        elif which == 'two':
            gamma = gamma_sqrt*np.sin(2*phi)
        else:
            print('bad option')
        return gamma


    def compute_omega_LOS(self, kwargs_shear):
        """
        This function computes the rotation (antisymmmetic) part of the LOS amplification
        matrix, assuming that it is defined as
        A_LOS = A_od * A_ds^-1 * A_os
        """

        # Define the complex shears
        gamma_od = kwargs_shear['gamma1_od'] + kwargs_shear['gamma2_od'] * 1j
        gamma_os = kwargs_shear['gamma1_os'] + kwargs_shear['gamma2_os'] * 1j
        gamma_ds = kwargs_shear['gamma1_ds'] + kwargs_shear['gamma2_ds'] * 1j

        # Compute the rotation
        kappaomega = (gamma_ds * np.conj(gamma_os)
                      - gamma_od * np.conj(gamma_os)
                      + gamma_od * np.conj(gamma_ds)
                     ) / (1 - gamma_ds) / (1 - np.conj(gamma_ds))
        omega = np.imag(kappaomega)

        return omega


    def Einstein_radius_point_lens(self, mass, distances):
        """
        This function returns the Einstein radius [in arcsec] of a point lens
        with mass [in solar masses] and with the various distances [in Mpc].
        """

        rS_sun = 2.95e3 / (3.086e22) # Schwarszchild radius of the sun in Mpc
        d_od = distances['od']
        d_os = distances['os']
        d_ds = distances['ds']

        theta_E = np.sqrt(2 * rS_sun * mass * d_ds / d_os / d_od) # in rad
        theta_E = self.angle_conversion(theta_E, 'to arcsecs')

        return theta_E


    def get_dataframe(self, kwargs_dict):

        dataframe = pd.DataFrame(kwargs_dict)

        return dataframe

    def combine_dataframes(self, dataframe_list):

        dataframe = pd.concat(dataframe_list, axis=1)

        return dataframe

    def save_input_kwargs(self, image_settings, dataframe):
        '''
        saves input kwargs dataframe to file
        '''

        dataframe.to_csv(str(self.path) + '/datasets/'+str(image_settings['image_name'])+'_input_kwargs.csv',
                              index = False)

    def append_from_starting_index(self, path, image_settings, dataframe):
        '''
        appends new dataframe from custom starting index to original dataframe

        use at your own peril!
        '''

        original_input_kwargs_dataframe = pd.read_csv(str(path) + '/datasets/' + str(image_settings['image__name']) + '_input_kwargs.csv')

        # check the starting index is on a blank row
        # this could be broken by trying to fill a gap in the df which is too small for your new df to fit
        # in that case you'd end up overwriting stuff lower down in the df
        # never mind, it doesn't work... to be implemented... maybe
        # if pd.isnull(original_input_kwargs_dataframe.at[settings['starting_index'], 'kappa_os']) == True:
        #     pass
        # else:
        #     raise ValueError('That row isn\'t empty.')

        # compute the total number of rows the input kwargs df with blank rows should have
        # the starting index + number of new rows - number of rows in df to be appended
        total_rows_new_df = settings['starting_index'] + settings['number_of_images'] - dataframe.shape[0]

        # add the requisite number of blank rows to the og df
        original_df_blanks = original_input_kwargs_dataframe.reindex(list(range(0, total_rows_new_df))).reset_index(drop=True)

        # append the new df to the df with blanks
        final_df = pd.concat([original_df_blanks, dataframe], ignore_index = True)

        return final_df

    def rename_kwargs(self, baryons, halo, lens_light, source):
        baryons = baryons.rename(
                index=str, columns={
                    'k_eff_bar': 'k_eff',
                    'R_sersic_bar': 'R_sersic',
                    'n_sersic_bar': 'n_sersic',
                    'x_bar': 'center_x',
                    'y_bar': 'center_y',
                    'e1_bar': 'e1',
                    'e2_bar': 'e2'})

        halo = halo.rename(
            index=str, columns={
                'x_nfw': 'center_x',
                'y_nfw': 'center_y',
                'e1_nfw': 'e1',
                'e2_nfw': 'e2'})

        if lens_light is not None:
            lens_light = lens_light.rename(
                index=str, columns={
                    'magnitude_ll': 'magnitude',
                    'R_sersic_ll': 'R_sersic',
                    'n_sersic_ll': 'n_sersic',
                    'x_ll': 'center_x',
                    'y_ll': 'center_y',
                    'e1_ll': 'e1',
                    'e2_ll': 'e2'})
        else:
            lens_light = None

        source = source.rename(
            index=str, columns={
                'magnitude_sl': 'magnitude',
                'R_sersic_sl': 'R_sersic',
                'n_sersic_sl': 'n_sersic',
                'x_sl': 'center_x',
                'y_sl': 'center_y',
                'e1_sl': 'e1',
                'e2_sl': 'e2'})
        return baryons, halo, lens_light, source


def estimate_quality(input_kwargs, snr_cut=1):
    """
    This function estimates the quality of an image as a sort of
    cumulated signal-to-noise ratio (SNR), according to

    quality = Sum(SNR[p]) for each pixel p if SNR[p] > snr_cut.

    This definition implies that the quality increases with the
    absolute SNR in the pixels, and it also increases with the
    number of pixels with an SNR above 1, thereby accounting for
    the resolution of the image and the extension of the lensed
    image.

    Importantly, we do not include the lens light when defining
    this quality criterion.
    """

    qualities = []

    # Extract data
    los_cols = ['kappa_os', 'gamma1_os', 'gamma2_os', 'omega_os',
            'kappa_od', 'gamma1_od', 'gamma2_od', 'omega_od',
            'kappa_ds', 'gamma1_ds', 'gamma2_ds', 'omega_ds',
            'kappa_los', 'gamma1_los', 'gamma2_los', 'omega_los']
    bar_cols = ['R_sersic_bar', 'n_sersic_bar', 'k_eff_bar', 'e1_bar', 'e2_bar', 'x_bar', 'y_bar', 'mass_bar']
    nfw_cols = ['Rs', 'alpha_Rs', 'x_nfw', 'y_nfw', 'e1_nfw', 'e2_nfw', 'virial_mass_nfw']
    sl_cols = ['magnitude_sl', 'R_sersic_sl', 'n_sersic_sl', 'x_sl', 'y_sl', 'e1_sl', 'e2_sl']

    los        = input_kwargs.loc[:, los_cols]
    baryons    = input_kwargs.loc[:, bar_cols]
    halo       = input_kwargs.loc[:, nfw_cols]
    source     = input_kwargs.loc[:, sl_cols]
    Einstein_radii = input_kwargs.loc[:, 'theta_E']

    # Rename the keys for lenstronomy
    util = Utilities(cosmo=None, path=None)
    baryons, halo, lens_light, source = util.rename_kwargs(baryons, halo, None, source)

    # Convert dataframes into lists of dictionaries
    kwargs_los = los.to_dict('records')
    kwargs_bar = baryons.to_dict('records')
    kwargs_nfw = halo.to_dict('records')
    kwargs_sl  = source.to_dict('records')

    # Eliminate useless keys
    for kwargs in kwargs_bar:
        del kwargs['mass_bar']
    for kwargs in kwargs_nfw:
        del kwargs['virial_mass_nfw']


    # PRODUCE IMAGES AND EVALUATE QUALITY

    for i in range(len(kwargs_bar)):

        # lens models
        lens_model_list = ['LOS', 'SERSIC_ELLIPSE_POTENTIAL', 'NFW_ELLIPSE']
        lens_light_model_list = [] # no lens light to evaluate image quality
        source_model_list = ['SERSIC_ELLIPSE'] # we don't include source perturbations here
        kwargs_model = {'lens_model_list': lens_model_list,
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

        # define kwargs for the lens, source, lens_light
        kwargs_lens = [kwargs_los[i], kwargs_bar[i], kwargs_nfw[i]]
        kwargs_source = [kwargs_sl[i]]
        kwargs_lens_light = []

        # compute the size of the image from the Einstein radius
        theta_E = Einstein_radii[i] # in arcsec
        beta = np.sqrt(kwargs_sl[i]['center_x']**2
                       + kwargs_sl[i]['center_y']**2) # source offset
        image_size = 4 * (theta_E + beta)
        numpix = int(image_size / pixel_size)

        # simulation API
        sim = SimAPI(numpix=numpix,
                     kwargs_single_band=kwargs_band,
                     kwargs_model=kwargs_model)

        # extract data kwargs
        kwargs_data = sim.kwargs_data

        # convert magnitudes into amplitudes
        kwargs_lens_light, kwargs_source, ps = sim.magnitude2amplitude(kwargs_lens_light_mag=None, kwargs_source_mag=kwargs_source)

        # generate image
        imSim = sim.image_model_class(kwargs_numerics)
        image = imSim.image(kwargs_lens=kwargs_lens,
                            kwargs_source=kwargs_source)
        # add noise
        image_noisy = image + sim.noise_for_model(model=image)
        kwargs_data['image_data'] = image_noisy

        # extract noise amplitude
        background_rms = kwargs_data['background_rms']

        # compute quality
        quality = 0
        for signal in np.nditer(image_noisy):
            snr = signal / background_rms
            if snr > snr_cut:
                quality += snr
        qualities.append(quality)


    return qualities


def estimate_Einstein_radius(R_sersic, n_sersic, k_eff, Rs, alpha_Rs, guess=1):
    """
    This functions estimates the Einstein radius [in arcsec] of a composite
    model made of a Sérsic profile and an NFW profile, assumed to be concentric
    and axially symmetric. The Einstein radius the solves the equation
    alpha(theta_E) = theta_E ,
    where alpha is the total displacement angle of the composite lens.
    """

    lens = LensModel(lens_model_list=['SERSIC', 'NFW'])
    kwargs_sersic = {'R_sersic': R_sersic,
                     'n_sersic': n_sersic,
                     'k_eff': k_eff}
    kwargs_nfw = {'Rs': Rs, 'alpha_Rs': alpha_Rs}
    kwargs_lens = [kwargs_sersic, kwargs_nfw]

    func = lambda theta: lens.alpha(x=theta, y=0, kwargs=kwargs_lens)[0] - theta
    theta_E = fsolve(func, guess)[0]

    return theta_E

def estimate_quality_lens_light(input_kwargs, snr_cut=1):
    """
    Compute image quality including lens light in the calculation
    """

    qualities = []

    # Extract data
    los_cols = ['kappa_os', 'gamma1_os', 'gamma2_os', 'omega_os',
            'kappa_od', 'gamma1_od', 'gamma2_od', 'omega_od',
            'kappa_ds', 'gamma1_ds', 'gamma2_ds', 'omega_ds',
            'kappa_los', 'gamma1_los', 'gamma2_los', 'omega_los']
    bar_cols = ['R_sersic_bar', 'n_sersic_bar', 'k_eff_bar', 'e1_bar', 'e2_bar', 'x_bar', 'y_bar', 'mass_bar']
    nfw_cols = ['Rs', 'alpha_Rs', 'x_nfw', 'y_nfw', 'e1_nfw', 'e2_nfw', 'virial_mass_nfw']
    sl_cols = ['magnitude_sl', 'R_sersic_sl', 'n_sersic_sl', 'x_sl', 'y_sl', 'e1_sl', 'e2_sl']
    ll_cols = ['magnitude_ll', 'R_sersic_ll', 'n_sersic_ll', 'x_ll', 'y_ll', 'e1_ll', 'e2_ll']

    los        = input_kwargs.loc[:, los_cols]
    baryons    = input_kwargs.loc[:, bar_cols]
    halo       = input_kwargs.loc[:, nfw_cols]
    source     = input_kwargs.loc[:, sl_cols]
    lens_light =  input_kwargs.loc[:, ll_cols]
    Einstein_radii = input_kwargs.loc[:, 'theta_E']

    # Rename the keys for lenstronomy
    util = Utilities(cosmo=None, path=None)
    baryons, halo, lens_light, source = util.rename_kwargs(baryons, halo, lens_light, source)

    # Convert dataframes into lists of dictionaries
    kwargs_los = los.to_dict('records')
    kwargs_bar = baryons.to_dict('records')
    kwargs_nfw = halo.to_dict('records')
    kwargs_sl  = source.to_dict('records')
    kwargs_ll  = lens_light.to_dict('records')

    # Eliminate useless keys
    for kwargs in kwargs_bar:
        del kwargs['mass_bar']
    for kwargs in kwargs_nfw:
        del kwargs['virial_mass_nfw']


    # PRODUCE IMAGES AND EVALUATE QUALITY

    for i in range(len(kwargs_bar)):

        # lens models
        lens_model_list = ['LOS', 'SERSIC_ELLIPSE_POTENTIAL', 'NFW_ELLIPSE']
        lens_light_model_list = ['SERSIC_ELLIPSE']
        source_model_list = ['SERSIC_ELLIPSE'] # we don't include source perturbations here
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

        # define kwargs for the lens, source, lens_light
        kwargs_lens = [kwargs_los[i], kwargs_bar[i], kwargs_nfw[i]]
        kwargs_source = [kwargs_sl[i]]
        kwargs_lens_light = [kwargs_ll[i]]

        # compute the size of the image from the Einstein radius
        theta_E = Einstein_radii[i] # in arcsec
        beta = np.sqrt(kwargs_sl[i]['center_x']**2
                       + kwargs_sl[i]['center_y']**2) # source offset
        image_size = 4 * (theta_E + beta)
        numpix = int(image_size / pixel_size)

        # simulation API
        sim = SimAPI(numpix=numpix,
                     kwargs_single_band=kwargs_band,
                     kwargs_model=kwargs_model)

        # extract data kwargs
        kwargs_data = sim.kwargs_data

        # convert magnitudes into amplitudes
        kwargs_lens_light, kwargs_source, ps = sim.magnitude2amplitude(kwargs_lens_light_mag=kwargs_lens_light, kwargs_source_mag=kwargs_source)


        # generate image
        imSim = sim.image_model_class(kwargs_numerics)
        image = imSim.image(kwargs_lens=kwargs_lens,
                            kwargs_lens_light=kwargs_lens_light,
                            kwargs_source=kwargs_source)
        # add noise
        image_noisy = image + sim.noise_for_model(model=image)
        kwargs_data['image_data'] = image_noisy

        # extract noise amplitude
        background_rms = kwargs_data['background_rms']

        # compute quality
        quality = 0
        for signal in np.nditer(image_noisy):
            snr = signal / background_rms
            if snr > snr_cut:
                quality += snr
        qualities.append(quality)


    return qualities
