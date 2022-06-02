import numpy as np
import pandas as pd
from lenstronomy.Workflow.fitting_sequence import FittingSequence

from analosis.image.image_generator import Image
im = Image()


class MCMC:

    def __init__(self, settings, baryons, halo, los, lens_light, source, kwargs_data, kwargs_psf, kwargs_numerics, path):
        rings_to_rule_them_all = 1

        self.mcmc(settings, baryons, halo, los, lens_light, source, kwargs_data, kwargs_psf, kwargs_numerics, path)

    def mcmc(self, settings, baryons, halo, los, lens_light, source, kwargs_data, kwargs_psf, kwargs_numerics, path):

        if settings['scenario'] == 'distributed haloes':
            lens_fit_list = ['LOS_MINIMAL', 'EPL']
        elif settings['scenario'] == 'composite lens':
            pass

        if settings['complexity'] == 'perfect':
            # we need to deal with the error that happens by selecting perfect
            # but not having the non-minimal LOS parameters defined
            # currently you get a key error regarding kappa_os if you run this setting
            lens_fit_list = ['LOS', 'SERSIC_ELLIPSE_POTENTIAL', 'NFW_ELLIPSE']
        elif settings['complexity'] == 'perfect minimal':
            lens_fit_list = ['LOS_MINIMAL', 'SERSIC_ELLIPSE_POTENTIAL', 'NFW_ELLIPSE']
        elif settings['complexity'] == 'minimal spherical halo':
            lens_fit_list = ['LOS_MINIMAL', 'SERSIC_ELLIPSE_POTENTIAL', 'NFW_ELLIPSE']
        else:
            raise ValueError('I didn\'t implement that setting yet.')

        kwargs_los = los.to_dict('records')
        kwargs_bar = baryons.to_dict('records')
        kwargs_nfw = halo.to_dict('records')
        kwargs_sl  = source.to_dict('records')
        kwargs_ll = lens_light.to_dict('records')

        kwargs_likelihood = {'source_marg': True}

        walker_ratio = 10

        chain_list = []
        kwargs_result = []
        output_gamma1_od = []
        output_gamma2_od = []
        output_gamma1_los = []
        output_gamma2_los = []
        output_omega_los = []


        for i in range(settings['number_of_images']):

            # Initialise the lists of parameters
            fixed_lens = []
            kwargs_lens_init = []
            kwargs_lens_sigma = []
            kwargs_lower_lens = []
            kwargs_upper_lens = []

            # Parameters for the lens
            # Specify which parameters are fixed and, for the parameters that will be
            # explored, choose their initial value, step size, lower and higher limits

            # Line-of-sight parameters
            # omega_LOS should not be fixed! the LOS shears in combination induce a small rotation
            # allowing for freedom in omega_LOS accounts for this and prevents bias in the shears
            fixed_lens.append({'kappa_od': 0.0, 'kappa_los': 0.0, 'omega_od': 0.0}) #, 'omega_los':0.0})

            kwargs_lens_init.append({'gamma1_od': kwargs_los[i]['gamma1_od'], 'gamma2_od': kwargs_los[i]['gamma2_od'],
                                     'gamma1_los': kwargs_los[i]['gamma1_los'], 'gamma2_los': kwargs_los[i]['gamma2_los'],
                                     'omega_los': kwargs_los[i]['omega_los']})

            gamma_sigma = 0.001
            omega_sigma = 0.0001
            gamma_prior = 0.15
            omega_prior = 0.01

            kwargs_lens_sigma.append({'gamma1_od': gamma_sigma, 'gamma2_od': gamma_sigma,
                                      'gamma1_los': gamma_sigma, 'gamma2_los': gamma_sigma,
                                      'omega_los': omega_sigma})

            kwargs_lower_lens.append({'gamma1_od': -gamma_prior, 'gamma2_od': -gamma_prior,
                                      'gamma1_los': -gamma_prior, 'gamma2_los': -gamma_prior,
                                      'omega_los': -omega_prior})

            kwargs_upper_lens.append({'gamma1_od': gamma_prior, 'gamma2_od': gamma_prior,
                                      'gamma1_los': gamma_prior, 'gamma2_los': gamma_prior,
                                      'omega_los': omega_prior})

            # SERSIC_ELLIPSE_POTENTIAL
            fixed_lens.append({'center_x': 0.0, 'center_y': 0.0})

            kwargs_lens_init.append({'k_eff': kwargs_bar[i]['k_eff'], 'R_sersic': kwargs_bar[i]['R_sersic'],
                                      'n_sersic': kwargs_bar[i]['n_sersic'], 'e1': kwargs_bar[i]['e1'], 'e2': kwargs_bar[i]['e2']})

            kwargs_lens_sigma.append({'k_eff': 0.01, 'R_sersic': 0.01, 'n_sersic': 0.01,
                                      'e1': 0.01, 'e2': 0.01})

            kwargs_lower_lens.append({'k_eff': 0, 'R_sersic': 0.0, 'n_sersic': 1.0,
                                      'e1': -1, 'e2': -1})

            kwargs_upper_lens.append({'k_eff': 0.5, 'R_sersic': 1.0, 'n_sersic': 8.0,
                                      'e1': 1, 'e2': 1})

            # NFW
            if settings['complexity'] == 'minimal spherical halo':
                fixed_lens.append({'e1': 0.0, 'e2': 0.0})
                kwargs_lens_init.append({'Rs': kwargs_nfw[i]['Rs'], 'alpha_Rs': kwargs_nfw[i]['alpha_Rs'],
                                         'center_x': kwargs_nfw[i]['center_x'], 'center_y': kwargs_nfw[i]['center_y']})
                kwargs_lens_sigma.append({'Rs': 0.01, 'alpha_Rs': 0.01,
                                          'center_x': 0.01, 'center_y': 0.01})
                kwargs_lower_lens.append({'Rs': 0.0, 'alpha_Rs': 0.0,
                                          'center_x': -0.2, 'center_y': -0.2})
                kwargs_upper_lens.append({'Rs': 15.0, 'alpha_Rs': 2.0,
                                          'center_x': 0.2, 'center_y': 0.2})
            else:
                fixed_lens.append({})
                kwargs_lens_init.append({'Rs': kwargs_nfw[i]['Rs'], 'alpha_Rs': kwargs_nfw[i]['alpha_Rs'],
                                         'center_x': kwargs_nfw[i]['center_x'], 'center_y': kwargs_nfw[i]['center_y'],
                                         'e1': kwargs_nfw[i]['e1'], 'e2': kwargs_nfw[i]['e2']})
                kwargs_lens_sigma.append({'Rs': 0.01, 'alpha_Rs': 0.01,
                                          'center_x': 0.01, 'center_y': 0.01,
                                          'e1': 0.01, 'e2': 0.01})
                kwargs_lower_lens.append({'Rs': 0.0, 'alpha_Rs': 0.0,
                                          'center_x': -0.2, 'center_y': -0.2,
                                          'e1': -1, 'e2': -1})
                kwargs_upper_lens.append({'Rs': 15.0, 'alpha_Rs': 2.0,
                                          'center_x': 0.2, 'center_y': 0.2,
                                          'e1': 1, 'e2': 1})

            lens_params = [kwargs_lens_init,
                           kwargs_lens_sigma,
                           fixed_lens,
                           kwargs_lower_lens,
                           kwargs_upper_lens]

            # SOURCE MODEL

            source_model_list = ['SERSIC_ELLIPSE']

            # Initialise the lists of parameters
            fixed_source = []
            kwargs_source_init = []
            kwargs_source_sigma = []
            kwargs_lower_source = []
            kwargs_upper_source = []


            # Define parameters
            fixed_source.append({})
            kwargs_source_init.append({'R_sersic': kwargs_sl[i]['R_sersic'], 'n_sersic': kwargs_sl[i]['n_sersic'],
                                       'center_x': kwargs_sl[i]['center_x'], 'center_y': kwargs_sl[i]['center_y'],
                                       'e1': kwargs_sl[i]['e1'], 'e2': kwargs_sl[i]['e2']})
            kwargs_source_sigma.append({'R_sersic': 0.001, 'n_sersic': 0.001,
                                        'center_x': 0.01, 'center_y': 0.01,
                                        'e1': 0.01, 'e2': 0.01})
            kwargs_lower_source.append({'R_sersic': 0.001, 'n_sersic': 2.0,
                                        'center_x': -1, 'center_y': -1,
                                        'e1': -1, 'e2': -1})
            kwargs_upper_source.append({'R_sersic': 1.0, 'n_sersic': 7.0,
                                         'center_x': 1, 'center_y': 1,
                                         'e1': 1, 'e2': 1})

            source_params = [kwargs_source_init, kwargs_source_sigma,
                             fixed_source, kwargs_lower_source, kwargs_upper_source]

            if settings['lens_light'] == True:
                lens_light_model_list = ['SERSIC_ELLIPSE']

                # lens light model
                fixed_lens_light = []
                kwargs_lens_light_init = []
                kwargs_lens_light_sigma = []
                kwargs_lower_lens_light = []
                kwargs_upper_lens_light = []

                # Define parameters
                fixed_lens_light.append({'center_x': 0.0, 'center_y': 0.0})
                kwargs_lens_light_init.append({'R_sersic': kwargs_ll[i]['R_sersic'], 'n_sersic': kwargs_ll[i]['n_sersic'],
                                               'e1': kwargs_ll[i]['e1'], 'e2': kwargs_ll[i]['e2']})
                kwargs_lens_light_sigma.append({'R_sersic': 0.001, 'n_sersic': 0.001, 'e1': 0.01, 'e2': 0.01})
                kwargs_lower_lens_light.append({'R_sersic': 0, 'n_sersic': 2.0,   'e1': -1.0, 'e2': -1.0,})
                kwargs_upper_lens_light.append({'R_sersic': 1.0,  'n_sersic': 7.0,   'e1': 1.0,  'e2': 1.0})

                lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma,
                                    fixed_lens_light, kwargs_lower_lens_light, kwargs_upper_lens_light]

                kwargs_params = {'lens_model': lens_params,
                                 'source_model': source_params,
                                 'lens_light_model': lens_light_params}

                kwargs_model = {'lens_model_list': lens_fit_list,
                                'source_light_model_list': source_model_list,
                                'lens_light_model_list': lens_light_model_list}

            elif settings['lens_light'] == False:
                kwargs_params = {'lens_model': lens_params,
                                 'source_model': source_params}

                kwargs_model = {'lens_model_list': lens_fit_list,
                                'source_light_model_list': source_model_list}
            else:
                print('Something went wrong with the lens light settings.')

            multi_band_list = [[kwargs_data, kwargs_psf, kwargs_numerics]]

            kwargs_data_joint = {'multi_band_list': multi_band_list,
                                 'multi_band_type': 'multi-linear'}
            kwargs_constraints = {}


            print('Starting MCMC')
            fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints,
                                          kwargs_likelihood, kwargs_params)

            fitting_kwargs_list = [['MCMC',
                                    {'n_burn': settings['n_burn'], 'n_run': settings['n_run'],
                                     'walkerRatio': walker_ratio, 'sigma_scale': 1.,
                                     'backup_filename': str(path) + '/chains/'+ str(settings['job_name']) + '_' + str(i) + '.h5'}]]

            chain_list.append(fitting_seq.fit_sequence(fitting_kwargs_list))
            kwargs_result.append(fitting_seq.best_fit())

            sampler_type, samples_mcmc, param_mcmc, dist_mcmc  = chain_list[i][0]

            number_of_walkers = len(param_mcmc)*walker_ratio

            print('the number of walkers in this chain is', number_of_walkers)

            output_gamma1_od.append(kwargs_result[i]['kwargs_lens'][0]['gamma1_od'])
            output_gamma2_od.append(kwargs_result[i]['kwargs_lens'][0]['gamma2_od'])
            output_gamma1_los.append(kwargs_result[i]['kwargs_lens'][0]['gamma1_los'])
            output_gamma2_los.append(kwargs_result[i]['kwargs_lens'][0]['gamma2_los'])
            output_omega_los.append(kwargs_result[i]['kwargs_lens'][0]['omega_los'])

        output_los_kwargs_dataframe = pd.DataFrame(columns = ['gamma1_od', 'gamma2_od',
                                                              'gamma1_los', 'gamma2_los', 'omega_los'])

        output_los_kwargs_dataframe['gamma1_od'] = output_gamma1_od
        output_los_kwargs_dataframe['gamma2_od'] = output_gamma2_od
        output_los_kwargs_dataframe['gamma1_los'] = output_gamma1_los
        output_los_kwargs_dataframe['gamma2_los'] = output_gamma2_los
        output_los_kwargs_dataframe['omega_los'] = output_omega_los


        # save the best-fit los kwargs according to emcee -- should be roughly the same as the chain consumer returned values
        output_los_kwargs_dataframe.to_csv(str(path) + '/datasets/output_kwargs.csv', index = False)


        return None
