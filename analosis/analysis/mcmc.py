import numpy as np
import pandas as pd
import os
import pickle
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from multiprocessing import cpu_count
from analosis.image.image_generator import Image

im = Image()
ncpu = cpu_count()

class MCMC:

    def __init__(self, settings, parameters, baryons, halo, los, lens_light, Einstein_radii,
                 source, path):

        rings_to_rule_them_all = 1

        self.mcmc(settings, parameters, baryons, halo, los, lens_light, Einstein_radii,
                  source,path)

    def mcmc(self, settings, parameters, baryons, halo, los, lens_light, Einstein_radii,
             source, path):

        # if settings['scenario'] == 'distributed haloes':
        #     lens_fit_list = ['LOS_MINIMAL', 'EPL']
        # elif settings['scenario'] == 'composite lens':
        #     pass

        if settings['complexity'] == 'perfect':
            lens_fit_list = ['LOS', 'SERSIC_ELLIPSE_POTENTIAL', 'NFW_ELLIPSE']
        elif settings['complexity'] == 'power_law':
            lens_fit_list = ['LOS_MINIMAL', 'EPL']
        elif settings['complexity'] in ['perfect_minimal',
                                        'missing_offset',
                                        'missing_foreground_shear',
                                        'missing_halo_ellipticity',
                                        'missing_offset_ellipticity']:
            lens_fit_list = ['LOS_MINIMAL', 'SERSIC_ELLIPSE_POTENTIAL', 'NFW_ELLIPSE']
        else:
            raise ValueError('I didn\'t implement that setting yet.')

        kwargs_los = los.to_dict('records')
        kwargs_bar = baryons.to_dict('records')
        kwargs_nfw = halo.to_dict('records')
        kwargs_sl  = source.to_dict('records')
        kwargs_ll  = lens_light.to_dict('records')

        kwargs_likelihood = {'source_marg': True}

        # load the hyperdata
        hyperfile = str(path)+'/datasets/'+str(settings['job_name'])+'_hyperdata.pickle'
        infile = open(hyperfile,'rb')
        hyper_data = pickle.load(infile)
        infile.close()

        # global setting for number of walkers per sampled parameter
        walker_ratio = 10

        chain_list = []
        kwargs_result = []
        if settings['complexity'] == 'perfect':
            output_gamma1_os = []
            output_gamma2_os = []
            output_gamma1_od = []
            output_gamma2_od = []
            output_gamma1_ds = []
            output_gamma2_ds = []
        else:
            output_gamma1_od = []
            output_gamma2_od = []
            output_gamma1_los = []
            output_gamma2_los = []
            output_omega_los = []

        # the maximum number of iterations is your total data less the index you start at
        # the maximum is always run i.e. all images are fit unless a starting index is specified
        max_iterations = len(hyper_data) - settings['starting_index']

        for n in range(max_iterations):

            # this ensures you access the right part of your kwargs and hyperdata for the given iteration
            i = n + settings['starting_index']

            # check if the file with the custom starting index already exists
            if settings['starting_index'] > 0:
                test_file = str(path) + '/chains/' + settings['job_name'] + '_' + str(settings['complexity']) +'_' + str(i + settings['starting_index']) + '.h5'
                if os.path.exists(test_file):
                    raise ValueError('That chain file already exists; change your starting index or set it to zero to overwrite the job.')
                else:
                    pass
            else:
                pass

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
            # we have to have a big if/else for perfect vs perfect minimal models
            # common prior boundaries and step sizes
            gamma_sigma = 0.001
            omega_sigma = 0.00001
            gamma_prior = 0.5
            omega_prior = 0.001

            if settings['complexity'] == 'perfect':
                # notice that we can't just append to the already existing fixed_lens object
                # this creates a structure like [{}, {}]
                # whereas it needs to be [{,}]
                fixed_lens.append({'kappa_od': 0.0, 'kappa_os': 0.0, 'kappa_ds': 0.0,
                                   'omega_od': 0.0, 'omega_os': 0.0, 'omega_ds': 0.0})
                kwargs_lens_init.append({'gamma1_od': kwargs_los[i]['gamma1_od'] + np.random.normal(0.0, gamma_sigma), 'gamma2_od': kwargs_los[i]['gamma2_od'] + np.random.normal(0.0, gamma_sigma),
                                         'gamma1_os': kwargs_los[i]['gamma1_os'] + np.random.normal(0.0, gamma_sigma), 'gamma2_os': kwargs_los[i]['gamma2_os'] + np.random.normal(0.0, gamma_sigma),
                                         'gamma1_ds': kwargs_los[i]['gamma1_ds'] + np.random.normal(0.0, gamma_sigma), 'gamma2_ds': kwargs_los[i]['gamma2_ds'] + np.random.normal(0.0, gamma_sigma)})

                kwargs_lens_sigma.append({'gamma1_od': gamma_sigma, 'gamma2_od': gamma_sigma,
                                          'gamma1_os': gamma_sigma, 'gamma2_os': gamma_sigma,
                                          'gamma1_ds': gamma_sigma, 'gamma2_ds': gamma_sigma})

                kwargs_lower_lens.append({'gamma1_od': -gamma_prior, 'gamma2_od': -gamma_prior,
                                          'gamma1_os': -gamma_prior, 'gamma2_os': -gamma_prior,
                                          'gamma1_ds': -gamma_prior, 'gamma2_ds': -gamma_prior})

                kwargs_upper_lens.append({'gamma1_od': gamma_prior, 'gamma2_od': gamma_prior,
                                          'gamma1_os': gamma_prior, 'gamma2_os': gamma_prior,
                                          'gamma1_ds': gamma_prior, 'gamma2_ds': gamma_prior})

            elif settings['complexity'] == 'missing_foreground_shear':
                fixed_lens.append({'kappa_od': 0.0, 'gamma1_od':0.0, 'gamma2_od':0.0,
                                   'kappa_los': 0.0, 'omega_od': 0.0})

                kwargs_lens_init.append({'gamma1_los': kwargs_los[i]['gamma1_los'] + np.random.normal(0.0, gamma_sigma),
                                         'gamma2_los': kwargs_los[i]['gamma2_los'] + np.random.normal(0.0, gamma_sigma),
                                         'omega_los': kwargs_los[i]['omega_los'] + np.random.normal(0.0, omega_sigma)})

                kwargs_lens_sigma.append({'gamma1_los': gamma_sigma,
                                          'gamma2_los': gamma_sigma,
                                          'omega_los': omega_sigma})

                kwargs_lower_lens.append({'gamma1_los': -gamma_prior,
                                          'gamma2_los': -gamma_prior,
                                          'omega_los': -omega_prior})

                kwargs_upper_lens.append({'gamma1_los': gamma_prior,
                                          'gamma2_los': gamma_prior,
                                          'omega_los': omega_prior})
            else:
                # minimal model params
                # omega_LOS should not be fixed! the LOS shears in combination induce a small rotation
                # allowing for freedom in omega_LOS accounts for this and prevents bias in the shears
                fixed_lens.append({'kappa_od': 0.0, 'kappa_los': 0.0, 'omega_od': 0.0}) #, 'omega_los':0.0})

                kwargs_lens_init.append({'gamma1_od': kwargs_los[i]['gamma1_od'] + np.random.normal(0.0, gamma_sigma), 'gamma2_od': kwargs_los[i]['gamma2_od'] + np.random.normal(0.0, gamma_sigma),
                                         'gamma1_los': kwargs_los[i]['gamma1_los'] + np.random.normal(0.0, gamma_sigma), 'gamma2_los': kwargs_los[i]['gamma2_los'] + np.random.normal(0.0, gamma_sigma),
                                         'omega_los': kwargs_los[i]['omega_los'] + np.random.normal(0.0, omega_sigma)})

                kwargs_lens_sigma.append({'gamma1_od': gamma_sigma, 'gamma2_od': gamma_sigma,
                                          'gamma1_los': gamma_sigma, 'gamma2_los': gamma_sigma,
                                          'omega_los': omega_sigma})

                kwargs_lower_lens.append({'gamma1_od': -gamma_prior, 'gamma2_od': -gamma_prior,
                                          'gamma1_los': -gamma_prior, 'gamma2_los': -gamma_prior,
                                          'omega_los': -omega_prior})

                kwargs_upper_lens.append({'gamma1_od': gamma_prior, 'gamma2_od': gamma_prior,
                                          'gamma1_los': gamma_prior, 'gamma2_los': gamma_prior,
                                          'omega_los': omega_prior})


            if settings['complexity'] == 'power_law':
                fixed_lens.append({'center_x': 0.0, 'center_y': 0.0})
                kwargs_lens_init.append({'theta_E': Einstein_radii[i] + np.random.normal(0.0, 0.001), 'gamma': 2.0 + np.random.normal(0.0, 0.01),
                                         'e1': kwargs_bar[i]['e1'] + np.random.normal(0.0, 0.01), 'e2': kwargs_bar[i]['e2'] + np.random.normal(0.0, 0.01)})
                kwargs_lens_sigma.append({'theta_E': 0.001, 'gamma': 0.01,
                                         'e1': 0.01, 'e2': 0.01})
                kwargs_lower_lens.append({'theta_E': 0.3, 'gamma': 1.0,
                                         'e1': -0.5, 'e2': -0.5})
                kwargs_upper_lens.append({'theta_E': 3.0, 'gamma': 3.0,
                                         'e1': 0.5, 'e2': 0.5})

            else:
                # SERSIC_ELLIPSE_POTENTIAL
                fixed_lens.append({'center_x': 0.0, 'center_y': 0.0})

                kwargs_lens_init.append({'k_eff': kwargs_bar[i]['k_eff']+ np.random.normal(0.0, 0.01), 'R_sersic': kwargs_bar[i]['R_sersic']+ np.random.normal(0.0, 0.01),
                                          'n_sersic': kwargs_bar[i]['n_sersic']+ np.random.normal(0.0, 0.01),
                                          'e1': kwargs_bar[i]['e1']+ np.random.normal(0.0, 0.01), 'e2': kwargs_bar[i]['e2']+ np.random.normal(0.0, 0.01)})

                kwargs_lens_sigma.append({'k_eff': 0.01, 'R_sersic': 0.01, 'n_sersic': 0.01,
                                          'e1': 0.01, 'e2': 0.01})

                kwargs_lower_lens.append({'k_eff': 0.0, 'R_sersic': 0.0, 'n_sersic': 1.0,
                                          'e1': -0.5, 'e2': -0.5})

                kwargs_upper_lens.append({'k_eff': 2.0, 'R_sersic': 1.0, 'n_sersic': 8.0,
                                          'e1': 0.5, 'e2': 0.5})

                # NFW
                # common priors and step sizes
                Rs_sigma = 0.01
                Rs_prior_lower = 5.0
                Rs_prior_upper = 20.0
                alpha_sigma = 0.01
                alpha_prior_lower = 1.0
                alpha_prior_upper = 4.0
                center_nfw_sigma = 0.01
                center_nfw_prior = 0.2
                e_nfw_sigma = 0.01
                e_nfw_prior = 0.5

                if settings['complexity'] == 'missing_halo_ellipticity':
                    fixed_lens.append({'e1': 0.0, 'e2': 0.0})
                    kwargs_lens_init.append({'Rs': kwargs_nfw[i]['Rs']+ np.random.normal(0.0, Rs_sigma), 'alpha_Rs': kwargs_nfw[i]['alpha_Rs'] + np.random.normal(0.0, alpha_sigma),
                                             'center_x': kwargs_nfw[i]['center_x'] + np.random.normal(0.0, center_nfw_sigma),
                                             'center_y': kwargs_nfw[i]['center_y'] + np.random.normal(0.0, center_nfw_sigma)})

                    kwargs_lens_sigma.append({'Rs': Rs_sigma, 'alpha_Rs': alpha_sigma,
                                              'center_x': center_nfw_sigma, 'center_y': center_nfw_sigma})

                    kwargs_lower_lens.append({'Rs': Rs_prior_lower, 'alpha_Rs': alpha_prior_lower,
                                              'center_x': -center_nfw_prior, 'center_y': -center_nfw_prior})

                    kwargs_upper_lens.append({'Rs': Rs_prior_upper, 'alpha_Rs': alpha_prior_upper,
                                              'center_x': center_nfw_prior, 'center_y': center_nfw_prior})

                elif settings['complexity'] == 'missing_offset':
                    fixed_lens.append({'center_x': 0.0, 'center_y': 0.0})
                    kwargs_lens_init.append({'Rs': kwargs_nfw[i]['Rs']+ np.random.normal(0.0, Rs_sigma), 'alpha_Rs': kwargs_nfw[i]['alpha_Rs']+ np.random.normal(0.0, alpha_sigma),
                                             'e1': kwargs_nfw[i]['e1']+ np.random.normal(0.0, e_nfw_sigma), 'e2': kwargs_nfw[i]['e2']+ np.random.normal(0.0, e_nfw_sigma)})

                    kwargs_lens_sigma.append({'Rs': Rs_sigma, 'alpha_Rs': alpha_sigma,
                                              'e1': e_nfw_sigma, 'e2': e_nfw_sigma})

                    kwargs_lower_lens.append({'Rs': Rs_prior_lower, 'alpha_Rs': alpha_prior_lower,
                                              'e1': -e_nfw_prior, 'e2': -e_nfw_prior})

                    kwargs_upper_lens.append({'Rs': Rs_prior_upper, 'alpha_Rs': alpha_prior_upper,
                                              'e1': e_nfw_prior, 'e2': e_nfw_prior})

                elif settings['complexity'] == 'missing_offset_ellipticity':
                    fixed_lens.append({'center_x': 0.0, 'center_y': 0.0, 'e1': 0.0, 'e2': 0.0})
                    kwargs_lens_init.append({'Rs': kwargs_nfw[i]['Rs']+ np.random.normal(0.0, Rs_sigma), 'alpha_Rs': kwargs_nfw[i]['alpha_Rs']+ np.random.normal(0.0, alpha_sigma)})

                    kwargs_lens_sigma.append({'Rs': Rs_sigma, 'alpha_Rs': alpha_sigma})

                    kwargs_lower_lens.append({'Rs': Rs_prior_lower, 'alpha_Rs': alpha_prior_lower})

                    kwargs_upper_lens.append({'Rs': Rs_prior_upper, 'alpha_Rs': alpha_prior_upper})

                else:
                    fixed_lens.append({})
                    kwargs_lens_init.append({'Rs': kwargs_nfw[i]['Rs']+ np.random.normal(0.0, Rs_sigma), 'alpha_Rs': kwargs_nfw[i]['alpha_Rs']+ np.random.normal(0.0, alpha_sigma),
                                             'center_x': kwargs_nfw[i]['center_x']+ np.random.normal(0.0, center_nfw_sigma),
                                             'center_y': kwargs_nfw[i]['center_y']+ np.random.normal(0.0, center_nfw_sigma),
                                             'e1': kwargs_nfw[i]['e1']+ np.random.normal(0.0, e_nfw_sigma), 'e2': kwargs_nfw[i]['e2']+ np.random.normal(0.0, e_nfw_sigma)})

                    kwargs_lens_sigma.append({'Rs': Rs_sigma, 'alpha_Rs': alpha_sigma,
                                              'center_x': center_nfw_sigma, 'center_y': center_nfw_sigma,
                                              'e1': e_nfw_sigma, 'e2': e_nfw_sigma})

                    kwargs_lower_lens.append({'Rs': Rs_prior_lower, 'alpha_Rs': alpha_prior_lower,
                                              'center_x': -center_nfw_prior, 'center_y': -center_nfw_prior,
                                              'e1': -e_nfw_prior, 'e2': -e_nfw_prior})

                    kwargs_upper_lens.append({'Rs': Rs_prior_upper, 'alpha_Rs': alpha_prior_upper,
                                              'center_x': center_nfw_prior, 'center_y': center_nfw_prior,
                                              'e1': e_nfw_prior, 'e2': e_nfw_prior})

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
            kwargs_source_init.append({'R_sersic': kwargs_sl[i]['R_sersic']+ np.random.normal(0.0, 0.001), 'n_sersic': kwargs_sl[i]['n_sersic']+ np.random.normal(0.0, 0.001),
                                       'center_x': kwargs_sl[i]['center_x']+ np.random.normal(0.0, 0.01), 'center_y': kwargs_sl[i]['center_y']+ np.random.normal(0.0, 0.01),
                                       'e1': kwargs_sl[i]['e1']+ np.random.normal(0.0, 0.01), 'e2': kwargs_sl[i]['e2']+ np.random.normal(0.0, 0.01)})

            kwargs_source_sigma.append({'R_sersic': 0.001, 'n_sersic': 0.001,
                                        'center_x': 0.01, 'center_y': 0.01,
                                        'e1': 0.01, 'e2': 0.01})

            kwargs_lower_source.append({'R_sersic': 0.0, 'n_sersic': 2.0,
                                        'center_x': -0.5, 'center_y': -0.5,
                                        'e1': -0.5, 'e2': -0.5})

            kwargs_upper_source.append({'R_sersic': 1.0, 'n_sersic': 7.0,
                                         'center_x': 0.5, 'center_y': 0.5,
                                         'e1': 0.5, 'e2': 0.5})

            source_params = [kwargs_source_init, kwargs_source_sigma,
                             fixed_source, kwargs_lower_source, kwargs_upper_source]

            if parameters['lens_light'] == True:
                lens_light_model_list = ['SERSIC_ELLIPSE']

                # lens light model
                fixed_lens_light = []
                kwargs_lens_light_init = []
                kwargs_lens_light_sigma = []
                kwargs_lower_lens_light = []
                kwargs_upper_lens_light = []

                # Define parameters
                fixed_lens_light.append({'center_x': 0.0, 'center_y': 0.0})
                kwargs_lens_light_init.append({'R_sersic': kwargs_ll[i]['R_sersic']+ np.random.normal(0.0, 0.001), 'n_sersic': kwargs_ll[i]['n_sersic']+ np.random.normal(0.0, 0.001),
                                               'e1': kwargs_ll[i]['e1']+ np.random.normal(0.0, 0.01), 'e2': kwargs_ll[i]['e2']+ np.random.normal(0.0, 0.01)})
                kwargs_lens_light_sigma.append({'R_sersic': 0.001, 'n_sersic': 0.001, 'e1': 0.01, 'e2': 0.01})
                kwargs_lower_lens_light.append({'R_sersic': 0, 'n_sersic': 2.0,   'e1': -0.5, 'e2': -0.5})
                kwargs_upper_lens_light.append({'R_sersic': 1.0,  'n_sersic': 7.0,   'e1': 0.5,  'e2': 0.5})

                lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma,
                                    fixed_lens_light, kwargs_lower_lens_light, kwargs_upper_lens_light]

                kwargs_params = {'lens_model': lens_params,
                                 'source_model': source_params,
                                 'lens_light_model': lens_light_params}

                kwargs_model = {'lens_model_list': lens_fit_list,
                                'source_light_model_list': source_model_list,
                                'lens_light_model_list': lens_light_model_list}

            elif parameters['lens_light'] == False:
                kwargs_params = {'lens_model': lens_params,
                                 'source_model': source_params}

                kwargs_model = {'lens_model_list': lens_fit_list,
                                'source_light_model_list': source_model_list}
            else:
                print('Something went wrong with the lens light settings.')


            kwargs_data = hyper_data[i][0]
            kwargs_psf = hyper_data[i][1]
            kwargs_numerics = hyper_data[i][2]

            multi_band_list = [[kwargs_data, kwargs_psf, kwargs_numerics]]

            kwargs_data_joint = {'multi_band_list': multi_band_list,
                                 'multi_band_type': 'multi-linear'}
            kwargs_constraints = {}


            print('Starting MCMC')
            fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints,
                                          kwargs_likelihood, kwargs_params)

            fitting_kwargs_list = [['MCMC',
                                    {'n_burn': settings['n_burn'], 'n_run': settings['n_run'],
                                     'walkerRatio': walker_ratio, 'sigma_scale': 10.,
                                     'threadCount': ncpu,
                                     'backup_filename': str(path) + '/chains/'
                                                       + str(settings['job_name']) + '_'
                                                       + str(settings['complexity']) + '_'
                                                       + str(i) + '.h5'}]]

            chain_list.append(fitting_seq.fit_sequence(fitting_kwargs_list))
            kwargs_result.append(fitting_seq.best_fit())

            sampler_type, samples_mcmc, param_mcmc, dist_mcmc  = chain_list[n][0]

            np.savetxt(str(path) + '/datasets/' + str(settings['job_name']) + '_' +str(settings['complexity'])+ '_sampled_params.csv',
                       param_mcmc, delimiter=',',  fmt='%s')

        #     if settings['complexity'] == 'perfect':
        #         output_gamma1_od.append(kwargs_result[i]['kwargs_lens'][0]['gamma1_od'])
        #         output_gamma2_od.append(kwargs_result[i]['kwargs_lens'][0]['gamma2_od'])
        #         output_gamma1_os.append(kwargs_result[i]['kwargs_lens'][0]['gamma1_os'])
        #         output_gamma2_os.append(kwargs_result[i]['kwargs_lens'][0]['gamma2_os'])
        #         output_gamma1_ds.append(kwargs_result[i]['kwargs_lens'][0]['gamma1_ds'])
        #         output_gamma2_ds.append(kwargs_result[i]['kwargs_lens'][0]['gamma2_ds'])
        #
        #         output_los_kwargs_dataframe = pd.DataFrame(columns = ['gamma1_od', 'gamma2_od',
        #                                                               'gamma1_os', 'gamma2_os',
        #                                                               'gamma1_ds', 'gamma2_ds'])
        #
        #         output_los_kwargs_dataframe['gamma1_od'] = output_gamma1_od
        #         output_los_kwargs_dataframe['gamma2_od'] = output_gamma2_od
        #         output_los_kwargs_dataframe['gamma1_os'] = output_gamma1_os
        #         output_los_kwargs_dataframe['gamma2_os'] = output_gamma2_os
        #         output_los_kwargs_dataframe['gamma1_ds'] = output_gamma1_ds
        #         output_los_kwargs_dataframe['gamma2_ds'] = output_gamma2_ds
        #     else:
        #         output_gamma1_od.append(kwargs_result[i]['kwargs_lens'][0]['gamma1_od'])
        #         output_gamma2_od.append(kwargs_result[i]['kwargs_lens'][0]['gamma2_od'])
        #         output_gamma1_los.append(kwargs_result[i]['kwargs_lens'][0]['gamma1_los'])
        #         output_gamma2_los.append(kwargs_result[i]['kwargs_lens'][0]['gamma2_los'])
        #         output_omega_los.append(kwargs_result[i]['kwargs_lens'][0]['omega_los'])
        #
        #         output_los_kwargs_dataframe = pd.DataFrame(columns = ['gamma1_od', 'gamma2_od',
        #                                                               'gamma1_los', 'gamma2_los', 'omega_los'])
        #
        #         output_los_kwargs_dataframe['gamma1_od'] = output_gamma1_od
        #         output_los_kwargs_dataframe['gamma2_od'] = output_gamma2_od
        #         output_los_kwargs_dataframe['gamma1_los'] = output_gamma1_los
        #         output_los_kwargs_dataframe['gamma2_los'] = output_gamma2_los
        #         output_los_kwargs_dataframe['omega_los'] = output_omega_los
        #
        #
        # # save the best-fit los kwargs according to emcee -- should be roughly the same as the chain consumer returned values
        # output_los_kwargs_dataframe.to_csv(str(path) + '/datasets/'+ str(settings['job_name']) + '_' + str(settings['complexity']) +'_output_kwargs.csv', index = False)


        return None
