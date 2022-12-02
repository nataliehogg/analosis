index = 0

# set the path to analosis relative to this script
import sys
sys.path.append('../..')

from analosis.run import Run

cosmology = {'id': 'planck18', 'H0': 67.4, 'Om': 0.315}

image_settings = {'generate_image': False,
                  'number_of_images': 64,
                  'image_name': 'golden_sample',
                  'maximum_shear': 0.025,
                  'Einstein_radius_min': 0.5,
                  'min_aspect_ratio_source': 0.7,
                  'min_aspect_ratio_baryons': 0.7,
                  'min_aspect_ratio_nfw': 0.7,
                  'maximum_source_offset_factor': 1.0, 
                  'source_perturbations': [0.03, 0.03, 0.03], 
                  'sigma_halo_offset': 300} 

mcmc_settings = {'MCMC': True, 
                 'split': True,
                 'split_index': index,
                 'complexity': 'perfect_minimal',
                 'number_of_runs': 8, 
                 'starting_index': 0, 
                 'sampler': 'ZEUS', 
                 'mu': 1e3,
                 'tune': True,
                 'splitr_callback': False,
                 'job_name': 'split_batch_pm_{}'.format(index), 
                 'n_burn': 500, 
                 'n_run': 1500} 

result = Run(cosmology, image_settings, mcmc_settings)
