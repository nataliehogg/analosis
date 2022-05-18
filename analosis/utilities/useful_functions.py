import numpy as np
#import matplotlib.pyplot as plt
from lenstronomy.LensModel.Profiles.sersic_utils import SersicUtil

class Utilities:
    """
    This class contains useful functions.
    """

    def __init__(self, cosmo):
        self.cosmo = cosmo
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


    def colorbar(self, mappable):
        # thanks to Joseph Long! https://joseph-long.com/writing/colorbars/
        last_axes = plt.gca()
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size = '5%', pad = 0.05)
        cbar = fig.colorbar(mappable, cax = cax)
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
        Computes convergence at half-light radius for a Sérsec profile,
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
        
        theta_E = np.sqrt(2 * rS_sun * d_ds / d_os / d_od) # in rad
        theta_E = self.angle_conversion(theta_E, 'to arcsecs')
        
        return theta_E
        
        
        
