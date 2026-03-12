import numpy as np


class BoxyDisky:
    """
    Randomly draws a power-law lens with an aligned m=4 boxy/disky perturbation.
    """

    def __init__(
        self,
        Einstein_radius,
        e1,
        e2,
        a4_max=0.05,
        gamma=2.0,
        center_x=0.0,
        center_y=0.0,
    ):
        self.kwargs = {
            'theta_E_bodi': Einstein_radius,
            'gamma_bodi': gamma,
            'e1_bodi': e1,
            'e2_bodi': e2,
            'x_bodi': center_x,
            'y_bodi': center_y,
            'a4_a_bodi': np.random.uniform(-a4_max, a4_max),
        }
