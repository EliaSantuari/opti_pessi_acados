import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver


class OcpQuadruped:

    def __init__(self, com_height, g, mass, inertia, wc, wdc, wp, wa, wdt, wtheta, wdtheta, u_min=None, u_max=None):
        self.g = g
        self.m = mass
        self.i = inertia
        self.w = np.sqrt(g/com_height) # Frequency constant
        self.wc = wc
        self.wdc = wdc
        self.wp = wp
        self.wa = wa
        self.wdt = wdt
        self.wtheta = wtheta
        self.wdtheta = wdt

        if (u_min is not None):
            self.u_min = u_min
        if (u_max is not None):
            self.u_max = u_max

        

        

    