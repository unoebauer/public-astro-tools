#!/usr/bin/env python
import numpy as np
import astropy.constants as c
import scipy.optimize
import scipy.integrate
"""
This module provides tools to calculate the structure of steady radiative
shocks based on equilibrium and non-equilibrium diffusion using the solution
procedures developed by LR07 and LE08 respectively.

All tools operate with non-dimensional quantities, using the definitions in
LE08 (unless stated otherwise). The non-dimensionalization works such that the
downstream non-dimensional density and temperature are both 1.

References
----------
    LR07: Lowrie & Rauenzahn 2007, Shock Waves (2007) 16:445–453
            DOI 10.1007/s00193-007-0081-2
    LE08: Lowrie & Edwards 2008, Shock Waves (2008) 18:129–143
            DOI 10.1007/s00193-008-0143-0
"""


a_rad = 4. * c.sigma_sb.cgs.value / c.c.cgs.value


class shock_base(object):
    def __init__(self, M0, P0, kappa, gamma=5./3.):
        """Basic calculator for the shock jump.

        Parameters
        ----------
        M0 : float
            downstream Mach number
        P0 : float
            non-dimensional constant (see LR07, Eq. 3); combination of
            reference values used in the non-dimensionalization process.
        kappa : float
            radiation diffusivity (definition according to LE08)
        gamma : float
            adiabatic index (default 5/3)
        """

        self.v0 = M0
        self.rho0 = 1.
        self.T0 = 1.

        self.P0 = P0
        self.M0 = M0
        self.gamma = gamma
        self.kappa = kappa

        self.check_shock_entropy_condition()

    def check_shock_entropy_condition(self):
        """Check whether initial setup satisfies the shock entropy condition

        Raises
        ------
        ValueError
            if entropy condition is violated
        """

        # radiation modified sound speed; LR07 Eq (11)
        a0star = np.sqrt(1. + 4. / 9. * self.P0 * (self.gamma - 1.) *
                         (4. * self.P0 * self.gamma +
                          3. * (5. - 3. * self.gamma)) /
                         (1. + 4. * self.P0 * self.gamma * (self.gamma - 1.)))

        # entropy condition; see LR07, Eq. (9)
        if self.M0 < a0star:
            raise ValueError("Violation of Entropy Condition")

    def calculate_shock_jump(self):
        """Calculate overall shock jump

        Returns
        -------
        rho1 : float
            upstream non-dimensional density
        v1 : float
            upstream non-dimensional velocity
        T1 : float
            upstream non-dimensional temperature
        """

        # helper functions used in Eq. (12) of LR07
        def f1(T):
            return (3. * (self.gamma + 1.) * (T - 1.) -
                    self.P0 * self.gamma * (self.gamma - 1.) * (7. + T**4))

        def f2(T):
            return (12. * (self.gamma - 1.)**2 * T *
                    (3. + self.gamma * self.P0 * (1. + 7. * T**4)))

        # System of equations determining rho_1, i.e. Eqs. (12) and (13) in
        # LR07
        def func(x):
            rho = x[0]
            T = x[1]

            y = np.zeros(2)
            # LR07, Eq (12)
            y[0] = ((f1(T) + np.sqrt(f1(T)**2 + f2(T))) /
                    (6. * (self.gamma - 1.) * T) - rho)
            # LR07, Eq (13)
            y[1] = (3 * rho * (rho * T - 1.) + self.gamma * self.P0 * rho *
                    (T**4 - 1.) - 3. * self.gamma * (rho - 1.) * self.M0**2)

            return y

        # initial guess
        x0 = np.array([10, 1000])

        res = scipy.optimize.fsolve(func, x0)

        # Overall shock jump
        self.rho1 = res[0]
        self.T1 = res[1]
        self.v1 = self.M0 / self.rho1


class eqdiff_shock_calculator(shock_base):
    def __init__(self, M0, P0, kappa, gamma=5./3.):
        super(eqdiff_shock_calculator, self).__init__(M0, P0, kappa,
                                                      gamma=gamma)
        self.calculate_shock_jump()

    def precursor_region(self, xpr):

        def m(T):
            return (0.5 * (self.gamma * self.M0**2 + 1.) +
                    self.gamma * self.P0 / 6. * (1. - T**4))

        def rho(T):
            return (m(T) - np.sqrt(m(T)**2 - self.gamma * T * self.M0**2)) / T

        def f3(dens, T):
            # Note: Missprint in Lowrie & Rauenzahn 2007, eq. 24:
            # rho has to be replaced by 1!
            return (6. * dens**2 * (T - 1.) / (self.gamma - 1.) +
                    3. * (1. - dens**2) * self.M0**2 +
                    8. * self.P0 * (T**4 - 1) * dens)

        def func(T, x):

            dens = rho(T)
            return self.M0 * f3(dens, T) / (24. * self.kappa * dens**2 * T**3)

        Tpr = scipy.integrate.odeint(func, self.T1, xpr)
        rhopr = rho(Tpr)
        vpr = self.M0 / rhopr

        return xpr, rhopr, Tpr, vpr

    def sample_shock(self, xmin=-0.25, xmax=0.25, N=2000):

        xpr = np.linspace(0, xmin, N)
        xrel = np.linspace(0, xmax, 2)

        xpr, rhopr, Tpr, vpr = self.precursor_region(xpr)

        Mpr = vpr[0] / np.sqrt(Tpr[0])
        if Mpr > 1.:
            rhos = (rhopr[0] * (self.gamma + 1.) * Mpr**2 /
                    (2. + (self.gamma - 1.) * Mpr**2))
            vs = self.M0 / rhos
            Ts = (Tpr[0] * (1. - self.gamma + 2. * self.gamma * Mpr**2) *
                  (2. + (self.gamma - 1.) * Mpr**2) /
                  ((self.gamma + 1.)**2 * Mpr**2))
        else:
            rhos = None
            vs = None
            Ts = None

        x = np.append(xpr[::-1], xrel)
        rho = np.append(rhopr[::-1], np.ones(2) * self.rho1)
        T = np.append(Tpr[::-1], np.ones(2) * self.T1)
        v = np.append(vpr[::-1], np.ones(2) * self.v1)
        M = v / np.sqrt(T)

        return x, v, M, rho, T, rhos, Ts, vs
