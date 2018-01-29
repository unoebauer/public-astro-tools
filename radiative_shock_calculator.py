#!/usr/bin/env python
import collections
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

        self._check_shock_entropy_condition()
        self._calculate_shock_jump()

    def _check_shock_entropy_condition(self):
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

    def _calculate_shock_jump(self):
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
            # LR07, Eq. (12)
            y[0] = ((f1(T) + np.sqrt(f1(T)**2 + f2(T))) /
                    (6. * (self.gamma - 1.) * T) - rho)
            # LR07, Eq. (13)
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
        """
        """
        super(eqdiff_shock_calculator, self).__init__(M0, P0, kappa,
                                                      gamma=gamma)
        # Zel'dovich spike
        self.Ts = None
        self.rhos = None
        self.vs = None

    def _precursor_region(self, xpr):
        """Calculate shock state in the precursor region

        The precursor region is at x < 0.

        Parameters
        ----------
        xpr : float or np.ndarray
            dimensionless coordinate (negative per definition)

        Returns
        -------
        precursor : collections.namedtuple
            precursor state, containing dimensionless position, density,
            temperature and velocity
        """

        # LE07, Eq. (26)
        def m(T):
            return (0.5 * (self.gamma * self.M0**2 + 1.) +
                    self.gamma * self.P0 / 6. * (1. - T**4))

        # LE07, Eq. (25)
        def rho(T):
            return (m(T) - np.sqrt(m(T)**2 - self.gamma * T * self.M0**2)) / T

        # auxiliary function for LE07, Eq. (24)
        def f3(dens, T):
            return (6. * dens**2 * (T - 1.) / (self.gamma - 1.) +
                    3. * (1. - dens**2) * self.M0**2 +
                    8. * self.P0 * (T**4 - dens) * dens)

        # LE07, Eq. (24)
        def func(T, x):

            dens = rho(T)
            return self.M0 * f3(dens, T) / (24. * self.kappa * dens**2 * T**3)

        Tpr = scipy.integrate.odeint(func, self.T1, xpr)
        rhopr = rho(Tpr)
        # LE07, Eq. (21)
        vpr = self.M0 / rhopr

        result = collections.namedtuple('precursor',
                                        ['x', 'rho', 'T', 'v'])
        precursor = result(x=xpr, rho=rhopr, T=Tpr, v=vpr)

        return precursor

    def sample_shock(self, xmin=-0.25, xmax=0.25, Nsamples=1024):
        """Sample the shock structure.

        This routine also checks for the existence of a Zel'dovich spike and
        sets the attributes rhos, vs, Ts.

        Parameters
        ----------
        xmin : float
            dimensionless left limit of domain (default -0.25)
        xmax : float
            dimensionless right limit of domain (default 0.25)
        Nsamples : int
            number of equidistant points which sample the precursor region
            (between xmin and 0)

        Returns
        -------
        shock : collections.namedtuple
            shock structure containing the dimensionless position, density,
            temperature, velocity, Mach number
        """

        assert(xmin < 0)
        assert(xmax > 0)

        # Precursor region
        xpr = np.linspace(0, xmin, Nsamples)

        # Relaxation region; only two points necessary since
        xrel = np.linspace(0, xmax, 2)

        xpr, rhopr, Tpr, vpr = self._precursor_region(xpr)

        # Mach number of p-state; see LR07, Eq. (27d)
        Mpr = vpr[0] / np.sqrt(Tpr[0])

        # checking for the presence of a Zel'dovich spike
        if Mpr > 1.:

            # Zel'dovich spike density; see LR07, Eq. (27a)
            self.rhos = (rhopr[0] * (self.gamma + 1.) * Mpr**2 /
                         (2. + (self.gamma - 1.) * Mpr**2))

            # Zel'dovich spike velocity; see LR07, Eq. (27b)
            self.vs = self.M0 / self.rhos
            # Zel'dovich spike temperature; see LR07, Eq. (27c)
            self.Ts = (Tpr[0] * (1. - self.gamma + 2. * self.gamma * Mpr**2) *
                       (2. + (self.gamma - 1.) * Mpr**2) /
                       ((self.gamma + 1.)**2 * Mpr**2))

        x = np.append(xpr[::-1], xrel)
        rho = np.append(rhopr[::-1], np.ones(2) * self.rho1)
        T = np.append(Tpr[::-1], np.ones(2) * self.T1)
        v = np.append(vpr[::-1], np.ones(2) * self.v1)
        M = v / np.sqrt(T)

        result = collections.namedtuple('shock', ['x', 'rho', 'T', 'v', 'M'])
        shock = result(x=x, rho=rho, T=T, v=v, M=M)

        return shock


class noneqdiff_shock_calculator(shock_base):

    def __init__(self, M0, P0, kappa, sigma, gamma=5./3.):
        super(noneqdiff_shock_calculator, self).__init__(M0, P0, kappa * P0,
                                                         gamma=gamma)
        self.sigma = sigma
        self.kappa = kappa

        # epsilon used in LE08, Eqs. (42), (43) and (44)
        self.eps = 1e-6

        # epsilon use in steps 2 and 3 of the LE08 solution strategy
        self.epsasp = 1e-3

        # LE08 Eqs. (16) and (18)
        self.Cp = 1. / (self.gamma - 1.)
        self.Km = 3. * (self.gamma * self.M0**2 + 1.) + self.gamma * self.P0

    def dthetadx(self, v, rho, T, theta):
        # LE08, Eq. (18)
        return (v * (6. * self.Cp * rho * (T - 1.) +
                     3. * rho * (v**2 - self.M0**2) +
                     8. * self.P0 * (theta**4 - rho)) /
                (24. * self.P0 * self.kappa * theta**3))

    def theta_fTM(self, T, rho):
        # LE08, Eq. (41)
        return np.sqrt(np.sqrt(1. / self.gamma / self.P0 *
                               (self.Km - 3. * self.gamma * self.M0**2 / rho -
                                3. * T * rho)))

    def determine_epsilon_state(self, domain="zero"):
        assert(domain in ["zero", "one"])

        eps = self.eps
        epsasp = self.epsasp

        if domain == "zero":
            # precursor region
            root = -1

            # set initial values to state 0
            rho = 1.
            T = 1.
            M = self.M0
            v = M
        else:
            # relaxation region
            root = 1
            eps = -eps
            epsasp = -epsasp

            # set initial values to state 1
            rho = self.rho1
            T = self.T1
            v = self.v1
            M = v / np.sqrt(T)

        # radiation temperature equals gas temperature
        theta = T

        # b and d of quadratic equation, LE08 Eq. (17)
        b = self.Km - self.gamma * self.P0 * theta**4
        d = np.sqrt(b**2 - 36. * self.gamma * self.M0**2 * T)

        # root of LE08 Eqs. (61) and (62) the same as in (17),
        # which in relaxation region is problem-dependent
        if domain == "one":
            rhop = (b + root * d) / (6. * T)
            rhom = (b - root * d) / (6. * T)
            if np.fabs(rhom - rho) < np.fabs(rhop - rho):
                print("changing sign of root")
                root = -root

        # LE08 Eq. (61)
        drhodT = -1. / T * (rho + root * 3. * self.gamma * self.M0**2 / d)
        # LE08 Eq. (62)
        drhodtheta = (-2. / 3. * self.P0 * self.gamma * theta**3 / T *
                      (1. + root *
                       (self.Km - self.gamma * self.P0 * theta**4) / d))
        # LE08 Eq. (59)
        c1 = self.M0 / (24. * self.P0 * self.kappa * rho**2 * theta**3)
        # LE08 Eq. (60)
        c2 = self.P0 / (3. * self.Cp * self.M0 * (M**2 - 1.))
        # LE08 Eq. (55)
        dGdT = c1 * (6. * self.Cp * rho * (2. * drhodT * (T - 1.) + rho) -
                     6. * self.M0**2 * rho * drhodT +
                     8. * self.P0 * (drhodT * (theta**4 - 2. * rho)))
        # LE08 Eq. (56)
        dGdtheta = c1 * (12. * self.Cp * drhodtheta * rho * (T - 1.) -
                         6. * self.M0**2 * rho * drhodtheta +
                         8. * self.P0 * (drhodtheta * (theta**4 - 2. * rho) +
                                         4. * rho * theta**3))
        # LE08 Eq. (57)
        dFdT = c2 * (4. * v * theta**3 * dGdT -
                     12. * self.sigma * (self.gamma * M**2 - 1.) * T**3)
        # LE08 Eq. (58)
        dFdtheta = c2 * (4. * v * theta**3 * dGdtheta +
                         12. * self.sigma * (self.gamma * M**2 - 1.) *
                         theta**3)

        # Solving LE08 Eq. (54): difficulty here is the selection of the root.
        # The root which produces drho/dtheta > 0 is selected.
        # Trial an error to decide
        root2 = root
        dTdtheta = ((dFdT - dGdtheta - root2 * np.sqrt(
            (dFdT - dGdtheta)**2 + 4. * dGdT * dFdtheta)) / (2. * dGdT))
        if (drhodT * dTdtheta + drhodtheta) <= 0:
            print("changing sign of root2")
            root2 = - root2
            dTdtheta = (dFdT - dGdtheta - root2 * np.sqrt(
                (dFdT - dGdtheta)**2 + 4. * dGdT * dFdtheta)) / (2. * dGdT)

        # determine epsilon state according to LE08, Eqs. (42), (43) and (44)
        # LE08, Eq. (42)
        thetaeps = theta + eps
        # LE08, Eq. (43)
        Teps = T + eps * dTdtheta
        # LE08, Eq. (17) for epsilon state
        rhoeps = (self.Km - self.gamma * self.P0 * thetaeps**4 + root *
                  np.sqrt((self.Km - self.gamma * self.P0 * thetaeps**4)**2 -
                          36. * self.gamma * self.M0**2 * Teps)) / (6. * Teps)
        veps = self.M0 / rhoeps
        Meps = veps / np.sqrt(Teps)

        # LE08, Eq. (18) for epsilon state
        dthetaepsdx = self.dthetadx(veps, rhoeps, Teps, thetaeps)
        # LE08, Eq. (44)
        x0 = -eps / dthetaepsdx

        # LE08 equation system (37), (38)
        def func(y, M):

            x = y[0]
            T = y[1]

            v = M * np.sqrt(T)
            rho = self.M0 / (np.sqrt(T) * M)
            theta = self.theta_fTM(T, rho)

            tmp = self.dthetadx(v, rho, T, theta)
            # LE08, Eq. (25)
            r = 3. * rho * self.sigma * (theta**4 - T**4)
            # LE08, Eq. (39)
            ZD = (4. * self.M0 * theta**3 * tmp +
                  (self.gamma - 1.) / (self.gamma + 1.) *
                  (self.gamma * M**2 + 1.) * r)
            # LE08, Eq. (24)
            ZN = 4. * self.M0 * theta**3 * tmp + (self.gamma * M**2 - 1.) * r

            # LE08, Eq. (37)
            dxdM = (-6. * self.M0 * rho * T /
                    ((self.gamma + 1.) * self.P0 * M) * ((M**2 - 1.) / ZD))
            # LE08, Eq. (38)
            dTdM = (-2. * (self.gamma - 1.) / (self.gamma + 1.) *
                    T * ZN / (M * ZD))

            return np.array([dxdM, dTdM])

        # integration runs over M
        Minteg = np.linspace(Meps, 1. + epsasp, 1000)
        res = scipy.integrate.odeint(func, np.array([x0, Teps]), Minteg)

        # determine remaining state parameters of relaxation/precursor region
        xprerel = res[:, 0]
        Mprerel = Minteg
        Tprerel = res[:, 1]
        rhoprerel = self.M0 / Mprerel / np.sqrt(Tprerel)
        thetaprerel = self.theta_fTM(Tprerel, rhoprerel)
        vprerel = Mprerel * np.sqrt(Tprerel)

        # necessary in case of a continuous shock
        dxdMprerel = func([xprerel[-1], Tprerel[-1]], Minteg[-1])[0]

        return (xprerel, vprerel, Mprerel, rhoprerel, Tprerel, thetaprerel,
                dxdMprerel)

    def connect_precursor_relaxation_domains(self, diagnostic_plots=False):

        xpre, vpre, Mpre, rhopre, Tpre, thetapre, dxdMpre = \
            self.determine_epsilon_state(domain="zero")
        xrel, vrel, Mrel, rhorel, Trel, thetarel, dxdMrel = \
            self.determine_epsilon_state(domain="one")

        ppre = rhopre * Tpre / self.gamma
        prel = rhorel * Trel / self.gamma

        if thetapre[-1] > thetarel[-1]:
            # precursor and relaxation region are connected by a shock
            print("Discontinuous case")

            # both regions have to be translated so that theta is continuous
            # and the density fulfils the hydrodynamic jump conditions
            # see e.g. Clarke & Carswell 2007 for the density jump condition
            loc = []
            dtheta = []
            rhojump = []

            # for now, matching is achieved by a brute force approach -
            # not elegant but works
            for i in range(len(xpre)-1, 100, -1):
                for j in range(len(xrel)-1, 100, -1):
                    loc.append((i, j))

                    # deviation from continuity of radiative temperature
                    dtheta.append((thetapre[i] - thetarel[j]))
                    # deviation from density jump condition
                    rhojump.append((rhorel[j] / rhopre[i] -
                                    ((self.gamma + 1.) * prel[j] +
                                     (self.gamma - 1.) * ppre[i]) /
                                    ((self.gamma + 1.) * ppre[i] +
                                     (self.gamma - 1.) * prel[j])))

            # index of location of hydrodynamic shock in precursor
            # and relaxation region
            i = np.argmin(np.fabs(dtheta) / np.max(np.fabs(dtheta)) +
                          np.fabs(rhojump) / np.max(np.fabs(rhojump)))

            ipre = loc[i][0]
            irel = loc[i][1]

            # determine translation offset
            dxpre = -xpre[ipre]
            dxrel = -xrel[irel]

            x = np.append((xpre + dxpre)[:ipre], ((xrel + dxrel)[:irel])[::-1])
            theta = np.append(thetapre[:ipre], (thetarel[:irel])[::-1])
            T = np.append(Tpre[:ipre], (Trel[:irel])[::-1])
            rho = np.append(rhopre[:ipre], (rhorel[:irel])[::-1])
            M = np.append(Mpre[:ipre], (Mrel[:irel])[::-1])
            v = np.append(vpre[:ipre], (vrel[:irel])[::-1])

        else:
            # precursor and relaxation region connect smoothly
            print("Continuous Case")

            # determine endpoints of relaxation and precursor region according
            # to LE08 Eq. (65)
            xl = self.epsasp * dxdMpre
            xr = -self.epsasp * dxdMrel

            # determine translation offset
            dxpre = xl - xpre[-1]
            dxrel = xr - xrel[-1]

            x = np.append((xpre + dxpre), ((xrel + dxrel))[::-1])
            theta = np.append(thetapre, (thetarel)[::-1])
            T = np.append(Tpre, (Trel)[::-1])
            rho = np.append(rhopre, (rhorel)[::-1])
            M = np.append(Mpre, (Mrel)[::-1])
            v = np.append(vpre, (vrel)[::-1])

        return x, v, M, rho, T, theta

    def sample_shock(self, xmin=-0.25, xmax=0.25):

        x, v, M, rho, T, theta = \
            self.connect_precursor_relaxation_domains()

        # if necessary attach state 0 at the left
        if xmin < np.min(x):
            np.insert(x, 0, xmin)
            np.insert(v, 0, self.M0)
            np.insert(M, 0, self.M0)
            np.insert(rho, 0, 1)
            np.insert(T, 0, 1)
            np.insert(theta, 0, 1)

        # if necessary attach state 1 at the right
        if xmax > np.max(x):
            x = np.append(x, xmax)
            v = np.append(v, self.v1)
            M = np.append(M, self.v1 / np.sqrt(self.T1))
            rho = np.append(rho, self.rho1)
            T = np.append(T, self.T1)
            theta = np.append(theta, self.T1)

        return x, v, M, rho, T, theta

    def save_shock_data(self, fname, xmin=-0.25, xmax=0.25):

        x, v, M, rho, T, theta = self.sample_shock(xmin=xmin, xmax=xmax)

        np.savetxt(fname, np.array([x, v, M, rho, T, theta]).T)

        return x, v, M, rho, T, theta
