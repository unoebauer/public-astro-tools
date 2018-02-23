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
            radiation diffusivity (definition according to LR07!)
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
        self.M1 = self.v1 / np.sqrt(self.T1)


class eqdiff_shock_calculator(shock_base):
    def __init__(self, M0, P0, kappa, gamma=5./3.):
        """Shock structure calculator based on equilibrium diffusion


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
    def __init__(self, M0, P0, kappa, sigma, gamma=5./3.,
                 eps=1e-6, epsasp=1e-6, Msamples=1024, epsrel=None,
                 epsasprel=None, matching_mode="theta_rhojump"):
        """Shock structure calculator based on non-equilibrium diffusion

        With this tool, the structure of steady radiative shocks can be
        calculated using the solution strategy described by LE08.

        Parameters
        ----------
        M0 : float
            downstream Mach number
        P0 : float
            non-dimensional constant (see LE08, Eq. 4); combination of
            reference values used in the non-dimensionalization process.
        kappa : float
            radiation diffusivity (definition according to LE08!)
        sigma : float
            total dimensionless cross section
        gamma : float
            adiabatic index (default 5/3)
        eps : float
            numerical parameter for the solution strategy; defines epsilon
            state which is used to avoid zero derivatives in limiting down- and
            upstream states (see LE08, Sec. 5); may have to be increased in
            order to avoid numerical stability problems in case of large Mach
            numbers that lead to continuous solutions (default 1e-6).
        epsasp : float
            numerical parameter for the solution strategy; used to avoid the
            singularity at the adiabatic sonic point (see LE08, Sec. 5 and
            Appendix B); should be small; with larger values, artefacts at the
            interface between the precursor and relaxation region become more
            pronounced (default 1e-6)
        Msamples : int
            numerical parameter for the solution strategy; set the number of
            integration points used in the determination of the precursor and
            relaxation region (default 1024).
        """
        super(noneqdiff_shock_calculator, self).__init__(M0, P0, kappa * P0,
                                                         gamma=gamma)

        self.matching_mode = matching_mode
        assert(matching_mode in ["theta_rhojump", "theta_only"])

        self.odeint_mode = "odeint"

        self.sigma = sigma
        # Reset kappa - this is necessary since the definition of kappa in LR07
        # and LE08 differs by a factor of P0
        self.kappa = kappa

        self.Msamples = Msamples

        # epsilon used in LE08, Eqs. (42), (43) and (44)
        self.eps = eps

        # epsilon use in steps 2 and 3 of the LE08 solution strategy
        self.epsasp = epsasp

        if epsrel is None:
            epsrel = eps
        if epsasprel is None:
            epsasprel = epsasp

        self.epsrel = epsrel
        self.epsasprel = epsasprel

        # LE08 Eqs. (16) and (18)
        self.Cp = 1. / (self.gamma - 1.)
        self.Km = 3. * (self.gamma * self.M0**2 + 1.) + self.gamma * self.P0

    def _dthetadx(self, v, rho, T, theta):
        """spatial derivative of radiation temperature

        Parameters
        ----------
        v : float or np.ndarray
            dimensionless velocity
        rho : float or np.ndarray
            dimensionless density
        T : float or np.ndarray
            dimensionless temperature
        theta : float or np.ndarray
            dimensionless radiation temperature

        Returns
        -------
        dthetadx : float or np.ndarray
            dimensionless spatial derivative of radiation temperature
        """
        # LE08, Eq. (18)
        return (v * (6. * self.Cp * rho * (T - 1.) +
                     3. * rho * (v**2 - self.M0**2) +
                     8. * self.P0 * (theta**4 - rho)) /
                (24. * self.P0 * self.kappa * theta**3))

    def _theta_fTM(self, T, rho):
        """radiation temperature as a function of temperature and density

        WARNING: at large Mach numbers and small values for eps, numerical
        inaccuracies may lead to negative values in the root and a breakdown of
        the solution strategy. Increasing the value for eps has typically
        helped to avoid this problem.

        Parameters
        ----------
        T : float or np.ndarray
            dimensionless temperature
        rho : float or np.ndarray
            dimensionless density

        Returns
        -------
        theta : float or np.ndarray
            dimensionless radiation temperature
        """
        # LE08, Eq. (41)
        tmp = (1. / self.gamma / self.P0 *
               (self.Km - 3. * self.gamma * self.M0**2 / rho -
                3. * T * rho))
        return np.sqrt(np.sqrt(tmp))

    def _solve_precursor_relaxation_region(self, domain="zero"):
        """Solves shock structure in precursor or relaxation region

        This routine preforms the bulk of the work. It determines first the
        epsilon state and then performs a numerical integration over the Mach
        number from the epsilon state towards the adiabatic sonic point.

        Parameters
        ----------
        domain : str
            either 'zero' or 'one'; determines whether the precursor or the
            relaxation region is treated (default 'zero')

        Returns
        -------
        precursor_relaxation : namedtuple
            physical state in the precursor or relaxation region in terms of
            the dimensionless location, velocity, Mach number, density,
            temperature, radiation temperature, derivative of the position
            with respect to the Mach number.
        """
        assert(domain in ["zero", "one"])

        if domain == "zero":
            # precursor region
            print("Precursor region")
            eps = self.eps
            epsasp = self.epsasp
            root = -1

            # set initial values to state 0
            rho = 1.
            T = 1.
            M = self.M0
            v = M
        else:
            # relaxation region
            print("Relaxation region")
            root = 1
            eps = -self.epsrel
            epsasp = -self.epsasprel

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
        dthetaepsdx = self._dthetadx(veps, rhoeps, Teps, thetaeps)
        # LE08, Eq. (44)
        x0 = -eps / dthetaepsdx

        # LE08 equation system (37), (38)
        def func(y, M):

            # x = y[0]
            T = y[1]

            # LE08, Eq. (14)
            v = M * np.sqrt(T)
            # LE08, Eq. (13)
            rho = self.M0 / v
            theta = self._theta_fTM(T, rho)

            tmp = self._dthetadx(v, rho, T, theta)
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

        def func2(M, y):

            res = func(y, M)
            return res

        # integration runs over M
        Minteg = np.logspace(np.log10(Meps), np.log10(1. + epsasp),
                             self.Msamples)
        func0 = np.array([0, Teps])

        # Scipy uses lsoda per default; creates problems for Roth & Kasen 2015
        # M=70 shock; Potentially, other solvers may work better
        # TODO: Try scipy.integrate.ode tools
        if self.odeint_mode == "odeint":
            res = scipy.integrate.odeint(func, func0, Minteg)

        elif self.odeint_mode == "ode":
            r = scipy.integrate.ode(func2).set_integrator('lsoda')
            r.set_initial_value(func0, t=Minteg[0])

            res = np.zeros((len(Minteg), 2))
            res[0, :] = func0

            for i, Mi in enumerate(Minteg[1:]):
                res[i, :] = r.integrate(Mi)
                if not r.successful():
                    print("Error in Integration")
                    print("i ", i)
                    raise Exception
        else:
            raise ValueError("Unknown 'odeint_mode'")

        # determine remaining state parameters of relaxation/precursor region
        xprerel = res[:, 0]
        Mprerel = Minteg
        Tprerel = res[:, 1]

        rhoprerel = self.M0 / Mprerel / np.sqrt(Tprerel)
        thetaprerel = self._theta_fTM(Tprerel, rhoprerel)
        vprerel = Mprerel * np.sqrt(Tprerel)

        # insert 0 or 1 state at the beginning
        xprerel = np.insert(xprerel, 0, x0)
        Mprerel = np.insert(Mprerel, 0, M)
        Tprerel = np.insert(Tprerel, 0, T)
        rhoprerel = np.insert(rhoprerel, 0, rho)
        thetaprerel = np.insert(thetaprerel, 0, theta)
        vprerel = np.insert(vprerel, 0, v)
        pprerel = rhoprerel * Tprerel / self.gamma

        # necessary in case of a continuous shock
        dxdMprerel = func([xprerel[-1], Tprerel[-1]], Minteg[-1])[0]

        result = collections.namedtuple(
            'precursor_relaxation',
            ['x', 'v', 'M', 'rho', 'T', 'p', 'theta', 'dxdM'])
        precursor_relaxation = result(x=xprerel, v=vprerel, M=Mprerel,
                                      rho=rhoprerel, T=Tprerel, p=pprerel,
                                      theta=thetaprerel, dxdM=dxdMprerel)

        return precursor_relaxation

    def _solve_precursor_region(self):

        precursor = self._solve_precursor_relaxation_region(domain="zero")

        return precursor

    def _solve_relaxation_region(self):

        relaxation = self._solve_precursor_relaxation_region(domain="one")

        return relaxation

    def _connect_domains(self):
        """Connect the solutions in the precursor and the relaxation regions

        Depending on the radiation temperature around the adiabatic sonic
        point, the two regimes are connected via an embedded hydrodynamic shock
        or joined smoothly.

        Returns
        ------
        shock : namedtuple
            shock structure in terms of the dimensionless location, velocity,
            Mach number, density, temperature, radiation temperature.
        """

        precursor = self._solve_precursor_region()
        xpre, vpre, Mpre, rhopre, Tpre, ppre, thetapre, dxdMpre = precursor
        self.precursor = precursor

        # sanity check
        if np.isnan(xpre).sum() > 0:
            print("Problems in precursor: {:d} NaNs in x".format(
                np.isnan(xpre).sum()))

        relaxation = self._solve_relaxation_region()
        xrel, vrel, Mrel, rhorel, Trel, prel, thetarel, dxdMrel = relaxation
        self.relaxation = relaxation

        # sanity check
        if np.isnan(xrel).sum() > 0:
            print("Problems in relaxation: {:d} NaNs in x".format(
                np.isnan(xrel).sum()))

        ppre = rhopre * Tpre / self.gamma
        prel = rhorel * Trel / self.gamma

        # decide whether there is an embedded shock or not
        if thetapre[-1] > thetarel[-1]:
            # precursor and relaxation region are connected by a shock
            print("Discontinuous case: embedded hydrodynamic shock")
            self.case = "shock"

            # both regions have to be translated so that theta is continuous
            # and the density fulfils the hydrodynamic jump conditions
            # see e.g. Clarke & Carswell 2007 for the density jump condition

            if self.matching_mode == "theta_rhojump":
                x = rhorel
                y = (rhopre *
                     ((self.gamma + 1.) * prel + (self.gamma - 1.) * ppre) /
                     ((self.gamma + 1.) * ppre + (self.gamma - 1.) * prel))
                X, Y = np.meshgrid(x, y)

                xt = thetarel
                yt = thetapre
                Xt, Yt = np.meshgrid(xt, yt)

                imatch = np.unravel_index(np.argmin((np.fabs(X - Y) / X) +
                                                    (np.fabs(Xt - Yt) / Xt),
                                                    axis=None), Xt.shape)

                ipre = imatch[0]
                irel = imatch[1]

            elif self.matching_mode == "theta_only":
                # Quite some effort went into determining the best way to match
                # the precursor and relaxation region in the presence of a
                # shock; in the end it turned out that simply checking for the
                # where the relaxation branch theta intercepts the precursor
                # branch theta is the most reliable strategy:
                try:
                    imatch = np.where(
                        np.diff(np.sign(thetapre - thetarel)) != 0)[0][0] + 1
                except IndexError:
                    print("Error in matching precursor and relaxation branch:")
                    print("precursor and relaxation theta do not intercept")
                    print("Try reducing 'eps' and 'epsrel'")
                    print("Try increasing 'Msamples'")
                    raise

                ipre = imatch
                irel = imatch
            else:
                raise ValueError("Unknown matching_mode")

            # determine translation offset
            dxpre = -xpre[ipre]
            dxrel = -xrel[irel]

            self.Ms = Mrel[irel]

            # TODO: calculate deviation from hydrodynamic jump condition

            irel += 1
            ipre += 1

            x = np.append((xpre + dxpre)[:ipre],
                          ((xrel + dxrel)[:irel])[::-1])
            theta = np.append(thetapre[:ipre], (thetarel[:irel])[::-1])
            T = np.append(Tpre[:ipre], (Trel[:irel])[::-1])
            rho = np.append(rhopre[:ipre], (rhorel[:irel])[::-1])
            M = np.append(Mpre[:ipre], (Mrel[:irel])[::-1])
            v = np.append(vpre[:ipre], (vrel[:irel])[::-1])

        else:
            # precursor and relaxation region connect smoothly
            print("Continuous case: no embedded hydrodynamic shock")

            self.case = "continuous"
            self.Ms = 1.

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

        p = rho * T / self.gamma
        result = collections.namedtuple(
            'shock', ['x', 'v', 'M', 'rho', 'T', 'p', 'theta'])
        shock = result(x=x, v=v, M=M, rho=rho, T=T, p=p, theta=theta)

        return shock

    def calculate_shock_structure(self, xmin=None, xmax=None):
        """Calculate the shock structure

        This is one of the main routines which should be used to calculate the
        shock structure. The spatial gridding follows directly from the M
        integration and will not be equidistant. Also, the extent of the
        spatial domain is automatically set by the integration. If instead a
        uniform gridding is desired, the 'sample_shock' routine should be
        called.

        Returns
        -------
        shock : namedtuple
            shock structure in terms of the dimensionless location, velocity,
            Mach number, density, temperature, radiation temperature.
        """

        shock = self._connect_domains()
        x, v, M, rho, T, p, theta = shock
        if xmin is not None:
            if xmin < x[0]:
                x = np.insert(x, 0, xmin)
                v = np.insert(v, 0, self.M0)
                M = np.insert(M, 0, self.M0)
                rho = np.insert(rho, 0, 1.)
                T = np.insert(T, 0, 1.)
                theta = np.insert(theta, 0, 1.)
            else:
                indices = (x >= xmin)
                x = x[indices]
                v = v[indices]
                M = M[indices]
                rho = rho[indices]
                T = T[indices]
                theta = theta[indices]

        if xmax is not None:
            if xmax > x[-1]:
                x = np.append(x, xmax)
                v = np.append(v, self.v1)
                M = np.append(M, self.M1)
                rho = np.append(rho, self.rho1)
                T = np.append(T, self.T1)
                theta = np.append(theta, self.T1)
            else:
                indices = (x <= xmax)
                x = x[indices]
                v = v[indices]
                M = M[indices]
                rho = rho[indices]
                T = T[indices]
                theta = theta[indices]

        p = rho * T / self.gamma

        result = collections.namedtuple(
            'shock', ['x', 'v', 'M', 'rho', 'T', 'p', 'theta'])
        shock = result(x=x, v=v, M=M, rho=rho, T=T, p=p, theta=theta)
        return shock
