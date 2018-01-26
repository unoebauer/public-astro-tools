#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as c
import constants as csts
import scipy.optimize
import scipy.integrate
"""

References
----------
    LR07: Lowrie & Rauenzahn 2007, Shock Waves (2007) 16:445–453
            DOI 10.1007/s00193-007-0081-2
    LE08: Lowrie & Edwards 2008, Shock Waves (2008) 18:129–143
            DOI 10.1007/s00193-008-0143-0
"""


a_rad = 4. * c.sigma_sb.cgs.value / c.c.cgs.value


class shock_calculator(object):
    def __init__(self):

        self.P0 = None
        self.M0 = None
        self.kappa = None

    def set_values(self, rho, u, T, L, gamma, kappa):

        self.gamma = gamma
        self.rhotilde = rho
        self.Ttilde = T
        self.Ltilde = L

        p = self.rhotilde / csts.CGS_UNIFIED_ATOMIC_MASS * csts.CGS_BOLTZMANN_CONSTANT * self.Ttilde
        self.atilde = np.sqrt(self.gamma * self.p / self.rhotilde)

        self.P0 = a_rad * self.Ttilde**4 / self.rhotilde / self.atilde**2
        self.kappa = a_rad * self.Ttilde**4 * csts.CGS_SPEED_OF_LIGHT / 3. / kappa / self.Ltilde / self.rhotilde / self.atilde**3

        self.M0 = self.u / self.atilde

        self.check_M0_consistency()

    def set_values_nondim(self, gamma, M0, P0, kappa):

        self.P0 = P0
        self.M0 = M0
        self.gamma = gamma
        self.kappa = kappa

        self.check_M0_consistency()

    def check_M0_consistency(self):

        a0star = np.sqrt(1. + 4. / 9. * self.P0 * (self.gamma - 1.) *
                         (4. * self.P0 * self.gamma +
                          3. * (5. - 3. * self.gamma)) /
                         (1. + 4. * self.P0 * self.gamma * (self.gamma - 1.)))

        if self.M0 < a0star:
            raise ValueError("Violation of Entropy Condition")

    def determine_state_one(self):
        """Calculate overall shock jump"""
        # TODO: rename routine to something more meaningful

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

class eqdiff_shock_calculator(shock_calculator):
    def __init__(self):
        super(eqdiff_shock_calculator, self).__init__()

    def precursor_region(self, xpr):

        self.determine_state_one()

        m = lambda T: 0.5 * (self.gamma * self.M0**2 + 1.) + self.gamma * self.P0 / 6. * (1. - T**4)
        rho = lambda T: (m(T) - np.sqrt(m(T)**2 - self.gamma * T * self.M0**2)) / T
        #Note: Missprint in Lowrie & Rauenzahn 2007, eq. 24: rho has to be replaced by 1!
        f3 = lambda dens, T: 6. * dens**2 * (T - 1.) / (self.gamma - 1.) + 3. * (1. - dens**2) * self.M0**2 + 8. * self.P0 * (T**4 - 1) * dens

        def func(T, x):

            dens = rho(T)
            return self.M0 * f3(dens, T) / (24. * self.kappa * dens**2 * T**3)

        Tpr = scipy.integrate.odeint(func, self.T1, xpr)
        rhopr = rho(Tpr)
        vpr = self.M0 / rhopr

        return xpr, rhopr, Tpr, vpr

    def sample_shock(self, xmin = -0.25, xmax = 0.25, N = 2000):


        xpr = np.linspace(0, xmin, N)
        xrel = np.linspace(0, xmax, 2)

        xpr, rhopr, Tpr, vpr = self.precursor_region(xpr)

        Mpr = vpr[0] / np.sqrt(Tpr[0])
        if Mpr > 1.:
            rhos = rhopr[0] * (self.gamma + 1.) * Mpr**2 / (2. + (self.gamma - 1.) * Mpr**2)
            vs = self.M0 / rhos
            Ts = Tpr[0] * (1. - self.gamma + 2. * self.gamma * Mpr**2) * (2. + (self.gamma - 1.) * Mpr**2) / ((self.gamma + 1.)**2 * Mpr**2)
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


class noneqdiff_shock_calculator(shock_calculator):
    def __init__(self):
        super(noneqdiff_shock_calculator, self).__init__()

        #epsilon used in eqs. 42, 43 and 44 of LE08
        self.eps = 1e-6

        #epsilon use in steps 2 and 3 of the solution strategy of LE08
        self.epsasp = 1e-3


    def set_values_nondim(self, gamma, M0, P0, kappa, sigma):
        super(noneqdiff_shock_calculator, self).set_values_nondim(gamma, M0, P0, kappa)

        self.sigma = sigma

        #see LE08 eqs 16 and 18
        self.Cp = 1. / (self.gamma - 1.)
        self.Km = 3. * (self.gamma * self.M0**2 + 1.) + self.gamma * self.P0


    def dthetadx(self, v, rho, T, theta):
        #LE08, eq. 18: Misprint in paper, last rho should be replaced by 1
        return v * (6. * self.Cp * rho * (T - 1.) + 3. * rho * (v**2 - self.M0**2) + 8. * self.P0 * (theta**4 - 1.)) / (24. * self.P0 * self.kappa * theta**3)

    def theta_fTM(self, T, rho):
        #LE08, eq. 41
        return np.sqrt(np.sqrt(1. / self.gamma / self.P0 * (self.Km - 3. * self.gamma * self.M0**2 / rho - 3. * T * rho)))

    def determine_epsilon_state(self, domain = "zero"):
        assert(domain in ["zero", "one"])

        eps = self.eps
        epsasp = self.epsasp

        if domain is "zero":
            #precursor region
            root = -1

            #set initial values to state 0
            rho = 1.
            T = 1.
            M = self.M0
            v = M

        else:
            #relaxation region
            root = 1
            eps = -eps
            epsasp = -epsasp

            #set initial values to state 1
            rho = self.rho1
            T = self.T1
            v = self.v1
            M = v / np.sqrt(T)

        #radiation temperature equals gas temperature
        theta = T

        #b and d of quadratic equation, LE08 eq. 17
        b = self.Km - self.gamma * self.P0 * theta**4
        d = np.sqrt(b**2 - 36. * self.gamma * self.M0**2 * T)

        #root of LE08 eqs 61 and 62 the same as in 17, which in relaxation region is problem-dependent
        if domain is "one":
            rhop = (b + root * d) / (6. * T)
            rhom = (b - root * d) / (6. * T)
            if np.fabs(rhom - rho) < np.fabs(rhop - rho):
                print("changing sign of root")
                root = -root

        #LE08 eqs. 55 - 62
        drhodT = -1. / T * (rho + root * 3. * self.gamma * self.M0**2 / d)
        drhodtheta = -2. / 3. * self.P0 * self.gamma * theta**3 / T * (1. + root * (self.Km - self.gamma * self.P0 * theta**4) / d)
        c1 = self.M0 / (24. * self.P0 * self.kappa * rho**2 * theta**3)
        c2 = self.P0 / (3. * self.Cp * self.M0 * (M**2 - 1.))
        dGdT = c1 * (6. * self.Cp * rho * (2. * drhodT * (T - 1.) + rho) - 6. * self.M0**2 * rho * drhodT + 8. * self.P0 * (drhodT * (theta**4 -2. * rho)))
        dGdtheta = c1 * (12. * self.Cp * drhodtheta * rho * (T - 1.) - 6. * self.M0**2 *rho * drhodtheta + 8. * self.P0 * (drhodtheta * (theta**4 - 2. * rho) + 4. * rho * theta**3))
        dFdT = c2 * (4. * v * theta**3 * dGdT - 12. * self.sigma * (self.gamma * M**2 - 1.) * T**3)
        dFdtheta = c2 * (4. * v * theta**3 * dGdtheta + 12. * self.sigma * (self.gamma * M**2 - 1.) * theta**3)

        #determination of LE08 eq 54: difficulty here is the selection of the root. The root which produces drho/dtheta > 0 is selected. Trial an error to decide
        root2 = root
        dTdtheta = (dFdT - dGdtheta - root2 * np.sqrt((dFdT - dGdtheta)**2 + 4. * dGdT * dFdtheta)) / (2. * dGdT)
        if (drhodT * dTdtheta + drhodtheta) <= 0:
            print("changing sign of root2")
            root2 = - root2
            dTdtheta = (dFdT - dGdtheta - root2 * np.sqrt((dFdT - dGdtheta)**2 + 4. * dGdT * dFdtheta)) / (2. * dGdT)


        #determine epsilon state according to LE08, eqs 42, 43 and 44
        thetaeps = theta + eps
        Teps = T + eps * dTdtheta
        rhoeps = (self.Km - self.gamma * self.P0 * thetaeps**4 + root * np.sqrt((self.Km - self.gamma * self.P0 * thetaeps**4)**2 - 36. * self.gamma * self.M0**2 * Teps)) / (6. * Teps)
        veps = self.M0 / rhoeps
        Meps = veps / np.sqrt(Teps)
        dthetaepsdx = self.dthetadx(veps, rhoeps, Teps, thetaeps)
        x0 = -eps / dthetaepsdx


        #LE08 equation system 37, 38
        def func(y, M):

            x = y[0]
            T = y[1]

            v = M * np.sqrt(T)
            rho = self.M0 / (np.sqrt(T) * M)
            theta = self.theta_fTM(T, rho)

            tmp = self.dthetadx(v, rho, T, theta)
            r = 3. * rho * self.sigma * (theta**4 - T**4)
            ZD = 4. * self.M0 * theta**3 * tmp + (self.gamma - 1.) / (self.gamma + 1.) * (self.gamma * M**2 + 1.) * r
            ZN = 4. * self.M0 * theta**3 * tmp + (self.gamma * M**2 - 1.) * r

            dxdM = -6. * self.M0 * rho * T / ((self.gamma + 1.) * self.P0 * M) * ((M**2 - 1.) / ZD)
            dTdM = -2. * (self.gamma - 1.) / (self.gamma + 1.) * T * ZN / (M * ZD)

            return np.array([dxdM, dTdM])


        #integration runs over M
        Minteg = np.linspace(Meps, 1. + epsasp, 1000)
        res = scipy.integrate.odeint(func, np.array([x0, Teps]), Minteg)

        #determine remaining state parameters of relaxation/precursor region
        xprerel = res[:,0]
        Mprerel = Minteg
        Tprerel = res[:,1]
        rhoprerel = self.M0 / Mprerel / np.sqrt(Tprerel)
        thetaprerel = self.theta_fTM(Tprerel, rhoprerel)
        vprerel = Mprerel * np.sqrt(Tprerel)

        #necessary in case of a continuous shock
        dxdMprerel = func([xprerel[-1], Tprerel[-1]], Minteg[-1])[0]

        return xprerel, vprerel, Mprerel, rhoprerel, Tprerel, thetaprerel, dxdMprerel

    def connect_precursor_relaxation_domains(self, diagnostic_plots = False):
        self.determine_state_one()

        xpre, vpre, Mpre, rhopre, Tpre, thetapre, dxdMpre = self.determine_epsilon_state(domain = "zero")
        xrel, vrel, Mrel, rhorel, Trel, thetarel, dxdMrel = self.determine_epsilon_state(domain = "one")

        ppre = rhopre * Tpre / self.gamma
        prel = rhorel * Trel / self.gamma

        if thetapre[-1] > thetarel[-1]:
            #precursor and relaxation region are connected by a shock
            print("Discontinuous case")

            #both regions have to be translated so that theta is continuous and the density fulfils the hydrodynamic jump conditions
            #see e.g. Clarke & Carswell 2007 for the density jump condition
            loc = []
            dtheta = []
            rhojump = []

            #for now, matching is achieved by a brute force approach - not elegant but works
            for i in xrange(len(xpre)-1, 100, -1):
                for j in xrange(len(xrel)-1, 100, -1):
                    loc.append((i,j))

                    #deviation from continuity of radiative temperature
                    dtheta.append((thetapre[i] - thetarel[j]))
                    #deviation from density jump condition
                    rhojump.append((rhorel[j] / rhopre[i] - ((self.gamma + 1.) * prel[j] + (self.gamma - 1.) * ppre[i]) / ((self.gamma + 1.) * ppre[i] + (self.gamma - 1.) * prel[j])))


            #index of location of hydrodynamic shock in precursor and relaxation region
            i = np.argmin(np.fabs(dtheta) / np.max(np.fabs(dtheta)) + np.fabs(rhojump) / np.max(np.fabs(rhojump)))


            if diagnostic_plots:
                plt.figure()
                plt.subplot(311)
                plt.plot(np.fabs(dtheta) / np.max(np.fabs(dtheta)))
                plt.axvline(i, ls = "dashed", color = "black")
                plt.subplot(312)
                plt.plot(np.fabs(rhojump) / np.max(np.fabs(rhojump)))
                plt.axvline(i, ls = "dashed", color = "black")
                plt.subplot(313)
                plt.plot(np.fabs(dtheta) / np.max(np.fabs(dtheta)) + np.fabs(rhojump) / np.max(np.fabs(rhojump)))
                plt.axvline(i, ls = "dashed", color = "black")

            ipre = loc[i][0]
            irel = loc[i][1]

            #determine translation offset
            dxpre = -xpre[ipre]
            dxrel = -xrel[irel]

            x = np.append((xpre + dxpre)[:ipre], ((xrel + dxrel)[:irel])[::-1])
            theta = np.append(thetapre[:ipre], (thetarel[:irel])[::-1])
            T = np.append(Tpre[:ipre], (Trel[:irel])[::-1])
            rho = np.append(rhopre[:ipre], (rhorel[:irel])[::-1])
            M = np.append(Mpre[:ipre], (Mrel[:irel])[::-1])
            v = np.append(vpre[:ipre], (vrel[:irel])[::-1])



        else:
            #precursor and relaxation region connect smoothly
            print("Continuous Case")


            #determine endpoints of relaxation and precursor region according to LE08 eq. 65
            xl = self.epsasp * dxdMpre
            xr = -self.epsasp * dxdMrel

            #determine translation offset
            dxpre = xl - xpre[-1]
            dxrel = xr - xrel[-1]

            x = np.append((xpre + dxpre), ((xrel + dxrel))[::-1])
            theta = np.append(thetapre, (thetarel)[::-1])
            T = np.append(Tpre, (Trel)[::-1])
            rho = np.append(rhopre, (rhorel)[::-1])
            M = np.append(Mpre, (Mrel)[::-1])
            v = np.append(vpre, (vrel)[::-1])


        if diagnostic_plots:
            plt.figure()
            plt.subplot(221)
            plt.plot(xpre + dxpre, Tpre, ls = "solid", color = "blue")
            plt.plot(xrel + dxrel, Trel, ls = "solid", color = "blue")
            plt.plot(x, T, ls = "solid", color = "black")
            plt.plot(xpre + dxpre, thetapre, ls = "dashed", color = "blue")
            plt.plot(xrel + dxrel, thetarel, ls = "dashed", color = "blue")
            plt.plot(x, theta, ls = "dashed", color = "black")
            plt.subplot(222)
            plt.plot(xpre + dxpre, rhopre, ls = "solid", color = "red")
            plt.plot(xrel + dxrel, rhorel, ls = "solid", color = "red")
            plt.plot(x, rho, ls = "solid", color = "black")
            plt.subplot(223)
            plt.plot(xpre + dxpre, vpre, ls = "solid", color = "green")
            plt.plot(xrel + dxrel, vrel, ls = "solid", color = "green")
            plt.plot(x, v, ls = "solid", color = "black")
            plt.subplot(224)
            plt.plot(xpre + dxpre, Mpre, ls = "solid", color = "cyan")
            plt.plot(xrel + dxrel, Mrel, ls = "solid", color = "cyan")
            plt.plot(x, M, ls = "solid", color = "black")

        return x, v, M, rho, T, theta

    def sample_shock(self, xmin = -0.25, xmax = 0.25, diagnostic_plots = False):

        x, v, M, rho, T, theta = self.connect_precursor_relaxation_domains(diagnostic_plots = diagnostic_plots)

        #if necessary attach state 0 at the left
        if xmin < np.min(x):
            np.insert(x, 0, xmin)
            np.insert(v, 0, self.M0)
            np.insert(M, 0, self.M0)
            np.insert(rho, 0, 1)
            np.insert(T, 0, 1)
            np.insert(theta, 0, 1)

        #if necessary attach state 1 at the right
        if xmax > np.max(x):
            x = np.append(x, xmax)
            v = np.append(v, self.v1)
            M = np.append(M, self.v1 / np.sqrt(self.T1))
            rho = np.append(rho, self.rho1)
            T = np.append(T, self.T1)
            theta = np.append(theta, self.T1)

        return x, v, M, rho, T, theta

    def save_shock_data(self, fname, xmin = -0.25, xmax = 0.25, diagnostic_plots = False):

        x, v, M, rho, T, theta = self.sample_shock(xmin = xmin, xmax = xmax, diagnostic_plots = diagnostic_plots)

        np.savetxt(fname, np.array([x, v, M, rho, T, theta]).T)

        return x, v, M, rho, T, theta






if __name__ == "__main__":

    gamma = 5./3.

    ## M = 1.05
    ## M = 1.2
    ## M = 1.4
    M = 2
    ## M = 3
    ## M = 4
    ## M = 5
    M = 40
    ## M = 70

    P0 = 1e-4
    kappa = 1.
    sigma = 1e6

    ## xmin = -0.01
    ## xmax = 0.01
    xmin = -0.25
    xmax = 0.05
    ## xmin = -0.3
    ## xmax = 0.05

    noneqdiff = noneqdiff_shock_calculator()
    eqdiff = eqdiff_shock_calculator()

    noneqdiff.set_values_nondim(gamma, M, P0, kappa, sigma)
    eqdiff.set_values_nondim(gamma, M, P0, kappa * P0)


    x_n, v_n, M_n, rho_n, T_n, theta_n = noneqdiff.save_shock_data('test.dat', xmin = xmin, xmax = xmax, diagnostic_plots = False)
    x_e, v_e, M_e, rho_e, T_e, rho_s, T_s, v_s = eqdiff.sample_shock(xmin = xmin, xmax = xmax)

    plt.figure()
    plt.subplot(221)
    plt.plot(x_n, T_n, ls = "solid", color = "blue")
    plt.plot(x_e, T_e, ls = "dashed", color = "blue")
    plt.plot(x_n, theta_n, ls = "dotted", color = "blue")
    plt.xlim([xmin, xmax])
    plt.subplot(222)
    plt.plot(x_n, rho_n, ls = "solid", color = "red")
    plt.plot(x_e, rho_e, ls = "dashed", color = "red")
    plt.xlim([xmin, xmax])
    plt.subplot(223)
    plt.plot(x_n, v_n, ls = "solid", color = "green")
    plt.plot(x_e, v_e, ls = "dashed", color = "green")
    plt.xlim([xmin, xmax])
    plt.subplot(224)
    plt.plot(x_n, M_n, ls = "solid", color = "cyan")
    plt.plot(x_e, M_e, ls = "dashed", color = "cyan")
    plt.xlim([xmin, xmax])

    plt.show()
