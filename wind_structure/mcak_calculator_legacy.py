#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import scipy.integrate as integ
import constants as csts
import useful_stuff as ufs
import matplotlib.pyplot as plt

Rsun = 6.955e10

class wind_calculator_mv08(object):
    """
    Wind structure calculator according to Mueller and Vink 2008 (MV08)
    """
    def __init__(self, Mstar = 40, Lstar = 10**5.5, Teff = 4e4, Gamma = 0.214, gamma = 0.462, delta = 0.6811, r0 = 1.0015, g0 = 17661, rc = 1.0098, a = 18.16, vterm = 3240, beta = 0.731, Mdot = 10**-6.046):
        """
        Keyword Arguments:
        (all default values are according to test calculation of section 4 in MV08)

        Mstar -- stellar mass in solar masses (default 40)
        Lstar -- stellar luminosity in solar luminosities (default 10**5.5)
        Teff -- effective temperature of star in Kelvin (default 40000)
        Gamma -- Edington factor with respect to electron scattering (default 0.214)
        gamma -- line acceleration parameter, c.f. MV08 Eq. 14 (default 0.462)
        delta -- line acceleration parameter, c.f. MV08 Eq. 14 (default 0.6811)
        r0 -- line acceleration parameter, c.f. MV08 Eq. 14 (default 1.0015)
        g0 -- line acceleration parameter, c.f. MV08 Eq. 14 (default 17661)
        rc -- dimensionless critical radius (default 1.0098)
        a -- isothermal sound speed in km/s (default 18.16)
        vterm -- terminal wind speed in km/s (default 3240)
        beta -- exponent of approximate beta-type velocity law (default 0.731)
        Mdot -- mass-loss rate in Msun/yr (default 10**-6.046)
        """

        self.Mstar = Mstar
        self.Lstar = Lstar
        self.Teff = Teff
        self.Mdot = Mdot
        self.vterm = vterm

        self.Gamma = Gamma
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        self.a = a
        self.rc = rc
        self.r0 = r0
        self.g0 = g0

        self._Rstar = None
        self._vesc = None
        self._vcrit = None

    def reset(self):
        """
        resets all the quantities derived from the principle parameters of the
        calculator
        """
        self._Rstar = None
        self._vesc = None
        self._vcrit = None

    @property
    def Gamma(self):
        """
        Eddington factor
        """
        return self._Gamma

    @Gamma.setter
    def Gamma(self, value):
        self.reset()
        self._Gamma = value

    @property
    def Mstar(self):
        """
        stellar mass (cgs)
        """
        return self._Mstar

    @Mstar.setter
    def Mstar(self, value):
        self.reset()
        self._Mstar = value * csts.CGS_SOLAR_MASS

    @property
    def Lstar(self):
        """
        luminosity of star (cgs)
        """
        return self._Lstar

    @Lstar.setter
    def Lstar(self, value):
        self.reset()
        self._Lstar = value * ufs.lumsun

    @property
    def Teff(self):
        """
        effective temperature of star (cgs)
        """
        return self._Teff

    @Teff.setter
    def Teff(self, value):
        self.reset()
        self._Teff = value

    @property
    def a(self):
        """
        isothermal sound speed (cgs)
        """
        return self._a

    @a.setter
    def a(self, value):
        self.reset()
        self._a = value * 1e5

    @property
    def vterm(self):
        """
        terminal wind speed (cgs)
        """
        return self._vterm

    @vterm.setter
    def vterm(self, value):
        self.reset()
        self._vterm = value * 1e5

    @property
    def Mdot(self):
        """
        mass loss rate (cgs)
        """
        return self._Mdot

    @Mdot.setter
    def Mdot(self, value):
        self.reset()
        self._Mdot = value * csts.CGS_SOLAR_MASS / 86400. / 365.

    @property
    def Rstar(self):
        """
        stellar radius (photospheric radius)
        """
        if self._Rstar is None:
            self._Rstar =  np.sqrt(self.Lstar / (4. * np.pi * csts.CGS_STEFAN_BOLTZMANN_CONSTANT * self.Teff**4))
        return self._Rstar

    @property
    def vesc(self):
        """
        escape speed from solar surface
        """
        if self._vesc is None:
            self._vesc = np.sqrt(2. * csts.CGS_GRAVITATIONAL_CONSTANT * self.Mstar * (1. - self.Gamma) / self.Rstar)
        return self._vesc

    @property
    def vcrit(self):
        """
        dimensionless critical wind speed, Eq. 11 in MV08
        """
        if self._vcrit is None:
            self._vcrit = self.vesc / self.a / np.sqrt(2)
        return self._vcrit


    def print(self):
        """
        Prints out some basic parameters and derived quantities
        """

        print(self.Rstar, self.Rstar / Rsun)
        print(self.vesc * 1e-5)
        print(self.vcrit)


    def calculate_velocity(self, r):
        """
        approximate wind velocity, according to Eq. 39 in MV08; applicable only
        to supersonic part of wind

        Arguments:
        r -- dimensionless radius, c.f. MV08 Eq. 9)

        Returns:
        v -- dimensionless velocity (MV08, Eq. 11) at radius r
        """

        return np.sqrt(2. / self.r0 * (self.vcrit**2 * (self.r0 / r - self.r0**(1. - 1. / self.delta)) + self.g0 / self.delta / (1. + self.gamma) * (1. - self.r0 / r**self.delta)**(1. + self.gamma)))

    def calculate_velocity_beta(self, r):
        """
        approximate beta-type wind velocity, according to MV08, Sec. 2.5.6 

        Arguments:
        r -- dimensionless radius, c.f. MV08 Eq. 9)

        Returns:
        v -- dimensionless velocity (MV08, Eq. 11) at radius r
        """

        return self.vterm / self.a * (1. - self.r0 / r)**self.beta


class wind_calculator_base(object):
    def __init__(self, Mstar = 52.5, Lstar = 1e6, Teff = 4.2e4, alpha = 0.6, k = 0.5, Gamma = 0, sigma = 0.3):

        self.Mstar = Mstar
        self.Lstar = Lstar
        self.Teff = Teff

        self.alpha = alpha
        self.k = k
        self.Gamma = Gamma
        self.sigma = sigma

        self._Mdot_cak = None
        self._vterm_cak = None
        self._Mdot = None
        self._vterm = None
        self._eta = None
        self._vth = None
        self._vesc = None
        self._t = None
        self._M = None
        self._Rstar = None

    def reset(self):

        self._Mdot_cak = None
        self._vterm_cak = None
        self._Mdot = None
        self._vterm = None
        self._vesc = None
        self._vth = None
        self._eta = None
        self._t = None
        self._M = None


    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self.reset()
        self._alpha = value

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, value):
        self.reset()
        self._k = value

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self.reset()
        self._sigma = value

    @property
    def vesc(self):
        if self._vesc is None:
            self._vesc = np.sqrt(2. * csts.CGS_GRAVITATIONAL_CONSTANT * self.Mstar * (1. - self.Gamma) / self.Rstar)
        return self._vesc

    @property
    def vth(self):
        if self._vth is None:
            self._vth = np.sqrt(2. * self._Teff * csts.CGS_BOLTZMANN / 1. / csts.CGS_UNIFIED_ATOMIC_MASS)
        return self._vth

    @vth.setter
    def vth(self, value):
        self._vth = value

    @property
    def Gamma(self):
        return self._Gamma

    @Gamma.setter
    def Gamma(self, value):
        self.reset()
        self._Gamma = value

    @property
    def Mstar(self):
        return self._Mstar

    @Mstar.setter
    def Mstar(self, value):
        self.reset()
        self._Mstar = value * csts.CGS_SOLAR_MASS

    @property
    def Lstar(self):
        return self._Lstar

    @Lstar.setter
    def Lstar(self, value):
        self.reset()
        self._Lstar = value * ufs.lumsun

    @property
    def Teff(self):
        return self._Teff

    @Teff.setter
    def Teff(self, value):
        self.reset()
        self._Teff = value

    @property
    def Rstar(self):
        if self._Rstar is None:
            self._Rstar =  np.sqrt(self.Lstar / (4. * np.pi * csts.CGS_STEFAN_BOLTZMANN_CONSTANT * self.Teff**4))

        return self._Rstar

    @Rstar.setter
    def Rstar(self, value):
        self._Rstar = value

    @property
    def Mdot_cak(self):
        if self._Mdot_cak is None:
            self._Mdot_cak = 4. * np.pi / self.sigma / self.vth * (self.sigma / 4. / np.pi)**(1. / self.alpha) * ((1. - self.alpha) / self.alpha)**((1. - self.alpha) / self.alpha) * (self.alpha * self.k)**(1. / self.alpha) * (self.Lstar / csts.CGS_SPEED_OF_LIGHT)**(1. / self.alpha) * (csts.CGS_GRAVITATIONAL_CONSTANT * self.Mstar * (1. - self.Gamma))**((self.alpha - 1.) / self.alpha)
        return self._Mdot_cak

    @property
    def Mdot(self):
        if self._Mdot is None:
            self._Mdot = self.Mdot_cak
        return self._Mdot

    @property
    def vterm_cak(self):
        if self._vterm_cak is None:
            self._vterm_cak = np.sqrt(self.alpha / (1. - self.alpha)) * self.vesc
        return self._vterm_cak

    @property
    def vterm(self):
        if self._vterm is None:
            self._vterm = self.vterm_cak
        return self._vterm

    @property
    def eta(self):
        if self._eta is None:
            self._eta = self.Mdot * self.vterm * csts.CGS_SPEED_OF_LIGHT / self.Lstar
        return self._eta

    @property
    def t(self):
        if self._t is None:
            self._t = self.Mdot * self.sigma * self.vth / (2. * np.pi * self.vterm**2 * self.Rstar)
        return self._t

    @property
    def M(self):
        if self._M is None:
            self._M = self.k * self.t**(-self.alpha)
        return self._M

    def calculate_velocity(self, x):

        return self.vterm * np.sqrt(1. - 1. / x)

    def calculate_density(self, x):

        r = self.Rstar * x

        return self.Mdot / (4. * np.pi * r**2 * self.calculate_velocity(x))

    def print(self):

        print("Stellar Parameters:")
        print("----------------")
        print("Mstar: %e (%.2f Msun)" % (self.Mstar, self.Mstar / csts.CGS_SOLAR_MASS))
        print("Lstar: %e (%.2e Lsun)" % (self.Lstar, self.Lstar / ufs.lumsun))
        print("Teffr: %e" % self.Teff)
        print("Rstar: %e (%.2f Rsun)" % (self.Rstar, self.Rstar / Rsun))
        print("Gamma: %f" % self.Gamma)
        print("\nCAK Parameters:")
        print("----------------")
        print("k: %f" % self.k)
        print("alpha: %f" % self.alpha)
        print("sigma: %f" % self.sigma)
        print("vth: %e" % self.vth)

class wind_calculator_cak75(wind_calculator_base):
    def __init__(self, Mstar = 52.5, Lstar = 1e6, Teff = 4.2e4, alpha = 0.6, k = 0.5, Gamma = 0, sigma = 0.3):
        super(wind_calculator_cak75, self).__init__(Mstar = Mstar, Lstar = Lstar, Teff = Teff, alpha = alpha, k = k, Gamma = Gamma, sigma = sigma) 

    def calculate_velocity(self, x):

        return self.vterm * np.sqrt(1. - 1. / x)

    def calculate_density(self, x):

        r = self.Rstar * x

        return self.Mdot / (4. * np.pi * r**2 * self.calculate_velocity(x))

    def print(self):

        print("\n================")
        print("Castor et al. 1975 wind calculation:")
        super(wind_calculator_cak75, self).print()
        print("\nResulting Wind Parameters:")
        print("----------------")
        print("t: %e" % self.t)
        print("M: %e" % self.M)
        print("vterm: %e" % self.vterm)
        print("Mdot: %e (%.3e Msun/year)" % (self.Mdot, self.Mdot / csts.CGS_SOLAR_MASS * 365. * 86400.))
        print("eta: %f" % self.eta)
        print("================")

class wind_calculator_fa86(wind_calculator_base):
    def __init__(self, Mstar = 52.5, Lstar = 1e6, Teff = 4.2e4, alpha = 0.6, k = 0.5, Gamma = 0, sigma = 0.3):
        super(wind_calculator_fa86, self).__init__(Mstar = Mstar, Lstar = Lstar, Teff = Teff, alpha = alpha, k = k, Gamma = Gamma, sigma = sigma) 

    @property
    def Mdot(self):
        if self._Mdot is None:
            self._Mdot = self.Mdot_cak * 0.5 * (self.vesc * 1e-8)**(-0.3)
        return self._Mdot

    @property
    def vterm(self):
        if self._vterm is None:
            self._vterm = self.vesc * 2.2 * self.alpha / (1. - self.alpha) * (self.vesc * 1e-8)**0.2
        return self._vterm

    def print(self):

        print("\n================")
        print("Friend and Abbott 1986 wind calculation:")
        super(wind_calculator_fa86, self).print()
        print("\nResulting Wind Parameters:")
        print("----------------")
        print("vterm: %e" % self.vterm)
        print("Mdot: %e (%.3e Msun/year)" % (self.Mdot, self.Mdot / csts.CGS_SOLAR_MASS * 365. * 86400.))
        print("eta: %f" % self.eta)
        print("================")




class wind_calculator_kppa89(wind_calculator_base):
    def __init__(self, Mstar = 52.5, Lstar = 1e6, Teff = 4.2e4, alpha = 0.6, k = 0.5, Gamma = 0, sigma = 0.3, beta = 0.8):
        super(wind_calculator_kppa89, self).__init__(Mstar = Mstar, Lstar = Lstar, Teff = Teff, alpha = alpha, k = k, Gamma = Gamma, sigma = sigma)

        self.beta = beta

        self._f1 = None

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._vterm = None
        self._beta = value

    @property
    def f1(self):
        if self._f1 is None:
            self._f1 = 1. / (self.alpha + 1.)
        return self._f1

    @property
    def Mdot(self):
        if self._Mdot is None:
            self._Mdot = self.f1**(1. / self.alpha) * self.Mdot_cak
        return self._Mdot

    @property
    def vterm(self):
        if self._vterm is None:
            self._vterm = self.vterm_cak * np.sqrt(integ.quad(self.Z, 0, 1)[0])
        return self._vterm

    def h(self, x):

        return (x - 1.) / self.beta

    def f(self, x):

        return 1. / (self.alpha + 1.) * x**2 / (1. - self.h(x)) * (1. - (1. - 1. / x**2 + self.h(x) / x**2)**(self.alpha + 1.))

    def fN(self, x):

        return self.f(x) / self.f1

    def Z(self, u):

        x = 1. / u

        return self.fN(x)**(1. / (1. - self.alpha)) * (1. + np.sqrt(2. / self.alpha * (1. - (1. / self.fN(x))**(1. / (1. - self.alpha)))))

    def calculate_velocity(self, x):

        u = 1. / x
        I = integ.quad(self.Z, u, 1)[0]
        vesc2 = 2. * csts.CGS_GRAVITATIONAL_CONSTANT * self.Mstar * (1. - self.Gamma) / self.Rstar

        return np.sqrt(self.alpha / (1. - self.alpha) * vesc2 * I)

    def calculate_density(self, x):

        r = self.Rstar * x

        return self.Mdot / (4. * np.pi * r**2 * self.calculate_velocity(x))

    def print(self):

        print("\n================")
        print("Kudritzki et al. 1989 wind calculation:")
        super(wind_calculator_kppa89, self).print()
        print("\nResulting Wind Parameters:")
        print("----------------")
        print("vterm: %e" % self.vterm)
        print("Mdot: %e (%.3e Msun/year)" % (self.Mdot, self.Mdot / csts.CGS_SOLAR_MASS * 365. * 86400.))
        print("eta: %f" % self.eta)
        print("================")



class mcak_calculator_pauldrach(object):
    def __init__(self, k, alpha, Gamma, Rstar, Mstar, Lstar, aiso, sigma_e, v_th):

        self.Mstar = csts.CGS_SOLAR_MASS * Mstar
        self.Gamma = Gamma
        self.k = k
        self.alpha = alpha
        self.Lstar = Lstar
        self.Rstar = Rstar
        self.aiso = aiso
        self.sigma_e = sigma_e
        self.v_th = v_th

    def critical_u(self):

        self.uc = -1. / (1.04 * self.Rstar)

    def G(self, u, w, z, simple = False):

        if simple:
            nominator = 1. - (1. - (self.Rstar * u)**2)**(self.alpha + 1.)
            denominator = (self.alpha + 1.) * (self.Rstar * u)**2
        else:
            nominator = 1. - (1. - (self.Rstar * u)**2 - 2. * self.Rstar**2 * w * u / z)**(self.alpha + 1.)
            denominator = (self.alpha + 1.) * (self.Rstar * u)**2 * (1. + 2 * w / z * u)

        return nominator / denominator


    def h(self, u):

        return -csts.CGS_GRAVITATIONAL_CONSTANT * self.Mstar * (1. - self.Gamma) - 2 * self.aiso**2 / u

    def dhdu(self, u):

        return 2 * (self.aiso / u)**2

    def dGdu(self, u, w, z, simple = False):

        if simple:
            return 2. * (1. - (self.Rstar * u)**2)**self.alpha / u - 2. * (1. - (1. - (self.Rstar * u)**2)**(1. + self.alpha)) / (self.Rstar**2 * u**3 * (1. + self.alpha))

        else:
            coeff1 = 2. / (self.Rstar**2 * u**3 * (2. * u * w + z)**2 * (1. + self.alpha))
            coeff2 = ((z - self.Rstar**2 * u * (2. * w + u * z)) / z)**self.alpha

            return coeff1 * (-z * (3. * u * w + z) + coeff2 * (z * (3. * u * w + z) + self.Rstar**2 * u * (-w * (4. * u * w + z + u**2 * z) + (2. * u * w + z) * (w + u * z) * self.alpha)))

 ##   def dGdw(self, u, w, z, simple = False):

 ##       if simple:
 ##           return 0
 ##       else:
 ##           res1 = (2. * (1. - (self.Rstar * u)**2 - 2. * self.Rstar**2 * u * w / z)**self.alpha) / (u * (1. + 2. * u * w / z) * z)
 ##           res2 = (2. * (1. - (1. - (self.Rstar * u)**2 - (2. * self.Rstar**2 * u * 2 / z))**(1. + self.alpha))) / (self.Rstar**2 * u * (1. + 2. * u * w / z)**2 * z * (1. + self.alpha))
 ##           return res1 + res2

    def C1(self, u, w, z, simple = False):

        return self.dhdu(u) - self.h(u) * self.dGdu(u, w, z, simple = simple) / (self.G(u, w, z, simple = simple) * (1. - self.alpha))

    def wc_calc(self, u, w, z, simple = False):

        return 0.5 * self.aiso**2 - (self.alpha / (1. - self.alpha)) * self.h(u) / np.sqrt(2. * self.C1(u, w, z, simple = simple) / self.aiso**2)

    def zc_calc(self, u, w, z, simple = False):

        return np.sqrt(self.aiso**2 * self.C1(u, w, z, simple = simple) + 0.5) - self.h(u) * self.alpha / (1. - self.alpha)

    def dFdu(self, u, w, z, simple = False):

        return -self.dhdu(u) - self.C * z**self.alpha * self.dGdu(u, w, z, simple = simple)

    def dFdw(self, u, w, z, simple = False):

        return z * self.aiso**2 / (2. * w**2)# - self.C * z**self.alpha * self.dGdw(u, w, z, simple = simple)

    def dFdz(self, u, w, z, simple = False):

        return (1. - self.aiso**2 / (2. * w)) - self.alpha * self.C * self.G(u, w, z, simple = simple) * z**(self.alpha - 1.)

    def dudz(self, u, w, z, simple = False):

        return -self.dFdz(u, w, z, simple = simple) / (self.dFdu(u, w, z, simple = simple) + z * self.dFdw(u, w, z, simple = simple))

    def dwdz(self, u, w, z, simple = False):

        return z * self.dudz(u, w, z, simple = simple)

    def duwdz(self, z, uw, simple):

        u = uw[0]
        w = uw[1]

        return np.array([self.dudz(u, w, z, simple = simple), self.dwdz(u, w, z, simple = simple)])

    def intergrate_beyond_critical_point(self):

        dz = self.zc * 0.1
        z1 = 7 * self.zc

        #r = integ.ode(self.duwdz).set_integrator('dopri5')
        r = integ.ode(self.duwdz).set_integrator('zvode', method='bdf')
        r.set_initial_value(np.array([self.uc, self.wc]), self.zc).set_f_params(True)

        u = []
        w = []

        while r.successful() and r.t < z1:
            r.integrate(r.t+dz)
            u.append(r.y[0])
            w.append(r.y[-1])
            print("z = %g u = %g w = %g; r/R = %g; v = %g" % (r.t, r.y[0], r.y[1], -1 / r.y[0] / self.Rstar, np.sqrt(2 * r.y[1])))

        u = np.array(u)
        w = np.array(w)

        plt.plot(-1 / u / self.Rstar, np.sqrt(2. * w))




    def critical_point_iteration(self):

        self.critical_u()
        wcstart = self.wc_calc(self.uc, 1, 1, simple = True)
        zcstart = self.zc_calc(self.uc, 1, 1, simple = True)

        print("w", wcstart, "z", zcstart)

        for i in xrange(50):

            wcnew = self.wc_calc(self.uc, wcstart, zcstart)
            zcnew = self.zc_calc(self.uc, wcstart, zcstart)

            rel_change = np.fabs((wcnew - wcstart) / wcnew) + np.fabs((zcnew - zcstart) / zcnew)

            wcstart = wcnew
            zcstart = zcnew
            print("w", wcstart, "z", zcstart)

            if rel_change < 1e-7:
                break
        else:
            print("WARNING: converged solution at critical point not found after %d iterations" % (i+1))

        self.wc = wcstart
        self.zc = zcstart

        self.C = - (1. / (1. - self.alpha)) * self.h(self.uc) / (self.G(self.uc, self.wc, self.zc) * self.zc**self.alpha)


        self.Mdot = (4. * np.pi) / (self.sigma_e * self.v_th) * (self.C / (self.Gamma * csts.CGS_GRAVITATIONAL_CONSTANT * self.Mstar * self.k))**(-1. / self.alpha)

        print("INFORMATION: critical point values:\nu = %e\nw = %e\nz = %e\nC = %e\nMdot [Msun/yr] = %e\nv [km/s] = %f" % (self.uc, self.wc, self.zc, self.C, self.Mdot / csts.CGS_SOLAR_MASS * 86400. * 365., np.sqrt(self.wc * 2) * 1e-5))

def main2():

    k = 0.415106878632
    alpha = 0.577633849139
    Gamma = 0.502

    Mstar = 52.5
    Rstar = 1.3170257344e+12
    Lstar = 3.846e39

    aiso = 1670814.76810567

    v_th = 2642755.54691
    sigma_e = 0.3

    x = 1. + np.linspace(0, 100, 1e3)
    beta = 0.8

    ppk_wind = mcak_calculator_pauldrach(k, alpha, Gamma, Rstar, Mstar, Lstar, aiso, sigma_e, v_th)
    ppk_wind.critical_point_iteration()
    ppk_wind.intergrate_beyond_critical_point()
    kppa_wind = wind_calculator_kppa89(Mstar = Mstar, Lstar = Lstar / ufs.lumsun, Teff = 4.2e4, alpha = alpha, k = k, Gamma = Gamma, sigma = sigma_e, beta = beta)
    ax = plt.gca()
    ax.plot(x, [kppa_wind.calculate_velocity(xi) for xi in x])
    plt.show()

def main3():

    k = 0.415106878632
    alpha = 0.577633849139
    Gamma = 0.502

    Mstar = 52.5
    Rstar = 1.3170257344e+12
    Lstar = 3.846e39

    aiso = 1670814.76810567

    v_th = 2642755.54691
    sigma_e = 0.3

    x = 1. + np.logspace(-3, 3, 1e3)

    cak_wind = wind_calculator_cak75(Mstar = Mstar, Lstar = Lstar / ufs.lumsun, Teff = 4.2e4, alpha = alpha, k = k, Gamma = Gamma, sigma = sigma_e)
    cak_wind.print()

    beta = 0.5
    kppa_wind = wind_calculator_kppa89(Mstar = Mstar, Lstar = Lstar / ufs.lumsun, Teff = 4.2e4, alpha = alpha, k = k, Gamma = Gamma, sigma = sigma_e, beta = beta)
    v1 = np.array([kppa_wind.calculate_velocity(xi) for xi in x])
    r1 = np.array([kppa_wind.calculate_density(xi) for xi in x])

    beta = 1.0
    kppa_wind = wind_calculator_kppa89(Mstar = Mstar, Lstar = Lstar / ufs.lumsun, Teff = 4.2e4, alpha = alpha, k = k, Gamma = Gamma, sigma = sigma_e, beta = beta)
    v2 = np.array([kppa_wind.calculate_velocity(xi) for xi in x])
    r2 = np.array([kppa_wind.calculate_density(xi) for xi in x])

    beta = 0.8
    kppa_wind = wind_calculator_kppa89(Mstar = Mstar, Lstar = Lstar / ufs.lumsun, Teff = 4.2e4, alpha = alpha, k = k, Gamma = Gamma, sigma = sigma_e, beta = beta)
    plt.fill_between(x - 1, v1, v2, color = "0.7")
    plt.plot(x - 1, [kppa_wind.calculate_velocity(xi) for xi in x], ls = "dashed", color = "black")
    plt.plot(x - 1, cak_wind.calculate_velocity(x), ls = "dotted", color = "black")
    ax = plt.gca()
    ax.set_xscale("log")

    plt.figure()
    plt.fill_between(x, r1, r2, color = "0.7")
    plt.plot(x, [kppa_wind.calculate_density(xi) for xi in x], ls = "dashed", color = "black")
    plt.plot(x, cak_wind.calculate_density(x), ls = "dotted", color = "black")
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.show()

def main4():

    k = 0.38
    alpha = 0.60
    Gamma = 0.50

    Mstar = 52.5
    Rstar = 1.3170257344e+12
    Lstar = 3.846e39

    aiso = 1670814.76810567

    v_th = 2642755.54691
    sigma_e = 0.3

    x = 1. + np.logspace(-3, 3, 1e3)

    cak_wind = wind_calculator_cak75(Mstar = Mstar, Lstar = Lstar / ufs.lumsun, Teff = 4.2e4, alpha = alpha, k = k, Gamma = Gamma, sigma = sigma_e)
    cak_wind.print()

    beta = 0.5
    kppa_wind = wind_calculator_kppa89(Mstar = Mstar, Lstar = Lstar / ufs.lumsun, Teff = 4.2e4, alpha = alpha, k = k, Gamma = Gamma, sigma = sigma_e, beta = beta)
    v1 = np.array([kppa_wind.calculate_velocity(xi) for xi in x])
    r1 = np.array([kppa_wind.calculate_density(xi) for xi in x])

    beta = 1.0
    kppa_wind = wind_calculator_kppa89(Mstar = Mstar, Lstar = Lstar / ufs.lumsun, Teff = 4.2e4, alpha = alpha, k = k, Gamma = Gamma, sigma = sigma_e, beta = beta)
    v2 = np.array([kppa_wind.calculate_velocity(xi) for xi in x])
    r2 = np.array([kppa_wind.calculate_density(xi) for xi in x])

    beta = 0.8
    kppa_wind = wind_calculator_kppa89(Mstar = Mstar, Lstar = Lstar / ufs.lumsun, Teff = 4.2e4, alpha = alpha, k = k, Gamma = Gamma, sigma = sigma_e, beta = beta)
    plt.fill_between(x - 1, v1, v2, color = "0.7")
    plt.plot(x - 1, [kppa_wind.calculate_velocity(xi) for xi in x], ls = "dashed", color = "black")
    plt.plot(x - 1, cak_wind.calculate_velocity(x), ls = "dotted", color = "black")
    ax = plt.gca()
    ax.set_xscale("log")

    plt.figure()
    plt.fill_between(x, r1, r2, color = "0.7")
    plt.plot(x, [kppa_wind.calculate_density(xi) for xi in x], ls = "dashed", color = "black")
    plt.plot(x, cak_wind.calculate_density(x), ls = "dotted", color = "black")
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.show()

def main():

    k = 0.415106878632
    alpha = 0.577633849139
    Gamma = 0.502

    Mstar = 52.5
    Rstar = 1.3170257344e+12
    Lstar = 3.846e39

    aiso = 1670814.76810567

    v_th = 2642755.54691
    sigma_e = 0.3

    beta = 0.8

    x = 1. + np.logspace(-2, 2, 1e3)

    cak_wind = wind_calculator_cak75(Mstar = Mstar, Lstar = Lstar / ufs.lumsun, Teff = 4.2e4, alpha = alpha, k = k, Gamma = Gamma, sigma = sigma_e)
    cak_wind.print()
    fa_wind = wind_calculator_fa86(Mstar = Mstar, Lstar = Lstar / ufs.lumsun, Teff = 4.2e4, alpha = alpha, k = k, Gamma = Gamma, sigma = sigma_e)
    fa_wind.print()

    ppk_wind = mcak_calculator_pauldrach(k, alpha, Gamma, Rstar, Mstar, Lstar, aiso, sigma_e, v_th)
    ppk_wind.critical_point_iteration()

    kppa_wind = wind_calculator_kppa89(Mstar = Mstar, Lstar = Lstar / ufs.lumsun, Teff = 4.2e4, alpha = alpha, k = k, Gamma = Gamma, sigma = sigma_e, beta = beta)
    kppa_wind.print()
    v = kppa_wind.calculate_velocity(1.04)
    print("Critical Point info: Mdot[Msun/yr] = %e, v[km/s] = %f" % (kppa_wind.Mdot / csts.CGS_SOLAR_MASS * 86400. * 365., 1e-5 * v))

    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.plot(x, cak_wind.calculate_velocity(x))
    ax.plot(x, [kppa_wind.calculate_velocity(xi) for xi in x])
    ax.plot(x, kppa_wind.vterm * (1. - 1. / x)**0.8)
    ax.axhline(y = fa_wind.vterm, color = "red", ls = "dashed")

    ax = fig.add_subplot(222)
    ax.plot(np.log10(x - 1), cak_wind.calculate_velocity(x))
    ax.plot(np.log10(x - 1), [kppa_wind.calculate_velocity(xi) for xi in x])
    ax.plot(np.log10(x - 1), kppa_wind.vterm * (1. - 1. / x)**0.8)
    ax.plot(np.log10(x - 1), kppa_wind.vterm * (1. - 0.9983 / x)**0.83)
    ax.axhline(y = fa_wind.vterm, color = "red", ls = "dashed")

    ax = fig.add_subplot(223)
    ax.plot(x, cak_wind.calculate_density(x))
    ax.plot(x, [kppa_wind.calculate_density(xi) for xi in x])

    ax = fig.add_subplot(224)
    ax.plot(x, np.ones(len(x)) * cak_wind.Mdot)
    ax.plot(x, np.ones(len(x)) * kppa_wind.Mdot)
    ax.axhline(y = fa_wind.Mdot, color = "red", ls = "dashed")


    plt.show()


def main5():

    r = np.logspace(-3, 3, 512) + 1.

    tester = wind_calculator_mv08()
    tester.print()
    v = tester.calculate_velocity(r)
    vbeta = tester.calculate_velocity_beta(r)

    plt.plot(r, v)
    plt.plot(r, vbeta)
    plt.show()


if __name__ == "__main__":

    main5()

##    k = 0.415106878632
##    alpha = 0.577633849139
##    Gamma = 0.502
##
##    Mstar = 52.5
##    Rstar = 1.3170257344e+12
##    Lstar = 3.846e39
##
##    aiso = 1670814.76810567
##
##    v_th = 2642755.54691
##    sigma_e = 0.3
##
##    beta = 0.8
##
##
##    ppk_test = mcak_calculator_pauldrach(k, alpha, Gamma, Rstar, Mstar, Lstar, aiso, sigma_e, v_th)
##    ppk_test.critical_point_iteration()
##
##    print(ppk_test.dudz(ppk_test.uc, ppk_test.wc, ppk_test.zc, simple = True))
##    print(ppk_test.dwdz(ppk_test.uc, ppk_test.wc, ppk_test.zc, simple = True))
##
##    kppa_test = mcak_calculator_kudritzki(k, alpha, Gamma, Rstar, Mstar, Lstar, aiso, sigma_e, v_th)
##    kppa_test.calculate_mass_loss_rate(beta)
##    v = kppa_test.calculate_velocity(beta, 1.04)
##
##    print("Critical Point info: Mdot[Msun/yr] = %e, v[km/s] = %f" % (kppa_test.Mdot / csts.CGS_SOLAR_MASS * 86400. * 365., 1e-5 * v))
##
##    v = kppa_test.calculate_velocity(beta, 1e4)
##
##    print("Terminal velocity: v_infinity[km/s] = %f" % (v * 1e-5))

