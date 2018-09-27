from __future__ import print_function
import numpy as np
import scipy.integrate as integ
import astropy.units as units
import astropy.constants as csts
from astropy.utils.decorators import lazyproperty


def _test_unit(val, final_unit_string, initial_unit_string):
    try:
        val.to(final_unit_string)
    except AttributeError:
        val = val * units.Unit(initial_unit_string)
    val = val.to(final_unit_string)

    return val


class StarBase(object):
    def __init__(self, mass=52.5, lum=1e6, teff=4.2e4, gamma=0, sigma=0.3):

        self.mass = mass
        self.lum = lum
        self.teff = teff
        self.sigma = sigma
        self.gamma = gamma

    @property
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, val):
        val = _test_unit(val, "g", "solMass")
        self._mass = val

    @property
    def lum(self):
        return self._lum

    @lum.setter
    def lum(self, val):
        val = _test_unit(val, "erg/s", "solLum")
        self._lum = val

    @property
    def teff(self):
        return self._teff

    @teff.setter
    def teff(self, val):
        val = _test_unit(val, "K", "K")
        self._teff = val

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, val):
        val = _test_unit(val, "cm^2/g", "cm^2/g")
        self._sigma = val

    @lazyproperty
    def rad(self):
        rad = np.sqrt(self.lum /
                      (4 * np.pi * csts.sigma_sb * self.teff**4))
        return rad

    @lazyproperty
    def vth(self):
        vth = np.sqrt(2. * self.teff * csts.k_B / csts.u)
        return vth

    @lazyproperty
    def vesc(self):
        vesc = np.sqrt(2. * csts.G * self.mass * (1. - self.gamma) / self.rad)
        return vesc


class WindBase(object):
    def __init__(self, alpha=0.6, k=0.5):

        self.alpha = alpha
        self.k = k


class WindStructureBase(object):
    def __init__(self, mstar=52.5, lstar=1e6, teff=4.2e4, alpha=0.6, k=0.5,
                 gamma=0, sigma=0.3):

        self.star = StarBase(mass=mstar, lum=lstar, teff=teff, gamma=gamma,
                             sigma=sigma)
        self.wind = WindBase(alpha=alpha, k=k)

    @lazyproperty
    def eta(self):
        eta = (self.mdot * self.vterm * csts.c / self.star.lum).to("").value
        return eta

    @lazyproperty
    def t(self):
        t = (self.mdot * self.star.sigma * self.star.vth /
             (2. * np.pi * self.vterm**2 * self.star.rad)).to("")
        return t

    @lazyproperty
    def m(self):
        m = self.wind.k * self.t**(-self.wind.alpha)
        return m


class BaseVelocityDensityMixin(object):
    def v(self, x):
        """calculate wind velocity at given location

        Parameters
        ----------
        x : float, np.ndarray
            dimensionless position, i.e. r/Rstar

        Returns
        -------
        v : float, np.ndarry
            wind velocity
        """

        return (self.vterm * np.sqrt(1. - 1. / x)).to("km/s")

    def rho(self, x):
        """calculate wind density at given location

        Parameters
        ----------
        x : float, np.ndarray
            dimensionless position, i.e. r/Rstar

        Returns
        -------
        rho : float, np.ndarry
            wind density
        """

        r = self.star.rad * x

        return (self.mdot /
                (4. * np.pi * r**2 * self.v(x))).to("g/cm^3")


class BaseCakStructureMixin(object):

    @lazyproperty
    def mdot_cak(self):

        mdot_cak = ((4. * np.pi / (self.star.sigma * self.star.vth)) *
                    (self.star.sigma / (4. * np.pi))**(1. / self.wind.alpha) *
                    ((1. - self.wind.alpha) / self.wind.alpha)**(
                        (1. - self.wind.alpha) / self.wind.alpha) *
                    (self.wind.alpha * self.wind.k)**(1. / self.wind.alpha) *
                    (self.star.lum / csts.c)**(1. / self.wind.alpha) *
                    (csts.G * self.star.mass * (1. - self.star.gamma))**(
                        (self.wind.alpha - 1.) / self.wind.alpha))
        return mdot_cak.to("Msun/yr")

    @lazyproperty
    def vterm_cak(self):

        vterm_cak = (np.sqrt(self.wind.alpha / (1. - self.wind.alpha)) *
                     self.star.vesc)

        return vterm_cak.to("km/s")


class WindStructureCak75(WindStructureBase, BaseCakStructureMixin,
                         BaseVelocityDensityMixin):
    def __init__(self, mstar=52.5, lstar=1e6, teff=4.2e4, alpha=0.6, k=0.5,
                 gamma=0, sigma=0.3):
        super(WindStructureCak75, self).__init__(mstar=mstar, lstar=lstar,
                                                 teff=teff, alpha=alpha,
                                                 k=k, gamma=gamma,
                                                 sigma=sigma)

    @property
    def mdot(self):
        return self.mdot_cak

    @property
    def vterm(self):
        return self.vterm_cak


class WindStructureKppa89(WindStructureBase, BaseCakStructureMixin,
                          BaseVelocityDensityMixin):
    def __init__(self, mstar=52.5, lstar=1e6, teff=4.2e4, alpha=0.6, k=0.5,
                 gamma=0, sigma=0.3, beta=0.8):
        super(WindStructureKppa89, self).__init__(mstar=mstar, lstar=lstar,
                                                  teff=teff, alpha=alpha,
                                                  k=k, gamma=gamma,
                                                  sigma=sigma)

        self.f1 = 1. / (self.wind.alpha + 1.)
        self.beta = beta

    @lazyproperty
    def mdot(self):

        mdot = self.f1**(1. / self.wind.alpha) * self.mdot_cak

        return mdot

    @lazyproperty
    def vterm(self):

        vterm = self.vterm_cak * np.sqrt(integ.quad(self.Z, 0, 1)[0])

        return vterm

    def h(self, x):

        return (x - 1.) / self.beta

    def f(self, x):

        return (1. / (self.wind.alpha + 1.) * x**2 / (1. - self.h(x)) *
                (1. - (1. - 1. / x**2 + self.h(x) / x**2)**(
                    self.wind.alpha + 1.)))

    def fn(self, x):

        return self.f(x) / self.f1

    def z(self, u):

        x = 1. / u
        z = (self.fn(x)**(1. / (1. - self.wind.alpha)) *
             (1. + np.sqrt(2. / self.wind.alpha *
                           (1. - (1. / self.fn(x))**(
                               1. / (1. - self.wind.alpha))))))
        return z

    def _v_scalar(self, x):

        u = 1. / x
        I = integ.quad(self.z, u, 1)[0]
        vesc2 = (2. * csts.G * self.star.m *
                 (1. - self.star.gamma) / self.star.rad)
        v = np.sqrt(self.wind.alpha / (1. - self.wind.alpha) * vesc2 * I).to(
            "km/s")

        return v.value

    def v(self, x):

        if type(x) is np.ndarray:
            v = np.array([self._v_scalar(xi) for xi in x])
        else:
            v = self._v_scalar(x)

        v = v * units.km / units.s
        return v


class WindStructureFa86(WindStructureBase, BaseCakStructureMixin,
                        BaseVelocityDensityMixin):
    def __init__(self, mstar=52.5, lstar=1e6, teff=4.2e4, alpha=0.6, k=0.5,
                 gamma=0, sigma=0.3):
        super(WindStructureFa86, self).__init__(mstar=mstar, lstar=lstar,
                                                teff=teff, alpha=alpha,
                                                k=k, gamma=gamma,
                                                sigma=sigma)

    @lazyproperty
    def mdot(self):

        mdot = (self.mdot_cak * 0.5 *
                (self.star.vesc / (1e3 * units.km / units.s))**(-0.3))

        return mdot

    @lazyproperty
    def vterm(self):

        vterm = (self.star.vesc * 2.2 * self.wind.alpha /
                 (1. - self.wind.alpha) *
                 (self.star.vesc / (1e3 * units.km / units.s))**0.2)

        return vterm
