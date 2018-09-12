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


class star_base(object):
    def __init__(self, M=52.5, L=1e6, Teff=4.2e4, Gamma=0, sigma=0.3):

        self.M = M
        self.L = L
        self.Teff = Teff
        self.sigma = sigma
        self.Gamma = Gamma

    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, val):
        val = _test_unit(val, "g", "solMass")
        self._M = val

    @property
    def L(self):
        return self._L

    @L.setter
    def L(self, val):
        val = _test_unit(val, "erg/s", "solLum")
        self._L = val

    @property
    def Teff(self):
        return self._Teff

    @Teff.setter
    def Teff(self, val):
        val = _test_unit(val, "K", "K")
        self._Teff = val

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, val):
        val = _test_unit(val, "cm^2/g", "cm^2/g")
        self._sigma = val

    @lazyproperty
    def R(self):
        R = np.sqrt(self.L /
                    (4 * np.pi * csts.sigma_sb * self.Teff**4))
        return R

    @lazyproperty
    def vth(self):
        vth = np.sqrt(2. * self.Teff * csts.k_B / csts.u)
        return vth

    @lazyproperty
    def vesc(self):
        vesc = np.sqrt(2. * csts.G * self.M * (1. - self.Gamma) / self.R)
        return vesc


class wind_base(object):
    def __init__(self, alpha=0.6, k=0.5):

        self.alpha = alpha
        self.k = k


class wind_structure_base(object):
    def __init__(self, Mstar=52.5, Lstar=1e6, Teff=4.2e4, alpha=0.6, k=0.5,
                 Gamma=0, sigma=0.3):

        self.star = star_base(M=Mstar, L=Lstar, Teff=Teff, Gamma=Gamma,
                              sigma=sigma)
        self.wind = wind_base(alpha=alpha, k=k)

    @lazyproperty
    def eta(self):
        eta = (self.Mdot * self.vterm * csts.c / self.star.L).to("").value
        return eta

    @lazyproperty
    def t(self):
        t = (self.Mdot * self.star.sigma * self.star.vth /
             (2. * np.pi * self.vterm**2 * self.star.R)).to("")
        return t

    @lazyproperty
    def M(self):
        M = self.wind.k * self.t**(-self.wind.alpha)
        return M


class base_velocity_density_mixin(object):
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

        r = self.star.R * x

        return (self.Mdot /
                (4. * np.pi * r**2 * self.v(x))).to("g/cm^3")


class base_cak_structure_mixin(object):

    @lazyproperty
    def Mdot_cak(self):

        Mdot_cak = ((4. * np.pi / (self.star.sigma * self.star.vth)) *
                    (self.star.sigma / (4. * np.pi))**(1. / self.wind.alpha) *
                    ((1. - self.wind.alpha) / self.wind.alpha)**(
                        (1. - self.wind.alpha) / self.wind.alpha) *
                    (self.wind.alpha * self.wind.k)**(1. / self.wind.alpha) *
                    (self.star.L / csts.c)**(1. / self.wind.alpha) *
                    (csts.G * self.star.M * (1. - self.star.Gamma))**(
                        (self.wind.alpha - 1.) / self.wind.alpha))
        return Mdot_cak.to("Msun/yr")

    @lazyproperty
    def vterm_cak(self):

        vterm_cak = (np.sqrt(self.wind.alpha / (1. - self.wind.alpha)) *
                     self.star.vesc)

        return vterm_cak.to("km/s")


class wind_structure_cak75(wind_structure_base, base_cak_structure_mixin,
                           base_velocity_density_mixin):
    def __init__(self, Mstar=52.5, Lstar=1e6, Teff=4.2e4, alpha=0.6, k=0.5,
                 Gamma=0, sigma=0.3):
        super(wind_structure_cak75, self).__init__(Mstar=Mstar, Lstar=Lstar,
                                                   Teff=Teff, alpha=alpha,
                                                   k=k, Gamma=Gamma,
                                                   sigma=sigma)

    @property
    def Mdot(self):
        return self.Mdot_cak

    @property
    def vterm(self):
        return self.vterm_cak


class wind_structure_kppa89(wind_structure_base, base_cak_structure_mixin,
                            base_velocity_density_mixin):
    def __init__(self, Mstar=52.5, Lstar=1e6, Teff=4.2e4, alpha=0.6, k=0.5,
                 Gamma=0, sigma=0.3, beta=0.8):
        super(wind_structure_kppa89, self).__init__(Mstar=Mstar, Lstar=Lstar,
                                                    Teff=Teff, alpha=alpha,
                                                    k=k, Gamma=Gamma,
                                                    sigma=sigma)

        self.f1 = 1. / (self.wind.alpha + 1.)
        self.beta = beta

    @lazyproperty
    def Mdot(self):

        Mdot = self.f1**(1. / self.wind.alpha) * self.Mdot_cak

        return Mdot

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

    def fN(self, x):

        return self.f(x) / self.f1

    def Z(self, u):

        x = 1. / u
        Z = (self.fN(x)**(1. / (1. - self.wind.alpha)) *
             (1. + np.sqrt(2. / self.wind.alpha *
                           (1. - (1. / self.fN(x))**(
                               1. / (1. - self.wind.alpha))))))
        return Z

    def v(self, x):

        u = 1. / x
        I = integ.quad(self.Z, u, 1)[0]
        vesc2 = (2. * csts.G * self.star.M *
                 (1. - self.star.Gamma) / self.star.R)
        v = np.sqrt(self.wind.alpha / (1. - self.wind.alpha) * vesc2 * I).to(
            "km/s")
        return v
