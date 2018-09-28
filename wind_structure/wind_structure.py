# MIT License
#
# Copyright (c) 2017 Ulrich Noebauer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Module providing a number of basic calculators to determine the steady-state
structure of line-driven hot star winds analytically. In particular, the
predictions according to
 * Castor, Abbott and Klein 1975
 * Friend and Abbott 1986
 * Kudritzki, Pauldrach, Puls and Abbott 1989
are included. Throughout this module, we rely heavily on the standard text
book on stellar winds by Lamers and Cassinelli 1999.

Note
----
We assume a so-called frozen-in ionization throughout the wind in all the wind
structure calculators (i.e. delta=0 in terms of the CAK force multipliers).

References
----------
 * Castor, Abbott and Klein 1975 (CAK75)
   'Radiation-driven winds in Of stars', Astrophysical Journal,
   1975, 195, 157-174
 * Friend and Abbott 1986 (FA86)
   'The theory of radiatively driven stellar winds. III - Wind models with
   finite disk correction and rotation', Astrophysical Journal,
   1986, 311, 701-707
 * Kudritzki, Pauldrach, Puls and Abbott 1989 (KPPA89)
   'Radiation-driven winds of hot stars. VI - Analytical solutions for wind
   models including the finite cone angle effect', Astronomy & Astrophysics,
   1989, 219, 205-218
 * Noebauer and Sim 2015 (NS15)
   'Self-consistent modelling of line-driven hot-star winds with Monte Carlo
   radiation hydrodynamics', Monthly Notices of the Royal Astronomical Society,
   2015, 453, 3120-3134
 * Lamers and Cassinelli 1999 (LC99)
   'Introduction to Stellar Winds', Cambridge University Press, 1999
"""
from __future__ import print_function
import numpy as np
import scipy.integrate as integ
import astropy.units as units
import astropy.constants as csts
from astropy.utils.decorators import lazyproperty


def _test_unit(val, final_unit_string, initial_unit_string):
    """helper routine to add unit to a quantity and convert to different units

    If val is not yet an astropy.units.quantity, it is assumed that it is
    given in units specified by the initial_unit_string. The input is then
    returned after converting to the units given by the final_unit_string.

    Parameters
    ----------
    val : float, int, np.ndarray
        scalar or vector input, can either already be an astropy quantity or
        not
    final_unit_string : str
        string describing the desired final units for the input
    initial_unit_string : str
        string describing the assumed initial units of the input

    Returns
    -------
    res : astropy.units.quantity
        input converted to units given by final_unit_string
    """
    try:
        val.to(final_unit_string)
    except AttributeError:
        val = val * units.Unit(initial_unit_string)
    res = val.to(final_unit_string)

    return res


class StarBase(object):
    """Base class containing the fundamental properties of the central star

    Parameters
    ---------
    mass : float, astropy.units.quantity
        stellar mass, if dimensionless, it is assumed to be in units of solar
        masses (default 52.5)
    lum : float, astropy.units.quantity
        stellar luminosity, if dimensionless, it is assumed to be in units of
        solar luminosities (default 1e6)
    teff : float, astropy.units.quantity
        effective temperature, if dimensionless, it is assumed to be Kelvin
        (default 4.2e4)
    gamma : float
        Eddington factor due to electron scattering (default 0)
    sigma : float, astropy.units.quantity
        reference specific electron scattering cross section, if dimensionless
        it is assumed to be in units of cm^2/g (default 0.3)
    """
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
        """stellar radius"""
        rad = np.sqrt(self.lum /
                      (4 * np.pi * csts.sigma_sb * self.teff**4))
        return rad

    @lazyproperty
    def vth(self):
        """thermal velocity (see LC99, eqs. 8.8, 8.83)"""
        vth = np.sqrt(2. * self.teff * csts.k_B / csts.u)
        return vth

    @lazyproperty
    def vesc(self):
        """escape velocity from stellar surface, accounting for electron
        scattering (see LC99, eq. 2.39)"""
        vesc = np.sqrt(2. * csts.G * self.mass * (1. - self.gamma) / self.rad)
        return vesc


class WindBase(object):
    """Base class containing the fundamental properties of the wind

    Parameters
    ----------
    alpha : float
        CAK force multiplier parameter, see LC99, eq. 8.86 (default 0.6)
    k : float
        CAK force multiplier parameter, see LC99, eq. 8.86 (default 0.5)
    """
    def __init__(self, alpha=0.6, k=0.5):

        self.alpha = alpha
        self.k = k


class WindStructureBase(object):
    """Base class describing the basic structure of a star+wind system



    Parameters
    ----------
    mstar : float, astropy.units.quantity
        stellar mass, see StarBase for details (default 52.5)
    lstar : float, astropy.units.quantity
        stellar luminosity, see StarBase for details (default 1e6)
    teff : float, astropy.units.quantity
        effective temperature, see StarBase for details (default 4.2e4)
    alpha : float
        force multiplier parameter, see WindBase for details (default, 0.6)
    k : float
        force multiplier parameter, see WindBase for details (default, 0.5)
    gamma : float
        Eddington factor, see StarBase for details (default 0)
    sigma : float, astropy.units.quantity
        reference electron scattering cross section, see StarBase for details
        (default 0.3)
    """
    def __init__(self, mstar=52.5, lstar=1e6, teff=4.2e4, alpha=0.6, k=0.5,
                 gamma=0, sigma=0.3):

        self.star = StarBase(mass=mstar, lum=lstar, teff=teff, gamma=gamma,
                             sigma=sigma)
        self.wind = WindBase(alpha=alpha, k=k)

    @lazyproperty
    def eta(self):
        """wind efficiency (see LC99, eq. 8.20)"""
        eta = (self.mdot * self.vterm * csts.c / self.star.lum).to("").value
        return eta

    @lazyproperty
    def t(self):
        """CAK dimensionless optical depth, c.f. LC99, eq. 8.82 and 8.104"""
        t = (self.mdot * self.star.sigma * self.star.vth /
             (2. * np.pi * self.vterm**2 * self.star.rad)).to("")
        return t

    @lazyproperty
    def m(self):
        """CAK force multiplier, assuming frozen-in ionization, c.f. LC99, eq.
        8.86 with delta=0"""
        m = self.wind.k * self.t**(-self.wind.alpha)
        return m


class BaseVelocityDensityMixin(object):
    """Mixin class providing routines to calculate the wind velocity and
    density structure"""
    def v(self, x):
        """calculate wind velocity according to CAK at given location

        C.f. LC99, eq. 8.104

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

        C.f. LC99, eq. 3.1

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
    """Mixin class providing the CAK mass loss rate and terminal wind speed"""

    @lazyproperty
    def mdot_cak(self):
        """Mass-loss rate according to CAK75, see LC99, eq. 8.105"""

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
        """Terminal wind speed according to CAK75, see LC99, eq. 8.104"""

        vterm_cak = (np.sqrt(self.wind.alpha / (1. - self.wind.alpha)) *
                     self.star.vesc)

        return vterm_cak.to("km/s")


class WindStructureCak75(WindStructureBase, BaseCakStructureMixin,
                         BaseVelocityDensityMixin):
    """Wind Structure Calculator based on the approach by CAK75.

    The wind structure is determined based on a CAK line-driving force,
    assuming a frozen-in ionization state and a central point source.

    Paramters
    ---------
    see WindStructureBase
    """
    def __init__(self, mstar=52.5, lstar=1e6, teff=4.2e4, alpha=0.6, k=0.5,
                 gamma=0, sigma=0.3):
        super(WindStructureCak75, self).__init__(mstar=mstar, lstar=lstar,
                                                 teff=teff, alpha=alpha,
                                                 k=k, gamma=gamma,
                                                 sigma=sigma)

    @property
    def mdot(self):
        """mass loss rate, equal to basic CAK mass loss rate"""
        return self.mdot_cak.to("Msun/yr")

    @property
    def vterm(self):
        """terminal wind speed, equal to basic CAK terminal velocity"""
        return self.vterm_cak.to("km/s")


class WindStructureKppa89(WindStructureBase, BaseCakStructureMixin,
                          BaseVelocityDensityMixin):
    """Wind Structure Calculator based on the approach by KPPA89.

    The wind structure is determined based on a CAK line-driving force,
    assuming a frozen-in ionization state but taking the finite size of the
    central star into account.

    Paramters
    ---------
    see WindStructureBase
    beta : float
        exponent for the beta-type velocity law (default 0.8, see LC99, sec.
        8.9.2 iii)
    """
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
        """mass loss rate, see KPPA89, eq. 31"""

        mdot = self.f1**(1. / self.wind.alpha) * self.mdot_cak

        return mdot.to("Msun/yr")

    @lazyproperty
    def vterm(self):
        """terminal wind speed, see KPPA89, eq. 39"""

        vterm = self.vterm_cak * np.sqrt(integ.quad(self.z, 0, 1)[0])

        return vterm.to("km/s")

    def h(self, x):
        """see KPPA89, eq. 14"""

        return (x - 1.) / self.beta

    def f(self, x):
        """see KPPA89, eqs. 16, 15"""

        return (1. / (self.wind.alpha + 1.) * x**2 / (1. - self.h(x)) *
                (1. - (1. - 1. / x**2 + self.h(x) / x**2)**(
                    self.wind.alpha + 1.)))

    def fn(self, x):
        """see KPPA89, eq. 35"""

        return self.f(x) / self.f1

    def z(self, u):
        """see KPPA89, eq. 36"""

        x = 1. / u
        z = (self.fn(x)**(1. / (1. - self.wind.alpha)) *
             (1. + np.sqrt(2. / self.wind.alpha *
                           (1. - (1. / self.fn(x))**(
                               1. / (1. - self.wind.alpha))))))
        return z

    def _v_scalar(self, x):
        """see KPPA89, eq. 36"""

        u = 1. / x
        I = integ.quad(self.z, u, 1)[0]
        vesc2 = (2. * csts.G * self.star.mass *
                 (1. - self.star.gamma) / self.star.rad)
        v = np.sqrt(self.wind.alpha / (1. - self.wind.alpha) * vesc2 * I).to(
            "km/s")

        return v.value

    def v(self, x):
        """calculate wind velocity according to KPPA89 at given location

        Parameters
        ----------
        x : float, np.ndarray
            dimensionless position, i.e. r/Rstar

        Returns
        -------
        v : float, np.ndarry
            wind velocity
        """

        if type(x) is np.ndarray:
            v = np.array([self._v_scalar(xi) for xi in x])
        else:
            v = self._v_scalar(x)

        v = v * units.km / units.s
        return v.to("km/s")


class WindStructureFa86(WindStructureBase, BaseCakStructureMixin,
                        BaseVelocityDensityMixin):
    """Wind Structure Calculator based on the approach by FA86.

    The wind structure is determined based on a CAK line-driving force,
    assuming a frozen-in ionization state but taking the finite size of the
    central star into account. All expressions for the wind properties result
    from fits to the numerical simulations as presented by FA86.

    Paramters
    ---------
    see WindStructureBase
    """
    def __init__(self, mstar=52.5, lstar=1e6, teff=4.2e4, alpha=0.6, k=0.5,
                 gamma=0, sigma=0.3):
        super(WindStructureFa86, self).__init__(mstar=mstar, lstar=lstar,
                                                teff=teff, alpha=alpha,
                                                k=k, gamma=gamma,
                                                sigma=sigma)

    @lazyproperty
    def mdot(self):
        """see FA86, eq. 9"""

        mdot = (self.mdot_cak * 0.5 *
                (self.star.vesc / (1e3 * units.km / units.s))**(-0.3))

        return mdot.to("Msun/yr")

    @lazyproperty
    def vterm(self):
        """see FA86, eq. 8"""

        vterm = (self.star.vesc * 2.2 * self.wind.alpha /
                 (1. - self.wind.alpha) *
                 (self.star.vesc / (1e3 * units.km / units.s))**0.2)

        return vterm.to("km/s")

    def v(self, x):
        """calculate wind velocity according to KPPA89 at given location

        See FA86, eq. 11

        Parameters
        ----------
        x : float, np.ndarray
            dimensionless position, i.e. r/Rstar

        Returns
        -------
        v : float, np.ndarry
            wind velocity
        """
        v = self.vterm * (1. - 1. / x)**(0.8)

        return v.to("km/s")


def example():
    """Example application of the wind structure calculators.

    The parameters are adopted from NS15, table 3 and are appropriate for the
    O star zeta-Puppis.
    """

    import matplotlib
    import os
    if "DISPLAY" not in os.environ:
        matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    plt.rcParams["text.usetex"] = False

    mstar = 52.5 * units.solMass
    lstar = 1e6 * units.solLum
    teff = 4.2e4 * units.K
    sigma = 0.3 * units.cm**2 / units.g

    k = 0.381
    alpha = 0.595
    gamma = 0.502

    x = np.logspace(-2, 2, 512) + 1

    wind_cak75 = WindStructureCak75(mstar=mstar, lstar=lstar, teff=teff, k=k,
                                    alpha=alpha, gamma=gamma, sigma=sigma)
    wind_fa86 = WindStructureFa86(mstar=mstar, lstar=lstar, teff=teff, k=k,
                                  alpha=alpha, gamma=gamma, sigma=sigma)
    wind_kppa89 = WindStructureKppa89(mstar=mstar, lstar=lstar, teff=teff, k=k,
                                      alpha=alpha, gamma=gamma, sigma=sigma)

    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    plt.subplot(221)
    plt.plot(x, wind_cak75.v(x), label="CAK75")
    plt.plot(x, wind_fa86.v(x), ls="dashed", label="FA86")
    plt.plot(x, wind_kppa89.v(x), ls="dashdot", label="KPPA86")
    plt.yscale("log")
    plt.ylim([1e2, 3e3])
    plt.xlim([0.8, 10])
    plt.xlabel(r"$r/R_{\star}$")
    plt.ylabel(r"$v$ [km/s]")
    plt.legend(frameon=False)

    plt.subplot(222)
    plt.plot(x - 1, wind_cak75.v(x) / wind_cak75.vterm, label="CAK75")
    plt.plot(x - 1, wind_fa86.v(x) / wind_fa86.vterm, ls="dashed",
             label="FA86")
    plt.plot(x - 1, wind_kppa89.v(x) / wind_kppa89.vterm, ls="dashdot",
             label="KPPA86")
    plt.xscale("log")
    plt.xlim([1e-2, 1e2])
    plt.ylim([0, 1])
    plt.xlabel(r"$r/R_{\star} - 1$")
    plt.ylabel(r"$v/v_{\infty}$")

    plt.subplot(223)
    plt.plot(x, wind_cak75.rho(x), label="CAK75")
    plt.plot(x, wind_fa86.rho(x), ls="dashed", label="FA86")
    plt.plot(x, wind_kppa89.rho(x), ls="dashdot", label="KPPA86")
    plt.yscale("log")
    plt.ylim([1e-15, 1e-10])
    plt.xlim([0.8, 10])
    plt.xlabel(r"$r/R_{\star}$")
    plt.ylabel(r"$\rho$ $[\mathrm{g\,cm^{-3}}]$")

    plt.subplot(224)
    plt.plot(x, wind_cak75.mdot * np.ones(len(x)))
    plt.plot(x, wind_fa86.mdot * np.ones(len(x)), ls="dashed")
    plt.plot(x, wind_kppa89.mdot * np.ones(len(x)), ls="dashdot")
    plt.yscale("log")
    plt.xlim([0.8, 10])
    plt.ylim([1e-5, 1e-4])
    plt.xlabel(r"$r/R_{\star}$")
    plt.ylabel(r"$\dot M$ $[\mathrm{M_{\odot}\,yr^{-1}}]$")
    fig.savefig("wind_structure_example.pdf")
    plt.show()


if __name__ == "__main__":

    example()
