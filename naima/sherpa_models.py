# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import astropy.units as u

from sherpa.models.parameter import Parameter
from sherpa.models.model import ArithmeticModel, modelCacher1d

from . import models
from .utils import trapz_loglog

__all__ = ['InverseCompton', 'Synchrotron', 'PionDecay', 'Bremsstrahlung']


def _mergex(xlo, xhi):
    """
    We are assuming that points are consecutive, so that xlo[n]=xhi[n-1]
    This is usually valid for fits from a single spectrum, but breaks for
    simultaneous multiwavelength fitting
    """
    N = xlo.size
    x = np.zeros(N + 1)
    x[:N] = xlo.copy()
    x[-1] = xhi[-1]

    return x


class SherpaModel(ArithmeticModel):
    """ Base class for Sherpa models
    """

    def guess(self, dep, *args, **kwargs):
        # guess normalization from total flux
        if len(args) == 1:
            xlo = args[0]
            xhi = None
        else:
            xlo, xhi = args

        model = self.calc([p.val for p in self.pars], xlo, xhi)
        obsflux = trapz_loglog(dep, xlo)
        modflux = trapz_loglog(model, xlo)
        self.ampl.set(self.ampl.val * obsflux / modflux)

    @modelCacher1d
    def calc(self, p, x, xhi=None):
        # Sherpa provides xlo, xhi in KeV, we merge into a single array if bins
        # required
        if xhi is None:
            Eph = x * u.keV
        else:
            Eph = _mergex(x, xhi) * u.keV

        model = self.flux(p, Eph)

        # Do a trapz integration to obtain the photons per bin
        if xhi is None:
            photons = (model * Eph).to('1/(s cm2)').value
        else:
            photons = trapz_loglog(
                model, Eph, intervals=True).to('1/(s cm2)').value

        if p[-1]:
            print(self.thawedpars,
                  trapz_loglog(Eph * model, Eph).to('erg/(s cm2)'))

        return photons


class SherpaModelECPL(SherpaModel):
    """ Base class for Sherpa models with a PL or ECPL particle distribution
    """

    def __init__(self, name='Model'):
        self.name = name
        # Initialize ECPL parameters
        self.index = Parameter(name, 'index', 2.1, min=-10, max=10)
        self.ref = Parameter(name, 'ref', 60, min=0, frozen=True, units='TeV')
        self.ampl = Parameter(
            name, 'ampl', 100, min=0, max=1e60, hard_max=1e100, units='1e30/eV'
        )
        self.cutoff = Parameter(
            name, 'cutoff', 0, min=0, frozen=True, units='TeV')
        self.beta = Parameter(name, 'beta', 1, min=0, max=10, frozen=True)
        self.distance = Parameter(
            name, 'distance', 1, min=0, max=1e6, frozen=True, units='kpc')
        self.verbose = Parameter(name, 'verbose', 0, min=0, max=1, frozen=True)

    @staticmethod
    def _pdist(p):
        """ Return PL or ECPL instance based on parameters p """
        index, ref, ampl, cutoff, beta = p[:5]
        if cutoff == 0.0:
            pdist = models.PowerLaw(ampl * 1e30 * u.Unit('1/eV'), ref * u.TeV,
                                    index)
        else:
            pdist = models.ExponentialCutoffPowerLaw(
                ampl * 1e30 * u.Unit('1/eV'),
                ref * u.TeV,
                index,
                cutoff * u.TeV,
                beta=beta)
        return pdist


class InverseCompton(SherpaModelECPL):
    """ Sherpa model for Inverse Compton emission from a Power Law or
    Exponential Cutoff PowerLaw particle distribution

    See the :ref:`radiative`, `naima.models.InverseCompton` and
    `naima.models.ExponentialCutoffPowerLaw` documentation.
    """

    def __init__(self, name='IC'):
        self.name = name
        self.TFIR = Parameter(name, 'TFIR', 30, min=0, frozen=True, units='K')
        self.uFIR = Parameter(
            name, 'uFIR', 0.0, min=0, frozen=True,
            units='eV/cm3')  # , 0.2eV/cm3 typical in outer disk
        self.TNIR = Parameter(
            name, 'TNIR', 3000, min=0, frozen=True, units='K')
        self.uNIR = Parameter(
            name, 'uNIR', 0.0, min=0, frozen=True,
            units='eV/cm3')  # , 0.2eV/cm3 typical in outer disk
        # add ECPL params
        super(InverseCompton, self).__init__(name=name)
        # Initialize model
        ArithmeticModel.__init__(self, name, (
            self.index, self.ref, self.ampl, self.cutoff, self.beta, self.TFIR,
            self.uFIR, self.TNIR, self.uNIR, self.distance, self.verbose))
        self._use_caching = True
        self.cache = 10

    def flux(self, p, Eph):
        (index, ref, ampl, cutoff, beta, TFIR, uFIR, TNIR, uNIR, distance,
         verbose) = p

        # Build seedspec definition
        seedspec = ['CMB']
        if uFIR > 0.0:
            seedspec.append(['FIR', TFIR * u.K, uFIR * u.eV / u.cm**3])
        if uNIR > 0.0:
            seedspec.append(['NIR', TNIR * u.K, uNIR * u.eV / u.cm**3])

        ic = models.InverseCompton(
            self._pdist(p),
            seed_photon_fields=seedspec,
            Eemin=1 * u.GeV,
            Eemax=100 * u.TeV,
            Eed=100)

        return ic.flux(Eph, distance=distance * u.kpc).to('1/(s cm2 keV)')


class Synchrotron(SherpaModelECPL):
    """ Sherpa model for Synchrotron emission from a Power Law or Exponential
    Cutoff PowerLaw particle distribution

    See the :ref:`radiative`, `naima.models.Synchrotron` and
    `naima.models.ExponentialCutoffPowerLaw` documentation.
    """

    def __init__(self, name='Sync'):
        self.name = name
        self.B = Parameter(name, 'B', 1, min=0, max=10, frozen=True, units='G')
        # add ECPL params
        super(Synchrotron, self).__init__(name=name)
        # Initialize model
        ArithmeticModel.__init__(self, name, (self.index, self.ref, self.ampl,
                                              self.cutoff, self.beta, self.B,
                                              self.distance, self.verbose))
        self._use_caching = True
        self.cache = 10

    def flux(self, p, Eph):
        index, ref, ampl, cutoff, beta, B, distance, verbose = p
        sy = models.Synchrotron(self._pdist(p), B=B * u.G)

        return sy.flux(Eph, distance=distance * u.kpc).to('1/(s cm2 keV)')


class Bremsstrahlung(SherpaModelECPL):
    """ Sherpa model for Bremsstrahlung emission from a Power Law or Exponential
    Cutoff PowerLaw particle distribution

    See the :ref:`radiative`, `naima.models.Bremsstrahlung` and
    `naima.models.ExponentialCutoffPowerLaw` documentation.
    """

    def __init__(self, name='Bremsstrahlung'):
        self.name = name
        self.n0 = Parameter(
            name, 'n0', 1, min=0, max=1e20, frozen=True, units='1/cm3')
        self.weight_ee = Parameter(
            name, 'weight_ee', 1.088, min=0, max=10, frozen=True)
        self.weight_ep = Parameter(
            name, 'weight_ep', 1.263, min=0, max=10, frozen=True)
        # add ECPL params
        super(Bremsstrahlung, self).__init__(name=name)
        # Initialize model
        ArithmeticModel.__init__(self, name, (
            self.index, self.ref, self.ampl, self.cutoff, self.beta, self.n0,
            self.weight_ee, self.weight_ep, self.distance, self.verbose))
        self._use_caching = True
        self.cache = 10

    def flux(self, p, Eph):
        (index, ref, ampl, cutoff, beta, n0, weight_ee, weight_ep, distance,
         verbose) = p
        brems = models.Bremsstrahlung(
            self._pdist(p),
            n0=n0 / u.cm**3,
            weight_ee=weight_ee,
            weight_ep=weight_ep)

        return brems.flux(Eph, distance=distance * u.kpc).to('1/(s cm2 keV)')


class PionDecay(SherpaModelECPL):
    """ Sherpa model for Pion Decay emission from a Power Law or Exponential
    Cutoff PowerLaw particle distribution

    See the :ref:`radiative`, `naima.models.PionDecay` and
    `naima.models.ExponentialCutoffPowerLaw` documentation.
    """

    def __init__(self, name='pp'):
        self.name = name
        self.nh = Parameter(name, 'nH', 1, min=0, frozen=True, units='1/cm3')
        # add ECPL params
        super(PionDecay, self).__init__(name=name)
        # Initialize model
        ArithmeticModel.__init__(self, name, (self.index, self.ref, self.ampl,
                                              self.cutoff, self.beta, self.nh,
                                              self.distance, self.verbose))
        self._use_caching = True
        self.cache = 10

    def flux(self, p, Eph):
        index, ref, ampl, cutoff, beta, nh, distance, verbose = p
        pp = models.PionDecay(self._pdist(p), nh=nh * u.Unit('1/cm3'))

        return pp.flux(Eph, distance=distance * u.kpc).to('1/(s cm2 keV)')
