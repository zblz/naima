# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from .extern.validator import validate_scalar, validate_array

__all__ = ['ElectronOZM', 'ProtonOZM']

from astropy.extern import six
import logging
# Get a new logger to avoid changing the level of the astropy logger
log = logging.getLogger('gammafit.onezone')

# Constants and units
from astropy import units as u
# import constant values from astropy.constants
import astropy.constants as const
from astropy.constants import c, G, m_e, h, hbar, k_B, R_sun, sigma_sb, e, m_p
e = e.gauss

mec2 = (m_e * c ** 2).cgs

erad = (e ** 2. / mec2).cgs
sigt = (8 * np.pi / 3) * erad ** 2
ar = (4 * sigma_sb / c).to('erg/(cm3 K4)')

heaviside = lambda x: (np.sign(x) + 1) / 2.


class ElectronOZM(object):

    r"""Synchrotron and IC emission from a leptonic population.

    Computation of electron spectrum evolution and synchrotron and IC radiation
    from a homogeneous emitter.

    The particle distribution function has the form:

    .. math::

        \frac{dN(E_e)}{dE_e} = A\left(\frac{E_e}{E_0}\right)^{-\Gamma}
        \exp{(E_e/E_\mathrm{cutoff})^\beta},

    where :math:`A` is the normalization at :math:`E_0` and is embedded in the
    parameter `norm`, :math:`E_0` is the normalization energy (`norm_energy`),
    :math:`\Gamma` is the power-law index (`index`), :math:`E_\mathrm{cutoff}`
    is the cutoff energy (`cutoff`), and :math:`\beta` is the exponent of the
    exponential cutoff (`beta`).

    Parameters
    ----------
    Eph : :class:`~astropy.units.quantity.Quantity` array instance
        Array of desired output photon energies [eV].

    norm : float
        Normalization of emitted spectrum [1/cm2]. Defined as

        .. math::

            \mathcal{N} = \frac{A V}{4 \pi d^2}

        where :math:`A` is the normalization of the non-thermal particle
        distribution [1/cm3/eV] at enery `norm_energy`, :math:`V` is the
        emitting volume, and :math:`d` is the distance to the source.

    norm_energy : :class:`~astropy.units.quantity.Quantity` float instance, optional
        Electron energy [eV] for which normalization parameter :math:`A`
        applies. Should correspond to the decorrelation energy of the observed
        spectrum for the emission process in consideration.

    index : float (optional)
        Power-law index of the particle distribution function.

    cutoff : :class:`~astropy.units.quantity.Quantity` float instance, optional
        Cut-off energy [eV].

    beta : float (optional)
        Exponent of exponential energy cutoff argument.

    B : :class:`~astropy.units.quantity.Quantity` float instance, optional
        Isotropic magnetic field strength. Default: equipartition
        with CMB (3.24e-6 G)

    seedspec: string or iterable of strings (optional)
        A list of gray-body seed spectra to use for IC calculation.
        Each of the items of the iterable can be:

        - A string equal to ``CMB`` (default), ``NIR``, or ``FIR``, for which
          radiation fields with temperatures of 2.72 K, 70 K, and 5000 K, and
          energy densities of 0.261, 0.5, and 1 eV/cm³ will be used
        - A list of length three composed of:
            1. A name for the seed photon field
            2. Its temperature as a :class:`~astropy.units.Quantity` float
               instance.
            3. Its photon field energy density as a
               :class:`~astropy.units.Quantity` float instance. If the photon
               field energy density if set to 0, its blackbody energy density
               will be computed through the Stefan-Boltzman law.

    evolve_nelec : bool (optional)
        Whether to evolve electron spectrum until steady state. See Zabalza et
        al (2011), A&A 527, 9, and Khangulyan et al (2007) MNRAS 380, 320, for a
        detailed explanation of steady-state electron spectrum computation.


    Other parameters
    ----------------

    nbb : int (optional)
        Number of spectral points to be computed for seed spectra blackbody.
        Default: 10

    gmin : float (optional)
        Minimum electron energy in units of mc2. Default: 1e4

    gmax : float (optional)
        Maximum electron energy in units of mc2. Default: 3e10

    ngamd : int (optional)
        Number of electron spectrum points per energy decade. Critical for
        accurate IC spectrum in the deep Klein-Nishina regime. Default: 300

    glocut : float (optional)
        Low energy cutoff of injection spectrum in units of mec2. Electron can
        evolve down to ``gmin``, but will not be injected below ``glocut``.
        Default: 20 (10 MeV)

    Attributes
    ----------
    specsy : :class:`~astropy.units.quantity.Quantity` array instance [1/s/eV]
        Differential synchrotron spectrum:
        emitted synchrotron photons per unit
        energy per second at energies given by ``Eph``.

    sedsy : :class:`~astropy.units.quantity.Quantity` array instance [erg/s]
        Synchrotron Spectral Energy Distribution.

    specic : :class:`~astropy.units.quantity.Quantity` array instance [1/s/eV]
        Differential IC spectrum: emitted IC photons
        per unit energy per second
        at energies given by ``Eph``.

    sedic : :class:`~astropy.units.quantity.Quantity` array instance [erg/s]
        IC SED

    We : :class:`~astropy.units.quantity.Quantity` float instance [erg]
        Total energy in electrons.
    """

    def __init__(self,
                 outspecene,  # emitted photon energy array
                 norm,       # normalization
                 # Default parameter values #####
                 # injection
                 norm_energy=20e12 * u.eV,
                 # corresponding to a scattered energy of 1 TeV
                 index=2.0,
                 cutoff=30e16 * u.eV,  # Default to no cutoff in TeV region
                 beta=1.0,
                 # emitter physical properties
                 B=np.sqrt(8 * np.pi * 4.1817e-13) * u.G,
                 # equipartition with CMB energy density (G)
                 seedspec=['CMB', ],
                 remit=1e15 * u.cm,
                 # evolve particle spectrum to steady-state?
                 evolve_nelec=False,
                 #
                 nolog=False, debug=False, **kwargs):

        if nolog:
            log.setLevel(logging.FATAL)
        elif debug and not nolog:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)

        del debug

        computation_defaults = {
            # Seed spectrum properties
            'nbb': 10,
            'tad': 1e30 * u.s,
            # electron spectrum matrix (lorentz factors)
            'gmin': 1e4,
            'gmax': 3e10,
            'ngamd': 300,  # electron spectrum points per decade
            # Injection spectrum
            'glocut': (10 * u.MeV / mec2).decompose(),  # sharp low energy cutoff at gamma*mec2 = 10MeV
        }

        self.__dict__.update(**computation_defaults)
        self.__dict__.update(**locals())
        self.__dict__.update(**kwargs)

        validate_array('outspecene', self.outspecene,
                       domain='positive', ndim=1, physical_type='energy')
        validate_scalar('norm_energy', self.norm_energy,
                        domain='positive', physical_type='energy')
        validate_scalar(
            'cutoff', self.cutoff, domain='positive', physical_type='energy')
        validate_scalar(
            'B', self.B, domain='positive', physical_type='magnetic flux density')
        validate_scalar(
            'remit', self.remit, domain='positive', physical_type='length')
        validate_scalar(
            'tad', self.tad, domain='positive', physical_type='time')

        self._process_input_seed()

    def _process_input_seed(self):
        """
        take input list of seedspecs and fix them into usable format
        """

        Tcmb = 2.72548 * u.K  # 0.00057 K
        Tfir = 70 * u.K
        ufir = 0.2 * u.eV / u.cm ** 3
        Tnir = 5000 * u.K
        unir = 0.2 * u.eV / u.cm ** 3

        # Allow for seedspec definitions of the type 'CMB-NIR-FIR' or 'CMB'
        if type(self.seedspec) != list:
            self.seedspec = self.seedspec.split('-')

        self.seeduf = {}
        self.seedT = {}
        for idx, inseed in enumerate(self.seedspec):
            if isinstance(inseed, six.string_types):
                if inseed == 'CMB':
                    self.seedT[inseed] = Tcmb
                    self.seeduf[inseed] = 1.0
                elif inseed == 'FIR':
                    self.seedT[inseed] = Tfir
                    self.seeduf[inseed] = (ufir / (ar * Tfir ** 4)).decompose()
                elif inseed == 'NIR':
                    self.seedT[inseed] = Tnir
                    self.seeduf[inseed] = (unir / (ar * Tnir ** 4)).decompose()
                else:
                    log.warn('Will not use seed {0} because it is not '
                             'CMB, FIR or NIR'.format(inseed))
                    raise TypeError
            elif type(inseed) == list and len(inseed) == 3:
                name, T, uu = inseed
                validate_scalar('{0}-T'.format(name), T, domain='positive',
                                physical_type='temperature')
                self.seedspec[idx] = name
                self.seedT[name] = T
                if uu == 0:
                    self.seeduf[name] = 1.0
                else:
                    validate_scalar(
                        '{0}-u'.format(name), uu, domain='positive',
                        physical_type='pressure')  # pressure has same physical type as energy density
                    self.seeduf[name] = (uu / (ar * T ** 4)).decompose()
            else:
                log.warn(
                    'Unable to process seed photon field: {0}'.format(inseed))
                raise TypeError

    def generate_gam(self):
        """
        Generate gamma values
        """
        ngam = int(np.log10(self.gmax / self.gmin)) * self.ngamd
        self.gam = np.logspace(np.log10(self.gmin), np.log10(self.gmax), ngam)

    def _gdot_iso_ic_on_planck(self, electron_energy, soft_photon_temperature):
        """
        IC energy losses computation for isotropic interaction with a
        blackbody photon spectrum following Khangulyan, Aharonian, and
        Kelner 2013 (arXiv:1310.7971).
        electron_energy and soft_photon_temperature are in units of m_ec^2
        """
        Ktomec2 = 1.6863699549e-10
        soft_photon_temperature *= Ktomec2

        def g(x, a):
            tmp = 1 + a[2] * x ** a[3]
            tmp2 = a[0] * x ** a[1] / tmp + 1.
            return 1. / tmp2
        aiso = [0.682, -0.362, 1.281, 0.826]
        ciso = 5.68
        t = 4. * electron_energy * soft_photon_temperature
        G0 = ciso * t * np.log(1 + 0.722 * t / ciso) / (1 + ciso * t / 0.822)
        tmp = 2.63187357438e+16 * soft_photon_temperature ** 2
        Edotiso = tmp * G0 * g(t, aiso)
        return Edotiso

    def calc_gdot(self):
        """
        Compute electron synchrotron and IC energy losses
        """
# Calculem Qinj i generem el self.gam correcte!
        if not hasattr(self, 'gam'):
            log.info('Generating gam...')
            self.generate_gam()

# Synchrotron losses
        # astropy cgs magnetic units (in particular B**2) do not convert properly
        # strip units
        umag = self.B.to('G').value ** 2. / (8. * np.pi)
        cgdot = -(4. / 3.) * c.cgs.value * \
            sigt.cgs.value / mec2.cgs.value * self.gam ** 2.
        gdotsy = cgdot * umag * u.Unit('1/s')
        # gdoticthom = cgdot*np.sum(self.phe)
        # self.ticthom = self.gam/np.abs(gdoticthom)

# Adiabatic losses
        gdotad = -1.0 * self.gam / self.tad

# IC losses
        gdotic = np.zeros_like(self.gam) * u.Unit('1/s')
        for seedspec in self.seedspec:
            gdot = (self.seeduf[seedspec] *
                    self._gdot_iso_ic_on_planck(self.gam, self.seedT[seedspec].to('K').value)) * u.Unit('1/s')
            setattr(self, 'tic_' + seedspec, self.gam / np.abs(gdot))
            gdotic += gdot

        self.gdot = np.abs(gdotsy + gdotic + gdotad)

        self.tsy = self.gam / np.abs(gdotsy)
        self.tic = self.gam / np.abs(gdotic)

        self.ttot = self.gam / np.abs(self.gdot)

    def _calc_qinj(self):
        """
        Compute injection spectrum, return as array.
        """

        # convert parameters to gamma
        cutoff_gam = (self.cutoff / mec2).cgs.value
        norm_gam = (self.norm_energy / mec2).cgs.value

        qinj = self.norm * (self.gam / norm_gam) ** -self.index *\
            np.exp(-(self.gam / cutoff_gam) ** self.beta)

        qinj[np.where(self.gam < self.glocut)] = 0.

        return qinj

#       qint(Ee) = \int_Ee^\infty Q_inj(E') dE'
    def _calc_steady_state_nelec(self, qinj):
        r"""
        Evolve electron spectrum until steady state. See Zabalza et al (2011),
        A&A 527, 9, and Khangulyan et al (2007) MNRAS 380, 320, for a detailed
        explanation of steady-state electron spectrum computation.

        .. math::
            N(E_e) = \frac{1}{|\dot{\gamma}|}\int_{E_e}^{\infty} Q_{inj}(E^\prime) dE^\prime

        Parameters
        ----------
        qinj : array
            Injection spectrum.

        Output
        ------
        nelec : Steady-state electron distribution.
        """

        # Compute trapezium integration terms and then sum them (faster than
        # cumsum[::-1])
        dgam = np.diff(self.gam)
        tt = dgam * (qinj[1:] + qinj[:-1]) / 2.
        qint = np.array([np.sum(tt[i:]) for i in range(len(qinj))])
        return qint / self.gdot.value

    def calc_nelec(self):
        """
        Generate electron distribution
        """
        self.generate_gam()

        qinj = self._calc_qinj()

        if self.evolve_nelec:
            self.calc_gdot()
            log.info('calc_nelec: L_inj/4πd² = {0:.2e} erg/s/cm²'.format(
                np.trapz(qinj * self.gam * mec2, self.gam)))
            self.nelec = self._calc_steady_state_nelec(qinj)
        else:
            self.nelec = qinj

        self.We = np.trapz(self.nelec * (self.gam * mec2), self.gam)
        log.info('calc_nelec: W_e/4πd²   = {0:.2e} / cm²'.format(self.We))

    def calc_sy(self):
        """
        Compute sync for random magnetic field according to approximation of
        Aharonian, Kelner, Prosekin 2010
        """
        from scipy.special import cbrt

        if not hasattr(self, 'gam'):
            log.info('Calling calc_nelec to generate gam, nelec')
            self.calc_nelec()

        def Gtilde(x):
            """
            AKP10 Eq. D7

            Factor ~2 performance gain in using cbrt(x)**n vs x**(n/3.)
            """
            gt1 = 1.808 * cbrt(x) / np.sqrt(1 + 3.4 * cbrt(x) ** 2.)
            gt2 = 1 + 2.210 * cbrt(x) ** 2. + 0.347 * cbrt(x) ** 4.
            gt3 = 1 + 1.353 * cbrt(x) ** 2. + 0.217 * cbrt(x) ** 4.
            return gt1 * (gt2 / gt3) * np.exp(-x)

        log.debug('calc_sy: Starting synchrotron computation with AKB2010...')

        # 100 gamma points per energy decade is enough for accurate SYN
        newngam = 100 * np.log10(self.gmax / self.gmin)
        oldngam = self.gam.shape[0]
        ratio = int(np.round(oldngam / newngam))
        gam = self.gam[::ratio]
        nelec = self.nelec[::ratio]

        # strip units, ensuring correct conversion
        # astropy units do not convert correctly for gyroradius calculation when using
        # cgs (SI is fine, see https://github.com/astropy/astropy/issues/1687)
        CS1 = np.sqrt(3) * e.value ** 3 * self.B.to('G').value / (2 * np.pi * m_e.cgs.value * c.cgs.value ** 2
                                                                  * hbar.cgs.value * self.outspecene.to('erg').value)
        Ec = 3 * e.value * hbar.cgs.value * \
            self.B.to('G').value * gam ** 2 / (
                2 * (m_e * c).cgs.value)  # Critical energy, erg
        EgEc = self.outspecene.to('erg').value / np.vstack(Ec)
        dNdE = CS1 * Gtilde(EgEc)
        # return units
        spec = np.trapz(np.vstack(nelec) * dNdE, gam, axis=0) / u.s / u.erg

        # convert from 1/s/erg to 1/s/eV
        self.specsy = spec.to('1/(s eV)')
        self.sedsy = (spec * self.outspecene ** 2.).to('erg/s')

        totsylum = np.trapz(
            self.specsy * self.outspecene, self.outspecene).to('erg/s')
        log.info('calc_sy: L_sy/4πd²  = {0:.2e} / cm²'.format(totsylum))

    def _calc_specic(self, seed):
        log.debug(
            '_calc_specic: Computing IC on {0} seed photons...'.format(seed))

        def iso_ic_on_planck(electron_energy,
                             soft_photon_temperature, gamma_energy):
            """
            IC cross-section for isotropic interaction with a blackbody
            photon spectrum following Khangulyan, Aharonian, and Kelner 2013
            (arXiv:1310.7971).

            `electron_energy` and `gamma_energy` are in units of m_ec^2
            `soft_photon_temperature` is in units of K
            """
            Ktomec2 = 1.6863699549e-10
            soft_photon_temperature *= Ktomec2

            def g(x, a):
                tmp = 1 + a[2] * x ** a[3]
                tmp2 = a[0] * x ** a[1] / tmp + 1.
                return 1. / tmp2
            gamma_energy = np.vstack(gamma_energy)
            a3 = [0.192, 0.448, 0.546, 1.377]
            a4 = [1.69, 0.549, 1.06, 1.406]
            z = gamma_energy / electron_energy
            x = z / (1 - z) / (4. * electron_energy * soft_photon_temperature)
            tmp = 1.644934 * x
            F = (1.644934 + tmp) / (1. + tmp) * np.exp(-x)
            cross_section = F * (z ** 2 / (2 * (1 - z)) * g(x, a3) + g(x, a4))
            tmp = 2.6433905738281024e+16 * \
                (soft_photon_temperature / electron_energy) ** 2
            cross_section = tmp * cross_section
            condition = (
                (gamma_energy < electron_energy) * (electron_energy > 1))
            return np.where(condition, cross_section,
                            np.zeros_like(cross_section))

        uf = self.seeduf[seed]
        T = self.seedT[seed]

        Eph = (self.outspecene / mec2).cgs.value
        gamint = iso_ic_on_planck(self.gam, T.to('K').value, Eph)
        lum = uf * Eph * \
            np.trapz(self.nelec * gamint, self.gam) * u.Unit('1/s')

        return lum / self.outspecene  # return differential spectrum in 1/s/eV

    def calc_ic(self):
        """
        Compute IC spectrum using IC cross-section for isotropic interaction
        with a blackbody photon spectrum following Khangulyan, Aharonian, and
        Kelner 2013 (arXiv:1310.7971).
        """
        log.debug('calc_ic: Starting IC computation...')

        if not hasattr(self, 'gam'):
            log.info('Calling calc_nelec to generate gam, nelec')
            self.calc_nelec()

        self.specic = np.zeros(len(self.outspecene)) * u.Unit('1/(s eV)')
        self.sedic = np.zeros(len(self.outspecene)) * u.Unit('erg/s')

        for seedspec in self.seedspec:
            # Call actual computation, detached to allow changes in subclasses
            specic = self._calc_specic(seedspec).to('1/(s eV)')
            sedic = (specic * self.outspecene ** 2).to('erg/s')
            setattr(self, 'specic_' + seedspec, specic)
            setattr(self, 'sedic_' + seedspec, sedic)
            self.specic += specic
            self.sedic += sedic

        toticlum = np.trapz(
            self.specic * self.outspecene, self.outspecene).to('erg/s')
        log.info('calc_ic: L_ic/4πd²  = {0:.2e} / cm²'.format(toticlum))
        tev = np.where(self.outspecene > 100 * u.GeV)
        if len(tev[0]) > 0:
            tottevlum = np.trapz(self.specic[tev] * self.outspecene[tev],
                                 self.outspecene[tev]).to('erg/s')
            log.info('calc_ic: L_vhe/4πd² = {0:.2e} / cm²'.format(tottevlum))

    def calc_outspec(self):
        """
        Generate electron distribution and compute all spectra
        """
        self.calc_nelec()
        self.calc_sy()
        self.calc_ic()
        self.spec = self.specsy + self.specic
        self.sed = self.sedsy + self.sedic


class ProtonOZM(object):

    r"""OneZoneModel for pp interaction gamma-ray emission.

    Compute gamma-ray spectrum arising from the interaction of a relativistic
    proton distribution with stationary target protons.

    The particle distribution function has the form:

    .. math::

        \frac{dN(E_p)}{dE_p} = A\left(\frac{E_p}{E_0}\right)^{-\Gamma}
        \exp{(E_p/E_\mathrm{cutoff})^\beta},

    where :math:`A` is the normalization at :math:`E_0` and is embedded in the
    parameter `norm`, :math:`E_0` is the normalization energy (`norm_energy`),
    :math:`\Gamma` is the power-law index (`index`), :math:`E_\mathrm{cutoff}`
    is the cutoff energy (`cutoff`), and :math:`\beta` is the exponent of the
    exponential cutoff (`beta`).


    Parameters
    ----------
    Eph : :class:`~astropy.units.quantity.Quantity` array instance
        Array of desired output photon energies [eV].

    norm : float
        Normalization of emitted spectrum [1/cm2]. Defined as

        .. math::
            \mathcal{N} = \frac{A_p n_H V}{4 \pi d^2}

        where :math:`A_p` is the normalization of the non-thermal particle
        distribution [1/TeV] at enery `norm_energy`, :math:`n_H` is the
        number density of target protons, :math:`V` is the emitting volume, and
        :math:`d` is the distance to the source.

    norm_energy : :class:`~astropy.units.quantity.Quantity` float instance, optional
        Electron energy [eV] for which normalization parameter :math:`A`
        applies. Should correspond to the decorrelation energy of the observed
        spectrum for the emission process in consideration.

    index : float, optional
        Power-law index of the particle distribution function.

    cutoff : :class:`~astropy.units.quantity.Quantity` float instance, optional
        Cut-off energy [eV]. Default: None

    beta : float, optional
        Exponent of exponential energy cutoff argument. Default: 2

    Attributes
    ----------
    specpp : :class:`~astropy.units.quantity.Quantity` array instance [1/s/eV]
        Differential gamma-ray spectrum at energies given by `Eph`.

    sedpp : :class:`~astropy.units.quantity.Quantity` array instance [erg/s]
        Spectral energy distribution at energies given by `Eph`.

    Wp : :class:`~astropy.units.quantity.Quantity` float instance [erg*[1/cm5]]
        Total energy required in protons in units of erg/cm5. To obtain
        intrinsic total energy, this value should be multiplied by a factor
        4πd²/nH.

    References
    ----------

    Kelner, S.R., Aharonian, F.A., and Bugayov, V.V., 2006 PhysRevD 74, 034018 [KAB06]

    """

    def __init__(self,
                 outspecene,
                 norm,
                 # Injection spectrum properties
                 norm_energy=1e12 * u.eV,  # eV
                 index=2.0,
                 cutoff=None,  # eV
                 beta=1.0,
                 nolog=False, debug=False, **kwargs):

        if nolog:
            log.setLevel(logging.FATAL)
        elif debug and not nolog:
            log.setLevel(logging.DEBUG)
        else:
            log.setLevel(logging.INFO)

        self.__dict__.update(**locals())
        self.__dict__.update(**kwargs)

        validate_array('outspecene', self.outspecene,
                       domain='positive', ndim=1, physical_type='energy')
        validate_scalar('norm_energy', self.norm_energy,
                        domain='positive', physical_type='energy')
        if cutoff is not None:
            validate_scalar(
                'cutoff', self.cutoff, domain='positive', physical_type='energy')
        if hasattr(self, 'E_break'):
            validate_scalar(
                'E_break', self.E_break, domain='positive', physical_type='energy')

        self._update_values()

    def _update_values(self):
        """
        Convert ``norm_energy``, ``cutoff``, and ``E_break`` to TeV and save values
        as ``_norm_energy``, ``_cutoff``, and ``_E_break`` for use in
        integrands.
        """

        for var in ['norm_energy', 'cutoff', 'E_break']:
            if hasattr(self, var) and getattr(self, var) is not None:
                validate_scalar(
                    var, getattr(self, var), domain='positive', physical_type='energy')
                setattr(self, '_' + var, getattr(self, var).to('TeV').value)

    def Jp(self, Ep):
        """
        Particle distribution function [1/TeV]

        Parameters
        ----------
        Ep : :class:`~astropy.units.quantity.Quantity` array instance
            Proton energies [TeV]

        Returns
        -------
        Jp : :class:`~astropy.units.quantity.Quantity` array instance
            Particle distribution function in particles per TeV [1/TeV]
        """

        return self._Jp(Ep.to('TeV').value) * u.Unit('1/TeV')

    def _Jp(self, Ep):
        """
        Particle distribution function [1/TeV]

        Note: Quantities are not used in this function

        Parameters
        ----------
        Ep : float or array
            Eprot [TeV]

        Returns
        -------
        Jp : type(Ep)
            Particle distribution function in particles per TeV
        """

        if hasattr(self, 'index1') and hasattr(self, 'index2') and hasattr(self, '_E_break'):
            Jp = self.norm * np.where(Ep <= self._E_break,
                                     (Ep / self._norm_energy) ** -self.index1,
                                     ((self._E_break / self._norm_energy) ** (self.index2 - self.index1)
                                      * (Ep / self._norm_energy) ** -self.index2)
                                      )
        else:
            if self.cutoff is None:
                Jp = self.norm * (Ep / self._norm_energy) ** -self.index
            else:
                Jp = self.norm * ((Ep / self._norm_energy) ** -self.index *
                                  np.exp(-(Ep / self._cutoff) ** self.beta))
        return Jp

    def _Fgamma(self, x, Ep):
        """
        KAB06 Eq.58

        Note: Quantities are not used in this function

        Parameters
        ----------
        x : float
            Egamma/Eprot
        Ep : float
            Eprot [TeV]
        """
        L = np.log(Ep)
        B = 1.30 + 0.14 * L + 0.011 * L ** 2  # Eq59
        beta = (1.79 + 0.11 * L + 0.008 * L ** 2) ** -1  # Eq60
        k = (0.801 + 0.049 * L + 0.014 * L ** 2) ** -1  # Eq61
        xb = x ** beta

        F1 = B * (np.log(x) / x) * ((1 - xb) / (1 + k * xb * (1 - xb))) ** 4
        F2 = 1. / np.log(x) - (4 * beta * xb) / (1 - xb) - (
            4 * k * beta * xb * (1 - 2 * xb)) / (1 + k * xb * (1 - xb))

        return F1 * F2

    def _sigma_inel(self, Ep):
        """
        Inelastic cross-section for p-p interaction. KAB06 Eq. 73, 79

        Note: Quantities are not used in this function

        Parameters
        ----------
        Ep : float
            Eprot [TeV]

        Returns
        -------
        sigma_inel : float
            Inelastic cross-section for p-p interaction [1/cm2].

        """
        L = np.log(Ep)
        sigma = 34.3 + 1.88 * L + 0.25 * L ** 2
        if Ep <= 0.1:
            Eth = 1.22e-3
            sigma *= (1 - (Eth / Ep) ** 4) ** 2 * heaviside(Ep - Eth)
        return sigma * 1e-27  # convert from mbarn to cm2

    def _photon_integrand(self, x, Egamma):
        """
        Integrand of Eq. 72
        """
        try:
            return self._sigma_inel(Egamma / x) * self._Jp(Egamma / x) \
                * self._Fgamma(x, Egamma / x) / x
        except ZeroDivisionError:
            return np.nan

    def _calc_specpp_hiE(self, Egamma):
        """
        Spectrum computed as in Eq. 42 for Egamma >= 0.1 TeV
        """
        # Fixed quad with n=40 is about 15 times faster and is always within
        # 0.5% of the result of adaptive quad for Egamma>0.1
        # WARNING: It also produces artifacts for steep distributions (e.g.
        # Maxwellian) at ~500 GeV. Reverting to adaptative quadrature
        # from scipy.integrate import fixed_quad
        # result=c*fixed_quad(self._photon_integrand, 0., 1., args = [Egamma,
        # ], n = 40)[0]
        from scipy.integrate import quad
        Egamma = Egamma.to('TeV').value
        specpp = c.cgs.value * quad(
            self._photon_integrand, 0., 1., args=Egamma,
            epsrel=1e-3, epsabs=0)[0]

        return specpp * u.Unit('1/(s TeV)')

    # variables for delta integrand
    _c = c.cgs.value
    _Kpi = 0.17
    _mp = (m_p * c ** 2).to('TeV').value
    _m_pi = 1.349766e-4  # TeV/c2

    def _delta_integrand(self, Epi):
        Ep0 = self._mp + Epi / self._Kpi
        qpi = self._c * \
            (self.nhat / self._Kpi) * self._sigma_inel(Ep0) * self._Jp(Ep0)
        return qpi / np.sqrt(Epi ** 2 + self._m_pi ** 2)

    def _calc_specpp_loE(self, Egamma):
        """
        Delta-functional approximation for low energies Egamma < 0.1 TeV
        """
        from scipy.integrate import quad
        Egamma = Egamma.to('TeV').value
        Epimin = Egamma + self._m_pi ** 2 / (4 * Egamma)

        result = 2 * quad(self._delta_integrand, Epimin, np.inf, epsrel=1e-3,
                          epsabs=0)[0]

        return result * u.Unit('1/(s TeV)')

    def _calc_photon_spectrum(self):
        """
        Compute photon spectrum from pp interactions using Eq. 71 and Eq.58 of KAB06.
        """
        from scipy.integrate import quad
        self._update_values()

        # Before starting, show total proton energy above threshold
        Eth = 1.22e-3
        self.Wp = (quad(lambda x: x * self._Jp(x), Eth, np.Inf)[0] * u.TeV).to(
            'erg') / u.cm ** 5
        log.info('W_p(E>1.22 GeV)*[nH/4πd²] = {0:.2e}'.format(self.Wp))

        if not hasattr(self, 'Etrans'):
            # Energy at which we change from delta functional to accurate
            # calculation
            self.Etrans = 0.1 * u.TeV
        else:
            validate_scalar(
                'Etrans', self.Etrans, domain='positive', physical_type='energy')

        self.nhat = 1.  # initial value, works for index~2.1
        if np.any(self.outspecene < self.Etrans) and np.any(self.outspecene >= self.Etrans):
            # compute value of nhat so that delta functional matches accurate
            # calculation at 0.1TeV
            full = self._calc_specpp_hiE(self.Etrans)
            delta = self._calc_specpp_loE(self.Etrans)
            self.nhat *= (full / delta).decompose().value

        self.specpp = np.zeros(len(self.outspecene)) * u.Unit('1/(s TeV)')

        for i, Egamma in enumerate(self.outspecene):
            if Egamma >= self.Etrans:
                self.specpp[i] = self._calc_specpp_hiE(Egamma)
            else:
                self.specpp[i] = self._calc_specpp_loE(Egamma)

        self.sedpp = (self.specpp * self.outspecene ** 2).to('erg/s')  # erg/s
        self.specpp = self.specpp.to('1/(s eV)')

        totpplum = np.trapz(
            self.specpp * self.outspecene, self.outspecene).to('erg/s')
        log.info('L_pp*nH/4πd²  = {0:.2e} / cm2'.format(totpplum))

    def calc_outspec(self):
        """
        Compute photon spectrum from pp interactions using Eq. 71 and Eq.58 of KAB06.
        """
        self._calc_photon_spectrum()
