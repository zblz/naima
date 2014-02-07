# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import division
import numpy as np
np.seterr(all='ignore')

import logging
logging.basicConfig(level=logging.INFO)

## Constants and units
from astropy import constants
from astropy import units as u
# import constant values from astropy.constants
constants_to_import=['c','G','m_e','e','h','hbar','k_B','R_sun','sigma_sb']
cntdict={}
for cnt in constants_to_import:
    if cnt == 'e':
        value=getattr(constants,cnt).gauss.value
    else:
        value=getattr(constants,cnt).cgs.value
    cntdict[cnt]=value
globals().update(cntdict)

eV     = u.erg.to('eV')
TeV    = u.erg.to('TeV')
hz     = 1/h # hz in erg

hz2erg = h
hz2ev  = h*eV

mec2    = m_e*c**2
mec2eV  = mec2*eV    # m_e*c**2 in eV
mec2TeV = mec2eV/1e12
mec2GeV = mec2eV/1e9
Ktomec2 = 1.6863699549e-10

erad = e**2./mec2
sigt = (8*np.pi/3)*erad**2
ar   = 4*sigma_sb/c

heaviside = lambda x: (np.sign(x)+1)/2.

class _BogusLogger(object):
    # logger functions used for nolog
    def debug(self,s):
        pass
    def info(self,s):
        pass
    def warn(self,s):
        print('WARN:OneZoneModel: %s'%s)
        pass


class ElectronOZM(object):
    r"""Synchrotron and IC emission from a leptonic population

    Computation of electron spectrum evolution and synchrotron and IC radiation
    from a homogeneous emitter.

    The particle distribution function has the form:

    .. math::

        \frac{dN(E_e)}{dE_e}=A\left(\frac{E_e}{E_0}\right)^{-\Gamma}
        \exp{(E_e/E_\mathrm{cutoff})^\beta},

    where :math:`A` is the normalization at :math:`E_0` and is embedded in the
    parameter `norm`, :math:`E_0` is the normalization energy (`norm_energy`),
    :math:`\Gamma` is the power-law index (`index`), :math:`E_\mathrm{cutoff}`
    is the cutoff energy (`cutoff`), and :math:`\beta` is the exponent of the
    exponential cutoff (`beta`).

    Parameters
    ----------
    Eph : array
        Array of desired output photon energies [eV].

    norm : float
        Normalization of emitted spectrum [1/cm2]. Defined as

        .. math::

            \mathcal{N}=\frac{A V}{4 \pi d^2}

        where :math:`A` is the normalization of the non-thermal particle
        distribution [1/cm3/eV] at enery `norm_energy`, :math:`V` is the
        emitting volume, and :math:`d` is the distance to the source.

    norm_energy : float (optional)
        Electron energy [eV] for which normalization parameter :math:`A`
        applies. Should correspond to the decorrelation energy of the observed
        spectrum for the emission process in consideration.

    index : float (optional)
        Power-law index of the particle distribution function.

    cutoff : float (optional)
        Cut-off energy [eV].

    beta : float (optional)
        Exponent of exponential energy cutoff argument.

    B : float (optional)
        Isotropic magnetic field strength in microgauss. Default: equipartition with
        CMB (3.24e-6 G)

    seedspec: string or iterable of strings (optional)
        A list of gray-body seed spectra to use for IC calculation. Strings can be one or
        more of CMB, NIR, FIR, for which radiation fields with temperatures of
        2.72 K, 70 K, and 5000 K, and energy densities of 0.261, 0.5, and 1
        eV/cm:math:`^{-3}` will be used. Custom gray-bodies can be included by
        using a list composed of: name, gray body temperature (K), and photon
        field energy density (erg/cm:math:`^{-3}`). If the photon field energy
        density if set to 0, its blackbody energy density will be computed
        through the Stefan-Boltzman law. Default: ['CMB',]

    evolve_nelec : bool (optional)
        Whether to evolve electron spectrum until steady state. See
        Zabalza et al (2011), A&A 527, 9, and Khangulyan et al (2007) MNRAS 380,
        320, for a detailed explanation of steady-state electron spectrum
        computation.


    Other parameters
    ----------------

    bb : bool (optional)
        Should IC seed spectra be computed as a blackbody? If false,
        monochromatic seed spectra are used. Default: False

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
        Default: 20 (1e7 eV)

    Attributes
    ----------

    specsy : array [1/s/eV]
        Differential synchrotron spectrum: emitted synchrotron photons per unit
        energy per second at energies given by `Eph`.

    sedsy : array [erg/s]
        Synchrotron SED

    specic : array [1/s/eV]
        Differential IC spectrum: emitted IC photons per unit energy per second
        at energies given by `Eph`.

    specictev : array [1/s/TeV]
        Differential IC spectrum in units typically used by IACT community.

    sedic : array [erg/s]
        IC SED

    We : float [erg]
        Total energy in electrons.

    """


    def __init__(self,
        outspecene, # emitted photon energy array
        norm,       # normalization
        #### Default parameter values #####
        # injection
        norm_energy   = 20e12, # corresponding to a scattered energy of 1 TeV
        index         = 2.0,
        cutoff        = 30e16, # Default to no cutoff in TeV region
        beta          = 1.0,
        # emitter physical properties
        B        = np.sqrt(8*np.pi*4.1817e-13), #equipartition with CMB energy density (G)
        seedspec = ['CMB',],
        remit    = 1e15,
        # evolve particle spectrum to steady-state?
        evolve_nelec = False,
        #
        nolog=False,debug=False,**kwargs):

        if nolog:
            self.logger=_BogusLogger()
        else:
            self.logger=logging.getLogger('ElectronOZM')
            if debug:
                self.logger.setLevel(logging.DEBUG)
        del debug

        computation_defaults={
            # Seed spectrum properties
            'bb'     : False,
            'nbb'    : 10,
            'tad'    : 1e30,
            # electron spectrum matrix (lorentz factors)
            'gmin'   : 1e4,
            'gmax'   : 3e10,
            'ngamd'  : 300, # electron spectrum points per decade
            # Injection spectrum
            'glocut' : 1e7/mec2eV,  # sharp low energy cutoff at gamma*mec2 = 10MeV
            }

        self.__dict__.update(**computation_defaults)
        self.__dict__.update(**locals())
        self.__dict__.update(**kwargs)

        self._process_input_seed()

    def _process_input_seed(self):
        """
        take input list of seedspecs and fix them into usable format
        """

        Tcmb = 2.72548 # 0.00057 K
        Tfir = 70
        ufir = 0.5*u.eV.to('erg')
        Tnir = 5000
        unir = 1.0*u.eV.to('erg')

        # Allow for seedspec definitions of the type 'CMB-NIR-FIR' or 'CMB'
        if type(self.seedspec)!=list:
            self.seedspec=self.seedspec.split('-')

        self.seeduf=np.zeros(len(self.seedspec))
        self.seedT=np.zeros(len(self.seedspec))
        for idx,inseed in enumerate(self.seedspec):
            if type(inseed) == str:
                if inseed=='CMB':
                    self.seedT[idx]  = Tcmb
                    self.seeduf[idx] = 1.0
                elif inseed=='FIR':
                    self.seedT[idx]  = Tfir
                    self.seeduf[idx] = ufir/(ar*Tfir**4)
                elif inseed=='NIR':
                    self.seedT[idx]  = Tnir
                    self.seeduf[idx] = unir/(ar*Tnir**4)
            elif type(inseed)==list and len(inseed)==3:
                name,T,uu = inseed
                self.seedspec[idx] = name
                self.seedT[idx]    = T
                if uu == 0:
                    self.seeduf[idx] = 1.0
                else:
                    self.seeduf[idx] = uu/(ar*T**4)

    def generate_gam(self):
        """
        Generate gamma values
        """

        ngam=int(np.log10(self.gmax/self.gmin))*self.ngamd
        self.gam=np.logspace(np.log10(self.gmin),np.log10(self.gmax),ngam)


    def _gdot_iso_ic_on_planck(self,electron_energy,soft_photon_temperature):
        """
        IC energy losses computation for isotropic interaction with a
        blackbody photon spectrum following Khangulyan, Aharonian, and
        Kelner 2013 (arXiv:1310.7971).
        electron_energy and soft_photon_temperature are in units of m_ec^2
        """
        def g(x,a):
            tmp=1+a[2]*x**a[3]
            tmp2=a[0]*x**a[1]/tmp+1.
            return 1./tmp2
        aiso=[0.682,-0.362,1.281,0.826]
        ciso=5.68
        t=4.*electron_energy*soft_photon_temperature
        G0=ciso*t*np.log(1+0.722*t/ciso)/(1+ciso*t/0.822)
        tmp=2.63187357438e+16*soft_photon_temperature**2
        Edotiso=tmp*G0*g(t,aiso)
        return Edotiso

    def calc_gdot(self):
        """
        Compute electron synchrotron and IC energy losses
        """
# Calculem Qinj i generem el self.gam correcte!
        if not hasattr(self,'gam'):
            self.logger.info('Generating gam...')
            self.generate_gam()

## Synchrotron losses
        umag=self.B**2./(8.*np.pi)
        cgdot=-(4./3.)*c*sigt/mec2*self.gam**2.
        gdotsy=cgdot*umag
        #gdoticthom=cgdot*np.sum(self.phe)
        #self.ticthom=self.gam/np.abs(gdoticthom)

## Adiabatic losses
        gdotad=-1.0*self.gam/self.tad

## IC losses
        gdotic=np.zeros_like(self.gam)
        for idx, seedspec in enumerate(self.seedspec):
            gdot = self.seeduf[idx] * self._gdot_iso_ic_on_planck(self.gam,self.seedT[idx]*Ktomec2)
            setattr(self,'tic_'+seedspec,self.gam/np.abs(gdot))
            gdotic+=gdot

        self.gdot = np.abs(gdotsy+gdotic+gdotad)

        self.tsy  = self.gam/np.abs(gdotsy)
        self.tic  = self.gam/np.abs(gdotic)

        self.ttot = self.gam/np.abs(self.gdot)


    def _calc_qinj(self):
        """
        Compute injection spectrum, return as array.
        """

        # convert parameters to gamma
        cutoff_gam=self.cutoff/mec2eV
        norm_gam=self.norm_energy/mec2eV

        qinj=self.norm*(self.gam/norm_gam)**-self.index*np.exp(-(self.gam/cutoff_gam)**self.beta)

        qinj[np.where(self.gam<self.glocut)]=0.

        return qinj

#       qint(Ee)=\int_Ee^\infty Q_inj(E') dE'
    def _calc_steady_state_nelec(self,qinj):
        r"""
        Evolve electron spectrum until steady state. See Zabalza et al (2011),
        A&A 527, 9, and Khangulyan et al (2007) MNRAS 380, 320, for a detailed
        explanation of steady-state electron spectrum computation.

        .. math::
            N(E_e)=\frac{1}{|\dot{\gamma}|}\int_{E_e}^{\infty} Q_{inj}(E^\prime) dE^\prime

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
        dgam=np.diff(self.gam)
        tt=dgam*(qinj[1:]+qinj[:-1])/2.
        qint = np.array([np.sum(tt[i:]) for i in range(len(qinj))])
        return qint/self.gdot

    def calc_nelec(self):
        """
        Generate electron distribution
        """
        self.generate_gam()
        self.calc_gdot()

        qinj = self._calc_qinj()

        if self.evolve_nelec:
            self.logger.info('calc_nelec: L_inj/4πd² = {0:.2e} erg/s/cm²'.format(
                np.trapz(qinj*self.gam*mec2,self.gam)))
            self.nelec = self._calc_steady_state_nelec(qinj)
        else:
            self.nelec = qinj

        self.We=np.trapz(self.nelec*(self.gam*mec2),self.gam)
        self.logger.info('calc_nelec: W_e/4πd²   = {0:.2e} erg/cm²'.format(self.We))

    def calc_sy(self):
        """
        Compute sync for random magnetic field according to approximation of
        Aharonian, Kelner, Prosekin 2010
        """
        from scipy.special import cbrt

        if not hasattr(self,'outspecerg'):
            self.outspecerg=self.outspecene/eV
        if not hasattr(self,'gam'):
            self.logger.info('Calling calc_nelec to generate gam,nelec')
            self.calc_nelec()

        def Gtilde(x):
            """
            AKP10 Eq. D7

            Factor ~2 performance gain in using cbrt(x)**n vs x**(n/3.)
            """
            gt1=1.808*cbrt(x)/np.sqrt(1+3.4*cbrt(x)**2.)
            gt2=1+2.210*cbrt(x)**2.+0.347*cbrt(x)**4.
            gt3=1+1.353*cbrt(x)**2.+0.217*cbrt(x)**4.
            return gt1*(gt2/gt3)*np.exp(-x)

        self.logger.debug('calc_sy: Starting synchrotron computation with AKB2010...')

        # 100 gamma points per energy decade is enough for accurate SYN
        newngam=100*np.log10(self.gmax/self.gmin)
        oldngam=self.gam.shape[0]
        ratio=int(np.round(oldngam/newngam))
        gam=self.gam[::ratio]
        nelec=self.nelec[::ratio]

        CS1=np.sqrt(3)*e**3*self.B/(2*np.pi*m_e*c**2*hbar*self.outspecerg)
        Ec=3*e*hbar*self.B*gam**2/(2*m_e*c) # Critical energy, erg
        EgEc=self.outspecerg/np.vstack(Ec)
        dNdE=CS1*Gtilde(EgEc)
        spec=np.trapz(np.vstack(nelec)*dNdE,gam,axis=0)

        # convert from 1/s/erg to 1/s/eV
        self.specsy=spec/u.erg.to('eV')
        self.sedsy=spec*self.outspecerg**2.

        totsylum=np.trapz(self.specsy*self.outspecene,self.outspecerg)
        self.logger.info('calc_sy: L_sy/4πd²  = {0:.2e} erg/s/cm²'.format(totsylum))

    def _calc_specic(self,seed=None):
        self.logger.debug('_calc_specic: Computing IC on {0} seed photons...'.format(seed))

        def iso_ic_on_planck(electron_energy,
                soft_photon_temperature,gamma_energy):
            """
            IC cross-section for isotropic interaction with a blackbody
            photon spectrum following Khangulyan, Aharonian, and Kelner 2013
            (arXiv:1310.7971).

            `electron_energy`, `soft_photon_temperature`, and
            `gamma_energy` are in units of m_ec^2
            """
            def g(x,a):
                tmp=1+a[2]*x**a[3]
                tmp2=a[0]*x**a[1]/tmp+1.
                return 1./tmp2
            gamma_energy=np.vstack(gamma_energy)
            a3=[0.192,0.448,0.546,1.377]
            a4=[1.69,0.549,1.06,1.406]
            z=gamma_energy/electron_energy
            x=z/(1-z)/(4.*electron_energy*soft_photon_temperature)
            tmp=1.644934*x
            F=(1.644934+tmp)/(1.+tmp)*np.exp(-x)
            cross_section=F*(z**2/(2*(1-z))*g(x,a3)+g(x,a4))
            tmp=2.6433905738281024e+16*(soft_photon_temperature/electron_energy)**2
            cross_section=tmp*cross_section
            condition=((gamma_energy < electron_energy)*(electron_energy>1))
            return np.where(condition, cross_section,
                    np.zeros_like(cross_section))

        idx=self.seedspec.index(seed)
        uf=self.seeduf[idx]
        T=self.seedT[idx]*Ktomec2

        Eph=(self.outspecene/mec2eV)
        gamint = iso_ic_on_planck(self.gam,T,Eph)
        lum=uf*Eph*np.trapz(self.nelec*gamint,self.gam)

        return lum/self.outspecene # return differential spectrum in 1/s/eV

    def calc_ic(self):
        """
        Compute IC spectrum
        """
        self.logger.debug('calc_ic: Starting IC computation...')

        if not hasattr(self,'outspecerg'):
            self.outspecerg=self.outspecene/eV

        if not hasattr(self,'gam'):
            self.logger.info('Calling calc_nelec to generate gam,nelec')
            self.calc_nelec()

        for spec in ['specic','specictev','sedic']:
            setattr(self,spec,np.zeros_like(self.outspecene))

        for idx,seedspec in enumerate(self.seedspec):
            # Call actual computation, detached to allow changes in subclasses
            specic=self._calc_specic(seed=seedspec)
            specictev=specic/u.eV.to('TeV') # 1/s/TeV
            sedic=specic*self.outspecerg*self.outspecene # erg/s
            setattr(self,'specic_'+seedspec,specic)
            setattr(self,'specictev_'+seedspec,specictev)
            setattr(self,'sedic_'+seedspec,sedic)
            self.specic+=specic
            self.specictev+=specictev
            self.sedic+=sedic

        toticlum=np.trapz(self.specic*self.outspecene,self.outspecerg)
        self.logger.info('calc_ic: L_ic/4πd²  = {0:.2e} erg/s/cm²'.format(toticlum))
        tev=np.where(self.outspecene>1e11)
        if len(tev[0])>0:
            tottevlum=np.trapz(self.specic[tev]*self.outspecene[tev],self.outspecerg[tev])
            self.logger.info('calc_ic: L_vhe/4πd² = {0:.2e} erg/s/cm²'.format(tottevlum))

    def calc_outspec(self):
        """
        Generate electron distribution and compute all spectra
        """
        self.calc_nelec()
        self.calc_sy()
        self.calc_ic()
        self.spec = self.specsy + self.specic
        self.sed  = self.sedsy  + self.sedic


class ProtonOZM(object):
    r"""OneZoneModel for pp interaction gamma-ray emission

    Compute gamma-ray spectrum arising from the interaction of a relativistic
    proton distribution with stationary target protons.

    The particle distribution function has the form:

    .. math::

        \frac{dN(E_p)}{dE_p}=A\left(\frac{E_p}{E_0}\right)^{-\Gamma}
        \exp{(E_p/E_\mathrm{cutoff})^\beta},

    where :math:`A` is the normalization at :math:`E_0` and is embedded in the
    parameter `norm`, :math:`E_0` is the normalization energy (`norm_energy`),
    :math:`\Gamma` is the power-law index (`index`), :math:`E_\mathrm{cutoff}`
    is the cutoff energy (`cutoff`), and :math:`\beta` is the exponent of the
    exponential cutoff (`beta`).


    Parameters
    ----------
    Eph : array
        Array of desired output photon energies [eV].

    norm : float
        Normalization of emitted spectrum [1/cm2]. Defined as

        .. math::
            \mathcal{N}=\frac{A_p n_H V}{4 \pi d^2}

        where :math:`A_p` is the normalization of the non-thermal particle
        distribution [1/TeV] at enery `norm_energy`, :math:`n_H` is the
        number density of target protons, :math:`V` is the emitting volume, and
        :math:`d` is the distance to the source.

    norm_energy : float (optional)
        Electron energy [eV] for which normalization parameter :math:`A`
        applies. Should correspond to the decorrelation energy of the observed
        spectrum for the emission process in consideration.

    index : float (optional)
        Power-law index of the particle distribution function.

    cutoff : float (optional)
        Cut-off energy [eV]. Default: None

    beta : float (optional)
        Exponent of exponential energy cutoff argument. Default: 2

    Attributes
    ----------
    specpp : array [1/s/eV]
        Differential gamma-ray spectrum at energies given by `Eph`.

    specpptev : array [1/s/TeV]
        Differential gamma-ray spectrum at energies given by `Eph` in units
        typically used by IACT community.

    sedpp : array [erg/s]
        Spectral energy distribution at energies given by `Eph`.

    Wp : float [erg*[1/cm5]]
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
            norm_energy = 1e12, # eV
            index       = 2.0,
            cutoff      = None, # eV
            beta        = 1.0,
            nolog=False, debug=False, **kwargs):

        if nolog:
            self.logger=_BogusLogger()
        else:
            self.logger=logging.getLogger('ProtonOZM')
            if debug:
                self.logger.setLevel(logging.DEBUG)

        self.__dict__.update(**locals())
        self.__dict__.update(**kwargs)

    def Jp(self,Ep):
        """
        Particle distribution function [1/TeV]
        """
        norm_energy=self.norm_energy/1e12
        if self.cutoff==None:
            Jp = self.norm*(Ep/norm_energy)**-self.index
        else:
            cutoff=self.cutoff/1e12
            Jp = self.norm*((Ep/norm_energy)**-self.index*
                    np.exp(-(Ep/cutoff)**self.beta))
        return Jp

    def Fgamma(self,x,Ep):
        """
        KAB06 Eq.58

        Parameters
        ----------
        x : Egamma/Eprot
        Ep : Eprot [TeV]
        """
        L    = np.log(Ep)
        B    = 1.30+0.14*L+0.011*L**2 # Eq59
        beta = (1.79+0.11*L+0.008*L**2)**-1 # Eq60
        k    = (0.801+0.049*L+0.014*L**2)**-1 # Eq61
        xb   = x**beta

        F1 = B*(np.log(x)/x)*((1-xb)/(1+k*xb*(1-xb)))**4
        F2 = 1./np.log(x)-(4*beta*xb)/(1-xb)-(4*k*beta*xb*(1-2*xb))/(1+k*xb*(1-xb))

        return F1*F2

    def sigma_inel(self,Ep):
        """
        Inelastic cross-section for p-p interaction. KAB06 Eq. 73, 79
        """
        L = np.log(Ep)
        sigma = 34.3 + 1.88*L + 0.25*L**2
        if Ep<=0.1:
            Eth = 1.22e-3
            sigma *= (1-(Eth/Ep)**4)**2 * heaviside(Ep-Eth)
        return sigma*1e-27 # convert from mbarn to cm2

    def _photon_integrand(self,x,Egamma):
        """
        Integrand of Eq. 72
        """
        try:
            return self.sigma_inel(Egamma/x)*self.Jp(Egamma/x) \
                    *self.Fgamma(x,Egamma/x)/x
        except ZeroDivisionError:
            return np.nan

    def _calc_specpp_hiE(self,Egamma):
        """
        Spectrum computed as in Eq. 42 for Egamma >= 0.1 TeV
        """
        from scipy.integrate import fixed_quad,quad
        # Fixed quad with n=40 is about 15 times faster and is always within
        # 0.5% of the result of adaptive quad for Egamma>0.1 
        # WARNING: It also produces artifacts for steep distributions (e.g.
        # Maxwellian) at ~500 GeV. Reverting to adaptative quadrature
        #result=c*fixed_quad(self._photon_integrand,0.,1.,args=[Egamma,],n=40)[0]
        result=c*quad(self._photon_integrand,0.,1.,args=Egamma,epsrel=1e-3,epsabs=0)[0]

        return result

    def _calc_specpp_loE(self,Egamma):
        """
        Delta-functional approximation for low energies Egamma < 0.1 TeV
        """
        from scipy.integrate import fixed_quad,quad
        Kpi=0.17

        m_p=(constants.m_p*constants.c**2).to('TeV').value
        m_pi=1.349766e-4 #TeV/c2

        def delta_integrand(Epi):
            Ep0=m_p+Epi/Kpi
            qpi=c*(self.nhat/Kpi)*self.sigma_inel(Ep0)*self.Jp(Ep0)

            return qpi/np.sqrt(Epi**2+m_pi**2)

        Epimin=Egamma+m_pi**2/(4*Egamma)

        return 2*quad(delta_integrand,Epimin,np.inf,epsrel=1e-3,epsabs=0)[0]

    def _calc_photon_spectrum(self):
        """
        Compute photon spectrum from pp interactions using Eq. 71 and Eq.58 of KAB06.
        """
        from scipy.integrate import fixed_quad,quad
        # convert outspecene to TeV
        outspecene=self.outspecene*u.eV.to('TeV')

        # Before starting, show total proton energy above threshold
        Eth = 1.22e-3
        self.Wp = quad(lambda x: x*self.Jp(x),Eth,np.Inf)[0]*u.TeV.to('erg')
        self.logger.info('W_p(E>1.22 GeV)*[nH/4πd²] = {0:.2e} erg*[1/cm5]'.format(self.Wp))

        if not hasattr(self,'Etrans'):
            self.Etrans=0.1 # Energy at which we change from delta functional to accurate calculation

        self.nhat=1. # initial value, works for index~2.1
        if np.any(outspecene<self.Etrans) and np.any(outspecene>=self.Etrans):
            # compute value of nhat so that delta functional matches accurate calculation at 0.1TeV
            full=self._calc_specpp_hiE(self.Etrans)
            delta=self._calc_specpp_loE(self.Etrans)
            self.nhat*=full/delta

        self.specpp=np.zeros_like(outspecene)

        for i,Egamma in enumerate(outspecene):
            if Egamma>=self.Etrans:
                self.specpp[i]=self._calc_specpp_hiE(Egamma)
            else:
                self.specpp[i]=self._calc_specpp_loE(Egamma)

        self.sedpp=self.specpp*outspecene**2*u.TeV.to('erg') # erg/s
        self.specpptev=self.specpp.copy()
        self.specpp/=u.TeV.to('eV')

        totpplum=np.trapz(self.specpptev*outspecene,outspecene*u.TeV.to('erg'))
        self.logger.info('L_pp*nH/4πd²  = {0:.2e} erg/s'.format(totpplum))

    def calc_outspec(self):
        """
        Compute photon spectrum from pp interactions using Eq. 71 and Eq.58 of KAB06.
        """
        self._calc_photon_spectrum()
