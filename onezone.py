#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
np.seterr(invalid= 'ignore')
from scipy.special import cbrt
from scipy.integrate import quad

import logging
logging.basicConfig(level=logging.INFO)

## Constants and units
from astropy import constants
from astropy import units as u
# import constant values from astropy.constants
constants_to_import=['c','G','m_e','e','h','k_B','R_sun']
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

erad    = e**2./mec2
sigt    = (8*np.pi/3)*erad**2

heaviside = lambda x: (np.sign(x)+1)/2.

class ElectronOZM(object):
    """
    Computation of electron spectrum evolution and synchrotron and IC radiation from a homogeneous emitter.

    One Zone Model parameters
    -------------------------
    Eph : array
        Array of desired output photon energies [eV].

    norm : float
        Normalization of emitted spectrum [1/cm2]. Defined as

        .. math::
            \mathcal{N}=\frac{A V}{4 \pi d^2}

        where :math:`A` is the normalization of the non-thermal particle
        distribution [1/cm3/GeV] at enery `norm_energy`, :math:`V` is the
        emitting volume, and :math:`d` is the distance to the source.

    norm_energy : float (optional)
        Electron energy [GeV] for which normalization parameter :math:`A`
        applies. Should correspond to the decorrelation energy of the observed
        spectrum for the emission process in consideration.

    index : float (optional)
        Power-law index of the particle distribution function. The distribution
        function has the form:

        .. math::
            \frac{dN(E_e)}{E_e}=A\left(\frac{E_e}{E_0}\right)^{-\Gamma}
            \exp{(p/E_\mathrm{cutoff})^\beta},

        where :math:`A` is the normalization at :math:`E_0` and is embedded in
        the parameter `norm`, :math:`E_0` is the normalization energy
        parameter `norm_energy`, :math:`\Gamma` is the power-law index
        parameter `index`, :math:`E_\mathrm{cutoff}` is the cutoff energy
        parameter `cutoff`, and :math:`\beta` is the exponent of the exponential
        cutoff argument `beta`.

    cutoff : float (optional)
        Cut-off energy [TeV].

    beta : float (optional)
        Exponent of exponential energy cutoff argument.

    B : float (optional)
        Isotropic magnetic field strength in microgauss. Default: equipartition with
        CMB (3.24e-6 G)

    seedspec: string or iterable of strings or iterable of floats (optional)
        A list of seed spectra to use for IC calculation. Is string is one of
        CMB, NIR, FIR, precomputed tables will be used. Any floats indicate the
        temperature for a blackbody photon distribution and full computation
        will be performed. Default: ['CMB',]

    ssc : bool (optional)
        Whether to compute synchrotron self compton (IC on synchrotron generated
        photons). Default: False

    remit : float (optional)
        Emitter radius in cm. Only relevant for synchrotron self-absorption and
        synchrotron self Compton.

    evolve_nelec : bool (optional)
        Whether to evolve electron spectrum until steady state. See
        Zabalza et al (2011), A&A 527, 9, and Khangulyan et al (2007) MNRAS 380,
        320, for a detailed explanation of steady-state electron spectrum
        computation.


    Computation parameters (not present in class signature)
    -------------------------------------------------------

    bb : bool (optional)
        Should IC seed spectra be computed as a blackbody? If false,
        monochromatic seed spectra are used. Default: False

    nbb : int (optional)
        Number of spectral points to be computed for seed spectra blackbody.
        Default: 10

    gmin : float (optional)
        Minimum electron energy in units of mc2. Default: 1.0

    gmax : float (optional)
        Maximum electron energy in units of mc2. Default: 3e9

    ngamd : int (optional)
        Number of electron spectrum points per energy decade. Default: 400

    glocut : float (optional)
        Low energy cutoff of injection spectrum in units of mec2. Electron can
        evolve down to ``gmin``, but will not be injected below ``glocut``.
        Default: 20 (1e7 eV)

    Output (as class attributes)
    ----------------------------

    self.specsy : array [1/s/eV]
        Differential synchrotron spectrum: emitted synchrotron photons per unit
        energy per second at energies given by self.outspecene.

    self.sedsy : array [erg/s]
        Synchrotron SED := self.specsy*self.outspecene*self.outspecerg

    self.specic : array [1/s/eV]
        Differential IC spectrum: emitted IC photons per unit energy per second
        at energies given by self.outspecene.

    self.specictev : array [1/s/TeV]
        Differential IC spectrum in units typically used by IACT community.

    self.sedic : array [erg/s]
        IC SED := self.specic*self.outspecene*self.outspecerg

    """


    def __init__(self,
        outspecene, # emitted photon energy array
        norm,       # normalization
        #### Default parameter values #####
        # injection
        norm_energy   = 1.0,
        index         = 2.0,
        cutoff        = 30,
        beta          = 1.0,
        # emitter physical properties
        B        = np.sqrt(8*np.pi*4.1817e-13), #equipartition with CMB energy density (G)
        seedspec = ['CMB',],
        ssc      = False,
        remit    = 1e15,
        # evolve particle spectrum to steady-state?
        evolve_nelec = False,
        #
        nolog=False,debug=False,**kwargs):

        if nolog:
            self.logger=self
        else:
            self.logger=logging.getLogger('ElectronOZM')
            if debug:
                self.logger.setLevel(logging.DEBUG)

        computation_defaults={
            # Seed spectrum properties
            'bb'     : False,
            'nbb'    : 10,
            'tad'    : 1e30,
            # electron spectrum matrix (lorentz factors)
            'gmin'   : 1e0,
            'gmax'   : 3e9,
            'ngamd'  : 400, # electron spectrum points per decade
            # Injection spectrum
            'glocut' : 1e7/mec2eV,  # sharp low energy cutoff at gamma*mec2 = 10MeV
            }

        self.__dict__.update(**computation_defaults)
        self.__dict__.update(**locals())
        self.__dict__.update(**kwargs)

# logger functions used for nolog
    def debug(self,s):
        pass
    def info(self,s):
        pass
    def warn(self,s):
        print 'WARN:OneZoneModel: %s'%s
        pass

    def _calc_bb(self,Tseed,useed,nbb):
        self.logger.info('Computing blackbody spectrum for T = {0} K, u = {1:.2e} erg/cm3'.format(Tseed,useed))
        bbepeak=3*Tseed*k_B # in erg
        self.logger.debug("E_peakbb = {0:.2e} eV = {1:.2e} erg".format(bbepeak*eV,bbepeak))
        eminbb=bbepeak/np.sqrt(1000.) #erg
        emaxbb=bbepeak*np.sqrt(100.) #erg
        bbepeak*=eV
        photE=np.logspace(np.log10(eminbb),np.log10(emaxbb),self.nbb) # in erg
        fbb=photE/hz2erg # Hz
# Bnu in units of erg/cm3/Hz
        Bnu=(4.*np.pi/c)*(2*h*fbb**3./c**2.)/np.expm1((h*fbb)/(k_B*Tseed))
        clum=np.trapz(Bnu,fbb)
        self.logger.debug("Lum seed BB = {0} erg/cm3".format(clum))
        Bnu*=useed/clum
        phn=Bnu/hz2erg # 1/cm3
        clum=np.trapz(phn,photE)
        self.logger.debug("Lum seed, trapz(phn,photE) = {0:.3e} erg/cm3".format(clum))
        eint=(emaxbb/eminbb)**(1./float(nbb))
        phe=phn*photE*(eint-1)
        self.logger.debug("Lum seed, sum(phe)         = {0:.3e} erg/cm3".format(np.sum(phe)))
        return photE,phe,phn

    def _calc_mono(self,Tseed,useed):
        phe=np.array((useed,)) # erg/cm3
        photE=np.array((3*k_B*Tseed,)) # erg
        phn=phe/photE # 1/cm3
        self.logger.info('Setting monochromatic seed spectrum at E = {0:.2e} eV, u = {1:.2e} erg/cm3'.format(photE[0]*eV, phe[0]))
        return photE,phe,phn

    def calc_seedspec(self):
        """
        Compute seed photon spectrum for IC.
        Outputs (as attributes of class):
            phn: particle photon density
            photE: photon energies
        """

        Tcmb = 2.72548 # 0.00057 K
        ucmb = 0.261*u.eV.to('erg')

        if not hasattr(self,'Tfir'):
            self.Tfir = 70
            self.ufir = 0.5*u.eV.to('erg')

        if not hasattr(self,'Tnir'):
            self.Tnir = 5000
            self.unir = 1.0*u.eV.to('erg')

        seeddata={
                'CMB'  : [Tcmb,ucmb],
                'NIR'  : [self.Tnir,self.unir],
                'FIR'  : [self.Tfir,self.ufir],
                }

        # Allow for seedspec definitions of the type 'CMB-NIR-FIR'
        if type(self.seedspec)!=list:
            self.seedspec=self.seedspec.split('-')

        self.photE,self.phe,self.phn=[],[],[]

        for seed in self.seedspec:
            try:
                Tseed,useed=seeddata[seed]
            except KeyError:
                self.logger.debug('Seedspec {0} not available, removing it.'.format(seed))
                self.seedspec.remove(seed)
                continue
            if  self.bb:
                pe,pu,pn=self._calc_bb(Tseed,useed,self.nbb)
            else:
                pe,pu,pn=self._calc_mono(Tseed,useed)
            self.photE.append(pe)
            self.phe.append(pu)
            self.phn.append(pn)

    def generate_gam(self):
        """
        Generate gamma values
        """

        ngam=int(np.log10(self.gmax/self.gmin))*self.ngamd
        self.gam=np.logspace(np.log10(self.gmin),np.log10(self.gmax),ngam)

    def calc_gdot(self):
# Calculem Qinj i generem el self.gam correcte!
        if not hasattr(self,'gam'):
            self.logger.warn('Generating gam...')
            self.generate_gam()
        if not hasattr(self,'phn') or type(self.phn)!=list:
            self.logger.warn('Seed photon spectrum not found, calling calc_seedspec...')
            self.calc_seedspec()

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
# If seed photon spectrum is not moonochromatic, we need to iterate over photE0
            photE0=self.photE[idx]/mec2
#        change xx to photE0*np.hstack(gam), int over axis=1
            xx=4.0*photE0*np.vstack(self.gam) # shape of (photE,gam)
            t1=np.log(1.0+0.16*xx)
            t2=xx**2.0
            dsigma=(c*0.1387e-23*t1/(1.0+0.139e+1*xx)
                          *(1.0-0.46e-1*xx/(1+0.49e-1*t2))*np.vstack(self.gam))

            if self.phn[idx].size > 1:
                gdot = np.trapz((-1.*dsigma*self.phn[idx]/self.photE[idx]),self.photE[idx]) # shape of gam
            else:
                gdot = (-1*dsigma*self.phn[idx]).flatten()
            setattr(self,'tic_'+seedspec,self.gam/np.abs(gdot))
            gdotic+=gdot
            #gdotic/=self.photE

        self.gdot = np.abs(gdotsy+gdotic+gdotad)

        self.tsy  = self.gam/np.abs(gdotsy)
        self.tic  = self.gam/np.abs(gdotic)

        self.ttot = self.gam/np.abs(self.gdot)


    def _calc_qinj(self):
        """
        Compute injection spectrum, return as array.
        """

        # convert parameters to gamma
        cutoff_gam=self.cutoff/mec2TeV
        norm_gam=self.norm_energy/mec2GeV

        qinj=self.norm*(self.gam/norm_gam)**-self.index*np.exp(-(self.gam/cutoff_gam)**self.beta)

        qinj[np.where(self.gam<self.glocut)]=0.

        return qinj

#       qint(Ee)=\int_Ee^\infty Q_inj(E') dE'
    def _calc_steady_state_nelec(self,qinj):
        """
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
        if not hasattr(self,'phe'):
            self.calc_seedspec()
        self.generate_gam()
        self.calc_gdot()

        qinj = self._calc_qinj()

        if self.evolve_nelec:
            self.logger.debug('calc_nelec: L_inj*4πd² = {0:.2e} erg/s'.format(
                np.trapz(qinj*self.gam*mec2,self.gam)))
            self.nelec = self._calc_steady_state_nelec(qinj)
        else:
            self.nelec = qinj

        self.logger.debug('calc_nelec: E_e*4πd² = {0:.2e} erg'.format(
            np.trapz(self.nelec*self.gam*mec2,self.gam)))

    def calc_sy(self):
        self.logger.debug('calc_sy: Starting synchrotron computation...')

        CS1   = 6.2632e18                 # crit freq const in Hz (Pacholczyk)
        CS3   = 1.8652e-23*np.sqrt(2./3.) # Constant pot radiada * factor isotropia

        if not hasattr(self,'outspecerg'):
            self.outspecerg=self.outspecene/eV
        if not hasattr(self,'gam'):
            self.logger.warn('Calling calc_nelec to generate gam,nelec')
            self.calc_nelec()

        # Bogus (large) vemit and remit to compute synchrotron
        vemit=(4./3.)*np.pi*self.remit**3

        enehz = self.outspecene/hz2ev
        nelec = self.nelec/vemit # density required for sync calculation

        freqcrit=CS1*(self.gam*mec2)**2.*self.B
        xx=np.vstack(enehz)/freqcrit # shape of (outspecene,gam)

        Fx=1.85*cbrt(xx)*np.exp(-xx)*heaviside(10-xx) # 13.9 ms per loop
        #Fx=np.where(xx<10.,1.85*cbrt(xx)*np.exp(-xx),0) # 16.1 ms per loop

        Psy=CS3*self.B*Fx #Pot sync radiada per un elec, shape of (outspecene,gam)
        J=4.*np.pi*np.trapz(Psy*nelec,self.gam) # int over gam, shape of outspecene

        # Absorption
        CA1=(c**2./enehz**2.)*CS3*self.B # shape of outspecene
        CA2=nelec/(self.gam*mec2) # shape of gam
        CA3=Fx*(1./3.+2.*xx) # shape of (outspecene,gam)
        sabs=np.vstack(CA1)*CA2*CA3 # shape of (outspecene,gam)
        K=np.trapz(sabs,self.gam) #int over gamma, shape of outspecene
        tau=self.remit*K

        I=np.zeros_like(J)
        nz=np.where((J>0)*(K>0))
        I[nz]=(J[nz]/K[nz])*(-1.*np.expm1(-tau[nz])) # erg/(s*cm2*Hz)

        self.specsy=4.*np.pi*self.remit**2.*I/h/self.outspecene # 1/s
        self.sedsy=self.specsy*self.outspecene*self.outspecerg # erg/s

        totsylum=np.trapz(self.specsy*self.outspecene,self.outspecerg)
        self.logger.info('calc_sy: L_sy*4πd² = {0:.2e} erg/s'.format(totsylum))

    def _calc_specic(self,phn=None,photE=None,seed=None):
        if phn==None and type(phn)==list:
            phn=self.phn[0]
        if photE==None and type(photE)==list:
            photE=self.photE[0]
        self.logger.debug('_calc_specic: Computing IC on {0} seed photons...'.format(seed))
        if photE.size==1:
            photE=np.array((photE,))[:,np.newaxis]

        photE0=(photE/mec2)[:,np.newaxis]

        b=4*photE0*self.gam
        iclum=np.zeros_like(self.outspecene)
# Iterate over Eph to save memory
        for i,Eph in enumerate(self.outspecene/mec2eV):
            w=Eph/self.gam
            q=w/(b*(1-w))
            fic=(2*q*np.log(q)
                    +(1+2*q)*(1-q)
                    +(1./2.)*(b*q)**2*(1-q)/(1+b*q)
                    )
            gamint=fic*heaviside(1-q)*heaviside(q-1./(4*self.gam**2))
            gamint[np.isnan(gamint)]=0.

            # integral over gam
            lum=np.trapz(gamint.squeeze()*self.nelec/self.gam**2,self.gam)
            # integral over photE
            if phn.size>1:
                lum=np.trapz(lum*phn/photE**2,photE) # 1/s
            else:
                lum*=phn/photE

            iclum[i]=(3./4.)*sigt*c*(Eph*mec2)*lum

        return iclum/self.outspecene # return differential spectrum in 1/s/eV

    def calc_ic(self):
        self.logger.debug('calc_ic: Starting IC computation...')

        if not hasattr(self,'outspecerg'):
            self.outspecerg=self.outspecene/eV

        if not hasattr(self,'gam'):
            self.logger.warn('Calling calc_nelec to generate gam,nelec')
            self.calc_nelec()

        for spec in ['specic','specictev','sedic']:
            setattr(self,spec,np.zeros_like(self.outspecene))

        if self.ssc:
            self.calc_sy()
# compute density of synchrotron photons in rsync radius
            rsyn=self.remit
            umean=2.24
            syf=umean/(4*np.pi*rsyn**2*c)
            syphn=self.specsy*syf
            syphotE=self.outspecerg.copy()
            if 'SSC' in self.seedspec:
                idx=self.seedspec.index('SSC')
                self.phn[idx]=syphn
                self.photE[idx]=syphotE
            else:
                self.seedspec.append('SSC')
                self.phn.append(syphn)
                self.photE.append(syphotE)
        else:
            # check if there is SSC spectrum in seedspec, remove it
            if 'SSC' in self.seedspec:
                idx=self.seedspec.index('SSC')
                for ll in [self.seedspec,self.phe,self.photE,self.phn]:
                    del ll[idx]

        for idx,seedspec in enumerate(self.seedspec):
            # Call actual computation, detached to allow changes in subclasses
            specic=self._calc_specic(phn=self.phn[idx],photE=self.photE[idx],seed=seedspec)
            specictev=specic/1e12 # 1/s/TeV
            sedic=specic*self.outspecerg*self.outspecene # erg/s
            setattr(self,'specic_'+seedspec,specic)
            setattr(self,'specictev_'+seedspec,specictev)
            setattr(self,'sedic_'+seedspec,sedic)
            self.specic+=specic
            self.specictev+=specictev
            self.sedic+=sedic

        self.logger.debug('self.specic.shape={0}'.format(self.specic.shape))
        toticlum=np.trapz(self.specic,self.outspecerg)
        self.logger.info('calc_ic: L_ic*4πd² = {0:.2e} erg/s'.format(toticlum))
        tev=np.where(self.outspecene>1e11)
        if len(tev[0])>0:
            tottevlum=np.trapz(self.specic[tev],self.outspecerg[tev])
            self.logger.info('calc_ic: L_VHE*4πd² = {0:.2e} erg/s'.format(tottevlum))

    def calc_outspec(self):
        self.calc_nelec()
        self.calc_sy()
        self.calc_ic()
        self.spec = self.specsy + self.specic
        self.sed  = self.sedsy  + self.sedic


class ProtonOZM(object):
    """
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


    One Zone Model parameters
    -------------------------
    Eph : array
        Array of desired output photon energies [eV].

    norm : float
        Normalization of emitted spectrum [1/cm2]. Defined as

        .. math::
            \mathcal{N}=\frac{A_p n_p V}{4 \pi d^2}

        where :math:`A` is the normalization of the non-thermal particle
        distribution [1/cm3/GeV] at enery `norm_energy`, :math:`n_p` is the
        number density of target protons, :math:`V` is the emitting volume, and
        :math:`d` is the distance to the source.

    norm_energy : float (optional)
        Electron energy [GeV] for which normalization parameter :math:`A`
        applies. Should correspond to the decorrelation energy of the observed
        spectrum for the emission process in consideration.

    index : float (optional)
        Power-law index of the particle distribution function.

    cutoff : float (optional)
        Cut-off energy [TeV].

    beta : float (optional)
        Exponent of exponential energy cutoff argument.

    Output
    ------
    self.specpp : array [1/s/eV]
        Differential gamma-ray spectrum at energies given by ``self.outspecene``.

    self.sedpp : array [erg/s]
        Spectral energy distribution at energies given by ``self.outspecene``.

    References
    ----------

    Kelner, S.R., Aharonian, F.A., and Bugayov, V.V., 2006 PhysRevD 74, 034018 [KAB06]

    """

    def __init__(self,
            outspecene,
            norm,
            # Injection spectrum properties
            norm_energy = 1e3, # GeV
            index       = 2.0,
            cutoff      = 1e3, # TeV
            beta        = 1.0,
            nolog=False, debug=False, **kwargs):

        if nolog:
            self.logger=self
        else:
            self.logger=logging.getLogger('ProtonOZM')
            if debug:
                self.logger.setLevel(logging.DEBUG)

        # convert all to TeV
        outspecene*=1e12
        norm_energy/=1e3

        self.__dict__.update(**locals())
        self.__dict__.update(**kwargs)

# bogus logger functions used for nolog
    def debug(self,s):
        pass
    def info(self,s):
        pass
    def warn(self,s):
        print 'WARN:OneZoneModel: %s'%s
        pass

    def Jp(self,Ep):
        """
        Particle distribution function [1/cm3/TeV]
        """
        return self.norm*((Ep/self.norm_energy)**-self.index*
                np.exp(-(Ep/self.cutoff)**self.beta))

    def Fgamma(self,x,Ep):
        """
        KAB06 Eq.58

        Parameters
        ----------
        x : Egamma/Eprot
        Ep : Eprot [TeV]
        """
        L=np.log(Ep)
        B=1.30+0.14*L+0.011*L**2 # Eq59
        beta=(1.79+0.11*L+0.008*L**2)**-1 # Eq60
        k=(0.801+0.049*L+0.014*L**2)**-1 # Eq61
        xb=x**beta

        F1=B*(np.log(x)/x)*((1-xb)/(1+k*xb*(1-xb)))**4
        F2=1./np.log(x)-(4*beta*xb)/(1-xb)-(4*k*beta*xb*(1-2*xb))/(1+k*xb*(1-xb))

        return F1*F2

    def sigma_inel(self,Ep):
        """
        Inelastic cross-section for p-p interaction. KAB06 Eq. 79
        """
        L = np.log(Ep)
        Eth = 1.22e-3
        if Ep<=Eth:
            sigma = 0.0
        else:
            sigma = (34.3 + 1.88*L + 0.25*L**2)*(1-(Eth/Ep)**4)**2
        return sigma

    def photon_integrand(self,x,Egamma):
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
        return c*quad(self.photon_integrand,0.,1.,args=Egamma)[0]

    def _calc_specpp_loE(self,Egamma):
        """
        Delta-functional approximation for low energies Egamma < 0.1 TeV
        """
        Kpi=0.17

        m_p=(constants.m_p*constants.c**2).to('TeV').value
        m_pi=1.349766e-4 #TeV/c2

        def delta_integrand(Epi):
            Ep0=m_p+Epi/Kpi
            qpi=c*(self.nhat/Kpi)*self.sigma_inel(Ep0)*self.Jp(Ep0)

            return qpi/np.sqrt(Epi**2+m_pi**2)

        Epimin=Egamma+m_pi**2/(4*Egamma)

        return 2*quad(delta_integrand,Epimin,np.inf)[0]

    def calc_photon_spectrum(self):
        """
        Compute photon spectrum from pp interactions using Eq. 71 and Eq.58 of KAB06.
        """

        if np.any(self.outspecene<0.1):
            # compute value of nhat so that delta functional matches accurate calculation at 0.1TeV
            self.nhat=1. # initial value, works for index~2.1
            full=self._calc_specpp_hiE(0.1)
            delta=self._calc_specpp_loE(0.1)
            self.nhat*=full/delta

        self.specpp=np.zeros_like(self.outspecene)

        for i,Egamma in enumerate(self.outspecene):
            if Egamma>=0.1:
                self.specpp[i]=self._calc_specpp_hiE(Egamma)
            else:
                self.specpp[i]=self._calc_specpp_loE(Egamma)

        self.sedpp=self.specpp*self.outspecene**2*u.TeV.to('erg') # erg/s

