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

erad    = e**2./mec2
sigt    = (8*np.pi/3)*erad**2

heaviside = lambda x: (np.sign(x)+1)/2.

class ElectronOZM:
    """
    Computation of electron spectrum evolution and synchrotron and IC radiation from a homogeneous emitter.

    Parameters
    ----------
    B : float (optional)
        Isotropic magnetic field strength in Gauss. Default: equipartition with
        CMB (3.4e-6 G)

    remit : float (optional)
        Radius of the emitter in cm. Only used for synchrotron self-absorption
        calculation. Default: 1e14 cm

    tad : float or array_like (ngam,) (optional)
        Adiabatic loss timescale in seconds. Can be either a single value or an
        array of length ``ngam``, to indicate adiabatic loss timescales
        independent or dependent on electron energy, respectively. Default: 1e10 s

    seedspec: iterable (optional)
        A list of seed spectra to use for IC calculation. Currently supported:
        CMB, NIR, FIR. Default: ['CMB',]

    bb : bool (optional)
        Should IC seed spectra be computed as a blackbody? If false,
        monochromatic seed spectra are used. Default: False

    nbb : int (optional)
        Number of spectral points to be computed for seed spectra blackbody.
        Default: 10

    ssc : bool (optional)
        Whether to compute synchrotron self compton (IC on synchrotron generated
        photons). Default: False

    gmin : float (optional)
        Minimum electron energy in units of mc2. Default: 1.0

    gmax : float (optional)
        Maximum electron energy in units of mc2. Default: 3e9

    ngamd : int (optional)
        Number of electron sprectrum points per energy decade. Default: 400

    inj_norm : float (optional)
        Normalization of injection spectrum at electron energy ``inj_norm_gam``.

    inj_norm_gam : float (optional)
        Electron energy for injection spectrum normalization. Default: 2e5 (~100GeV)

    gammainj : float (optional)
        Spectral index of injection spectrum. Default: 2.0

    glocut : float (optional)
        Low energy cutoff of injection spectrum in units of mec2. Electron can
        evolve down to ``gmin``, but will not be injected below ``glocut``.
        Default: 20 (1e7 eV)

    ghicut : float (optional)
        Energy of high-energy exponential cutoff in units of mec2. Default: 4e7
        (2e13 eV)

    cutoffidx : float (optional)
        Exponent of exponential cutoff argument. An infinite ``cutoffidx`` can
        be used to indicate a sharp cutoff. Default: 1.0

    evolve_nelec : bool (optional)
        Whether to evolve electron spectrum until steady state. Se
        Zabalza et al (2011), A&A 527, 9, and Khangulyan et al (2007) MNRAS 380,
        320, for a detailed explanation of steady-state electron spectrum
        computation.

    nened : int (optional)
        Number of emitted spectrum points to be computed per decade of photon
        energy used by ``self.generate_outspecene()``. The output spectrum
        energies can alternatively be defined by modifying the class property
        outspecene with an array of photon energies in eV. Default: 10

    emin : float (optional)
        Minimum photon energy of emitted spectrum in eV used by
        ``self.generate_outspecene()``. Default: 1.0 eV

    emax : float (optional)
        Maximum photon energy of emitted spectrum in eV used by
        ``self.generate_outspecene()``. Default: 1e13 eV

    """

    def __init__(self,nolog=0,debug=0,**kwargs):
        if nolog:
            self.logger=self
        else:
            self.logger=logging.getLogger('OZM')
            if debug:
                self.logger.setLevel(logging.DEBUG)
        self.__dict__.update(**kwargs)

# logger functions used for nolog
    def debug(self,s):
        pass
    def info(self,s):
        pass
    def warn(self,s):
        print 'WARN:OneZoneModel: %s'%s
        pass

    def _set_default(self,attrlist):
        """
        Sets the attributes in attrlist to their default value given by DEF_attr
        if they has not already been set as self.attr
        """
        if type(attrlist)=='str':
            attrlist=[attrlist,]
        for attr in attrlist:
            if not hasattr(self,attr):
                setattr(self,attr,eval('self.DEF_'+attr))

    #### Default parameter values #####
    # emitter physical properties
    DEF_B     = np.sqrt(8*np.pi*4.1817e-13) #equipartition with CMB energy density (G)
    DEF_remit = 1e14
    DEF_tad   = 1e30
    # Seed spectrum properties
    DEF_seedspec = ['CMB',]
    DEF_bb       = False
    DEF_nbb      = 10
    DEF_ssc      = False
    # electron spectrum matrix (lorentz factors)
    DEF_gmin  = 1e0
    DEF_gmax  = 3e9
    DEF_ngamd = 400 # electron spectrum points per decade
    # Injection spectrum
    DEF_inj_norm     = 1e35
    DEF_inj_norm_gam = 2e5 # ~100 GeV
    DEF_gammainj     = 2
    DEF_glocut       = 1e7/mec2eV  # sharp low energy cutoff at gamma*mec2 = 10MeV
    DEF_ghicut       = 2e13/mec2eV # exponential high energy cutoff at gamma*mec2 = 20TeV
    DEF_cutoffidx    = 1.          # Exponent of exponential cutoff
    # evolve spectrum to steady-state?
    DEF_evolve_nelec = True
    # Emitted spectrum matrix (eV)
    DEF_nened = 10 # emitted spectrum points per decade
    DEF_emin  = 1.
    DEF_emax  = 1.e14
    #########

    def vemit(self):
        """
        Emitter radius from emitter volume
        """
        self._set_default(['remit'])
        return (4./3.)*np.pi*self.remit**3.

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
        self._set_default(['bb','nbb','seedspec'])

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

        self._set_default(['gmin','gmax','ngamd'])
        ngam=int(np.log10(self.gmax/self.gmin))*self.ngamd
        self.gam=np.logspace(np.log10(self.gmin),np.log10(self.gmax),ngam)

    def generate_outspecene(self):
        """
        Generate photon energies for which to compute output radiation
        """
        self._set_default(['emin','emax','nened'])

        self.outspecene=np.logspace(np.log10(self.emin),np.log10(self.emax),
               np.log10(self.emax/self.emin)*self.nened)
        self.outspecerg=self.outspecene/eV

    def _calc_qinj(self):
        """
        Compute injection spectrum, return as array.
        """
        self._set_default(['gammainj','glocut','ghicut','cutoffidx','inj_norm','inj_norm_gam',])

        qinj=self.inj_norm*(self.gam/self.inj_norm_gam)**-self.gammainj*np.exp(-(self.gam/self.ghicut)**self.cutoffidx)
        qinj[np.where(self.gam<self.glocut)]=0.

        self.logger.debug('calc_qinj:Injected power: %g erg/s'%(np.trapz(qinj*self.gam*mec2,self.gam)))
        return qinj

    def calc_gdot(self):
# Calculem Qinj i generem el self.gam correcte!
        if not hasattr(self,'gam'):
            self.logger.warn('Generating gam...')
            self.generate_gam()
        if not hasattr(self,'phn'):
            self.logger.warn('Seed photon spectrum not found, calling calc_seedspec...')
            self.calc_seedspec()
        elif type(self.phn)!=list:
            self.logger.warn('Seed photon spectrum not found, calling calc_seedspec...')
            self.calc_seedspec()
        self._set_default(['B','tad',])

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

        self.ttot=self.gam/np.abs(self.gdot)


#       qint(Ee)=\int_Ee^\infty Q_inj(E') dE'
    def evolve_e_spectrum(self):
        self._set_default(['evolve_nelec',])
        self.calc_gdot()
        qinj = self._calc_qinj()
        if self.evolve_nelec:
            # calculem els trapz terms i despres nomes els sumem
            dgam=np.diff(self.gam)
            tt=dgam*(qinj[1:]+qinj[:-1])/2.
            qint = np.array([np.sum(tt[i:]) for i in range(len(qinj))])

            self.nelec = qint/self.gdot
        else:
            self.nelec = qinj

        self.logger.debug('Total energy in electrons: {0:.2e} erg'.format(
            np.trapz(self.nelec*self.gam*mec2,self.gam)))

    def calc_sy(self):
        self.logger.debug('calc_sy')

        CS1   = 6.2632e18                 # crit freq const in Hz (Pacholczyk)
        CS3   = 1.8652e-23*np.sqrt(2./3.) # Constant pot radiada * factor isotropia

        if not hasattr(self,'outspecene'):
            self.logger.warn('Generating outspecene...')
            self.generate_outspecene()
        elif not hasattr(self,'outspecerg'):
            self.outspecerg=self.outspecene/eV
        if not hasattr(self,'gam'):
            self.logger.warn('Calling calc_nelec to generate gam,nelec')
            self.calc_nelec()

        enehz = self.outspecene/hz2ev
        nelec = self.nelec/self.vemit() # density required for sync calculation

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
#        print self.remit*K
        tau=self.remit*K

        I=np.zeros_like(J)
        nz=np.where((J>0)*(K>0))
        I[nz]=(J[nz]/K[nz])*(-1.*np.expm1(-tau[nz])) # erg/(s*cm2*Hz)

        self.specsy=4.*np.pi*self.remit**2.*I/h # 1/s
        self.sedsy=self.specsy*self.outspecerg # erg/s

        totsylum=np.trapz(self.specsy,self.outspecerg)
        self.logger.info('Synchrotron emitted power: {0:.2e} erg/s'.format(totsylum))

    def _calc_specic(self,phn=None,photE=None,seed=None):
        if phn==None and type(phn)==list:
            phn=self.phn[0]
        if photE==None and type(photE)==list:
            photE=self.photE[0]
        self.logger.debug('Computing IC on {0} seed photons...'.format(seed))
# Per a fer-ho sense iterar sobre Eph descomentar seguent paragraf
        # scattered photons: (N,1,1) np.dstack(foo).T
#        Eph=self.outspecene/mec2eV
#        Eph=Eph[:,np.newaxis,np.newaxis]
        # seed photons: (1,N,1) np.vstack
#        photE0=(self.photE/mec2)[np.newaxis,:,np.newaxis]
#        phn=(self.phe/self.photE)[np.newaxis,:,np.newaxis]
        # gam: (1,1,N)
        if photE.size==1:
            photE=np.array((photE,))[:,np.newaxis]

        photE0=(photE/mec2)[:,np.newaxis]
        #phn=(phn)[:,np.newaxis]

        b=4*photE0*self.gam
        iclum=np.zeros_like(self.outspecene)
# Iterate over Eph per save memory
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

            #self.logger.debug('{0} {1}'.format(lum.shape,Eph.shape))
            iclum[i]=(3./4.)*sigt*c*(Eph*mec2)*lum

        #iclum[np.isnan(iclum)]=0.
        return iclum

    def calc_ic(self):
        self.logger.debug('calc_ic')
        self._set_default(['ssc',])
#        d30=lambda x:np.expand_dims(np.expand_dims(x,1),1)
#        d31=lambda x:np.expand_dims(np.expand_dims(x,1),0)
#        d32=lambda x:np.expand_dims(np.expand_dims(x,0),0)

        if not hasattr(self,'outspecene'):
            self.logger.warn('Generating outspecene...')
            self.generate_outspecene()
        elif not hasattr(self,'outspecerg'):
            self.outspecerg=self.outspecene/eV

        if not hasattr(self,'gam'):
            self.logger.warn('Calling calc_nelec to generate gam,nelec')
            self.calc_nelec()

        self.specic,self.sedic=np.zeros_like(self.outspecene),np.zeros_like(self.outspecene)

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
            sedic=specic*self.outspecerg # erg/s
            setattr(self,'specic_'+seedspec,specic)
            setattr(self,'sedic_'+seedspec,sedic)
            self.specic+=specic
            self.sedic+=sedic

        self.logger.debug('self.specic.shape={0}'.format(self.specic.shape))
        toticlum=np.trapz(self.specic,self.outspecerg)
        self.logger.info('IC emitted power: %e erg/s'%toticlum)
        tev=np.where(self.outspecene>1e11)
        tottevlum=np.trapz(self.specic[tev],self.outspecerg[tev])
        self.logger.info('TeV (>100 GeV) luminosity: %e erg/s'%tottevlum)

    def calc_nelec(self):
        if not hasattr(self,'phe'):
            self.calc_seedspec()
        self.generate_gam()
        self.evolve_e_spectrum()

    def calc_outspec(self):
        self.generate_outspecene()
        self.calc_nelec()
        self.calc_sy()
        self.calc_ic()


class ProtonOZM:
    """
    References
    ----------

    KAB06: Kelner, S.R., Aharonian, F.A., and Bugayov, V.V., 2006 PhysRevD 74, 034018

    """

    def __init__(self,nolog=0,debug=0,**kwargs):
        if nolog:
            self.logger=self
        else:
            self.logger=logging.getLogger('OZM')
            if debug:
                self.logger.setLevel(logging.DEBUG)
        self.__dict__.update(**kwargs)

# logger functions used for nolog
    def debug(self,s):
        pass
    def info(self,s):
        pass
    def warn(self,s):
        print 'WARN:OneZoneModel: %s'%s
        pass

    def _set_default(self,attrlist):
        """
        Sets the attributes in attrlist to their default value given by DEF_attr
        if they has not already been set as self.attr
        """
        if type(attrlist)=='str':
            attrlist=[attrlist,]
        for attr in attrlist:
            if not hasattr(self,attr):
                setattr(self,attr,eval('self.DEF_'+attr))

    # Injection spectrum properties
    DEF_gammainj     = 2.0
    DEF_inj_norm     = 1e35
    DEF_inj_norm_ene = 1.0  # TeV
    DEF_cutoff_ene   = 1e3 # TeV
    DEF_cutoff_beta  = 1.0

    ## Proton energy array properties
    #DEF_eprot_min = 1e9
    #DEF_eprot_max = 1e15
    #DEF_eprotd    = 100

    # Target properties
    DEF_nH = 1.0 # 1/cm3

    # Output spectrum properties
    DEF_emin  = 1e-1
    DEF_emax  = 1e3
    DEF_nened = 10


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

    def generate_outspecene(self):
        """
        Generate photon energies for which to compute output radiation
        """
        self._set_default(['emin','emax','nened'])

        self.outspecene=np.logspace(np.log10(self.emin),np.log10(self.emax),
               np.log10(self.emax/self.emin)*self.nened)
        self.outspecerg=self.outspecene/TeV

    def Jp(self,Ep):
        """
        Following Eq. 74 of KAB06 so we can use the low-energy delta-functional
        approximation.
        """
        self._set_default(['gammainj','inj_norm','inj_norm_ene',
            'cutoff_ene','cutoff_beta'])

        return self.inj_norm*((Ep/self.inj_norm_ene)**self.gammainj*
                np.exp(-(Ep/self.cutoff_ene)**self.cutoff_beta))

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

    def photon_integrand(self,x,Egamma):
        """
        Integrand of Eq. 72
        """
        try:
            return self.sigma_inel(Egamma/x)*self.Jp(Egamma/x) \
                    *self.Fgamma(x,Egamma/x)/x
        except ZeroDivisionError:
            return np.nan

    def calc_photon_spectrum(self):
        """
        Compute photon spectrum from pp interactions using Eq. 71 and Eq.58 of KAB06.
        """
        if not hasattr(self,'outspecene'):
            self.generate_outspecene()

        self._set_default(['nH',])

        # Integrate over x=(0,1) (Eq72)
        I = np.array([ quad(self.photon_integrand,0.,1.,args=Egamma)[0] for Egamma in self.outspecene ])

        self.specpp = c*self.nH*I

