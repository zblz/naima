#!/usr/bin/env python
# vim: set fileencoding=utf8 :
from __future__ import division
import numpy as np
np.seterr(invalid= 'ignore')
from scipy.integrate import trapz,simps,quad,quadrature,romberg
from scipy.special import cbrt

import logging
logging.basicConfig(level=logging.INFO)

from astropy import constants
from astropy import units as u

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
mec2TeV = mec2*TeV   # m_e*c**2 in eV
mec2hz  = mec2*hz    # m_e*c**2 in Hz

erad    = e**2./mec2
sigt    = (8*np.pi/3)*erad**2

yr = u.yr.to('s')

heaviside=lambda x: (np.sign(x)+1)/2.

class OneZoneModel:

    def __init__(self,nolog=0,debug=0,**kwargs):
        # redirigir el logger a les funcions bogus de sota si ens crida multiprocessing
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
        Sets the attribute to its default value given by DEF_attr
        """
        if type(attrlist)=='str':
            attrlist=[attrlist,]
        for attr in attrlist:
            if not hasattr(self,attr):
                setattr(self,attr,eval('self.DEF_'+attr))

    #### Default parameter values #####
    # Accelerator and emitter physical properties
    DEF_remit = 5e12
    DEF_eta   = 20
    DEF_tad   = 1e30
    DEF_B     = 1  #G
    # Seed spectrum properties
    DEF_nbb      = 10
    DEF_bb       = False
    DEF_seedspec = 'star'
    DEF_ssc      = False
    # electron spectrum matrix (lorentz factors)
    DEF_gmin  = 1e0
    DEF_gmax  = 3e9
    DEF_ngamd = 400 # electron spectrum points per decade
    # Injection spectrum
    DEF_Linj      = 1e36 #erg/s
    DEF_gammainj  = 2
    DEF_glocut    = 1e7/mec2eV # minimum gamma*mec2 = 10MeV
    DEF_cutoffidx = 1.
    # fixed ghicut
    DEF_fixed_ghicut = False
    DEF_ghicut       = 1e8
    # fixed injection normalisation (useful for fitting)
    DEF_inj_norm     = None
    DEF_inj_norm_gam = 2e5 # ~100 GeV
    # evolve spectrum to steady-state?
    DEF_evolve_nelec=True
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

    def _get_star_props(self):
        """
        placeholder for binary system data fetching
        """
        Tstar=40000
        ustar=1
        return Tstar,ustar

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
        clum=trapz(Bnu,fbb)
        self.logger.debug("Lum seed BB = {0} erg/cm3".format(clum))
        Bnu*=useed/clum
        phn=Bnu/hz2erg # 1/cm3
        clum=trapz(phn,photE)
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

        Tstar,ustar = self._get_star_props()

        seeddata={
                'CMB'  : [Tcmb,ucmb],
                'NIR'  : [self.Tnir,self.unir],
                'FIR'  : [self.Tfir,self.ufir],
                'star' : [Tstar,ustar],
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

    def calc_ghicut(self):
        """
        Compute electron spectrum high energy cutoff from analytical formulae for
        IC-KN and syn losses
        """
        # only apply to star seedspecs, as others are not in KN
        try:
            idx=self.seedspec.index('star')
        except ValueError:
            self.ghicut=self.gmax
            return None
        gmaxsyn = 2.*self.B**(-1./2.)*(self.eta/1.e3)**(-1./2.)/mec2TeV # emc2
        if self.bb:
            pheint=trapz(self.phn[idx],self.photE[idx])
        else:
            pheint=self.phe[idx]
        gmaxic  = 9.e5*(self.B/pheint)**3.3*(self.eta/1.e3)**(-3.3)/mec2TeV # emc2
        self.ghicut=float(np.minimum(gmaxic,gmaxsyn))

    def calc_qinj(self,norm10G=False):
        """
        Compute injection spectrum, return as array.
        """
        self._set_default(['gammainj','glocut','cutoffidx','Linj','inj_norm','inj_norm_gam','remit','fixed_ghicut'])

        if self.fixed_ghicut:
            self._set_default(['ghicut',])
        else:
            try:
                ghicut_arg=int(np.where(np.diff(np.sign(self.ttot-self.tacc)))[0])
            except TypeError:
                ghicut_arg=len(self.gam)

            if ghicut_arg<len(self.gam)-1:
                self.ghicut=self.gam[ghicut_arg]
                self.logger.debug('Cutoff obtained from loss timescale comparison: gamma = {0:.2e}; {1:.2e} eV'.format(self.ghicut,self.ghicut*mec2eV))
            else:
                self.calc_ghicut()
                self.logger.debug('Cutoff obtained from analytical formulae: gamma = {0:.2e}; {1:.2e} eV'.format(self.ghicut,self.ghicut*mec2eV))
    
        if self.inj_norm!=None:
            qinj=self.inj_norm*(self.gam/self.inj_norm_gam)**-self.gammainj*np.exp(-(self.gam/self.ghicut)**self.cutoffidx)
            qinj[np.where(self.gam<self.glocut)]=0.
        else:
            qinj=(self.gam)**-self.gammainj*np.exp(-(self.gam/self.ghicut)**self.cutoffidx)
            qinj[np.where(self.gam<self.glocut)]=0.
            qinj*=self.Linj/trapz(qinj*self.gam*mec2,self.gam)

        self.logger.debug('calc_qinj:Injected power: %g erg/s'%(trapz(qinj*self.gam*mec2,self.gam)))
#convertir-ho a densitat
        qinj/=self.vemit()
        return qinj

    def equipartition_B(self):
        """
        Compute equipartition B from U_B=U_rel
        """
        Urel=np.trapz(self.nelec*self.gam*mec2,self.gam)
        return np.sqrt(Urel*8.*np.pi)

#    @profile
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
        self._set_default(['B','tad','eta'])

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
                gdot = trapz((-1.*dsigma*self.phn[idx]/self.photE[idx]),self.photE[idx]) # shape of gam
            else:
                gdot = (-1*dsigma*self.phn[idx]).flatten()
            setattr(self,'tic_'+seedspec,self.gam/np.abs(gdot))
            gdotic+=gdot
            #gdotic/=self.photE

        self.gdot = np.abs(gdotsy+gdotic+gdotad)

        self.tacc = self.eta*self.gam*mec2/(e*self.B*c)
        self.tsy  = self.gam/np.abs(gdotsy)
        self.tic  = self.gam/np.abs(gdotic)

        self.ttot=self.gam/np.abs(self.gdot)


#       qint(Ee)=\int_Ee^\infty Q_inj(E') dE'
    def evolve_e_spectrum(self):
        self._set_default(['evolve_nelec',])
        self.calc_gdot()
        qinj = self.calc_qinj()
        if self.evolve_nelec:
#calculem els trapz terms i despres nomes els sumem
            dgam=np.diff(self.gam)
            tt=dgam*(qinj[1:]+qinj[:-1])/2.
            qint = np.array([np.sum(tt[i:]) for i in range(len(qinj))])

            self.nelec = qint/self.gdot
        else:
            self.nelec = qinj

        self.logger.debug('Total energy in electrons: {0:.2e} erg'.format(
            self.vemit()*trapz(self.nelec*self.gam*mec2,self.gam)))

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

        freqcrit=CS1*(self.gam*mec2)**2.*self.B
        xx=np.vstack(enehz)/freqcrit # shape of (outspecene,gam)

        Fx=1.85*cbrt(xx)*np.exp(-xx)*heaviside(10-xx) # 13.9 ms per loop
        #Fx=np.where(xx<10.,1.85*cbrt(xx)*np.exp(-xx),0) # 16.1 ms per loop

        Psy=CS3*self.B*Fx #Pot sync radiada per un elec, shape of (outspecene,gam)
        J=4.*np.pi*trapz(Psy*self.nelec,self.gam) # int over gam, shape of outspecene

        # Absorption
        CA1=(c**2./enehz**2.)*CS3*self.B # shape of outspecene
        CA2=self.nelec/(self.gam*mec2) # shape of gam
        CA3=Fx*(1./3.+2.*xx) # shape of (outspecene,gam)
        sabs=np.vstack(CA1)*CA2*CA3 # shape of (outspecene,gam)
        K=trapz(sabs,self.gam) #int over gamma, shape of outspecene
#        print self.remit*K
        tau=self.remit*K

        I=np.zeros_like(J)
        nz=np.where((J>0)*(K>0))
        I[nz]=(J[nz]/K[nz])*(-1.*np.expm1(-tau[nz])) # erg/(s*cm2*Hz)

        self.specsy=4.*np.pi*self.remit**2.*I/h # 1/s
        self.sedsy=self.specsy*self.outspecerg # erg/s

        totsylum=trapz(self.specsy,self.outspecerg)
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

        #self.logger.debug('{0} {1}'.format(photE.shape,photE0.shape))
        b=4*photE0*self.gam
# Iterem sobre Eph per estalviar memoria
# TODO: ho hauriem de fer sobre nbb per minimitzar el numero de loops (o fer-ho piece-wise, però és un merder)
        iclum=np.zeros_like(self.outspecene)
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
            lum=trapz(gamint.squeeze()*self.nelec/self.gam**2,self.gam)
            # integral over photE
            if phn.size>1:
                lum=trapz(lum*phn/photE**2,photE) # 1/s
            else:
                lum*=phn/photE

            #self.logger.debug('{0} {1}'.format(lum.shape,Eph.shape))
            iclum[i]=(3./4.)*sigt*c*(Eph*mec2)*lum

        #iclum[np.isnan(iclum)]=0.
        return iclum*self.vemit()

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
        toticlum=trapz(self.specic,self.outspecerg)
        tev=np.where(self.outspecene>1e11)
        tottevlum=trapz(self.specic[tev],self.outspecerg[tev])
        self.logger.info('IC emitted power: %e erg/s'%toticlum)
        self.logger.info('TeV (>100 GeV) luminosity: %e erg/s'%tottevlum)

    def fitPL(self,ener,spec,ran):
        from scipy.optimize import curve_fit
        def linpl(x,N,alpha):
            return np.log10(N)-alpha*x
        pars,covar=curve_fit(linpl,np.log10(ener[ran]),np.log10(spec[ran]),
                p0=[1e33,2.5])
        return pars[0],pars[1]

    def calc_lums(self):
        if not hasattr(self,'specsy'):
            self.logger.warn('No specsy, calling calc_sy()...')
            self.calc_sy()
        if not hasattr(self,'specic'):
            self.logger.warn('No specic, calling calc_ic()...')
            self.calc_ic()

        if hasattr(self,'specicabs'):
            ppabs=True
        else:
            if hasattr(self,'calc_ppabs'):
                self.logger.warn('No specicabs, calling calc_ppabs()...')
                self.calc_ppabs()
                ppabs=True
            else:
                ppabs=False

        self.toticlum=trapz(self.specic,self.outspecene/eV)
        if ppabs:
            self.toticabslum=trapz(self.specicabs,self.outspecene/eV)
        self.totsylum=trapz(self.specsy,self.outspecene/eV)
        xray=np.where((self.outspecene>300)&(self.outspecene<10000))
        gev=np.where((self.outspecene>0.1e9)&(self.outspecene<10e9))
        fermi=np.where(self.outspecene>0.1e9)
        tev=np.where(self.outspecene>0.1e12)
        hess=np.where((self.outspecene>0.2e12)&(self.outspecene<5e12))
        totspec=(self.specic+self.specsy)
        if np.sum(totspec[xray]>0.):
            self.xlum=trapz(totspec[xray],self.outspecerg[xray]) # erg/s
            self.xnorm,self.xidx=self.fitPL(self.outspecene/1e3,totspec,xray)
        else:
            self.xlum=0.
        if np.sum(totspec[gev]>0.):
            self.gevlum=trapz(totspec[gev],self.outspecerg[gev]) # erg/s for 100MeV<E<10GeV
        else:
            self.gevlum=0.
        if np.sum(totspec[fermi]>0.):
            self.fermilum=trapz(totspec[fermi]/self.outspecerg[fermi],self.outspecerg[fermi]) # ph/s for E>100MeV
        else:
            self.fermilum=0.
        if np.sum(totspec[tev]>0.):
            self.tevlum=trapz(totspec[tev],self.outspecerg[tev]) # erg/s
            if ppabs:
                self.tevabslum=trapz(self.specicabs[tev],self.outspecerg[tev]) # erg/s
            tevfit=np.where((self.outspecene>0.1e12)&(self.outspecene<4e12)&(totspec>0))
        else:
            self.tevlum=0.
            if ppabs: 
                self.tevabslum=0.
        if np.sum(totspec[hess]>0.):
            self.hessphlum=trapz(totspec[hess]/self.outspecerg[hess],self.outspecerg[hess]) # ph/s for E>1TeV
            self.hesslum=trapz(totspec[hess],self.outspecerg[hess]) # erg/s for E>1TeV
        else:
            self.hessphlum=0.
            self.hesslum=0.

        self.logger.info('*** Sync ***')
        self.logger.info('Total synchrotron luminosity:  {0:.2e} erg/s'.format(self.totsylum))
        self.logger.info('X-ray (0.3-10 keV) luminosity: {0:.2e} erg/s'.format(self.xlum))
        self.logger.info('*** IC ***')
        self.logger.info('Total IC luminosity:           {0:.2e} erg/s'.format(self.toticlum))
        self.logger.info('VHE (>100 GeV) luminosity:     {0:.2e} erg/s'.format(self.tevlum))
        if ppabs:
            self.logger.info('VHE (>100 GeV) abs luminosity: {0:.2e} erg/s'.format(self.tevabslum))
        self.logger.info(' HE (>0.1 GeV) luminosity:     {0:.2e} erg/s'.format(self.gevlum))

    def calc_fluxes(self):
        if not hasattr(self,'tevlum'):
            self.logger.warn('No tevlum, calling calc_lums()...')
            self.calc_lums()

        if hasattr(self,'tevabslum'):
            ppabs=True
        else:
            ppabs=False

        if not hasattr(self,'dist'):
            self.logger.warn('System has no distance, not computing fluxes')
            return

        dfac=4.*np.pi*self.dist**2.
        self.xflux=self.xlum/dfac
        self.gevflux=self.gevlum/dfac
        self.fermiflux=self.fermilum/dfac
        self.tevflux=self.tevlum/dfac
        if ppabs:
            self.tevabsflux=self.tevabslum/dfac
        self.hessphflux=self.hessphlum/dfac
        self.hessflux=self.hesslum/dfac

        self.logger.info('*** Fluxes ***')
        self.logger.info('X-ray (0.3-10 keV) flux:       {0:.2e} erg/s/cm2'.format(self.xflux))
        self.logger.info('VHE (>100 GeV) flux:           {0:.2e} erg/s/cm2'.format(self.tevflux))
        if ppabs:
            self.logger.info('VHE (>100 GeV) abs flux:       {0:.2e} erg/s/cm2'.format(self.tevabsflux))
        self.logger.info(' HE (>0.1 GeV) flux:           {0:.2e} erg/s/cm2'.format(self.gevflux))

    equipartition=0

    def calc_nelec(self):
        if not hasattr(self,'phe'):
            self.calc_seedspec()
        self.generate_gam()
        if self.equipartition:
            self.calc_qinj()
            self.evolve_e_spectrum()
            relchange=1.
            while relchange>1e-3:
#                self.B=self.equipartition_B()
                self.evolve_e_spectrum()
                Bequi=self.equipartition_B()
                relchange=abs(Bequi-self.B)/self.B
                self.logger.debug('Bold: {0:.2f}, Beq: {1:.2f}, Change: {2:.5f}'.format(self.B,Bequi,relchange*100))
                self.B=(0.25*self.B+0.75*Bequi)
#                self.B=Bequi
            self.logger.info('Equipartition magnetic field: {0:.3f} G'.format(self.B))
        else:
            self.evolve_e_spectrum()

    def calc_outspec(self):
        self.generate_outspecene()
        self.calc_sy()
        self.calc_ic()
        self.calc_lums()

