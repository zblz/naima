"""
Wrappers around InverseCompton, Synchrotron and PionDecay to be used as sherpa models
"""

import numpy as np
import astropy.units as u

from sherpa.models.parameter import Parameter, tinyval
from sherpa.models.model import ArithmeticModel, modelCacher1d

__all__ = ['InverseCompton', 'Synchrotron', 'PionDecay', ]

from . import models
from .utils import trapz_loglog

def _mergex(xlo,xhi,midpoints=False):
    """
    We are assuming that points are consecutive, so that xlo[n]=xhi[n-1]
    This is usually valid for fits from a single spectrum, but breaks for
    simultaneous multiwavelength fitting
    """
    N=xlo.size
    x=np.zeros(N+1)
    x[:N]=xlo.copy()
    x[-1]=xhi[-1]

    if midpoints:
        mid=(xlo+xhi)/2.
        x=np.concatenate((x,mid))
        x.sort()

    return x

class InverseCompton(ArithmeticModel):
    def __init__(self,name='IC'):
        self.index   = Parameter(name, 'index', 2.0, min=-10, max=10)
        self.ref     = Parameter(name, 'ref', 20, min=0, frozen=True, units='TeV')
        self.ampl    = Parameter(name, 'ampl', 1, min=0, max=1e60, hard_max=1e100, units='1/eV')
        self.cutoff  = Parameter(name, 'cutoff', 0.0, min=0,frozen=True, units='TeV')
        self.beta    = Parameter(name, 'beta', 1, min=0, max=10, frozen=True)
        self.TFIR    = Parameter(name, 'TFIR', 70, min=0, frozen=True, units='K')
        self.uFIR    = Parameter(name, 'uFIR', 0.0, min=0, frozen=True, units='eV/cm3') # 0.2eV/cm3 typical in outer disk
        self.TNIR    = Parameter(name, 'TNIR', 3800, min=0, frozen=True, units='K')
        self.uNIR    = Parameter(name, 'uNIR', 0.0, min=0, frozen=True, units='eV/cm3') # 0.2eV/cm3 typical in outer disk
        self.verbose = Parameter(name, 'verbose', 0, min=0, frozen=True)
        ArithmeticModel.__init__(self,name,(self.index,self.ref,self.ampl,self.cutoff,self.beta,
            self.TFIR, self.uFIR, self.TNIR, self.uNIR, self.verbose))
        self._use_caching = True
        self.cache = 10

    def guess(self,dep,*args,**kwargs):
        # guess normalization from total flux
        xlo,xhi=args
        model=self.calc([p.val for p in self.pars],xlo,xhi)
        modflux=trapz_loglog(model,xlo)
        obsflux=trapz_loglog(dep*(xhi-xlo),xlo)
        self.ampl.set(self.ampl.val*obsflux/modflux)

    @modelCacher1d
    def calc(self,p,x,xhi=None):

        index,ref,ampl,cutoff,beta,TFIR,uFIR,TNIR,uNIR,verbose = p

        # Sherpa provides xlo, xhi in KeV, we merge into a single array if bins required
        if xhi is None:
            outspec = x * u.keV
        else:
            outspec = _mergex(x,xhi) * u.keV

        if cutoff == 0.0:
            pdist = models.PowerLaw(ampl * u.Unit('1/eV'), ref * u.TeV, index)
        else:
            pdist = models.ExponentialCutoffPowerLaw(ampl * u.Unit('1/eV'),
                    ref * u.TeV, index, cutoff * u.TeV, beta=beta)

        # Build seedspec definition
        seedspec=['CMB',]
        if uFIR>0.0:
            seedspec.append(['FIR',TFIR * u.K, uFIR * u.ev/u.cm**3])
        if uNIR>0.0:
            seedspec.append(['NIR',TNIR * u.K, uNIR * u.ev/u.cm**3])

        ic = models.InverseCompton(pdist, seed_photon_fields=seedspec,
                log10gmin=5, log10gmax=10, ngamd=100)

        model = ic.flux(outspec, distance=1*u.kpc).to('1/(s cm2 keV)')

        # Do a trapz integration to obtain the photons per bin
        if xhi is None:
            photons = (model * outspec).to('1/(s cm2)').value
        else:
            photons = trapz_loglog(model,outspec,intervals=True).to('1/(s cm2)').value

        if verbose:
            print self.thawedpars, trapz_loglog(outspec*model,outspec).to('erg/(s cm2)')

        return photons

class Synchrotron(ArithmeticModel):
    def __init__(self,name='IC'):
        self.index   = Parameter(name, 'index', 2.0, min=-10, max=10)
        self.ref     = Parameter(name, 'ref', 20, min=0, frozen=True, units='TeV')
        self.ampl    = Parameter(name, 'ampl', 1, min=0, max=1e60, hard_max=1e100, units='1/eV')
        self.cutoff  = Parameter(name, 'cutoff', 0.0, min=0,frozen=True, units='TeV')
        self.beta    = Parameter(name, 'beta', 1, min=0, max=10, frozen=True)
        self.B       = Parameter(name, 'B', 1, min=0, max=10, frozen=True, units='G')
        self.verbose = Parameter(name, 'verbose', 0, min=0, frozen=True)
        ArithmeticModel.__init__(self,name,(self.index,self.ref,self.ampl,self.cutoff,self.beta,self.B,self.verbose))
        self._use_caching = True
        self.cache = 10

    def guess(self,dep,*args,**kwargs):
        # guess normalization from total flux
        xlo,xhi=args
        model=self.calc([p.val for p in self.pars],xlo,xhi)
        modflux=trapz_loglog(model,xlo)
        obsflux=trapz_loglog(dep*(xhi-xlo),xlo)
        self.ampl.set(self.ampl.val*obsflux/modflux)

    @modelCacher1d
    def calc(self,p,x,xhi=None):

        index,ref,ampl,cutoff,beta,B,verbose = p

        # Sherpa provides xlo, xhi in KeV, we merge into a single array if bins required
        if xhi is None:
            outspec = x * u.keV
        else:
            outspec = _mergex(x,xhi) * u.keV

        if cutoff == 0.0:
            pdist = models.PowerLaw(ampl * u.Unit('1/eV'), ref * u.TeV, index)
        else:
            pdist = models.ExponentialCutoffPowerLaw(ampl * u.Unit('1/eV'),
                    ref * u.TeV, index, cutoff * u.TeV, beta=beta)

        sy = models.Synchrotron(pdist, B=B*u.G,
                log10gmin=5, log10gmax=10, ngamd=50)

        model = sy.flux(outspec, distance=1*u.kpc).to('1/(s cm2 keV)')

        # Do a trapz integration to obtain the photons per bin
        if xhi is None:
            photons = (model * outspec).to('1/(s cm2)').value
        else:
            photons = trapz_loglog(model,outspec,intervals=True).to('1/(s cm2)').value

        if verbose:
            print self.thawedpars, trapz_loglog(outspec*model,outspec).to('erg/(s cm2)')

        return photons

class PionDecay(ArithmeticModel):
    def __init__(self,name='pp'):
        self.index   = Parameter(name , 'index'   , 2.1 , min=-10 , max=10)
        self.ref     = Parameter(name , 'ref'     , 60  , min=0   , frozen=True  , units='TeV')
        self.ampl    = Parameter(name , 'ampl'    , 100 , min=0   , max=1e60     , hard_max=1e100 , units='1/eV')
        self.cutoff  = Parameter(name , 'cutoff'  , 0   , min=0   , frozen=True  , units='TeV')
        self.beta    = Parameter(name , 'beta'    , 1   , min=0   , max=10       , frozen=True)
        self.nh      = Parameter(name , 'nH'      , 1   , min=0   , frozen=True  , units='1/cm3')
        self.verbose = Parameter(name , 'verbose' , 0   , min=0   , frozen=True)
        ArithmeticModel.__init__(self,name,(self.index,self.ref,self.ampl,self.cutoff,self.beta,self.nh,self.verbose))
        self._use_caching = True
        self.cache = 10

    def guess(self,dep,*args,**kwargs):
        # guess normalization from total flux
        xlo,xhi=args
        model=self.calc([p.val for p in self.pars],xlo,xhi)
        modflux=trapz_loglog(model,xlo)
        obsflux=trapz_loglog(dep*(xhi-xlo),xlo)
        self.ampl.set(self.ampl.val*obsflux/modflux)

    @modelCacher1d
    def calc(self,p,x,xhi=None):

        index,ref,ampl,cutoff,beta,nh,verbose = p

        # Sherpa provides xlo, xhi in KeV, we merge into a single array if bins required
        if xhi is None:
            outspec = x * u.keV
        else:
            outspec = _mergex(x,xhi) * u.keV

        if cutoff == 0.0:
            pdist = models.PowerLaw(ampl * u.Unit('1/eV'), ref * u.TeV, index)
        else:
            pdist = models.ExponentialCutoffPowerLaw(ampl * u.Unit('1/eV'),
                    ref * u.TeV, index, cutoff * u.TeV, beta=beta)

        pp = models.PionDecay(pdist, nh=nh*u.Unit('1/cm3'))

        model = pp.flux(outspec, distance=1*u.kpc).to('1/(s cm2 keV)')

        # Do a trapz integration to obtain the photons per bin
        if xhi is None:
            photons = (model * outspec).to('1/(s cm2)').value
        else:
            photons = trapz_loglog(model,outspec,intervals=True).to('1/(s cm2)').value

        if verbose:
            print self.thawedpars, trapz_loglog(outspec*model,outspec).to('erg/(s cm2)')

        return photons
