#!/usr/bin/env python
import numpy as np
import gammafit
import astropy.units as u
from astropy.constants import m_e,c
from astropy.io import ascii

## Read data

data=ascii.read('CrabNebula_HESS_2006.dat')

## Set initial parameters

p0=np.array((2.5e-6,3.3,np.log10(48.0),))
labels=['norm','index','log10(cutoff)']

## Model definition

def ElectronIC(pars,data):

    norm   = pars[0]
    index  = pars[1]
    cutoff = (10**pars[2])*u.TeV

    outspecene = data['ene']

    ozm=gammafit.ElectronOZM(
            outspecene, norm,
            index=index,
            cutoff=cutoff,
            seedspec=['CMB',],
            norm_energy=10.*u.TeV,
            nolog=True,
            evolve_nelec=False,
            )

    ozm.calc_nelec()
    ozm.calc_ic()

    # convert to same units as observed differential spectrum
    model=ozm.specic.to('1/(s TeV)')/u.cm**2

    mec2=m_e*c**2

    nelec=ozm.nelec[:-1]*mec2.cgs.value*ozm.gam[:-1]*np.diff(ozm.gam)*u.Unit('erg')
    elec_energy=ozm.gam[:-1]*mec2.to('TeV')

    del ozm

    return model, model, (elec_energy,nelec)

## Prior definition

def lnprior(pars):
	"""
	Return probability of parameter values according to prior knowledge.
	Parameter limits should be done here through uniform prior ditributions
	"""

	logprob = gammafit.uniform_prior(pars[0],0.,np.inf) \
            + gammafit.uniform_prior(pars[1],-1,5)
            #+ gammafit.uniform_prior(pars[2],0.,np.inf) \
            #+ gammafit.uniform_prior(pars[3],0.5,1.5)

	return logprob

if __name__=='__main__':

## Run sampler

    sampler,pos = gammafit.run_sampler(data_table=data, p0=p0, labels=labels, model=ElectronIC,
            prior=lnprior, nwalkers=50, nburn=50, nrun=10, threads=4)

## Save sampler
    from astropy.extern import six
    from six.moves import cPickle
    sampler.pool=None
    cPickle.dump(sampler,open('CrabNebula_electron_sampler.pickle','wb'))

## Diagnostic plots

    gammafit.generate_diagnostic_plots('CrabNebula_electron',sampler,sed=True)


