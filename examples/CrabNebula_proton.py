#!/usr/bin/env python
import numpy as np
import gammafit

from astropy import units as u
from astropy.io import ascii

## Read data

data=ascii.read('CrabNebula_HESS_2006.dat')

## Set initial parameters

p0=np.array((5e-25,2.34,np.log10(80.),))
labels=['norm','index','log10(cutoff)']

## Model definition

def ppgamma(pars,data):

    enemid=np.sqrt(data['ene'][0]*data['ene'][-1])
    # peak gamma energy production is ~0.1*Ep, so enemid corresponds to Ep=10*enemid
    # If a cutoff is present, this should be reduced
    norm_ene = 5.*enemid

    norm   = pars[0]
    index  = pars[1]
    cutoff = (10**pars[2])*u.TeV

    ozm=gammafit.ProtonOZM(
            data['ene'], norm,
            cutoff      = cutoff,
            index       = index,
            norm_energy = norm_ene,
            nolog       = True,
            )

    ozm.calc_outspec()

    model=ozm.specpp.to('1/(s TeV)')/u.cm**2

    # compute proton distribution for blob
    Epmin=data['ene'][0]*1e-2
    Epmax=data['ene'][-1]*1e3

    protonene=np.logspace(np.log10(Epmin.value),np.log10(Epmax.value),50)*data['ene'].unit
    protondist=(ozm.Jp(protonene)*protonene**2).to('erg')

    del(ozm)

    return model, (data['ene'],model), (protonene,protondist)

## Prior definition

def lnprior(pars):
	"""
	Return probability of parameter values according to prior knowledge.
	Parameter limits should be done here through uniform prior ditributions
	"""

	logprob = gammafit.uniform_prior(pars[0],0.,np.inf) \
            + gammafit.uniform_prior(pars[1],-1,5)

	return logprob

if __name__=='__main__':

## Run sampler

    sampler,pos = gammafit.run_sampler(data_table=data, p0=p0, labels=labels,
            model=ppgamma, prior=lnprior, nwalkers=50, nburn=50, nrun=10,
            threads=4)

## Save sampler

    from astropy.extern import six
    from six.moves import cPickle
    sampler.pool=None
    cPickle.dump(sampler,open('CrabNebula_proton_sampler.pickle','wb'))

## Diagnostic plots

    gammafit.generate_diagnostic_plots('CrabNebula_proton',sampler,sed=True)

