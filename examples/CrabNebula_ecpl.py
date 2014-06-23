#!/usr/bin/env python
import numpy as np
import gammafit
import astropy.units as u
from astropy.io import ascii

## Read data

data=ascii.read('CrabNebula_HESS_2006.dat')

ene = u.Quantity(data['energy'])
ene0 = np.sqrt(ene[0]*ene[-1])

## Set initial parameters

p0=np.array((1.5e-12,2.4,np.log10(15.0),))
labels=['norm','index','log10(cutoff)']

## Model definition

from gammafit.models import ExponentialCutoffPowerLaw

# initialise an instance of ECPL
ECPL = ExponentialCutoffPowerLaw(1, ene0, 2, ene0)

def cutoffexp(pars,data):
    """
    Powerlaw with exponential cutoff

    Parameters:
        - 0: PL normalization
        - 1: PL index
        - 2: log10(cutoff energy)
    """

    ECPL.amplitude = pars[0]
    ECPL.alpha = pars[1]
    ECPL.e_cutoff = (10**pars[2])*u.TeV

    return ECPL(data) * data['flux'].unit

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
            model=cutoffexp, prior=lnprior, nwalkers=512, nburn=50, nrun=10,
            threads=4)

## Save sampler
    from astropy.extern import six
    from six.moves import cPickle
    sampler.pool=None
    cPickle.dump(sampler,open('CrabNebula_ecpl_sampler.pickle','wb'))

## Diagnostic plots
    gammafit.generate_diagnostic_plots('CrabNebula_ecpl',sampler,
            sed=True,last_step=False)


