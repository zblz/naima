#!/usr/bin/env python
import numpy as np
import gammafit
import astropy.units as u
from astropy.io import ascii

## Read data

data=ascii.read('CrabNebula_HESS_2006.dat')

ene = u.Quantity(data['ene'])
ene0 = np.sqrt(ene[0]*ene[-1])

## Set initial parameters

p0=np.array((1.5e-12,2.7,0.12,))
labels=['norm','alpha','beta']

## Model definition

from gammafit.models import LogParabola

# initialise an instance of ECPL
LP = LogParabola(1, ene0, 2, 0.5)

def logparabola(pars,data):

    LP.amplitude = pars[0]
    LP.alpha = pars[1]
    LP.beta = pars[2]

    return LP(data) * data['flux'].unit

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
            model=logparabola, prior=lnprior, nwalkers=128, nburn=50, nrun=10,
            threads=4)

## Save sampler
    from astropy.extern import six
    from six.moves import cPickle
    sampler.pool=None
    cPickle.dump(sampler,open('CrabNebula_logparabola_sampler.pickle','wb'))

## Diagnostic plots
    gammafit.generate_diagnostic_plots('CrabNebula_logparabola',sampler,
            sed=True,last_step=False)


