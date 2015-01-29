#!/usr/bin/env python
import numpy as np
import naima
import astropy.units as u
from astropy.io import ascii

## Read data

data=ascii.read('CrabNebula_HESS_2006.dat')

ene = u.Quantity(data['energy'])
ene0 = np.sqrt(ene[0]*ene[-1])

## Set initial parameters

p0=np.array((1.5e-12,2.7,0.12,))
labels=['norm','alpha','beta']

## Model definition

from naima.models import LogParabola

# initialise an instance of ECPL
flux_unit = data['flux'].unit
LP = LogParabola(1e-12 * flux_unit, ene0, 2, 0.5)

def logparabola(pars,data):

    LP.amplitude = pars[0] * flux_unit
    LP.alpha = pars[1]
    LP.beta = pars[2]

    return LP(data)

## Prior definition

def lnprior(pars):
	"""
	Return probability of parameter values according to prior knowledge.
	Parameter limits should be done here through uniform prior ditributions
	"""

	logprob = naima.uniform_prior(pars[0],0.,np.inf) \
                + naima.uniform_prior(pars[1],-1,5)

	return logprob

if __name__=='__main__':
## Run sampler

    sampler,pos = naima.run_sampler(data_table=data, p0=p0, labels=labels,
            model=logparabola, prior=lnprior, nwalkers=256, nburn=50, nrun=10,
            threads=4)

## Save sampler
    from astropy.extern import six
    from six.moves import cPickle
    sampler.pool=None
    cPickle.dump(sampler,open('CrabNebula_logparabola_sampler.pickle','wb'))

## Diagnostic plots
    naima.save_diagnostic_plots('CrabNebula_logparabola',sampler,
            sed=True,last_step=False)


