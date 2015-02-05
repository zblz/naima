#!/usr/bin/env python
import numpy as np
import naima
import astropy.units as u
from astropy.io import ascii

## Read data

data=ascii.read('CrabNebula_HESS_2006_ipac.dat')

## Model definition

from naima.models import LogParabola

def logparabola(pars,data):
    amplitude = pars[0] * data['flux'].unit
    alpha = pars[1]
    beta = pars[2]
    LP = LogParabola(amplitude, 1*u.TeV, alpha, beta)

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
    # Set initial parameters
    p0=np.array((1.5e-12,2.7,0.12,))
    labels=['norm','alpha','beta']

    # Run sampler
    sampler,pos = naima.run_sampler(data_table=data, p0=p0, labels=labels,
            model=logparabola, prior=lnprior, nwalkers=128, nburn=100, nrun=50,
            threads=4, prefit=True)

    # Save sampler
    from astropy.extern import six
    from six.moves import cPickle
    sampler.pool=None
    cPickle.dump(sampler,open('CrabNebula_logparabola_sampler.pickle','wb'))

    # Diagnostic plots
    naima.save_diagnostic_plots('CrabNebula_logparabola',sampler,
            sed=True,last_step=False)
    naima.save_results_table('CrabNebula_logparabola',sampler)


