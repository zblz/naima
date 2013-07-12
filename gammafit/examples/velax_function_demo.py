#!/usr/bin/python

import numpy as np
import gammafit

import astropy.units as u

## Read data

spec=np.loadtxt('velax_cocoon.spec')

ene=spec[:,0]
dene=gammafit.generate_energy_edges(ene)

flux=spec[:,1]
merr=spec[:,1]-spec[:,2]
perr=spec[:,3]-spec[:,1]
dflux=np.array(zip(merr,perr))

ul=(dflux[:,0]==0.)
cl=0.99

data=gammafit.build_data_dict(ene,dene,flux,dflux,ul,cl)

## Model definition

def cutoffexp(pars,data):
    """
    Powerlaw with exponential cutoff

    Parameters:
        - 0: PL normalization
        - 1: PL index
        - 2: cutoff energy
        - 3: cutoff exponent (beta)
    """

    x=data['ene']
    # take logarithmic mean of first and last data points as normalization energy
    x0=np.sqrt(x[0]*x[-1])

    N     = pars[0]
    gamma = pars[1]
    ecut  = pars[2]
    #beta  = pars[3]
    beta  = 1.

    return N*(x/x0)**-gamma*np.exp(-(x/ecut)**beta)

## Prior definition

def lnprior(pars):
	"""
	Return probability of parameter values according to prior knowledge.
	Parameter limits should be done here through uniform prior ditributions
	"""

	logprob = gammafit.uniform_prior(pars[0],0.,np.inf) \
            + gammafit.uniform_prior(pars[1],-1,5) \
			+ gammafit.uniform_prior(pars[2],0.,np.inf) \
			#+ gammafit.uniform_prior(pars[3],0.5,1.5)

	return logprob

## Set initial parameters

p0=np.array((1e-9,1.4,14.0,))
labels=['norm','index','cutoff','beta']

## Run sampler

sampler,pos = gammafit.run_sampler(data=data, p0=p0, labels=labels, model=cutoffexp,
        prior=lnprior, nwalkers=1000, nburn=100, nrun=100, threads=8)

## Diagnostic plots

gammafit.generate_diagnostic_plots('velax_function',sampler)

