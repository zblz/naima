#!/usr/bin/python

import numpy as np
import emcee_tools as et
from scipy import stats

import astropy.units as u

# read data

spec=np.loadtxt('velax_cocoon.spec')

ene=spec[:,0]
dene=et.generate_energy_edges(ene)

flux=spec[:,1]
merr=spec[:,1]-spec[:,2]
perr=spec[:,3]-spec[:,1]
dflux=np.array(zip(merr,perr))

ul=(dflux[:,0]==0.)
cl=0.99

# data is a dict with the fields:
# ene dene flux dflux ul cl
data={}
for val in ['ene', 'dene', 'flux', 'dflux', 'ul', 'cl']:
    data[val]=eval(val)

## Model definition

def cutoffexp(pars,data):
    """
    Powerlaw with exponential cutoff

    Parameters:
        - 0: PL index
        - 1: PL normalization
        - 2: cutoff energy
        - 3: cutoff exponent (beta)
    """

    x=data['ene']
    x0=stats.gmean(x)

    gamma = pars[0]
    N     = pars[1]
    ecut  = pars[2]
    beta  = pars[3]

    return N*(x/x0)**-gamma*np.exp(-(x/ecut)**beta)

## Prior definition

def lnprior(pars):
	"""
	Return probability of parameter values according to prior knowledge.
	Parameter limits should be done here through uniform prior ditributions
	"""

	logprob = et.uniform_prior(pars[0],-1,5) \
			+ et.uniform_prior(pars[1],0.,np.inf) \
			+ et.uniform_prior(pars[2],0.,np.inf) \
			+ et.uniform_prior(pars[3],0.25,np.inf)

	return logprob

## Set initial parameters

p0=np.array((2.0,1e-11,10.0,1.0))

## Run fit

sampler,pos = et.run_sampler(p0=p0,data=data,model=cutoffexp,prior=lnprior,nburn=100,nrun=100,threads=1)

## Diagnostic plots

## Corner plot
f = et.corner(sampler.flatchain,labels=['gamma','norm','ecut','beta'])
f.savefig('velax_corner.png')

## Chains

for par in range(4):
	f = et.plot_chain(sampler.chain,par)
	f.savefig('velax_chain_par{}.png'.format(par))

## Fit

f = et.plot_fit(sampler,xlabel='Energy',ylabel='Flux')
f.savefig('velax_fit.png')





