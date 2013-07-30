#!/usr/bin/python
import numpy as np
import gammafit

## Read data

spec=np.loadtxt('CrabNebula_HESS_2006.dat')

ene=spec[:,0]
flux=spec[:,3]
perr=spec[:,4]
merr=spec[:,5]
dflux=np.array(zip(merr,perr))

data=gammafit.build_data_dict(ene,None,flux,dflux)

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

    ene=data['ene']
    # take logarithmic mean of first and last data points as normalization
    # energy (reasonable approximation of decorrelation energy for gamma~2)
    ene0=np.sqrt(ene[0]*ene[-1])

    N     = pars[0]
    gamma = pars[1]
    ecut  = pars[2]

    return N*(ene/ene0)**-gamma*np.exp(-(ene/ecut))

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

p0=np.array((1.5e-12,2.4,15.0,))
labels=['norm','index','cutoff']

## Run sampler

sampler,pos = gammafit.run_sampler(data=data, p0=p0, labels=labels, model=cutoffexp,
        prior=lnprior, nwalkers=1000, nburn=100, nrun=100, threads=8)

## Diagnostic plots

gammafit.generate_diagnostic_plots('CrabNebula_function',sampler)

## Save sampler

#import cPickle as pickle
#sampler.pool=None
#pickle.dump(sampler,open('CrabNebula_function_sampler.pickle','wb'))

