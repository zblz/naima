#!/usr/bin/env python
import numpy as np
import gammafit

from astropy import units as u

## Read data

spec=np.loadtxt('CrabNebula_HESS_2006.dat')

ene=spec[:,0]
flux=spec[:,3]
perr=spec[:,4]
merr=spec[:,5]
dflux=np.array(list(zip(merr,perr)))

data=gammafit.build_data_dict(ene,None,flux,dflux)

## Set initial parameters

p0=np.array((5e-25,2.34,80.,))
labels=['norm','index','cutoff']

## Model definition

def ppgamma(pars,data):

    enemid=np.sqrt(data['ene'][0]*data['ene'][-1])
    # peak gamma energy production is ~0.1*Ep, so enemid corresponds to Ep=10*enemid
    # If a cutoff is present, this should be reduced
    norm_ene = 5.*enemid

    norm   = pars[0]
    index  = pars[1]
    cutoff = pars[2]

    ozm=gammafit.ProtonOZM(
            data['ene']*1e12, norm,
            cutoff      = cutoff*1e12,
            index       = index,
            norm_energy = norm_ene*1e12,
            nolog       = True,
            )

    ozm.calc_outspec()

    model=ozm.specpptev # 1/s/cm2/TeV

    # compute proton distribution for blob
    Epmin=data['ene'][0]*1e-2
    Epmax=data['ene'][-1]*1e3

    protonene=np.logspace(np.log10(Epmin),np.log10(Epmax),50)
    protondist=ozm.Jp(protonene)*protonene**2*u.TeV.to('erg')

    del(ozm)

    return model, np.array((data['ene'],model)), \
            np.array((protonene,protondist))

## Prior definition

def lnprior(pars):
	"""
	Return probability of parameter values according to prior knowledge.
	Parameter limits should be done here through uniform prior ditributions
	"""

	logprob = gammafit.uniform_prior(pars[0],0.,np.inf) \
            + gammafit.uniform_prior(pars[1],-1,5) \
			+ gammafit.uniform_prior(pars[2],0.,np.inf) \

	return logprob

## Run sampler

sampler,pos = gammafit.run_sampler(data=data, p0=p0, labels=labels, model=ppgamma,
        prior=lnprior, nwalkers=500, nburn=200, nrun=100, threads=8)

## Diagnostic plots

gammafit.generate_diagnostic_plots('CrabNebula_proton',sampler,converttosed=[True,False])

## Save sampler

#import cPickle as pickle
#sampler.pool=None
#pickle.dump(sampler,open('CrabNebula_function_sampler.pickle','wb'))

