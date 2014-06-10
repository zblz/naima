#!/usr/bin/env python
import numpy as np
import gammafit

from astropy import units as u

## Read data

spec=np.loadtxt('CrabNebula_HESS_2006.dat')

flux_unit = u.Unit('1/(cm2 s TeV)')

ene=spec[:,0]*u.TeV
flux=spec[:,3]*flux_unit
perr=spec[:,4]
merr=spec[:,5]
dflux=np.array((merr,perr))*flux_unit

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
    cutoff = pars[2]*u.TeV

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
	    + gammafit.uniform_prior(pars[2],0.,np.inf)

	return logprob

## Run sampler

sampler,pos = gammafit.run_sampler(data=data, p0=p0, labels=labels, model=ppgamma,
        prior=lnprior, nwalkers=500, nburn=100, nrun=50, threads=4)

## Diagnostic plots

gammafit.generate_diagnostic_plots('CrabNebula_proton',sampler,sed=[True,None])

## Save sampler

#import cPickle as pickle
#sampler.pool=None
#pickle.dump(sampler,open('CrabNebula_function_sampler.pickle','wb'))

