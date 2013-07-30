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

def ElectronIC(pars,data):

    norm   = pars[0]
    index  = pars[1]
    cutoff = pars[2]*1e12

    outspecene=data['ene']*1e12

    ozm=gammafit.ElectronOZM(
            outspecene, norm,
            index=index,
            cutoff=cutoff,
            seedspec=['CMB',],
            norm_energy=1e13,
            nolog=True,
            evolve_nelec=False,
            )

    ozm.calc_nelec()
    ozm.calc_ic()

    model=ozm.specictev # 1/s/cm2/TeV

    nelec=ozm.nelec[:-1]*gammafit.mec2*ozm.gam[:-1]*np.diff(ozm.gam)
    elec_energy=ozm.gam[:-1]*gammafit.mec2TeV

    del ozm

    return model, np.array((data['ene'],model)), np.array((elec_energy,nelec))

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

p0=np.array((3e18,3.2,45.0,))
labels=['norm','index','cutoff']

## Run sampler

sampler,pos = gammafit.run_sampler(data=data, p0=p0, labels=labels, model=ElectronIC,
        prior=lnprior, nwalkers=250, nburn=100, nrun=100, threads=8)

## Diagnostic plots

gammafit.generate_diagnostic_plots('CrabNebula_electron',sampler)

## Save sampler

#import cPickle as pickle
#sampler.pool=None
#pickle.dump(sampler,open('CrabNebula_function_sampler.pickle','wb'))

