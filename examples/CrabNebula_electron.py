#!/usr/bin/env python
import numpy as np
import gammafit
import astropy.units as u
from astropy.constants import m_e,c

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

p0=np.array((3e18,3.2,45.0,))
labels=['norm','index','cutoff']

## Model definition

def ElectronIC(pars,data):

    norm   = pars[0]
    index  = pars[1]
    cutoff = pars[2]*u.TeV

    outspecene = data['ene']

    ozm=gammafit.ElectronOZM(
            outspecene, norm,
            index=index,
            cutoff=cutoff,
            seedspec=['CMB',],
            norm_energy=10.*u.TeV,
            nolog=True,
            evolve_nelec=False,
            )

    ozm.calc_nelec()
    ozm.calc_ic()

    # convert to same units as observed differential spectrum
    model=ozm.specic.to('1/(s TeV)')/u.cm**2

    mec2=m_e*c**2

    nelec=ozm.nelec[:-1]*mec2.cgs.value*ozm.gam[:-1]*np.diff(ozm.gam)*u.Unit('erg')
    elec_energy=ozm.gam[:-1]*mec2.to('TeV')

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

## Run sampler

sampler,pos = gammafit.run_sampler(data=data, p0=p0, labels=labels, model=ElectronIC,
        prior=lnprior, nwalkers=500, nburn=100, nrun=50, threads=4)

## Diagnostic plots

gammafit.generate_diagnostic_plots('CrabNebula_electron',sampler,sed=[True,None])

## Save sampler

#import cPickle as pickle
#sampler.pool=None
#pickle.dump(sampler,open('CrabNebula_function_sampler.pickle','wb'))

