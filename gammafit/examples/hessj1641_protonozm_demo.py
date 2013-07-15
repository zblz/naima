#!/usr/bin/python

import numpy as np
import gammafit

import astropy.units as u

## Read data

dataf=np.loadtxt('hessj1641.spec')

ene=dataf[:,1]
dene=np.array(zip(dataf[:,3],dataf[:,2]))
flux=dataf[:,4]
dflux=np.array(zip(dataf[:,6],dataf[:,5]))
ul=(dflux[:,1]==0.)
cl=np.average(dflux[ul][:,0])

data=gammafit.build_data_dict(ene,dene,flux,dflux,ul,cl)

## Model definition

def ProtonOZM(pars,data):

    # Add two spectral points at edges of spectra
    outspecene=np.concatenate((
        (data['ene'][0]-2*data['dene'][0][0],),
        data['ene'],
        (data['ene'][-1]+2*data['dene'][-1][1],),
        ))

    emin=data['ene'][-data['ul']][0]
    emax=data['ene'][-data['ul']][-1]
    enemid=np.sqrt(emin*emax)
    # peak gamma energy production is ~0.1*Ep, so enemid corresponds to Ep=10*enemid
    # From tests, a better correspondence to the decorrelation energy of a
    # powerlaw is obtained with Ep=30*enemid
    norm_ene=30.*enemid

    norm=pars[0]
    index=pars[1]

    ozm=gammafit.ProtonOZM(outspecene*1e12,
            norm=norm,
            norm_energy=norm_ene*1e12,
            index=index,
            cutoff=1e16,
            nolog=True)

    ozm.calc_outspec()

    model=ozm.specpptev # 1/s/cm2/TeV

    modelfordata=model[1:-1]

    # compute proton distribution for blob
    Epmin=data['ene'][0]*1e-3
    Epmax=data['ene'][-1]*1e3

    protonene=np.logspace(np.log10(Epmin),np.log10(Epmax),30)
    protondist=ozm.Jp(protonene)*protonene**2*u.TeV.to('erg')

    del(ozm)

    return modelfordata , np.array((outspecene,model)), \
            np.array((protonene,protondist))


## Prior definition

def lnprior(pars):
	"""
	Return probability of parameter values according to prior knowledge.
	Parameter limits should be done here through uniform prior ditributions
	"""

	logprob = \
			+ gammafit.uniform_prior(pars[0],0.,np.inf)
            #+ gammafit.normal_prior(pars[1],2.,1.) \
			#+ gammafit.uniform_prior(pars[2],0.,150.)
			#+ gammafit.uniform_prior(pars[3],0.25,np.inf)

	return logprob

## Set initial parameters

p0=np.array((1e36,2.0))

## Run sampler

sampler,pos = gammafit.run_sampler(p0=p0,data=data,model=ProtonOZM,prior=lnprior,
        nwalkers=250,nburn=100,nrun=100,threads=8)

## Diagnostic plots

gammafit.generate_diagnostic_plots('hessj1641_prot',sampler,modelidxs=[0,1,],pdf=True)
