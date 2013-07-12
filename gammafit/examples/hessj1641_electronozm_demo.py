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

import onezone


for B in np.logspace(-7,-4,7):

    def ElectronOZM(pars,data):

        norm=pars[0]
        index=pars[1]

        outspecene=data['ene']*1e12

        ozm=onezone.ElectronOZM(outspecene, norm, index=index, B=B,
                norm_energy=5e13,
                cutoff=1e50, nolog=True, evolve_nelec=True,
                gmin=1e6,gmax=3e10)

        ozm.calc_nelec()
        ozm.calc_ic()

        model=ozm.specictev # 1/s/cm2/TeV

        nelec=ozm.nelec[:-1]*onezone.mec2*ozm.gam[:-1]*np.diff(ozm.gam)
        elec_energy=ozm.gam[:-1]*onezone.mec2TeV

        del ozm

        return model, np.array((data['ene'],model)), np.array((elec_energy,nelec))


## Prior definition

    def lnprior(pars):
        """
        Return probability of parameter values according to prior knowledge.
        Parameter limits should be done here through uniform prior ditributions
        """

        logprob = \
                + gammafit.uniform_prior(pars[0],0.,np.inf) \
                + gammafit.normal_prior(pars[1],2.,1.) \
                #+ gammafit.uniform_prior(pars[2],0.,150.)
                #+ gammafit.uniform_prior(pars[3],0.25,np.inf)

        return logprob

## Set initial parameters

    p0=np.array((1e12,2.0))
    labels=['norm','index']

## Run sampler

    sampler,pos = gammafit.run_sampler(p0=p0,data=data,model=ElectronOZM,prior=lnprior,
            labels=labels, nwalkers=250,nburn=100,nrun=100,threads=8)

## Diagnostic plots

    gammafit.generate_diagnostic_plots('hessj1641_elec_B{0:.1f}'.format(np.log10(B)),sampler,modelidxs=[0,1,])

