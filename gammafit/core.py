#!/usr/bin/env python
import numpy as np

from astropy import constants as const
from astropy import units as u

mec2TeV=(const.m_e*const.c**2).to('eV').value/1e12
mec2=(const.m_e*const.c**2).to('erg').value

__all__=["normal_prior","uniform_prior","get_sampler","run_sampler"]


## Likelihood functions

# Prior functions

def uniform_prior(value,umin,umax):
    if umin <= value <= umax:
        return 0.0
    else:
        return - np.inf

def normal_prior(value,mean,sigma):
    return - 0.5 * (2 * np.pi * sigma) - (value - mean) ** 2 / (2. * sigma)

# Probability function

def lnprobmodel(model,data):

    ul=data['ul']
    notul = -ul

    difference = model[notul]-data['flux'][notul]

    if np.rank(data['dflux'])>1:
# use different errors for model above or below data
        sign=difference>0
        loerr,hierr=1*-sign,1*sign
        logprob =  - difference**2/(2.*(loerr*data['dflux'][notul][:,0]+hierr*data['dflux'][notul][:,1])**2)
    else:
        logprob =  - difference**2/(2.*data['dflux'][notul]**2)

    totallogprob = np.sum(logprob)

    if np.sum(ul)>0:
        # deal with upper limits at CL set by data['cl']
        violated_uls = np.sum(model[ul]>data['flux'][ul])
        totallogprob += violated_uls * np.log(1.-data['cl'])

    return totallogprob

def lnprob(pars,data,modelfunc,priorfunc):

    lnprob_priors = priorfunc(pars)

# If prior is -np.inf, avoid calling the function as invalid calls may be made,
# and the result will be discarded anyway
    if not np.isinf(lnprob_priors):
        modelout = modelfunc(pars,data)

        # Save blobs or save model if no blobs given
        if type(modelout)==tuple:
            #print len(modelout)
            model = modelout[0]
            blob  = modelout[1:]
        else:
            model = modelout
            blob  = (np.array((data['ene'],modelout)),)

        lnprob_model  = lnprobmodel(model,data)
    else:
        lnprob_model = 0.0
        blob=None

    total_lnprob  = lnprob_model + lnprob_priors

    # Print parameters and total_lnprob
    outstr = '{:8.2g} '*len(pars) + '{:8.3g} '*3
    outargs = list(pars) + [lnprob_model,lnprob_priors,total_lnprob]
# TODO: convert following print to logger
    #print outstr.format(*outargs)

    return total_lnprob,blob

## Sampler funcs

def _run_mcmc(sampler,pos,nrun):
    for i, out in enumerate(sampler.sample(pos, iterations=nrun)):
        progress=int(100 * i / nrun)
        if progress%5==0:
            print("\nProgress of the run: {:.0f} percent ({} of {} steps)".format(int(progress),i,nrun))
            npars=out[0].shape[-1]
            paravg,parstd=[],[]
            for npar in range(npars):
                paravg.append(np.average(out[0][:,npar]))
                parstd.append(np.std(out[0][:,npar]))
            print("                            "+("{:-^10} "*npars).format(*sampler.labels))
            print("  Last ensemble average : "+("{:^10.3g} "*npars).format(*paravg))
            print("  Last ensemble std     : "+("{:^10.3g} "*npars).format(*parstd))
            print("  Last ensemble lnprob  :  avg: {:.3f}, max: {:.3f}".format(np.average(out[1]),np.max(out[1])))
    return sampler,out[0]

## Placeholder model: Powerlaw with exponential

def _cutoffexp(pars,data):
    """
    Powerlaw with exponential cutoff

    Parameters:
        - 0: PL normalization
        - 1: PL index
        - 2: cutoff energy
    """

    ene=data['ene']
    ene0=np.exp(np.average(np.log(ene)))

    N     = pars[0]
    gamma = pars[1]
    ecut  = pars[2]

    model = N*(ene/ene0)**-gamma*np.exp(-(ene/ecut))

    return model

# Placeholder prior and initial parameters
_prior=lambda x: 0.0
_p00=np.array((1e-11,2,10))

def get_sampler(nwalkers=500, nburn=30, guess=True, p0=_p00, data=None,
        model=_cutoffexp, prior=_prior, labels=None, threads=8):
    #TODO docstring

    import emcee


    if data==None:
        print 'Need to provide data!'
        raise TypeError

    # Add parameter labels if not provided or too short
    if labels == None:
        # First is normalization
        labels = ['norm',]+['par{}'.format(i) for i in range(1,len(p0))]
    elif len(labels)<len(p0):
        labels+=['par{}'.format(i) for i in range(len(labels),len(p0))]

    if guess:
        # guess normalization parameter from p0
        modelout=model(p0,data)
        if type(modelout)==tuple:
            spec=modelout[0]
        else:
            spec=modelout
        p0[labels.index('norm')]*=np.trapz(data['flux'],data['ene'])/np.trapz(spec,data['ene'])

    ndim=len(p0)

    sampler=emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=[data,model,prior],threads=threads)

    # Add data and parameters properties to sampler
    sampler.data = data
    sampler.labels = labels

    # Initialize walkers in a ball of relative size 10% in all dimensions
    p0var=np.array([ pp/10. for pp in p0])
    p0=emcee.utils.sample_ball(p0,p0var,nwalkers)

    print 'Burning in the walkers with {} steps...'.format(nburn)
    #burnin = sampler.run_mcmc(p0,nburn)
    #pos=burnin[0]
    sampler,pos=_run_mcmc(sampler,p0,nburn)

    return sampler,pos


def run_sampler(nrun=100,sampler=None,pos=None,**kwargs):
    #TODO docstring

    if sampler==None or pos==None:
        sampler,pos=get_sampler(**kwargs)

    print 'Walker burn in finished, running {} steps...'.format(nrun)
    sampler.reset()
    sampler,pos=_run_mcmc(sampler,pos,nrun)

    return sampler,pos
