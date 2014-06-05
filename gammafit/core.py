# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np

from astropy import constants as const

mec2TeV=(const.m_e*const.c**2).to('eV').value/1e12
mec2=(const.m_e*const.c**2).to('erg').value

__all__=["normal_prior","uniform_prior","get_sampler","run_sampler"]


## Likelihood functions

# Prior functions

def uniform_prior(value,umin,umax):
    """Uniform prior distribution.
    """
    if umin <= value <= umax:
        return 0.0
    else:
        return - np.inf

def normal_prior(value,mean,sigma):
    """Normal prior distribution.
    """
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
        if (type(modelout)==tuple or type(modelout)==list) and (type(modelout)!=np.ndarray):
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
    #outstr = '{:8.2g} '*len(pars) + '{:8.3g} '*3
    #outargs = list(pars) + [lnprob_model,lnprob_priors,total_lnprob]
# TODO: convert following print to logger
    #print outstr.format(*outargs)

    return total_lnprob,blob

## Sampler funcs

def _run_mcmc(sampler,pos,nrun):
    for i, out in enumerate(sampler.sample(pos, iterations=nrun)):
        progress=(100. * float(i) / float(nrun))
        if progress%5<(5./float(nrun)):
            print("\nProgress of the run: {0:.0f} percent ({1} of {2} steps)".format(int(progress),i,nrun))
            npars=out[0].shape[-1]
            paravg,parstd=[],[]
            for npar in range(npars):
                paravg.append(np.average(out[0][:,npar]))
                parstd.append(np.std(out[0][:,npar]))
            print("                            "+(" ".join(["{%i:-^10}"%i for i in range(npars)])).format(*sampler.labels))
            print("  Last ensemble average : "+(" ".join(["{%i:^10.3g}"%i for i in range(npars)])).format(*paravg))
            print("  Last ensemble std     : "+(" ".join(["{%i:^10.3g}"%i for i in range(npars)])).format(*parstd))
            print("  Last ensemble lnprob  :  avg: {0:.3f}, max: {1:.3f}".format(np.average(out[1]),np.max(out[1])))
    return sampler,out[0]

## Placeholder model: Powerlaw with exponential

def _cutoffexp(pars,data):
    """
    Powerlaw with exponential cutoff.

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
    """Make an MCMC sampler.
    
    Parameters
    ----------
    nwalkers : int
        Number of walkers
    nburn : int
        TODO
    guess : bool
        TODO
    p0 : array
        Initial position vector.
    data : TODO
        Data
    model : TODO
        TODO
    prior : TODO
        TODO
    labels : TODO
        TODO
    threads : int
        Number of threads to use for sampling.

    Returns
    -------
    sampler : `emcee.EnsembleSampler`
        Sampler
    pos : `numpy.array`
        Position array

    See also
    --------
    emcee.EnsembleSampler
    """
    import emcee

    if data==None:
        print('Need to provide data!')
        raise TypeError

    # Add parameter labels if not provided or too short
    if labels == None:
        # First is normalization
        labels = ['norm',]+['par{0}'.format(i) for i in range(1,len(p0))]
    elif len(labels)<len(p0):
        labels+=['par{0}'.format(i) for i in range(len(labels),len(p0))]

    if guess:
        # guess normalization parameter from p0
        modelout=model(p0,data)
        if (type(modelout)==tuple or type(modelout)==list) and (type(modelout)!=np.ndarray):
            spec=modelout[0]
        else:
            spec=modelout
        p0[labels.index('norm')]*=np.trapz(data['flux'],data['ene'])/np.trapz(spec,data['ene'])

    ndim=len(p0)

    sampler=emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=[data,model,prior],threads=threads)

    # Add data and parameters properties to sampler
    sampler.data = data
    sampler.labels = labels

    # Initialize walkers in a ball of relative size 2% in all dimensions
    p0var=np.array([ 0.02*pp for pp in p0])
    p0=emcee.utils.sample_ball(p0,p0var,nwalkers)

    if nburn>0:
        print('Burning in the {0} walkers with {1} steps...'.format(nwalkers,nburn))
        sampler,pos = _run_mcmc(sampler,p0,nburn)
    else:
        pos=p0

    return sampler,pos


def run_sampler(nrun=100,sampler=None,pos=None,**kwargs):
    """Run an MCMC sampler.
    
    Extra ``kwargs`` are passed to `get_sampler`.
    
    Parameters
    ----------
    nrun : int
        TODO
    sampler : `emcee.EnsembleSampler`
        Sampler
    pos : TODO
        TODO
    
    Returns
    -------
    sampler : `emcee.EnsembleSampler`
        Modified input ``sampler``
    pos : TODO
        TODO
    """

    if sampler==None or pos==None:
        sampler,pos=get_sampler(**kwargs)

    print('\nWalker burn in finished, running {0} steps...'.format(nrun))
    sampler.reset()
    sampler,pos=_run_mcmc(sampler,pos,nrun)

    return sampler,pos
