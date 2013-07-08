#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d

import emcee
try:
    from triangle import corner
except ImportError:
    print 'triangle.py not installed, corner plot will not be available'

from astropy import constants as const
from astropy import units as u

mec2TeV=(const.m_e*const.c**2).to('eV').value/1e12
mec2=(const.m_e*const.c**2).to('erg').value

## Placeholder model: Powerlaw with exponential

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

    N     = pars[0]
    gamma = pars[1]
    ecut  = pars[2]
    beta  = pars[3]

    model = N*(x/x0)**-gamma*np.exp(-(x/ecut)**beta)

    return model

p00=np.array((1e-11,2,10,1))

## Placeholder prior function: Flat

def flatprior(pars):
    return 0.0

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
    lnprob_priors = priorfunc(pars)
    total_lnprob  = lnprob_model + lnprob_priors

    # Print parameters and total_lnprob
    outstr = '{:6.2g} '*len(pars) + '{:5g}'
    outargs = list(pars) + [total_lnprob,]
# TODO: convert following print to logger
    #print outstr.format(*outargs)

    return total_lnprob,blob

## Sampler funcs

def _run_mcmc(sampler,pos,nrun):
    for i, out in enumerate(sampler.sample(pos, iterations=nrun)):
        progress=int(100 * i / nrun)
        if progress%5==0:
            print("Progress of the run: {0:.0f} percent".format(int(progress)))
    return sampler,out[0]


def get_sampler(nwalkers=600,nburn=30,guess=True,p0=p00,data=None,model=cutoffexp,prior=flatprior,
        threads=8):

    if data==None:
        print 'Need to provide data!'
        raise TypeError

    if guess:
        # guess normalization parameter from p0
        lp,blob=lnprob(p0,data,model,prior)
        ene=blob[0][0]
        spec=blob[0][1]
        p0[0]*=np.trapz(data['flux'],data['ene'])/np.trapz(spec,ene)

    ndim=len(p0)
    sampler=emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=[data,model,prior],threads=threads)
    sampler.data=data

    # Initialize walkers in a ball of relative size 10% in all dimensions
    p0var=np.array([ pp/10. for pp in p0])
    p0=emcee.utils.sample_ball(p0,p0var,nwalkers)

    print 'Burning in the walkers with {0} steps...'.format(nburn)
    #burnin = sampler.run_mcmc(p0,nburn)
    #pos=burnin[0]
    sampler,pos=_run_mcmc(sampler,p0,nburn)

    return sampler,pos


def run_sampler(nrun=100,sampler=None,pos=None,**kwargs):
    if sampler==None or pos==None:
        sampler,pos=get_sampler(**kwargs)

    print 'Walker burn in finished, running {0} steps...'.format(nrun)
    sampler.reset()
    sampler,pos=_run_mcmc(sampler,pos,nrun)

    return sampler,pos


## Plot funcs

def plot_chain(chain,p=None,**kwargs):
    if p==None:
        npars=chain.shape[-1]
        for pp in range(npars):
            _plot_chain_func(chain,pp,**kwargs)
        fig = None
    else:
        fig = _plot_chain_func(chain,p,**kwargs)

    return fig


def _plot_chain_func(chain,p=None,last_step=False):
    if len(chain.shape)>2:
        traces=chain[:,:,p]
        if last_step==True:
#keep only last step
            dist=traces[:,-1]
        else:
#convert chain to flatchain
            dist=traces.flatten()
    else:
# we have a flatchain, plot everything
        #dist=chain[:,p]
        print 'we need the chain to plot the traces, not a flatchain!'
        return None

    nwalkers=traces.shape[0]
    nsteps=traces.shape[1]

    logplot=False
    #if dist.max()/dist.min()>5:
        #logplot=True

    f=plt.figure()

    ax1=f.add_subplot(221)
    ax2=f.add_subplot(122)
    #ax3=f.add_subplot(223)

# plot five percent of the traces darker

    colors=np.where(np.arange(nwalkers)/float(nwalkers)>0.95,'#550000','0.5')

    for t,c in zip(traces,colors): #range(nwalkers):
        #ax1.plot(traces[i,:],c=colors[i])
        ax1.plot(t,c=c,lw=1)
    ax1.set_xlabel('step')
    ax1.set_ylabel('Parameter {0}'.format(p))
    ax1.set_title('Walker traces')
    if logplot:
        ax1.set_yscale('log')

    #nbins=25 if last_step else 100
    nbins=min(max(25,int(len(dist)/100.)),100)
    xlabel='Parameter {0}'.format(p)
    if logplot:
        dist=np.log10(dist)
        xlabel='log10('+xlabel+')'
    n,x,patch=ax2.hist(dist,nbins,histtype='stepfilled',color='#CC0000',lw=0,normed=1)
    kde=stats.kde.gaussian_kde(dist)
    ax2.plot(x,kde(x),c='k',label='KDE')
    #for m,ls,lab in zip([np.mean(dist),np.median(dist)],('--','-.'),('mean: {0:.4g}','median: {0:.4g}')):
        #ax2.axvline(m,ls=ls,c='k',alpha=0.5,lw=2,label=lab.format(m))
    ns=len(dist)
    quant=[0.01,0.16,0.5,0.84,0.99]
    sdist=np.sort(dist)
    xquant=[sdist[int(q*ns)] for q in quant]
    ax2.axvline(xquant[quant.index(0.5)],ls='--',c='k',alpha=0.5,lw=2,label='50% quantile')
    ax2.axvspan(xquant[quant.index(0.16)],xquant[quant.index(0.84)],color='0.5',alpha=0.25,label='68% CI')
    #ax2.legend()
    ax2.set_xlabel(xlabel)
    ax2.set_title('posterior distribution')
    ax2.set_ylim(top=n.max()*1.05)

    # Plot distribution parameters on lower-left

    mean,median,std=np.mean(dist),np.median(dist),np.std(dist)
    xmode=np.linspace(mean-np.sqrt(3)*std,mean+np.sqrt(3)*std,100)
    mode=xmode[np.argmax(kde(xmode))]

    if logplot:
        mode=10**mode
        mean=10**mean
        std=np.std(10**dist)
        xquant=[10**q for q in xquant]
        median=10**np.median(dist)
    else:
        median=np.median(dist)

    acort=[]
    npars=chain.shape[-1]
    try:
        import acor
        for npar in range(npars):
            acort.append(acor.acor(chain[:,:,npar])[0])
    except:
        acort=[np.nan,]*npars

    if last_step:
        clen='last step ensemble'
    else:
        clen='whole chain'

    quantiles=dict(zip(quant,xquant))

    f.text(0.1,0.45,'Walkers: {0} \nSteps in chain: {1} \n'.format(nwalkers,nsteps) + \
            'Autocorrelation times (steps): '+('{:.1f} '*npars).format(*acort) + '\n' +\
            'Distribution properties for the {clen}:\n \
       - mode: {mode:.3g} \n \
       - mean: {mean:.3g} \n \
       - median: {median:.3g} \n \
       - std: {std:.3g} \n \
       - 68% CI: ({quant16:.3g},{quant84:.3g})\n \
       - 99% CI: ({quant01:.3g},{quant99:.3g})\n \
       - mean +/- std CI: ({meanstd[0]:.3g},{meanstd[1]:.3g})'.format(
                mean=mean,median=median,std=std,
                quant01=quantiles[0.01],
                quant16=quantiles[0.16],
                quant84=quantiles[0.84],
                quant99=quantiles[0.99],
                meanstd=(mean-std,mean+std),clen=clen,mode=mode,),
            ha='left',va='top')


    f.tight_layout()

    #f.show()

    return f

def calc_fit_CI(sampler,modelidx=0,confs=[3,1],last_step=True):

    if last_step:
        model=np.array([m[modelidx][1] for m in sampler.blobs[-1]])
        dists=sampler.chain[-1].T
    else:
        nsteps=len(sampler.blobs)
        model=[]
        for i in range(nsteps):
            for m in sampler.blobs[i]:
                model.append(m[modelidx][1])
        model=np.array(model)
        dists=sampler.flatchain.T

    modelx=sampler.blobs[-1][0][modelidx][0]

    nwalkers=len(model)
    CI=[]
    for conf in confs:
        fmin=stats.norm.cdf(-conf)
        fmax=stats.norm.cdf(conf)
        #print conf,fmin,fmax
        ymin,ymax=[],[]
        for fr,y in ((fmin,ymin),(fmax,ymax)):
            nf=int((fr*nwalkers))
# TODO: logger
            #print conf,fr,nf
            for i,x in enumerate(modelx):
                ysort=np.sort(model[:,i])
                y.append(ysort[nf])
        CI.append((ymin,ymax))

    # Find best-fit parameters as those in the chain with a highest log
    # probability
    #argmaxlp=np.argmax(sampler.chain
    #lnprob=np.array([[blob[-1] for blob in step] for step in sampler.blobs])
    index=np.unravel_index(np.argmax(sampler.lnprobability),sampler.lnprobability.shape)
    MAPp=sampler.chain[index]
    model_MAP=sampler.blobs[index[1]][index[0]][modelidx][1]
    MAPvar=[np.std(dist) for dist in dists]
# TODO: logger
    print 'Maximum log probability: {0:.3g}'.format(sampler.lnprobability[index])
    print 'Maximum Likelihood results:'
    for p,v in zip(MAPp,MAPvar):
        print '{0:.2e} +/- {1:.2e}'.format(p,v)

    return modelx,CI,model_MAP

def plot_fit(sampler,modelidx=0,xlabel=None,ylabel=None,confs=[3,1],**kwargs):
    """
    Plot data with fit confidence regions.
    """

    modelx,CI,model_MAP=calc_fit_CI(sampler,modelidx=modelidx,confs=confs,**kwargs)
    data=sampler.data

    #f,axarr=plt.subplots(4,sharex=True)
    #ax1=axarr[0]
    #ax2=axarr[3]

    plotdata=False
    if modelidx==0:
        plotdata=True

    f=plt.figure()
    if plotdata:
        ax1=plt.subplot2grid((4,1),(0,0),rowspan=3)
        ax2=plt.subplot2grid((4,1),(3,0),sharex=ax1)
        for subp in [ax1,ax2]:
            f.add_subplot(subp)
    else:
        ax1=f.add_subplot(111)

    datacol='r'

    for (ymin,ymax),conf in zip(CI,confs):
        color=np.log(conf)/np.log(20)+0.4
        ax1.fill_between(modelx,ymax,ymin,lw=0.,color='{0}'.format(color),alpha=0.5,zorder=-10)
    ax1.plot(modelx,model_MAP,c='k',lw=3,zorder=-5)

    def plot_ulims(ax,x,y,xerr):
        ax.errorbar(x,y,xerr=xerr,ls='',
                color=datacol,elinewidth=2,capsize=0)
        ax.errorbar(x,0.75*y,yerr=0.25*y,ls='',lolims=True,
                color=datacol,elinewidth=2,capsize=5,zorder=10)

    if plotdata:
        ul=data['ul']
        notul=-ul
        ax1.errorbar(data['ene'][notul],data['flux'][notul],
                yerr=data['dflux'][notul].T, xerr=data['dene'][notul].T,
                zorder=100,marker='o',ls='', elinewidth=2,capsize=0,
                mec='w',mew=0,ms=8,color=datacol)
        if np.sum(ul)>0:
            plot_ulims(ax1,data['ene'][ul],data['flux'][ul],data['dene'][ul])

        modelfunc=interp1d(modelx,model_MAP)
        difference=data['flux'][notul]-modelfunc(data['ene'][notul])
        dflux=np.average(data['dflux'][notul],axis=1)
        ax2.errorbar(data['ene'][notul],difference/dflux,yerr=dflux/dflux, xerr=data['dene'][notul].T,
                zorder=100,marker='o',ls='', elinewidth=2,capsize=0,
                mec='w',mew=0,ms=8,color=datacol)
        ax2.axhline(0,c='k',lw=2,ls='--')

        from matplotlib.ticker import MaxNLocator
        ax2.yaxis.set_major_locator(MaxNLocator(integer='True',prune='upper'))

        ax2.set_ylabel(r'$\Delta\sigma$')


    ax1.set_xscale('log')
    ax1.set_yscale('log')
    if plotdata:
        ax2.set_xscale('log')
        for tl in ax1.get_xticklabels():
            tl.set_visible(False)


    if ylabel!=None:
        ax1.set_ylabel(ylabel)
    if xlabel!=None:
        if plotdata:
            ax2.set_xlabel(xlabel)
        else:
            ax1.set_xlabel(xlabel)

    f.subplots_adjust(hspace=0)

    #f.show()

    return f

## Convenience tools

def generate_energy_edges(ene):
    """Generate an array of energy edges from given energy array to be used as
    abcissa error bar limits when no energy uncertainty or energy band is
    provided

    Parameters
    ----------
    ene : array
        Array of energies

    Returns
    -------
    edge_array : array with shape (2,len(ene))
        Array of energy edge pairs corresponding to each given energy of the
        input array.
    """
    midene=np.sqrt((ene[1:]*ene[:-1]))
    elo,ehi=np.zeros_like(ene),np.zeros_like(ene)
    elo[1:]=ene[1:]-midene
    ehi[:-1]=midene-ene[:-1]
    elo[0]=ehi[0]
    ehi[-1]=elo[-1]
    return np.array(zip(elo,ehi))

def read_data(file,enecol=0,denecol=1,fluxcol=2,dfluxcol=3):
    """
    read data into data dict
    """
    pass

def generate_diagnostic_plots(outname,sampler,modelidxs=None):
    """
    Generate diagnostic plots:

    - A corner plot of sample density in the two dimensional parameter space of
      all parameter pairs of the run: ``outname_corner.png``
    - A plot for each of the chain parameters showing walker progression, final
      sample distribution and several statistical measures of this distribution:
      ``outname_chain_parN.png``
    - A plot for each of the models returned as blobs by the model function. The
      maximum likelihood model is shown, as well as the 1 and 3 sigma confidence
      level contours. The first model will be compared with observational data
      and residuals shown. ``outname_fit_modelN.png``

    Parameters
    ----------
    outname: str
        Name to be used to save diagnostic plot files.

    sampler: emcee.EnsembleSampler instance
        Sampler instance from which chains, blobs and data are read.

    modelidxs: iterable (optional)
        Model numbers to be plotted. Default: All returned in sampler.blobs

    """

    print 'Generating diagnostic plots'

    try:
        ## Corner plot
        f = corner(sampler.flatchain,labels=['norm','par1','par2','par3'])
        f.savefig('{0}_corner.png'.format(outname))
    except NameError:
        print 'triangle.py not installed, corner plot will not be available'

    ## Chains

    for par in range(sampler.chain.shape[-1]):
        f = plot_chain(sampler.chain,par)
        f.savefig('{0}_chain_par{1}.png'.format(outname,par))

    ## Fit

    if modelidxs==None:
        nmodels=len(sampler.blobs[-1][0])
        modelidxs=range(nmodels)

    for modelidx in modelidxs:
        if modelidx==0:
            labels=('Energy','Flux')
        else:
            labels=( None, None)
        f = plot_fit(sampler,xlabel=labels[0],ylabel=labels[1],modelidx=modelidx,last_step=False)
        f.savefig('{0}_fit_model{1}.png'.format(outname,modelidx))
