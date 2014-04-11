# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np

__all__ = ["plot_chain","plot_fit","plot_CI"]


def plot_chain(sampler,p=None,**kwargs):
    import matplotlib.pyplot as plt
    if p==None:
        npars=sampler.chain.shape[-1]
        for pp,label in zip(range(npars),sampler.labels):
            _plot_chain_func(sampler.chain,pp,label,**kwargs)
        fig = None
    else:
        fig = _plot_chain_func(sampler.chain,p,sampler.labels[p],**kwargs)

    return fig


def _plot_chain_func(chain,p,label,last_step=False):
    import matplotlib.pyplot as plt
    from scipy import stats
    if len(chain.shape)>2:
        traces=chain[:,:,p]
        if last_step==True:
            #keep only last step
            dist=traces[:,-1]
        else:
            #convert chain to flatchain
            dist=traces.flatten()
    else:
        print('we need the chain to plot the traces, not a flatchain!')
        return None

    nwalkers=traces.shape[0]
    nsteps=traces.shape[1]

    logplot=False
    if np.abs(dist.max()/dist.min())>10.:
        logplot=True

    f=plt.figure()

    ax1=f.add_subplot(221)
    ax2=f.add_subplot(122)

# plot five percent of the traces darker

    colors=np.where(np.arange(nwalkers)/float(nwalkers)>0.95,'#550000','0.5')

    ax1.set_rasterization_zorder(1)
    for t,c in zip(traces,colors): #range(nwalkers):
        ax1.plot(t,c=c,lw=1,alpha=0.9,zorder=0)
    ax1.set_xlabel('step number')
    ax1.set_ylabel(label)
    ax1.set_title('Walker traces')
    if logplot:
        ax1.set_yscale('log')

    #nbins=25 if last_step else 100
    nbins=min(max(25,int(len(dist)/100.)),100)
    xlabel=label
    if logplot:
        dist=np.log10(dist)
        xlabel='log10('+xlabel+')'
    n,x,patch=ax2.hist(dist,nbins,histtype='stepfilled',color='#CC0000',lw=0,normed=1)
    kde=stats.kde.gaussian_kde(dist)
    ax2.plot(x,kde(x),c='k',label='KDE')
    #for m,ls,lab in zip([np.mean(dist),np.median(dist)],('--','-.'),('mean: {0:.4g}','median: {0:.4g}')):
        #ax2.axvline(m,ls=ls,c='k',alpha=0.5,lw=2,label=lab.format(m))
    ns=len(dist)
    quant=[0.01,0.1,0.16,0.5,0.84,0.9,0.99]
    sdist=np.sort(dist)
    xquant=[sdist[int(q*ns)] for q in quant]
    ax2.axvline(xquant[quant.index(0.5)],ls='--',c='k',alpha=0.5,lw=2,label='50% quantile')
    ax2.axvspan(xquant[quant.index(0.16)],xquant[quant.index(0.84)],color='0.5',alpha=0.25,label='68% CI')
    #ax2.legend()
    ax2.set_xlabel(xlabel)
    ax2.set_title('posterior distribution')
    ax2.set_ylim(top=n.max()*1.05)

    # Print distribution parameters on lower-left

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

    try:
        import acor
        acort=acor.acor(traces)[0]
    except:
        acort=np.nan

    if last_step:
        clen='last ensemble'
    else:
        clen='whole chain'

    quantiles=dict(zip(quant,xquant))

    chain_props='Walkers: {0} \nSteps in chain: {1} \n'.format(nwalkers,nsteps) + \
            'Autocorrelation time: {0:.1f}'.format(acort) + '\n' +\
            'Gelman-Rubin statistic: {0:.3f}'.format(gelman_rubin_statistic(traces)) + '\n' +\
            'Distribution properties for the {clen}:\n \
    - mode: {mode:.3g} \n \
    - mean: {mean:.3g} \n \
    - median: {median:.3g} \n \
    - std: {std:.3g} \n \
    - 68% CI: ({quant16:.3g},{quant84:.3g})\n \
    - 99% CI: ({quant01:.3g},{quant99:.3g})\n \
    - mean +/- std CI: ({meanstd[0]:.3g},{meanstd[1]:.3g})\n'.format(
                mean=mean,median=median,std=std,
                quant01=quantiles[0.01],
                quant16=quantiles[0.16],
                quant84=quantiles[0.84],
                quant99=quantiles[0.99],
                meanstd=(mean-std,mean+std),clen=clen,mode=mode,)

    print('\n {0:-^50}\n'.format(label) + chain_props)
    f.text(0.05,0.45,chain_props,ha='left',va='top')

    #f.tight_layout()

    #f.show()

    return f

def gelman_rubin_statistic(chains):
    """
    Compute Gelman-Rubint statistic for convergence testing of Markov chains.

    Gelman & Rubin (1992), Statistical Science 7, pp. 457-511
    """
    # normalize it so it doesn't do strange things with very low values
    chains=chains.copy()/np.average(chains)
    eta=float(chains.shape[1])
    m=float(chains.shape[0])
    avgchain=np.average(chains,axis=1)

    W=np.sum(np.sum((chains.T-avgchain)**2,axis=1))/m/(eta-1)
    B=eta/(m-1)*np.sum((avgchain-np.mean(chains)))
    var=(1-1/eta)*W+1/eta*B

    return np.sqrt(var/W)



def calc_CI(sampler,modelidx=0,confs=[3,1],last_step=True):
    from scipy import stats

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

    nwalkers=len(model)-1
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

    return modelx,CI

def plot_CI(ax, sampler, modelidx=0,converttosed=False,confs=[3,1,0.5],**kwargs):

    envconf=1000000
    #confs+=[envconf,]

    modelx,CI=calc_CI(sampler,modelidx=modelidx,confs=confs,**kwargs)

    if converttosed:
# try to find whether modelx is eV or TeV
        toerg=1.6021765
        if np.max(modelx)>1e8:
            toerg/=1e12
        sedf=modelx**2*toerg # TeV to erg
    else:
        sedf=np.ones_like(modelx)

    for (ymin,ymax),conf in zip(CI,confs):
        if conf==envconf:
            for yy in (ymin,ymax):
                ax.plot(modelx,yy*sedf,lw=1.,color='0.7',ls=':',zorder=-10)
        else:
            color=np.log(conf)/np.log(20)+0.4
            ax.fill_between(modelx,ymax*sedf,ymin*sedf,lw=0.,color='{0}'.format(color),alpha=0.6,zorder=-10)
    #ax.plot(modelx,model_ML,c='k',lw=3,zorder=-5)


def find_ML(sampler,modelidx):
    """
    Find Maximum Likelihood parameters as those in the chain with a highest log
    probability
    """
    index=np.unravel_index(np.argmax(sampler.lnprobability),sampler.lnprobability.shape)
    MLp=sampler.chain[index]
    model_ML=sampler.blobs[index[1]][index[0]][modelidx][1]
    MLvar=[np.std(dist) for dist in sampler.flatchain.T]
    ML=sampler.lnprobability[index]

    return ML,MLp,MLvar,model_ML

def plot_fit(sampler,modelidx=0,xlabel=None,ylabel=None,confs=[3,1,0.5],
        converttosed=False,figure=None,residualCI=True,plotdata=False,**kwargs):
    """
    Plot data with fit confidence regions.
    """
    import matplotlib.pyplot as plt

    modelx=sampler.blobs[-1][0][modelidx][0]
    ML,MLp,MLvar,model_ML = find_ML(sampler,modelidx)
    infostr='Maximum log probability: {0:.3g}\n'.format(ML)
    infostr+='Maximum Likelihood values:\n'
    for p,v,label in zip(MLp,MLvar,sampler.labels):
        infostr+='{2:>10}: {0:>8.3g} +/- {1:<8.3g}\n'.format(p,v,label)

    # TODO: logger
    print(infostr)
    #infostr=''

    data=sampler.data

    if modelidx==0:
        plotdata=True

    if figure==None:
        f=plt.figure()
    else:
        f=figure

    if plotdata:
        ax1=plt.subplot2grid((4,1),(0,0),rowspan=3)
        ax2=plt.subplot2grid((4,1),(3,0),sharex=ax1)
        for subp in [ax1,ax2]:
            f.add_subplot(subp)
    else:
        ax1=f.add_subplot(111)

    datacol='r'

    if confs is not None:
        plot_CI(ax1,sampler,modelidx,converttosed=converttosed,confs=confs,**kwargs)
    else:
        residualCI=False

    def plot_ulims(ax,x,y,xerr):
        ax.errorbar(x,y,xerr=xerr,ls='',
                color=datacol,elinewidth=2,capsize=0)
        ax.errorbar(x,0.75*y,yerr=0.25*y,ls='',lolims=True,
                color=datacol,elinewidth=2,capsize=5,zorder=10)

    if plotdata:
        ul=data['ul']
        notul=-ul
        if converttosed:
            toerg=1.6021765
            if np.max(modelx)>1e8:
                toerg/=1e12
            sedf=data['ene']**2*toerg
        else:
            sedf=np.ones_like(data['ene'])

        ax1.errorbar(data['ene'][notul],data['flux'][notul]*sedf[notul],
                yerr=data['dflux'][notul].T*sedf[notul], xerr=data['dene'][notul].T,
                zorder=100,marker='o',ls='', elinewidth=2,capsize=0,
                mec='w',mew=0,ms=6,color=datacol)

        if np.any(ul):
            plot_ulims(ax1,data['ene'][ul],data['flux'][ul]*sedf[ul],data['dene'][ul].T)

        if len(model_ML)!=len(data['ene']):
            from scipy.interpolate import interp1d
            modelfunc=interp1d(modelx,model_ML)
            difference=data['flux'][notul]-modelfunc(data['ene'][notul])
        else:
            difference=data['flux'][notul]-model_ML[notul]

        dflux=np.average(data['dflux'][notul],axis=1)
        ax2.errorbar(data['ene'][notul],difference/dflux,yerr=dflux/dflux, xerr=data['dene'][notul].T,
                zorder=100,marker='o',ls='', elinewidth=2,capsize=0,
                mec='w',mew=0,ms=6,color=datacol)
        ax2.axhline(0,c='k',lw=2,ls='--')

        from matplotlib.ticker import MaxNLocator
        ax2.yaxis.set_major_locator(MaxNLocator(integer='True',prune='upper'))

        ax2.set_ylabel(r'$\Delta\sigma$')

        if len(model_ML)==len(data['ene']) and residualCI:
            modelx,CI=calc_CI(sampler,modelidx=modelidx,confs=confs,**kwargs)

            for (ymin,ymax),conf in zip(CI,confs):
                if conf<100:
                    color=np.log(conf)/np.log(20)+0.4
                    ax2.fill_between(modelx[notul],(np.array(ymax)[notul]-model_ML[notul])/dflux,
                            (np.array(ymin)[notul]-model_ML[notul])/dflux,lw=0.,
                            color='{0}'.format(color), alpha=0.6,zorder=-10)
            #ax.plot(modelx,model_ML,c='k',lw=3,zorder=-5)




    ax1.set_xscale('log')
    ax1.set_yscale('log')
    if plotdata:
        ax2.set_xscale('log')
        for tl in ax1.get_xticklabels():
            tl.set_visible(False)
    else:
        ndecades=5
        # restrict y axis to ndecades to avoid autoscaling deep exponentials
        xmin,xmax,ymin,ymax=ax1.axis()
        ymin=max(ymin,ymax/10**ndecades)
        ax1.set_ylim(bottom=ymin)
        # scale x axis to largest model_ML x point within ndecades decades of
        # maximum
        hi=np.where(model_ML>ymin)
        xmax=np.max(modelx[hi])
        ax1.set_xlim(right=10**np.ceil(np.log10(xmax)))



    ax1.text(0.05,0.05,infostr,ha='left',va='bottom',transform=ax1.transAxes,family='monospace')

    if ylabel!=None:
        ax1.set_ylabel(ylabel)
    if xlabel!=None:
        if plotdata:
            ax2.set_xlabel(xlabel)
        else:
            ax1.set_xlabel(xlabel)

    f.subplots_adjust(hspace=0)

    return f

