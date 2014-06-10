# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import astropy.units as u
from astropy.logger import log

__all__ = ["plot_chain","plot_fit","plot_data"]


def plot_chain(sampler,p=None,**kwargs):
    """Generate a diagnostic plot of the sampler chains.

    Parameters
    ----------
    sampler : `emcee.EnsembleSampler`
        Sampler containing the chains to be plotted.
    p : int (optional)
        Index of the parameter to plot. If omitted, all chains are plotted.
    last_step : bool (optional)
        Whether to plot the last step of the chain or the complete chain (default).

    Returns
    -------
    figure : `matplotlib.figure.Figure`
        Figure
    """
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
        log.warn('we need the full chain to plot the traces, not a flatchain!')
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

    log.info('\n {0:-^50}\n'.format(label) + chain_props)
    f.text(0.05,0.45,chain_props,ha='left',va='top')

    return f

def gelman_rubin_statistic(chains):
    """
    Compute Gelman-Rubin statistic for convergence testing of Markov chains.

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
    """Calculate confidence interval.
    """
    from scipy import stats

    model_unit = sampler.blobs[-1][0][modelidx][1].unit
    if last_step:
        model=np.array([m[modelidx][1].value for m in sampler.blobs[-1]])
    else:
        nsteps=len(sampler.blobs)
        model=[]
        for step in sampler.blobs:
            for walkerblob in step:
                model.append(walkerblob[modelidx][1].value)
        model=np.array(model)

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
        CI.append((np.array(ymin)*model_unit,np.array(ymax)*model_unit))

    return modelx,CI

# Define phsyical types
u.def_physical_type(u.erg / u.cm ** 2 / u.s, 'flux')
u.def_physical_type(u.Unit('1/(s cm2 erg)'), 'differential flux')
u.def_physical_type(u.Unit('1/(s erg)'), 'differential power')

def _sed_conversion(energy,flux,sed):
    """
    Manage conversion between differential spectrum and SED
    """

    model_pt = flux.unit.physical_type

    ones = np.ones(energy.shape)

    if sed:
        # SED
        f_unit = u.Unit('erg/s')
        if model_pt == 'power' or model_pt == 'flux':
            sedf = ones
        elif 'differential' in model_pt:
            sedf = (energy**2).to('eV erg')
        else:
            raise u.UnitsError('Model physical type ({0}) is not supported'.format(model_pt),
                    'Supported physical types are: power, flux, differential'
                    ' power, differential flux')

        if 'flux' in model_pt:
            f_unit /= u.cm**2
    elif sed is None:
        # Use original units units
        f_unit = flux.unit
        sedf = ones
    else:
        # Differential spectrum
        f_unit = u.Unit('1/(s TeV)')
        if model_pt == 'power' or model_pt == 'flux':
            # From SED to differential
            sedf = 1/(energy**2).to('eV erg')
        elif 'differential' in model_pt:
            sedf = ones
        else:
            raise u.UnitsError('Model physical type ({0}) is not supported'.format(model_pt),
                    'Supported physical types are: power, flux, differential'
                    ' power, differential flux')

        if 'flux' in model_pt:
            f_unit /= u.cm**2

    return f_unit,sedf


def plot_CI(ax, sampler, modelidx=0,sed=True,confs=[3,1,0.5],e_unit=u.eV,**kwargs):
    """Plot confidence interval.

    Parameters
    ----------
    ax : `matplotlib.Axes`
        Axes to plot on.
    sampler : `emcee.EnsembleSampler`
        Sampler
    modelidx : int, optional
        Model index. Default is 0
    sed : bool, optional
        Whether to plot SED or differential spectrum. If `None`, the units of
        the observed spectrum will be used.
    confs : list, optional
        List of confidence levels (in sigma) to use for generating the confidence intervals. Default is `[3,1,0.5]`
    e_unit : :class:`~astropy.units.Unit` or str parseable to unit
        Unit in which to plot energy axis.
    last_step : bool, optional
        Whether to only use the positions in the final step of the run (True, default) or the whole chain (False).
    """

    envconf=1000000
    #confs+=[envconf,]

    modelx,CI = calc_CI(sampler,modelidx=modelidx,confs=confs,**kwargs)
    modely = sampler.blobs[-1][0][modelidx][1]

    f_unit, sedf = _sed_conversion(modelx,modely,sed)

    for (ymin,ymax),conf in zip(CI,confs):
        if conf==envconf:
            for yy in (ymin,ymax):
                ax.plot(modelx.to(e_unit).value,
                        (yy*sedf).to(f_unit).value,
                        lw=1.,color='0.7',ls=':',zorder=-10)
        else:
            color=np.log(conf)/np.log(20)+0.4
            ax.fill_between(modelx.to(e_unit).value,
                    (ymax*sedf).to(f_unit).value,
                    (ymin*sedf).to(f_unit).value,
                    lw=0.,color='{0}'.format(color),
                    alpha=0.6,zorder=-10)

    #ax.plot(modelx,model_ML,c='k',lw=3,zorder=-5)


def find_ML(sampler,modelidx):
    """
    Find Maximum Likelihood parameters as those in the chain with a highest log
    probability.
    """
    index=np.unravel_index(np.argmax(sampler.lnprobability),sampler.lnprobability.shape)
    MLp=sampler.chain[index]
    model_ML=sampler.blobs[index[1]][index[0]][modelidx][1]
    MLvar=[np.std(dist) for dist in sampler.flatchain.T]
    ML=sampler.lnprobability[index]

    return ML,MLp,MLvar,model_ML

def plot_fit(sampler,modelidx=0,xlabel=None,ylabel=None,confs=[3,1,0.5],
        sed=False,figure=None,residualCI=True,plotdata=None,**kwargs):
    """
    Plot data with fit confidence regions.

    Additional ``kwargs`` are passed to `plotCI`.

    Parameters
    ----------
    sampler : `emcee.EnsembleSampler`
        Sampler with a stored chain.
    modelidx : int, optional
        Model index to plot.
    xlabel : str, optional
        Label for the ``x`` axis of the plot.
    ylabel : str, optional
        Label for the ``y`` axis of the plot.
    sed : bool, optional
        Whether to plot SED or differential spectrum.
    confs : list, optional
        List of confidence levels (in sigma) to use for generating the
        confidence intervals. Default is ``[3,1,0.5]``
    figure : `matplotlib.figure`, optional
        `matplotlib` figure to plot on. If omitted a new one will be generated.
    residualCI : bool, optional
        Whether to plot the confidence interval bands in the residuals subplot.

    """
    import matplotlib.pyplot as plt

    modelx=sampler.blobs[-1][0][modelidx][0]
    ML,MLp,MLvar,model_ML = find_ML(sampler,modelidx)
    infostr='Maximum log probability: {0:.3g}\n'.format(ML)
    infostr+='Maximum Likelihood values:\n'
    for p,v,label in zip(MLp,MLvar,sampler.labels):
        infostr+='{2:>10}: {0:>8.3g} +/- {1:<8.3g}\n'.format(p,v,label)

    log.info(infostr)

    data=sampler.data

    if modelidx==0 and plotdata is None:
        plotdata=True
    elif plotdata is None:
        plotdata=False

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
    e_unit=data['ene'].unit

    if confs is not None:
        plot_CI(ax1,sampler,modelidx,sed=sed,confs=confs,e_unit=e_unit,**kwargs)
    else:
        residualCI=False

    def plot_ulims(ax,x,y,xerr):
        """
        Plot upper limits as arrows with cap at value of upper limit.
        """
        ax.errorbar(x,y,xerr=xerr,ls='',
                color=datacol,elinewidth=2,capsize=0)
        ax.errorbar(x,0.75*y,yerr=0.25*y,ls='',lolims=True,
                color=datacol,elinewidth=2,capsize=5,zorder=10)

    if plotdata:
        # Check that physical types of data and model match
        modely=sampler.blobs[-1][0][modelidx][1]
        model_pt = modely.unit.physical_type
        data_pt = data['flux'].unit.physical_type
        if data_pt != model_pt:
            raise u.UnitsError('Model physical type ({0}) and spectral data physical'
                    ' type ({1}) do not match!'.format(model_pt,data_pt))

        f_unit, sedf = _sed_conversion(data['ene'],data['flux'],sed)

        ul=data['ul']
        notul=-ul

        ax1.errorbar(data['ene'][notul].to(e_unit).value,
                (data['flux'][notul]*sedf[notul]).to(f_unit).value,
                yerr=(data['dflux'][:,notul]*sedf[notul]).to(f_unit).value,
                xerr=(data['dene'][:,notul]).to(e_unit).value,
                zorder=100,marker='o',ls='', elinewidth=2,capsize=0,
                mec='w',mew=0,ms=6,color=datacol)

        if np.any(ul):
            plot_ulims(ax1,data['ene'][ul].to(e_unit).value,
                    (data['flux'][ul]*sedf[ul]).to(f_unit).value,
                    (data['dene'][:,ul]).to(e_unit).value)

        if len(model_ML)!=len(data['ene']):
            from scipy.interpolate import interp1d
            modelfunc=interp1d(modelx.to(e_unit).value,model_ML.value)
            difference=data['flux'][notul].value-modelfunc(data['ene'][notul])
            difference *= data['flux'].unit
        else:
            difference=data['flux'][notul]-model_ML[notul]

        dflux=np.mean(data['dflux'][:,notul],axis=0)
        ax2.errorbar(data['ene'][notul].to(e_unit).value,
                (difference/dflux).decompose().value,
                yerr=(dflux/dflux).decompose().value,
                xerr=data['dene'][:,notul].to(e_unit).value,
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
                    ax2.fill_between(modelx[notul].to(e_unit).value,
                            ((ymax[notul]-model_ML[notul])/dflux).decompose().value,
                            ((ymin[notul]-model_ML[notul])/dflux).decompose().value,
                            lw=0., color='{0}'.format(color), alpha=0.6,zorder=-10)
            #ax.plot(modelx,model_ML,c='k',lw=3,zorder=-5)




    ax1.set_xscale('log')
    ax1.set_yscale('log')
    if plotdata:
        ax2.set_xscale('log')
        for tl in ax1.get_xticklabels():
            tl.set_visible(False)
        ax1.set_xlim(10**np.floor(np.log10(np.min(data['ene']-data['dene'][0]).value)),
                10**np.ceil(np.log10(np.max(data['ene']+data['dene'][1]).value)))
    else:
        ndecades=5
        # restrict y axis to ndecades to avoid autoscaling deep exponentials
        xmin,xmax,ymin,ymax=ax1.axis()
        ymin=max(ymin,ymax/10**ndecades)
        ax1.set_ylim(bottom=ymin)
        # scale x axis to largest model_ML x point within ndecades decades of
        # maximum
        modelx = sampler.blobs[-1][0][modelidx][0]
        modely = sampler.blobs[-1][0][modelidx][1]
        f_unit, sedf = _sed_conversion(modelx,modely,sed)
        hi=np.where((model_ML*sedf).to(f_unit).value>ymin)
        xmax=np.max(modelx[hi])
        ax1.set_xlim(right=10**np.ceil(np.log10(xmax.to(e_unit).value)))


    ax1.text(0.05,0.05,infostr,ha='left',va='bottom',transform=ax1.transAxes,family='monospace')

    if ylabel is None:
        if sed:
            ax1.set_ylabel(r'$E^2\mathrm{{d}}N/\mathrm{{d}}E$ [{{{0}}}]'.format(u.Unit(f_unit)))
        else:
            ax1.set_ylabel(r'$\mathrm{{d}}N/\mathrm{{d}}E$ [{{{0}}}]'.format(u.Unit(f_unit)))
    else:
        ax1.set_ylabel(ylabel)

    if plotdata:
        xlaxis = ax2
    else:
        xlaxis = ax1

    if xlabel is None:
        xlaxis.set_xlabel('Energy [{0}]'.format(e_unit))
    else:
        xlaxis.set_xlabel(xlabel)

    f.subplots_adjust(hspace=0)

    return f


def plot_data(sampler,xlabel=None,ylabel=None,
        sed=False,figure=None,**kwargs):
    """
    Plot spectral data.

    Additional ``kwargs`` are passed to `plot_fit`, except ``confs`` and
    ``plotdata``.

    Parameters
    ----------
    sampler : `emcee.EnsembleSampler`
        Sampler with a stored chain.
    xlabel : str, optional
        Label for the ``x`` axis of the plot.
    ylabel : str, optional
        Label for the ``y`` axis of the plot.
    sed : bool, optional
        Whether to plot SED or differential spectrum.
    figure : `matplotlib.figure`, optional
        `matplotlib` figure to plot on. If omitted a new one will be generated.

    """
    for par in ['confs','plotdata']:
        if par in kwargs:
            kwargs.pop(par)

    f = plot_fit(sampler,confs=None,xlabel=xlabel,ylabel=ylabel,
        sed=sed,figure=figure,plotdata=True,**kwargs)

    return f
