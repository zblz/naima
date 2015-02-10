# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import astropy.units as u
from astropy.extern import six
from astropy import log
from astropy import table

from .utils import sed_conversion, validate_data_table

__all__ = ["plot_chain", "plot_fit", "plot_data", "plot_blob"]


def plot_chain(sampler, p=None, **kwargs):
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
    if p == None:
        npars = sampler.chain.shape[-1]
        for pp, label in zip(six.moves.range(npars), sampler.labels):
            _plot_chain_func(sampler, pp, **kwargs)
        fig = None
    else:
        fig = _plot_chain_func(sampler, p, **kwargs)

    return fig

def _latex_float(f, format=".3g"):
    """ http://stackoverflow.com/a/13490601
    """
    float_str = "{{0:{0}}}".format(format).format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0}\times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

def _plot_chain_func(sampler, p, last_step=False):
    chain = sampler.chain
    label = sampler.labels[p]

    import matplotlib.pyplot as plt
    # Plot everything in serif to match math exponents
    plt.rc('font', family='serif')

    from scipy import stats
    if len(chain.shape) > 2:
        traces = chain[:,:, p]
        if last_step == True:
            # keep only last step
            dist = traces[:, -1]
        else:
            # convert chain to flatchain
            dist = traces.flatten()
    else:
        log.warning('we need the full chain to plot the traces, not a flatchain!')
        return None

    nwalkers = traces.shape[0]
    nsteps = traces.shape[1]

    f = plt.figure()

    ax1 = f.add_subplot(221)
    ax2 = f.add_subplot(122)

    f.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9)

# plot five percent of the traces darker

    colors = np.where(np.arange(nwalkers)/float(nwalkers) > 0.95, '#550000', '0.5')

    ax1.set_rasterization_zorder(1)
    for t, c in zip(traces, colors):  # range(nwalkers):
        ax1.plot(t, c=c, lw=1, alpha=0.9, zorder=0)
    ax1.set_xlabel('step number')
    #[l.set_rotation(45) for l in ax1.get_yticklabels()]
    ax1.set_ylabel(label)
    ax1.yaxis.set_label_coords(-0.15, 0.5)
    ax1.set_title('Walker traces')

    # nbins=25 if last_step else 100
    nbins = min(max(25, int(len(dist)/100.)), 100)
    xlabel = label
    n, x, patch = ax2.hist(dist, nbins, histtype='stepfilled', color='#CC0000', lw=0, normed=1)
    kde = stats.kde.gaussian_kde(dist)
    ax2.plot(x, kde(x), c='k', label='KDE')
    # for m,ls,lab in zip([np.mean(dist),np.median(dist)],('--','-.'),('mean: {0:.4g}','median: {0:.4g}')):
        # ax2.axvline(m,ls=ls,c='k',alpha=0.5,lw=2,label=lab.format(m))
    quant = [16, 50, 84]
    xquant = np.percentile(dist, quant)
    quantiles = dict(six.moves.zip(quant, xquant))

    ax2.axvline(quantiles[50], ls='--', c='k', alpha=0.5, lw=2,
                label='50% quantile')
    ax2.axvspan(quantiles[16], quantiles[84], color='0.5', alpha=0.25,
                label='68% CI')
    # ax2.legend()
    [l.set_rotation(45) for l in ax2.get_xticklabels()]
    #[l.set_rotation(45) for l in ax2.get_yticklabels()]
    ax2.set_xlabel(xlabel)
    ax2.xaxis.set_label_coords(0.5, -0.1)
    ax2.set_title('posterior distribution')
    ax2.set_ylim(top=n.max() * 1.05)

    # Print distribution parameters on lower-left

    mean, median, std = np.mean(dist), np.median(dist), np.std(dist)
    xmode = np.linspace(mean-np.sqrt(3)*std, mean+np.sqrt(3)*std, 100)
    mode = xmode[np.argmax(kde(xmode))]
    median = np.median(dist)

    try:
        # EnsembleSample.get_autocorr_time was only added in the
        # recently released emcee 2.1.0 (2014-05-22), so make it optional
        autocorr = sampler.get_autocorr_time(window=chain.shape[1]/4.)[p]
        autocorr_message = '{0:.1f}'.format(autocorr)
    except AttributeError:
        autocorr_message = 'Not available. Update to emcee 2.1 or later.'

    if last_step:
        clen = 'last ensemble'
    else:
        clen = 'whole chain'

    maxlen = np.max([len(ilabel) for ilabel in sampler.labels])
    vartemplate = '{{2:>{0}}}: {{0:>8.3g}} +/- {{1:<8.3g}}\n'.format(maxlen)

    chain_props = 'Walkers: {0} \nSteps in chain: {1} \n'.format(nwalkers, nsteps) + \
            'Autocorrelation time: {0}\n'.format(autocorr_message) +\
            'Mean acceptance fraction: {0:.3f}\n'.format(np.mean(sampler.acceptance_fraction)) +\
            'Distribution properties for the {clen}:\n \
    - median: ${median}$ \n \
    - std: ${std}$ \n' .format(median=_latex_float(quantiles[50]), std=_latex_float(std), clen=clen) +\
'     - Median with uncertainties based on \n \
      the 16th and 84th percentiles ($\sim$1$\sigma$):\n'

    info_line = ' '*10 + '{label} = ${{{median}}}^{{+{uncs[1]}}}_{{-{uncs[0]}}}$'.format(
            label=label, median=_latex_float(quantiles[50]),
            uncs=(_latex_float(quantiles[50] - quantiles[16]),
                      _latex_float(quantiles[84] - quantiles[50])))

    chain_props += info_line


    if 'log10(' in label or 'log(' in label:
        nlabel = label.split('(')[-1].split(')')[0]
        ltype = label.split('(')[0]
        if ltype == 'log10':
            new_dist = 10**dist
        elif ltype == 'log':
            new_dist = np.exp(dist)

        quant = [16, 50, 84]
        quantiles = dict(six.moves.zip(quant, np.percentile(new_dist, quant)))

        label_template = '\n'+' '*10+'{{label:>{0}}}'.format(len(label))

        new_line = label_template.format(label=nlabel)
        new_line += ' = ${{{median}}}^{{+{uncs[1]}}}_{{-{uncs[0]}}}$'.format(
                    label=nlabel, median=_latex_float(quantiles[50]),
                    uncs=(_latex_float(quantiles[50] - quantiles[16]),
                          _latex_float(quantiles[84] - quantiles[50])))

        chain_props += new_line
        info_line += new_line

    log.info('{0:-^50}\n'.format(label) + info_line)
    f.text(0.05, 0.45, chain_props, ha='left', va='top')

    return f

def _process_blob(sampler, modelidx,last_step=True):
    """
    Process binary blob in sampler. If blob in position modelidx is:

    - a Quantity array of len(blob[i])=len(data['energy']: use blob as model, data['energy'] as modelx
    - a tuple: use first item as modelx, second as model
    - a Quantity scalar: return array of scalars
    """

    blob0 = sampler.blobs[-1][0][modelidx]
    if isinstance(blob0, u.Quantity):
        if blob0.size == sampler.data['energy'].size:
            # Energy array for blob is not provided, use data['energy']
            modelx = sampler.data['energy']
        elif blob0.size == 1:
            modelx = None

        if last_step:
            model = u.Quantity([m[modelidx] for m in sampler.blobs[-1]])
        else:
            nsteps = len(sampler.blobs)
            model = []
            for step in sampler.blobs:
                for walkerblob in step:
                    model.append(walkerblob[modelidx])
            model = u.Quantity(model)
    elif np.isscalar(blob0):
        modelx = None

        if last_step:
            model = u.Quantity([m[modelidx] for m in sampler.blobs[-1]])
        else:
            nsteps = len(sampler.blobs)
            model = []
            for step in sampler.blobs:
                for walkerblob in step:
                    model.append(walkerblob[modelidx])
            model = u.Quantity(model)
    elif (isinstance(blob0, list) or isinstance(blob0, tuple)):
        if (len(blob0) == 2 and isinstance(blob0[0], u.Quantity)
            and isinstance(blob0[1], u.Quantity)):
            # Energy array for model is item 0 in blob, model flux is item 1
            modelx = blob0[0]

            if last_step:
                model = u.Quantity([m[modelidx][1] for m in sampler.blobs[-1]])
            else:
                nsteps = len(sampler.blobs)
                model = []
                for step in sampler.blobs:
                    for walkerblob in step:
                        model.append(walkerblob[modelidx][1])
                model = u.Quantity(model)
        else:
            raise TypeError('Model {0} has wrong blob format'.format(modelidx))

    else:
        raise TypeError('Model {0} has wrong blob format'.format(modelidx))

    return modelx, model


def _get_model_pt(sampler, modelidx):
    blob0 = sampler.blobs[-1][0][modelidx]
    if isinstance(blob0, u.Quantity):
        pt = blob0.unit.physical_type
    elif len(blob0) == 2:
        pt = blob0[1].unit.physical_type
    else:
        raise TypeError('Model {0} has wrong blob format'.format(modelidx))

    return pt

def calc_CI(sampler, modelidx=0,confs=[3, 1],last_step=True):
    """Calculate confidence interval.
    """
    from scipy import stats

    modelx, model = _process_blob(sampler, modelidx, last_step=last_step)

    nwalkers = len(model)-1
    CI = []
    for conf in confs:
        fmin = stats.norm.cdf(-conf)
        fmax = stats.norm.cdf(conf)
        ymin, ymax = [], []
        for fr, y in ((fmin, ymin), (fmax, ymax)):
            nf = int((fr*nwalkers))
            for i, x in enumerate(modelx):
                ysort = np.sort(model[:, i])
                y.append(ysort[nf])

        # create an array from lists ymin and ymax preserving units
        CI.append((u.Quantity(ymin), u.Quantity(ymax)))

    return modelx, CI

def plot_CI(ax, sampler, modelidx=0, sed=True, confs=[3, 1, 0.5], e_unit=u.eV,
        label=None, **kwargs):
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

    modelx, CI = calc_CI(sampler, modelidx=modelidx,confs=confs,**kwargs)
    # pick first confidence interval curve for units
    f_unit, sedf = sed_conversion(modelx, CI[0][0].unit, sed)

    for (ymin, ymax), conf in zip(CI, confs):
        color = np.log(conf)/np.log(20)+0.4
        ax.fill_between(modelx.to(e_unit).value,
                (ymax * sedf).to(f_unit).value,
                (ymin * sedf).to(f_unit).value,
                lw=0., color='{0}'.format(color),
                alpha=0.6, zorder=-10)

    ML, MLp, MLerr, ML_model = find_ML(sampler, modelidx)
    ax.plot(ML_model[0].to(e_unit).value, (ML_model[1] * sedf).to(f_unit).value,
            color='r', lw=1.5, alpha=0.8)

    if label is not None:
        ax.set_ylabel('{0} [{1}]'.format(label,_latex_unit(f_unit)))

def plot_samples(ax, sampler, modelidx=0, sed=True, n_samples=100, e_unit=u.eV,
        last_step=False, label=None):
    """Plot a number of samples from the sampler chain.

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
    n_samples : int, optional
        Number of samples to plot. Default is 100.
    e_unit : :class:`~astropy.units.Unit` or str parseable to unit
        Unit in which to plot energy axis.
    last_step : bool, optional
        Whether to only use the positions in the final step of the run (True, default) or the whole chain (False).
    """

    modelx, model = _process_blob(sampler, modelidx, last_step=last_step)
    # pick first confidence interval curve for units
    f_unit, sedf = sed_conversion(modelx, model[0].unit, sed)

    for my in model[np.random.randint(len(model), size=n_samples)]:
        ax.plot(modelx.to(e_unit).value, (my * sedf).to(f_unit).value,
                color='k', alpha=0.1, lw=1)

    ML, MLp, MLerr, ML_model = find_ML(sampler, modelidx)
    ax.plot(ML_model[0].to(e_unit).value, (ML_model[1] * sedf).to(f_unit).value,
            color='r', lw=1.5, alpha=0.8)

    if label is not None:
        ax.set_ylabel('{0} [{1}]'.format(label,_latex_unit(f_unit)))

def find_ML(sampler, modelidx):
    """
    Find Maximum Likelihood parameters as those in the chain with a highest log
    probability.
    """
    index = np.unravel_index(np.argmax(sampler.lnprobability), sampler.lnprobability.shape)
    MLp = sampler.chain[index]
    blob = sampler.blobs[index[1]][index[0]][modelidx]
    if isinstance(blob, u.Quantity):
        modelx = sampler.data['energy'].copy()
        model_ML = blob.copy()
    elif len(blob) == 2:
        modelx = blob[0].copy()
        model_ML = blob[1].copy()
    else:
        raise TypeError('Model {0} has wrong blob format'.format(modelidx))

    MLerr = []
    for dist in sampler.flatchain.T:
        hilo = np.percentile(dist, [16., 84.])
        MLerr.append((hilo[1]-hilo[0])/2.)
    ML = sampler.lnprobability[index]

    return ML, MLp, MLerr, (modelx, model_ML)

def _latex_unit(unit):
    """ Hack to get a single line latex representation of a unit

        Will be obsolete with format='latex_inline' in astropy 1.0
    """
    l = unit.to_string('cds').split('.')
    out = ''
    for uni in l:
        try:
            int(uni[-1])
            if uni[-2] == '-':
                out += ' {0}$^{{{1}}}$'.format(uni[:-2], uni[-2:])
            else:
                out += ' {0}$^{1}$'.format(uni[:-1], uni[-1:])
        except ValueError:
            out += ' ' + uni

    return out[1:]

def plot_blob(sampler, blobidx=0, label=None, last_step=False, figure=None, **kwargs):
    """
    Plot a metadata blob as a fit to spectral data or value distribution

    Additional ``kwargs`` are passed to `plot_fit`.

    Parameters
    ----------
    sampler : `emcee.EnsembleSampler`
        Sampler with a stored chain.
    blobidx : int, optional
        Metadata blob index to plot.
    label : str, optional
        Label for the value distribution. Labels for the fit plot can be passed
        as ``xlabel`` and ``ylabel`` and will be passed to `plot_fit`.

    Returns
    -------
    figure : `matplotlib.pyplot.Figure`
        `matplotlib` figure instance containing the plot.
    """

    modelx, model = _process_blob(sampler, blobidx, last_step)

    if modelx is None:
        # Blob is scalar, plot distribution
        f = plot_distribution(model, label, figure=figure)
    else:
        f = plot_fit(sampler, modelidx=blobidx, last_step=last_step,
                label=label, figure=figure,**kwargs)

    return f

def plot_fit(sampler, modelidx=0, label=None, xlabel=None, ylabel=None,
        n_samples=100, confs=None, sed=True, figure=None, residualCI=True,
        plotdata=None, e_unit=None, data_color='r', **kwargs):
    """
    Plot data with fit confidence regions.

    Additional ``kwargs`` are passed to `plot_CI`.

    Parameters
    ----------
    sampler : `emcee.EnsembleSampler`
        Sampler with a stored chain.
    modelidx : int, optional
        Model index to plot.
    label : str, optional
        Label for the title of the plot.
    xlabel : str, optional
        Label for the ``x`` axis of the plot.
    ylabel : str, optional
        Label for the ``y`` axis of the plot.
    sed : bool, optional
        Whether to plot SED or differential spectrum.
    n_samples : int, optional
        If not ``None``, number of sample models to plot. If ``None``,
        confidence bands will be plotted instead of samples. Default is 100.
    confs : list, optional
        List of confidence levels (in sigma) to use for generating the
        confidence intervals. Default is to plot sample models instead of
        confidence bands.
    figure : `matplotlib.figure.Figure`, optional
        `matplotlib` figure to plot on. If omitted a new one will be generated.
    residualCI : bool, optional
        Whether to plot the confidence interval bands in the residuals subplot.
    plotdata : bool, optional
        Wheter to plot data on top of model confidence intervals. Default is
        True if the physical types of the data and the model match.
    e_unit : `~astropy.units.Unit`
        Units for the energy axis of the plot. The default is to use the units
        of the energy array of the observed data.
    data_color : str
        Matplotlib color for the data points.

    """
    import matplotlib.pyplot as plt

    # Plot everything in serif to match math exponents
    plt.rc('font', family='serif')

    if confs is None and n_samples is None:
        # We actually only want to plot the data, so let's go there
        return plot_data(sampler.data, xlabel=xlabel, ylabel=ylabel, sed=sed, figure=figure,
                e_unit=e_unit, data_color=data_color, **kwargs)

    ML, MLp, MLerr, model_ML = find_ML(sampler, modelidx)
    infostr = 'Maximum log probability: {0:.3g}\n'.format(ML)
    infostr += 'Maximum Likelihood values:\n'
    maxlen = np.max([len(ilabel) for ilabel in sampler.labels])
    vartemplate = '{{2:>{0}}}: {{0:>8.3g}} +/- {{1:<8.3g}}\n'.format(maxlen)
    for p, v, ilabel in zip(MLp, MLerr, sampler.labels):
        infostr += vartemplate.format(p, v, ilabel)

    # log.info(infostr)

    data = sampler.data
    ul = data['ul']
    notul = -ul

    plotresiduals = False
    if modelidx == 0 and plotdata is None:
        plotdata = True
        if confs is not None or n_samples is not None:
            plotresiduals = True
    elif plotdata is None:
        plotdata = False

    if plotdata:
        # Check that physical types of data and model match
        model_pt = _get_model_pt(sampler, modelidx)
        data_pt = data['flux'].unit.physical_type
        if data_pt != model_pt:
            log.info('Model physical type ({0}) and spectral data physical'
                    ' type ({1}) do not match for blob {2}! Not plotting data.'.format(model_pt, data_pt, modelidx))
            plotdata = False

    if figure == None:
        f = plt.figure()
    else:
        f = figure

    if plotdata and plotresiduals:
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
        ax2 = plt.subplot2grid((4, 1), (3, 0), sharex=ax1)
        for subp in [ax1, ax2]:
            f.add_subplot(subp)
    else:
        ax1 = f.add_subplot(111)

    if e_unit is None:
        e_unit = data['energy'].unit

    if confs is not None:
        plot_CI(ax1, sampler,modelidx,sed=sed,confs=confs,e_unit=e_unit,
                label=label, **kwargs)
    elif n_samples is not None:
        plot_samples(ax1, sampler, modelidx, sed=sed, n_samples=n_samples,
                e_unit=e_unit, label=label)
    else:
        residualCI = False

    if plotdata:

        _plot_data_to_ax(data, ax1, e_unit=e_unit, sed=sed,
                data_color=data_color, ylabel=ylabel)

        if plotresiduals:
            if len(model_ML) != len(data['energy']):
                from scipy.interpolate import interp1d
                modelfunc = interp1d(model_ML[0].to(e_unit).value, model_ML[1].value)
                difference = data['flux'][notul].value-modelfunc(data['energy'][notul])
                difference *= data['flux'].unit
            else:
                difference = data['flux'][notul]-model_ML[1][notul]

            dflux = np.mean(data['dflux'][:, notul], axis=0)
            ax2.errorbar(data['energy'][notul].to(e_unit).value,
                    (difference / dflux).decompose().value,
                    yerr=(dflux / dflux).decompose().value,
                    xerr=data['dene'][:, notul].to(e_unit).value,
                    zorder=100, marker='o', ls='', elinewidth=2, capsize=0,
                    mec='w', mew=0, ms=6, color=data_color)
            ax2.axhline(0, c='k', lw=2, ls='--')

            from matplotlib.ticker import MaxNLocator
            ax2.yaxis.set_major_locator(MaxNLocator(integer='True', prune='upper'))

            ax2.set_ylabel(r'$\Delta\sigma$')

            if len(model_ML) == len(data['energy']) and residualCI:
                modelx, CI = calc_CI(sampler, modelidx=modelidx,
                                     confs=confs, **kwargs)

                for (ymin, ymax), conf in zip(CI, confs):
                    if conf < 100:
                        color = np.log(conf)/np.log(20)+0.4
                        ax2.fill_between(modelx[notul].to(e_unit).value,
                                ((ymax[notul]-model_ML[1][notul])
                                 / dflux).decompose().value,
                                ((ymin[notul]-model_ML[1][notul])
                                 / dflux).decompose().value,
                                lw=0., color='{0}'.format(color), alpha=0.6, zorder=-10)
                # ax.plot(modelx,model_ML,c='k',lw=3,zorder=-5)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    if plotdata and plotresiduals:
        ax2.set_xscale('log')
        for tl in ax1.get_xticklabels():
            tl.set_visible(False)
    else:
        if sed:
            ndecades = 10
        else:
            ndecades = 20
        # restrict y axis to ndecades to avoid autoscaling deep exponentials
        xmin, xmax, ymin, ymax = ax1.axis()
        ymin = max(ymin, ymax/10**ndecades)
        ax1.set_ylim(bottom=ymin)
        # scale x axis to largest model_ML x point within ndecades decades of
        # maximum
        f_unit, sedf = sed_conversion(model_ML[0], model_ML[1].unit, sed)
        hi = np.where((model_ML[1]*sedf).to(f_unit).value > ymin)
        xmax = np.max(model_ML[0][hi])
        ax1.set_xlim(right=10 ** np.ceil(np.log10(xmax.to(e_unit).value)))

    if confs is not None:
        ax1.text(0.05, 0.05, infostr, ha='left', va='bottom',
                transform=ax1.transAxes, family='monospace')

    if label is not None:
        ax1.set_title(label)

    if plotdata and plotresiduals:
        xlaxis = ax2
    else:
        xlaxis = ax1

    if xlabel is None:
        xlaxis.set_xlabel('Energy [{0}]'.format(_latex_unit(e_unit)))
    else:
        xlaxis.set_xlabel(xlabel)

    f.subplots_adjust(hspace=0)

    return f

def plot_data(input_data, xlabel=None, ylabel=None, sed=True, figure=None,
        e_unit=None, data_color='r', **kwargs):
    """
    Plot spectral data.

    Additional ``kwargs`` are passed to `plot_fit`, except ``confs`` and
    ``plotdata``.

    Parameters
    ----------
    input_data : `emcee.EnsembleSampler`, `astropy.table.Table`, or `dict`
        Spectral data to plot. Can be given as a data table, a dict generated
        with `validate_data_table` or a `emcee.EnsembleSampler` with a data
        property.
    xlabel : str, optional
        Label for the ``x`` axis of the plot.
    ylabel : str, optional
        Label for the ``y`` axis of the plot.
    sed : bool, optional
        Whether to plot SED or differential spectrum.
    figure : `matplotlib.figure.Figure`, optional
        `matplotlib` figure to plot on. If omitted a new one will be generated.
    e_unit : `astropy.unit.Unit`, optional
        Units for energy axis. Defaults to those of the data.
    data_color : str
        Matplotlib color for the data points.
    """

    import matplotlib.pyplot as plt

    # Plot everything in serif to match math exponents
    plt.rc('font', family='serif')

    if isinstance(input_data, table.Table):
        data = validate_data_table(input_data)
    elif hasattr(input_data,'data'):
        data = input_data.data
    elif isinstance(input_data, dict) and 'energy' in input_data.keys():
        data = input_data
    else:
        log.warning('input_data format not know, no plotting data!')
        return None

    if figure == None:
        f = plt.figure()
    else:
        f = figure

    if len(f.axes) > 0:
        ax1 = f.axes[0]
    else:
        ax1 = f.add_subplot(111)

    # try to get units from previous plot in figure
    try:
        old_e_unit = u.Unit(ax1.get_xlabel().split('[')[-1].split(']')[0])
    except ValueError:
        old_e_unit = u.Unit('')

    if e_unit is None and old_e_unit.physical_type == 'energy':
        e_unit = old_e_unit
    elif e_unit is None:
        e_unit = data['energy'].unit

    _plot_data_to_ax(data, ax1, e_unit=e_unit, sed=sed, data_color=data_color,
            ylabel=ylabel)

    if xlabel is not None:
        ax1.set_xlabel(xlabel)
    elif xlabel is None and ax1.get_xlabel() == '':
        ax1.set_xlabel('Energy [{0}]'.format(_latex_unit(e_unit)))

    ax1.autoscale()

    return f


def _plot_data_to_ax(data, ax1, e_unit=None, sed=True, data_color='r',
        ylabel=None):
    """ Plots data errorbars and upper limits onto ax.
    X label is left to plot_data and plot_fit because they depend on whether
    residuals are plotted.
    """

    if e_unit is None:
        e_unit = data['energy'].unit

    def plot_ulims(ax, x, y, xerr):
        """
        Plot upper limits as arrows with cap at value of upper limit.
        """
        ax.errorbar(x, y, xerr=xerr, ls='',
                color=data_color, elinewidth=2, capsize=0)
        ax.errorbar(x, 0.75 * y, yerr=0.25*y, ls='', lolims=True,
                color=data_color, elinewidth=2, capsize=5, zorder=10)

    f_unit, sedf = sed_conversion(data['energy'], data['flux'].unit, sed)

    ul = data['ul']
    notul = -ul

    # Hack to show y errors compatible with 0 in loglog plot
    yerr = data['dflux'][:, notul]
    y = data['flux'][notul].to(yerr.unit)
    bad_err = np.where((y-yerr[0]) <= 0.)
    yerr[0][bad_err] = y[bad_err]*(1.-1e-7)

    ax1.errorbar(data['energy'][notul].to(e_unit).value,
            (data['flux'][notul] * sedf[notul]).to(f_unit).value,
            yerr=(yerr * sedf[notul]).to(f_unit).value,
            xerr=(data['dene'][:, notul]).to(e_unit).value,
            zorder=100, marker='o', ls='', elinewidth=2, capsize=0,
            mec='w', mew=0, ms=6, color=data_color)

    if np.any(ul):
        plot_ulims(ax1, data['energy'][ul].to(e_unit).value,
                (data['flux'][ul] * sedf[ul]).to(f_unit).value,
                (data['dene'][:, ul]).to(e_unit).value)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    xmin = 10 ** np.floor(np.log10(np.min(data['energy'] - data['dene'][0]).value))
    xmax = 10 ** np.ceil(np.log10(np.max(data['energy'] + data['dene'][1]).value))
    ax1.set_xlim(xmin, xmax)
    # avoid autoscaling to errorbars to 0
    if np.any(data['dflux'][:, notul][0] >= data['flux'][notul]):
        elo  = ((data['flux'][notul] * sedf[notul]).to(f_unit).value -
                (data['dflux'][0][notul] * sedf[notul]).to(f_unit).value)
        gooderr = np.where(data['dflux'][0][notul] < data['flux'][notul])
        ymin = 10 ** np.floor(np.log10(np.min(elo[gooderr])))
        ax1.set_ylim(bottom=ymin)

    if ylabel is None:
        if sed:
            ax1.set_ylabel(r'$E^2\mathsf{{d}}N/\mathsf{{d}}E$'
                ' [{0}]'.format(_latex_unit(u.Unit(f_unit))))
        else:
            ax1.set_ylabel(r'$\mathsf{{d}}N/\mathsf{{d}}E$'
                    ' [{0}]'.format(_latex_unit(u.Unit(f_unit))))
    else:
        ax1.set_ylabel(ylabel)


def plot_distribution(samples, label, figure=None):

    from scipy import stats
    import matplotlib.pyplot as plt


    quant = [16, 50, 84]
    quantiles = dict(six.moves.zip(quant, np.percentile(samples, quant)))
    std = np.std(samples)

    if isinstance(samples[0], u.Quantity):
        unit = samples[0].unit
    else:
        unit = ''

    if isinstance(std, u.Quantity):
        std = std.value

    dist_props = '{label} distribution properties:\n \
    - median: ${median}$ {unit}, std: ${std}$ {unit}\n \
    - Median with uncertainties based on \n \
      the 16th and 84th percentiles ($\sim$1$\sigma$):\n\
          {label} = ${{{median}}}^{{+{uncs[1]}}}_{{-{uncs[0]}}}$ {unit}'.format(
                  label=label, median=_latex_float(quantiles[50]),
                  uncs=(_latex_float(quantiles[50] - quantiles[16]),
                        _latex_float(quantiles[84] - quantiles[50])), std=_latex_float(std), unit=unit)

    if figure is None:
        f = plt.figure()
    else:
        f = figure

    f.text(0.1, 0.23, dist_props, ha='left', va='top')

    ax = f.add_subplot(111)
    f.subplots_adjust(bottom=0.35)

    histnbins = min(max(25, int(len(samples)/100.)), 100)
    xlabel = label
    n, x, patch = ax.hist(samples, histnbins, histtype='stepfilled', color='#CC0000', lw=0, normed=1)
    if isinstance(samples, u.Quantity):
        samples_nounit = samples.value
    else:
        samples_nunit = samples

    kde = stats.kde.gaussian_kde(samples_nounit)
    ax.plot(x, kde(x), c='k', label='KDE')

    ax.axvline(quantiles[50], ls='--', c='k', alpha=0.5, lw=2,
                label='50% quantile')
    ax.axvspan(quantiles[16], quantiles[84], color='0.5', alpha=0.25,
                label='68% CI')
    # ax.legend()
    [l.set_rotation(45) for l in ax.get_xticklabels()]
    #[l.set_rotation(45) for l in ax.get_yticklabels()]
    if unit != '':
        xlabel += ' [{0}]'.format(unit)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.set_title('posterior distribution of {0}'.format(label))
    ax.set_ylim(top=n.max() * 1.05)

    return f
