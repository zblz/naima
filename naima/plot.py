# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import astropy.units as u
from astropy.extern import six
from astropy import log
from functools import partial
from emcee import autocorr

from .utils import sed_conversion, validate_data_table
from .extern.validator import validate_array
from .extern.interruptible_pool import InterruptiblePool as Pool

__all__ = ["plot_chain", "plot_fit", "plot_data", "plot_blob", "plot_corner"]

marker_cycle = ['o', 's', 'd', 'p', '*']
# from seaborn: sns.color_palette('deep',6)
color_cycle = [(0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
               (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
               (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
               (0.5058823529411764, 0.4470588235294118, 0.6980392156862745),
               (0.8, 0.7254901960784313, 0.4549019607843137),
               (0.39215686274509803, 0.7098039215686275, 0.803921568627451)]


def plot_chain(sampler, p=None, **kwargs):
    """Generate a diagnostic plot of the sampler chains.

    Parameters
    ----------
    sampler : `emcee.EnsembleSampler`
        Sampler containing the chains to be plotted.
    p : int (optional)
        Index of the parameter to plot. If omitted, all chains are plotted.
    last_step : bool (optional)
        Whether to plot the last step of the chain or the complete chain
        (default).

    Returns
    -------
    figure : `matplotlib.figure.Figure`
        Figure
    """
    if p is None:
        npars = sampler.chain.shape[-1]
        for pp in six.moves.range(npars):
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


def round2(x, n):
    y = round(x, n)
    if n < 1:
        y = str(int(y))
    else:
        # preserve trailing zeroes
        y = ('{{0:.{0}f}}'.format(n)).format(x)
    return y


def _latex_value_error(val, elo, ehi=0, tol=0.25):
    order = int(np.log10(abs(val)))
    if order > 2 or order < -2:
        val /= 10**order
        elo /= 10**order
        ehi /= 10**order
    else:
        order = 0
    nlo = -int(np.floor(np.log10(elo)))
    if elo * 10 ** nlo < 2:
        nlo += 1
    if ehi:
        # elo = round(elo,nlo)
        nhi = -int(np.floor(np.log10(ehi)))
        if ehi * 10 ** nhi < 2:
            nhi += 1
        # ehi = round(ehi,nhi)
        if np.abs(elo - ehi) / ((elo + ehi) / 2.) > tol:
            n = max(nlo, nhi)
            string = '{0}^{{+{1}}}_{{-{2}}}'.format(* [
                round2(x, nn) for x, nn in zip([val, ehi, elo], [n, nhi, nlo])
            ])
        else:
            e = (elo + ehi) / 2.
            n = -int(np.floor(np.log10(e)))
            if e * 10**n < 2:
                n += 1
            string = '{0} \pm {1}'.format(* [round2(x, n) for x in [val, e]])
    else:
        string = '{0} \pm {1}'.format(* [round2(x, nlo) for x in [val, elo]])
    if order != 0:
        string = '(' + string + r')\times10^{{{0}}}'.format(order)

    return '$' + string + '$'


def _plot_chain_func(sampler, p, last_step=False):
    chain = sampler.chain
    label = sampler.labels[p]

    import matplotlib.pyplot as plt

    from scipy import stats
    if len(chain.shape) > 2:
        traces = chain[:, :, p]
        if last_step:
            # keep only last step
            dist = traces[:, -1]
        else:
            # convert chain to flatchain
            dist = traces.flatten()
    else:
        log.warning(
            'we need the full chain to plot the traces, not a flatchain!')
        return None

    nwalkers = traces.shape[0]
    nsteps = traces.shape[1]

    f = plt.figure()

    ax1 = f.add_subplot(221)
    ax2 = f.add_subplot(122)

    f.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9)

    # plot five percent of the traces darker

    if nwalkers < 60:
        thresh = 1 - 3. / nwalkers
    else:
        thresh = 0.95
    red = np.arange(nwalkers) / float(nwalkers) >= thresh

    ax1.set_rasterization_zorder(1)
    for t in traces[-red]:  # range(nwalkers):
        ax1.plot(t, color=(0.1,) * 3, lw=1.0, alpha=0.25, zorder=0)
    for t in traces[red]:
        ax1.plot(t, color=color_cycle[0], lw=1.5, alpha=0.75, zorder=0)
    ax1.set_xlabel('step number')
    # [l.set_rotation(45) for l in ax1.get_yticklabels()]
    ax1.set_ylabel(label)
    ax1.yaxis.set_label_coords(-0.15, 0.5)
    ax1.set_title('Walker traces')

    nbins = min(max(25, int(len(dist) / 100.)), 100)
    xlabel = label
    n, x, _ = ax2.hist(
        dist,
        nbins,
        histtype='stepfilled',
        color=color_cycle[0],
        lw=0,
        normed=1)
    kde = stats.kde.gaussian_kde(dist)
    ax2.plot(x, kde(x), color='k', label='KDE')
    quant = [16, 50, 84]
    xquant = np.percentile(dist, quant)
    quantiles = dict(six.moves.zip(quant, xquant))

    ax2.axvline(
        quantiles[50],
        ls='--',
        color='k',
        alpha=0.5,
        lw=2,
        label='50% quantile')
    ax2.axvspan(
        quantiles[16],
        quantiles[84],
        color=(0.5,) * 3,
        alpha=0.25,
        label='68% CI',
        lw=0)
    # ax2.legend()
    for l in ax2.get_xticklabels():
        l.set_rotation(45)
    ax2.set_xlabel(xlabel)
    ax2.xaxis.set_label_coords(0.5, -0.1)
    ax2.set_title('posterior distribution')
    ax2.set_ylim(top=n.max() * 1.05)

    # Print distribution parameters on lower-left

    try:
        try:
            ac = sampler.get_autocorr_time()[p]
        except AttributeError:
            ac = autocorr.integrated_time(
                np.mean(
                    chain, axis=0), axis=0, fast=False)[p]
        autocorr_message = '{0:.1f}'.format(ac)
    except autocorr.AutocorrError:
        # Raised when chain is too short for meaningful auto-correlation
        # estimation
        autocorr_message = None

    if last_step:
        clen = 'last ensemble'
    else:
        clen = 'whole chain'

    chain_props = 'Walkers: {0} \nSteps in chain: {1} \n'.format(nwalkers,
                                                                 nsteps)
    if autocorr_message is not None:
        chain_props += 'Autocorrelation time: {0}\n'.format(autocorr_message)
    chain_props += 'Mean acceptance fraction: {0:.3f}\n'.format(
                        np.mean(sampler.acceptance_fraction)) +\
                   'Distribution properties for the {clen}:\n \
    $-$ median: ${median}$, std: ${std}$ \n \
    $-$ median with uncertainties based on \n \
      the 16th and 84th percentiles ($\sim$1$\sigma$):\n'.format(
              median=_latex_float(quantiles[50]),
              std=_latex_float(np.std(dist)), clen=clen)

    info_line = ' ' * 10 + label + ' = ' + _latex_value_error(
        quantiles[50], quantiles[50] - quantiles[16], quantiles[84] -
        quantiles[50])

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

        label_template = '\n' + ' ' * 10 + '{{label:>{0}}}'.format(len(label))

        new_line = label_template.format(label=nlabel)
        new_line += ' = ' + _latex_value_error(quantiles[50], quantiles[50] -
                                               quantiles[16], quantiles[84] -
                                               quantiles[50])

        chain_props += new_line
        info_line += new_line

    log.info('{0:-^50}\n'.format(label) + info_line)
    f.text(0.05, 0.45, chain_props, ha='left', va='top')

    return f


def _process_blob(sampler, modelidx, last_step=False, energy=None):
    """
    Process binary blob in sampler. If blob in position modelidx is:

    - a Quantity array of len(blob[i])=len(data['energy']: use blob as model,
      data['energy'] as modelx
    - a tuple: use first item as modelx, second as model
    - a Quantity scalar: return array of scalars
    """

    # Allow process blob to be used by _calc_samples and _calc_ML by sending
    # only blobs, not full sampler
    if hasattr(sampler, 'blobs'):
        blob0 = sampler.blobs[-1][0][modelidx]
        blobs = sampler.blobs
        energy = sampler.data['energy']
    else:
        blobs = [sampler]
        blob0 = sampler[0][modelidx]
        last_step = True

    if isinstance(blob0, u.Quantity):
        if blob0.size == energy.size:
            # Energy array for blob is not provided, use data['energy']
            modelx = energy
        elif blob0.size == 1:
            modelx = None

        if last_step:
            model = u.Quantity([m[modelidx] for m in blobs[-1]])
        else:
            model = []
            for step in blobs:
                for walkerblob in step:
                    model.append(walkerblob[modelidx])
            model = u.Quantity(model)
    elif np.isscalar(blob0):
        modelx = None

        if last_step:
            model = u.Quantity([m[modelidx] for m in blobs[-1]])
        else:
            model = []
            for step in blobs:
                for walkerblob in step:
                    model.append(walkerblob[modelidx])
            model = u.Quantity(model)
    elif (isinstance(blob0, list) or isinstance(blob0, tuple)):
        if (len(blob0) == 2 and isinstance(blob0[0], u.Quantity) and
                isinstance(blob0[1], u.Quantity)):
            # Energy array for model is item 0 in blob, model flux is item 1
            modelx = blob0[0]

            if last_step:
                model = u.Quantity([m[modelidx][1] for m in blobs[-1]])
            else:
                model = []
                for step in blobs:
                    for walkerblob in step:
                        model.append(walkerblob[modelidx][1])
                model = u.Quantity(model)
        else:
            raise TypeError('Model {0} has wrong blob format'.format(modelidx))

    else:
        raise TypeError('Model {0} has wrong blob format'.format(modelidx))

    return modelx, model


def _read_or_calc_samples(sampler,
                          modelidx=0,
                          n_samples=100,
                          last_step=False,
                          e_range=None,
                          e_npoints=100,
                          threads=None):
    """Get samples from blob or compute them from chain and sampler.modelfn
    """

    if not e_range:
        # return the results saved in blobs
        modelx, model = _process_blob(sampler, modelidx, last_step=last_step)
    else:
        # prepare bogus data for calculation
        e_range = validate_array(
            'e_range', u.Quantity(e_range), physical_type='energy')
        e_unit = e_range.unit
        energy = np.logspace(
            np.log10(e_range[0].value), np.log10(e_range[1].value),
            e_npoints) * e_unit
        data = {
            'energy': energy,
            'flux': np.zeros(energy.shape) * sampler.data['flux'].unit
        }
        # init pool and select parameters
        chain = sampler.chain[-1] if last_step else sampler.flatchain
        pars = chain[np.random.randint(len(chain), size=n_samples)]
        blobs = []

        p = Pool(threads)
        modelouts = p.map(partial(sampler.modelfn, data=data), pars)
        p.close()
        p.terminate()

        for modelout in modelouts:
            if isinstance(modelout, np.ndarray):
                blobs.append([modelout])
            else:
                blobs.append(modelout)

        modelx, model = _process_blob(
            blobs, modelidx=modelidx, energy=data['energy'])

    return modelx, model


def _calc_ML(sampler, modelidx=0, e_range=None, e_npoints=100):
    """Get ML model from blob or compute them from chain and sampler.modelfn
    """

    ML, MLp, MLerr, ML_model = find_ML(sampler, modelidx)

    if e_range is not None:
        # prepare bogus data for calculation
        e_range = validate_array(
            'e_range', u.Quantity(e_range), physical_type='energy')
        e_unit = e_range.unit
        energy = np.logspace(
            np.log10(e_range[0].value), np.log10(e_range[1].value),
            e_npoints) * e_unit
        data = {
            'energy': energy,
            'flux': np.zeros(energy.shape) * sampler.data['flux'].unit
        }
        modelout = sampler.modelfn(MLp, data)

        if isinstance(modelout, np.ndarray):
            blob = modelout
        else:
            blob = modelout[modelidx]

        if isinstance(blob, u.Quantity):
            modelx = data['energy'].copy()
            model_ML = blob.copy()
        elif len(blob) == 2:
            modelx = blob[0].copy()
            model_ML = blob[1].copy()
        else:
            raise TypeError('Model {0} has wrong blob format'.format(modelidx))

        ML_model = (modelx, model_ML)

    return ML, MLp, MLerr, ML_model


def _calc_CI(sampler,
             modelidx=0,
             confs=[3, 1],
             last_step=False,
             e_range=None,
             e_npoints=100,
             threads=None):
    """Calculate confidence interval.
    """
    from scipy import stats

    # If we are computing the samples for the confidence intervals, we need at
    # least one sample to constrain the highest confidence band
    # 1 sigma -> 6 samples
    # 2 sigma -> 43 samples
    # 3 sigma -> 740 samples
    # 4 sigma -> 31574 samples
    # 5 sigma -> 3488555 samples
    # We limit it to 1000 samples and warn that it might not be enough
    if e_range:
        maxconf = np.max(confs)
        minsamples = min(100, int(1 / stats.norm.cdf(-maxconf) + 1))
        if minsamples > 1000:
            log.warning(
                'In order to sample the confidence band for {0} sigma,'
                ' {1} new samples need to be computed, but we are limiting'
                ' it to 1000 samples, so the confidence band might not be'
                ' well constrained.'
                ' Consider reducing the maximum'
                ' confidence significance or using the samples stored in'
                ' the sampler by setting e_range'
                ' to None'.format(maxconf, minsamples))
            minsamples = 1000
    else:
        minsamples = None

    modelx, model = _read_or_calc_samples(
        sampler,
        modelidx,
        last_step=last_step,
        e_range=e_range,
        e_npoints=e_npoints,
        n_samples=minsamples,
        threads=threads)

    nwalkers = len(model) - 1
    CI = []
    for conf in confs:
        fmin = stats.norm.cdf(-conf)
        fmax = stats.norm.cdf(conf)
        ymin, ymax = [], []
        for fr, y in ((fmin, ymin), (fmax, ymax)):
            nf = int((fr * nwalkers))
            for i in six.moves.range(len(modelx)):
                ysort = np.sort(model[:, i])
                y.append(ysort[nf])

        # create an array from lists ymin and ymax preserving units
        CI.append((u.Quantity(ymin), u.Quantity(ymax)))

    return modelx, CI


def _plot_MLmodel(ax, sampler, modelidx, e_range, e_npoints, e_unit, sed):
    """compute and plot ML model"""

    ML, MLp, MLerr, ML_model = _calc_ML(
        sampler, modelidx, e_range=e_range, e_npoints=e_npoints)
    f_unit, sedf = sed_conversion(ML_model[0], ML_model[1].unit, sed)

    ax.loglog(
        ML_model[0].to(e_unit).value, (ML_model[1] * sedf).to(f_unit).value,
        color='k',
        lw=2,
        alpha=0.8)


def plot_CI(ax,
            sampler,
            modelidx=0,
            sed=True,
            confs=[3, 1, 0.5],
            e_unit=u.eV,
            label=None,
            e_range=None,
            e_npoints=100,
            threads=None,
            last_step=False):
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
        List of confidence levels (in sigma) to use for generating the
        confidence intervals. Default is `[3,1,0.5]`
    e_unit : :class:`~astropy.units.Unit` or str parseable to unit
        Unit in which to plot energy axis.
    e_npoints : int, optional
        How many points to compute for the model samples and ML model if
        `e_range` is set.
    threads : int, optional
        How many parallel processing threads to use when computing the samples.
        Defaults to the number of available cores.
    last_step : bool, optional
        Whether to only use the positions in the final step of the run (True,
        default) or the whole chain (False).
    """
    confs.sort(reverse=True)

    modelx, CI = _calc_CI(
        sampler,
        modelidx=modelidx,
        confs=confs,
        e_range=e_range,
        e_npoints=e_npoints,
        last_step=last_step,
        threads=threads)
    # pick first confidence interval curve for units
    f_unit, sedf = sed_conversion(modelx, CI[0][0].unit, sed)

    for (ymin, ymax), conf in zip(CI, confs):
        color = np.log(conf) / np.log(20) + 0.4
        ax.fill_between(
            modelx.to(e_unit).value, (ymax * sedf).to(f_unit).value,
            (ymin * sedf).to(f_unit).value,
            lw=0.001,
            color=(color,) * 3,
            alpha=0.6,
            zorder=-10)

    _plot_MLmodel(ax, sampler, modelidx, e_range, e_npoints, e_unit, sed)

    if label is not None:
        ax.set_ylabel('{0} [{1}]'.format(label, f_unit.to_string(
            'latex_inline')))


def plot_samples(ax,
                 sampler,
                 modelidx=0,
                 sed=True,
                 n_samples=100,
                 e_unit=u.eV,
                 e_range=None,
                 e_npoints=100,
                 threads=None,
                 label=None,
                 last_step=False):
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
    e_range : list of `~astropy.units.Quantity`, length 2, optional
        Limits in energy for the computation of the model samples and ML model.
        Note that setting this parameter will mean that the samples for the
        model are recomputed and depending on the model speed might be quite
        slow.
    e_npoints : int, optional
        How many points to compute for the model samples and ML model if
        `e_range` is set.
    threads : int, optional
        How many parallel processing threads to use when computing the samples.
        Defaults to the number of available cores.
    last_step : bool, optional
        Whether to only use the positions in the final step of the run (True,
        default) or the whole chain (False).
    """

    modelx, model = _read_or_calc_samples(
        sampler,
        modelidx,
        last_step=last_step,
        e_range=e_range,
        e_npoints=e_npoints,
        threads=threads)
    # pick first model sample for units
    f_unit, sedf = sed_conversion(modelx, model[0].unit, sed)

    sample_alpha = min(5. / n_samples, 0.5)
    for my in model[np.random.randint(len(model), size=n_samples)]:
        ax.loglog(
            modelx.to(e_unit).value, (my * sedf).to(f_unit).value,
            color=(0.1,) * 3,
            alpha=sample_alpha,
            lw=1.0)

    _plot_MLmodel(ax, sampler, modelidx, e_range, e_npoints, e_unit, sed)

    if label is not None:
        ax.set_ylabel('{0} [{1}]'.format(label, f_unit.to_string(
            'latex_inline')))


def find_ML(sampler, modelidx):
    """
    Find Maximum Likelihood parameters as those in the chain with a highest log
    probability.
    """
    index = np.unravel_index(
        np.argmax(sampler.lnprobability), sampler.lnprobability.shape)
    MLp = sampler.chain[index]
    if modelidx is not None and hasattr(sampler, 'blobs'):
        blob = sampler.blobs[index[1]][index[0]][modelidx]
        if isinstance(blob, u.Quantity):
            modelx = sampler.data['energy'].copy()
            model_ML = blob.copy()
        elif len(blob) == 2:
            modelx = blob[0].copy()
            model_ML = blob[1].copy()
        else:
            raise TypeError('Model {0} has wrong blob format'.format(modelidx))
    elif modelidx is not None and hasattr(sampler, 'modelfn'):
        blob = _process_blob(
            [sampler.modelfn(MLp, sampler.data)],
            modelidx,
            energy=sampler.data['energy'])
        modelx, model_ML = blob[0], blob[1][0]
    else:
        modelx, model_ML = None, None

    MLerr = []
    for dist in sampler.flatchain.T:
        hilo = np.percentile(dist, [16., 84.])
        MLerr.append((hilo[1] - hilo[0]) / 2.)
    ML = sampler.lnprobability[index]

    return ML, MLp, MLerr, (modelx, model_ML)


def plot_blob(sampler,
              blobidx=0,
              label=None,
              last_step=False,
              figure=None,
              **kwargs):
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
    if label is None:
        label = 'Model output {0}'.format(blobidx)

    if modelx is None:
        # Blob is scalar, plot distribution
        f = plot_distribution(model, label, figure=figure)
    else:
        f = plot_fit(
            sampler,
            modelidx=blobidx,
            last_step=last_step,
            label=label,
            figure=figure,
            **kwargs)

    return f


def plot_fit(sampler,
             modelidx=0,
             label=None,
             sed=True,
             last_step=False,
             n_samples=100,
             confs=None,
             ML_info=False,
             figure=None,
             plotdata=None,
             plotresiduals=None,
             e_unit=None,
             e_range=None,
             e_npoints=100,
             threads=None,
             xlabel=None,
             ylabel=None,
             ulim_opts={},
             errorbar_opts={}):
    """
    Plot data with fit confidence regions.

    Parameters
    ----------
    sampler : `emcee.EnsembleSampler`
        Sampler with a stored chain.
    modelidx : int, optional
        Model index to plot.
    label : str, optional
        Label for the title of the plot.
    sed : bool, optional
        Whether to plot SED or differential spectrum.
    last_step : bool, optional
        Whether to use only the samples of the last step in the run when
        showing either the model samples or the confidence intervals.
    n_samples : int, optional
        If not ``None``, number of sample models to plot. If ``None``,
        confidence bands will be plotted instead of samples. Default is 100.
    confs : list, optional
        List of confidence levels (in sigma) to use for generating the
        confidence intervals. Default is to plot sample models instead of
        confidence bands.
    ML_info : bool, optional
        Whether to plot information about the maximum likelihood parameters and
        the standard deviation of their distributions. Default is True.
    figure : `matplotlib.figure.Figure`, optional
        `matplotlib` figure to plot on. If omitted a new one will be generated.
    plotdata : bool, optional
        Wheter to plot data on top of model confidence intervals. Default is
        True if the physical types of the data and the model match.
    plotresiduals : bool, optional
        Wheter to plot the residuals with respect to the maximum likelihood
        model. Default is True if ``plotdata`` is True and either ``confs`` or
        ``n_samples`` are set.
    e_unit : `~astropy.units.Unit`, optional
        Units for the energy axis of the plot. The default is to use the units
        of the energy array of the observed data.
    e_range : list of `~astropy.units.Quantity`, length 2, optional
        Limits in energy for the computation of the model samples and ML model.
        Note that setting this parameter will mean that the samples for the
        model are recomputed and depending on the model speed might be quite
        slow.
    e_npoints : int, optional
        How many points to compute for the model samples and ML model if
        `e_range` is set.
    threads : int, optional
        How many parallel processing threads to use when computing the samples.
        Defaults to the number of available cores.
    xlabel : str, optional
        Label for the ``x`` axis of the plot.
    ylabel : str, optional
        Label for the ``y`` axis of the plot.
    ulim_opts : dict
        Option for upper-limit plotting. Available options are capsize (arrow
        width) and height_fraction (arrow length in fraction of flux value).
    errorbar_opts : dict
        Addtional options to pass to `matplotlib.plt.errorbar` for plotting the
        spectral flux points.

    """
    import matplotlib.pyplot as plt

    ML, MLp, MLerr, model_ML = find_ML(sampler, modelidx)
    infostr = 'Maximum log probability: {0:.3g}\n'.format(ML)
    infostr += 'Maximum Likelihood values:\n'
    maxlen = np.max([len(ilabel) for ilabel in sampler.labels])
    vartemplate = '{{2:>{0}}}: {{0:>8.3g}} +/- {{1:<8.3g}}\n'.format(maxlen)
    for p, v, ilabel in zip(MLp, MLerr, sampler.labels):
        infostr += vartemplate.format(p, v, ilabel)

    # log.info(infostr)

    data = sampler.data

    if e_range is None and not hasattr(sampler, 'blobs'):
        e_range = data['energy'][[0, -1]] * np.array((1. / 3., 3.))

    if len(model_ML[0]) == len(data['energy']) and plotdata is None:
        plotdata = True
    elif plotdata is None:
        plotdata = False

    if plotresiduals is None and plotdata and (confs is not None or n_samples):
        plotresiduals = True

    if confs is None and not n_samples and plotdata and not plotresiduals:
        # We actually only want to plot the data, so let's go there
        return plot_data(
            sampler.data,
            xlabel=xlabel,
            ylabel=ylabel,
            sed=sed,
            figure=figure,
            e_unit=e_unit,
            ulim_opts=ulim_opts,
            errorbar_opts=errorbar_opts)

    if figure is None:
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
        plot_CI(
            ax1,
            sampler,
            modelidx,
            sed=sed,
            confs=confs,
            e_unit=e_unit,
            label=label,
            e_range=e_range,
            e_npoints=e_npoints,
            last_step=last_step,
            threads=threads)
    elif n_samples:
        plot_samples(
            ax1,
            sampler,
            modelidx,
            sed=sed,
            n_samples=n_samples,
            e_unit=e_unit,
            label=label,
            e_range=e_range,
            e_npoints=e_npoints,
            last_step=last_step,
            threads=threads)
    else:
        # plot only ML model
        _plot_MLmodel(ax1, sampler, modelidx, e_range, e_npoints, e_unit, sed)

    xlaxis = ax1
    if plotdata:
        _plot_data_to_ax(
            data,
            ax1,
            e_unit=e_unit,
            sed=sed,
            ylabel=ylabel,
            ulim_opts=ulim_opts,
            errorbar_opts=errorbar_opts)
        if plotresiduals:
            _plot_residuals_to_ax(
                data,
                model_ML,
                ax2,
                e_unit=e_unit,
                sed=sed,
                errorbar_opts=errorbar_opts)
            xlaxis = ax2
            for tl in ax1.get_xticklabels():
                tl.set_visible(False)
        xmin = 10**np.floor(
            np.log10(
                np.min(data['energy'] - data['energy_error_lo']).to(e_unit)
                .value))
        xmax = 10**np.ceil(
            np.log10(
                np.max(data['energy'] + data['energy_error_hi']).to(e_unit)
                .value))
        ax1.set_xlim(xmin, xmax)
    else:
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        if sed:
            ndecades = 10
        else:
            ndecades = 20
        # restrict y axis to ndecades to avoid autoscaling deep exponentials
        xmin, xmax, ymin, ymax = ax1.axis()
        ymin = max(ymin, ymax / 10**ndecades)
        ax1.set_ylim(bottom=ymin)
        # scale x axis to largest model_ML x point within ndecades decades of
        # maximum
        f_unit, sedf = sed_conversion(model_ML[0], model_ML[1].unit, sed)
        hi = np.where((model_ML[1] * sedf).to(f_unit).value > ymin)
        xmax = np.max(model_ML[0][hi])
        ax1.set_xlim(right=10**np.ceil(np.log10(xmax.to(e_unit).value)))

    if e_range:
        # ensure that xmin/xmax contains e_range
        xmin, xmax, ymin, ymax = ax1.axis()
        xmin = min(xmin, e_range[0].to(e_unit).value)
        xmax = max(xmax, e_range[1].to(e_unit).value)
        ax1.set_xlim(xmin, xmax)

    if ML_info and (confs is not None or n_samples):
        ax1.text(
            0.05,
            0.05,
            infostr,
            ha='left',
            va='bottom',
            transform=ax1.transAxes,
            family='monospace')

    if label is not None:
        ax1.set_title(label)

    if xlabel is None:
        xlaxis.set_xlabel('Energy [{0}]'.format(
            e_unit.to_string('latex_inline')))
    else:
        xlaxis.set_xlabel(xlabel)

    f.subplots_adjust(hspace=0)

    return f


def _plot_ulims(ax,
                x,
                y,
                xerr,
                color,
                capsize=5,
                height_fraction=0.25,
                elinewidth=2):
    """
    Plot upper limits as arrows with cap at value of upper limit.

    uplim behaviour has been fixed in matplotlib 1.4
    """
    ax.errorbar(
        x, y, xerr=xerr, ls='', color=color, elinewidth=elinewidth, capsize=0)

    from distutils.version import LooseVersion
    import matplotlib
    mpl_version = LooseVersion(matplotlib.__version__)
    if mpl_version >= LooseVersion('1.4.0'):
        ax.errorbar(
            x,
            y,
            yerr=height_fraction * y,
            ls='',
            uplims=True,
            color=color,
            elinewidth=elinewidth,
            capsize=capsize,
            zorder=10)
    else:
        ax.errorbar(
            x, (1 - height_fraction) * y,
            yerr=height_fraction * y,
            ls='',
            lolims=True,
            color=color,
            elinewidth=elinewidth,
            capsize=capsize,
            zorder=10)


def _plot_data_to_ax(data_all,
                     ax1,
                     e_unit=None,
                     sed=True,
                     ylabel=None,
                     ulim_opts={},
                     errorbar_opts={}):
    """ Plots data errorbars and upper limits onto ax.
    X label is left to plot_data and plot_fit because they depend on whether
    residuals are plotted.
    """

    if e_unit is None:
        e_unit = data_all['energy'].unit

    f_unit, sedf = sed_conversion(data_all['energy'], data_all['flux'].unit,
                                  sed)

    if 'group' not in data_all.keys():
        data_all['group'] = np.zeros(len(data_all))

    groups = np.unique(data_all['group'])

    for g in groups:
        data = data_all[np.where(data_all['group'] == g)]
        _, sedfg = sed_conversion(data['energy'], data['flux'].unit, sed)

        # wrap around color and marker cycles
        color = color_cycle[int(g) % len(color_cycle)]
        marker = marker_cycle[int(g) % len(marker_cycle)]

        ul = data['ul']
        notul = -ul

        # Hack to show y errors compatible with 0 in loglog plot
        yerr_lo = data['flux_error_lo'][notul]
        y = data['flux'][notul].to(yerr_lo.unit)
        bad_err = np.where((y - yerr_lo) <= 0.)
        yerr_lo[bad_err] = y[bad_err] * (1. - 1e-7)
        yerr = u.Quantity((yerr_lo, data['flux_error_hi'][notul]))
        xerr = u.Quantity((data['energy_error_lo'], data['energy_error_hi']))

        opts = dict(
            zorder=100,
            marker=marker,
            ls='',
            elinewidth=2,
            capsize=0,
            mec=color,
            mew=0.1,
            ms=5,
            color=color)
        opts.update(**errorbar_opts)

        ax1.errorbar(
            data['energy'][notul].to(e_unit).value,
            (data['flux'][notul] * sedfg[notul]).to(f_unit).value,
            yerr=(yerr * sedfg[notul]).to(f_unit).value,
            xerr=xerr[:, notul].to(e_unit).value,
            **opts)

        if np.any(ul):
            if 'elinewidth' in errorbar_opts:
                ulim_opts['elinewidth'] = errorbar_opts['elinewidth']

            _plot_ulims(ax1, data['energy'][ul].to(e_unit).value,
                        (data['flux'][ul] * sedfg[ul]).to(f_unit).value,
                        (xerr[:, ul]).to(e_unit).value, color, **ulim_opts)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    xmin = 10**np.floor(
        np.log10(
            np.min(data['energy'] - data['energy_error_lo']).to(e_unit).value))
    xmax = 10**np.ceil(
        np.log10(
            np.max(data['energy'] + data['energy_error_hi']).to(e_unit).value))
    ax1.set_xlim(xmin, xmax)
    # avoid autoscaling to errorbars to 0
    notul = -data_all['ul']
    if np.any(data_all['flux_error_lo'][notul] >= data_all['flux'][notul]):
        elo = (
            (data_all['flux'][notul] * sedf[notul]).to(f_unit).value -
            (data_all['flux_error_lo'][notul] * sedf[notul]).to(f_unit).value)
        gooderr = np.where(
            data_all['flux_error_lo'][notul] < data_all['flux'][notul])
        ymin = 10**np.floor(np.log10(np.min(elo[gooderr])))
        ax1.set_ylim(bottom=ymin)

    if ylabel is None:
        if sed:
            ax1.set_ylabel(r'$E^2\mathrm{{d}}N/\mathrm{{d}}E$'
                           ' [{0}]'.format(
                               u.Unit(f_unit).to_string('latex_inline')))
        else:
            ax1.set_ylabel(r'$\mathrm{{d}}N/\mathrm{{d}}E$'
                           ' [{0}]'.format(
                               u.Unit(f_unit).to_string('latex_inline')))
    else:
        ax1.set_ylabel(ylabel)


def _plot_residuals_to_ax(data_all,
                          model_ML,
                          ax,
                          e_unit=u.eV,
                          sed=True,
                          errorbar_opts={}):
    """Function to compute and plot residuals in units of the uncertainty"""
    if 'group' not in data_all.keys():
        data_all['group'] = np.zeros(len(data_all))

    groups = np.unique(data_all['group'])

    MLf_unit, MLsedf = sed_conversion(model_ML[0], model_ML[1].unit, sed)
    MLene = model_ML[0].to(e_unit)
    MLflux = (model_ML[1] * MLsedf).to(MLf_unit)

    ax.axhline(0, color='k', lw=1, ls='--')

    interp = False
    if (data_all['energy'].size != MLene.size or
            not np.allclose(data_all['energy'].value, MLene.value)):
        interp = True
        from scipy.interpolate import interp1d
        modelfunc = interp1d(MLene.value, MLflux.value, bounds_error=False)

    for g in groups:
        groupidx = np.where(data_all['group'] == g)

        data = data_all[groupidx]

        notul = -data['ul']
        df_unit, dsedf = sed_conversion(data['energy'], data['flux'].unit, sed)
        ene = data['energy'].to(e_unit)
        xerr = u.Quantity((data['energy_error_lo'], data['energy_error_hi']))
        flux = (data['flux'] * dsedf).to(df_unit)
        dflux = (data['flux_error_lo'] + data['flux_error_hi']) / 2.
        dflux = (dflux * dsedf).to(df_unit)[notul]

        if interp:
            difference = flux[notul] - modelfunc(ene[notul]) * flux.unit
        else:
            difference = flux[notul] - MLflux[groupidx][notul]

        # wrap around color and marker cycles
        color = color_cycle[int(g) % len(color_cycle)]
        marker = marker_cycle[int(g) % len(marker_cycle)]

        opts = dict(
            zorder=100,
            marker=marker,
            ls='',
            elinewidth=2,
            capsize=0,
            mec=color,
            mew=0.1,
            ms=6,
            color=color)
        opts.update(errorbar_opts)

        ax.errorbar(
            ene[notul].value, (difference / dflux).decompose().value,
            yerr=(dflux / dflux).decompose().value,
            xerr=xerr[:, notul].to(e_unit).value,
            **opts)

    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(
        MaxNLocator(
            5, integer='True', prune='upper', symmetric=True))

    ax.set_ylabel(r'$\Delta\sigma$')
    ax.set_xscale('log')


def plot_data(input_data,
              xlabel=None,
              ylabel=None,
              sed=True,
              figure=None,
              e_unit=None,
              ulim_opts={},
              errorbar_opts={}):
    """
    Plot spectral data.

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
    ulim_opts : dict
        Options for upper-limit plotting. Available options are capsize (arrow
        width) and height_fraction (arrow length in fraction of flux value).
    errorbar_opts : dict
        Addtional options to pass to `matplotlib.plt.errorbar` for plotting the
        spectral flux points.
    """

    import matplotlib.pyplot as plt

    try:
        data = validate_data_table(input_data)
    except TypeError:
        if hasattr(input_data, 'data'):
            data = input_data.data
        elif isinstance(input_data, dict) and 'energy' in input_data.keys():
            data = input_data
        else:
            log.warning('input_data format not know, no plotting data!')
            return None

    if figure is None:
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

    _plot_data_to_ax(
        data,
        ax1,
        e_unit=e_unit,
        sed=sed,
        ylabel=ylabel,
        ulim_opts=ulim_opts,
        errorbar_opts=errorbar_opts)

    if xlabel is not None:
        ax1.set_xlabel(xlabel)
    elif xlabel is None and ax1.get_xlabel() == '':
        ax1.set_xlabel(r'$\mathrm{Energy}$' + ' [{0}]'.format(
            e_unit.to_string('latex_inline')))

    ax1.autoscale()

    return f


def plot_distribution(samples, label, figure=None):
    """ Plot a distribution and print statistics about it"""

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
    $-$ median: ${median}$ {unit}, std: ${std}$ {unit}\n \
    $-$ Median with uncertainties based on \n \
      the 16th and 84th percentiles ($\sim$1$\sigma$):\n\
          {label} = {value_error} {unit}'.format(
        label=label,
        median=_latex_float(quantiles[50]),
        std=_latex_float(std),
        value_error=_latex_value_error(quantiles[50],
                                       quantiles[50] - quantiles[16],
                                       quantiles[84] - quantiles[50]),
        unit=unit)

    if figure is None:
        f = plt.figure()
    else:
        f = figure

    ax = f.add_subplot(111)
    f.subplots_adjust(bottom=0.40, top=0.93, left=0.06, right=0.95)

    f.text(0.2, 0.27, dist_props, ha='left', va='top')

    histnbins = min(max(25, int(len(samples) / 100.)), 100)
    xlabel = '' if label is None else label
    n, x, _ = ax.hist(
        samples,
        histnbins,
        histtype='stepfilled',
        color=color_cycle[0],
        lw=0,
        normed=1)
    if isinstance(samples, u.Quantity):
        samples_nounit = samples.value
    else:
        samples_nounit = samples

    kde = stats.kde.gaussian_kde(samples_nounit)
    ax.plot(x, kde(x), color='k', label='KDE')

    ax.axvline(
        quantiles[50],
        ls='--',
        color='k',
        alpha=0.5,
        lw=2,
        label='50% quantile')
    ax.axvspan(
        quantiles[16],
        quantiles[84],
        color=(0.5,) * 3,
        alpha=0.25,
        label='68% CI',
        lw=0)
    # ax.legend()
    for l in ax.get_xticklabels():
        l.set_rotation(45)
    # [l.set_rotation(45) for l in ax.get_yticklabels()]
    if unit != '':
        xlabel += ' [{0}]'.format(unit)
    ax.set_xlabel(xlabel)
    ax.set_title('posterior distribution of {0}'.format(label))
    ax.set_ylim(top=n.max() * 1.05)

    return f


def plot_corner(sampler, show_ML=True, **kwargs):
    """
    A plot that summarizes the parameter samples by showing them as individual
    histograms and 2D histograms against each other. The maximum likelihood
    parameter vector is indicated by a cross.

    This function is a thin wrapper around `corner.corner`, found at
    https://github.com/corner/corner.py.

    Parameters
    ----------
    sampler : `emcee.EnsembleSampler`
        Sampler with a stored chain.
    show_ML : bool, optional
        Whether to show the maximum likelihood parameter vector as a cross on
        the 2D histograms.
    """
    import matplotlib.pyplot as plt
    oldlw = plt.rcParams['lines.linewidth']
    plt.rcParams['lines.linewidth'] = 0.7
    try:
        from corner import corner

        if show_ML:
            _, MLp, _, _ = find_ML(sampler, 0)
        else:
            MLp = None

        corner_opts = {
            'labels': sampler.labels,
            'truths': MLp,
            'quantiles': [0.16, 0.5, 0.84],
            'verbose': False,
            'truth_color': color_cycle[0],
        }

        corner_opts.update(kwargs)

        f = corner(sampler.flatchain, **corner_opts)
    except ImportError:
        log.warning('The corner package is not installed;'
                    ' corner plot not available')
        f = None

    plt.rcParams['lines.linewidth'] = oldlw

    return f
