# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.io import ascii
from astropy import log
from astropy.extern import six
import warnings

HAS_PYYAML = True
try:
    import yaml
except ImportError:
    HAS_PYYAML = False

__all__ = ["save_diagnostic_plots", "save_results_table"]

def save_diagnostic_plots(outname, sampler, modelidxs=None, pdf=False, sed=None,
        blob_labels=None, **kwargs):
    """
    Generate diagnostic plots.

    - A plot for each of the chain parameters showing walker progression, final
      sample distribution and several statistical measures of this distribution:
      ``outname_chain_parN.png`` (see `naima.plot_chain`).
    - A corner plot of sample density in the two dimensional parameter space of
      all parameter pairs of the run, with the Maximum Likelihood parameter
      vector indicated in blue: ``outname_corner.png`` (see `triangle.corner`).
    - A plot for each of the models returned as blobs by the model function. The
      maximum likelihood model is shown, as well as the 1 and 3 sigma confidence
      level contours. The first model will be compared with observational data
      and residuals shown. ``outname_fit_modelN.png`` (see `naima.plot_fit` and
      `naima.plot_blob`).

    Parameters
    ----------
    outname : str
        Name to be used to save diagnostic plot files.

    sampler : `emcee.EnsembleSampler` instance
        Sampler instance from which chains, blobs and data are read.

    modelidxs : iterable of integers, optional
        Model numbers to be plotted. Default: All returned in sampler.blobs

    blob_labels : list of strings, optional
        Label for each of the outputs of the model. They will be used as title
        for the corresponding plot.

    pdf : bool, optional
        Whether to save plots to multipage pdf.
    """

    from .plot import plot_chain, plot_blob

    if pdf:
        from matplotlib import pyplot as plt
        plt.rc('pdf', fonttype=42)
        log.info( 'Generating diagnostic plots in file '
                '{0}_plots.pdf'.format(outname))
        from matplotlib.backends.backend_pdf import PdfPages
        outpdf = PdfPages('{0}_plots.pdf'.format(outname))

    # Chains

    for par, label in six.moves.zip(six.moves.range(sampler.chain.shape[-1]),
                                    sampler.labels):
        try:
            log.info('Plotting chain of parameter {0}...'.format(label))
            f = plot_chain(sampler, par, **kwargs)
            if pdf:
                f.savefig(outpdf, format='pdf')
            else:
                if 'log(' in label or 'log10(' in label:
                    label = label.split('(')[-1].split(')')[0]
                f.savefig('{0}_chain_{1}.png'.format(outname, label))
            del f
        except Exception as e:
            log.warning('plot_chain failed for paramter'
                    ' {0} ({1}): {2}'.format(label,par,e))

    # Corner plot

    try:
        from triangle import corner
        from .plot import find_ML

        log.info('Plotting corner plot...')

        ML, MLp, MLvar, model_ML = find_ML(sampler, 0)
        f = corner(sampler.flatchain, labels=sampler.labels,
                   truths=MLp, quantiles=[0.16, 0.5, 0.84],
                   verbose=False, **kwargs)
        if pdf:
            f.savefig(outpdf, format='pdf')
        else:
            f.savefig('{0}_corner.png'.format(outname))
        del f
    except ImportError:
        log.warning('triangle_plot not installed, corner plot not available')

    # Fit

    if modelidxs is None:
        nmodels = len(sampler.blobs[-1][0])
        modelidxs = list(six.moves.range(nmodels))

    if sed is None:
        sed = [None for idx in modelidxs]
    elif isinstance(sed, bool):
        sed = [sed for idx in modelidxs]

    if blob_labels is None:
        blob_labels = ['Model output {0}'.format(idx) for idx in modelidxs]
    elif len(modelidxs)==1 and isinstance(blob_labels, str):
        blob_labels = [blob_labels,]
    elif len(blob_labels) < len(modelidxs):
        # Add labels
        n = len(blob_labels)
        blob_labels += ['Model output {0}'.format(idx) for idx in modelidxs[n:]]

    for modelidx, plot_sed, label in six.moves.zip(modelidxs, sed, blob_labels):

        try:
            log.info('Plotting {0}...'.format(label))
            f = plot_blob(sampler, blobidx=modelidx, label=label,
                          sed=plot_sed, n_samples=100, **kwargs)
            if pdf:
                f.savefig(outpdf, format='pdf')
            else:
                f.savefig('{0}_model{1}.png'.format(outname, modelidx))
            del f
        except Exception as e:
            log.warning('plot_blob failed for model output {0}: {1}'.format(par,e))

    if pdf:
        outpdf.close()


def save_results_table(outname, sampler, table_format='ascii.ecsv',
        convert_log=True, last_step=True, **kwargs):
    """
    Save an ASCII table with the results stored in the `~emcee.EnsembleSampler`.

    The table contains the median, 16th and 84th percentile confidence region
    (~1sigma) for each parameter.

    Parameters
    ----------
    outname : str
        Root name to be used to save the table. ``_results.dat`` will be
        appended for the output filename.

    sampler : `emcee.EnsembleSampler` instance
        Sampler instance from which chains, blobs and data are read.

    table_format : str, optional
        Format of the saved table. Must be a format string accepted by
        `astropy.table.Table.write`, see the `astropy unified file read/write
        interface documentation
        <https://astropy.readthedocs.org/en/latest/io/unified.html>`_. Only the
        ``ascii.ecsv`` and ``ascii.ipac`` formats are able to preserve all the
        information stored in the ``run_info`` dictionary of the sampler.
        Defaults to ``ascii.ecsv`` if available (only in astropy > v1.0), else
        ``ascii.ipac``.

    convert_log : bool, optional
        Whether to convert natural or base-10 logarithms into original values in
        addition to saving the logarithm value.

    last_step : bool, optional
        Whether to only use the positions in the final step of the run (True,
        default) or the whole chain (False).

    Returns
    -------

    table : `~astropy.table.Table`
        Table with the results.
    """

    if not hasattr(ascii,'ecsv') and table_format == 'ascii.ecsv':
        table_format = 'ascii.ipac'
        log.warning("ECSV format not available (only in astropy >= v1.0),"
                " falling back to {0}...".format(table_format))
    elif not HAS_PYYAML and table_format == 'ascii.ecsv':
        table_format = 'ascii.ipac'
        log.warning("PyYAML package is required for ECSV format,"
                " falling back to {0}...".format(table_format))
    elif table_format not in ['ascii.ecsv','ascii.ipac']:
        log.warning('The chosen table format does not have an astropy'
                ' writer that suppports metadata writing, no run info'
                ' will be saved to the file!')

    file_extension = 'dat'
    if table_format == 'ascii.ecsv':
        file_extension = 'ecsv'

    labels = sampler.labels
    maxlenlabel = max([len(l) for l in labels])

    if last_step == True:
        dists = sampler.chain[:,-1,:]
    else:
        dists = sampler.flatchain

    quant = [16, 50, 84]
    # Do we need more info on the distributions?
    t=Table(names=['label','median','err_lo','err_hi'],
            dtype=['S{0}'.format(maxlenlabel),'f8','f8','f8'])
    t['label'].description   = 'Name of the parameter'
    t['median'].description  = 'Median of the posterior distribution function'
    t['err_lo'].description = ('Difference between the median and the'
                ' {0}th percentile of the pdf, ~1sigma lower uncertainty'.format(quant[0]))
    t['err_hi'].description = ('Difference between the {0}th percentile'
                ' and the median of the pdf, ~1sigma upper uncertainty'.format(quant[2]))

    metadata = {}
    # Start with info from the distributions used for storing the results
    metadata['n_samples']= dists.shape[0]
    # And add all info stored in the sampler.run_info dict
    if hasattr(sampler,'run_info'):
        metadata.update(sampler.run_info)

    if table_format == 'ascii.ipac':
        # Only keywords are written to IPAC tables
        t.meta['keywords'] = {}
        for di in metadata.items():
            t.meta['keywords'][di[0]]={'value':di[1]}
    else:
        # Save it directly in meta for readability in ECSV
        t.meta.update(metadata)

    for p,label in enumerate(labels):
        dist = dists[:,p]
        xquant = np.percentile(dist, quant)
        quantiles = dict(six.moves.zip(quant, xquant))
        med = quantiles[50]
        lo,hi = med - quantiles[16], quantiles[84] - med

        t.add_row((label, med, lo, hi))

        if convert_log and ('log10(' in label or 'log(' in label):
            nlabel = label.split('(')[-1].split(')')[0]
            ltype = label.split('(')[0]
            if ltype == 'log10':
                new_dist = 10**dist
            elif ltype == 'log':
                new_dist = np.exp(dist)

            quant = [16, 50, 84]
            quantiles = dict(six.moves.zip(quant, np.percentile(new_dist, quant)))
            med = quantiles[50]
            lo,hi = med - quantiles[16], quantiles[84] - med

            t.add_row((nlabel, med, lo, hi))


    t.write('{0}_results.{1}'.format(outname,file_extension),format=table_format)

    return t
