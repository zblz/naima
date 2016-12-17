# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import astropy.units as u
from astropy.table import Table, QTable
from astropy import log
from astropy.extern import six
from astropy.utils.exceptions import AstropyUserWarning
import warnings
import h5py
import os

from .plot import find_ML

try:
    import yaml
    HAS_PYYAML = True
except ImportError:
    HAS_PYYAML = False

__all__ = [
    "save_diagnostic_plots", "save_results_table", "save_run", "read_run"
]


def save_diagnostic_plots(outname,
                          sampler,
                          modelidxs=None,
                          pdf=False,
                          sed=True,
                          blob_labels=None,
                          last_step=False,
                          dpi=100):
    """
    Generate diagnostic plots.

    - A plot for each of the chain parameters showing walker progression, final
      sample distribution and several statistical measures of this
      distribution: ``outname_chain_parN.png`` (see `naima.plot_chain`).
    - A corner plot of sample density in the two dimensional parameter space of
      all parameter pairs of the run, with the Maximum Likelihood parameter
      vector indicated in blue: ``outname_corner.png`` (see `corner.corner`).
    - A plot for each of the models returned as blobs by the model function.
      The maximum likelihood model is shown, as well as the 1 and 3 sigma
      confidence level contours. The first model will be compared with
      observational data and residuals shown. ``outname_fit_modelN.png`` (see
      `naima.plot_fit` and `naima.plot_blob`).

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

    from .plot import plot_chain, plot_blob, plot_corner
    from matplotlib import pyplot as plt
    # This function should never be interactive
    old_interactive = plt.rcParams['interactive']
    plt.rcParams['interactive'] = False

    if pdf:
        plt.rc('pdf', fonttype=42)
        log.info('Saving diagnostic plots in file '
                 '{0}_plots.pdf'.format(outname))
        from matplotlib.backends.backend_pdf import PdfPages
        outpdf = PdfPages('{0}_plots.pdf'.format(outname))

    # Chains

    for par, label in six.moves.zip(
            six.moves.range(sampler.chain.shape[-1]), sampler.labels):
        try:
            log.info('Plotting chain of parameter {0}...'.format(label))
            f = plot_chain(sampler, par, last_step=last_step)
            if pdf:
                f.savefig(outpdf, format='pdf', dpi=dpi)
            else:
                if 'log(' in label or 'log10(' in label:
                    label = label.split('(')[-1].split(')')[0]
                f.savefig('{0}_chain_{1}.png'.format(outname, label), dpi=dpi)
            f.clf()
            plt.close(f)
        except Exception as e:
            log.warning('plot_chain failed for paramter'
                        ' {0} ({1}): {2}'.format(label, par, e))

    # Corner plot

    log.info('Plotting corner plot...')

    f = plot_corner(sampler)
    if f is not None:
        if pdf:
            f.savefig(outpdf, format='pdf', dpi=dpi)
        else:
            f.savefig('{0}_corner.png'.format(outname), dpi=dpi)
        f.clf()
        plt.close(f)

    # Fit

    if modelidxs is None:
        nmodels = len(sampler.blobs[-1][0])
        modelidxs = list(six.moves.range(nmodels))

    if isinstance(sed, bool):
        sed = [sed for idx in modelidxs]

    if blob_labels is None:
        blob_labels = ['Model output {0}'.format(idx) for idx in modelidxs]
    elif len(modelidxs) == 1 and isinstance(blob_labels, str):
        blob_labels = [blob_labels]
    elif len(blob_labels) < len(modelidxs):
        # Add labels
        n = len(blob_labels)
        blob_labels += ['Model output {0}'.format(idx)
                        for idx in modelidxs[n:]]

    for modelidx, plot_sed, label in six.moves.zip(
            modelidxs, sed, blob_labels):

        try:
            log.info('Plotting {0}...'.format(label))
            f = plot_blob(
                sampler,
                blobidx=modelidx,
                label=label,
                sed=plot_sed,
                n_samples=100,
                last_step=last_step)
            if pdf:
                f.savefig(outpdf, format='pdf', dpi=dpi)
            else:
                f.savefig('{0}_model{1}.png'.format(outname, modelidx),
                          dpi=dpi)
            f.clf()
            plt.close(f)
        except Exception as e:
            log.warning('plot_blob failed for {0}: {1}'.format(label, e))

    if pdf:
        outpdf.close()

    # set interactive back to original
    plt.rcParams['interactive'] = old_interactive


def save_results_table(outname,
                       sampler,
                       format='ascii.ecsv',
                       convert_log=True,
                       last_step=False,
                       include_blobs=True):
    """
    Save an ASCII table with the results stored in the
    `~emcee.EnsembleSampler`.

    The table contains the median, 16th and 84th percentile confidence region
    (~1sigma) for each parameter.

    Parameters
    ----------
    outname : str
        Root name to be used to save the table. ``_results.dat`` will be
        appended for the output filename.

    sampler : `emcee.EnsembleSampler` instance
        Sampler instance from which chains, blobs and data are read.

    format : str, optional
        Format of the saved table. Must be a format string accepted by
        `astropy.table.Table.write`, see the `astropy unified file read/write
        interface documentation
        <https://astropy.readthedocs.org/en/latest/io/unified.html>`_. Only the
        ``ascii.ecsv`` and ``ascii.ipac`` formats are able to preserve all the
        information stored in the ``run_info`` dictionary of the sampler.
        Defaults to ``ascii.ecsv`` if available (only in astropy > v1.0), else
        ``ascii.ipac``.

    convert_log : bool, optional
        Whether to convert natural or base-10 logarithms into original values
        in addition to saving the logarithm value.

    last_step : bool, optional
        Whether to only use the positions in the final step of the run (True,
        default) or the whole chain (False).

    include_blobs : bool, optional
        Whether to save the distribution properties of the scalar blobs in the
        sampler. Default is True.

    Returns
    -------

    table : `~astropy.table.Table`
        Table with the results.
    """

    if not HAS_PYYAML and format == 'ascii.ecsv':
        format = 'ascii.ipac'
        log.warning("PyYAML package is required for ECSV format,"
                    " falling back to {0}...".format(format))
    elif format not in ['ascii.ecsv', 'ascii.ipac']:
        log.warning('The chosen table format does not have an astropy'
                    ' writer that suppports metadata writing, no run info'
                    ' will be saved to the file!')

    file_extension = 'dat'
    if format == 'ascii.ecsv':
        file_extension = 'ecsv'

    log.info('Saving results table in {0}_results.{1}'.format(outname,
                                                              file_extension))

    labels = sampler.labels

    if last_step:
        dists = sampler.chain[:, -1, :]
    else:
        dists = sampler.flatchain

    quant = [16, 50, 84]
    # Do we need more info on the distributions?
    t = Table(
        names=['label', 'median', 'unc_lo', 'unc_hi'],
        dtype=['S72', 'f8', 'f8', 'f8'])
    t['label'].description = 'Name of the parameter'
    t['median'].description = 'Median of the posterior distribution function'
    t['unc_lo'].description = (
        'Difference between the median and the'
        ' {0}th percentile of the pdf, ~1sigma lower uncertainty'.format(quant[
            0]))
    t['unc_hi'].description = (
        'Difference between the {0}th percentile'
        ' and the median of the pdf, ~1sigma upper uncertainty'
        .format(quant[2])
    )

    metadata = {}
    # Start with info from the distributions used for storing the results
    metadata['n_samples'] = dists.shape[0]
    # save ML parameter vector and best/median loglikelihood
    ML, MLp, MLerr, _ = find_ML(sampler, None)
    metadata['ML_pars'] = [float(p) for p in MLp]
    metadata['MaxLogLikelihood'] = float(ML)

    # compute and save BIC
    BIC = len(MLp) * np.log(len(sampler.data)) - 2 * ML
    metadata['BIC'] = BIC

    # And add all info stored in the sampler.run_info dict
    if hasattr(sampler, 'run_info'):
        metadata.update(sampler.run_info)

    for p, label in enumerate(labels):
        dist = dists[:, p]
        xquant = np.percentile(dist, quant)
        quantiles = dict(six.moves.zip(quant, xquant))
        med = quantiles[50]
        lo, hi = med - quantiles[16], quantiles[84] - med

        t.add_row((label, med, lo, hi))

        if convert_log and ('log10(' in label or 'log(' in label):
            nlabel = label.split('(')[-1].split(')')[0]
            ltype = label.split('(')[0]
            if ltype == 'log10':
                new_dist = 10**dist
            elif ltype == 'log':
                new_dist = np.exp(dist)

            quantiles = dict(
                six.moves.zip(quant, np.percentile(new_dist, quant)))
            med = quantiles[50]
            lo, hi = med - quantiles[16], quantiles[84] - med

            t.add_row((nlabel, med, lo, hi))

    if include_blobs:
        nblobs = len(sampler.blobs[-1][0])
        for idx in range(nblobs):
            blob0 = sampler.blobs[-1][0][idx]

            IS_SCALAR = False
            if isinstance(blob0, u.Quantity):
                if blob0.size == 1:
                    IS_SCALAR = True
                    unit = blob0.unit
            elif np.isscalar(blob0):
                IS_SCALAR = True
                unit = None

            if IS_SCALAR:
                if last_step:
                    blobl = [m[idx] for m in sampler.blobs[-1]]
                else:
                    blobl = []
                    for step in sampler.blobs:
                        for walkerblob in step:
                            blobl.append(walkerblob[idx])
                if unit:
                    dist = np.array([b.value for b in blobl])
                    metadata['blob{0}_unit'.format(idx)] = unit.to_string()
                else:
                    dist = np.array(blobl)

                quantiles = dict(
                    six.moves.zip(quant, np.percentile(dist, quant)))
                med = quantiles[50]
                lo, hi = med - quantiles[16], quantiles[84] - med

                t.add_row(('blob{0}'.format(idx), med, lo, hi))

    if format == 'ascii.ipac':
        # Only keywords are written to IPAC tables
        t.meta['keywords'] = {}
        for di in metadata.items():
            t.meta['keywords'][di[0]] = {'value': di[1]}
    else:
        if format == 'ascii.ecsv':
            # there can be no numpy arrays in the metadata (YAML doesn't like
            # them)
            for di in list(metadata.items()):
                if type(di[1]).__module__ == np.__name__:
                    try:
                        # convert arrays
                        metadata[di[0]] = [np.asscalar(a) for a in di[1]]
                    except TypeError:
                        # convert scalars
                        metadata[di[0]] = np.asscalar(di[1])
        # Save it directly in meta for readability in ECSV
        t.meta.update(metadata)

    t.write('{0}_results.{1}'.format(outname, file_extension), format=format)

    return t


def save_run(filename, sampler, compression=True, clobber=False):
    """
    Save the sampler chain, data table, parameter labels, metadata blobs, and
    run information to a hdf5 file.

    The data table and parameter labels stored in the sampler will also be
    saved to the hdf5 file.

    Parameters
    ----------
    filename : str
        Filename for hdf5 file. If the filename extension is not 'h5' or
        'hdf5', the suffix '_chain.h5' will be appended to the filename.

    sampler : `emcee.EnsembleSampler` instance
        Sampler instance for which chain and run information is saved.

    compression : bool, optional
        Whether gzip compression is applied to the dataset on write. Default is
        True.

    clobber : bool, optional
        Whether to overwrite the output filename if it exists.
    """

    if filename.split('.')[-1] not in ['h5', 'hdf5']:
        filename += '_chain.h5'

    if os.path.exists(filename) and not clobber:
        log.warning(
            'Not writing file because file exists and clobber is False')
        return

    f = h5py.File(filename, 'w')
    group = f.create_group('sampler')
    group.create_dataset(
        'chain', data=sampler.chain, compression=compression)
    group.create_dataset(
        'lnprobability', data=sampler.lnprobability, compression=compression)

    # blobs
    blob = sampler.blobs[-1][0]
    for idx, item in enumerate(blob):
        if isinstance(item, u.Quantity):
            # scalar or array quantity
            units = [item.unit.to_string()]
        elif isinstance(item, float):
            units = ['']
        elif isinstance(item, tuple) or isinstance(item, list):
            arearrs = np.all([isinstance(x, np.ndarray) for x in item])
            if arearrs:
                units = []
                for x in item:
                    if isinstance(x, u.Quantity):
                        units.append(x.unit.to_string())
                    else:
                        units.append('')
        else:
            log.warning(
                'blob number {0} has unknown format and cannot be saved '
                'in HDF5 file'
            )
            continue

        # traverse blobs list. This will probably be slow and there should be a
        # better way
        blob = []
        for step in sampler.blobs:
            for walkerblob in step:
                blob.append(walkerblob[idx])
        blob = u.Quantity(blob).value

        blobdataset = group.create_dataset(
            'blob{0}'.format(idx), data=blob, compression=compression)
        if len(units) > 1:
            for j, unit in enumerate(units):
                blobdataset.attrs['unit{0}'.format(j)] = unit
        else:
            blobdataset.attrs['unit'] = units[0]

    if hasattr(sampler, 'data'):
        data = group.create_dataset(
            'data',
            data=Table(sampler.data).as_array(),
            compression=compression)

        for col in sampler.data.colnames:
            f['sampler/data'].attrs[col + 'unit'] = str(sampler.data[col].unit)

        for key in sampler.data.meta:
            val = sampler.data.meta[key]
            try:
                data.attrs[key] = val
            except TypeError:
                try:
                    data.attrs[key] = str(val)
                except:
                    warnings.warn(
                        "Attribute `{0}` of type {1} of the data table"
                        " of the sampler cannot be written to HDF5 files"
                        "- skipping".format(key, type(val)), AstropyUserWarning
                    )

    # add all run info to group attributes
    if hasattr(sampler, 'run_info'):
        for key in sampler.run_info.keys():
            val = sampler.run_info[key]
            try:
                group.attrs[key] = val
            except TypeError:
                group.attrs[key] = str(val)

    # add other sampler info to the attrs
    group.attrs['acceptance_fraction'] = np.mean(sampler.acceptance_fraction)

    # add labels as individual attrs (there might be a better way)
    for i, label in enumerate(sampler.labels):
        group.attrs['label{0}'.format(i)] = label

    f.close()


class _result(object):
    """
    Minimal emcee.EnsembleSampler like container for chain results
    """

    @property
    def flatchain(self):
        s = self.chain.shape
        return self.chain.reshape(s[0] * s[1], s[2])

    @property
    def flatlnprobability(self):
        return self.lnprobability.flatten()


def read_run(filename, modelfn=None):
    """
    Read chain from a hdf5 saved with `save_run`.

    This function will also read the labels, data table, and metadata blobs
    stored in the original sampler. If you want to use the result object with
    `plot_fit` and setting the ``e_range`` parameter, you must provide the
    model function with the `modelfn` argument given that functions cannot be
    serialized in hdf5 files.

    Parameters
    ----------
    filename : str
        Filename of the hdf5 containing the chain, lnprobability, and blob
        arrays in the group 'sampler'

    modelfn : function, optional
        Model function to be attached to the returned sampler

    Returns
    -------
    result : class
        Container object with same properties as an `emcee.EnsembleSampler`
        resulting from a sampling run. This object can be passed onto
        `~naima.plot_fit`, `~naima.plot_chain`, and `~naima.plot_corner` for
        analysis as you would do with a `emcee.EnsembleSampler` instance.
    """

    # initialize empty sampler class to return
    result = _result()
    result.modelfn = modelfn
    result.run_info = {}

    f = h5py.File(filename, 'r')
    # chain and lnprobability
    result.chain = np.array(f['sampler/chain'])
    result.lnprobability = np.array(f['sampler/lnprobability'])

    # blobs
    result.blobs = []
    shape = result.chain.shape[:2]
    blobs = []
    blobrank = []
    for i in range(100):
        # first read each of the blobs and convert to Quantities
        try:
            ds = f['sampler/blob{0}'.format(i)]
            rank = np.ndim(ds[0])
            blobrank.append(rank)
            if rank <= 1:
                blobs.append(u.Quantity(ds.value, unit=ds.attrs['unit']))
            else:
                blob = []
                for j in range(np.ndim(ds[0])):
                    blob.append(
                        u.Quantity(
                            ds.value[:, j, :],
                            unit=ds.attrs['unit{0}'.format(j)]))
                blobs.append(blob)
        except KeyError:
            break

    # Now organize in an awful list of lists of arrays
    for step in range(shape[1]):
        steplist = []
        for walker in range(shape[0]):
            n = step * shape[0] + walker
            walkerblob = []
            for j in range(len(blobs)):
                if blobrank[j] <= 1:
                    walkerblob.append(blobs[j][n])
                else:
                    blob = []
                    for k in range(blobrank[j]):
                        blob.append(blobs[j][k][n])
                    walkerblob.append(blob)
            steplist.append(walkerblob)
        result.blobs.append(steplist)

    # run info
    result.run_info = dict(f['sampler'].attrs)
    result.acceptance_fraction = f['sampler'].attrs['acceptance_fraction']
    # labels
    result.labels = []
    for i in range(result.chain.shape[2]):
        result.labels.append(f['sampler'].attrs['label{0}'.format(i)])

    # data
    data = Table(np.array(f['sampler/data']))
    data.meta.update(f['sampler/data'].attrs)
    for col in data.colnames:
        if f['sampler/data'].attrs[col + 'unit'] != 'None':
            data[col].unit = f['sampler/data'].attrs[col + 'unit']
    result.data = QTable(data)

    return result
