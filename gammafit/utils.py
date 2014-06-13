# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import astropy.units as u
from astropy.extern import six
from astropy import log

from .plot import plot_fit, plot_chain

__all__ = ["generate_energy_edges",
           "build_data_dict", "generate_diagnostic_plots"]

# Convenience tools


def generate_energy_edges(ene):
    """Generate energy bin edges from given energy array.

    Generate an array of energy edges from given energy array to be used as
    abcissa error bar limits when no energy uncertainty or energy band is
    provided.

    Parameters
    ----------
    ene : `astropy.units.Quantity` array instance
        1-D array of energies with associated phsyical units.

    Returns
    -------
    edge_array : `astropy.units.Quantity` array instance of shape ``(2,len(ene))``
        Array of energy edge pairs corresponding to each given energy of the
        input array.
    """
    midene = np.sqrt((ene[1:] * ene[:-1]))
    elo, ehi = np.zeros_like(ene), np.zeros_like(ene)
    elo[1:] = ene[1:] - midene
    ehi[:-1] = midene - ene[:-1]
    elo[0] = ehi[0]
    ehi[-1] = elo[-1]
    return np.array((elo, ehi)) * ene.unit


def build_data_dict(ene, dene, flux, dflux, ul=None, cl=0.99):
    """
    Read data into data dict.

    Parameters
    ----------

    ene : array (Nene)
        Spectrum energies

    dene : array (2,Nene) or None
        Difference from energy points to lower (row 0) and upper (row 1)
        energy edges. Currently only used on plots. If ``None`` is given, they
        will be generated with function `generate_energy_edges`.

    flux : array (Nene)
        Spectrum flux values.

    dflux : array (2,Nene) or (Nene)
        Spectrum flux uncertainties. If shape is (2,Nene), rows 0 and 1
        correspond to lower and upper uncertainties, respectively.

    ul : array of bool (optional)
        Boolean array indicating which of the flux values given in ``flux``
        correspond to upper limits.

    cl : float (optional)
        Confidence level of the flux upper limits given by ``ul``.

    Returns
    -------
    data : dict
        Data stored in a `dict`.
    """
    if ul == None:
        ul = np.array((False,) * len(ene))

    if dene == None:
        dene = generate_energy_edges(ene)

    # data is a dict with the fields:
    # ene dene flux dflux ul cl
    data = {}
    for val in ['ene', 'dene', 'flux', 'dflux', 'ul', 'cl']:
        data[val] = eval(val)

    return data


def generate_diagnostic_plots(outname, sampler, modelidxs=None, pdf=False, sed=None, **kwargs):
    """
    Generate diagnostic plots.

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
    outname : str
        Name to be used to save diagnostic plot files.

    sampler : `emcee.EnsembleSampler` instance
        Sampler instance from which chains, blobs and data are read.

    modelidxs : iterable (optional)
        Model numbers to be plotted. Default: All returned in sampler.blobs

    pdf : bool (optional)
        Whether to save plots to multipage pdf.
    """

    if pdf:
        from matplotlib import pyplot as plt
        plt.rc('pdf', fonttype=42)
        print(
            'Generating diagnostic plots in file {0}_plots.pdf'.format(outname))
        from matplotlib.backends.backend_pdf import PdfPages
        outpdf = PdfPages('{0}_plots.pdf'.format(outname))

    # Chains

    for par, label in zip(six.moves.range(sampler.chain.shape[-1]), sampler.labels):
        try:
            f = plot_chain(sampler, par, **kwargs)
            if pdf:
                f.savefig(outpdf, format='pdf')
            else:
                if 'log(' in label or 'log10(' in label:
                    label = label.split('(')[-1].split(')')[0]
                f.savefig('{0}_chain_{1}.png'.format(outname, label))
            del f
        except Exception as e:
            log.warn('plot_chain failed for paramter {0}: {1}'.format(par,e))

    # Corner plot

    try:
        from triangle import corner
        from .plot import find_ML

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
        print('triangle_plot not installed, corner plot not available')

    # Fit

    if modelidxs is None:
        nmodels = len(sampler.blobs[-1][0])
        modelidxs = list(range(nmodels))

    if sed is None:
        sed = [None for idx in modelidxs]
    elif isinstance(sed, bool):
        sed = [sed for idx in modelidxs]

    for modelidx, plot_sed in zip(modelidxs, sed):
        try:
            blob0 = sampler.blobs[-1][0][modelidx]
            if isinstance(blob0, u.Quantity):
                modelx = sampler.data['ene']
                modely = blob0
            elif len(blob0) == 2:
                modelx = blob0[0]
                modely = blob0[1]
            else:
                raise TypeError
            assert(len(modelx) == len(modely))
        except (TypeError, AssertionError):
            log.warn(
                'Not plotting model {0} because of wrong blob format'.format(modelidx))
            continue

        try:
            e_unit = modelx.unit
            f_unit = modely.unit
        except AttributeError:
            log.warn(
                'Not plotting model {0} because of lack of units'.format(modelidx))
            continue

        f = plot_fit(sampler, modelidx=modelidx, sed=plot_sed, **kwargs)
        if pdf:
            f.savefig(outpdf, format='pdf')
        else:
            f.savefig('{0}_fit_model{1}.png'.format(outname, modelidx))
        del f

    if pdf:
        outpdf.close()
