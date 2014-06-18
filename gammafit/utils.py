# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import astropy.units as u
from astropy.extern import six
from astropy import log
from .extern.validator import validate_array, validate_scalar

__all__ = ["generate_energy_edges", "sed_conversion",
           "build_data_dict", "generate_diagnostic_plots"]

# Input validation tools

def validate_column(data_table,key,pt,domain='positive'):
    try:
        column = data_table[key]
        array = validate_array(key, u.Quantity(column,unit=column.unit), physical_type=pt, domain=domain)
    except KeyError as e:
        raise TypeError('Data table does not contain required column "{0}"'.format(key))

    return array

def validate_data_table(data_table):

    data = {}

    flux_types = ['flux','differential flux','power','differential power']

    # Energy and flux arrays
    data['ene'] = validate_column(data_table,'ene','energy')
    data['flux'] = validate_column(data_table,'flux',flux_types)

    # Flux uncertainties
    if 'flux_error' in data_table.keys():
        dflux = validate_column(data_table,'flux_error',flux_types)
        data['dflux'] = u.Quantity((dflux,dflux))
    elif 'flux_error_lo' in data_table.keys() and 'flux_error_hi' in data_table.keys():
        data['dflux'] = u.Quantity((
            validate_column(data_table,'flux_error_lo',flux_types),
            validate_column(data_table,'flux_error_hi',flux_types)))
    else:
        raise TypeError('Data table does not contain required column'
                        ' "flux_error" or columns "flux_error_lo" and "flux_error_hi"')

    # Energy bin edges
    if 'ene_width' in data_table.keys():
        ene_width = validate_column(data_table,'ene_width', 'energy')
        data['dene'] = u.Quantity((ene_width/2.,ene_width/2.))
    elif 'ene_lo' in data_table.keys() and 'ene_hi' in data_table.keys():
        ene_lo = validate_column(data_table,'ene_lo', 'energy')
        ene_hi = validate_column(data_table,'ene_hi', 'energy')
        data['dene'] = u.Quantity((data['ene']-ene_lo,ene_hi-data['ene']))
    else:
        data['dene'] = generate_energy_edges(data['ene'])

    # Upper limit flags
    if 'ul' in data_table.keys():
        # Check if it is a integer or boolean flag
        ul_col = data_table['ul']
        if ul_col.dtype.type is np.int_ or ul_col.dtype.type is np.bool_:
            data['ul'] = np.array(ul_col, dtype=np.bool)
        elif ul_col.dtype.type is np.string_:
            strbool = True
            for ul in ul_col:
                if ul != 'True' and ul != 'False':
                    strbool = False
            if strbool:
                data['ul'] = np.array((eval(ul) for ul in ul_col),dtype=np.bool)
            else:
                raise TypeError ('UL column is in wrong format')

    if 'cl' in data_table.meta['keywords'].keys():
        data['cl'] = validate_scalar('cl',data_table.meta['keywords']['cl']['value'])
    else:
        data['cl'] = 0.9
        if 'ul' in data_table.keys():
            log.warn('"cl" keyword not provided in input data table, upper limits'
                    'will be assumed to be at 90% confidence level')

    return data



# Convenience tools

def sed_conversion(energy, model_unit, sed):
    """
    Manage conversion between differential spectrum and SED
    """

    model_pt = model_unit.physical_type

    ones = np.ones(energy.shape)

    if sed:
        # SED
        f_unit = u.Unit('erg/s')
        if model_pt == 'power' or model_pt == 'flux' or model_pt == 'energy':
            sedf = ones
        elif 'differential' in model_pt:
            sedf = (energy ** 2)
        else:
            raise u.UnitsError(
                'Model physical type ({0}) is not supported'.format(model_pt),
                'Supported physical types are: power, flux, differential'
                ' power, differential flux')

        if 'flux' in model_pt:
            f_unit /= u.cm ** 2
        elif 'energy' in model_pt:
            # particle energy distributions
            f_unit = u.erg

    elif sed is None:
        # Use original units
        f_unit = model_unit
        sedf = ones
    else:
        # Differential spectrum
        f_unit = u.Unit('1/(s TeV)')
        if 'differential' in model_pt:
            sedf = ones
        elif model_pt == 'power' or model_pt == 'flux' or model_pt == 'energy':
            # From SED to differential
            sedf = 1 / (energy**2)
        else:
            raise u.UnitsError(
                'Model physical type ({0}) is not supported'.format(model_pt),
                'Supported physical types are: power, flux, differential'
                ' power, differential flux')

        if 'flux' in model_pt:
            f_unit /= u.cm ** 2
        elif 'energy' in model_pt:
            # particle energy distributions
            f_unit = u.Unit('1/TeV')

    log.debug(
        'Converted from {0} ({1}) into {2} ({3}) for sed={4}'.format(model_unit, model_pt,
        f_unit, f_unit.physical_type, sed))

    return f_unit, sedf



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

    from .plot import plot_fit, plot_chain

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
