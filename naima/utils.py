# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import astropy.units as u
from astropy.table import Table, QTable
from astropy import log
import warnings
import ast
from .extern.validator import validate_array, validate_scalar

__all__ = [
    "generate_energy_edges", "sed_conversion", "build_data_table", "estimate_B"
]

# Input validation tools


def validate_column(data_table, key, pt, domain='positive'):
    try:
        column = data_table[key]
        array = validate_array(
            key,
            u.Quantity(
                column, unit=column.unit),
            physical_type=pt,
            domain=domain)
    except KeyError:
        raise TypeError(
            'Data table does not contain required column "{0}"'.format(key))

    return array


def validate_data_table(data_table, sed=None):
    """
    Validate all columns of a data table. If a list of tables is passed, all
    tables will be validated and then concatenated

    Parameters
    ----------

    data_table : `astropy.table.Table` or list of `astropy.table.Table`.

    sed : bool, optional
        Whether to convert the fluxes to SED. If unset, all data tables are
        converted to the format of the first data table.
    """
    if isinstance(data_table, Table) or isinstance(data_table, QTable):
        data_table = [data_table]

    try:
        for dt in data_table:
            if not isinstance(dt, Table) and not isinstance(dt, QTable):
                raise TypeError(
                    'An object passed as data_table is not an astropy Table!')
    except TypeError:
        raise TypeError(
            'Argument passed to validate_data_table is not a table and '
            'not a list'
        )

    def dt_sed_conversion(dt, sed):
        f_unit, sedf = sed_conversion(dt['energy'], dt['flux'].unit, sed)
        # roundtrip to Table to change the units
        t = Table(dt)
        for col in ['flux', 'flux_error_lo', 'flux_error_hi']:
            t[col].unit = f_unit
        ndt = QTable(t)

        ndt['flux'] = (dt['flux'] * sedf).to(f_unit)
        ndt['flux_error_lo'] = (dt['flux_error_lo'] * sedf).to(f_unit)
        ndt['flux_error_hi'] = (dt['flux_error_hi'] * sedf).to(f_unit)

        return ndt

    data_list = []
    for group, dt in enumerate(data_table):
        dt_val = _validate_single_data_table(dt, group=group)
        data_list.append(dt_val)

    # concatenate input data tables
    data_new = data_list[0].copy()
    f_pt = data_new['flux'].unit.physical_type
    if sed is None:
        sed = f_pt in ['flux', 'power']

    data_new = dt_sed_conversion(data_new, sed)

    for dt in data_list[1:]:
        nf_pt = dt['flux'].unit.physical_type
        if (('flux' in nf_pt and 'power' in f_pt) or
                ('power' in nf_pt and 'flux' in f_pt)):
            raise TypeError(
                'The physical types of the data tables could not be '
                'matched: Some are in flux and others in luminosity units')

        dt = dt_sed_conversion(dt, sed)

        for row in dt:
            data_new.add_row(row)

    return data_new


def _validate_single_data_table(data_table, group=0):

    data = QTable()

    flux_types = ['flux', 'differential flux', 'power', 'differential power']

    # Energy and flux arrays
    data['energy'] = validate_column(data_table, 'energy', 'energy')
    data['flux'] = validate_column(data_table, 'flux', flux_types)

    # Flux uncertainties
    if 'flux_error' in data_table.keys():
        dflux = validate_column(data_table, 'flux_error', flux_types)
        data['flux_error_lo'] = dflux
        data['flux_error_hi'] = dflux
    elif 'flux_error_lo' in data_table.keys(
    ) and 'flux_error_hi' in data_table.keys():
        data['flux_error_lo'] = validate_column(data_table, 'flux_error_lo',
                                                flux_types)
        data['flux_error_hi'] = validate_column(data_table, 'flux_error_hi',
                                                flux_types)
    else:
        raise TypeError('Data table does not contain required column'
                        ' "flux_error" or columns "flux_error_lo"'
                        ' and "flux_error_hi"')

    if 'group' in data_table.colnames:
        # avoid overwriting groups
        data['group'] = data_table['group']
    else:
        data['group'] = [group] * len(data['energy'])

    # Energy bin edges
    if 'energy_width' in data_table.keys():
        energy_width = validate_column(data_table, 'energy_width', 'energy')
        data['energy_error_lo'] = energy_width / 2.
        data['energy_error_hi'] = energy_width / 2.
    elif 'energy_error' in data_table.keys():
        energy_error = validate_column(data_table, 'energy_error', 'energy')
        data['energy_error_lo'] = energy_error
        data['energy_error_hi'] = energy_error
    elif ('energy_error_lo' in data_table.keys() and
          'energy_error_hi' in data_table.keys()):
        energy_error_lo = validate_column(data_table, 'energy_error_lo',
                                          'energy')
        data['energy_error_lo'] = energy_error_lo
        energy_error_hi = validate_column(data_table, 'energy_error_hi',
                                          'energy')
        data['energy_error_hi'] = energy_error_hi
    elif 'energy_lo' in data_table.keys() and 'energy_hi' in data_table.keys():
        energy_lo = validate_column(data_table, 'energy_lo', 'energy')
        data['energy_error_lo'] = (data['energy'] - energy_lo)
        energy_hi = validate_column(data_table, 'energy_hi', 'energy')
        data['energy_error_hi'] = (energy_hi - data['energy'])
    else:
        data['energy_error_lo'], data[
            'energy_error_hi'] = generate_energy_edges(
                data['energy'], groups=data['group'])

    # Upper limit flags
    if 'ul' in data_table.keys():
        # Check if it is a integer or boolean flag
        ul_col = data_table['ul']
        if ul_col.dtype.type is np.int_ or ul_col.dtype.type is np.bool_:
            data['ul'] = np.array(ul_col, dtype=np.bool)
        elif ul_col.dtype.type is np.str_:
            strbool = True
            for ul in ul_col:
                if ul != 'True' and ul != 'False':
                    strbool = False
            if strbool:
                data['ul'] = np.array(
                    [ast.literal_eval(ul) for ul in ul_col], dtype=np.bool)
            else:
                raise TypeError('UL column is in wrong format')
        else:
            raise TypeError('UL column is in wrong format')
    else:
        data['ul'] = np.array([False] * len(data['energy']))

    if 'flux_ul' in data_table.keys():
        data['flux'][data['ul']] = u.Quantity(
            data_table['flux_ul'])[data['ul']]

    HAS_CL = False
    if 'keywords' in data_table.meta.keys():
        if 'cl' in data_table.meta['keywords'].keys():
            HAS_CL = True
            CL = validate_scalar('cl',
                                 data_table.meta['keywords']['cl']['value'])
            data['cl'] = [CL] * len(data['energy'])

    if not HAS_CL:
        data['cl'] = [0.9] * len(data['energy'])
        if np.sum(data['ul']) > 0:
            log.warning(
                '"cl" keyword not provided in input data table, upper limits'
                ' will be assumed to be at 90% confidence level')

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
            sedf = (energy**2)
        else:
            raise u.UnitsError(
                'Model physical type ({0}) is not supported'.format(model_pt),
                'Supported physical types are: power, flux, differential'
                ' power, differential flux')

        if 'flux' in model_pt:
            f_unit /= u.cm**2
        elif 'energy' in model_pt:
            # particle energy distributions
            f_unit = u.erg
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
            f_unit /= u.cm**2
        elif 'energy' in model_pt:
            # particle energy distributions
            f_unit = u.Unit('1/TeV')

    log.debug('Converted from {0} ({1}) into {2} ({3}) for sed={4}'.format(
        model_unit, model_pt, f_unit, f_unit.physical_type, sed))

    return f_unit, sedf


def trapz_loglog(y, x, axis=-1, intervals=False):
    """
    Integrate along the given axis using the composite trapezoidal rule in
    loglog space.

    Integrate `y` (`x`) along given axis in loglog space.

    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like, optional
        Independent variable to integrate over.
    axis : int, optional
        Specify the axis.

    Returns
    -------
    trapz : float
        Definite integral as approximated by trapezoidal rule in loglog space.
    """
    try:
        y_unit = y.unit
        y = y.value
    except AttributeError:
        y_unit = 1.
    try:
        x_unit = x.unit
        x = x.value
    except AttributeError:
        x_unit = 1.

    y = np.asanyarray(y)
    x = np.asanyarray(x)

    slice1 = [slice(None)] * y.ndim
    slice2 = [slice(None)] * y.ndim
    slice1[axis] = slice(None, -1)
    slice2[axis] = slice(1, None)

    if x.ndim == 1:
        shape = [1] * y.ndim
        shape[axis] = x.shape[0]
        x = x.reshape(shape)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Compute the power law indices in each integration bin
        b = np.log10(y[slice2] / y[slice1]) / np.log10(x[slice2] / x[slice1])

        # if local powerlaw index is -1, use \int 1/x = log(x); otherwise use
        # normal powerlaw integration
        trapzs = np.where(
            np.abs(b + 1.) > 1e-10, (y[slice1] * (
                x[slice2] * (x[slice2] / x[slice1])**b - x[slice1])) / (b + 1),
            x[slice1] * y[slice1] * np.log(x[slice2] / x[slice1]))

    tozero = (y[slice1] == 0.) + (y[slice2] == 0.) + (x[slice1] == x[slice2])
    trapzs[tozero] = 0.

    if intervals:
        return trapzs * x_unit * y_unit

    ret = np.add.reduce(trapzs, axis) * x_unit * y_unit

    return ret


def _generate_energy_edges_single(ene):
    """Generate energy edges for single group"""
    midene = np.sqrt((ene[1:] * ene[:-1]))
    elo, ehi = np.zeros(len(ene)) * ene.unit, np.zeros(len(ene)) * ene.unit
    elo[1:] = ene[1:] - midene
    ehi[:-1] = midene - ene[:-1]
    elo[0] = ene[0] * (1 - ene[0] / (ene[0] + ehi[0]))
    ehi[-1] = elo[-1]
    return u.Quantity([elo, ehi])


def generate_energy_edges(ene, groups=None):
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
    energy_err_lo, energy_error_hi : `astropy.units.Quantity` arrays
        Arrays of low and high energy edges corresponding to each given energy
        of the input array.
    """
    if groups is None or len(ene) != len(groups):
        return _generate_energy_edges_single(ene)
    else:
        eloehi = np.zeros((2, len(ene))) * ene.unit
        for g in np.unique(groups):
            group_edges = _generate_energy_edges_single(ene[groups == g])
            eloehi[:, groups == g] = group_edges
        # hstack throws away units
        return eloehi


def build_data_table(energy,
                     flux,
                     flux_error=None,
                     flux_error_lo=None,
                     flux_error_hi=None,
                     energy_width=None,
                     energy_lo=None,
                     energy_hi=None,
                     ul=None,
                     cl=None):
    """
    Read data into data dict.

    Parameters
    ----------

    energy : :class:`~astropy.units.Quantity` array instance
        Observed photon energy array [physical type ``energy``]

    flux : :class:`~astropy.units.Quantity` array instance
        Observed flux array [physical type ``flux`` or ``differential flux``]

    flux_error, flux_error_hi, flux_error_lo : :class:`~astropy.units.Quantity` array instance
        68% CL gaussian uncertainty of the flux [physical type ``flux`` or
        ``differential flux``]. Either ``flux_error`` (symmetrical uncertainty)
        or ``flux_error_hi`` and ``flux_error_lo`` (asymmetrical uncertainties)
        must be provided.

    energy_width, energy_lo, energy_hi : :class:`~astropy.units.Quantity` array instance, optional
        Width of the energy bins [physical type ``energy``]. Either
        ``energy_width`` (bin width) or ``energy_lo`` and ``energy_hi``
        (Energies of the lower and upper bin edges) can be provided. If none
        are provided, ``generate_energy_edges`` will be used.

    ul : boolean or int array, optional
        Boolean array indicating which of the flux values given in ``flux``
        correspond to upper limits.

    cl : float, optional
        Confidence level of the flux upper limits given by ``ul``.

    Returns
    -------
    data : :class:`astropy.table.QTable`
        Data stored in an astropy Table.
    """
    table = QTable()

    if cl is not None:
        cl = validate_scalar('cl', cl)
        table.meta['keywords'] = {'cl': {'value': cl}}

    table['energy'] = energy

    if energy_width is not None:
        table['energy_width'] = energy_width
    elif energy_lo is not None and energy_hi is not None:
        table['energy_lo'] = energy_lo
        table['energy_hi'] = energy_hi

    table['flux'] = flux

    if flux_error is not None:
        table['flux_error'] = flux_error
    elif flux_error_lo is not None and flux_error_hi is not None:
        table['flux_error_lo'] = flux_error_lo
        table['flux_error_hi'] = flux_error_hi
    else:
        raise TypeError('Flux error not provided!')

    if ul is not None:
        ul = np.array(ul, dtype=np.int)
        table['ul'] = ul

    table.meta['comments'] = ['Table generated with naima.build_data_table']

    # test table units, format, etc
    validate_data_table(table)

    return table


def estimate_B(xray_table,
               vhe_table,
               photon_energy_density=0.261 * u.eV / u.cm**3):
    """ Estimate magnetic field from synchrotron to Inverse Compton luminosity
    ratio

    Estimate the magnetic field from the ratio of X-ray to gamma-ray emission
    according to:

    .. math::

        \\frac{L_\mathrm{xray}}{L_\gamma} =
        \\frac{u_\mathrm{B}}{u_\mathrm{ph}} =
        \\frac{B^2}{ 8 \pi u_\mathrm{ph}}

    where :math:`L_\mathrm{xray}` is the X-ray luminosity, :math:`L_\gamma` is
    the gamma-ray luminosity, and :math:`u_\mathrm{ph}` is the seed photon
    field energy density.

    Note that this assumes that the ratio of observed fluxes is equal to the
    ratio of bolometric synchrotron and IC luminosities, and that IC proceeds
    in the Thomson regims. This assumption is safe as long as the X-ray and
    gamma-ray emission contain the bulk of the bolometric emission (i.e., the
    peak in the SED is in the X-ray and gamma-ray observed bands). Even if the
    assumption does not hold, this is a good starting point for the magnetic
    field when doing simultaneous X-ray and gamma-ray spectral fits.

    Parameters
    ----------

    xray_table : :class:`~astropy.table.Table`
        Data table (see :ref:`dataformat` for details on the format) containing
        the X-ray spectrum.
    vhe_table : :class:`~astropy.table.Table`
        Data table (see :ref:`dataformat` for details on the format) containing
        the HE/VHE gamma-ray spectrum.

    photon_energy_density : :class:`~astropy.units.Quantity` float, optional
        Energy density of the seed photon field for IC emission. Defaults to
        0.261 eV/cm3, the energy density of the CMB.

    Returns
    -------

    B : :class:`~astropy.units.Quantity` float
        Estimate of the magnetic flux density at the emitter.
    """

    xray = validate_data_table(xray_table, sed=False)
    vhe = validate_data_table(vhe_table, sed=False)

    xray_lum = trapz_loglog(xray['flux'] * xray['energy'], xray['energy'])
    vhe_lum = trapz_loglog(vhe['flux'] * vhe['energy'], vhe['energy'])

    uph = (photon_energy_density.to('erg/cm3')).value

    B0 = (np.sqrt((xray_lum / vhe_lum).decompose().value * 8 * np.pi * uph) *
          u.G).to('uG')

    return B0
