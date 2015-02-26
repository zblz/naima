# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.utils.data import get_pkg_data_filename
import astropy.units as u

from ..utils import validate_data_table, sed_conversion, generate_energy_edges, build_data_table, estimate_B

from astropy.io import ascii

# Read data
fname = get_pkg_data_filename('data/CrabNebula_HESS_ipac.dat')
data_table = ascii.read(fname)

# Read spectrum with symmetric flux errors
fname_sym = get_pkg_data_filename('data/CrabNebula_HESS_ipac_symmetric.dat')
data_table_sym = ascii.read(fname_sym)

def test_validate_energy_error_types():
    for etype in ['edges','error','width','errors']:
        fname = get_pkg_data_filename( 
                'data/CrabNebula_HESS_ipac_energy_{0}.dat'.format(etype))
        dt = ascii.read(fname)
        data = validate_data_table(dt)

def test_sed():
    fname = get_pkg_data_filename('data/Fake_ipac_sed.dat')
    validate_data_table(ascii.read(fname))
    validate_data_table([ascii.read(fname),])

def test_concatenation():
    fname0 = get_pkg_data_filename('data/Fake_ipac_sed.dat')
    dt0 = ascii.read(fname0)

    for sed in [True, False]:
        validate_data_table([dt0,data_table],sed=sed)
        validate_data_table([data_table,dt0],sed=sed)
        validate_data_table([dt0,dt0],sed=sed)

def test_validate_data_types():
    data_table2 = data_table.copy()
    data_table2['energy'].unit=''
    with pytest.raises(TypeError):
        data = validate_data_table(data_table2)

def test_validate_missing_column():
    data_table2 = data_table.copy()
    data_table2.remove_column('energy')
    with pytest.raises(TypeError):
        data = validate_data_table(data_table2)
    data_table2 = data_table_sym.copy()
    data_table2.remove_column('flux_error')
    with pytest.raises(TypeError):
        data = validate_data_table(data_table2)

def test_validate_string_uls():
    from astropy.table import Column
    data_table2 = data_table.copy()

    # replace uls column with valid strings
    data_table2.remove_column('ul')
    data_table2.add_column(Column(name='ul',dtype=str, data=['False',]*len(data_table2)))
    data_table2['ul'][1] = 'True'

    data = validate_data_table(data_table2)

    assert np.sum(data['ul']) == 1
    assert np.sum(-data['ul']) == len(data_table2)-1

    # put an invalid value
    data_table2['ul'][2] = 'invalid'

    with pytest.raises(TypeError):
        data = validate_data_table(data_table2)

def test_validate_cl():
    data_table2 = data_table.copy()

    # use invalid value
    data_table2.meta['keywords']['cl']['value']='test'
    with pytest.raises(TypeError):
        data = validate_data_table(data_table2)

    # remove cl
    data_table2.meta['keywords'].pop('cl')
    data = validate_data_table(data_table2)
    assert np.all(data['cl'] == 0.9)

def test_build_data_table():
    ene = np.logspace(-2,2,20) * u.TeV
    flux = (ene/(1*u.TeV))**-2 * u.Unit('1/(cm2 s TeV)')
    flux_error_hi = 0.2 * flux
    flux_error_lo = 0.1 * flux
    ul = np.zeros(len(ene))
    ul[0] = 1

    dene = generate_energy_edges(ene)

    table = build_data_table(ene, flux, flux_error_hi=flux_error_hi, flux_error_lo=flux_error_lo, ul=ul)
    table = build_data_table(ene, flux, flux_error_hi=flux_error_hi, flux_error_lo=flux_error_lo, ul=ul, cl=0.99)
    table = build_data_table(ene, flux, flux_error=flux_error_hi, energy_width=dene[0])
    table = build_data_table(ene, flux, flux_error=flux_error_hi, energy_lo=(ene-dene[0]), energy_hi=(ene+dene[1]))

    # no flux_error
    with pytest.raises(TypeError):
        table = build_data_table(ene, flux)

    # errors in energy physical type validation
    with pytest.raises(TypeError):
        table = build_data_table(ene.value, flux, flux_error=flux_error_hi)

    with pytest.raises(TypeError):
        table = build_data_table(ene.value*u.Unit('erg/(cm2 s)'), flux, flux_error=flux_error_hi)

def test_estimate_B():

    fname = get_pkg_data_filename('data/CrabNebula_Fake_Xray.dat')
    xray = ascii.read(fname)

    B = estimate_B(xray, data_table)

    assert_allclose(B.to('uG'), 0.4848756912803697 * u.uG)

