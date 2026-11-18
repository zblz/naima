# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import ascii
from astropy.tests.helper import pytest
from numpy.testing import assert_allclose

from naima.utils import (
    build_data_table,
    estimate_B,
    generate_energy_edges,
    trapz_loglog,
    validate_data_table,
)

# Read data
fname = Path(__file__).parent / "data/CrabNebula_HESS_ipac.dat"
data_table = ascii.read(fname)

# Read spectrum with symmetric flux errors
fname_sym = Path(__file__).parent / "data/CrabNebula_HESS_ipac_symmetric.dat"
data_table_sym = ascii.read(fname_sym)


def test_validate_energy_error_types():
    for etype in ["edges", "error", "width", "errors"]:
        fname = Path(__file__).parent / f"data/CrabNebula_HESS_ipac_energy_{etype}.dat"

        dt = ascii.read(fname)
        validate_data_table(dt)


def test_sed():
    fname = Path(__file__).parent / "data/Fake_ipac_sed.dat"
    validate_data_table(ascii.read(fname))
    validate_data_table([ascii.read(fname)])


def test_concatenation():
    fname0 = Path(__file__).parent / "data/Fake_ipac_sed.dat"
    dt0 = ascii.read(fname0)

    for sed in [True, False]:
        validate_data_table([dt0, data_table], sed=sed)
        validate_data_table([data_table, dt0], sed=sed)
        validate_data_table([dt0, dt0], sed=sed)


def test_validate_data_types():
    data_table2 = data_table.copy()
    data_table2["energy"].unit = ""
    with pytest.raises(TypeError):
        validate_data_table(data_table2)


def test_validate_missing_column():
    data_table2 = data_table.copy()
    data_table2.remove_column("energy")
    with pytest.raises(TypeError):
        validate_data_table(data_table2)
    data_table2 = data_table_sym.copy()
    data_table2.remove_column("flux_error")
    with pytest.raises(TypeError):
        validate_data_table(data_table2)


def test_validate_string_uls():
    from astropy.table import Column

    data_table2 = data_table.copy()

    # replace uls column with valid strings
    data_table2.remove_column("ul")
    data_table2.add_column(
        Column(name="ul", dtype=str, data=["False"] * len(data_table2))
    )
    data_table2["ul"][1] = "True"

    data = validate_data_table(data_table2)

    assert np.sum(data["ul"]) == 1
    assert np.sum(~data["ul"]) == len(data_table2) - 1

    # put an invalid value
    data_table2["ul"][2] = "foo"

    with pytest.raises(TypeError):
        validate_data_table(data_table2)


def test_validate_cl():
    data_table2 = data_table.copy()

    # use invalid value
    data_table2.meta["keywords"]["cl"]["value"] = "test"
    with pytest.raises(TypeError):
        data = validate_data_table(data_table2)

    # remove cl
    data_table2.meta["keywords"].pop("cl")
    data = validate_data_table(data_table2)
    assert np.all(data["cl"] == 0.9)


def test_build_data_table():
    ene = np.logspace(-2, 2, 20) * u.TeV
    flux = (ene / (1 * u.TeV)) ** -2 * u.Unit("1/(cm2 s TeV)")
    flux_error_hi = 0.2 * flux
    flux_error_lo = 0.1 * flux
    ul = np.zeros(len(ene))
    ul[0] = 1

    dene = generate_energy_edges(ene)

    build_data_table(
        ene,
        flux,
        flux_error_hi=flux_error_hi,
        flux_error_lo=flux_error_lo,
        ul=ul,
    )
    build_data_table(
        ene,
        flux,
        flux_error_hi=flux_error_hi,
        flux_error_lo=flux_error_lo,
        ul=ul,
        cl=0.99,
    )
    build_data_table(ene, flux, flux_error=flux_error_hi, energy_width=dene[0])
    build_data_table(
        ene,
        flux,
        flux_error=flux_error_hi,
        energy_lo=(ene - dene[0]),
        energy_hi=(ene + dene[1]),
    )

    # no flux_error
    with pytest.raises(TypeError):
        build_data_table(ene, flux)

    # errors in energy physical type validation
    with pytest.raises(TypeError):
        build_data_table(ene.value, flux, flux_error=flux_error_hi)

    with pytest.raises(TypeError):
        build_data_table(
            ene.value * u.Unit("erg/(cm2 s)"), flux, flux_error=flux_error_hi
        )


def test_estimate_B():
    fname = Path(__file__).parent / "data/CrabNebula_Fake_Xray.dat"
    xray = ascii.read(fname)

    B = estimate_B(xray, data_table)

    assert_allclose(B.to("uG"), 0.4848756912803697 * u.uG)


def test_reversed_table_order():
    """Regression test for #247: table order should not affect energy sorting."""
    from astropy.table import QTable

    t_lo = QTable()
    t_lo["energy"] = [1, 2, 3] * u.keV
    t_lo["flux"] = [1e-11, 8e-12, 5e-12] * u.Unit("erg/(cm2 s)")
    t_lo["flux_error"] = [1e-12, 1e-12, 1e-12] * u.Unit("erg/(cm2 s)")

    t_hi = QTable()
    t_hi["energy"] = [1, 2, 3] * u.TeV
    t_hi["flux"] = [1e-12, 8e-13, 5e-13] * u.Unit("erg/(cm2 s)")
    t_hi["flux_error"] = [1e-13, 1e-13, 1e-13] * u.Unit("erg/(cm2 s)")

    data_lohi = validate_data_table([t_lo, t_hi])
    data_hilo = validate_data_table([t_hi, t_lo])

    # Energies must be sorted regardless of input order
    assert np.all(np.diff(data_lohi["energy"]) > 0)
    assert np.all(np.diff(data_hilo["energy"]) > 0)

    # Energy and flux values must match (units may differ based on first table)
    assert_allclose(
        data_lohi["energy"].to(u.keV).value, data_hilo["energy"].to(u.keV).value
    )
    assert_allclose(
        data_lohi["flux"].to(data_lohi["flux"].unit).value,
        data_hilo["flux"].to(data_lohi["flux"].unit).value,
    )


def test_trapz_loglog_steep_bin():
    """
    A bin with an extremely steep local power-law index must not overflow
    (https://github.com/zblz/naima/issues/263)
    """
    x = np.array([1.0, 1.001, 1.002, 1.003])
    y = np.array([0.0, 1e-250, 1e250, 1e255])
    result = trapz_loglog(y, x)
    assert np.isfinite(result)

    # steep bin integral is y2 * x2 / (b + 1), negligible next to the last bin
    b = (np.log(y[2]) - np.log(y[1])) / np.log(x[2] / x[1])
    steep = trapz_loglog(y[1:3], x[1:3])
    assert_allclose(steep, y[2] * x[2] / (b + 1), rtol=1e-6)


@pytest.mark.parametrize(
    "y, x",
    [
        # y2/y1 overflows
        ([1e-200, 1e200], [1.0, 2.0]),
        # y2/y1 underflows
        ([1e200, 1e-200], [1.0, 2.0]),
    ],
)
def test_trapz_loglog_no_overflow(y, x):
    y, x = np.array(y), np.array(x)
    # the bin is dominated by the larger endpoint: y x / |b + 1|
    b = (np.log(y[1]) - np.log(y[0])) / np.log(x[1] / x[0])
    imax = np.argmax(y)
    assert_allclose(trapz_loglog(y, x), y[imax] * x[imax] / abs(b + 1), rtol=1e-10)


def test_trapz_loglog_no_product_overflow():
    """y x overflows although the integral over the narrow bin does not"""
    x = np.array([1e10, 1e10 * (1 + 1e-12)])
    y = np.array([1e300, 1e300])
    assert_allclose(trapz_loglog(y, x), 1e300 * (x[1] - x[0]), rtol=1e-6)


def test_trapz_loglog_index_near_minus_one():
    """The integral is continuous and precise across a local index of -1"""
    x = np.array([1.0, 2.0])
    for eps in [0.0, 1e-12, -1e-12, 1e-9, 1e-6]:
        b = -1.0 + eps
        y = x**b
        expected = np.log(2.0) if eps == 0 else np.expm1(eps * np.log(2.0)) / eps
        assert_allclose(trapz_loglog(y, x), expected, rtol=1e-12)


def test_trapz_loglog_negative_and_zero():
    x = np.array([1.0, 2.0, 3.0])
    assert_allclose(trapz_loglog(-(x**2), x), -26.0 / 3, rtol=1e-12)
    # the bin touching zero is dropped, the other is y = x - 1 as a power law
    b = np.log(2.0) / np.log(1.5)
    assert_allclose(
        trapz_loglog(np.array([0.0, 1.0, 2.0]), x), (6.0 - 2.0) / (b + 1), rtol=1e-12
    )
    # a sign change falls back to y1 x1 log(x2/x1), as before
    assert_allclose(trapz_loglog(np.array([-1.0, 1.0]), x[:2]), -np.log(2.0))


@pytest.mark.parametrize("ratio", [0.9, 1e-3, 1e-9])
def test_trapz_loglog_decreasing_x(ratio):
    """Integrating towards lower x, over narrow and wide bins"""
    x = np.array([1.0, ratio])
    # integral of x**-2 from 1 to ratio
    assert_allclose(trapz_loglog(x**-2, x), 1.0 - 1.0 / ratio, rtol=1e-12)


def test_trapz_loglog_axis():
    x = np.logspace(0, 2, 5)
    y = np.vstack([x**-2, 3 * x**0.5, x**-1])
    expected = [trapz_loglog(row, x) for row in y]
    assert_allclose(trapz_loglog(y, x, axis=1), expected, rtol=1e-14)
    assert_allclose(trapz_loglog(y.T, x, axis=0), expected, rtol=1e-14)
    assert_allclose(trapz_loglog(y.T, x[:, None], axis=0), expected, rtol=1e-14)
