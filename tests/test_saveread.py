# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import ascii
from astropy.tests.helper import pytest

from naima.analysis import read_run, save_run
from naima.model_fitter import InteractiveModelFitter
from naima.plot import plot_chain, plot_data, plot_fit
from naima.utils import validate_data_table

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import emcee  # noqa

    HAS_EMCEE = True
except ImportError:
    HAS_EMCEE = False


fname = Path(__file__).parent / "data/CrabNebula_HESS_ipac.dat"
data_table = ascii.read(str(fname))


@pytest.mark.skipif("not HAS_EMCEE")
def test_roundtrip(simple_sampler, tmp_path):
    sampler = simple_sampler
    filename = tmp_path / "naima_test_sampler.hdf5"
    save_run(filename, sampler)
    assert os.path.exists(filename)
    nresult = read_run(filename)

    assert np.allclose(sampler.get_chain(), nresult.get_chain())
    assert np.allclose(sampler.get_chain(flat=True), nresult.get_chain(flat=True))
    assert np.allclose(sampler.get_log_prob(), nresult.get_log_prob())
    assert np.allclose(sampler.get_log_prob(flat=True), nresult.get_log_prob(flat=True))

    nwalkers, nsteps = sampler.get_chain().shape[:2]
    sampler_blobs = sampler.get_blobs()
    new_blobs = nresult.get_blobs()
    assert sampler_blobs.shape == new_blobs.shape
    j, k = nwalkers // 2, nsteps // 2
    for l_index in range(len(sampler_blobs[j][k])):
        b0 = sampler_blobs[j][k][l_index]
        b1 = new_blobs[j][k][l_index]
        if isinstance(b0, tuple) or isinstance(b0, list):
            for b0m, b1m in zip(b0, b1):
                assert np.allclose(b0m, b1m)
        else:
            assert np.allclose(b0, b1)

    for key in sampler.run_info.keys():
        assert np.all(sampler.run_info[key] == nresult.run_info[key])

    for i in range(len(sampler.labels)):
        assert sampler.labels[i] == nresult.labels[i]

    for col in sampler.data.colnames:
        assert np.allclose(
            u.Quantity(sampler.data[col]).value,
            u.Quantity(nresult.data[col]).value,
        )
        assert str(sampler.data[col].unit) == str(nresult.data[col].unit)
    validate_data_table(nresult.data)

    assert np.allclose(
        np.mean(sampler.acceptance_fraction), nresult.acceptance_fraction
    )


@pytest.mark.skipif("not HAS_MATPLOTLIB or not HAS_EMCEE")
def test_plot_fit(simple_sampler, tmp_path):
    filename = tmp_path / "naima_test_sampler.hdf5"
    save_run(filename, simple_sampler, clobber=True)
    nresult = read_run(filename, modelfn=simple_sampler.modelfn)

    plot_data(nresult)
    plot_fit(nresult, 0)
    plot_fit(nresult, 0, e_range=[0.1, 10] * u.TeV)
    plot_fit(nresult, 0, sed=False)
    plt.close("all")


@pytest.mark.skipif("not HAS_MATPLOTLIB or not HAS_EMCEE")
def test_plot_chain(simple_sampler, tmp_path):
    filename = tmp_path / "naima_test_sampler.hdf5"
    save_run(filename, simple_sampler, clobber=True)
    nresult = read_run(filename, modelfn=simple_sampler.modelfn)

    for i in range(nresult.get_chain().shape[2]):
        plot_chain(nresult, i)
    plt.close("all")


@pytest.mark.skipif("not HAS_MATPLOTLIB or not HAS_EMCEE")
def test_imf(simple_sampler, tmp_path):
    filename = tmp_path / "naima_test_sampler.hdf5"
    save_run(filename, simple_sampler, clobber=True)
    nresult = read_run(filename, modelfn=simple_sampler.modelfn)

    imf = InteractiveModelFitter(
        nresult.modelfn, nresult.get_chain()[-1][-1], nresult.data
    )
    imf.do_fit("test")
    from naima.core import lnprobmodel

    lnprobmodel(nresult.modelfn(imf.pars, nresult.data)[0], nresult.data)
    plt.close("all")
