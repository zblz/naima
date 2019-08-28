# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os

import astropy.units as u
import numpy as np
from astropy.io import ascii
from astropy.tests.helper import pytest
from astropy.utils.data import get_pkg_data_filename

from ..analysis import read_run, save_run
from ..core import run_sampler, uniform_prior
from ..model_fitter import InteractiveModelFitter
from ..models import ExponentialCutoffPowerLaw
from ..plot import plot_chain, plot_data, plot_fit
from ..utils import validate_data_table
from .fixtures import simple_sampler as sampler

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except:
    HAS_MATPLOTLIB = False

try:
    import emcee

    HAS_EMCEE = True
except:
    HAS_EMCEE = False


fname = get_pkg_data_filename("data/CrabNebula_HESS_ipac.dat")
data_table = ascii.read(fname)


@pytest.mark.skipif("not HAS_EMCEE")
def test_roundtrip(sampler):
    save_run("test_chain.h5", sampler, clobber=True)
    assert os.path.exists("test_chain.h5")
    nresult = read_run("test_chain.h5")

    assert np.allclose(sampler.chain, nresult.chain)
    assert np.allclose(sampler.flatchain, nresult.flatchain)
    assert np.allclose(sampler.lnprobability, nresult.lnprobability)
    assert np.allclose(sampler.flatlnprobability, nresult.flatlnprobability)

    nwalkers, nsteps = sampler.chain.shape[:2]
    j, k = int(nsteps / 2), int(nwalkers / 2)
    for l in range(len(sampler.blobs[j][k])):
        b0 = sampler.blobs[j][k][l]
        b1 = nresult.blobs[j][k][l]
        if isinstance(b0, tuple) or isinstance(b0, list):
            for m in range(len(b0)):
                assert b0[m].unit == b1[m].unit
                assert np.allclose(b0[m].value, b1[m].value)
        else:
            if isinstance(b0, u.Quantity):
                assert b0.unit == b1.unit
                assert np.allclose(b0.value, b1.value)
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
def test_plot_fit(sampler):
    save_run("test_chain.h5", sampler, clobber=True)
    nresult = read_run("test_chain.h5", modelfn=sampler.modelfn)

    plot_data(nresult)
    plot_fit(nresult, 0)
    plot_fit(nresult, 0, e_range=[0.1, 10] * u.TeV)
    plot_fit(nresult, 0, sed=False)
    plt.close("all")


@pytest.mark.skipif("not HAS_MATPLOTLIB or not HAS_EMCEE")
def test_plot_chain(sampler):
    save_run("test_chain.h5", sampler, clobber=True)
    nresult = read_run("test_chain.h5", modelfn=sampler.modelfn)

    for i in range(nresult.chain.shape[2]):
        plot_chain(nresult, i)
    plt.close("all")


@pytest.mark.skipif("not HAS_MATPLOTLIB or not HAS_EMCEE")
def test_imf(sampler):
    save_run("test_chain.h5", sampler, clobber=True)
    nresult = read_run("test_chain.h5", modelfn=sampler.modelfn)

    imf = InteractiveModelFitter(
        nresult.modelfn, nresult.chain[-1][-1], nresult.data
    )
    imf.do_fit("test")
    from naima.core import lnprobmodel

    lnprobmodel(nresult.modelfn(imf.pars, nresult.data)[0], nresult.data)
    plt.close("all")
