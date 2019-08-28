# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
from glob import glob

import astropy.units as u
import numpy as np
from astropy.io import ascii
from astropy.tests.helper import pytest
from astropy.utils.data import get_pkg_data_filename

from ..analysis import save_diagnostic_plots, save_results_table
from ..core import run_sampler, uniform_prior
from ..plot import plot_chain, plot_data, plot_fit
from .fixtures import sampler

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import emcee

    HAS_EMCEE = True
except ImportError:
    HAS_EMCEE = False


# Read data
fname = get_pkg_data_filename("data/CrabNebula_HESS_ipac.dat")
data_table = ascii.read(fname)

fname2 = get_pkg_data_filename("data/CrabNebula_Fake_Xray.dat")
data_table2 = ascii.read(fname2)
data_list = [data_table2, data_table]


@pytest.mark.skipif("not HAS_MATPLOTLIB or not HAS_EMCEE")
@pytest.mark.parametrize("last_step", [True, False])
@pytest.mark.parametrize("convert_log", [True, False])
@pytest.mark.parametrize("include_blobs", [True, False])
@pytest.mark.parametrize("format", ["ascii.ipac", "ascii.ecsv", "ascii"])
def test_results_table(sampler, last_step, convert_log, include_blobs, format):
    # set one keyword to a numpy array to try an break ecsv
    sampler.run_info["test"] = np.random.randn(3)
    save_results_table(
        "test_table",
        sampler,
        convert_log=convert_log,
        last_step=last_step,
        format=format,
        include_blobs=include_blobs,
    )

    os.unlink(glob("test_table_results*")[0])


@pytest.mark.skipif("not HAS_MATPLOTLIB or not HAS_EMCEE")
@pytest.mark.parametrize("last_step", [True, False])
@pytest.mark.parametrize("p", [None, 1])
def test_chain_plots(sampler, last_step, p):
    plot_chain(sampler, last_step=last_step, p=p)
    plt.close("all")


@pytest.mark.skipif("not HAS_MATPLOTLIB or not HAS_EMCEE")
@pytest.mark.parametrize("idx", range(4))
@pytest.mark.parametrize("sed", [True, False])
@pytest.mark.parametrize("last_step", [True, False])
@pytest.mark.parametrize("confs", [[1, 2], None])
@pytest.mark.parametrize("n_samples", [100, None])
@pytest.mark.parametrize("e_range", [[1 * u.GeV, 100 * u.TeV], None])
def test_fit_plots(sampler, idx, sed, last_step, confs, n_samples, e_range):
    # plot models with correct format
    plot_fit(
        sampler,
        modelidx=idx,
        sed=sed,
        last_step=last_step,
        plotdata=True,
        confs=confs,
        n_samples=n_samples,
        e_range=e_range,
    )
    plt.close("all")


@pytest.mark.skipif("not HAS_MATPLOTLIB or not HAS_EMCEE")
@pytest.mark.parametrize("threads", [None, 1, 4])
def test_threads_in_samples(sampler, threads):
    plot_fit(
        sampler,
        n_samples=100,
        threads=threads,
        e_range=[1 * u.GeV, 100 * u.TeV],
        e_npoints=20,
    )
    plt.close("all")


@pytest.mark.skipif("not HAS_MATPLOTLIB or not HAS_EMCEE")
@pytest.mark.parametrize("sed", [True, False])
def test_plot_data(sampler, sed):
    plot_data(sampler, sed=sed)
    plt.close("all")


@pytest.mark.skipif("not HAS_MATPLOTLIB or not HAS_EMCEE")
def test_plot_data_reuse_fig(sampler):
    # change the energy units between calls
    data = sampler.data
    fig = plot_data(data, sed=True)
    data["energy"] = (data["energy"] * 1000).to("keV")
    plot_data(data, sed=True, figure=fig)
    plt.close("all")


@pytest.mark.skipif("not HAS_MATPLOTLIB or not HAS_EMCEE")
@pytest.mark.parametrize("data_tables", [data_table, data_table2, data_list])
def test_plot_data_tables(sampler, data_tables):
    plot_data(data_tables)
    plt.close("all")


@pytest.mark.skipif("not HAS_MATPLOTLIB or not HAS_EMCEE")
def test_fit_data_units(sampler):
    plot_fit(sampler, modelidx=0, sed=None)
    plt.close("all")


@pytest.mark.skipif("not HAS_MATPLOTLIB or not HAS_EMCEE")
def test_diagnostic_plots(sampler):
    # Diagnostic plots
    # try to plot all models, including those with wrong format/units

    blob_labels = [
        "Model",
        "Flux",
        "Model",
        "Particle Distribution",
        "Broken PL",
        "Wrong",
        "Wrong",
        "Wrong",
        "Scalar",
        "Scalar without units",
    ]

    save_diagnostic_plots("test_function_1", sampler, blob_labels=blob_labels)
    save_diagnostic_plots(
        "test_function_2", sampler, sed=True, blob_labels=blob_labels[:4]
    )
    save_diagnostic_plots("test_function_3", sampler, pdf=True)
