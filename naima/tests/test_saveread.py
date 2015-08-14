# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np

import astropy.units as u
from astropy.tests.helper import pytest
from astropy.utils.data import get_pkg_data_filename
from astropy.extern import six

try:
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except:
    HAS_MATPLOTLIB = False

try:
    import emcee
    HAS_EMCEE = True
except:
    HAS_EMCEE = False

from ..analysis import save_chain, read_chain
from ..plot import plot_data, plot_fit, plot_chain
from ..model_fitter import InteractiveModelFitter

from .test_plotting import sampler

@pytest.mark.skipif('not HAS_EMCEE')
def test_roundtrip(sampler):
    save_chain('test', sampler)
    nresult = read_chain('test_chain.h5')

    assert np.allclose(sampler.chain, nresult.chain)
    assert np.allclose(sampler.flatchain, nresult.flatchain)
    assert np.allclose(sampler.lnprobability, nresult.lnprobability)
    assert np.allclose(sampler.flatlnprobability, nresult.flatlnprobability)

    for key in sampler.run_info.keys():
        assert np.all(sampler.run_info[key] == nresult.run_info[key])

    assert np.allclose(sampler.acceptance_fraction, nresult.acceptance_fraction)


@pytest.mark.skipif('not HAS_MATPLOTLIB or not HAS_EMCEE')
def test_plot_fit(sampler):
    nresult = read_chain('test_chain.h5', data=sampler.data,
            modelfn=sampler.modelfn, labels=sampler.labels)

    f = plot_data(nresult)
    f = plot_fit(nresult, 0)
    f = plot_fit(nresult, 0, e_range=[0.1,10]*u.TeV)
    f = plot_fit(nresult, 0, sed=False)

@pytest.mark.skipif('not HAS_MATPLOTLIB or not HAS_EMCEE')
def test_plot_chain(sampler):
    nresult = read_chain('test_chain.h5', data=sampler.data,
            modelfn=sampler.modelfn, labels=sampler.labels)

    for i in range(nresult.chain.shape[2]):
        f = plot_chain(nresult, i)

@pytest.mark.skipif('not HAS_MATPLOTLIB or not HAS_EMCEE')
def test_imf(sampler):
    nresult = read_chain('test_chain.h5', data=sampler.data,
            modelfn=sampler.modelfn, labels=sampler.labels)

    imf = InteractiveModelFitter(nresult.modelfn, nresult.chain[-1][-1],
            nresult.data)
    imf.do_fit('test')
    from naima.core import lnprobmodel
    lnprobmodel(nresult.modelfn(imf.pars,nresult.data)[0], nresult.data)
