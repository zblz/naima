# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np

import astropy.units as u
from astropy.tests.helper import pytest
from astropy.utils.data import get_pkg_data_filename
from astropy.extern import six
from astropy.io import ascii
import os

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

from ..analysis import save_run, read_run
from ..plot import plot_data, plot_fit, plot_chain
from ..model_fitter import InteractiveModelFitter
from ..utils import validate_data_table
from ..core import run_sampler, uniform_prior
from ..models import ExponentialCutoffPowerLaw

fname = get_pkg_data_filename('data/CrabNebula_HESS_ipac.dat')
data_table = ascii.read(fname)

def cutoffexp(pars, data):
    x = data['energy'].copy()
    ECPL = ExponentialCutoffPowerLaw(np.exp(pars[0])*u.Unit('1/(cm2 s TeV)'),
            1*u.TeV, pars[1], 10**pars[2]*u.TeV)
    flux = ECPL(x)

    # save a particle energy distribution
    ene = np.logspace(np.log10(x[0].value) - 1,
                      np.log10(x[-1].value) + 1, 100) * x.unit
    ECPL.amplitude = np.exp(pars[0])*u.Unit('1/(TeV)')
    model_part = ECPL(ene)

    # add a scalar value to test plot_distribution
    model4 = np.trapz(flux*x,x).to('erg/(cm2 s)')
    # and without units
    model5 = model4.value

    return flux, (ene, model_part), model4, model5

# Prior definition
def lnprior(pars):
    logprob = uniform_prior(pars[1], -1, 5)
    return logprob

# Set initial parameters
p0=np.array((np.log(1.8e-12),2.4,np.log10(15.0),))
labels=['log(norm)','index','log10(cutoff)']

# Run sampler
@pytest.fixture(scope='module')
def sampler():
    sampler, pos = run_sampler(
        data_table=data_table, p0=p0, labels=labels, model=cutoffexp,
        prior=lnprior, nwalkers=10, nburn=2, nrun=10, threads=1)
    return sampler

@pytest.mark.skipif('not HAS_EMCEE')
def test_roundtrip(sampler):
    save_run('test_chain.h5', sampler, clobber=True)
    assert os.path.exists('test_chain.h5')
    nresult = read_run('test_chain.h5')

    assert np.allclose(sampler.chain, nresult.chain)
    assert np.allclose(sampler.flatchain, nresult.flatchain)
    assert np.allclose(sampler.lnprobability, nresult.lnprobability)
    assert np.allclose(sampler.flatlnprobability, nresult.flatlnprobability)

    nwalkers, nsteps = sampler.chain.shape[:2]
    j, k = int(nsteps/2), int(nwalkers/2)
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
        assert np.allclose(u.Quantity(sampler.data[col]).value,
                u.Quantity(nresult.data[col]).value)
        assert str(sampler.data[col].unit) == str(nresult.data[col].unit)
    validate_data_table(nresult.data)

    assert np.allclose(np.mean(sampler.acceptance_fraction),
            nresult.acceptance_fraction)


@pytest.mark.skipif('not HAS_MATPLOTLIB or not HAS_EMCEE')
def test_plot_fit(sampler):
    nresult = read_run('test_chain.h5', modelfn=sampler.modelfn)

    f = plot_data(nresult)
    f = plot_fit(nresult, 0)
    f = plot_fit(nresult, 0, e_range=[0.1,10]*u.TeV)
    f = plot_fit(nresult, 0, sed=False)

@pytest.mark.skipif('not HAS_MATPLOTLIB or not HAS_EMCEE')
def test_plot_chain(sampler):
    nresult = read_run('test_chain.h5', modelfn=sampler.modelfn)

    for i in range(nresult.chain.shape[2]):
        f = plot_chain(nresult, i)

@pytest.mark.skipif('not HAS_MATPLOTLIB or not HAS_EMCEE')
def test_imf(sampler):
    nresult = read_run('test_chain.h5', modelfn=sampler.modelfn)

    imf = InteractiveModelFitter(nresult.modelfn, nresult.chain[-1][-1],
            nresult.data)
    imf.do_fit('test')
    from naima.core import lnprobmodel
    lnprobmodel(nresult.modelfn(imf.pars,nresult.data)[0], nresult.data)
