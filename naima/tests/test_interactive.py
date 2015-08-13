# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np

import astropy.units as u
from astropy.tests.helper import pytest
from astropy.utils.data import get_pkg_data_filename
from astropy.extern import six
from astropy.io import ascii

try:
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except:
    HAS_MATPLOTLIB = False

from ..models import ExponentialCutoffPowerLaw
from ..model_fitter import InteractiveModelFitter

# Read data
fname = get_pkg_data_filename('data/CrabNebula_HESS_ipac.dat')
data = ascii.read(fname)

def modelfn(pars,data):
    ECPL = ExponentialCutoffPowerLaw(10**pars[0] * u.Unit('1/(cm2 s TeV)'), 1*u.TeV,
            pars[1], 10**pars[2] * u.TeV)
    return ECPL(data)

def modelfn2(pars, data):
    return modelfn(pars, data), (1,2,3)*u.m

labels = ['log10(norm)', 'index', 'log10(cutoff)']
p0 = np.array((-12, 2.7, np.log10(14)))

e_range = [100*u.GeV, 100*u.TeV]

@pytest.mark.skipif('not HAS_MATPLOTLIB')
def test_modelwidget_inputs():
    for dt in [data, None]:
        for er in [e_range, None]:
            for model in [modelfn, modelfn2]:
                imf = InteractiveModelFitter(model, p0, labels=labels,
                        data=dt, e_range=er)
                imf.update('test')

    for labs in [labels, labels[:2], None]:
        imf = InteractiveModelFitter(model, p0, labels=labs)
    for sed in [True, False]:
        for dt in [data, None]:
            imf = InteractiveModelFitter(model, p0, data=dt, labels=labels, sed=sed)
    p0[1] = -2.7
    imf = InteractiveModelFitter(model, p0, labels=labels)
    labels[0] = 'norm'
    imf = InteractiveModelFitter(model, p0, labels=labels)

@pytest.mark.skipif('not HAS_MATPLOTLIB')
def test_modelwidget_funcs():
    imf = InteractiveModelFitter(modelfn, p0, data=data, labels=labels, auto_update=False)
    assert imf.autoupdate is False
    imf.update_autoupdate('test')
    assert imf.autoupdate is True
    imf.parsliders[0].val *= 2
    imf.update_if_auto('test')
    imf.close_fig('test')

    imf = InteractiveModelFitter(modelfn, p0, labels=labels, auto_update=False)
    imf.update('test')

