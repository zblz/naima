# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.tests.helper import pytest
from ..utils import trapz_loglog

try:
    from sherpa import ui
    HAS_SHERPA = True
except ImportError:
    HAS_SHERPA = False

energies = np.logspace(8, 10, 10)  # 0.1 to 10 TeV in keV
test_spec_points = (1e-20 * (energies / 1e9) ** -0.7 *
                    (1 + 0.2 * np.random.randn(energies.size)))
test_err_points = 0.2 * test_spec_points

elo = energies[:-1]
ehi = energies[1:]
test_spec_int = trapz_loglog(test_spec_points, energies, intervals=True)
test_err_int = 0.2 * test_spec_int


@pytest.mark.skipif('not HAS_SHERPA')
def test_electron_models():
    """
    test import
    """

    from ..sherpa_models import InverseCompton, Synchrotron, Bremsstrahlung

    for modelclass in [InverseCompton, Synchrotron, Bremsstrahlung]:
        model = modelclass()

        model.ampl = 1e-8
        model.index = 2.1

        print(model)

        # point calc
        output = model.calc([p.val for p in model.pars], energies)

        # test as well ECPL
        model.cutoff = 100

        # integrated
        output = model.calc([p.val for p in model.pars], elo, xhi=ehi)

        if modelclass is InverseCompton:
            # Perform a fit to fake data
            ui.load_arrays(1, energies, test_spec_points, test_err_points)
            ui.set_model(model)
            ui.guess()
            ui.fit()

            # add FIR and NIR components and test verbose
            model.uNIR.set(1.0)
            model.uFIR.set(1.0)
            model.verbose.set(1)

            # test with integrated data
            ui.load_arrays(1, elo, ehi, test_spec_int, test_err_int,
                           ui.Data1DInt)
            ui.set_model(model)
            ui.guess()
            ui.fit()


@pytest.mark.skipif('not HAS_SHERPA')
def test_proton_model():
    """
    test import
    """

    from ..sherpa_models import PionDecay

    model = PionDecay()

    model.ampl = 1e36
    model.index = 2.1

    # point calc
    output = model.calc([p.val for p in model.pars], energies)

    # integrated
    output = model.calc([p.val for p in model.pars], elo, xhi=ehi)

    # test as well ECPL
    model.cutoff = 1000

    # Perform a fit to fake data
    ui.load_arrays(1, energies, test_spec_points, test_err_points)
    ui.set_model(model)
    ui.guess()
    # Actual fit is too slow for tests
    # ui.fit()

    # test with integrated data
    ui.load_arrays(1, elo, ehi, test_spec_int, test_err_int, ui.Data1DInt)
    ui.set_model(model)
    ui.guess()
    # Actual fit is too slow for tests
    # ui.fit()
