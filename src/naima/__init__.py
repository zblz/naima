# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
`naima` is a Python package for computation of non-thermal radiation from
relativistic particle populations. It includes tools to perform MCMC fitting of
radiative models to X-ray, GeV, and TeV spectra using `~emcee`, an
affine-invariant ensemble sampler for Markov Chain Monte Carlo.  `naima` uses
MCMC fitting of non-thermal X-ray, GeV, and TeV spectra to constrain the
properties of their parent relativistic particle distributions.

"""

from . import models
from .analysis import *
from .core import *
from .model_fitter import *
from .plot import *
from .utils import *

from .version import get_version

__version__ = get_version()
del get_version


try:
    import sherpa
    from . import sherpa_models
except ImportError:
    pass
