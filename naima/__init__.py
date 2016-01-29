# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
`naima` is a Python package for computation of non-thermal radiation from
relativistic particle populations. It includes tools to perform MCMC fitting of
radiative models to X-ray, GeV, and TeV spectra using `~emcee`, an
affine-invariant ensemble sampler for Markov Chain Monte Carlo.  `naima` uses
MCMC fitting of non-thermal X-ray, GeV, and TeV spectra to constrain the
properties of their parent relativistic particle distributions.

"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

from .core import *
from .plot import *
from .utils import *
from .analysis import *
from .model_fitter import *

from . import models

try:
    import sherpa
    from . import sherpa_models
except ImportError:
    pass
