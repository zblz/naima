# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
`naima` uses MCMC fitting of non-thermal X-ray, GeV, and TeV spectra
to constrain the properties of their parent relativistic particle distributions.

The workhorse of naima is the powerful `~emcee`
affine-invariant ensemble sampler for Markov chain Monte Carlo.
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

from . import models

try:
    import sherpa
    from . import sherpa_models
except ImportError:
    pass
