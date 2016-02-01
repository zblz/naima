.. _MCMC:

Model fitting
=============

Naima can derive the best-fit and uncertainty distributions of spectral
model parameters through Markov Chain Monte Carlo (MCMC) sampling of their
likelihood distributions. The following will only describe the implementation of
Naima to do so, but a full explanation of MCMC or the sampling algorithm can
be found in `MacKay (2003)
<http://www.inference.phy.cam.ac.uk/mackay/itila/book.html>`_, and
`Foreman-Mackey et al. (2013) <http://arxiv.org/abs/1202.3665>`_. It is also
advisable to consult `the documentation for the emcee package
<http://dan.iel.fm/emcee/current/>`_, which is used for the MCMC sampling.

If you use the MCMC fitting in your research, please cite the ``emcee`` package
through the publication `Foreman-Mackey, 
D., Hogg, D.W., Lang, D., & Goodman, J. 2013, PASP, 125, 306
<http://adsabs.harvard.edu/abs/2013PASP..125..306F>`_.

The measurements and uncertainties in the provided spectrum are assumed to be
correct, Gaussian, and independent (note that this is unlikely to be the case,
see `Overcoming the Gaussian error assumption`_ on how this might be tackled in
the future).  Under this assumption, the likelihood of observed data given the
spectral model :math:`S(\vec{p};E)`, for a parameter vector :math:`\vec{p}`, is

.. math::
    \mathcal{L} = \prod^N_{i=1} \frac{1}{\sqrt{2 \pi \sigma^2_i}} 
                \exp\left(-\frac{(S(\vec{p};E_i) - F_i)^2}{2\sigma^2_i}\right),

where :math:`(F_i,\sigma_i)` are the flux measurement and uncertainty at an
energy :math:`E_i` over :math:`N` spectral measurements. Taking the logarithm,

.. math::
    \ln\mathcal{L} = K - \sum^N_{i=1} \frac{(S(\vec{p};E_i) - F_i)^2}{2\sigma^2_i}.

Given that the MCMC procedure will sample the areas of the distribution with
maximum value of the objective function, it is useful to define the objective
function as the log-likelihood disregarding constant factors:

.. math::
    \ln\mathcal{L} \propto  \sum^N_{i=1} \frac{(S(\vec{p};E_i) - F_i)^2}{\sigma^2_i}.

The :math:`\ln\mathcal{L}` function in this assumption can be related to the
:math:`\chi^2` parameter as :math:`\chi^2=-2\ln\mathcal{L}`, so that
maximization of the log-likelihood is equivalent to a minimization of
:math:`\chi^2`.

In addition to the likelihood from the observed spectral points, a prior
likelihood factor should be considered for all parameters. This prior likelihood
encodes our prior knowledge of the probability distribution of a given model
parameter. If a given parameter is constrained by a previous measurement, it can
be considered using a normal distribution (`naima.normal_prior`). If you need to
constrain a parameter to be within a certain range, a uniform prior can be used
(`naima.uniform_prior`). For parameters expected to have a flat prior in
log-space (e.g., normalizations, cutoff energies, etc.) you can either sample
the logarithm of the parameter or use a log-uniform prior
(`naima.log_uniform_prior`).
    
The combination of the prior and data likelihood functions is passed onto the
`emcee.EnsembleSampler`, and the MCMC run is started. `emcee
<http://dan.iel.fm/emcee/current/>`_ uses an affine-invariant MCMC sampler
(`Goodman & Weare 2010 <http://msp.org/camcos/2010/5-1/p04.xhtml>`_) that has
the advantage of being able to sample complex parameter spaces without any
tuning required. In addition, having multiple simultaneous *walkers* improves
the efficiency of the sampling and reduces the number of
computationally-expensive likelihood calls required.

The sampler works best by using as many samplers as possible, and starting them
in a compact ball around the best fitting parameter values. After a *burn-in*
period (there is no clear answer to many steps it takes for the sampler to
stabilize), the samples that are accepted by the sampler are recorded, along
with the model spectrum, and any metadata blobs provided by the model function
(see :ref:`blobs`). These can then be accessed through the sampler object, and
Naima provides several convenience functions to analyse the results and
compare them to the input spectrum: see :ref:`plotting`.


Overcoming the Gaussian error assumption
----------------------------------------

Naima provides an alternative to MCMC fitting by providing wrappers around
the radiative models that can be used in `sherpa`_. This package allows to take
into account instrument response functions that include bin correlation. See
:ref:`sherpamod` for more details on these sherpa wrappers.

.. _sherpa: http://cxc.cfa.harvard.edu/sherpa/

However, within the framework of MCMC fitting in Naima, several approaches
will be considered for inclusion in the future to overcome the assumption of
correct, Gaussian, independent errors.

- The first is a probabilistic approach to the uncertainty, by including
  generative models for the uncertainties in the NLL function. Such models could
  account for systematics and bin correlation. Their parameters would be fitted
  simultaneously with the model parameters, obtaining model parameter
  distributions that take them into account through marginalisation.
- An alternative approach to avoid bin correlation would be to call an external
  program that can do forward-folding comparison of models. However, doing this
  requires a full set of Instrument Response Functions that might not be
  available for all published data. 
- A proper Poisson statistic could also be used is the fit was performed in
  counts space rather than in flux space. For this, the effective area and
  exposure in each bin would be required to convert between the model flux and
  the expected counts.


How to select which model best fits the data?
---------------------------------------------

Model selection is the method through which a given model is selected from a set
of possible models. You can find a good overview of the process and possible
pitfalls in a `blog post by Jake VanderPlas
<https://jakevdp.github.io/blog/2015/08/07/frequentism-and-bayesianism-5-model-selection/>`_.
In general, computing the `Bayes factor
<https://en.wikipedia.org/wiki/Bayes_factor>`_ for all competing models tends to
be the best way to gauge which model provides a better fit. However, computing
the Bayes Factor is often non-trivial, and a simpler way to obtain an estimate
is using the `Bayesian Information Criterion (BIC)
<https://en.wikipedia.org/wiki/Bayesian_Information_Criterion>`_. The BIC for a
Naima run can be found as a metadata keyword in the results table saved with
`naima.save_results_table`. Note that the BIC is only a valid approximation for
the Bayes factor when the number of datapoints is much larger than the number of
parameters.
