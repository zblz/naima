# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy import log

__all__ = ["normal_prior", "uniform_prior", "get_sampler", "run_sampler"]

# Likelihood functions

# Prior functions


def uniform_prior(value, umin, umax):
    """Uniform prior distribution.
    """
    if umin <= value <= umax:
        return 0.0
    else:
        return - np.inf


def normal_prior(value, mean, sigma):
    """Normal prior distribution.
    """
    return - 0.5 * (2 * np.pi * sigma) - (value - mean) ** 2 / (2. * sigma)

# Probability function


def lnprobmodel(model, data):

    ul = data['ul']
    notul = -ul

    difference = model[notul] - data['flux'][notul]

    if np.rank(data['dflux']) > 1:
# use different errors for model above or below data
        sign = difference > 0
        loerr, hierr = 1 * -sign, 1 * sign
        logprob = - difference ** 2 / (2. * (loerr * data['dflux'][0][notul] +
                                             hierr * data['dflux'][1][notul]) ** 2)
    else:
        logprob = - difference ** 2 / (2. * data['dflux'][notul] ** 2)

    totallogprob = np.sum(logprob)

    if np.sum(ul) > 0:
        # deal with upper limits at CL set by data['cl']
        violated_uls = np.sum(model[ul] > data['flux'][ul])
        totallogprob += violated_uls * np.log(1. - data['cl'])

    return totallogprob


def lnprob(pars, data, modelfunc, priorfunc):

    if priorfunc is None:
        lnprob_priors = 0.0
    else:
        lnprob_priors = priorfunc(pars)

# If prior is -np.inf, avoid calling the function as invalid calls may be made,
# and the result will be discarded anyway
    if not np.isinf(lnprob_priors):
        modelout = modelfunc(pars, data)

        # Save blobs or save model if no blobs given
        if ((type(modelout) == tuple or type(modelout) == list)
                and (type(modelout) != np.ndarray)):
            # print len(modelout)
            model = modelout[0]
            blob = modelout[1:]
        else:
            model = modelout
            blob = (modelout, )

        lnprob_model = lnprobmodel(model, data)
    else:
        lnprob_model = 0.0
        blob = None

    total_lnprob = lnprob_model + lnprob_priors

    return total_lnprob, blob

# Sampler funcs


def _run_mcmc(sampler, pos, nrun):
    for i, out in enumerate(sampler.sample(pos, iterations=nrun)):
        progress = (100. * float(i) / float(nrun))
        if progress % 5 < (5. / float(nrun)):
            print("\nProgress of the run: {0:.0f} percent"
                  " ({1} of {2} steps)".format(int(progress), i, nrun))
            npars = out[0].shape[-1]
            paravg, parstd = [], []
            for npar in range(npars):
                paravg.append(np.average(out[0][:, npar]))
                parstd.append(np.std(out[0][:, npar]))
            print("                            " +
                  (" ".join(["{%i:-^10}" % i for i in range(npars)])
                   ).format(*sampler.labels))
            print("  Last ensemble average : " +
                  (" ".join(["{%i:^10.3g}" % i for i in range(npars)])
                   ).format(*paravg))
            print("  Last ensemble std     : " +
                  (" ".join(["{%i:^10.3g}" % i for i in range(npars)])
                   ).format(*parstd))
            print("  Last ensemble lnprob  :  avg: {0:.3f}, max: {1:.3f}".format(
                np.average(out[1]), np.max(out[1])))
    return sampler, out[0]


def get_sampler(data=None, p0=None, model=None, prior=None,
                nwalkers=500, nburn=100,
                guess=True, labels=None, threads=4):
    """Generate a new MCMC sampler.

    Parameters
    ----------
    data : dict
        Dictionary containing the observed spectrum.
    p0 : array
        Initial position vector. The distribution for the ``nwalkers`` walkers
        will be computed as a multidimensional gaussian of width 5% around the
        initial position vector ``p0``.
    model : function
        A function that takes a vector in the parameter space and the data
        dictionary, and returns the expected fluxes at the energies in the
        spectrum. Additional return objects will be saved as blobs in the
        sampler chain, see `the emcee documentation for the
        format
        <http://dan.iel.fm/emcee/current/user/advanced/#arbitrary-metadata-blobs>`_.
    prior : function, optional
        A function that takes a vector in the parameter space and returns the
        log-likelihood of the Bayesian prior. Parameter limits can be specified
        through a uniform prior, returning 0. if the vector is within the
        parameter bounds and ``-np.inf`` otherwise.
    nwalkers : int, optional
        The number of Goodman & Weare “walkers”. Default is 500.
    nburn : int, optional
        Number of burn-in steps. After ``nburn`` steps, the sampler is reset and
        chain history discarded. It is necessary to settle the sampler into the
        maximum of the parameter space density. Default is 100.
    labels : iterable of strings, optional
        Labels for the parameters included in the position vector `p0`. If not
        provided ``['par1','par2', ... ,'parN']`` will be used.
    threads : int, optional
        Number of threads to use for sampling. Default is 4.
    guess : bool, optional
        Whether to attempt to guess the normalization (first) parameter of the
        model. Default is True.

    Returns
    -------
    sampler : :class:`~emcee.EnsembleSampler` instance
        Ensemble sampler with walker positions after `nburn` burn-in steps.
    pos : :class:`~numpy.array`
        Final position vector array.

    See also
    --------
    emcee.EnsembleSampler
    """
    import emcee

    if data is None:
        log.warn('Data dictionary is missing!')
        raise TypeError

    if model is None:
        log.warn('Model function is missing!')
        raise TypeError

    # Add parameter labels if not provided or too short
    if labels is None:
        # First is normalization
        labels = ['norm', ] + ['par{0}'.format(i) for i in range(1, len(p0))]
    elif len(labels) < len(p0):
        labels += ['par{0}'.format(i) for i in range(len(labels), len(p0))]

    if guess:
        # guess normalization parameter from p0
        modelout = model(p0, data)
        if ((type(modelout) == tuple or type(modelout) == list)
                and (type(modelout) != np.ndarray)):
            spec = modelout[0]
        else:
            spec = modelout
        p0[labels.index('norm')] *= np.trapz(
            data['flux'], data['ene']) / np.trapz(spec, data['ene'])

    ndim = len(p0)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=[data, model, prior], threads=threads)

    # Add data and parameters properties to sampler
    sampler.data = data
    sampler.labels = labels

    # Initialize walkers in a ball of relative size 5% in all dimensions
    p0var = np.array([0.05 * pp for pp in p0])
    p0 = emcee.utils.sample_ball(p0, p0var, nwalkers)

    if nburn > 0:
        print(
            'Burning in the {0} walkers with {1} steps...'.format(nwalkers, nburn))
        sampler, pos = _run_mcmc(sampler, p0, nburn)
    else:
        pos = p0

    return sampler, pos


def run_sampler(nrun=100, sampler=None, pos=None, **kwargs):
    """Run an MCMC sampler.

    If no sampler or initial position vector is provided, extra ``kwargs`` are
    passed to `get_sampler` to generate a new sampler.

    Parameters
    ----------
    nrun : int, optional
        Number of steps to run
    sampler : :class:`~emcee.EnsembleSampler` instance, optional
        Sampler.
    pos : :class:`~numpy.ndarray`, optional
        A list of initial position vectors for the walkers. It should have
        dimensions of ``(nwalkers,dim)``, where ``dim`` is the number of free
        parameters. `emcee.utils.sample_ball` can be used to generate a
        multidimensional gaussian distribution around a single initial position.

    Returns
    -------
    sampler : :class:`~emcee.EnsembleSampler` instance
        Sampler containing the paths of the walkers during the ``nrun`` steps.
    pos : array
        List of final position vectors after the run.
    """

    if sampler is None or pos is None:
        sampler, pos = get_sampler(**kwargs)

    print('\nWalker burn in finished, running {0} steps...'.format(nrun))
    sampler.reset()
    sampler, pos = _run_mcmc(sampler, pos, nrun)

    return sampler, pos
