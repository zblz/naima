# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy import log
import astropy
import astropy.units as u
import warnings

from .utils import validate_data_table, sed_conversion

__all__ = ["normal_prior", "uniform_prior", "get_sampler", "run_sampler"]

# Define phsyical types used in plot and utils.validate_data_table
u.def_physical_type(u.erg / u.cm ** 2 / u.s, 'flux')
u.def_physical_type(u.Unit('1/(s cm2 erg)'), 'differential flux')
u.def_physical_type(u.Unit('1/(s erg)'), 'differential power')
u.def_physical_type(u.Unit('1/TeV'), 'differential energy')
u.def_physical_type(u.Unit('1/cm3'), 'number density')

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

    # Check if conversion is required
    model_is_sed = model.unit.physical_type in ['power','flux']
    data_is_sed = data['flux'].unit.physical_type in ['power','flux']

    if model_is_sed != data_is_sed:
        unit, sed_factor = sed_conversion(data['energy'], model.unit, data_is_sed)
        model = (model * sed_factor).to(data['flux'].unit)

    ul = data['ul']
    notul = -ul

    difference = model[notul] - data['flux'][notul]

    if data['dflux'].ndim > 1:
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
        # If model is not in blobs, save model+blobs
        if ((type(modelout) == tuple or type(modelout) == list)
                and (type(modelout) != np.ndarray)):
            model = modelout[0]

            MODEL_IN_BLOB = False
            for blob in modelout[1:]:
                if np.all(blob == model):
                    MODEL_IN_BLOB=True

            if MODEL_IN_BLOB:
                blob = modelout[1:]
            else:
                blob = modelout
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
                paravg.append(np.median(out[0][:, npar]))
                parstd.append(np.std(out[0][:, npar]))
            print("                           " +
                  (" ".join(["{%i:-^15}" % i for i in range(npars)])
                   ).format(*sampler.labels))
            print("  Last ensemble median : " +
                  (" ".join(["{%i:^15.3g}" % i for i in range(npars)])
                   ).format(*paravg))
            print("  Last ensemble std    : " +
                  (" ".join(["{%i:^15.3g}" % i for i in range(npars)])
                   ).format(*parstd))
            print("  Last ensemble lnprob :  avg: {0:.3f}, max: {1:.3f}".format(
                np.average(out[1]), np.max(out[1])))
    return sampler, out[0]


def get_sampler(data_table=None, p0=None, model=None, prior=None,
                nwalkers=500, nburn=100,
                guess=True, prefit=False, labels=None, threads=4, data_sed=None):
    """Generate a new MCMC sampler.

    Parameters
    ----------
    data_table : `~astropy.table.Table` or list of `~astropy.table.Table`
        Table containing the observed spectrum. If multiple tables are passed as
        a string, they will be concatenated in the order given. Each table needs
        at least these columns, with the appropriate associated units (with the
        physical type indicated in brackets below) as either a
        `~astropy.units.Unit` instance or parseable string:

        - ``energy``: Observed photon energy [``energy``]
        - ``flux``: Observed fluxes [``flux`` or ``differential flux``]
        - ``flux_error``: 68% CL gaussian uncertainty of the flux [``flux`` or
          ``differential flux``]. It can also be provided as ``flux_error_lo``
          and ``flux_error_hi`` (see below).

        Optional columns:

        - ``energy_width``: Width of the energy bin [``energy``], or
        - ``energy_error``: Half-width of the energy bin [``energy``], or
        - ``energy_error_lo`` and ``energy_error_hi``: Distance from bin center
          to lower and upper bin edges [``energy``], or
        - ``energy_lo`` and ``energy_hi``: Energy edges of the corresponding
          energy bin [``energy``]
        - ``flux_error_lo`` and ``flux_error_hi``: 68% CL gaussian lower and
          upper uncertainties of the flux.
        - ``ul``: Flag to indicate that a flux measurement is an upper limit.
        - ``flux_ul``: Upper limit to the flux. If not present, the ``flux``
          column will be taken as an upper limit for those measurements with the
          ``ul`` flag set to True or 1.

        The ``keywords`` metadata field of the table can be used to provide the
        confidence level of the upper limits with the keyword ``cl``, which
        defaults to 90%. The `astropy.io.ascii` reader can recover all
        the needed information from ASCII tables in the
        :class:`~astropy.io.ascii.Ipac` and :class:`~astropy.io.ascii.Daophot`
        formats, and everything except the ``cl`` keyword from tables in the
        :class:`~astropy.io.ascii.Sextractor`.  For the latter format, the cl
        keyword can be added after reading the table with::

            data.meta['keywords']['cl']=0.99

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
        Labels for the parameters included in the position vector ``p0``. If not
        provided ``['par1','par2', ... ,'parN']`` will be used.
    threads : int, optional
        Number of threads to use for sampling. Default is 4.
    guess : bool, optional
        Whether to attempt to guess the normalization (first) parameter of the
        model. Default is True.
    prefit : bool, optional
        Whether to attempt to find the maximum likelihood parameters with
        `scipy.optimize.minimize` and use them as starting point of the MCMC
        run. The parameter values in `p0` will be used as starting points for
        the minimization.
    data_sed : bool, optional
        When providing more than one data table, whether to convert them to SED
        format. If unset or None, all tables will be converted to the format of
        the first table.

    Returns
    -------
    sampler : :class:`~emcee.EnsembleSampler` instance
        Ensemble sampler with walker positions after ``nburn`` burn-in steps.
    pos : :class:`~numpy.ndarray`
        Final position vector array.

    See also
    --------
    emcee.EnsembleSampler
    """
    import emcee

    if data_table is None:
        raise TypeError ('Data table is missing!')
    else:
        data = validate_data_table(data_table,sed=data_sed)

    if model is None:
        raise TypeError ('Model function is missing!')

    # Add parameter labels if not provided or too short
    if labels is None:
        # First is normalization
        labels = ['norm', ] + ['par{0}'.format(i) for i in range(1, len(p0))]
    elif len(labels) < len(p0):
        labels += ['par{0}'.format(i) for i in range(len(labels), len(p0))]


    # Check that the model returns fluxes in same physical type as data
    modelout = model(p0, data)
    if ((type(modelout) == tuple or type(modelout) == list)
            and (type(modelout) != np.ndarray)):
        spec = modelout[0]
    else:
        spec = modelout

    # check whether both can be converted to same physical type through sed_conversion
    try:
        # If both can be converted to differential flux, they can be compared
        # Otherwise, sed_conversion will raise a u.UnitsError
        sed_conversion(data['energy'], spec.unit, False)
        sed_conversion(data['energy'], data['flux'].unit, False)
    except u.UnitsError:
        raise u.UnitsError('The physical type of the model and data units are not compatible,'
                ' please modify your model or data so they match:\n'
                ' Model units: {0} [{1}]\n Data units: {2} [{3}]\n'.format(
                    spec.unit, spec.unit.physical_type,
                    data['flux'].unit, data['flux'].unit.physical_type))

    if guess:
        normNames = ['norm', 'ampl', 'Norm', 'Ampl']
        normNames += ['log(norm', 'log(ampl', 'log(Norm', 'log(Ampl']
        normNames += ['log10(norm', 'log10(ampl', 'log10(Norm', 'log10(Ampl']
        idxs = []
        for l in normNames:
            for l2 in labels:
                if l2.startswith(l):
                    # check with startswith to include normalization, amplitude, etc.
                    idxs.append(labels.index(l2))

        if len(idxs) == 1:

            nunit, sedf = sed_conversion(data['energy'],spec.unit,False)
            currFlux = np.trapz(data['energy']*(spec*sedf).to(nunit), data['energy'])
            nunit, sedf = sed_conversion(data['energy'],data['flux'].unit,False)
            dataFlux = np.trapz(data['energy']*(data['flux']*sedf).to(nunit), data['energy'])
            ratio = (dataFlux / currFlux)
            if labels[idxs[0]].startswith('log('):
                p0[idxs[0]] += np.log(ratio)
            elif labels[idxs[0]].startswith('log10('):
                p0[idxs[0]] += np.log10(ratio)
            else:
                p0[idxs[0]] *= ratio

        elif len(idxs) == 0:
            log.warning('No label starting with [{0}] found: not applying'
                    ' normalization guess.'.format(','.join(normNames)))
        elif len(idxs) > 1:
            log.warning('More than one label starting with [{0}] found:'
                    ' not applying normalization guess.'.format(','.join(normNames)))

    P0_IS_ML = False
    if prefit:
        try:
            import scipy.optimize as op
            flat_prior = lambda *args: 0.0
            nll = lambda *args: -lnprob(*args)[0]
            log.info('Attempting to find Maximum Likelihood parameters...')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = op.minimize(nll, p0, args=(data, model, flat_prior),
                        method='Nelder-Mead',options={'maxfev':500})
            if result['success'] or result['status']==1:
                # also keep result if we have reached maxiter, it is likely
                # better than p0
                log.info('   Initial parameters: {0}'.format(p0))
                log.info('   New ML parameters : {0}'.format(result['x']))
                log.info('   lnprob(p0): {0:.3f}'.format(-result['fun']))
                p0 = result['x']
                P0_IS_ML = True
            else:
                log.warning('Maximum Likelihood procedure failed to converge,'
                        ' using original parameters for MCMC')
        except ImportError:
            log.warning('Scipy is required for the prefit, using '
                        'original parameters for MCMC.')

    ndim = len(p0)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=[data, model, prior], threads=threads)

    # Add data and parameters properties to sampler
    sampler.data_table = data_table
    sampler.data = data
    sampler.labels = labels
    # Add run_info dict
    sampler.run_info = {'n_walkers':nwalkers,
                        'n_burn': nburn,
                        # convert from np.float to regular float
                        'p0': [float(p) for p in p0],
                        'guess':guess,
                        }

    # Initialize walkers in a ball of relative size 0.5% in all dimensions if the
    # parameters have been fit to their ML values, or to 10% otherwise
    spread = 0.005 if P0_IS_ML else 0.1
    p0var = np.array([spread * pp for pp in p0])
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

    sampler.run_info['n_run'] = nrun
    sampler.run_info['p0_run_median'] = [float(p) for p in np.median(pos,axis=0)]

    print('\nWalker burn in finished, running {0} steps...'.format(nrun))
    sampler.reset()
    sampler, pos = _run_mcmc(sampler, pos, nrun)

    return sampler, pos
