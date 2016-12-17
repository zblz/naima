# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import numpy as np
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
from .extern.validator import (validate_scalar, validate_array,
                               validate_physical_type)
from .radiative import Synchrotron, InverseCompton, PionDecay, Bremsstrahlung
from .model_utils import memoize

__all__ = [
    'Synchrotron', 'InverseCompton', 'PionDecay', 'Bremsstrahlung',
    'BrokenPowerLaw', 'ExponentialCutoffPowerLaw', 'PowerLaw', 'LogParabola',
    'ExponentialCutoffBrokenPowerLaw', 'TableModel', 'EblAbsorptionModel'
]


def _validate_ene(ene):
    from astropy.table import Table

    if isinstance(ene, dict) or isinstance(ene, Table):
        try:
            ene = validate_array(
                'energy', u.Quantity(ene['energy']), physical_type='energy')
        except KeyError:
            raise TypeError('Table or dict does not have \'ene\' column')
    else:
        if not isinstance(ene, u.Quantity):
            ene = u.Quantity(ene)
        validate_physical_type('energy', ene, physical_type='energy')

    return ene


class PowerLaw(object):
    """
    One dimensional power law model.

    Parameters
    ----------
    amplitude : float
        Model amplitude.
    e_0 : `~astropy.units.Quantity` float
        Reference energy
    alpha : float
        Power law index

    See Also
    --------
    PowerLaw, BrokenPowerLaw, LogParabola

    Notes
    -----
    Model formula (with :math:`A` for ``amplitude``, :math:`\\alpha` for
    ``alpha``):

        .. math:: f(E) = A (E / E_0) ^ {-\\alpha}

    """

    param_names = ['amplitude', 'e_0', 'alpha']
    _memoize = False
    _cache = {}
    _queue = []

    def __init__(self, amplitude, e_0, alpha):
        self.amplitude = amplitude
        self.e_0 = validate_scalar(
            'e_0', e_0, domain='positive', physical_type='energy')
        self.alpha = alpha

    @staticmethod
    def eval(e, amplitude, e_0, alpha):
        """One dimensional power law model function"""

        xx = e / e_0
        return amplitude * xx**(-alpha)

    @memoize
    def _calc(self, e):
        return self.eval(
            e.to('eV').value, self.amplitude,
            self.e_0.to('eV').value, self.alpha)

    def __call__(self, e):
        """One dimensional power law model function"""
        e = _validate_ene(e)
        return self._calc(e)


class ExponentialCutoffPowerLaw(object):
    """
    One dimensional power law model with an exponential cutoff.

    Parameters
    ----------
    amplitude : float
        Model amplitude
    e_0 : `~astropy.units.Quantity` float
        Reference point
    alpha : float
        Power law index
    e_cutoff : `~astropy.units.Quantity` float
        Cutoff point
    beta : float
        Cutoff exponent

    See Also
    --------
    PowerLaw, BrokenPowerLaw, LogParabola

    Notes
    -----
    Model formula (with :math:`A` for ``amplitude``, :math:`\\alpha` for
    ``alpha``, and :math:`\\beta` for ``beta``):

        .. math:: f(E) = A (E / E_0) ^ {-\\alpha}
                    \\exp (- (E / E_{cutoff}) ^ \\beta)

    """

    param_names = ['amplitude', 'e_0', 'alpha', 'e_cutoff', 'beta']
    _memoize = False
    _cache = {}
    _queue = []

    def __init__(self, amplitude, e_0, alpha, e_cutoff, beta=1.0):
        self.amplitude = amplitude
        self.e_0 = validate_scalar(
            'e_0', e_0, domain='positive', physical_type='energy')
        self.alpha = alpha
        self.e_cutoff = validate_scalar(
            'e_cutoff', e_cutoff, domain='positive', physical_type='energy')
        self.beta = beta

    @staticmethod
    def eval(e, amplitude, e_0, alpha, e_cutoff, beta):
        """One dimensional power law with an exponential cutoff model function
        """

        xx = e / e_0
        return amplitude * xx**(-alpha) * np.exp(-(e / e_cutoff)**beta)

    @memoize
    def _calc(self, e):
        return self.eval(
            e.to('eV').value, self.amplitude,
            self.e_0.to('eV').value, self.alpha,
            self.e_cutoff.to('eV').value, self.beta)

    def __call__(self, e):
        """One dimensional power law with an exponential cutoff model function
        """
        e = _validate_ene(e)
        return self._calc(e)


class BrokenPowerLaw(object):
    """
    One dimensional power law model with a break.

    Parameters
    ----------
    amplitude : float
        Model amplitude at the break energy
    e_0 : `~astropy.units.Quantity` float
        Reference point
    e_break : `~astropy.units.Quantity` float
        Break energy
    alpha_1 : float
        Power law index for x < x_break
    alpha_2 : float
        Power law index for x > x_break

    See Also
    --------
    PowerLaw, ExponentialCutoffPowerLaw, LogParabola

    Notes
    -----
    Model formula (with :math:`A` for ``amplitude``, :math:`E_0` for ``e_0``,
    :math:`\\alpha_1` for ``alpha_1`` and :math:`\\alpha_2` for ``alpha_2``):

        .. math::

            f(E) = \\left \\{
                     \\begin{array}{ll}
                       A (E / E_0) ^ {-\\alpha_1} & : E < E_{break} \\\\
                       A (E_{break}/E_0) ^ {\\alpha_2-\\alpha_1}
                           (E / E_0) ^ {-\\alpha_2} & :  E > E_{break} \\\\
                     \\end{array}
                   \\right.
    """

    param_names = ['amplitude', 'e_0', 'e_break', 'alpha_1', 'alpha_2']
    _memoize = False
    _cache = {}
    _queue = []

    def __init__(self, amplitude, e_0, e_break, alpha_1, alpha_2):
        self.amplitude = amplitude
        self.e_0 = validate_scalar(
            'e_0', e_0, domain='positive', physical_type='energy')
        self.e_break = validate_scalar(
            'e_break', e_break, domain='positive', physical_type='energy')
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2

    @staticmethod
    def eval(e, amplitude, e_0, e_break, alpha_1, alpha_2):
        """One dimensional broken power law model function"""
        K = np.where(e < e_break, 1, (e_break / e_0)**(alpha_2 - alpha_1))
        alpha = np.where(e < e_break, alpha_1, alpha_2)
        return amplitude * K * (e / e_0)**-alpha

    @memoize
    def _calc(self, e):
        return self.eval(
            e.to('eV').value, self.amplitude,
            self.e_0.to('eV').value,
            self.e_break.to('eV').value, self.alpha_1, self.alpha_2)

    def __call__(self, e):
        """One dimensional broken power law model function"""
        e = _validate_ene(e)
        return self._calc(e)


class ExponentialCutoffBrokenPowerLaw(object):
    """
    One dimensional power law model with a break.

    Parameters
    ----------
    amplitude : float
        Model amplitude at the break point
    e_0 : `~astropy.units.Quantity` float
        Reference point
    e_break : `~astropy.units.Quantity` float
        Break energy
    alpha_1 : float
        Power law index for x < x_break
    alpha_2 : float
        Power law index for x > x_break
    e_cutoff : `~astropy.units.Quantity` float
        Exponential Cutoff energy
    beta : float, optional
        Exponential cutoff rapidity. Default is 1.

    See Also
    --------
    PowerLaw, ExponentialCutoffPowerLaw, LogParabola

    Notes
    -----
    Model formula (with :math:`A` for ``amplitude``, :math:`E_0` for ``e_0``,
    :math:`\\alpha_1` for ``alpha_1``, :math:`\\alpha_2` for ``alpha_2``,
    :math:`E_{cutoff}` for ``e_cutoff``, and :math:`\\beta` for ``beta``):

        .. math::


            f(E) = \\exp(-(E / E_{cutoff})^\\beta)\\left \\{
                     \\begin{array}{ll}
                       A (E / E_0) ^ {-\\alpha_1}    & : E < E_{break} \\\\
                       A (E_{break}/E_0) ^ {\\alpha_2-\\alpha_1}
                            (E / E_0) ^ {-\\alpha_2} & : E > E_{break} \\\\
                     \\end{array}
                   \\right.

    """

    param_names = [
        'amplitude', 'e_0', 'e_break', 'alpha_1', 'alpha_2', 'e_cutoff', 'beta'
    ]
    _memoize = False
    _cache = {}
    _queue = []

    def __init__(self,
                 amplitude,
                 e_0,
                 e_break,
                 alpha_1,
                 alpha_2,
                 e_cutoff,
                 beta=1.0):
        self.amplitude = amplitude
        self.e_0 = validate_scalar(
            'e_0', e_0, domain='positive', physical_type='energy')
        self.e_break = validate_scalar(
            'e_break', e_break, domain='positive', physical_type='energy')
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.e_cutoff = validate_scalar(
            'e_cutoff', e_cutoff, domain='positive', physical_type='energy')
        self.beta = beta

    @staticmethod
    def eval(e, amplitude, e_0, e_break, alpha_1, alpha_2, e_cutoff, beta):
        """One dimensional broken power law model function"""
        K = np.where(e < e_break, 1, (e_break / e_0)**(alpha_2 - alpha_1))
        alpha = np.where(e < e_break, alpha_1, alpha_2)
        ee2 = e / e_cutoff
        return amplitude * K * (e / e_0)**-alpha * np.exp(-(ee2**beta))

    @memoize
    def _calc(self, e):
        return self.eval(
            e.to('eV').value, self.amplitude,
            self.e_0.to('eV').value,
            self.e_break.to('eV').value, self.alpha_1, self.alpha_2,
            self.e_cutoff.to('eV').value, self.beta)

    def __call__(self, e):
        """One dimensional broken power law model with exponential cutoff
        function"""
        e = _validate_ene(e)
        return self._calc(e)


class LogParabola(object):
    """
    One dimensional log parabola model (sometimes called curved power law).

    Parameters
    ----------
    amplitude : float
        Model amplitude
    e_0 : `~astropy.units.Quantity` float
        Reference point
    alpha : float
        Power law index
    beta : float
        Power law curvature

    See Also
    --------
    PowerLaw, BrokenPowerLaw, ExponentialCutoffPowerLaw

    Notes
    -----
    Model formula (with :math:`A` for ``amplitude`` and :math:`\\alpha` for
    ``alpha`` and :math:`\\beta` for ``beta``):

        .. math::

            f(e) = A \\left(\\frac{E}{E_{0}}\\right)^
                {- \\alpha - \\beta \\log{\\left (\\frac{E}{E_{0}} \\right )}}

    """

    param_names = ['amplitude', 'e_0', 'alpha', 'beta']
    _memoize = False
    _cache = {}
    _queue = []

    def __init__(self, amplitude, e_0, alpha, beta):
        self.amplitude = amplitude
        self.e_0 = validate_scalar(
            'e_0', e_0, domain='positive', physical_type='energy')
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def eval(e, amplitude, e_0, alpha, beta):
        """One dimenional log parabola model function"""

        ee = e / e_0
        eeponent = -alpha - beta * np.log(ee)
        return amplitude * ee**eeponent

    @memoize
    def _calc(self, e):
        return self.eval(
            e.to('eV').value, self.amplitude,
            self.e_0.to('eV').value, self.alpha, self.beta)

    def __call__(self, e):
        """One dimensional curved power law function"""
        e = _validate_ene(e)
        return self._calc(e)


class TableModel(object):
    """
    A model generated from a table of energy and value arrays.

    The units returned will be the units of the values array provided at
    initialization. The model will return values interpolated in
    log-space, returning 0 for energies outside of the limits of the provided
    energy array.

    Parameters
    ----------
    energy : `~astropy.units.Quantity` array
        Array of energies at which the model values are given
    values : array
        Array with the values of the model at energies ``energy``.
    amplitude : float
        Model amplitude that is multiplied to the supplied arrays. Defaults to
        1.
    """

    def __init__(self, energy, values, amplitude=1):
        from scipy.interpolate import interp1d
        self._energy = validate_array(
            'energy', energy, domain='positive', physical_type='energy')
        self._values = values
        self.amplitude = amplitude

        loge = np.log10(self._energy.to('eV').value)
        try:
            self.unit = self._values.unit
            logy = np.log10(self._values.value)
        except AttributeError:
            self.unit = u.Unit('')
            logy = np.log10(self._values)

        self._interplogy = interp1d(
            loge, logy, fill_value=-np.Inf, bounds_error=False, kind='cubic')

    def __call__(self, e):
        e = _validate_ene(e)
        interpy = np.power(10, self._interplogy(np.log10(e.to('eV').value)))
        return self.amplitude * interpy * self.unit


class EblAbsorptionModel(TableModel):
    """
    A TableModel containing the different absorption values from a specific
    model.

    It returns dimensionless opacity values, that could be multiplied to any
    model.

    Parameters
    ----------
    redshift : float
        Redshift considered for the absorption evaluation.
    ebl_absorption_model : {'Dominguez'}
        Name of the EBL absorption model to use (Dominguez by default).

    Notes
    -----
    Dominguez model refers to the Dominguez 2011 EBL model. Current
    implementation does NOT perform an interpolation in the redshift, so it
    just uses the closest z value from the finely binned tau_dominguez11.npz
    file (delta_z=0.01).

    See Also
    --------
    TableModel
    """

    def __init__(self, redshift, ebl_absorption_model='Dominguez'):

        # check that the redshift is a positive scalar
        if not isinstance(redshift, u.Quantity):
            redshift *= u.dimensionless_unscaled

        self.redshift = validate_scalar(
            'redshift',
            redshift,
            domain='positive',
            physical_type='dimensionless')

        self.model = ebl_absorption_model

        if self.model == 'Dominguez':
            """Table generated by Alberto Dominguez containing tau vs energy
            [TeV] vs redshift.  Energy is defined between 1 GeV and 100 TeV, in
            500 bins uniform in log(E).  Redshift is defined between 0.01 and
            4, in steps of 0.01.  """
            filename = get_pkg_data_filename(
                os.path.join('data', 'tau_dominguez11.npz'))
            taus_table = np.load(filename)['arr_0']
            redshift_list = np.arange(0.01, 4, 0.01)
            energy = taus_table['energy'] * u.TeV
            if self.redshift >= 0.01:
                colname = 'col%s' % (
                    2 + (np.abs(redshift_list - self.redshift)).argmin())
                table_values = taus_table[colname]
                # Set maximum value of the log(Tau) to 150, as it is high
                # enough.  This solves later overflow problems.
                table_values[table_values > 150.] = 150.
                taus = 10**table_values * u.dimensionless_unscaled
            elif self.redshift < 0.01:
                taus = 10**np.zeros(len(taus_table[
                    'energy'])) * u.dimensionless_unscaled
        else:
            raise ValueError('Model should be one of: ["Dominguez"]')

        super(EblAbsorptionModel, self).__init__(energy, taus)

    def transmission(self, e):
        e = _validate_ene(e)
        taus = np.zeros(len(e))
        for i in range(0, len(e)):
            if e[i].to('GeV').value < 1.:
                taus[i] = 0.
            elif e[i].to('TeV').value > 100.:
                taus[i] = np.log10(6000.)
            else:
                taus[i] = np.log10(self(e[i]))
        return np.exp(-taus)
