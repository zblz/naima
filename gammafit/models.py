# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import astropy.units as u
from .extern.validator import validate_scalar, validate_array, validate_physical_type
from .radiative import Synchrotron, InverseCompton, PionDecay, Bremsstrahlung

__all__ = ['Synchrotron', 'InverseCompton', 'PionDecay', 'Bremsstrahlung',
           'BrokenPowerLaw', 'ExponentialCutoffPowerLaw', 'PowerLaw',
           'LogParabola', 'ExponentialCutoffBrokenPowerLaw' ]

def _validate_ene(ene):
    from astropy.table import Table

    if isinstance(ene, dict) or isinstance(ene, Table):
        try:
            ene = validate_array('energy',u.Quantity(ene['energy']),physical_type='energy')
        except KeyError:
            raise TypeError('Table or dict does not have \'ene\' column')
    else:
        if not isinstance(ene,u.Quantity):
            ene = u.Quantity(ene)
        validate_physical_type('energy',ene,physical_type='energy')

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

    def __init__(self, amplitude, e_0, alpha):
        self.amplitude = amplitude
        self.e_0 = validate_scalar('e_0', e_0, domain='positive', physical_type='energy')
        self.alpha = alpha

    @staticmethod
    def eval(e, amplitude, e_0, alpha):
        """One dimensional power law model function"""

        xx = e / e_0
        return amplitude * xx ** (-alpha)

    def __call__(self,e):
        """One dimensional power law model function"""

        e = _validate_ene(e)

        return self.eval(e.to('eV').value, self.amplitude,
                self.e_0.to('eV').value, self.alpha)


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

        .. math:: f(E) = A (E / E_0) ^ {-\\alpha} \\exp (- (E / E_{cutoff}) ^ \\beta)

    """

    param_names = ['amplitude', 'e_0', 'alpha', 'e_cutoff', 'beta']

    def __init__(self, amplitude, e_0, alpha, e_cutoff, beta=1.0):
        self.amplitude = amplitude
        self.e_0 = validate_scalar('e_0', e_0, domain='positive', physical_type='energy')
        self.alpha = alpha
        self.e_cutoff = validate_scalar('e_cutoff', e_cutoff, domain='positive', physical_type='energy')
        self.beta = beta

    @staticmethod
    def eval(e, amplitude, e_0, alpha, e_cutoff, beta):
        """One dimensional power law with an exponential cutoff model function"""

        xx = e / e_0
        return amplitude * xx ** (-alpha) * np.exp(-(e / e_cutoff) ** beta)

    def __call__(self,e):
        """One dimensional power law with an exponential cutoff model function"""

        e = _validate_ene(e)

        return self.eval(e.to('eV').value, self.amplitude,
                self.e_0.to('eV').value, self.alpha,
                self.e_cutoff.to('eV').value, self.beta)

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
                       A (E_{break}/E_0) ^ {\\alpha_2-\\alpha_1} (E / E_0) ^ {-\\alpha_2} & :  E > E_{break} \\\\
                     \\end{array}
                   \\right.
    """

    param_names = ['amplitude', 'e_0', 'e_break', 'alpha_1', 'alpha_2']

    def __init__(self, amplitude, e_0, e_break, alpha_1, alpha_2):
        self.amplitude = amplitude
        self.e_0 = validate_scalar('e_0', e_0, domain='positive', physical_type='energy')
        self.e_break = validate_scalar('e_break', e_break, domain='positive', physical_type='energy')
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2

    @staticmethod
    def eval(e, amplitude, e_0, e_break, alpha_1, alpha_2):
        """One dimensional broken power law model function"""
        K = np.where(e < e_break, 1, (e_break / e_0) ** (alpha_2 - alpha_1))
        alpha = np.where(e < e_break, alpha_1, alpha_2)
        return amplitude * K * (e / e_0) ** -alpha

    def __call__(self,e):
        """One dimensional power law model function"""

        e = _validate_ene(e)

        return self.eval(e.to('eV').value, self.amplitude,
                self.e_0.to('eV').value, self.e_break.to('eV').value,
                self.alpha_1, self.alpha_2)

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
                       A (E / E_0) ^ {-\\alpha_1}                                         & : E < E_{break} \\\\
                       A (E_{break}/E_0) ^ {\\alpha_2-\\alpha_1} (E / E_0) ^ {-\\alpha_2} & : E > E_{break} \\\\
                     \\end{array}
                   \\right.

    """

    param_names = ['amplitude', 'e_0', 'e_break', 'alpha_1', 'alpha_2']

    def __init__(self, amplitude, e_0, e_break, alpha_1, alpha_2, e_cutoff, beta=1.0):
        self.amplitude = amplitude
        self.e_0 = validate_scalar('e_0', e_0, domain='positive', physical_type='energy')
        self.e_break = validate_scalar('e_break', e_break, domain='positive', physical_type='energy')
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.e_cutoff = validate_scalar('e_cutoff', e_cutoff, domain='positive', physical_type='energy')
        self.beta = beta

    @staticmethod
    def eval(e, amplitude, e_0, e_break, alpha_1, alpha_2, e_cutoff, beta):
        """One dimensional broken power law model function"""
        K = np.where(e < e_break, 1, (e_break / e_0) ** (alpha_2 - alpha_1))
        alpha = np.where(e < e_break, alpha_1, alpha_2)
        ee2 = e / e_cutoff
        return amplitude * K * (e / e_0) ** -alpha * np.exp(- (ee2 ** beta))

    def __call__(self,e):
        """One dimensional power law model function"""

        e = _validate_ene(e)

        return self.eval(e.to('eV').value, self.amplitude,
                self.e_0.to('eV').value, self.e_break.to('eV').value,
                self.alpha_1, self.alpha_2, self.e_cutoff.to('eV').value,
                self.beta)

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
    Model formula (with :math:`A` for ``amplitude`` and :math:`\\alpha` for ``alpha`` and :math:`\\beta` for ``beta``):

        .. math:: f(e) = A \\left(\\frac{E}{E_{0}}\\right)^{- \\alpha - \\beta \\log{\\left (\\frac{E}{E_{0}} \\right )}}

    """

    param_names = ['amplitude', 'e_0', 'alpha', 'beta']

    def __init__(self, amplitude, e_0, alpha, beta):
        self.amplitude = amplitude
        self.e_0 = validate_scalar('e_0', e_0, domain='positive', physical_type='energy')
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def eval(e, amplitude, e_0, alpha, beta):
        """One dimenional log parabola model function"""

        ee = e / e_0
        eeponent = -alpha - beta * np.log(ee)
        return amplitude * ee ** eeponent

    def __call__(self,e):
        """One dimensional power law model function"""

        e = _validate_ene(e)

        return self.eval(e.to('eV').value, self.amplitude,
                self.e_0.to('eV').value,
                self.alpha, self.beta)

