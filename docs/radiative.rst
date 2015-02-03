.. _radiative:

Radiative Models
================

``naima`` contains several radiative models that can be used to compute the
non-thermal emission from populations of relativistic electrons or protons.
Below there is a brief explanation of each model, and full details on the
physics and motivation for the implementation can be found in the referenced
articles.

All the radiative models take a particle distribution function as a first
argument. Several particle distribution functions can be found in the
`naima.models` module: `~naima.models.PowerLaw`,
`~naima.models.ExponentialCutoffPowerLaw`, `~naima.models.LogParabola`,
`~naima.models.BrokenPowerLaw`, and
`~naima.models.ExponentialCutoffBrokenPowerLaw`. To use them as a particle
distribution function the units of their amplitude must be in particles per unit
energy (i.e., convertible to ``1/eV``). See below in `IC`_ for details about
the inverse Compton model.::

    >>> ECPL = naima.models.ExponentialCutoffPowerLaw(1e36*u.Unit('1/eV'), 1*u.TeV, 2.1, 13*u.TeV)
    >>> IC = naima.models.InverseCompton(ECPL, seed_photon_fields=['CMB'])

The parameters of the particle distribution can be accessed through the
``particle_distribution`` attribute of the radiative model, e.g.::

    >>> IC.particle_distribution.index = 1.8
    >>> print(ECPL.index)
    1.8

In addition, the same particle distribution instance can be used for several
radiative models simultaneously, for example when computing the synchrotron and
IC emission from an electron population::

    >>> SYN = naima.models.Synchrotron(ECPL, B=100*u.uG)
    >>> SYN.particle_distribution is IC.particle_distribution
    True

Once instantiated, the emission spectra from radiative models can be obtained
through the ``flux`` (differential flux) and ``sed`` (spectral energy
distribution) methods::

    >>> spectrum_energy = np.logspace(-1,14,1000)*u.eV
    >>> sed_IC = IC.sed(spectrum_energy, distance=1.5*u.kpc)
    >>> sed_SYN = SYN.sed(spectrum_energy, distance=1.5*u.kpc)

These spectra can then be analysed or plotted:

.. plot::

    import naima
    import astropy.units as u

    # Define models
    ECPL = naima.models.ExponentialCutoffPowerLaw(1e36*u.Unit('1/eV'),
            1*u.TeV, 2.1, 13*u.TeV)

    IC_CMB = naima.models.InverseCompton(ECPL, seed_photon_fields=['CMB'])
    IC_FIR = naima.models.InverseCompton(ECPL, seed_photon_fields=['FIR'])
    IC_NIR = naima.models.InverseCompton(ECPL, seed_photon_fields=['NIR'])
    IC_CMB.particle_distribution.index = 1.8
    SYN = naima.models.Synchrotron(ECPL, B=100*u.uG)

    # Compute SEDs
    spectrum_energy = np.logspace(-1,14,1000)*u.eV
    sed_IC_CMB = IC_CMB.sed(spectrum_energy, distance=1.5*u.kpc)
    sed_IC_FIR = IC_FIR.sed(spectrum_energy, distance=1.5*u.kpc)
    sed_IC_NIR = IC_NIR.sed(spectrum_energy, distance=1.5*u.kpc)
    sed_IC_tot = sed_IC_CMB + sed_IC_FIR + sed_IC_NIR
    sed_SYN = SYN.sed(spectrum_energy, distance=1.5*u.kpc)

    # Plot
    plt.figure(figsize=(8,4))
    plt.rc('font', family='serif')
    plt.loglog(spectrum_energy,sed_IC_CMB,lw=1,
            ls='-',label='IC (CMB)',c='0.25')
    plt.loglog(spectrum_energy,sed_IC_FIR,lw=1,
            ls='--',label='IC (FIR)',c='0.25')
    plt.loglog(spectrum_energy,sed_IC_NIR,lw=1,
            ls=':',label='IC (NIR)',c='0.25')
    plt.loglog(spectrum_energy,sed_IC_tot,lw=2,
            label='IC (total)',c='r')
    plt.loglog(spectrum_energy,sed_SYN,lw=2,label='Sync',c='b')
    plt.xlabel('Photon energy [{0}]'.format(
            naima.plot._latex_unit(spectrum_energy.unit)))
    plt.ylabel('$E^2 dN/dE$ [{0}]'.format(
            naima.plot._latex_unit(sed_SYN.unit)))
    plt.ylim(bottom=1e-15)
    plt.tight_layout()
    plt.legend(loc='lower left')


.. _IC:

Inverse Compton radiative model
-------------------------------

The inverse Compton (IC) scattering of soft photons by relativistic electrons is
the main gamma-ray production channel for electron populations (`Blumenthal &
Gould 1970`_). Often, the seed photon field will be a blackbody or a diluted
blackbody, and the calculation of IC must be done taking this into account.
``naima`` implements the analytical approximations to IC upscattering of
blackbody radiation developed by `Khangulyan et al. (2014)`_. These have the
advantage of being computationally cheap compared to a numerical integration
over the spectrum of the blackbody, and remain accurate within one percent over
a wide range of energies. Both the isotropic IC and anisotropic IC
approximations are available in ``naima``. If you use this class in your
research, please consult and cite `Khangulyan, D., Aharonian, F.A., & Kelner,
S.R.  2014, Astrophysical Journal, 783, 100
<http://adsabs.harvard.edu/abs/2014ApJ...783..100K>`_.

.. _Khangulyan et al. (2014): http://adsabs.harvard.edu/abs/2014ApJ...783..100K

The implementation in ``naima`` allows to specify which blackbody seed photon
fields to use in the calculation, and provides the three dominant galactic
photon fields at the location of the Solar System through the `CMB` (Cosmic
Microwave Background), `FIR` (far-infrared dust emission), and `NIR`
(near-infrared stellar emission) keywords. The seed photon fields can be
selected though the `seed_photon_fields` parameter of the
`~naima.models.InverseCompton` model. This parameter should be provided with a
list of items, each of which can be either:

    * A string equal to ``CMB`` (default), ``NIR``, or ``FIR``, for which
      radiation fields with temperatures of 2.72 K, 70 K, and 5000 K, and
      energy densities of 0.261, 0.5, and 1 eV/cm³ will be used, or

    * A list of length three (isotropic source) or four (anisotropic source)
      composed of:

        1. A name for the seed photon field
        2. Its temperature as a :class:`~astropy.units.Quantity` float
           instance.
        3. Its photon field energy density as a
           :class:`~astropy.units.Quantity` float instance. If the photon
           field energy density if set to 0, its blackbody energy density
           will be computed through the Stefan-Boltzman law.
        4. Optional: The angle between the seed photon direction and the scattered
           photon direction as a :class:`~astropy.units.Quantity` float
           instance. If this is provided, the anisotropic IC differential
           cross-section will be used.

.. _SY:

Synchrotron radiative model
---------------------------

Synchrotron radiation is produced by all charged particles in the presence of
magnetic fields, and is ubiquitous in the emitted spectrum of leptonic sources.
A full description and derivation of its properties can be found in `Blumenthal
& Gould (1970)`_. The derivation of the spectrum is usually done considering a
uniform magnetic field direction, but that is rarely thought to be the case in
astrophysical sources. Considering random magnetic fields results in a shift of
the maximum emissivity from :math:`E_\mathrm{peak}=0.29 E_\mathrm{c}` to
:math:`0.23 E_c`, where :math:`E_c` is the synchrotron characteristic energy. The
`~naima.models.Synchrotron` class implements the parametrization of the
emissivity function of synchrotron radiation in random magnetic fields presented
by `Aharonian et al. (2010; Appendix D)`_. This parametrization is particularly
useful as it avoids using special functions, and achieves an accuracy of 0.2%
over the entire range of emission energy.

If you use this class in your research, please cite `Aharonian, F.A., Kelner,
S.R., & Prosekin, A.Y. 2010, Physical Review D, 82, 043002
<http://adsabs.harvard.edu/abs/2010PhRvD..82d3002A>`_. 

.. _Aharonian et al. (2010; Appendix D):
        http://adsabs.harvard.edu/abs/2010PhRvD..82d3002A

.. _BR:

Nonthermal Bremsstrahlung radiative model
-----------------------------------------

Nonthermal bremsstrahlung radiation arises when a population of relativistic
particles interact with a thermal particle population (see `Blumenthal & Gould
1970`_). For the computation of the bremsstrahlung emission spectrum, The
`~naima.models.Bremsstrahlung` class implements the approximation of `Baring et
al. (1999)`_ to the original cross-section presented by `Haug (1975)`_.
Electron-electron bremsstrahlung is implemented for the complete energy range,
whereas electron-ion bremsstrahlung is at the moment only available for photon
energies above 10 MeV. The normalization of the emission, and importance of the
electron-electron versus the electron-ion channels, are given by the class
arguments ``n0`` (ion total number density), ``weight_ee`` (weight of the e-e
channel, given by :math:`\sum_i Z_i X_i`), and ``weight_ep`` (weight of the e-p
channel, given by  :math:`\sum_i Z_i^2 X_i`). The defaults for ``weight_ee`` and
``weight_ep`` correspond to a fully ionised medium with solar abundances.

If you use this class in your research, please cite `Baring, M.G., Ellison,
D.C., Reynolds, S.P., Grenier, I.A., & Goret, P. 1999, Astrophysical Journal,
513, 311 <http://adsabs.harvard.edu/abs/1999ApJ...513..311B>`_.

.. _Baring et al. (1999): http://adsabs.harvard.edu/abs/1999ApJ...513..311B
.. _Haug (1975): http://adsabs.harvard.edu/abs/1975ZNatA..30.1099H

.. _PP:

Pion Decay radiative model
--------------------------

The main gamma-ray production for relativistic protons are p-p interactions
followed by pion decay, which results in a photon with :math:`E_\gamma >
100\,\mathrm{MeV}`. Until recently, the only parametrizations available for the
integral cross-section and photon emission spectra were either only applicable
to limited energy ranges, or were given as extensive numerical tables (e.g.,
`Kelner et al. 2006 <http://ukads.nottingham.ac.uk/abs/2006PhRvD..74c4018K>`_;
`Kamae et al. 2006 <http://ukads.nottingham.ac.uk/abs/2006ApJ...647..692K>`_).
By considering Monte Carlo results and a compilation of accelerator data on p-p
interactions, `Kafexhiu et al. (2014)
<http://adsabs.harvard.edu/abs/2014PhRvD..90l3014K>`_ were able to develop
analytic parametrizations to the energy spectra and production rates of gamma
rays from p-p interactions. The `~naima.models.PionDecay` class uses an
implementation of the formulae presented in their paper, and gives the choice of
which high-energy model to use (from the parametrization to the different Monte
Carlo results) through the `hiEmodel` parameter. If you use this class in your
research, please consult and cite `Kafexhiu, E., Aharonian, F., Taylor, A.M., &
Vila, G.S. 2014, Physical Review D, 90, 123014
<http://adsabs.harvard.edu/abs/2014PhRvD..90l3014K>`_. 


.. _Blumenthal & Gould 1970: 
        http://ukads.nottingham.ac.uk/abs/1970RvMP...42..237B
.. _Blumenthal & Gould (1970): 
        http://ukads.nottingham.ac.uk/abs/1970RvMP...42..237B

