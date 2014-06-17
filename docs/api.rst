Detailed description of functions and arguments (API)
=====================================================

.. currentmodule:: gammafit

Radiative Models
----------------

.. autoclass:: gammafit.ElectronOZM
    :members:

.. autoclass:: gammafit.ProtonOZM
    :members:

MCMC fitting
------------

.. autofunction:: gammafit.get_sampler
.. autofunction:: gammafit.run_sampler

.. autofunction:: normal_prior
.. autofunction:: uniform_prior

MCMC results plotting 
---------------------

.. autofunction:: gammafit.plot_chain
.. autofunction:: gammafit.plot_fit
.. autofunction:: gammafit.plot_data
.. autofunction:: gammafit.generate_diagnostic_plots


Convenience functions
---------------------

.. autofunction:: gammafit.generate_energy_edges
.. autofunction:: gammafit.sed_conversion
.. autofunction:: gammafit.build_data_dict
