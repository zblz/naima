Spectral model fitting
======================

.. currentmodule:: naima

- MCMC sampling
    - `naima.get_sampler`
    - `naima.run_sampler`
- Interactive model fitting
    - `naima.InteractiveModelFitter`
- Saving and retrieving the parameter chain of a run
    - `naima.save_run`
    - `naima.read_run`
- Plotting and analysis
    - `naima.plot_chain`
    - `naima.plot_corner`
    - `naima.plot_fit`
    - `naima.plot_blob`
    - `naima.save_diagnostic_plots`
    - `naima.save_results_table`
- Priors
    - `naima.normal_prior`
    - `naima.uniform_prior`
    - `naima.log_uniform_prior`
    
API
---

.. autofunction:: get_sampler
.. autofunction:: run_sampler
.. autoclass:: InteractiveModelFitter
.. autofunction:: save_run
.. autofunction:: read_run
.. autofunction:: plot_chain
.. autofunction:: plot_corner
.. autofunction:: plot_fit
.. autofunction:: plot_data
.. autofunction:: plot_blob
.. autofunction:: save_diagnostic_plots
.. autofunction:: save_results_table
.. autofunction:: normal_prior
.. autofunction:: uniform_prior
