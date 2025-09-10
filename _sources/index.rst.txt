.. EAO documentation master file, created by
   sphinx-quickstart on Mon Oct  5 08:38:06 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the Energy Asset Optimization (EAO) documentation!
==============================================================


The EAO package is a modular Python framework, designed to enable practitioners to 
design, build and optimize energy and commodity trading portfolios using linear or mixed integer programming 
as well as stochastic linear programming. It provides an implementation of

* standard assets such as contracts, transport and storages
* addition of new asset types
* their combination to complex portfolios using network structures
* (de-) serialization to JSON
* basic input & output functionality

We found that the approach is useful for modeling very different problem settings, such as decentral and renewable power 
generation, green power supply and PPAs and sector coupling in ad-hoc analysis, market modeling or daily operation.

You can find the code along with some sample notebooks here:
`GitHub repository <https://github.com/RIVACON/EAO>`_

And an extensive technical report here:
`Report <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3842822>`_

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   eao
   samples
   


Indices and tables
==================

* :ref:`modindex`
* :ref:`search`
