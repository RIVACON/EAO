
[![CI](https://github.com/RIVACON/EAO/actions/workflows/unittests_doc.yml/badge.svg)](https://github.com/RIVACON/EAO/actions/workflows/unittests_doc.yml)
[![documentation](https://img.shields.io/badge/-documentation-blue.svg)](https://rivacon.github.io/EAO/)
![Static Badge](https://img.shields.io/badge/License%20PLACEHOLDER-yellow)
![Test Coverage](coverage-badge.svg)
[![RIVACON](https://img.shields.io/badge/powered%20by-RIVACON-lightgrey.svg)](https://www.rivacon.com/en/)

# The Energy Asset Optimization (EAO) package

The EAO package is a modular Python framework, designed to enable practitioners to use, build and optimize energy and commodity trading portfolios using linear or mixed integer programming as well as stochastic linear programming. It provides an implementation of
- standard assets such as contracts, transport and storages
- addition of new asset types
- their combination to complex portfolios using network structures
- (de-) serialization to JSON
- basic input & output functionality

We found that the approach is useful for modeling very different problem settings, such as decentral and renewable power generation, green power supply and PPAs and sector coupling in ad-hoc analysis, market modeling or daily operation.

You can find the documentation along with several sample notebooks here:
[`EAO documentation`](https://RIVACON.github.io/EAO)

And an extensive technical report here:
[`Report`](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3842822)

To install:
pip install eaopack

