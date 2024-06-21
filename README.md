# Badges <!-- omit in toc -->

| Badges         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| :------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Registry       | [![RSD](https://img.shields.io/badge/rsd-eitprocessing-00a3e3.svg)](https://www.research-software.nl/software/eitprocessing) [![workflow pypi badge](https://img.shields.io/pypi/v/eitprocessing.svg?colorB=blue)](https://pypi.python.org/project/eitprocessing/) [![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](git@github.com:EIT-ALIVE/eitprocessing)                                                                                   |
| License        | [![github license badge](https://img.shields.io/github/license/EIT-ALIVE/eitprocessing)](git@github.com:EIT-ALIVE/eitprocessing)                                                                                                                                                                                                                                                                                                                                                                             |
| Citation       | [![DOI](https://zenodo.org/badge/617944717.svg)](https://zenodo.org/badge/latestdoi/617944717)                                                                                                                                                                                                                                                                                                                                                                                                               |
| Fairness       | [![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9147/badge)](https://www.bestpractices.dev/projects/9147) [![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)                                                                                                                                                                                                  |
| GitHub Actions | ![build](https://github.com/EIT-ALIVE/eitprocessing/actions/workflows/build.yml/badge.svg) ![dash_actions](https://github.com/EIT-ALIVE/eitprocessing/actions/workflows/dash_actions.yml/badge.svg) ![lint](https://github.com/EIT-ALIVE/eitprocessing/actions/workflows/lint.yml/badge.svg) ![documentation](https://github.com/EIT-ALIVE/eitprocessing/actions/workflows/documentation.yml/badge.svg) ![cffconvert](https://github.com/EIT-ALIVE/eitprocessing/actions/workflows/cffconvert.yml/badge.svg) |
| Python         | ![Python](https://img.shields.io/badge/python-3.10-blue.svg)                                                                                                                                                                                                                                                                                                                                                                                                                                                 |

# Contents <!-- omit in toc -->

- [Installation](#installation-1)
  - [Virtual environment](#virtual-environment)
  - [Install using `pip`](#install-using-pip)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Credits](#credits)

The project setup is documented in [project_setup](/project_setup.md). Feel free to remove this document (and/or the link to this document) if you don't need it.
## Installation

[Electrical Impedance Tomography](https://en.wikipedia.org/wiki/Electrical_impedance_tomography) (EIT) is a noninvasive
and radiation-free continuous imaging tool for monitoring respiratory mechanics. eitprocessing aims to provide a
versatile, user-friendly, reproducible and reliable workflow for the processing and analysis of EIT data and related
waveform data, like pressures and flow.

eitprocessing includes tools to load data exported from EIT-devices from several manufacturers, including Dräger, SenTec
and Timpel, as well as data from other sources. Several pre-processing tools and analysis tools are provided.

<!-- TODO when available, add summarisation and reporting -->
<!-- TODO extend with short list of available tools when applicable -->

[eit_dash](https://github.com/EIT-ALIVE/eit_dash) provides an accompanying GUI.

We welcome any [contributions or suggestions](CONTRIBUTING.md)

# Installation

## Virtual environment

It is advised to install eitprocessing in a dedicated virtual environment. See e.g. [Install packages in a virtual
environment using pip and
venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) or [Getting started
with conda](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html).

For conda (using 'eit-alive' as example environment name; you can choose your own):
```
conda create -n eit-alive python=3.10
conda activate eit-alive
```

## Install using `pip`

eitprocessing can be installed from PyPi as follows:

- For basic use: `pip install eitprocessing`
- For full developer options (testing, etc): 
  - `git clone git@github.com:EIT-ALIVE/eitprocessing.git`
  - `cd eitprocessing`
  - `pip install -e ".[dev]"`

# Documentation

Please see our [usage documentation](https://eit-alive.github.io/eitprocessing/) for a detailed explanation of the package.

# Contributing

If you want to contribute to the development of eitprocessing,
have a look at the [contribution guidelines](CONTRIBUTING.md) and the [developer documentation](README.dev.md).

# Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
