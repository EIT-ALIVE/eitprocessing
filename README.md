# EITprocessing <!-- omit in toc -->

## Introduction

[Electrical Impedance Tomography](https://en.wikipedia.org/wiki/Electrical_impedance_tomography) (EIT) is a noninvasive
and radiation-free continuous imaging tool for monitoring respiratory mechanics. eitprocessing aims to provide a
versatile, user-friendly, reproducible and reliable workflow for the processing and analysis of EIT data and related
waveform data, like pressures and flow.

`eitprocessing` includes tools to load data exported from EIT-devices from several manufacturers, including Dräger, SenTec
and Timpel, as well as data from other sources. Several pre-processing tools and analysis tools are provided.

<!-- TODO when available, add summarisation and reporting -->
<!-- TODO extend with short list of available tools when applicable -->

| Badges         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| :------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Registry       | [![RSD](https://img.shields.io/badge/rsd-eitprocessing-00a3e3.svg)](https://www.research-software.nl/software/eitprocessing) [![workflow pypi badge](https://img.shields.io/pypi/v/eitprocessing.svg?colorB=blue)](https://pypi.python.org/project/eitprocessing/) [![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](git@github.com:EIT-ALIVE/eitprocessing)                                                                                                                                                              |
| License        | [![github license badge](https://img.shields.io/github/license/EIT-ALIVE/eitprocessing)](git@github.com:EIT-ALIVE/eitprocessing)                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Citation       | [![DOI](https://zenodo.org/badge/617944717.svg)](https://zenodo.org/badge/latestdoi/617944717)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Fairness       | [![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9147/badge)](https://www.bestpractices.dev/projects/9147) [![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)                                                                                                                                                                                                                                                                             |
| GitHub Actions | ![build](https://github.com/EIT-ALIVE/eitprocessing/actions/workflows/testing.yml/badge.svg) ![lint](https://github.com/EIT-ALIVE/eitprocessing/actions/workflows/lint.yml/badge.svg) ![documentation](https://github.com/EIT-ALIVE/eitprocessing/actions/workflows/test_build_documentation.yml/badge.svg) ![cffconvert](https://github.com/EIT-ALIVE/eitprocessing/actions/workflows/cffconvert.yml/badge.svg) [![Coverage Status](https://coveralls.io/repos/github/EIT-ALIVE/eitprocessing/badge.svg?branch=main)](https://coveralls.io/github/EIT-ALIVE/eitprocessing?branch=main) |
| Python Support | ![Python](https://img.shields.io/badge/python-3.10-blue.svg) ![Python](https://img.shields.io/badge/python-3.11-blue.svg) ![Python](https://img.shields.io/badge/python-3.12-blue.svg) ![Python](https://img.shields.io/badge/python-3.13-blue.svg)                                                                                                                                                                                                                                                                                                                                     |
| Linting        | [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)                                                                                                                                                                                                                                                                                                                                                                                                                            |

## Contents <!-- omit in toc -->

- [Introduction](#introduction)
- [Installation](#installation)
  - [Install from PyPi](#install-from-pypi)
  - [Developer install](#developer-install)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Credits](#credits)

## Installation <!-- --8<-- [start:install] -->

It is advised to install eitprocessing in a dedicated virtual environment. See e.g. [Install packages in a virtual
environment using pip and
venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) or [Getting started
with conda](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html).

### Install from PyPi

eitprocessing can be installed from PyPi as follows:

```bash
pip install eitprocessing
```

### Developer install

For full developer options (testing, etc):

```bash
git clone git@github.com:EIT-ALIVE/eitprocessing.git
cd eitprocessing
pip install -e ".[dev]"
```

<!-- --8<-- [end:install] -->

## Documentation

Please see our [user documentation](https://eit-alive.github.io/eitprocessing/) for a detailed explanation of the package.

## Contributing

We welcome any contributions or suggestions. If you want to contribute to the development of eitprocessing,
have a look at the [contribution guidelines](CONTRIBUTING.md) and the [developer documentation](README.dev.md).

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
