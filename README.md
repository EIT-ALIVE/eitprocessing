## Badges

| fair-software.eu recommendations | |
| :-- | :--  |
| (1/5) code repository              | [![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](git@github.com:EIT-ALIVE/eitprocessing) |
| (2/5) license                      | [![github license badge](https://img.shields.io/github/license/EIT-ALIVE/eitprocessing)](git@github.com:EIT-ALIVE/eitprocessing) |
| (3/5) community registry           | [![RSD](https://img.shields.io/badge/rsd-eitprocessing-00a3e3.svg)](https://www.research-software.nl/software/eitprocessing) [![workflow pypi badge](https://img.shields.io/pypi/v/eitprocessing.svg?colorB=blue)](https://pypi.python.org/project/eitprocessing/) |
| (4/5) citation                     |  [![DOI](https://zenodo.org/badge/617944717.svg)](https://zenodo.org/badge/latestdoi/617944717) |
| (5/5) checklist                    | [![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9147/badge)](https://www.bestpractices.dev/projects/9147) |
| howfairis                          | [![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu) |
| **GitHub Actions**                 | &nbsp; |
| Build                              | [![build](git@github.com:EIT-ALIVE/eitprocessing/actions/workflows/build.yml/badge.svg)](git@github.com:EIT-ALIVE/eitprocessing/actions/workflows/build.yml) |
| Citation data consistency               | [![cffconvert](git@github.com:EIT-ALIVE/eitprocessing/actions/workflows/cffconvert.yml/badge.svg)](git@github.com:EIT-ALIVE/eitprocessing/actions/workflows/cffconvert.yml) |
| MarkDown link checker              | [![markdown-link-check](git@github.com:EIT-ALIVE/eitprocessing/actions/workflows/markdown-link-check.yml/badge.svg)](git@github.com:EIT-ALIVE/eitprocessing/actions/workflows/markdown-link-check.yml) |

## How to use eitprocessing

Processing of data from electrical impedance tomography and other respiratory monitoring tools.

[Electrical Impedance Tomography](https://en.wikipedia.org/wiki/Electrical_impedance_tomography) (EIT) is a noninvasive and radiation-free continuous imaging tool for monitoring respiratory
mechanics.
eitprocessing aims to provide a versatile, user-friendly, reproducible and reliable workflow for the processing and
analysis of EIT data and related waveform data, like pressures and flow.

eitprocessing includes tools to load data exported from EIT-devices from several manufacturers, including Dräger, SenTec and
Timpel, as well as data from other sources. 
Several pre-processing tools and analysis tools are provided. 
<!-- TODO when available, add summarisation and reporting -->
<!-- TODO extend with short list of available tools when applicable -->

[eit_dash](https://github.com/EIT-ALIVE/eit_dash) provides an accompanying GUI. 


## Installation

It is advised to install eitprocessing in a dedicated virtual environment. See e.g. [Install packages in a virtual environment using pip and venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) or
[Getting started with conda](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html). 

eitprocessing can be installed from PyPi as follows:

- Install
  - For basic use: `pip install eitprocessing`
  - For full developer options (testing, etc): `pip install "eitprocessing[dev]"`

## Documentation

We have custom documentation on a dedicated page [here](https://eit-alive.github.io/eitprocessing/)

## Contributing

If you want to contribute to the development of eitprocessing,
have a look at the [contribution guidelines](CONTRIBUTING.md) and the [developer documentation](README.dev.md).

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
