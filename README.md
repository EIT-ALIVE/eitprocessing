## Badges

(Customize these badges with your own links, and check https://shields.io/ or https://badgen.net/ to see which other badges are available.)

| fair-software.eu recommendations | |
| :-- | :--  |
| (1/5) code repository              | [![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](git@github.com:EIT-ALIVE/eitprocessing) |
| (2/5) license                      | [![github license badge](https://img.shields.io/github/license/EIT-ALIVE/eitprocessing)](git@github.com:EIT-ALIVE/eitprocessing) |
| (3/5) community registry           | [![RSD](https://img.shields.io/badge/rsd-eitprocessing-00a3e3.svg)](https://www.research-software.nl/software/eitprocessing) [![workflow pypi badge](https://img.shields.io/pypi/v/eitprocessing.svg?colorB=blue)](https://pypi.python.org/project/eitprocessing/) |
| (4/5) citation                     |  [![DOI](https://zenodo.org/badge/617944717.svg)](https://zenodo.org/badge/latestdoi/617944717) |
| (5/5) checklist                    | [![workflow cii badge](https://bestpractices.coreinfrastructure.org/projects/<replace-with-created-project-identifier>/badge)](https://bestpractices.coreinfrastructure.org/projects/<replace-with-created-project-identifier>) |
| howfairis                          | [![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu) |
| **Other best practices**           | &nbsp; |
| Documentation                      | [![Documentation Status](https://readthedocs.org/projects/eitprocessing/badge/?version=latest)](https://eitprocessing.readthedocs.io/en/latest/?badge=latest) |
| **GitHub Actions**                 | &nbsp; |
| Build                              | [![build](git@github.com:EIT-ALIVE/eitprocessing/actions/workflows/build.yml/badge.svg)](git@github.com:EIT-ALIVE/eitprocessing/actions/workflows/build.yml) |
| Citation data consistency               | [![cffconvert](git@github.com:EIT-ALIVE/eitprocessing/actions/workflows/cffconvert.yml/badge.svg)](git@github.com:EIT-ALIVE/eitprocessing/actions/workflows/cffconvert.yml) |
| MarkDown link checker              | [![markdown-link-check](git@github.com:EIT-ALIVE/eitprocessing/actions/workflows/markdown-link-check.yml/badge.svg)](git@github.com:EIT-ALIVE/eitprocessing/actions/workflows/markdown-link-check.yml) |

## How to use eitprocessing

Processing of lung image data from electrical impedance tomography

The project setup is documented in [project_setup.md](project_setup.md). Feel free to remove this document (and/or the link to this document) if you don't need it.

## Installation

Install eitprocessing as follows:
- Create fresh environment
  - Make sure you are in your base environment: `conda activate`
  - Create a new environment: `conda create -n <envname> python=3.10`
  - Activate new environment: `conda activate <envname>`
- Clone and install
  - Clone the repository: `git clone git@github.com:EIT-ALIVE/eitprocessing.git`
  - Install:
    - For basic use: `pip install -e .`
    - For full developer options (testing, etc): `pip install -e .[dev]`

## Documentation

Include a link to your project's full documentation here.

## Contributing

If you want to contribute to the development of eitprocessing,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
