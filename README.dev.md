# `eitprocessing` developer documentation

If you're looking for user documentation, go [here](README.md).


## Contributions

We welcome all contributions to this open-source project, as long as they follow our
[code of conduct](https://github.com/EIT-ALIVE/eitprocessing/blob/main/CODE_OF_CONDUCT.md).
We appreciate it if you adhere to our naming and style conventions below.


### Conventions

#### Readability vs complexity/correctness
While we implement "best practices" as much as possible, it is important to state that sometimes
readibility or simplicity is more important than absolute correctness.
It is hard to define the precise balance we are looking for, so instead we will refer
to the [Zen of python](https://peps.python.org/pep-0020/).

Note that all contrubtions to this project will be published under our [Apache 2.0 licence]
(https://github.com/EIT-ALIVE/eitprocessing/blob/main/LICENSE).

#### Branch naming convention
Please try to adhere to the following branch naming convention:
<github-issue-number>_<brief-description>_<username>.
E.g., `042_life_universe_everything_douglasadams`.

This allows, at a single glance, to see in the issue that you're
addressing, a hint of what the issue is, and who you are.
Also, it simplifies tab autocompletion when switching to your branch.

#### PR naming convention
Please use an [angular convention type](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#type),
followed by a semicolon and then a description when creating a PR.
E.g., `feat: added module to calculate the answer to life, the universe, and everything`


### Creating a PR

#### Code review and continuous integration
All contributions to the project are subject to code review and require at least one
approving review before they can be merged onto the main branch.

We have set up continuous integration for linting and testing, among other things. Please ensure
that all checks pass before requesting code review.

Please create a "draft PR" until your work is ready for review, as this will avoid triggering
the CI prematurely (which uses unnecessary computing power, see [here]
(https://blog.esciencecenter.nl/reduce-reuse-recycle-save-the-planet-one-github-action-at-a-time-4ab602255c3f)).

You can run the [tests](#testing-locally) and [linter](#linting-locally) locally before activating the CI.


#### Testing locally

Make sure you have developer options installed as described in the [README](README.md)
(otherwise run: `pip install -e .[dev]` on the repository folder in your environment)

For testing all you need to do is run:
```shell
pytest
```

Furthermore, you can determine the coverage, i.e. how much of the package's code is actually
executed during tests, by running (after running `pytest`):

```shell
coverage run
# This runs tests and stores the result in a `.coverage` file.
# To see the results on the command line, run
coverage report
```

`coverage` can also generate output in HTML and other formats; see `coverage help` for more information.

#### Linting locally

For linting we will use [prospector](https://pypi.org/project/prospector/).
To sort imports we will use [isort](https://pycqa.github.io/isort/). Note that if you use VS Code,
sorting is automated upon file saving.

```shell
# linter
prospector

# recursively check import style for the eitprocessing module only
isort --check-only eitprocessing

# recursively check import style for the eitprocessing module only and show
# any proposed changes as a diff
isort --check-only --diff eitprocessing

# recursively fix import style for the eitprocessing module only
isort eitprocessing
```


# The following sections are untested
## Generating the API docs

```shell
cd docs
make html
```

The documentation will be in `docs/_build/html`

If you do not have `make` use

```shell
sphinx-build -b html docs docs/_build/html
```

To find undocumented Python objects run

```shell
cd docs
make coverage
cat _build/coverage/python.txt
```

To [test snippets](https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html) in documentation run

```shell
cd docs
make doctest
```

## Versioning

Bumping the version across all files is done with [bumpversion](https://github.com/c4urself/bump2version), e.g.

```shell
bumpversion major
bumpversion minor
bumpversion patch
```

## Making a release

This section describes how to make a release in 3 parts:

1. preparation
1. making a release on PyPI
1. making a release on GitHub

### (1/3) Preparation

1. Update the <CHANGELOG.md> (don't forget to update links at bottom of page)
2. Verify that the information in `CITATION.cff` is correct, and that `.zenodo.json` contains equivalent data
3. Make sure the [version has been updated](#versioning).
4. Run the unit tests with `pytest -v`

### (2/3) PyPI

In a new terminal, without an activated virtual environment or an env directory:

```shell
# prepare a new directory
cd $(mktemp -d eitprocessing.XXXXXX)

# fresh git clone ensures the release has the state of origin/main branch
git clone git@github.com:EIT-ALIVE/eitprocessing .

# prepare a clean virtual environment and activate it
python3 -m venv env
source env/bin/activate

# make sure to have a recent version of pip and setuptools
python3 -m pip install --upgrade pip setuptools

# install runtime dependencies and publishing dependencies
cd eitprocessing
python3 -m pip install --no-cache-dir .
python3 -m pip install --no-cache-dir .[publishing]

# clean up any previously generated artefacts
rm -rf eitprocessing.egg-info
rm -rf dist

# create the source distribution and the wheel
python3 setup.py sdist bdist_wheel

# upload to test pypi instance (requires credentials)
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

Visit
[https://test.pypi.org/project/eitprocessing](https://test.pypi.org/project/eitprocessing)
and verify that your package was uploaded successfully. Keep the terminal open, we'll need it later.

In a new terminal, without an activated virtual environment or an env directory:

```shell
cd $(mktemp -d eitprocessing-test.XXXXXX)

# prepare a clean virtual environment and activate it
python3 -m venv env
source env/bin/activate

# make sure to have a recent version of pip and setuptools
pip install --upgrade pip setuptools

# install from test pypi instance:
python3 -m pip -v install --no-cache-dir \
--index-url https://test.pypi.org/simple/ \
--extra-index-url https://pypi.org/simple eitprocessing
```

Check that the package works as it should when installed from pypitest.

Then upload to pypi.org with:

```shell
# Back to the first terminal,
# FINAL STEP: upload to PyPI (requires credentials)
twine upload dist/*
```

### (3/3) GitHub

Don't forget to also make a [release on GitHub](git@github.com:EIT-ALIVE/eitprocessing/releases/new). If your repository uses the GitHub-Zenodo integration this will also trigger Zenodo into making a snapshot of your repository and sticking a DOI on it.
