# `eitprocessing` developer documentation

If you're looking for user documentation, go [here](README.md).

## Contributions

We welcome all contributions to this open-source project, as long as they follow our
[code of conduct](https://github.com/EIT-ALIVE/eitprocessing/blob/main/CODE_OF_CONDUCT.md).
We appreciate it if you adhere to our naming and style [conventions](#conventions) below.

Please follow these steps:

1. (**important**) announce your plan to the rest of the community _before you start working_. This announcement should be in the form of a (new) issue;
1. (**important**) wait until some kind of consensus is reached about your idea being a good idea;
1. if needed, fork the repository to your own Github profile and create your own feature branch off of the latest master commit. While working on your feature branch, make sure to stay up to date with the master branch by pulling in changes, possibly from the 'upstream' repository (follow the instructions [here](https://help.github.com/articles/configuring-a-remote-for-a-fork/) and [here](https://help.github.com/articles/syncing-a-fork/));
1. make sure the existing tests still work by running `pytest` (see also [here](#testing-locally));
1. add your own tests (if necessary);
1. update or expand the documentation;
1. update the `CHANGELOG.md` file with change;
1. push your feature branch to (your fork of) the eitprocessing repository on GitHub;
1. create the pull request, e.g. following the instructions [here](https://help.github.com/articles/creating-a-pull-request/).

In case you feel like you've made a valuable contribution, but you don't know how to write or run tests for it, or how to generate the documentation: don't let this discourage you from making the pull request; we can help you! Just go ahead and submit the pull request, but keep in mind that you might be asked to append additional commits to your pull request.

### Conventions

#### Readability vs complexity/correctness

While we implement "best practices" as much as possible, it is important to state that sometimes
readibility or simplicity is more important than absolute correctness.
It is hard to define the precise balance we are looking for, so instead we will refer
to the [Zen of python](https://peps.python.org/pep-0020/).

Note that all contrubtions to this project will be published under our [Apache 2.0 licence]
(<https://github.com/EIT-ALIVE/eitprocessing/blob/main/LICENSE>).

#### Docstrings

We use the [google convention](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
for writing docstrings.

#### Code formatting

We use the [Black formatter](https://pypi.org/project/black/) to format code. If you use Visual
Studio Code, the [extension by
Microsoft](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter) is a good
place to start. This extension is currently in preview, but seems to work more reliably than older implementations.

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
E.g., `feat: added module to calculate the answer to life, the universe, and everything`.

### Creating a PR

#### Code review and continuous integration

All contributions to the project are subject to code review and require at least one
approving review before they can be merged onto the main branch.

We have set up continuous integration for linting and testing, among other things. Please ensure
that all checks pass before requesting code review.

Please create a "draft PR" until your work is ready for review, as this will avoid triggering
the CI prematurely (which uses unnecessary computing power, see [here](https://blog.esciencecenter.nl/reduce-reuse-recycle-save-the-planet-one-github-action-at-a-time-4ab602255c3f)).

You can run the [tests](#testing-locally) and [linter](#linting-and-formatting) locally before activating
the CI.

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

#### Linting and Formatting

We use [ruff](https://docs.astral.sh/ruff/) for linting, sorting imports and formatting of python (notebook) files. The configurations of `ruff` are set in [pyproject.toml](pyproject.toml) file.

If you are using VS code, please install and activate the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) to automatically format and check linting.

Otherwise, please ensure check both linting (`ruff fix .`) and formatting (`ruff format .`) before requesting a review.

We use [prettier](https://prettier.io/) for formatting most other files. If you are editing or adding non-python files and using VS code, the [Prettier extension](https://marketplace.visualstudio.com/items?itemName=esbenp.prettier-vscode) can be installed to auto-format these files as well.

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

## Making a release

0. Make sure you have all required developers tools installed `pip install -e .'[dev]'`
1. Branch from `main` and prepare the branch for the release (e.g., removing the unnecessary files, fix minor bugs if necessary).
2. Ensure all tests pass `pytest -v` and that linting (`ruff check`) and formatting (`ruff format --check`) conventions
   are adhered to.
3. Bump the version using [bumpversion](https://github.com/c4urself/bump2version): `bumpversion <level>`
   where level must be one of the following ([following semantic versioning conventions](https://semver.org/)):
   - `major`: when API-incompatible changes have been made
   - `minor`: when functionality was added in a backward compatible manner
   - `path`: when backward compatible bug fixes were made
4. Merge the release branch into `main`.
5. Go to https://github.com/EIT-ALIVE/eitprocessing/releases and draft a new release; create a new tag for the release, generate release notes automatically and adjust them, and finally publish the release as latest. This will trigger [a GitHub action](https://github.com/EIT-ALIVE/eitprocessing/actions/workflows/release.yml) that will take care of publishing the package on PyPi.
