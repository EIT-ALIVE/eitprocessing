# `eitprocessing` developer documentation

If you're looking for user documentation, go [here](README.md).

## Contributions

We welcome all contributions to this open-source project, as long as they follow our
[code of conduct](https://github.com/EIT-ALIVE/eitprocessing/blob/main/CODE_OF_CONDUCT.md).
We also ask you to adhere to our [naming and style conventions](#conventions).

We appreciate if you follow the steps below. Don't be discouraged if you struggle with any of these: if you feel you
have made or can make a valuable contribution. We are happy to help, so please reach out! Do keep in mind that you might
be asked to append additional commits or make changes to your pull request.

1. announce your plan to the rest of the community _before you start working_. This announcement should be done via GitHub in the form of a (new) [issue](https://github.com/EIT-ALIVE/eitprocessing/issues);
2. wait until some kind of consensus is reached about your idea being a good idea;
3. if needed, fork the repository to your own Github profile and create your own feature branch off of the latest master commit. While working on your feature branch, make sure to stay up to date with the main branch by pulling in changes, possibly from the 'upstream' repository (follow the instructions [here](https://help.github.com/articles/configuring-a-remote-for-a-fork/) and [here](https://help.github.com/articles/syncing-a-fork/));
4. make sure the existing [tests still work](#testing-locally) by running `pytest`;
5. add your own tests (recommended);
6. update or expand the documentation;
7. push your feature branch (or fork) to the eitprocessing repository on GitHub;
8. [create a pull request](https://help.github.com/articles/creating-a-pull-request/), following our [PR conventions]()
   and link it to the issue in step 1;
9. ensure that all automatically generated checks pass and update make changes as required to solve any resulting issues;
   - it can be tricky to discover what some of the problems mean, feel free to reach out if you have difficulties finding out.
10. request a review of your PR once you are happy with its state or if you require feedback.

Note that all contrubtions to this project will be published under our [Apache 2.0 licence]
(<https://github.com/EIT-ALIVE/eitprocessing/blob/main/LICENSE>).

### Conventions

#### Readability vs complexity/correctness

While we implement "best practices" as much as possible, it is important to state that sometimes
readibility or simplicity is more important than absolute correctness.
It is hard to define the precise balance we are looking for, so instead we will refer
to the [Zen of python](https://peps.python.org/pep-0020/).

#### Linting and formatting

We use the [Ruff formatter](https://pypi.org/project/black/) to format code. If you use Visual
Studio Code, the [extension by
Microsoft](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter) is a good
place to start. This extension is currently in preview, but seems to work more reliably than older implementations.

#### Docstrings

We use the [google convention](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
for writing docstrings.

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
5. On the [Releases page](https://github.com/EIT-ALIVE/eitprocessing/releases):
   1. Click "Draft a new release"
   2. By convention, use `v<version number>` as both the release title and as a tag for the release.
   3. Click "Generate release notes" to automatically load release notes from merged PRs since the last release.
   4. Adjust the notes as required
   5. Ensure that "Set as latest release" is checked and that both other boxes are unchecked.
   6. Hit "Publish release". This will automatically trigger [the GitHub action](https://github.com/EIT-ALIVE/eitprocessing/actions/workflows/release.yml) that will take care of publishing the package on PyPi.
