# `eitprocessing` developer documentation

If you're looking for user documentation, go [here](README.md).

## Contributions

We welcome all contributions to this open-source project, as long as they follow our
[code of conduct](https://github.com/EIT-ALIVE/eitprocessing/blob/main/CODE_OF_CONDUCT.md).
We appreciate it if you adhere to our naming and style [conventions](#conventions) below.

Please follow these steps:

1. (**important**) announce your plan to the rest of the community _before you start working_. This announcement should be in the form of a (new) issue;
1. (**important**) wait until some kind of consensus is reached about your idea being a good idea;
1. if needed, fork the repository to your own GitHub profile and create your own feature branch off of the latest master commit. While working on your feature branch, make sure to stay up to date with the master branch by pulling in changes, possibly from the 'upstream' repository (follow the instructions [here](https://help.github.com/articles/configuring-a-remote-for-a-fork/) and [here](https://help.github.com/articles/syncing-a-fork/));
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

Note that all contributions to this project will be published under our [Apache 2.0 licence]
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
`<github-issue-number>_<brief-description>_<username>`.
E.g., `042_life_universe_everything_douglasadams`.

This allows, at a single glance, to see in the issue that you're
addressing, a hint of what the issue is, and who you are.
Also, it simplifies tab autocompletion when switching to your branch.

#### PR naming convention

Please use an [angular convention type](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#type),
followed by a semicolon and then a description when creating a PR.
E.g., `feat: added module to calculate the answer to life, the universe, and everything`.

### Creating a PR

#### Branching strategy

We use a workflow where `main` always contains the latest stable release version of `eitprocessing` and where `develop` contains the next release version under construction.

When creating a new feature, one should branch from `develop`.
When a feature is finished, a PR to pull the feature into `develop` should be created. After one or multiple features
have been pulled into `develop`, the [release workflow](#making-a-release) can be triggered to automatically create the
new feature (minor) release originating from `develop`.

For bug fixes that can't wait on a feature release, one should branch from `main`.
When the bug fix is finished, the [release workflow](#making-a-release) can be triggered originating from
the created branch, usually with a patch update.

In principle, no releases should originate from branches other than `develop` and bug fix branches.

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

##### Downloading test data
Some tests require access to test data. You can download the test data from Zenodo via the button below. Note that for
some reason downloading all files at ones results in a corrupted zip file. Please download the files one by one.

[![](https://zenodo.org/badge/DOI/10.5281/zenodo.17423608.svg)](https://doi.org/10.5281/zenodo.17423608)

Test data should reside in the `test_data/` folder in the root of the repository.

Alternatively, use zenodo-get to download the data directly into the `test_data/` folder:

Using `uv`:

```shell
mkdir -p test_data
cd test_data
uv tool run zenodo_get 17423608
cd -
```

Using `pip`:

```shell
pip install zenodo-get
mkdir -p test_data
cd test_data
zenodo_get 17423608
cd -
```


##### Running tests

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

If you are using VS code, please install and activate the [Ruff
extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) to automatically format and check
linting. Make sure to use the ruff version installed in your environment.

Otherwise, please ensure check both linting (`ruff fix .`) and formatting (`ruff format .`) before requesting a review.

We use [prettier](https://prettier.io/) for formatting most other files. If you are editing or adding non-python files and using VS code, the [Prettier extension](https://marketplace.visualstudio.com/items?itemName=esbenp.prettier-vscode) can be installed to auto-format these files as well.

## Making a release

### Automated release workflow

0. **IMP0RTANT:** [Create a PR](#creating-a-pr) for the release branch (usually `develop`) and make sure that all checks pass!
   - if everything goes well, this PR will automatically be closed after the draft release is created.
1. Navigate to [Draft Github Release](https://github.com/EIT-ALIVE/eitprocessing/actions/workflows/release_github.yml)
   on the [Actions](https://github.com/EIT-ALIVE/eitprocessing/actions) tab.
2. On the right hand side, you can select the level increase ("patch", "minor", or "major") and which branch to release from.
   - [Follow semantic versioning conventions](https://semver.org/) to chose the level increase:
     - `patch`: when backward compatible bug fixes were made
     - `minor`: when functionality was added in a backward compatible manner
     - `major`: when API-incompatible changes have been made
   - If the release branch is not `develop`, the workflow will attempt to merge the changes into develop as well. If
     succesfull, the release branch will be deleted from the remote repository.
   - Note that you cannot release from `main` (the default shown) using the automated workflow. To release from `main`
     directly, you must [create the release manually](#manually-create-a-release).
3. Visit [Actions](https://github.com/EIT-ALIVE/eitprocessing/actions) tab to check whether everything went as expected.
4. Navigate to the [Releases](https://github.com/EIT-ALIVE/eitprocessing/releases) tab and click on the newest draft
   release that was just generated.
5. Click on the edit (pencil) icon on the right side of the draft release.
6. Check/adapt the release notes and make sure that everything is as expected.
7. Check that "Set as the latest release is checked".
8. Click green "Publish Release" button to convert the draft to a published release on GitHub.
   - This will automatically trigger [another GitHub workflow](https://github.com/EIT-ALIVE/eitprocessing/actions/workflows/release.yml) that will take care of publishing the package on PyPi.

#### Updating the token

NOTE: the current token (associated to @DaniBodor) allowing to bypass branch protection will expire on June 20th, 2025. To update the token do the following:

1. [Create a personal access token](https://github.com/settings/tokens/new) from a GitHub user account with admin
   priviliges for this repo.
2. Check all the "repo" boxes and the "workflow" box, set an expiration date, and give the token a note.
3. Click green "Generate token" button on the bottom
4. Copy the token immediately, as it will not be visible again later.
5. Navigate to the [secrets settings](https://github.com/EIT-ALIVE/eitprocessing/settings/secrets/actions).
6. Edit the `GH_PAT` key giving your access token as the new value.

### Manually create a release

0. Make sure you have all required developers tools installed `pip install -e .'[dev]'`.
1. Create a `release` branch from `main` and merge the changes into this branch.
   - Ensure that the `release` branch is ready to be merged back into `main` (e.g., removing the unnecessary files, fix minor bugs if necessary).
   - Also see our [branching strategy](#branching-strategy) above.
2. Ensure all tests pass `pytest -v` and that linting (`ruff check`) and formatting (`ruff format --check`) conventions
   are adhered to.
3. Bump the version using [bump-my-version](https://github.com/callowayproject/bump-my-version): `bump-my-version bump <level>`
   where level must be one of the following ([following semantic versioning conventions](https://semver.org/)):
   - `major`: when API-incompatible changes have been made
   - `minor`: when functionality was added in a backward compatible manner
   - `patch`: when backward compatible bug fixes were made
4. Merge the release branch into `main` and `develop`.
5. On the [Releases page](https://github.com/EIT-ALIVE/eitprocessing/releases):
   1. Click "Draft a new release"
   2. By convention, use `v<version number>` as both the release title and as a tag for the release.
   3. Click "Generate release notes" to automatically load release notes from merged PRs since the last release.
   4. Adjust the notes as required.
   5. Ensure that "Set as latest release" is checked and that both other boxes are unchecked.
   6. Hit "Publish release".
      - This will automatically trigger a [GitHub
        workflow](https://github.com/EIT-ALIVE/eitprocessing/actions/workflows/release.yml) that will take care of publishing
        the package on PyPi.
