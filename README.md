# Schedule optimization for a conference

In this project, we optimize a schedule for a conference based on time/room constraints and attendance preferences of the users.
This project is only a backend in which the actual solving is done.
The interaction with a graphical frontend is done via import/export of `json` files.

The general workflow is:
1. Read in the constraints from a `.json` file. The input format is specified on [this wiki page](https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format#input--output-format).
2. Construct an integer linear program (ILP) from the constraints using [PuLP](https://coin-or.github.io/pulp/). The ILP formulation is described on [this wiki page](https://github.com/Die-KoMa/ak-plan-optimierung/wiki/New-LP-Formulation).
3. Solve the ILP using a solver supported by PuLP, e.g. HiGHS or Gurobi.
4. Output the solution into a `.json` file. The output format is specified on [this wiki page](https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format#input--output-format).


### Setup

To install this project, simply run
```sh
$ pip install git+https://github.com/Die-KoMa/ak-plan-optimierung.git
```
To run the solver, simply call
```sh
$ python -m akplan.solve PATH_TO_JSON_INPUT
```
For a list of available cli options, run `python -m akplan.solve --help`.

### Development setup

For a development setup, clone this repository and run `pip install -e .` in the repository directory.
Further, install the tool [`nox`](https://nox.thea.codes/en/stable/).

To see all available `nox` sessions, run `nox --list`:
```
* test -> Run pytest on all test cases.
* fast-test -> Run pytest on fast test cases.
* lint -> Check code conventions.
* typing -> Check type hints.
* format -> Fix common convention problems automatically.
* coverage -> Check test coverage and generate a html report.
* coverage-clean -> Remove the code coverage website.
```
A session can then be called via `nox -s <session_name>`.
