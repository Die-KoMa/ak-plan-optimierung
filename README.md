# Schedule optimization for a conference

In this project, we optimize a schedule for a conference based on time/room constraints and attendance preferences of the users.
This project is only a backend in which the actual solving is done.
The interaction with a graphical frontend is done via import/export of `json` files.

The general workflow is:
1. Read in the constraints from a `.json` file. The input format is specified on [this wiki page](https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format#input--output-format).
2. Construct an integer linear program (ILP) from the constraints using [PuLP](https://coin-or.github.io/pulp/). The ILP formulation is described on [this wiki page](https://github.com/Die-KoMa/ak-plan-optimierung/wiki/LP-formulation).
3. Solve the ILP using a solver supported by PuLP, e.g. HiGHS or Gurobi.
4. Output the solution into a `.json` file. The output format is specified on [this wiki page](https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format#input--output-format).


### Setup

To run `pulp_solve.py` you need to install [PuLP](https://coin-or.github.io/pulp/) and possibly the actual solver you want to run.
