"""Solving the MILPs for conference scheduling."""

import argparse
import json
import multiprocessing
from collections.abc import Iterable
from dataclasses import asdict
from itertools import combinations, product, repeat
from pathlib import Path
from typing import Any, Literal, TypeVar, cast, overload

from pulp import (
    LpConstraint,
    LpMaximize,
    LpProblem,
    LpSolution,
    LpSolutionInfeasible,
    LpSolutionNoSolutionFound,
    LpStatus,
    LpStatusInfeasible,
    getSolver,
    lpSum,
)

from . import constraints
from .util import (
    LPVarDicts,
    PartialSolvedVarDict,
    ProblemIds,
    ProblemProperties,
    ScheduleAtom,
    SchedulingInput,
    SolvedVarDict,
    VarDict,
    _construct_constraint_name,
)

T = TypeVar("T")


def create_lp(
    input_data: SchedulingInput,
    output_file: str | None = "koma-plan.lp",
    show_progress: bool = False,
    chunksize: int = 1,
) -> tuple[LpProblem, dict[str, VarDict]]:
    """Create the MILP problem as pulp object.

    Creates the problem with all constraints, preferences and the objective function.

    For a specification of the input JSON format, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format

    For a specification of the MILP, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/LP-formulation

    The MILP models each person to have three kinds of prefences for an AK:
    0 (no preference), 1 (weak preference) and `mu` (strong preference).
    The choice of `mu` is an hyperparameter of the MILP that weights the
    balance between weak and strong preferences.

    Args:
        input_data (SchedulingInput): The input data used to construct the MILP.
        output_file (str, optional): If not None, the created LP is written
            as an `.lp` file to this location. Defaults to `koma-plan.lp`.
        show_progress (bool): Whether progress bars should be displayed.
            Defaults to False.
        chunksize (int): The size of chunks that are passed to processes of the
            multiprocessing Pool. Defaults to 1.


    Returns:
        A tuple (`lp_problem`, `dec_vars`) where `lp_problem` is the
        constructed MILP instance and `dec_vars` is the nested dictionary
        containing the MILP variables.
    """
    ids = ProblemIds.init_from_problem(input_data)
    props = ProblemProperties.init_from_problem(input_data, ids=ids)

    # Create decision variables
    var = LPVarDicts.init_from_ids(
        ak_ids=ids.ak,
        room_ids=ids.room,
        timeslot_ids=ids.timeslot,
        block_ids=ids.block_dict.keys(),
        person_ids=ids.person,
    )

    # Create problem
    prob = LpProblem("MLPKoMa", sense=LpMaximize)

    # Set objective function
    # \sum_{P,A} \frac{P_{P,A}}{\sum_{P_{P,A}}\neq 0} T_{P,A}
    prob += (
        lpSum(
            [
                pref * var.person[ak_id][person_id] / len(preferences)
                for person_id, preferences in props.weighted_preferences.items()
                for ak_id, pref in preferences.items()
            ]
        ),
        "cost_function",
    )

    tasks: list[constraints.TaskItem[Any]] = []

    def _create_task_item(
        func: constraints.ConstraintFunc[T], params: Iterable[T]
    ) -> None:
        """Construct task item and add it to the list. Used to facilitate type checking."""
        task_item: constraints.TaskItem[T] = (func, params)
        tasks.append(task_item)

    _create_task_item(
        constraints._max_one_ak_per_person_and_time,
        zip(repeat(var), product(combinations(ids.ak, 2), ids.timeslot, ids.person)),
    )
    _create_task_item(
        constraints._max_one_ak_per_room_and_time,
        zip(repeat(var), product(combinations(ids.ak, 2), ids.timeslot, ids.room)),
    )
    _create_task_item(
        constraints._ak_durations, zip(repeat(var), repeat(props), ids.ak)
    )
    _create_task_item(constraints._ak_single_block, zip(repeat(var), ids.ak))
    _create_task_item(
        constraints._ak_single_block_per_block,
        zip(repeat(var), repeat(props), product(ids.ak, ids.block_dict.items())),
    )
    _create_task_item(constraints._at_most_one_room_per_ak, zip(repeat(var), ids.ak))
    _create_task_item(constraints._at_least_one_room_per_ak, zip(repeat(var), ids.ak))
    _create_task_item(
        constraints._not_more_people_than_interested,
        zip(repeat(var), repeat(props), ids.ak),
    )
    _create_task_item(
        constraints._room_sizes,
        zip(repeat(var), repeat(props), product(ids.room, ids.ak)),
    )
    _create_task_item(
        constraints._time_impossible_for_person,
        zip(repeat(var), product(ids.person, ids.timeslot, ids.ak)),
    )
    _create_task_item(constraints._room_for_ak, zip(repeat(var), ids.ak))
    _create_task_item(
        constraints._time_impossible_for_room,
        zip(repeat(var), repeat(props), product(ids.room, ids.timeslot, ids.ak)),
    )
    _create_task_item(
        constraints._ak_conflict,
        zip(repeat(var), product(ids.timeslot, ids.conflict_pairs)),
    )

    # INITIAL VALUES #

    # PersonNotInterestedInAK
    # For all A, Z, R, P: If P_{P, A} = 0: Y_{A,Z,R,P} = 0 (non-dummy P)
    for person_id, preferences in props.weighted_preferences.items():
        # aks not in pref_aks have P_{P,A} = 0 implicitly
        for ak_id in ids.ak.difference(preferences.keys()):
            var.person[ak_id][person_id].setInitialValue(0)
            var.person[ak_id][person_id].fixValue()
    for ak_id, persons in props.required_persons.items():
        for person_id in persons:
            var.person[ak_id][person_id].setInitialValue(1)
            var.person[ak_id][person_id].fixValue()

    # TimeImpossibleForPerson
    # Real person P cannot attend AKs with timeslot Z
    for person_id, timeslot_id in product(ids.person, ids.timeslot):
        if props.participant_time_constraints[person_id].difference(
            props.fulfilled_time_constraints[timeslot_id]
        ):
            var.person_time[person_id][timeslot_id].setInitialValue(0)
            var.person_time[person_id][timeslot_id].fixValue()

    # TimeImpossibleForAK
    for ak_id, timeslot_id in product(ids.ak, ids.timeslot):
        if props.ak_time_constraints[ak_id].difference(
            props.fulfilled_time_constraints[timeslot_id]
        ):
            var.time[ak_id][timeslot_id].setInitialValue(0)
            var.time[ak_id][timeslot_id].fixValue()

    # RoomImpossibleForAK
    for ak_id, room_id in product(ids.ak, ids.room):
        if props.ak_room_constraints[ak_id].difference(
            props.fulfilled_room_constraints[room_id]
        ):
            var.room[ak_id][room_id].setInitialValue(0)
            var.room[ak_id][room_id].fixValue()

    # COMPLEX LOGIC CONSTRAINTS #

    for ak_id, (block_id, block) in product(ids.ak, ids.block_dict.items()):
        # AKContiguous
        for timeslot_idx, timeslot_id_a in enumerate(block):
            for timeslot_id_b in block[timeslot_idx + props.ak_durations[ak_id] :]:
                prob += lpSum(
                    [var.time[ak_id][timeslot_id_a], var.time[ak_id][timeslot_id_b]]
                ) <= 1, _construct_constraint_name(
                    "AKContiguous",
                    ak_id,
                    str(block_id),
                    timeslot_id_a,
                    timeslot_id_b,
                )

    # RoomImpossibleForPerson
    # Real person P cannot attend AKs with room R
    for person_id, room_id in product(ids.person, ids.room):
        if props.participant_room_constraints[person_id].difference(
            props.fulfilled_room_constraints[room_id]
        ):
            for ak_id in ids.ak:
                constraint_sum = lpSum(
                    [var.room[ak_id][room_id], var.person[ak_id][person_id]]
                )
    prob.extend

    if input_data.config.max_num_timeslots_before_break > 0:
        # PersonNeedsBreak
        # Any real person needs a break after some number of time slots
        # So in each block at most  consecutive timeslots can be active for any person
        for person_id, (_block_id, block) in product(
            ids.person, ids.block_dict.items()
        ):
            for i in range(
                len(block) - input_data.config.max_num_timeslots_before_break - 1
            ):
                sum_of_vars = lpSum(
                    [
                        var.person_time[person_id][timeslot_id]
                        for timeslot_id in block[
                            i : i + input_data.config.max_num_timeslots_before_break + 1
                        ]
                    ]
                )

                prob += (
                    sum_of_vars <= input_data.config.max_num_timeslots_before_break,
                    _construct_constraint_name("BreakForPerson", person_id, block[i]),
                )

    # AK dependencies
    for ak in input_data.aks:
        other_ak_ids = ak.properties.get("dependencies", [])
        for other_ak_id, (idx, timeslot_id) in product(
            other_ak_ids, enumerate(ids.sorted_timeslot)
        ):
            constraint_sum = lpSum(
                [
                    var.time[ak.id][succ_timeslot_id]
                    for succ_timeslot_id in ids.sorted_timeslot[idx:]
                ]
            )
            constraint = var.time[other_ak_id][timeslot_id] <= constraint_sum

            prob += constraint, _construct_constraint_name(
                "AKDependenciesDoneBeforeAK", ak.id, timeslot_id, other_ak_id
            )

    all_constraints: list[tuple[str, LpConstraint]] = []

    with multiprocessing.Pool(processes=4) as pool:
        for task_func, task_params in tasks:
            for result in pool.imap_unordered(
                task_func, task_params, chunksize=chunksize
            ):
                if result is not None:
                    all_constraints.append(result)

    prob.extend(all_constraints)

    # Fix Values for already scheduled aks
    for scheduled_ak in input_data.scheduled_aks:
        if scheduled_ak.room_id is not None:
            var.room[scheduled_ak.ak_id][scheduled_ak.room_id].setInitialValue(1)
            if not input_data.config.allow_changing_rooms:
                var.room[scheduled_ak.ak_id][scheduled_ak.room_id].fixValue()
        for person_id in scheduled_ak.participant_ids:
            var.person[scheduled_ak.ak_id][person_id].setInitialValue(1)
            var.person[scheduled_ak.ak_id][person_id].fixValue()
        for timeslot_id in scheduled_ak.timeslot_ids:
            var.time[scheduled_ak.ak_id][timeslot_id].setInitialValue(1)
            var.time[scheduled_ak.ak_id][timeslot_id].fixValue()

    # The problem data is written to an .lp file
    if output_file is not None:
        prob.writeLP(output_file)

    return prob, var.to_export_dict()


def export_scheduling_result(
    input_data: SchedulingInput,
    solution: dict[str, SolvedVarDict],
    allow_unscheduled_aks: bool = False,
) -> dict[int, ScheduleAtom]:
    """Create a dictionary from the solved MILP.

    For a specification of the output format, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format

    Args:
        input_data(SchedulingInput): The Scheduling instance.
        solution: Nested dicts with the values of the decision variables
            relevant to generate the output.
        allow_unscheduled_aks (bool): Whether not scheduling an AK is allowed or not.
            Defaults to False.

    Returns:
        dict: The constructed output dict (as specified).
    """
    ids = ProblemIds.init_from_problem(input_data)

    @overload
    def _get_id(
        ak_id: int, var_key: str, allow_multiple: Literal[True], allow_none: bool
    ) -> list[int]: ...

    @overload
    def _get_id(
        ak_id: int, var_key: str, allow_multiple: Literal[False], allow_none: bool
    ) -> int | None: ...

    def _get_id(
        ak_id: int, var_key: str, allow_multiple: bool, allow_none: bool
    ) -> int | list[int] | None:
        matched_ids = [id for id, val in solution[var_key][ak_id].items() if val > 0]
        if not allow_multiple and len(matched_ids) > 1:
            raise ValueError(f"AK {ak_id} is assigned multiple {var_key}")
        elif len(matched_ids) == 0 and not allow_none:
            raise ValueError(f"no {var_key} assigned to ak {ak_id}")
        else:
            if allow_multiple:
                return matched_ids
            else:
                return matched_ids[0] if len(matched_ids) > 0 else None

    scheduled_ak_dict: dict[int, ScheduleAtom] = {
        ak_id: ScheduleAtom(
            ak_id=ak_id,
            room_id=_get_id(
                ak_id=ak_id,
                var_key="Room",
                allow_multiple=False,
                allow_none=allow_unscheduled_aks,
            ),
            timeslot_ids=_get_id(
                ak_id=ak_id,
                var_key="Time",
                allow_multiple=True,
                allow_none=allow_unscheduled_aks,
            ),
            participant_ids=_get_id(
                ak_id=ak_id, var_key="Part", allow_multiple=True, allow_none=True
            ),
        )
        for ak_id in ids.ak
    }

    return scheduled_ak_dict


def solve_scheduling(
    input_data: SchedulingInput,
    solver_name: str | None = None,
    output_lp_file: str | None = "koma-plan.lp",
    show_progress: bool = False,
    **solver_kwargs: dict[str, Any],
) -> tuple[LpProblem, dict[str, PartialSolvedVarDict]]:
    """Solve the scheduling problem.

    Solves the ILP scheduling problem described by the input data using an ILP
    formulation.

    For a specification of the input format, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format

    For a specification of the ILP used, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/New-LP-formulation

    The ILP models each person to have three kinds of prefences for an AK:
    0 (no preference), 1 (weak preference) and `mu` (strong preference).
    The choice of `mu` is an hyperparameter of the ILP that weights the
    balance between weak and strong preferences.

    Args:
        input_data (SchedulingInput): The input data used to construct the ILP.
        solver_name (str, optional): The solver to use. If None, uses pulp's
            default solver. Defaults to None.
        output_lp_file (str, optional): If not None, the created LP is written
            as an `.lp` file to this location. Defaults to `koma-plan.lp`.
        show_progress (bool): Whether progress bars should be displayed during
            LP creation. Defaults to False.
        **solver_kwargs: kwargs are passed to the solver.

    Returns:
        A tuple (`lp_problem`, `solution`) where `lp_problem` is the
        constructed and solved MILP instance and `solution` contains
        the nested dicts with the solution.
    """
    lp_problem, dec_vars = create_lp(
        input_data, output_lp_file, show_progress=show_progress
    )

    if not solver_name:
        # The problem is solved using PuLP's default solver
        solver_name = "PULP_CBC_CMD"
    solver = getSolver(solver_name, **solver_kwargs)
    lp_problem.solve(solver)

    print("Status:", LpStatus[lp_problem.status])
    print("Solution Status:", LpSolution[lp_problem.sol_status])
    if lp_problem.status == LpStatusInfeasible and output_lp_file is not None:
        if lp_problem.solver and lp_problem.solver.name == "GUROBI":
            # compute irreducible inconsistent subsystem for debugging, cf.
            # https://www.gurobi.com/documentation/current/refman/py_model_computeiis.html
            lp_problem.solverModel.computeIIS()
            iis_path = Path(output_lp_file)
            iis_path = iis_path.parent / f"{iis_path.stem}-iis.ilp"
            lp_problem.solverModel.write(str(iis_path))

    def value_processing(value: float | None) -> int | None:
        if value is None:
            return None
        return round(value)

    solution = {
        var_key: {
            ak_id: {id: value_processing(var.value()) for id, var in vars.items()}
            for ak_id, vars in vars_dict.items()
        }
        for var_key, vars_dict in dec_vars.items()
    }

    return (lp_problem, solution)


def _check_for_partial_solve(
    solution: dict[str, PartialSolvedVarDict],
    solved_lp_problem: LpProblem,
) -> dict[str, SolvedVarDict]:
    if any(None in d.values() for dd in solution.values() for d in dd.values()):
        print(
            "Warning: some variables are not assigned a value "
            f"with solution status {solved_lp_problem.sol_status}."
        )
        raise ValueError
    return cast(dict[str, SolvedVarDict], solution)


def process_solved_lp(
    solved_lp_problem: LpProblem,
    solution: dict[str, PartialSolvedVarDict],
    input_data: SchedulingInput,
) -> dict[int, ScheduleAtom] | None:
    """Process the solved LP model and create a schedule output.

    Args:
        solved_lp_problem (LpProblem): The pulp LP problem object after the optimizer ran.
        solution (nested dict containing the MILP variables): The solution to the problem.
        input_data (SchedulingInput): The input data used to construct the ILP.
            If set, the input data is added to the output schedule dict. Defaults to None.

    Returns:
        A dictionary containing the scheduled aks as well as the scheduling
        input.
    """
    if solved_lp_problem.sol_status in [
        LpSolutionInfeasible,
        LpSolutionNoSolutionFound,
    ]:
        return None

    try:
        checked_solution = _check_for_partial_solve(solution, solved_lp_problem)
    except ValueError:
        return None

    return export_scheduling_result(
        input_data,
        checked_solution,
        allow_unscheduled_aks=input_data.config.allow_unscheduled_aks,
    )


def main() -> None:
    """Run solve_scheduling from CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver",
        type=str,
        default=None,
        help="The solver to use. If None, uses pulp's default solver. Defaults to None.",
    )
    parser.add_argument(
        "--solver-path",
        type=str,
        help="If set, this value is passed as the `path` argument to the solver.",
    )
    parser.add_argument(
        "--warm-start",
        action="store_true",
        default=False,
        help="If set, passes the `warmStart` flag to the solver.",
    )
    parser.add_argument(
        "--timelimit",
        type=float,
        default=None,
        help="Timelimit as stopping criterion (in seconds)",
    )
    parser.add_argument(
        "--gap_rel", type=float, default=None, help="Relative gap as stopping criterion"
    )
    parser.add_argument(
        "--gap_abs", type=float, default=None, help="Absolute gap as stopping criterion"
    )
    parser.add_argument(
        "--threads", type=int, default=None, help="Number of threads to use"
    )
    parser.add_argument(
        "path", type=str, help="Path of the JSON input file to the solver."
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for the solver")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Json File to output the calculated schedule to. If not specified, "
        "the prefix 'out-' is added to the input file name and it is stored in the "
        "current working directory.",
    )
    parser.add_argument(
        "--override-output",
        action="store_true",
        help="If set, overrides the output file if it exists.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="If set, shows progress bars during LP generation.",
    )
    args = parser.parse_args()

    solver_kwargs = {}
    if args.solver_path:
        solver_kwargs["path"] = args.solver_path
    if args.warm_start:
        solver_kwargs["warmStart"] = True
    if args.timelimit:
        solver_kwargs["timeLimit"] = args.timelimit
    if args.gap_rel:
        solver_kwargs["gapRel"] = args.gap_rel
    if args.gap_abs:
        solver_kwargs["gapAbs"] = args.gap_abs
    if args.threads:
        solver_kwargs["threads"] = args.threads
    # if args.seed:
    #     solver_kwargs["RandomC"] = args.seed

    json_file = Path(args.path)
    assert json_file.suffix == ".json"
    with json_file.open("r") as f:
        input_dict = json.load(f)

    if args.output is None:
        args.output = Path.cwd() / f"out-{json_file.name}"

    if args.output.exists() and not args.override_output:
        raise ValueError(
            f"Output file {args.output} already exists. We do not simply override it."
        )

    # create directory tree
    args.output.parent.mkdir(exist_ok=True, parents=True)

    scheduling_input = SchedulingInput.from_dict(input_dict)

    solved_lp_problem, solution = solve_scheduling(
        scheduling_input,
        args.solver,
        show_progress=args.progress,
        **solver_kwargs,
    )

    schedule = process_solved_lp(
        solved_lp_problem,
        solution,
        input_data=scheduling_input,
    )

    # here we replace the old scheduled aks in the input
    # because these are also part of the produced schedule
    if schedule is not None:
        out_dict = {
            "scheduled_aks": list(map(asdict, schedule.values())),
            "input": scheduling_input.to_dict(),
        }
        with args.output.open("w") as ff:
            json.dump(out_dict, ff)
        print(f"Stored result at {args.output}")


if __name__ == "__main__":
    main()
