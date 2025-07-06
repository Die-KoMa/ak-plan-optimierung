"""Solving the MILPs for conference scheduling."""

import argparse
import json
import logging
from collections.abc import Iterable
from dataclasses import asdict
from itertools import combinations, product
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, TypeVar, cast, get_args, overload

import linopy
import numpy as np
import pandas as pd
import xarray as xr

from . import types
from .util import (
    ProblemIds,
    ProblemProperties,
    ScheduleAtom,
    SchedulingInput,
    SolverConfig,
    _construct_constraint_name,
    default_num_threads,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def create_lp(
    input_data: SchedulingInput, solver_dir: str | None = None
) -> linopy.Model:
    """Create the MILP problem as linopy model.

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
        solver_dir (str, optional): Path where linopy's temporary files like the lp file
            or the intermediate solution file should be stored.
            The default None results in taking the default temporary directory.

    Returns:
        The constructed linopy model instance.
    """
    time_lp_construction_start = perf_counter()

    logger.debug("Start construction of the LP")

    ids = ProblemIds.init_from_problem(input_data)
    props = ProblemProperties.init_from_problem(input_data, ids=ids)

    logger.debug("IDs and Properties initialized")

    # TODO: Consider chunking
    m = linopy.Model(force_dim_names=True, solver_dir=solver_dir)

    # set initial values by setting lower/upper value of variable
    def _init_lower_upper(coords: list[pd.Index]) -> tuple[xr.DataArray, xr.DataArray]:
        return xr.DataArray(0, coords=coords), xr.DataArray(1, coords=coords)

    person_lower, person_upper = _init_lower_upper([ids.ak, ids.person])
    # required aks have P_{P,A} = 1 implicitly
    person_lower = person_lower.where(~props.required_persons, 1)
    # aks without preferences have P_{P,A} = 0 implicitly
    person_upper = person_upper.where(
        (props.preferences != 0) | props.required_persons, 0
    )

    # TimeImpossibleForPerson
    time_impossible_for_person_mask = (
        props.participant_time_constraints & (~props.fulfilled_time_constraints)
    ).any("time_constraint")
    person_time_lower, person_time_upper = _init_lower_upper([ids.person, ids.timeslot])
    person_time_upper = person_time_upper.where(~time_impossible_for_person_mask, 0)

    # TimeImpossibleForAK
    time_impossible_for_ak_mask = (
        props.ak_time_constraints & (~props.fulfilled_time_constraints)
    ).any("time_constraint")
    time_lower, time_upper = _init_lower_upper([ids.ak, ids.timeslot])
    time_upper = time_upper.where(~time_impossible_for_ak_mask, 0)

    # RoomImpossibleForAK
    room_impossible_for_ak_mask = (
        props.ak_room_constraints & (~props.fulfilled_room_constraints)
    ).any("room_constraint")
    room_lower, room_upper = _init_lower_upper([ids.ak, ids.room])
    room_upper = room_upper.where(~room_impossible_for_ak_mask, 0)

    # Fix Values for already scheduled aks
    for scheduled_ak in input_data.scheduled_aks:
        if (
            scheduled_ak.room_id is not None
            and not input_data.config.allow_changing_rooms
        ):
            room_lower.loc[scheduled_ak.ak_id, scheduled_ak.room_id] = 1

        person_lower.loc[scheduled_ak.ak_id, scheduled_ak.participant_ids] = 1
        time_lower.loc[scheduled_ak.ak_id, scheduled_ak.timeslot_ids] = 1

    # construct variables

    # all variables are binary
    # but we use 'integer' since we manually set the lower/upper bound to fix values
    room = m.add_variables(
        name="Room",
        integer=True,
        coords=[ids.ak, ids.room],
        lower=room_lower,
        upper=room_upper,
    )
    time = m.add_variables(
        name="Time",
        integer=True,
        coords=[ids.ak, ids.timeslot],
        lower=time_lower,
        upper=time_upper,
    )
    block = m.add_variables(name="Block", integer=True, coords=[ids.ak, ids.block])
    person = m.add_variables(
        name="Part",
        integer=True,
        coords=[ids.ak, ids.person],
        lower=person_lower,
        upper=person_upper,
    )
    person_time = m.add_variables(
        name="Working",
        integer=True,
        coords=[ids.person, ids.timeslot],
        lower=person_time_lower,
        upper=person_time_upper,
    )
    logger.debug("Variables added")

    # Set objective function
    # \sum_{P,A} \frac{P_{P,A}}{\sum_{P_{P,A}}\neq 0} T_{P,A}

    # TODO: Do we want to include 'required' AKs?
    num_prefs_per_person = (props.preferences != 0).sum(
        "ak"
    ) + props.required_persons.sum("ak")
    weighted_prefs = (props.preferences / num_prefs_per_person).where(
        num_prefs_per_person != 0
    )
    m.add_objective((weighted_prefs * person).sum(), sense="max")
    logger.debug("Objective added")

    c = time + person
    for ak_id1, ak_id2 in combinations(ids.ak, 2):
        m.add_constraints(
            (c.loc[ak_id1] + c.loc[ak_id2] <= 3),
            name=_construct_constraint_name("MaxOneAKPerPersonAndTime", ak_id1, ak_id2),
        )
    logger.debug("Constraints MaxOneAKPerPersonAndTime added")

    c = time + room
    for ak_id1, ak_id2 in combinations(ids.ak, 2):
        m.add_constraints(
            (c.loc[ak_id1] + c.loc[ak_id2] <= 3),
            name=_construct_constraint_name("MaxOneAKPerRoomAndTime", ak_id1, ak_id2),
        )
    logger.debug("Constraints MaxOneAKPerRoomAndTime added")

    m.add_constraints((time.sum("timeslot") >= props.ak_durations), name="AKDuration")
    logger.debug("Constraints AKDuration added")
    m.add_constraints((block.sum("block") <= 1), name="AKSingleBlock")
    logger.debug("Constraints AKSingleBlock added")

    m.add_constraints(
        (time - props.ak_durations * block).where(props.block_mask) <= 0,
        name="AKBlockAssign",
    )
    logger.debug("Constraints AKBlockAssign added")

    m.add_constraints(
        lhs=person.sum("person") + props.ak_num_interested * room,
        sign="<=",
        rhs=props.ak_num_interested + props.room_capacities,
        mask=props.ak_num_interested > props.room_capacities,
        name="Roomsize",
    )
    logger.debug("Constraints Roomsize added")

    m.add_constraints(room.sum("room") <= 1, name="AtMostOneRoomPerAK")
    logger.debug("Constraints AtMostOneRoomPerAK added")
    m.add_constraints(room.sum("room") >= 1, name="AtLeastOneRoomPerAK")
    logger.debug("Constraints AtLeastOneRoomPerAK added")
    m.add_constraints(
        person.sum("person") <= props.ak_num_interested,
        name="NotMorePeopleThanInterested",
    )
    logger.debug("Constraints NotMorePeopleThanInterested added")
    m.add_constraints(time + person - person_time <= 1, name="TimePersonVar")
    logger.debug("Constraints TimePersonVar added")
    m.add_constraints(room.sum("room") >= 1, name="RoomForAK")
    logger.debug("Constraints RoomForAK added")

    room_impossible_for_person_mask = (
        props.participant_room_constraints & (~props.fulfilled_room_constraints)
    ).any("room_constraint")
    m.add_constraints(
        room + person <= 1,
        name="RoomImpossibleForPerson",
        mask=room_impossible_for_person_mask,
    )
    logger.debug("Constraints RoomImpossibleForPerson added")

    time_impossible_for_room_mask = (
        props.room_time_constraints & (~props.fulfilled_time_constraints)
    ).any("time_constraint")
    m.add_constraints(
        room + time <= 1,
        name="TimeImpossibleForRoom",
        mask=time_impossible_for_room_mask,
    )
    logger.debug("Constraints TimeImpossibleForRoom added")

    for ak_a, ak_b in props.conflict_pairs:
        m.add_constraints(
            time.loc[ak_a] + time.loc[ak_b] <= 1,
            name=_construct_constraint_name("AKConflict", ak_a, ak_b),
        )
    logger.debug("Constraints AKConflict added")

    # TODO vectorize
    for ak_id, (block_id, block_lst) in product(ids.ak, ids.block_dict.items()):
        # AKContiguous
        for timeslot_idx, timeslot_id_a in enumerate(block_lst):
            for timeslot_id_b in block_lst[
                timeslot_idx + props.ak_durations.loc[ak_id].item() :
            ]:
                m.add_constraints(
                    time.loc[ak_id, [timeslot_id_a, timeslot_id_b]].sum("timeslot")
                    <= 1,
                    name=_construct_constraint_name(
                        "AKContiguous",
                        ak_id,
                        block_id,
                        timeslot_id_a,
                        timeslot_id_b,
                    ),
                )
    logger.debug("Constraints AKContiguous added")

    # TODO vectorize
    if input_data.config.max_num_timeslots_before_break > 0:
        # PersonNeedsBreak
        # Any real person needs a break after some number of time slots
        # So in each block at most  consecutive timeslots can be active for any person
        for block_entry in ids.block_dict.values():
            for idx in range(
                len(block_entry) - input_data.config.max_num_timeslots_before_break - 1
            ):
                block_subset = block_entry[
                    idx : idx + input_data.config.max_num_timeslots_before_break + 1
                ]
                m.add_constraints(
                    lhs=person_time.loc[:, block_subset].sum("timeslot"),
                    sign="<=",
                    rhs=input_data.config.max_num_timeslots_before_break,
                    name=_construct_constraint_name("BreakForPerson", block_entry[idx]),
                )
    logger.debug("Constraints BreakForPerson added")

    # TODO vectorize
    # AK dependencies
    for ak_id in ids.ak:
        if ak_id not in props.dependencies:
            continue
        other_ak_ids = props.dependencies[ak_id]
        for idx, timeslot_id in enumerate(ids.timeslot):
            m.add_constraints(
                lhs=time.loc[ak_id, ids.timeslot[idx:]].sum("timeslot")
                - time.loc[other_ak_ids, timeslot_id],
                sign=">=",
                rhs=0,
                name=_construct_constraint_name(
                    "AKDependenciesDoneBeforeAK", ak_id, timeslot_id
                ),
            )
    logger.debug("Constraints AKDependenciesDoneBeforeAK added")

    time_lp_construction_end = perf_counter()
    logger.info(
        "LP constructed. Time elapsed: %.1fs",
        time_lp_construction_end - time_lp_construction_start,
    )
    return m


def export_scheduling_result(
    input_data: SchedulingInput,
    solution: types.ExportTuple,
    allow_unscheduled_aks: bool = False,
) -> dict[types.AkId, ScheduleAtom]:
    """Create a dictionary from the solved MILP.

    For a specification of the output format, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format

    Args:
        input_data(SchedulingInput): The Scheduling instance.
        solution: Named tuple with the solution values of the decision variables.
        allow_unscheduled_aks (bool): Whether not scheduling an AK is allowed or not.
            Defaults to False.

    Returns:
        list: The constructed output list (as specified).
    """
    ids = ProblemIds.init_from_problem(input_data)

    @overload
    def _get_id(
        ak_id: types.AkId,
        var_key: str,
        allow_multiple: Literal[True],
        allow_none: bool,
        coord: str | None = None,
    ) -> np.ndarray: ...

    @overload
    def _get_id(
        ak_id: types.AkId,
        var_key: str,
        allow_multiple: Literal[False],
        allow_none: bool,
        coord: str | None = None,
    ) -> types.Id | None: ...

    def _get_id(
        ak_id: types.AkId,
        var_key: str,
        allow_multiple: bool,
        allow_none: bool,
        coord: str | None = None,
    ) -> Any:
        if coord is None:
            coord = var_key
        ak_row = getattr(solution, var_key).loc[ak_id]
        matched_ids = ak_row.where(ak_row > 0, drop=True).coords[coord]
        if not allow_multiple and matched_ids.size > 1:
            raise ValueError(f"AK {ak_id} is assigned multiple {var_key}")
        elif matched_ids.size == 0 and not allow_none:
            raise ValueError(f"no {var_key} assigned to ak {ak_id}")
        else:
            if allow_multiple:
                return matched_ids.data.tolist()
            else:
                return matched_ids.item() if matched_ids.size > 0 else None

    scheduled_aks: dict[types.AkId, ScheduleAtom] = {
        ak_id: ScheduleAtom(
            ak_id=ak_id,
            room_id=_get_id(
                ak_id=ak_id,
                var_key="room",
                allow_multiple=False,
                allow_none=allow_unscheduled_aks,
            ),
            timeslot_ids=_get_id(
                ak_id=ak_id,
                var_key="time",
                coord="timeslot",
                allow_multiple=True,
                allow_none=allow_unscheduled_aks,
            ),
            participant_ids=_get_id(
                ak_id=ak_id, var_key="person", allow_multiple=True, allow_none=True
            ),
        )
        for ak_id in ids.ak
    }

    return scheduled_aks


def solve_scheduling(
    input_data: SchedulingInput,
    solver_config: SolverConfig,
    solver_name: str | None = None,
) -> tuple[linopy.Model, types.ExportTuple] | None:
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
        solver_config (SolverConfig): The config of the solver to apply.
        solver_name (str, optional): The solver to use. If None, uses a
            default solver choice. Defaults to None.

    Returns:
        If a solution is found, a tuple (`lp_problem`, `solution`)
        where `lp_problem` is the constructed and solved linopy MILP model
        and `solution` contains the named tuple with the solution.
        If the model is infeasible, None is returned instead.

    Raises:
        ValueError: if no solvers are installed.
    """
    if not linopy.available_solvers:
        raise ValueError(
            "No linopy solvers available! "
            "Consider installing any solver of "
            f"{get_args(types.SupportedSolver)}."
        )

    if solver_name is None:
        for solver_candidate in get_args(types.SupportedSolver):
            if solver_candidate in linopy.available_solvers:
                solver_name = cast(str, solver_candidate)
                break
        else:
            solver_name = linopy.available_solvers[0]
            logger.warning(
                "No supported solver available. "
                f"Solver {solver_name} will be used with default config values."
            )

    model = create_lp(input_data, solver_dir=solver_config.solver_dir)

    status, term_cond = model.solve(
        keep_files=solver_config.solver_dir is not None,
        solver_name=solver_name,
        **solver_config.generate_kwargs(solver_name),
    )

    logger.info(f"Termination Condition: {term_cond}")
    logger.info(f"Solution status: {status}")

    if term_cond == "infeasible":
        if model.solver_name == "gurobi":
            model.print_infeasibilities()
        else:
            logger.warning(
                "To calculate the IIS of the infeasible model, use 'gurobi' as a solver"
            )
        return None
    solution = types.ExportTuple(
        room=model.variables["Room"].solution.round(),
        time=model.variables["Time"].solution.round(),
        person=model.variables["Part"].solution.round(),
    )
    return (model, solution)


def process_solved_lp(
    model: linopy.Model,
    solution: types.ExportTuple,
    input_data: SchedulingInput,
) -> dict[types.AkId, ScheduleAtom] | None:
    """Process the solved LP model and create a schedule output.

    Args:
        model (linopy.Model): The linopy LP model object after the optimizer ran.
        solution (nested dict containing the MILP variables): The solution to the problem.
        input_data (SchedulingInput): The input data used to construct the ILP.

    Returns:
        A dict mapping each AK ID to its scheduleing or None if scheduling failed.
    """
    if model.status != "ok":
        return None

    # TODO: Test if a check for partial solutions is necessary

    return export_scheduling_result(
        input_data,
        solution,
        allow_unscheduled_aks=input_data.config.allow_unscheduled_aks,
    )


def calc_changed_fixed_schedule_atoms(
    input_atoms: Iterable[ScheduleAtom],
    schedule_atoms: Iterable[ScheduleAtom],
    ignore_room_change: bool = False,
    ignore_timeslots_change: bool = False,
    ignore_participants_change: bool = True,
) -> list[ScheduleAtom]:
    """
    Check if all scheduling atoms of the input are still contained in the output schedule.

    Args:
        input_atoms (iterable of ScheduleAtoms): The fixed schedule atoms of the input.
        schedule_atoms (iterable of ScheduleAtoms): An iterable of the scheduled atoms.
        ignore_room_change (bool): If True, room changes are ignored in the check.
        ignore_timeslots_change (bool): If True, timeslots changes are ignored in the check.
        ignore_participants_change (bool): If True, participants changes
            are ignored in the check.

    Returns:
        The list of all schedule atoms of the input that are not contained in the output.
    """

    def _stripped_atom_set(atom_it: Iterable[ScheduleAtom]) -> set[ScheduleAtom]:
        return {
            atom.stripped_copy(
                strip_room=ignore_room_change,
                strip_participants=ignore_participants_change,
                strip_timeslots=ignore_timeslots_change,
            )
            for atom in atom_it
        }

    input_data_atom_set = _stripped_atom_set(input_atoms)
    schedule_atom_set = _stripped_atom_set(schedule_atoms)
    changed_schedule_set = input_data_atom_set - schedule_atom_set

    return sorted(changed_schedule_set)


def main() -> None:
    """Run solve_scheduling from CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver",
        type=str,
        default=None,
        help=(
            "The solver to use. We currently only support passing CLI args to solvers in "
            f"{get_args(types.SupportedSolver)}. If None, chooses a default "
            "from installed solvers. Defaults to None."
        ),
    )
    parser.add_argument(
        "--solver-dir",
        type=str,
        default=None,
        help=(
            "Path where linopy's temporary files like the lp file "
            "or the intermediate solution file should be stored. "
            "The default None results in taking the default temporary directory "
            " and an automatic removal after the solving is done."
        ),
    )
    parser.add_argument(
        "--solver-io-api",
        type=str,
        choices=["direct", "lp", "mps"],
        default="direct",
        help=(
            "API to use for communicating with the solver, must be one of "
            "{'lp', 'mps', 'direct'}. If set to 'lp'/'mps' the problem is written to an "
            "LP/MPS file which is then read by the solver. If set to "
            "'direct' the problem is communicated to the solver via the solver "
            "specific API, e.g. gurobipy. This may lead to faster run times. "
            "Defaults to 'direct'."
        ),
    )
    parser.add_argument(
        "--solver-warmstart-fn",
        type=str,
        default=None,
        help="Optional path of the basis file which should be used to warmstart the solving.",
    )
    parser.add_argument(
        "--timelimit",
        type=float,
        default=None,
        help="Timelimit as stopping criterion (in seconds)",
    )
    parser.add_argument(
        "--gap-rel", type=float, default=None, help="Relative gap as stopping criterion"
    )
    parser.add_argument(
        "--gap-abs", type=float, default=None, help="Absolute gap as stopping criterion"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of threads to use. Defaults to #CPUs minus 1.",
    )
    parser.add_argument(
        "--loglevel",
        type=str.lower,
        choices=["error", "warning", "info", "debug"],
        default="info",
        help="Select logging level. Defaults to 'info'.",
    )
    parser.add_argument(
        "path", type=str, help="Path of the JSON input file to the solver."
    )
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
    args = parser.parse_args()

    # set logging level
    numeric_loglevel = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_loglevel, int):
        raise ValueError(f"Invalid log level: {args.loglevel}")
    logging.basicConfig(
        level=numeric_loglevel,
        format="[%(levelname)s] %(asctime)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.threads is None:
        # default threads to number of available CPUs minus 1
        args.threads = default_num_threads()

    # disable duplicate logging from gurobi logger
    # https://linopy.readthedocs.io/en/latest/gurobi-double-logging.html
    gurobi_logger = logging.getLogger("gurobipy")
    gurobi_logger.propagate = False

    solver_config = SolverConfig(
        solver_dir=args.solver_dir,
        solver_io_api=args.solver_io_api,
        warmstart_fn=args.solver_warmstart_fn,
        time_limit=args.timelimit,
        gap_rel=args.gap_rel,
        gap_abs=args.gap_abs,
        threads=args.threads,
    )

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

    solution_tuple = solve_scheduling(
        scheduling_input,
        solver_config,
        args.solver,
    )

    if solution_tuple is None:
        # if no solution was found, exit
        return

    schedule = process_solved_lp(*solution_tuple, input_data=scheduling_input)

    if schedule is None:
        # if no schedule was calculated, exit
        return

    changed_fixed_schedule_atoms = calc_changed_fixed_schedule_atoms(
        scheduling_input.scheduled_aks,
        schedule.values(),
        ignore_room_change=scheduling_input.config.allow_changing_rooms,
    )

    # check if all fixed schedule atoms of the input are carried over to the output
    # if not: print warning with affected AKs
    if changed_fixed_schedule_atoms:
        string_repr = [
            f"\t(AK {atom.ak_id}, Room {atom.room_id}, Timeslots {sorted(atom.timeslot_ids)})"
            for atom in changed_fixed_schedule_atoms
        ]

        logger.warning(
            "Some fixed scheduling was NOT respected in the output! "
            "The following entries of the input are affected:\n%s",
            "\n".join(string_repr),
        )

    out_dict = {
        "scheduled_aks": list(map(asdict, schedule.values())),
        "input": scheduling_input.to_dict(),
    }
    with args.output.open("w") as ff:
        json.dump(out_dict, ff)
    logger.info("Stored result at %s", args.output)


if __name__ == "__main__":
    main()
