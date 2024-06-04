"""Solving the MILPs for conference scheduling."""

import argparse
import json
from collections import defaultdict
from itertools import chain, combinations, product
from pathlib import Path
from typing import Any, Iterable, cast

from pulp import (
    LpBinary,
    LpMaximize,
    LpProblem,
    LpSolution,
    LpSolutionInfeasible,
    LpSolutionNoSolutionFound,
    LpStatus,
    LpStatusInfeasible,
    LpVariable,
    getSolver,
    lpSum,
    value,
)

from .util import AKData, ParticipantData, RoomData, SchedulingInput, TimeSlotData


def process_pref_score(preference_score: int, required: bool, mu: float) -> float:
    """Process the input preference score for the MILP constraints.

    Args:
        preference_score (int): The input score of preference: not interested (0),
            weakly interested (1), strongly interested (2) or required (-1).
        required (bool): Whether the participant is required for the AK or not.
        mu (float): The weight associated with a strong preference for an AK.

    Returns:
        float: The processed preference score: Required AKs are weighted with 0,
        weakly preferred AKs with 1 and strongly preferred AKs with `mu`.

    Raises:
        ValueError: if `preference_score` is not in [0, 1, 2, -1].
    """
    if required or preference_score == -1:
        return 0
    elif preference_score in [0, 1]:
        return preference_score
    elif preference_score == 2:
        return mu
    else:
        raise ValueError(preference_score)


def process_room_cap(room_capacity: int, num_participants: int) -> int:
    """ Process the input room capacity for the MILP constraints.

    Args:
        room_capacity (int): The input room capacity: infinite (-1) or actual capacity >=0
        num_participants (int): The total number of participants (needed to model infinity)

    Retruns:
        int: The processed room capacity: Rooms with infinite capacity or capacity larger than
        num_participants are set to num_participants. Rooms with a smaller non-negative capacity
        hold their capacity.

    Raises:
        ValueError: if 'room_capacity' < -1
    """
    if room_capacity == -1:
        return num_participants
    if room_capacity >= num_participants:
        return num_participants
    if room_capacity < 0:
        raise ValueError(
            f"Invalid room capacity {room_capacity}. "
            "Room capacity must be non-negative or -1."
        )
    return room_capacity


def _construct_constraint_name(name: str, *args: str) -> str:
    return name + "_" + "_".join(args)


def get_ids(
    input_data: SchedulingInput,
) -> tuple[set[str], set[str], set[str], set[str]]:
    """Create id sets from scheduling input."""

    def _retrieve_ids(
        input_iterable: Iterable[AKData | ParticipantData | RoomData | TimeSlotData],
    ) -> set[str]:
        return {obj.id for obj in input_iterable}

    ak_ids = _retrieve_ids(input_data.aks)
    participant_ids = _retrieve_ids(input_data.participants)
    room_ids = _retrieve_ids(input_data.rooms)
    timeslot_ids = _retrieve_ids(chain.from_iterable(input_data.timeslot_blocks))
    return ak_ids, participant_ids, room_ids, timeslot_ids


def create_lp(
    input_data: SchedulingInput,
    mu: float,
    output_file: str | None = "koma-plan.lp",
) -> tuple[
    LpProblem,
    tuple[
        dict[str, dict[str, LpVariable]],
        dict[str, dict[str, LpVariable]],
        dict[str, dict[str, LpVariable]],
    ],
]:
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
        mu (float): The weight associated with a strong preference for an AK.
        output_file (str, optional): If not None, the created LP is written
            as an `.lp` file to this location. Defaults to `koma-plan.lp`.

    Returns:
        A tuple (`lp_problem`, `dec_vars`) where `lp_problem` is the
        constructed MILP instance and `dec_vars` is the nested dictionary
        containing the MILP variables.
    """
    # Get ids from input_dict
    ak_ids, person_ids, room_ids, timeslot_ids = get_ids(input_data)

    timeslot_idx_dict = {
        timeslot.id: timeslot_idx
        for block in input_data.timeslot_blocks
        for timeslot_idx, timeslot in enumerate(block)
    }
    block_idx_dict = {
        block_idx: [timeslot.id for timeslot in block]
        for block_idx, block in enumerate(input_data.timeslot_blocks)
    }
    # Get values needed from the input_dict
    room_capacities = {
        room.id: process_room_cap(room.capacity, len(person_ids))
        for room in input_data.rooms
    }
    ak_durations = {ak.id: ak.duration for ak in input_data.aks}

    # dict of real participants only (without dummy participants)
    # with numerical preferences
    weighted_preference_dict = {
        person.id: {
            pref.ak_id: process_pref_score(
                pref.preference_score,
                pref.required,
                mu=mu,
            )
            for pref in person.preferences
        }
        for person in input_data.participants
    }
    required_persons = {
        ak_id: {
            person.id
            for person in input_data.participants
            for pref in person.preferences
            if pref.ak_id == ak_id and pref.required
        }
        for ak_id in ak_ids
    }
    ak_num_interested = {
        ak_id: len(required_persons[ak_id])
        + sum(
            1
            for person, prefs in weighted_preference_dict.items()
            if ak_id in prefs.keys() and prefs[ak_id] != 0
        )
        for ak_id in ak_ids
    }

    # Get constraints from input_dict
    participant_time_constraint_dict = {
        participant.id: set(participant.time_constraints)
        for participant in input_data.participants
    }
    participant_room_constraint_dict = {
        participant.id: set(participant.room_constraints)
        for participant in input_data.participants
    }

    ak_time_constraint_dict = {ak.id: set(ak.time_constraints) for ak in input_data.aks}
    ak_room_constraint_dict = {ak.id: set(ak.room_constraints) for ak in input_data.aks}

    room_time_constraint_dict = {
        room.id: set(room.time_constraints) for room in input_data.rooms
    }

    fulfilled_time_constraints = {
        timeslot.id: set(timeslot.fulfilled_time_constraints)
        for block in input_data.timeslot_blocks
        for timeslot in block
    }
    fulfilled_room_constraints = {
        room.id: set(room.fulfilled_room_constraints) for room in input_data.rooms
    }

    # Create problem
    prob = LpProblem("MLPKoMa", sense=LpMaximize)

    # Create decision variables
    room_var: dict[str, dict[str, LpVariable]] = LpVariable.dicts(
        "Room", (ak_ids, room_ids), cat=LpBinary
    )
    time_var: dict[str, dict[str, LpVariable]] = LpVariable.dicts(
        "Time",
        (ak_ids, timeslot_ids),
        cat=LpBinary,
    )
    block_var: dict[str, dict[int, LpVariable]] = LpVariable.dicts(
        "Block", (ak_ids, block_idx_dict.keys()), cat=LpBinary
    )
    person_var: dict[str, dict[str, LpVariable]] = LpVariable.dicts(
        "Part", (ak_ids, person_ids), cat=LpBinary
    )

    # Set objective function
    # \sum_{P,A} \frac{P_{P,A}}{\sum_{P_{P,A}}\neq 0} T_{P,A}
    prob += (
        lpSum(
            [
                pref * person_var[ak_id][person_id] / len(preferences)
                for person_id, preferences in weighted_preference_dict.items()
                for ak_id, pref in preferences.items()
            ]
        ),
        "cost_function",
    )

    # Add constraints
    # for all x, a, a', t time[a][t]+F[a][x]+time[a'][t]+F[a'][x] <= 3
    # a,a' AKs, t timeslot, x Person or Room
    for (ak_id1, ak_id2), timeslot_id in product(combinations(ak_ids, 2), timeslot_ids):
        for person_id in person_ids:
            constraint = lpSum(
                [
                    time_var[ak_id1][timeslot_id],
                    time_var[ak_id2][timeslot_id],
                    person_var[ak_id1][person_id],
                    person_var[ak_id2][person_id],
                ]
            )
            prob += constraint <= 3, _construct_constraint_name(
                "MaxOneAKPerPersonAndTime",
                ak_id1,
                ak_id2,
                timeslot_id,
                person_id,
            )
        # MaxOneAKPerRoomAndTime
        for room_id in room_ids:
            constraint = lpSum(
                [
                    time_var[ak_id1][timeslot_id],
                    time_var[ak_id2][timeslot_id],
                    room_var[ak_id1][room_id],
                    room_var[ak_id2][room_id],
                ]
            )
            prob += constraint <= 3, _construct_constraint_name(
                "MaxOneAKPerRoomAndTime",
                ak_id1,
                ak_id2,
                timeslot_id,
                room_id,
            )

    for ak_id in ak_ids:
        # AKDurations
        prob += lpSum(
            [time_var[ak_id][timeslot_id] for timeslot_id in timeslot_ids]
        ) >= ak_durations[ak_id], _construct_constraint_name("AKDuration", ak_id)
        # AKSingleBlock
        prob += lpSum(
            [block_var[ak_id][block_id] for block_id in block_idx_dict]
        ) <= 1, _construct_constraint_name("AKSingleBlock", ak_id)
        for block_id, block in block_idx_dict.items():
            prob += lpSum(
                [time_var[ak_id][timeslot_id] for timeslot_id in block]
            ) <= ak_durations[ak_id] * block_var[ak_id][
                block_id
            ], _construct_constraint_name(
                "AKSingleBlock", ak_id, str(block_id)
            )
            # AKConsecutive
            for timeslot_id_a, timeslot_id_b in combinations(block, 2):
                if (
                    abs(
                        timeslot_idx_dict[timeslot_id_a]
                        - timeslot_idx_dict[timeslot_id_b]
                    )
                    >= ak_durations[ak_id]
                ):
                    prob += lpSum(
                        [time_var[ak_id][timeslot_id_a], time_var[ak_id][timeslot_id_b]]
                    ) <= 1, _construct_constraint_name(
                        "AKConsecutive",
                        ak_id,
                        str(block_id),
                        timeslot_id_a,
                        timeslot_id_b,
                    )

    # Roomsizes
    for room_id, ak_id in product(room_ids, ak_ids):
        if ak_num_interested[ak_id] > room_capacities[room_id]:
            prob += lpSum(
                [person_var[ak_id][person_id] for person_id in person_ids]
            ) + ak_num_interested[ak_id] * room_var[ak_id][room_id] <= ak_num_interested[
                ak_id
            ] + room_capacities[
                room_id
            ], _construct_constraint_name(
                "Roomsize", room_id, ak_id
            )
    for ak_id in ak_ids:
        prob += lpSum(
            [room_var[ak_id][room_id] for room_id in room_ids]
        ) <= 1, _construct_constraint_name("AtMostOneRoomPerAK", ak_id, room_id)
        prob += lpSum(
            [room_var[ak_id][room_id] for room_id in room_ids]
        ) >= 1, _construct_constraint_name("AtLeastOneRoomPerAK", ak_id, room_id)
        # We need this constraint so the Roomsize is correct
        prob += lpSum(
            [person_var[ak_id][person_id] for person_id in person_ids]
        ) <= ak_num_interested[ak_id], _construct_constraint_name(
            "NotMorePeopleThanInterested", ak_id
        )

    # PersonNotInterestedInAK
    # For all A, Z, R, P: If P_{P, A} = 0: Y_{A,Z,R,P} = 0 (non-dummy P)
    for person_id, preferences in weighted_preference_dict.items():
        # aks not in pref_aks have P_{P,A} = 0 implicitly
        for ak_id in ak_ids.difference(preferences.keys()):
            person_var[ak_id][person_id].setInitialValue(0)
            person_var[ak_id][person_id].fixValue()
    for ak_id, persons in required_persons.items():
        for person_id in persons:
            person_var[ak_id][person_id].setInitialValue(1)
            person_var[ak_id][person_id].fixValue()

    for person_id in weighted_preference_dict:
        # TimeImpossibleForPerson
        # Real person P cannot attend AKs with timeslot Z
        for timeslot_id in timeslot_ids:
            if participant_time_constraint_dict[person_id].difference(
                fulfilled_time_constraints[timeslot_id]
            ):
                for ak_id in ak_ids:
                    prob += lpSum(
                        [time_var[ak_id][timeslot_id], person_var[ak_id][person_id]]
                    ) <= 1, _construct_constraint_name(
                        "TimeImpossibleForPerson",
                        person_id,
                        timeslot_id,
                        ak_id,
                    )

        # RoomImpossibleForPerson
        # Real person P cannot attend AKs with room R
        for room_id in room_ids:
            if participant_room_constraint_dict[person_id].difference(
                fulfilled_room_constraints[room_id]
            ):
                for ak_id in ak_ids:
                    prob += lpSum(
                        [room_var[ak_id][room_id], person_var[ak_id][person_id]]
                    ) <= 1, _construct_constraint_name(
                        "RoomImpossibleFor Person", person_id, room_id, ak_id
                    )

    for ak_id in ak_ids:
        # TimeImpossibleForAK
        for timeslot_id in timeslot_ids:
            if ak_time_constraint_dict[ak_id].difference(
                fulfilled_time_constraints[timeslot_id]
            ):
                time_var[ak_id][timeslot_id].setInitialValue(0)
                time_var[ak_id][timeslot_id].fixValue()
        # RoomImpossibleForAK
        for room_id in room_ids:
            if ak_room_constraint_dict[ak_id].difference(
                fulfilled_room_constraints[room_id]
            ):
                room_var[ak_id][room_id].setInitialValue(0)
                room_var[ak_id][room_id].fixValue()
        prob += lpSum(
            [room_var[ak_id][room_id] for room_id in room_ids]
        ) >= 1, _construct_constraint_name("RoomForAK", ak_id)

        # TimeImpossibleForRoom
        for room_id, timeslot_id in product(room_ids, timeslot_ids):
            if room_time_constraint_dict[room_id].difference(
                fulfilled_time_constraints[timeslot_id]
            ):
                prob += lpSum(
                    [room_var[ak_id][room_id], time_var[ak_id][timeslot_id]]
                ) <= 1, _construct_constraint_name(
                    "TimeImpossibleForRoom", room_id, timeslot_id, ak_id
                )

    # The problem data is written to an .lp file
    if output_file is not None:
        prob.writeLP(output_file)

    return prob, (room_var, time_var, person_var)


def export_scheduling_result(
    solved_lp_problem: LpProblem,
    allow_unscheduled_aks: bool = False,
) -> dict[str, Any]:
    """Create a dictionary from the solved MILP.

    For a specification of the output format, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format

    Args:
        solved_lp_problem (LpProblem): The solved MILP instance.
        allow_unscheduled_aks (bool): Whether not scheduling an AK is allowed or not.
            Defaults to False.

    Returns:
        dict: The constructed output dict (as specified).

    Raises:
        ValueError: might be raised if the solution of the MILP is infeasible or
            if an AK is not scheduled and allow_unscheduled_aks is False.
    """
    var_value_dict: dict[str, dict[str, dict[str, int]]] = {
        "Room": defaultdict(dict),
        "Time": defaultdict(dict),
        "Block": defaultdict(dict),
        "Part": defaultdict(dict),
    }

    def _get_val(var: LpVariable) -> int:
        ret_val = (
            # We know there is distinction works as intended for gurobi, PuLP and HiGHS.
            # There might be other solvers (supported by pulp) that need special handling.
            var.solverVar.X
            if solved_lp_problem.solver and solved_lp_problem.solver.name == "GUROBI"
            else value(var)
        )
        return cast(int, round(ret_val))

    for var in solved_lp_problem.variables():
        var_cat, idx0, idx1 = var.name.split("_")
        var_value_dict[var_cat][idx0][idx1] = _get_val(var)

    scheduled_ak_dict: dict[str, dict[str, list[str] | str]] = defaultdict(dict)

    # Handle rooms differntly because we want special handling if
    # AKs are scheduled into more than one room.
    # Also the field `ak_id` is set initially.
    for ak_id, set_room_ids in var_value_dict["Room"].items():
        sum_matched_rooms = sum(set_room_ids.values())
        if sum_matched_rooms == 1:
            room_for_ak = max(set_room_ids.keys(), key=set_room_ids.__getitem__)
            scheduled_ak_dict[ak_id]["ak_id"] = ak_id
            scheduled_ak_dict[ak_id]["room_id"] = room_for_ak
        elif sum_matched_rooms == 0:
            if allow_unscheduled_aks:
                continue
            else:
                raise ValueError(f"no room assigned to ak {ak_id}")
        else:
            raise ValueError(f"AK {ak_id} is assigned multiple rooms")

    def _assign_matched_ids(var_key: str, scheduled_ak_key: str, name: str) -> None:
        for ak_id, set_ids in var_value_dict[var_key].items():
            matched_ids = [idx for idx, val in set_ids.items() if val > 0]
            if matched_ids:
                scheduled_ak_dict[ak_id][scheduled_ak_key] = matched_ids
            elif not allow_unscheduled_aks:
                raise ValueError(f"AK {ak_id} has no assigned {name}")

    _assign_matched_ids(
        var_key="Time", scheduled_ak_key="timeslot_ids", name="timeslots"
    )
    _assign_matched_ids(
        var_key="Part", scheduled_ak_key="participant_ids", name="participants"
    )

    return {"scheduled_aks": list(scheduled_ak_dict.values())}


def solve_scheduling(
    input_data: SchedulingInput,
    mu: float,
    solver_name: str | None = None,
    output_lp_file: str | None = "koma-plan.lp",
    **solver_kwargs: dict[str, Any],
) -> LpProblem:
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
        mu (float): The weight associated with a strong preference for an AK.
        solver_name (str, optional): The solver to use. If None, uses pulp's
            default solver. Defaults to None.
        output_lp_file (str, optional): If not None, the created LP is written
            as an `.lp` file to this location. Defaults to `koma-plan.lp`.
        **solver_kwargs: kwargs are passed to the solver.

    Returns:
        The LpProblem instance after the solver ran. This instance encodes the
        LP formulation as well as the variable assignment.
    """
    lp_problem, _dec_vars = create_lp(input_data, mu, output_lp_file)

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

    return lp_problem


def process_solved_lp(
    solved_lp_problem: LpProblem,
    input_data: SchedulingInput | None = None,
    allow_unscheduled_aks: bool = False,
) -> dict[str, Any] | None:
    """Process the solved LP model and create a schedule output.

    Args:
        solved_lp_problem (LpProblem): The pulp LP problem object after the optimizer ran.
        input_data (SchedulingInput, optional): The input data used to construct the ILP.
            If set, the input data is added to the output schedule dict. Defaults to None.
        allow_unscheduled_aks (bool): Whether not scheduling an AK is allowed or not.
            Defaults to False.

    Returns:
        A dictionary containing the scheduled aks as well as the scheduling
        input.
    """
    if solved_lp_problem.sol_status in [
        LpSolutionInfeasible,
        LpSolutionNoSolutionFound,
    ]:
        return None
    output_dict = export_scheduling_result(
        solved_lp_problem, allow_unscheduled_aks=allow_unscheduled_aks
    )
    if input_data is not None:
        output_dict["input"] = input_data.to_dict()
    return output_dict


def main() -> None:
    """Run solve_scheduling from CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mu",
        type=float,
        default=2,
        help="The weight associated with a strong preference for an AK.",
    )
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
        "--disallow-unscheduled-aks",
        action="store_true",
        default=False,
        help="If set, we do not allow aks to not be scheduled. "
        "Otherwise, the solver is allowed to not schedule AKs.",
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

    solved_lp_problem = solve_scheduling(
        scheduling_input,
        args.mu,
        args.solver,
        **solver_kwargs,
    )

    output_dict = process_solved_lp(
        solved_lp_problem,
        input_data=scheduling_input,
        allow_unscheduled_aks=not args.disallow_unscheduled_aks,
    )

    if output_dict is not None:
        with args.output.open("w") as ff:
            json.dump(output_dict, ff)
        print(f"Stored result at {args.output}")


if __name__ == "__main__":
    main()
