"""Solving the MILPs for conference scheduling."""

import argparse
import json
from dataclasses import asdict
from itertools import chain, combinations, product
from pathlib import Path
from typing import Any, Iterable, Literal, overload

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
)

from .util import (
    AKData,
    ParticipantData,
    RoomData,
    ScheduleAtom,
    SchedulingInput,
    TimeSlotData,
)


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
    """Process the input room capacity for the MILP constraints.

    Args:
        room_capacity (int): The input room capacity: infinite (-1) or actual capacity >=0
        num_participants (int): The total number of participants (needed to model infinity)

    Returns:
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


def get_ak_name(input_data: SchedulingInput, ak_id: str) -> str:
    """Get name string for an AK."""
    ak_names = [
        ak.info["name"]
        for ak in input_data.aks
        if ak.id == ak_id and "name" in ak.info.keys()
    ]
    return ", ".join(ak_names)


def create_lp(
    input_data: SchedulingInput,
    output_file: str | None = "koma-plan.lp",
) -> tuple[LpProblem, dict[str, dict[str, dict[str, LpVariable]]]]:
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
                mu=input_data.config.mu,
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
    for ak_id, persons in required_persons.items():
        if len(persons) == 0:
            print(
                f"Warning: AK {get_ak_name(input_data, ak_id)} with id {ak_id} "
                "has no required persons. Who owns this?"
            )
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
    person_time_var: dict[str, dict[str, LpVariable]] = LpVariable.dicts(
        "Working",
        (person_ids, timeslot_ids),
        cat=LpBinary,
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
        constraint = lpSum(time_var[ak_id].values()) >= ak_durations[ak_id]
        prob += constraint, _construct_constraint_name("AKDuration", ak_id)

        # AKSingleBlock
        constraint = lpSum(block_var[ak_id].values()) <= 1
        prob += constraint, _construct_constraint_name("AKSingleBlock", ak_id)
        for block_id, block in block_idx_dict.items():
            constraint_sum = lpSum(
                [time_var[ak_id][timeslot_id] for timeslot_id in block]
            )
            prob += (
                constraint_sum <= ak_durations[ak_id] * block_var[ak_id][block_id],
                _construct_constraint_name("AKSingleBlock", ak_id, str(block_id)),
            )
            # AKContiguous
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
                        "AKContiguous",
                        ak_id,
                        str(block_id),
                        timeslot_id_a,
                        timeslot_id_b,
                    )

    # Roomsizes
    for room_id, ak_id in product(room_ids, ak_ids):
        if ak_num_interested[ak_id] > room_capacities[room_id]:
            constraint_sum = lpSum(person_var[ak_id].values())
            constraint_sum += ak_num_interested[ak_id] * room_var[ak_id][room_id]
            constraint = (
                constraint_sum <= ak_num_interested[ak_id] + room_capacities[room_id]
            )
            prob += constraint, _construct_constraint_name("Roomsize", room_id, ak_id)
    for ak_id in ak_ids:
        prob += lpSum(room_var[ak_id].values()) <= 1, _construct_constraint_name(
            "AtMostOneRoomPerAK", ak_id, room_id
        )
        prob += lpSum(room_var[ak_id].values()) >= 1, _construct_constraint_name(
            "AtLeastOneRoomPerAK", ak_id, room_id
        )
        # We need this constraint so the Roomsize is correct
        constraint_sum = lpSum(person_var[ak_id].values())
        prob += constraint_sum <= ak_num_interested[ak_id], _construct_constraint_name(
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
            for ak_id in ak_ids:
                constraint_sum = lpSum(
                    [time_var[ak_id][timeslot_id], person_var[ak_id][person_id]]
                )
                prob += (
                    constraint_sum <= person_time_var[person_id][timeslot_id] + 1,
                    _construct_constraint_name(
                        "TimePersonVar",
                        person_id,
                        timeslot_id,
                        ak_id,
                    ),
                )
            if participant_time_constraint_dict[person_id].difference(
                fulfilled_time_constraints[timeslot_id]
            ):
                person_time_var[person_id][timeslot_id].setInitialValue(0)
                person_time_var[person_id][timeslot_id].fixValue()

        # RoomImpossibleForPerson
        # Real person P cannot attend AKs with room R
        for room_id in room_ids:
            if participant_room_constraint_dict[person_id].difference(
                fulfilled_room_constraints[room_id]
            ):
                for ak_id in ak_ids:
                    constraint_sum = lpSum(
                        [room_var[ak_id][room_id], person_var[ak_id][person_id]]
                    )
                    prob += constraint_sum <= 1, _construct_constraint_name(
                        "RoomImpossibleForPerson", person_id, room_id, ak_id
                    )

        # PersonNeedsBreak
        # Any real person needs a break after some number of time slots
        # So in each block at most  consecutive timeslots can be active for any person
        if input_data.config.max_num_timeslots_before_break > 0:
            for _block_id, block in block_idx_dict.items():
                for i in range(
                    len(block) - input_data.config.max_num_timeslots_before_break - 1
                ):
                    sum_of_vars = lpSum(
                        [
                            person_time_var[person_id][timeslot_id]
                            for timeslot_id in block[
                                i : i
                                + input_data.config.max_num_timeslots_before_break
                                + 1
                            ]
                        ]
                    )

                    prob += (
                        sum_of_vars <= input_data.config.max_num_timeslots_before_break,
                        _construct_constraint_name(
                            "BreakForPerson", person_id, block[i]
                        ),
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

    # AK conflicts
    conflict_pairs: set[tuple[str, str]] = set()
    for ak in input_data.aks:
        other_ak_ids: list[str] = ak.properties.get("conflicts", [])
        conflict_pairs.update(
            [
                (ak.id, other_ak_id) if ak.id < other_ak_id else (other_ak_id, ak.id)
                for other_ak_id in other_ak_ids
            ]
        )

    for timeslot_id, (ak_a, ak_b) in product(timeslot_ids, conflict_pairs):
        prob += (
            lpSum([time_var[ak_a][timeslot_id], time_var[ak_b][timeslot_id]]) <= 1,
            _construct_constraint_name("AKConflict", ak_a, ak_b, timeslot_id),
        )

    # AK dependencies
    sorted_timeslot_ids = sorted(timeslot_ids)
    for ak in input_data.aks:
        other_ak_ids = ak.properties.get("dependencies", [])
        if not other_ak_ids:
            continue
        for idx, timeslot_id in enumerate(sorted_timeslot_ids):
            constraint_sum = lpSum(
                [
                    time_var[ak_dependency][succ_timeslot_id]
                    for ak_dependency, succ_timeslot_id in product(
                        other_ak_ids, sorted_timeslot_ids[idx:]
                    )
                ]
            )
            constraint = constraint_sum <= time_var[ak.id][timeslot_id]

            prob += constraint, _construct_constraint_name(
                "AKDependenciesDoneBeforeAK", ak.id, timeslot_id
            )

    # Fix Values for already scheduled aks
    for scheduled_ak in input_data.scheduled_aks:
        if scheduled_ak.room_id is not None:
            room_var[scheduled_ak.ak_id][scheduled_ak.room_id].setInitialValue(1)
            if not input_data.config.allow_changing_rooms:
                room_var[scheduled_ak.ak_id][scheduled_ak.room_id].fixValue()
        for person_id in scheduled_ak.participant_ids:
            person_var[scheduled_ak.ak_id][person_id].setInitialValue(1)
            person_var[scheduled_ak.ak_id][person_id].fixValue()
        for timeslot_id in scheduled_ak.timeslot_ids:
            time_var[scheduled_ak.ak_id][timeslot_id].setInitialValue(1)
            time_var[scheduled_ak.ak_id][timeslot_id].fixValue()

    # The problem data is written to an .lp file
    if output_file is not None:
        prob.writeLP(output_file)

    return prob, {"Room": room_var, "Time": time_var, "Part": person_var}


def export_scheduling_result(
    input_data: SchedulingInput,
    solution: dict[str, dict[str, dict[str, int]]],
    allow_unscheduled_aks: bool = False,
) -> dict[str, ScheduleAtom]:
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
    ak_ids, person_ids, room_ids, timeslot_ids = get_ids(input_data)

    @overload
    def _get_id(
        ak_id: str, var_key: str, allow_multiple: Literal[True], allow_none: bool
    ) -> list[str]: ...

    @overload
    def _get_id(
        ak_id: str, var_key: str, allow_multiple: Literal[False], allow_none: bool
    ) -> str | None: ...

    def _get_id(
        ak_id: str, var_key: str, allow_multiple: bool, allow_none: bool
    ) -> str | list[str] | None:
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

    scheduled_ak_dict: dict[str, ScheduleAtom] = {
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
        for ak_id in ak_ids
    }

    return scheduled_ak_dict


def solve_scheduling(
    input_data: SchedulingInput,
    solver_name: str | None = None,
    output_lp_file: str | None = "koma-plan.lp",
    **solver_kwargs: dict[str, Any],
) -> tuple[LpProblem, dict[str, dict[str, dict[str, int]]]]:
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
        **solver_kwargs: kwargs are passed to the solver.

    Returns:
        A tuple (`lp_problem`, `solution`) where `lp_problem` is the
        constructed and solved MILP instance and `solution` contains
        the nested dicts with the solution.
    """
    lp_problem, dec_vars = create_lp(input_data, output_lp_file)

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

    solution = {
        var_key: {
            ak_id: {id: round(var.value()) for id, var in vars.items()}
            for ak_id, vars in vars_dict.items()
        }
        for var_key, vars_dict in dec_vars.items()
    }

    return (lp_problem, solution)


def process_solved_lp(
    solved_lp_problem: LpProblem,
    solution: dict[str, dict[str, dict[str, LpVariable]]],
    input_data: SchedulingInput,
) -> dict[str, ScheduleAtom] | None:
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
    return export_scheduling_result(
        input_data,
        solution,
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
