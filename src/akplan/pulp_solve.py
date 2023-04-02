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
    LpStatus,
    LpVariable,
    getSolver,
    lpSum,
    value,
)

from .util import AKData, ParticipantData, RoomData, SchedulingInput, TimeSlotData


def process_pref_score(preference_score: int, required: bool, mu: float) -> float:
    if required or preference_score == -1:
        return 0
    elif preference_score in [0, 1]:
        return preference_score
    elif preference_score == 2:
        return mu
    else:
        raise NotImplementedError(preference_score)


def _construct_constraint_name(name: str, *args: str) -> str:
    return name + "_" + "_".join(args)


def get_ids(
    input_data: SchedulingInput,
) -> tuple[set[str], set[str], set[str], set[str]]:
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
    num_people = len(person_ids)

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
    room_capacities = {room.id: room.capacity for room in input_data.rooms}
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
                    prob += time_var[ak_id][timeslot_id_a] + time_var[ak_id][
                        timeslot_id_b
                    ] <= 1, _construct_constraint_name(
                        "AKConsecutive",
                        ak_id,
                        str(block_id),
                        timeslot_id_a,
                        timeslot_id_b,
                    )

    # Roomsizes
    for room_id, ak_id in product(room_ids, ak_ids):
        prob += lpSum(
            [person_var[ak_id][person_id] for person_id in person_ids]
        ) + num_people * room_var[ak_id][room_id] <= num_people + room_capacities[
            room_id
        ], _construct_constraint_name(
            "Roomsizes", room_id, ak_id
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
                    prob += time_var[ak_id][timeslot_id] + person_var[ak_id][
                        person_id
                    ] <= 1, _construct_constraint_name(
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
                    prob += room_var[ak_id][room_id] + person_var[ak_id][
                        person_id
                    ] <= 1, _construct_constraint_name(
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
                prob += room_var[ak_id][room_id] + time_var[ak_id][
                    timeslot_id
                ] <= 1, _construct_constraint_name(
                    "TimeImpossibleForRoom", room_id, timeslot_id, ak_id
                )

    # The problem data is written to an .lp file
    if output_file is not None:
        prob.writeLP(output_file)

    return prob, (room_var, time_var, person_var)


def export_scheduling_result(
    input_data: SchedulingInput,
    solved_lp_problem: LpProblem,
    dec_vars: tuple[
        dict[str, dict[str, LpVariable]],
        dict[str, dict[str, LpVariable]],
        dict[str, dict[str, LpVariable]],
    ],
    allow_unscheduled_aks: bool = False,
) -> dict[str, Any]:
    ak_ids, person_ids, room_ids, timeslot_ids = get_ids(input_data)

    (room_var, time_var, person_var) = dec_vars

    def _get_val(var: LpVariable) -> int:
        ret_val = (
            round(var.solverVar.X)
            if solved_lp_problem.solver.name == "GUROBI"
            else value(var)
        )
        return cast(int, ret_val)

    tmp_res_dir: dict[str, dict[str, dict[str, set[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )
    tmp_res_dir = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    for ak_id in ak_ids:
        room_for_ak = None
        for room_id in room_ids:
            if _get_val(room_var[ak_id][room_id]) == 1:
                if room_for_ak is None:
                    room_for_ak = room_id
                else:
                    raise ValueError(f"AK {ak_id} is assigned multiple rooms")
        if room_for_ak is None:
            if allow_unscheduled_aks:
                continue
            else:
                raise ValueError(f"no room assigned to ak {ak_id}")
        for timeslot_id in timeslot_ids:
            if _get_val(time_var[ak_id][timeslot_id]) == 1:
                tmp_res_dir[ak_id][room_for_ak]["timeslot_ids"].add(timeslot_id)
        for person_id in person_ids:
            if _get_val(person_var[ak_id][person_id]) == 1:
                tmp_res_dir[ak_id][room_for_ak]["participant_ids"].add(person_id)

    output_dict: dict[str, Any] = {}
    output_dict["scheduled_aks"] = [
        {
            "ak_id": ak_id,
            "room_id": room_id,
            "timeslot_ids": list(subsubdict["timeslot_ids"]),
            "participant_ids": list(subsubdict["participant_ids"]),
        }
        for ak_id, subdict in tmp_res_dir.items()
        for room_id, subsubdict in subdict.items()
    ]
    output_dict["input"] = input_data.to_dict()

    return output_dict


def solve_scheduling(
    input_data: SchedulingInput,
    mu: float,
    solver_name: str | None = None,
    output_lp_file: str | None = "koma-plan.lp",
    output_json_file: str | None = "output.json",
    **solver_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Solve the scheduling problem.

    Solves the MILP scheduling problem described by the input data using an MILP
    formulation.

    For a specification of the input format, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format

    For a specification of the MILP used, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/LP-formulation

    The MILP models each person to have three kinds of prefences for an AK:
    0 (no preference), 1 (weak preference) and `mu` (strong preference).
    The choice of `mu` is an hyperparameter of the MILP that weights the
    balance between weak and strong preferences.

    Args:
        input_data (SchedulingInput): The input data used to construct the MILP.
        mu (float): The weight associated with a strong preference for an AK.
        solver_name (str, optional): The solver to use. If None, uses pulp's
            default solver. Defaults to None.
        output_lp_file (str, optional): If not None, the created LP is written
            as an `.lp` file to this location. Defaults to `koma-plan.lp`.
        output_json_file (str, optional): If not None, the result dict is written
            as an `.json` file to this location. Defaults to `output.json`.
        **solver_kwargs: kwargs are passed to the solver.

    Returns:
        A dictionary containing the scheduled aks as well as the scheduling
        input.
    """
    lp_problem, dec_vars = create_lp(input_data, mu, output_lp_file)

    if not solver_name:
        # The problem is solved using PuLP's default solver
        solver_name = "PULP_CBC_CMD"
    solver = getSolver(solver_name, **solver_kwargs)
    lp_problem.solve(solver)

    # The status of the solution is printed to the screen
    print("Status:", LpStatus[lp_problem.status])

    return export_scheduling_result(input_data, lp_problem, dec_vars)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu", type=float, default=2)
    parser.add_argument("--solver", type=str, default=None)
    parser.add_argument("--solver-path", type=str)
    parser.add_argument("--warm-start", action="store_true", default=False)
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
    parser.add_argument("path", type=str)
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
        solver_kwargs["Threads"] = args.threads

    json_file = Path(args.path)
    assert json_file.suffix == ".json"
    with json_file.open("r") as f:
        input_dict = json.load(f)

    solve_scheduling(
        SchedulingInput.from_dict(input_dict), args.mu, args.solver, **solver_kwargs
    )


if __name__ == "__main__":
    main()
