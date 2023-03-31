import argparse
import json
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path
from typing import Dict, Optional, Set

from pulp import (
    LpAffineExpression,
    LpBinary,
    LpInteger,
    LpMaximize,
    LpProblem,
    LpStatus,
    LpVariable,
    getSolver,
    lpSum,
    value,
)


def process_pref_score(
    preference_score: int, required: bool, mu: float
) -> float:
    if required or preference_score == -1:
        return 0
    elif preference_score in [0, 1]:
        return preference_score
    elif preference_score == 2:
        return mu
    else:
        raise NotImplementedError(preference_score)


def _construct_constraint_name(name: str, *args) -> str:
    return name + "_" + "_".join(args)


def create_lp(
    input_dict: Dict[str, object], mu: float, args: argparse.Namespace
) -> None:
    """Create the MILP problem as pulp object and solve it.

    Creates the problem with all constraints, preferences and the objective function.
    Runs the solver on the created instance and stores the output as a json file.

    For a specification of the input JSON format, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format

    For a specification of the MILP, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/LP-formulation

    The MILP models each person to have three kinds of prefences for an AK:
    0 (no preference), 1 (weak preference) and `mu` (strong preference).
    The choice of `mu` is an hyperparameter of the MILP that weights the
    balance between weak and strong preferences.

    Args:
        input_dict (dict): The input dictionary as read from the input JSON file.
        mu (float): The weight associated with a strong preference for an AK.
        args (argparse.Namespace): CLI arguments, used to pass options for the
            MILP solver.
    """

    # Get ids from input_dict
    def _retrieve_val_set(object_key: str, val_key: str) -> Set:
        return {obj[val_key] for obj in input_dict[object_key]}

    ak_ids = _retrieve_val_set("aks", "id")
    room_ids = _retrieve_val_set("rooms", "id")
    person_ids = _retrieve_val_set("participants", "id")
    timeslot_ids = {
        timeslot["id"]: timeslot_idx
        for block in input_dict["timeslots"]["blocks"]
        for timeslot_idx, timeslot in enumerate(block)
    }
    block_ids = {
        block_idx: {timeslot["id"] for timeslot in block}
        for block_idx, block in enumerate(input_dict["timeslots"]["blocks"])
    }
    num_people = len(person_ids)

    # Get values needed from the input_dict
    room_capacities = {
        room["id"]: room["capacity"] for room in input_dict["rooms"]
    }
    ak_durations = {ak["id"]: ak["duration"] for ak in input_dict["aks"]}
    weighted_preference_dict = {
        person["id"]: {
            pref["ak_id"]: process_pref_score(
                pref["preference_score"],
                pref["required"],
                mu=mu,
            )
            for pref in person["preferences"]
        }
        for person in input_dict["participants"]
    }
    required_persons = {
        ak_id: {
            person["id"]
            for person in input_dict["participants"]
            for pref in person["preferences"]
            if pref["ak_id"] == ak_id and pref["required"]
        }
        for ak_id in ak_ids
    }

    # Get constraints from input_dict
    participant_time_constraint_dict = {
        participant["id"]: set(participant["time_constraints"])
        for participant in input_dict["participants"]
    }
    participant_room_constraint_dict = {
        participant["id"]: set(participant["room_constraints"])
        for participant in input_dict["participants"]
    }
    ak_time_constraint_dict = {
        ak["id"]: set(ak["time_constraints"]) for ak in input_dict["aks"]
    }
    ak_room_constraint_dict = {
        ak["id"]: set(ak["room_constraints"]) for ak in input_dict["aks"]
    }
    room_time_constraint_dict = {
        room["id"]: set(room["time_constraints"])
        for room in input_dict["rooms"]
    }

    fulfilled_time_constraints = {
        timeslot["id"]: set(timeslot["fulfilled_time_constraints"])
        for block in input_dict["timeslots"]["blocks"]
        for timeslot in block
    }
    fulfilled_room_constraints = {
        room["id"]: set(room["fulfilled_room_constraints"])
        for room in input_dict["rooms"]
    }

    # Create problem
    prob = LpProblem("MLPKoMa", sense=LpMaximize)

    # Create decision variables
    room_var = LpVariable.dicts("Room", (ak_ids, room_ids), cat=LpBinary)
    time_var = LpVariable.dicts(
        "Time",
        (ak_ids, timeslot_ids),
        cat=LpBinary,
    )
    block_var = LpVariable.dicts("Block", (ak_ids, block_ids), cat=LpBinary)
    person_var = LpVariable.dicts("Part", (ak_ids, person_ids), cat=LpBinary)

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
    for (ak_id1, ak_id2), timeslot_id in product(
        combinations(ak_ids, 2), timeslot_ids
    ):
        for person_id in person_ids:
            prob += time_var[ak_id1][timeslot_id] + time_var[ak_id2][
                timeslot_id
            ] + person_var[ak_id1][person_id] + person_var[ak_id2][
                person_id
            ] <= 3, _construct_constraint_name(
                "MaxOneAKPerPersonAndTime",
                ak_id1,
                ak_id2,
                timeslot_id,
                person_id,
            )
        # MaxOneAKPerRoomAndTime
        for room_id in room_ids:
            prob += time_var[ak_id1][timeslot_id] + time_var[ak_id2][
                timeslot_id
            ] + room_var[ak_id1][room_id] + room_var[ak_id2][
                room_id
            ] <= 3, _construct_constraint_name(
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
        ) >= ak_durations[ak_id], _construct_constraint_name(
            "AKDuration", ak_id
        )
        # AKSingleBlock
        prob += lpSum(
            [block_var[ak_id][block_id] for block_id in block_ids]
        ) <= 1, _construct_constraint_name("AKSingleBlock", ak_id)
        for block_id, block in block_ids.items():
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
                        timeslot_ids[timeslot_id_a]
                        - timeslot_ids[timeslot_id_b]
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
        ) + num_people * room_var[ak_id][
            room_id
        ] <= num_people + room_capacities[
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
                    prob += room_var[ak_id][room_id] + person_id[ak_id][
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
    prob.writeLP("koma-plan.lp")

    kwargs_dict = {}
    if args.solver_path:
        kwargs_dict["path"] = args.solver_path
    if args.warm_start:
        kwargs_dict["warmStart"] = True
    if args.timelimit:
        kwargs_dict["timeLimit"] = args.timelimit
    if args.gap_rel:
        kwargs_dict["gapRel"] = args.gap_rel
    if args.gap_abs:
        kwargs_dict["gapAbs"] = args.gap_abs
    if args.threads:
        kwargs_dict["Threads"] = args.threads

    if args.solver:
        solver = getSolver(args.solver, **kwargs_dict)
    else:
        solver = None
    # The problem is solved using PuLP's choice of Solver
    res = prob.solve(solver)

    # The status of the solution is printed to the screen
    print("Status:", LpStatus[prob.status])

    tmp_res_dir = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    for ak_id in ak_ids:
        room_for_ak = None
        for room_id in room_ids:
            if value(room_var[ak_id][room_id]) == 1:
                room_for_ak = room_id
        for timeslot_id in timeslot_ids:
            if value(time_var[ak_id][timeslot_id]) == 1:
                tmp_res_dir[ak_id][room_for_ak]["timeslot_ids"].add(
                    timeslot_id
                )
        for person_id in person_ids:
            if value(person_var[ak_id][person_id]) == 1:
                tmp_res_dir[ak_id][room_for_ak]["participant_ids"].add(
                    person_id
                )

    output_dict = {}
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
    output_dict["input"] = input_dict

    with open("output.json", "w") as output_file:
        json.dump(output_dict, output_file)


def main():
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
        "--gap_rel",
        type=float,
        default=None,
        help="Relative gap as stopping criterion",
    )
    parser.add_argument(
        "--gap_abs",
        type=float,
        default=None,
        help="Absolute gap as stopping criterion",
    )
    parser.add_argument(
        "--threads", type=int, default=None, help="Number of threads to use"
    )
    parser.add_argument("path", type=str)
    args = parser.parse_args()

    json_file = Path(args.path)
    assert json_file.suffix == ".json"
    # Load input json file
    with json_file.open("r") as fp:
        input_dict = json.load(fp)

    create_lp(input_dict, args.mu, args)


if __name__ == "__main__":
    main()
