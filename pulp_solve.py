import argparse
import json
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path
from typing import Dict, Optional, Set

from pulp import (
    LpAffineExpression,
    LpBinary,
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    getSolver,
    lpSum,
    value,
)


_DUMMY_PARTICIPANT_PREFIX = "DUMMY_PARTICIPANT"


def process_pref_score(preference_score: int, required: bool, mu: float) -> float:
    if required or preference_score == -1:
        return 0
    elif preference_score in [0, 1]:
        return preference_score
    elif preference_score == 2:
        return mu
    else:
        raise NotImplementedError(preference_score)


## TODO könnte auch mit DUMMY_PARTICIPANT_{uuid.uuid4()}_{ak_id} für sichere eindeutigkeit gemacht werden
def get_dummy_participant_id(ak_id: str, dummy_prefix: Optional[str] = None) -> str:
    if not dummy_prefix:
        dummy_prefix = _DUMMY_PARTICIPANT_PREFIX
    return f"{dummy_prefix}_{ak_id}"


def is_participant_dummy(
    participant_id: str, dummy_prefix: Optional[str] = None
) -> bool:
    if not dummy_prefix:
        dummy_prefix = _DUMMY_PARTICIPANT_PREFIX
    return participant_id.startswith(dummy_prefix)


def _construct_constraint_name(name: str, *args) -> str:
    return name + "_" + "_".join(args)


def _set_decision_variable(
    dec_vars: Dict[str, Dict[str, Dict[str, Dict[str, LpVariable]]]],
    ak_id: str,
    timeslot_id: str,
    room_id: str,
    participant_id: str,
    value: float,
    name: Optional[str] = None,
) -> None:
    """Force a decision variable to be a fixed value."""
    if name is not None:
        name = _construct_constraint_name(
            name, ak_id, timeslot_id, room_id, participant_id
        )
    dec_vars[ak_id][timeslot_id][room_id][participant_id].setInitialValue(value)
    dec_vars[ak_id][timeslot_id][room_id][participant_id].fixValue()


def create_lp(
    input_dict: Dict[str, object], mu: float, args: argparse.Namespace
) -> None:
    """Create the milp problem as pulp object and solve it.

    Creates the problem with all constraints, preferences and the objective function.
    Runs the solver on the created instance and stores the output as a json file.

    TODO: Extend docstring.
    """
    # Get values needed from the input_dict
    room_capacities = {room["id"]: room["capacity"] for room in input_dict["rooms"]}
    ak_durations = {ak["id"]: ak["duration"] for ak in input_dict["aks"]}

    # dict of real participants only (without dummy participants) with their preferences dicts
    real_preferences_dict = {
        participant["id"]: participant["preferences"]
        for participant in input_dict["participants"]
    }

    # dict of real participants only (without dummy participants) with numerical preferences
    weighted_preference_dict = {
        participant["id"]: {
            pref["ak_id"]: process_pref_score(
                pref["preference_score"],
                pref["required"],
                mu=mu,
            )
            for pref in participant["preferences"]
        }
        for participant in input_dict["participants"]
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
        room["id"]: set(room["time_constraints"]) for room in input_dict["rooms"]
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

    # Get ids from input_dict
    def _retrieve_val_set(object_key: str, val_key: str) -> Set:
        return {obj[val_key] for obj in input_dict[object_key]}

    ak_ids = _retrieve_val_set("aks", "id")
    room_ids = _retrieve_val_set("rooms", "id")
    timeslot_ids = {
        timeslot["id"]
        for block in input_dict["timeslots"]["blocks"]
        for timeslot in block
    }

    participant_ids = _retrieve_val_set("participants", "id")
    participant_ids = (
        participant_ids.union(  # contains all participants ids (incl. dummy ids)
            {get_dummy_participant_id(ak_id) for ak_id in ak_ids}
        )
    )

    timeslot_block_ids = {
        timeslot["id"]: (block_idx, timeslot_idx)
        for block_idx, block in enumerate(input_dict["timeslots"]["blocks"])
        for timeslot_idx, timeslot in enumerate(block)
    }

    # Create problem
    prob = LpProblem("MLPKoMa", sense=LpMinimize)

    # Create decision variables
    dec_vars = LpVariable.dicts(
        "DecVar", (ak_ids, timeslot_ids, room_ids, participant_ids), cat=LpBinary
    )

    # Set objective function
    # \sum_{P,A,Z,R} - \frac{P_{P,A}}{S_A\sum_{P_{P,A}\neq 0}1} Y_{A,Z,R,P}
    cost_func = LpAffineExpression()
    for participant_id, preferences in real_preferences_dict.items():
        normalizing_factor = len(preferences)
        for ak_id in ak_ids:
            coeff = -weighted_preference_dict[participant_id].get(ak_id, 0)
            coeff /= ak_durations[ak_id] * normalizing_factor
            affine_constraint = lpSum(
                [
                    dec_vars[ak_id][timeslot_id][room_id][participant_id]
                    for timeslot_id, room_id in product(timeslot_ids, room_ids)
                ]
            )
            cost_func += coeff * affine_constraint

    prob += cost_func, "cost_function"

    # Add constraints

    # MaxOneAKperPersonAndTime
    # for all Z, P \neq P_A: \sum_{A, R} Y_{A, Z, R, P} <= 1
    for timeslot_id in timeslot_ids:
        for participant_id in real_preferences_dict:
            affine_constraint = lpSum(
                [
                    dec_vars[ak_id][timeslot_id][room_id][participant_id]
                    for ak_id, room_id in product(ak_ids, room_ids)
                ]
            )
            prob += affine_constraint <= 1, _construct_constraint_name(
                "MaxOneAKperPersonAndTime", timeslot_id, participant_id
            )

    # AKLength
    # for all A: \sum_{Z, R} Y_{A, Z, R, P_A} = S_A
    for ak_id in ak_ids:
        affine_constraint = lpSum(
            [
                dec_vars[ak_id][timeslot_id][room_id][get_dummy_participant_id(ak_id)]
                for timeslot_id, room_id in product(timeslot_ids, room_ids)
            ]
        )
        prob += affine_constraint == ak_durations[ak_id], _construct_constraint_name(
            "AKLength", ak_id
        )

    # NoPartialParticipation
    ## TODO FIXME BUG: Muss =1 oder =0 sein!
    # for all A, P \neq P_A: \frac{1}{S_A} \sum_{Z, R} Y_{A, Z, R, P} <= 1
    # and
    # PersonNeededForAK (stronger than the above)
    # for all A, P \neq P_A: if P essential for A: \sum_{Z,R}Y_{A,Z,R,P}=S_A
    for ak_id in ak_ids:
        for participant_id, preferences in real_preferences_dict.items():
            affine_constraint = lpSum(
                [
                    dec_vars[ak_id][timeslot_id][room_id][participant_id]
                    for ak_id, room_id in product(ak_id, room_ids)
                ]
            )
            for pref in preferences:
                if pref["ak_id"] == ak_id:
                    if pref[
                        "required"
                    ]:  # participant is essential for ak -> set constraint for "PersonNeededForAK"
                        prob += (
                            affine_constraint == ak_durations[ak_id],
                            _construct_constraint_name(
                                "PersonNeededForAK",
                                ak_id,
                                participant_id,
                            ),
                        )  ## TODO Check for fixed value
                    else:  # participant is not essential -> set constraint for "NoPartialParticipation"
                        affine_constraint *= 1 / ak_durations[ak_id]
                        prob += affine_constraint <= 1, _construct_constraint_name(
                            "NoPartialParticipation",
                            ak_id,
                            participant_id,
                        )
                    break
            else:  # participant is not essential -> set constraint for "NoPartialParticipation"
                affine_constraint *= 1 / ak_durations[ak_id]
                prob += affine_constraint <= 1, _construct_constraint_name(
                    "NoPartialParticipation",
                    ak_id,
                    participant_id,
                )

    # FixedAKRooms
    ## TODO FIXME BUG: Muss =1 oder =0 sein!
    # for all A, R: \frac{1}{S_A} \sum_{Z} Y_{A, Z, R, P_A} <= 1
    for ak_id, room_id in product(ak_ids, room_ids):
        affine_constraint = lpSum(
            [
                dec_vars[ak_id][timeslot_id][room_id][get_dummy_participant_id(ak_id)]
                for timeslot_id in timeslot_ids
            ]
        )
        affine_constraint *= 1 / ak_durations[ak_id]
        prob += affine_constraint <= 1, _construct_constraint_name(
            "FixedAKRooms", ak_id, room_id
        )

    # AKConsecutive
    # AKs happen consecutively
    # \forall A,R,Z_{(a, b)},Z_{(c,d)} \text{ s.t. } (a \neq c \vee |b-d| \ge S_A ) : Y_{A,Z_{(a,b)},R,P_A} + Y_{A,Z_{(c,d)},R,P_A} \leq 1
    for ak_id, room_id in product(ak_ids, room_ids):
        for timeslot_id_a, timeslot_id_b in combinations(timeslot_ids, 2):
            block_idx_a, slot_in_block_idx_a = timeslot_block_ids[timeslot_id_a]
            block_idx_b, slot_in_block_idx_b = timeslot_block_ids[timeslot_id_b]
            if (
                block_idx_a != block_idx_b
                or abs(slot_in_block_idx_a - slot_in_block_idx_b) >= ak_durations[ak_id]
            ):  # if two timeslots are too far apart to be consecutive
                affine_constraint = lpSum(
                    [
                        dec_vars[ak_id][timeslot_id_a][room_id][
                            get_dummy_participant_id(ak_id)
                        ],
                        dec_vars[ak_id][timeslot_id_b][room_id][
                            get_dummy_participant_id(ak_id)
                        ],
                    ]
                )
                prob += (
                    affine_constraint <= 1,
                    _construct_constraint_name(  # forbid the ak to happen in both of them
                        "AKConsecutive", ak_id, room_id, timeslot_id_a, timeslot_id_b
                    ),
                )

    # PersonVisitingAKAtRightTimeAndRoom
    # for all A, Z, R, P\neq P_A: 0 <= Y_{A, Z, R, P_A} - Y_{A, Z, R, P}
    for ak_id, timeslot_id, room_id, participant_id in product(
        ak_ids, timeslot_ids, room_ids, participant_ids
    ):  ## TODO dies geht auch durch die dummy participants durch. Ist das notwendig?
        affine_constraint = LpAffineExpression(
            dec_vars[ak_id][timeslot_id][room_id][get_dummy_participant_id(ak_id)]
        )
        affine_constraint -= LpAffineExpression(
            dec_vars[ak_id][timeslot_id][room_id][participant_id]
        )
        prob += affine_constraint >= 0, _construct_constraint_name(
            "PersonVisitingAKAtRightTimeAndRoom",
            ak_id,
            timeslot_id,
            room_id,
            participant_id,
        )

    # Roomsizes
    # for all R, Z: \sum_{A, P\neq P_A} Y_{A, Z, R, P} <= K_R
    for room_id, timeslot_id in product(room_ids, timeslot_ids):
        affine_constraint = lpSum(
            [
                dec_vars[ak_id][timeslot_id][room_id][participant_id]
                for ak_id, participant_id in product(ak_ids, real_preferences_dict)
            ]
        )
        prob += affine_constraint <= room_capacities[
            room_id
        ], _construct_constraint_name("Roomsizes", room_id, timeslot_id)

    # DummyPersonOneAk
    # for all Z, R, A'\neq A: Y_{A', Z, R, P_A} = 0
    for timeslot_id, room_id, ak_id, dummy_ak_id in product(
        timeslot_ids, room_ids, ak_ids, ak_ids
    ):
        if ak_id == dummy_ak_id:
            continue
        _set_decision_variable(
            dec_vars,
            ak_id,
            timeslot_id,
            room_id,
            get_dummy_participant_id(dummy_ak_id),
            value=0,
            name="DummyPersonOneAk",
        )

    # PersonNotInterestedInAK
    # For all A, Z, R, P: If P_{P, A} = 0: Y_{A,Z,R,P} = 0 (non-dummy P)
    for participant_id, preferences in real_preferences_dict.items():
        pref_aks = {
            pref["ak_id"] for pref in preferences
        }  # aks not in pref_aks have P_{P,A} = 0 implicitly
        for ak_id, timeslot_id, room_id in product(
            ak_ids.difference(pref_aks), timeslot_ids, room_ids
        ):
            _set_decision_variable(
                dec_vars,
                ak_id,
                timeslot_id,
                room_id,
                participant_id,
                value=0,
                name="PersonNotInterestedInAK",
            )

    for participant_id in real_preferences_dict:
        # TimeImpossibleForPerson
        # Real person P cannot attend AKs with timeslot Z
        # \forall A,Z,R,P: If P cannot attend at Z: Y_{A,Z,R,P}=0
        for timeslot_id in timeslot_ids:
            if participant_time_constraint_dict[participant_id].difference(
                fulfilled_time_constraints[timeslot_id]
            ):
                for ak_id, room_id in product(ak_ids, room_ids):
                    _set_decision_variable(
                        dec_vars,
                        ak_id,
                        timeslot_id,
                        room_id,
                        participant_id,
                        value=0,
                        name="TimeImpossibleForPerson",
                    )

        # RoomImpossibleForPerson
        # Real person P cannot attend AKs with room R
        # \forall A,Z,R,P: If P cannot attend in R: Y_{A,Z,R,P}=0
        for room_id in room_ids:
            if participant_room_constraint_dict[participant_id].difference(
                fulfilled_room_constraints[room_id]
            ):
                for ak_id, timeslot_id in product(ak_ids, timeslot_ids):
                    _set_decision_variable(
                        dec_vars,
                        ak_id,
                        timeslot_id,
                        room_id,
                        participant_id,
                        value=0,
                        name="RoomImpossibleForPerson",
                    )

    for ak_id in ak_ids:
        # TimeImpossibleForAK
        # \forall A,Z,R,P: If A cannot happen in timeslot Z: Y_{A,Z,R,P} = 0
        for timeslot_id in timeslot_ids:
            if ak_time_constraint_dict[ak_id].difference(
                fulfilled_time_constraints[timeslot_id]
            ):
                for participant_id, room_id in product(participant_ids, room_ids):
                    _set_decision_variable(
                        dec_vars,
                        ak_id,
                        timeslot_id,
                        room_id,
                        participant_id,
                        value=0,
                        name="TimeImpossibleForAK",
                    )
        # RoomImpossibleForAK
        # \forall A,Z,R,P: If A cannot happen in room R: Y_{A,Z,R,P} = 0
        for room_id in room_ids:
            if ak_room_constraint_dict[ak_id].difference(
                fulfilled_room_constraints[room_id]
            ):
                for participant_id, timeslot_id in product(
                    participant_ids, timeslot_ids
                ):
                    _set_decision_variable(
                        dec_vars,
                        ak_id,
                        timeslot_id,
                        room_id,
                        participant_id,
                        value=0,
                        name="RoomImpossibleForAK",
                    )

    # TimeImpossibleForRoom
    # \forall A,Z,R,P: If room R is not available in timeslot Z: Y_{A,Z,R,P}=0
    for room_id, timeslot_id in product(room_ids, timeslot_ids):
        if room_time_constraint_dict[room_id].difference(
            fulfilled_time_constraints[timeslot_id]
        ):
            for participant_id, ak_id in product(participant_ids, ak_ids):
                _set_decision_variable(
                    dec_vars,
                    ak_id,
                    timeslot_id,
                    room_id,
                    participant_id,
                    value=0,
                    name="TimeImpossibleForRoom",
                )

    ## TODO was passiert hier, i have no idea. Wäre cool wenn du das kommentieren könntest @Felix
    def _add_impossible_constraints(
        prob: LpProblem,
        constraint_supplier_type: str,
        constraint_requester_type: str,
        name: str,
    ):
        idx_sets = [ak_ids, timeslot_ids, room_ids, participant_ids]
        requested_constraint_dict = {
            "timeslot_id": {
                "ak_id": ak_time_constraint_dict,
                "participant_id": participant_time_constraint_dict,
                "room_id": room_time_constraint_dict,
            },
            "room_id": {
                "ak_id": ak_room_constraint_dict,
                "participant_id": participant_room_constraint_dict,
            },
        }
        supplied_constraint_dict = {
            "timeslot_id": fulfilled_time_constraints,
            "room_id": fulfilled_room_constraints,
        }

        id_dict = {
            "timeslot_id": timeslot_ids,
            "ak_id": ak_ids,
            "room_id": room_ids,
            "participant_id": participant_ids,
        }

        requester_ids = id_dict[constraint_requester_type]
        supplier_ids = id_dict[constraint_supplier_type]
        other_ids = {
            label: id_dict[label]
            for label in id_dict
            if label not in [constraint_requester_type, constraint_supplier_type]
        }

        for requester_id, supplier_id in product(requester_ids, supplier_ids):
            supplied_constraints = supplied_constraint_dict[supplier_id]
            requested_constraints = requested_constraint_dict[constraint_supplier_type][
                constraint_requester_type
            ][requester_id]
            if requested_constraints.difference(supplied_constraints):
                for lbl_ids in product(other_ids.values()):
                    kwargs_dict = {
                        key: lbl_ids[key_idx] for key_idx, key in enumerate(other_ids)
                    }
                    kwargs_dict[constraint_supplier_type] = supplier_id
                    kwargs_dict[constraint_requester_type] = requester_id

                    _set_decision_variable(value=0, name=name, **kwargs_dict)

    # The problem data is written to an .lp file
    prob.writeLP("koma-plan.lp")

    kwargs_dict = {}
    if args.solver_path:
        kwargs_dict["path"] = args.solver_path
    if args.warm_start:
        kwargs_dict["warmStart"] = True

    if args.solver:
        solver = getSolver(args.solver, **kwargs_dict)
    else:
        solver = None
    # The problem is solved using PuLP's choice of Solver
    res = prob.solve(solver)

    # The status of the solution is printed to the screen
    print("Status:", LpStatus[prob.status])

    tmp_res_dir = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    for ak_id, timeslot_id, room_id, participant_id in product(
        ak_ids, timeslot_ids, room_ids, participant_ids
    ):
        if value(dec_vars[ak_id][timeslot_id][room_id][participant_id]) == 1:
            tmp_res_dir[ak_id][room_id]["timeslot_ids"].add(timeslot_id)
            tmp_res_dir[ak_id][room_id]["participant_ids"].add(participant_id)

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
