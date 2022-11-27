from itertools import combinations, product
from typing import Dict, Optional, Set, Tuple

from pulp import (
    lpSum,
    LpAffineExpression,
    LpBinary,
    LpStatus,
    LpVariable,
    LpProblem,
)


def process_pref_score(preference_score: int, required: bool, mu: float):
    if preference_score in [0, 1]:
        return preference_score
    elif preference_score == 2:
        return mu
    else:
        raise NotImplementedError


def create_lp(input_dict: Dict[str, object], mu: float):
    id_map = lambda x: x["id"]

    room_capacities = {room["id"]: room["capacity"] for room in input_dict["rooms"]}
    ak_durations = {ak["id"]: ak["duration"] for ak in input_dict["aks"]}
    preferences_dict = {
        participant["id"]: participant["preferences"]
        for participant in input_dict["participants"]
    }

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
        for block in input_dict["timeslots"]
        for timeslot in block
    }
    fulfilled_room_constraints = {
        room["id"]: set(room["fulfilled_room_constraints"])
        for room in input_dict["rooms"]
    }

    def _retrieve_val_set(object_key: str, val_key: str) -> Set:
        return {obj[val_key] for obj in input_dict[object_key]}

    ak_ids = _retrieve_val_set("aks", "id")
    room_ids = _retrieve_val_set("rooms", "id")
    participant_ids = _retrieve_val_set("participants", "id")
    timeslot_id = {
        timeslot["id"] for block in input_dict["timeslots"] for timeslot in block
    }

    # TODO Add dummy to participants


    prob = LpProblem("MLP KoMa")

    dec_vars = LpVariable.dicts(
        "DecVar", (ak_ids, timeslot_ids, room_ids, participant_ids), cat=LpBinary
    )


    def get_dummy_id(ak_id: str) -> str:
        raise NotImplementedError


    def is_dummy(participant_id: str) -> bool:
        raise NotImplementedError


    def get_block_based_idx(timeslot_id: str) -> Tuple(int, int):
        pass


    def _construct_constraint_name(name: str, *args) -> str:
        return name + "_" + "_".join(args)


    def _set_decision_variable(
        prob: LpProblem,
        dec_vars: Dict[str, Dict[str, Dict[str, Dict[str, LpVariable]]]],
        ak_id: str,
        timeslot_id: str,
        room_id: str,
        participant_id: str,
        value: float,
        name: Optional[str] = None,
    ) -> None:
        if name is not None:
            name = _construct_constraint_name(
                name, ak_id, timeslot_id, room_id, participant_id
            )
        affine_constraint = LpAffineExpression(
            dec_vars[ak_id][timeslot_id][room_id][participant_id]
        )
        prob += affine_constraint == value, name


    cost_func = LpAffineExpression()
    for participant_id in participant_ids:
        normalizing_factor = len(preferences_dict[participant_id])
        for ak_id in ak_ids:
            coeff = -weighted_preference_dict[participant_id][ak_id]
            coeff /= aks[ak_id]["duration"] * normalizing_factor
            affine_constraint = lpSum(
                [
                    dec_vars[ak_id][timeslot_id][room_id][participant_id]
                    for timeslot_id, room_id in product(timeslot_ids, room_ids)
                ]
            )
            cost_func += coeff * affine_constraint

    prob += cost_func, "cost_function"

    # for all Z, P \neq P_A: \sum_{A, R} Y_{A, Z, R, P} <= 1
    for timeslot_id in timeslot_ids:
        for participant_id in participant_ids:
            if is_dummy(participant_id):
                continue
            affine_constraint = lpSum(
                [
                    dec_vars[ak_id][timeslot_id][room_id][participant_id]
                    for ak_id, room_id in product(ak_ids, room_ids)
                ]
            )
            prob += affine_constraint <= 1, _construct_constraint_name(
                "MaxOneAKperPersonAndTime", timeslot_id, participant_id
            )

    # for all A: \sum_{Z, R} Y_{A, Z, R, P_A} = S_A
    for ak_id in ak_ids:
        affine_constraint = lpSum(
            [
                dec_vars[ak_id][timeslot_id][room_id][get_dummy_id(ak_id)]
                for timeslot_id, room_id in product(timeslot_ids, room_ids)
            ]
        )
        prob += affine_constraint == aks[ak_id]["duration"], _construct_constraint_name(
            "AKLength", ak_id
        )

    # for all A, P \neq P_A: \frac{1}{S_A} \sum_{Z, R} Y_{A, Z, R, P} <= 1
    for ak_id, participant_id in product(ak_ids, participant_ids):
        if is_dummy(participant_id):
            continue
        affine_constraint = lpSum(
            [
                dec_vars[ak_id][timeslot_id][room_id][participant_id]
                for ak_id, room_id in product(ak_id, room_ids)
            ]
        )
        for pref in preferences_dict[participant_id]:
            if pref["ak_id"] == ak_id:
                if pref["required"]:
                    prob += (
                        affine_constraint == aks[ak_id]["duration"],
                        _construct_constraint_name(
                            "PersonNeededForAK",
                            ak_id,
                            participant_id,
                        ),
                    )  ## TODO Check for fixed value
                else:
                    affine_constraint *= 1 / aks[ak_id]["duration"]
                    prob += affine_constraint <= 1, _construct_constraint_name(
                        "NoPartialParticipation",
                        ak_id,
                        participant_id,
                    )
                break
        else:
            affine_constraint *= 1 / aks[ak_id]["duration"]
            prob += affine_constraint <= 1, _construct_constraint_name(
                "NoPartialParticipation",
                ak_id,
                participant_id,
            )


    # for all A, R: \sum_{Z} Y_{A, Z, R, P_A} <= 1
    for ak_id, room_id in product(ak_ids, room_ids):
        affine_constraint = lpSum(
            [
                dec_vars[ak_id][timeslot_id][room_id][get_dummy_id(ak_id)]
                for timeslot_id in timeslot_ids
            ]
        )
        affine_constraint *= 1 / aks[ak_id]["duration"]
        prob += affine_constraint <= 1, _construct_constraint_name(
            "FixedAKRooms", ak_id, room_id
        )

    # Ein AK findet konsekutiv statt:
    for ak_id, room_id in product(ak_ids, room_ids):
        for timeslot_id_a, timeslot_id_b in combinations(timeslot_ids, 2):
            block_idx_a, slot_in_block_idx_a = get_block_based_idx(timeslot_id_a)
            block_idx_b, slot_in_block_idx_b = get_block_based_idx(timeslot_id_b)

            if (
                block_idx_a != block_idx_b
                or abs(slot_in_block_idx_a - slot_in_block_idx_b) >= aks[ak_id]["duration"]
            ):
                affine_constraint = lpSum(
                    [
                        dec_vars[ak_id, timeslot_id_a, room_id, get_dummy_id(ak_id)],
                        dec_vars[ak_id, timeslot_id_b, room_id, get_dummy_id(ak_id)],
                    ]
                )
                prob += affine_constraint <= 1, _construct_constraint_name(
                    "AKConsecutive", ak_id, room_id, timeslot_id_a, timeslot_id_b
                )

    # for all A, Z, R, P\neq P_A: 0 <= Y_{A, Z, R, P_A} - Y_{A, Z, R, P}
    for ak_id, timeslot_id, room_id, participant_id in product(
        ak_ids, timeslot_ids, room_ids, participant_ids
    ):
        affine_constraint = LpAffineExpression(
            dec_vars[ak_id][timeslot_id][room_id][get_dummy_id(ak_id)]
        )
        affine_constraint -= LpAffineExpression(
            dec_vars[ak_id][timeslot_id][room_id][participant_id]
        )
        prob += affine_constraint >= 0, "PersonVisitingAKAtRightTimeAndRoom"

    # for all R, Z: \sum_{A, P\neq P_A} Y_{A, Z, R, P} <= K_R
    for room_id, timeslot_id in product(room_ids, timeslot_ids):
        affine_constraint = lpSum(
            [
                dec_vars[ak_id][timeslot_id][room_id][participant_id]
                for ak_id, participant_id in product(ak_ids, participant_ids)
                if not is_dummy(participant_id)
            ]
        )
        prob += affine_constraint <= rooms[room_id]["capacity"], "Roomsizes"

    # for all Z, R, A'\neq A: Y_{A', Z, R, P_A} = 0
    for timeslot_id, room_id, ak_id, dummy_ak_id in product(
        timeslot_ids, room_ids, ak_ids, ak_ids
    ):
        if ak_id == dummy_ak_id:
            continue
        _set_decision_variable(
            prob,
            ak_id,
            timeslot_id,
            room_id,
            get_dummy_id(dummy_ak_id),
            value=0,
            name="DummyPersonOneAk",
        )

    # If P_{P, A} = 0: Y_{A,*,*,P} = 0 (non-dummy P)
    for participant_id, preferences in preferences_dict.items():
        if is_dummy(participant_id):
            continue
        pref_aks = {pref["ak_id"] for pref in preferences}
        for ak_id, timeslot_id, room_id in product(
            ak_ids.difference(pref_aks), timeslot_ids, room_ids
        ):
            _set_decision_variable(
                prob,
                ak_id,
                timeslot_id,
                room_id,
                participant_id,
                value=0,
                name="PersonNotInterestedInAK",
            )

    for participant_id in participant_ids:
        if is_dummy(participant_id):
            continue
        for timeslot_id in timeslot_ids:
            if participant_time_constraint_dict[participant_id].difference(
                fulfilled_time_constraints[timeslot_id]
            ):
                for ak_id, room_id in product(ak_ids, room_ids):
                    _set_decision_variable(
                        prob,
                        ak_id,
                        timeslot_id,
                        room_id,
                        participant_id,
                        value=0,
                        name="TimeImpossipleForPerson",
                    )

        for room_id in room_ids:
            if participant_room_constraint_dict[participant_id].difference(
                fulfilled_room_constraints[room_id]
            ):
                for ak_id, timeslot_id in product(ak_ids, timeslot_ids):
                    _set_decision_variable(
                        prob,
                        ak_id,
                        timeslot_id,
                        room_id,
                        participant_id,
                        value=0,
                        name="RoomImpossibleForPerson",
                    )

    for ak_id in ak_ids:
        for timeslot_id in timeslot_ids:
            if ak_time_constraint_dict[ak_id].difference(
                fulfilled_time_constraints[timeslot_id]
            ):
                for participant_id, room_id in product(participant_ids, room_ids):
                    _set_decision_variable(
                        prob,
                        ak_id,
                        timeslot_id,
                        room_id,
                        participant_id,
                        value=0,
                        name="TimeImpossipleForAK",
                    )

        for room_id in room_ids:
            if ak_room_constraint_dict[ak_id].difference(
                fulfilled_room_constraints[room_id]
            ):
                for participant_id, timeslot_id in participant_ids, timeslot_ids:
                    _set_decision_variable(
                        prob,
                        ak_id,
                        timeslot_id,
                        room_id,
                        participant_id,
                        value=0,
                        name="RoomImpossibleForPerson",
                    )


    for room_id, timeslot_id in product(room_ids, timeslot_ids):
        if room_time_constraint_dict[room_id].difference(
            fulfilled_time_constraints[timeslot_id]
        ):
            for participant_id, ak_id in product(participant_ids, ak_ids):
                _set_decision_variable(
                    prob,
                    ak_id,
                    timeslot_id,
                    room_id,
                    participant_id,
                    value=0,
                    name="TimeImpossibleForRoom",
                )


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

                    _set_decision_variable(prob, value=0, name=name, **kwargs_dict)


    ## TODO: Set intitial value for eq constraints

    # The problem data is written to an .lp file
    prob.writeLP("Sudoku.lp")

    # The problem is solved using PuLP's choice of Solver
    prob.solve()

    # The status of the solution is printed to the screen
    print("Status:", LpStatus[prob.status])
