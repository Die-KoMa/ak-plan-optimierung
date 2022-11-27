from itertools import combinations, product
from typing import Dict, Optional, Tuple

from pulp import (
    lpSum,
    LpAffineExpression,
    LpBinary,
    LpStatus,
    LpVariable,
    LpProblem,
)

prob = LpProblem("MLP KoMa")

rooms = {"room_id": "room_dict"}
aks = {"ak_id": "ak_dict"}

# TODO Add dummy to participants
weighted_preference_dict = {"participant_id": {
    "ak_id": 0
}}
preferences_dict = {"participant_id": "preferences_set"}
participant_time_constraint_dict = {"participant_id": "time_constraint_set"}
participant_room_constraint_dict = {"participant_id": "room_constraint_set"}
ak_time_constraint_dict = {"ak_id": "time_constraint_set"}
ak_room_constraint_dict = {"ak_id": "room_constraint_set"}
room_time_constraint_dict = {"room_id": "time_constraint_set"}


fulfilled_time_constraints = {"timeslot_id": "fulfilled_time_constraint_set"}
fulfilled_room_constraints = {"room_id": "fulfilled_room_constraint_set"}

ak_ids = {}
timeslot_ids = {}
room_ids = {}
participant_ids = {}

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
        coeff = -weighted_preference_dict[participant_id][ak_id]#
        coeff /= aks[ak_id]["duration"] * normalizing_factor
        affine_constraint = lpSum([
            dec_vars[ak_id][timeslot_id][room_id][participant_id]
            for timeslot_id, room_id in product(timeslot_ids, room_ids)
        ])
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
                for ak_id in ak_ids
                for room_id in room_ids
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
            for timeslot_id in timeslot_ids
            for room_id in room_ids
        ]
    )
    prob += affine_constraint == aks[ak_id]["duration"], "AKLength"

# for all A, P \neq P_A: \frac{1}{S_A} \sum_{Z, R} Y_{A, Z, R, P} <= 1
for ak_id in ak_ids:
    for participant_id in participant_ids:
        if is_dummy(participant_id):
            continue
        affine_constraint = lpSum(
            [
                dec_vars[ak_id][timeslot_id][room_id][participant_id]
                for ak_id in ak_ids
                for room_id in room_ids
            ]
        )
        for pref in preferences_dict[participant_id]:
            if pref["ak_id"] == ak_id:
                if pref["required"]:
                    prob += (
                        affine_constraint == aks[ak_id]["duration"],
                        "PersonNeededForAK",
                    )  ## TODO Check for fixed value
                else:
                    affine_constraint *= 1 / aks[ak_id]["duration"]
                    prob += affine_constraint <= 1, "NoPartialParticipation"
                break
        else:
            affine_constraint *= 1 / aks[ak_id]["duration"]
            prob += affine_constraint <= 1, "NoPartialParticipation"


# for all A, R: \sum_{Z} Y_{A, Z, R, P_A} <= 1
for ak_id in ak_ids:
    for room_id in room_ids:
        affine_constraint = lpSum(
            [
                dec_vars[ak_id][timeslot_id][room_id][get_dummy_id(ak_id)]
                for timeslot_id in timeslot_ids
            ]
        )
        affine_constraint *= 1 / aks[ak_id]["duration"]
        prob += affine_constraint <= 1, "FixedAKRooms"

# Ein AK findet konsekutiv statt:
for ak_id in ak_ids:
    for room_id in room_ids:
        for timeslot_id_a in timeslot_ids:
            for timeslot_id_b in timeslot_ids:
                block_idx_a, slot_in_block_idx_a = get_block_based_idx(timeslot_id_a)
                block_idx_b, slot_in_block_idx_b = get_block_based_idx(timeslot_id_b)

                if (
                    block_idx_a != block_idx_b
                    or abs(slot_in_block_idx_a - slot_in_block_idx_b)
                    >= aks[ak_id]["duration"]
                ):
                    affine_constraint = lpSum(
                        [
                            dec_vars[
                                ak_id, timeslot_id_a, room_id, get_dummy_id(ak_id)
                            ],
                            dec_vars[
                                ak_id, timeslot_id_b, room_id, get_dummy_id(ak_id)
                            ],
                        ]
                    )
                    prob += affine_constraint <= 1, "AKConsecutive"

# for all A, Z, R, P\neq P_A: 0 <= Y_{A, Z, R, P_A} - Y_{A, Z, R, P}
for ak_id in ak_ids:
    for timeslot_id in timeslot_ids:
        for room_id in room_ids:
            for participant_id in participant_ids:
                affine_constraint = LpAffineExpression(
                    dec_vars[ak_id][timeslot_id][room_id][get_dummy_id(ak_id)]
                )
                affine_constraint -= LpAffineExpression(
                    dec_vars[ak_id][timeslot_id][room_id][participant_id]
                )
                prob += affine_constraint >= 0, "PersonVisitingAKAtRightTimeAndRoom"

# for all R, Z: \sum_{A, P\neq P_A} Y_{A, Z, R, P} <= K_R
for room_id in room_ids:
    for timeslot_id in timeslot_ids:
        affine_constraint = lpSum(
            [
                dec_vars[ak_id][timeslot_id][room_id][participant_id]
                for ak_id in ak_ids
                for participant_id in participant_ids
                if not is_dummy(participant_id)
            ]
        )
        prob += affine_constraint <= rooms[room_id]["capacity"], "Roomsizes"

# for all Z, R, A'\neq A: Y_{A', Z, R, P_A} = 0
for timeslot_id in timeslot_ids:
    for room_id in room_ids:
        for ak_id in ak_ids:
            for dummy_ak_id in ak_ids:
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
    for ak_id in ak_ids.difference(pref_aks):
        for timeslot_id in timeslot_ids:
            for room_id in room_ids:
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
            for ak_id in ak_ids:
                for room_id in room_ids:
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
            for ak_id in ak_ids:
                for timeslot_id in timeslot_ids:
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
            for participant_id in participant_ids:
                for room_id in room_ids:
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
            for participant_id in participant_ids:
                for timeslot_id in timeslot_ids:
                    _set_decision_variable(
                        prob,
                        ak_id,
                        timeslot_id,
                        room_id,
                        participant_id,
                        value=0,
                        name="RoomImpossibleForPerson",
                    )


for room_id in room_ids:
    for timeslot_id in timeslot_ids:
        if room_time_constraint_dict[room_id].difference(
            fulfilled_time_constraints[timeslot_id]
        ):
            for participant_id in participant_ids:
                for ak_id in ak_ids:
                    _set_decision_variable(
                        prob,
                        ak_id,
                        timeslot_id,
                        room_id,
                        participant_id,
                        value=0,
                        name="TimeImpossibleForRoom",
                    )


## TODO: Set intitial value for eq constraints

# The problem data is written to an .lp file
prob.writeLP("Sudoku.lp")

# The problem is solved using PuLP's choice of Solver
prob.solve()

# The status of the solution is printed to the screen
print("Status:", LpStatus[prob.status])
