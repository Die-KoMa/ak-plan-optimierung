"""Functions and types for constructing the LP constraints."""

from collections.abc import Callable, Iterable
from typing import TypeVar

from pulp import LpConstraint, lpSum

from .util import LPVarDicts, ProblemProperties, _construct_constraint_name

T = TypeVar("T")

ConstraintItem = tuple[str, LpConstraint]
ConstraintItemOrNone = ConstraintItem | None
ConstraintFunc = Callable[[T], ConstraintItemOrNone]
TaskItem = tuple[ConstraintFunc[T], Iterable[T]]


def _max_one_ak_per_person_and_time(
    packed: tuple[LPVarDicts, tuple[tuple[int, int], int, int]],
) -> tuple[str, LpConstraint]:
    var, ((ak_id1, ak_id2), timeslot_id, person_id) = packed
    constraint = lpSum(
        [
            var.time[ak_id1][timeslot_id],
            var.time[ak_id2][timeslot_id],
            var.person[ak_id1][person_id],
            var.person[ak_id2][person_id],
        ]
    )
    return (
        _construct_constraint_name(
            "MaxOneAKPerPersonAndTime",
            ak_id1,
            ak_id2,
            timeslot_id,
            person_id,
        ),
        constraint <= 3,
    )


def _max_one_ak_per_room_and_time(
    packed: tuple[LPVarDicts, tuple[tuple[int, int], int, int]],
) -> tuple[str, LpConstraint]:
    var, ((ak_id1, ak_id2), timeslot_id, room_id) = packed
    constraint = lpSum(
        [
            var.time[ak_id1][timeslot_id],
            var.time[ak_id2][timeslot_id],
            var.room[ak_id1][room_id],
            var.room[ak_id2][room_id],
        ]
    )
    return (
        _construct_constraint_name(
            "MaxOneAKPerRoomAndTime",
            ak_id1,
            ak_id2,
            timeslot_id,
            room_id,
        ),
        constraint <= 3,
    )


def _ak_durations(
    packed: tuple[LPVarDicts, ProblemProperties, int],
) -> tuple[str, LpConstraint]:
    var, props, ak_id = packed
    constraint = lpSum(var.time[ak_id].values()) >= props.ak_durations[ak_id]
    return _construct_constraint_name("AKDuration", ak_id), constraint


def _ak_single_block(packed: tuple[LPVarDicts, int]) -> tuple[str, LpConstraint]:
    var, ak_id = packed
    constraint = lpSum(var.block[ak_id].values()) <= 1
    return _construct_constraint_name("AKSingleBlock", ak_id), constraint


def _ak_single_block_per_block(
    packed: tuple[LPVarDicts, ProblemProperties, tuple[int, tuple[int, list[int]]]],
) -> tuple[str, LpConstraint]:
    var, props, (ak_id, (block_id, block)) = packed
    constraint_sum = lpSum([var.time[ak_id][timeslot_id] for timeslot_id in block])
    return (
        _construct_constraint_name("AKSingleBlock", ak_id, str(block_id)),
        constraint_sum <= props.ak_durations[ak_id] * var.block[ak_id][block_id],
    )


def _room_sizes(
    packed: tuple[LPVarDicts, ProblemProperties, tuple[int, int]],
) -> tuple[str, LpConstraint] | None:
    var, props, (room_id, ak_id) = packed
    if props.ak_num_interested[ak_id] > props.room_capacities[room_id]:
        constraint_sum = lpSum(var.person[ak_id].values())
        constraint_sum += props.ak_num_interested[ak_id] * var.room[ak_id][room_id]
        constraint = (
            constraint_sum
            <= props.ak_num_interested[ak_id] + props.room_capacities[room_id]
        )
        return _construct_constraint_name("Roomsize", room_id, ak_id), constraint
    return None


def _at_most_one_room_per_ak(
    packed: tuple[LPVarDicts, int],
) -> tuple[str, LpConstraint]:
    var, ak_id = packed
    return (
        _construct_constraint_name("AtMostOneRoomPerAK", ak_id),
        lpSum(var.room[ak_id].values()) <= 1,
    )


def _at_least_one_room_per_ak(
    packed: tuple[LPVarDicts, int],
) -> tuple[str, LpConstraint]:
    var, ak_id = packed
    return (
        _construct_constraint_name("AtLeastOneRoomPerAK", ak_id),
        lpSum(var.room[ak_id].values()) >= 1,
    )


def _not_more_people_than_interested(
    packed: tuple[LPVarDicts, ProblemProperties, int],
) -> tuple[str, LpConstraint]:
    var, props, ak_id = packed
    # We need this constraint so the Roomsize is correct
    constraint_sum = lpSum(var.person[ak_id].values())
    return (
        _construct_constraint_name("NotMorePeopleThanInterested", ak_id),
        constraint_sum <= props.ak_num_interested[ak_id],
    )


def _time_impossible_for_person(
    packed: tuple[LPVarDicts, tuple[int, int, int]],
) -> tuple[str, LpConstraint]:
    var, (person_id, timeslot_id, ak_id) = packed
    constraint_sum = lpSum([var.time[ak_id][timeslot_id], var.person[ak_id][person_id]])
    return (
        _construct_constraint_name(
            "TimePersonVar",
            person_id,
            timeslot_id,
            ak_id,
        ),
        constraint_sum <= var.person_time[person_id][timeslot_id] + 1,
    )


def _room_impossible_for_person(
    packed: tuple[LPVarDicts, tuple[int, int, int]],
) -> tuple[str, LpConstraint]:
    var, (person_id, room_id, ak_id) = packed
    constraint_sum = lpSum([var.room[ak_id][room_id], var.person[ak_id][person_id]])
    return (
        _construct_constraint_name(
            "RoomImpossibleForPerson", person_id, room_id, ak_id
        ),
        constraint_sum <= 1,
    )


def _room_for_ak(packed: tuple[LPVarDicts, int]) -> tuple[str, LpConstraint]:
    var, ak_id = packed
    return (
        _construct_constraint_name("RoomForAK", ak_id),
        lpSum([var.room[ak_id].values()]) >= 1,
    )


def _time_impossible_for_room(
    packed: tuple[LPVarDicts, ProblemProperties, tuple[int, int, int]],
) -> tuple[str, LpConstraint] | None:
    var, props, (room_id, timeslot_id, ak_id) = packed

    if props.room_time_constraints[room_id].difference(
        props.fulfilled_time_constraints[timeslot_id]
    ):
        return (
            _construct_constraint_name(
                "TimeImpossibleForRoom", room_id, timeslot_id, ak_id
            ),
            lpSum([var.room[ak_id][room_id], var.time[ak_id][timeslot_id]]) <= 1,
        )
    return None


def _ak_conflict(
    packed: tuple[LPVarDicts, tuple[int, tuple[int, int]]],
) -> tuple[str, LpConstraint]:
    var, (timeslot_id, (ak_a, ak_b)) = packed
    return (
        _construct_constraint_name("AKConflict", ak_a, ak_b, timeslot_id),
        lpSum([var.time[ak_a][timeslot_id], var.time[ak_b][timeslot_id]]) <= 1,
    )
