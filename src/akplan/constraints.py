"""Functions and types for constructing the LP constraints."""

from collections.abc import Callable, Iterable
from typing import Literal, TypeAlias, TypeVar

from pulp import const
from pulp.mps_lp import MPSCoefficient, MPSConstraint

from .types import AkId, Block, BlockId, PersonId, RoomId, TimeslotId
from .util import ExportLPVarDicts, ProblemProperties, _construct_constraint_name

T = TypeVar("T")

ConstraintItem: TypeAlias = MPSConstraint
ConstraintFunc = Callable[[T], ConstraintItem | None]
TaskItem = tuple[ConstraintFunc[T], Iterable[T]]
PulpConstraintSense = Literal[
    const.LpConstraintLE, const.LpConstraintEQ, const.LpConstraintGE
]


def var_sum_coeffs(*variables: tuple[str, float]) -> list[MPSCoefficient]:
    return [MPSCoefficient(name=var_name, value=coeff) for var_name, coeff in variables]


def var_sum(*variables: str) -> list[MPSCoefficient]:
    return [MPSCoefficient(name=var_name, value=1.0) for var_name in variables]


def construct_constraint(
    name: str,
    coefficients: list[MPSCoefficient],
    rhs: float,
    sense: PulpConstraintSense = const.LpConstraintLE,
) -> MPSConstraint:
    return MPSConstraint(
        name=name,
        sense=sense,
        coefficients=coefficients,
        pi=None,
        constant=-rhs,
    )


def _max_one_ak_per_person_and_time(
    packed: tuple[tuple[AkId, AkId], TimeslotId, PersonId],
    var: ExportLPVarDicts,
    props: ProblemProperties,
) -> ConstraintItem:
    (ak_id1, ak_id2), timeslot_id, person_id = packed
    coefficients = var_sum(
        var.time[ak_id1][timeslot_id],
        var.time[ak_id2][timeslot_id],
        var.person[ak_id1][person_id],
        var.person[ak_id2][person_id],
    )
    name = _construct_constraint_name(
        "MaxOneAKPerPersonAndTime",
        ak_id1,
        ak_id2,
        timeslot_id,
        person_id,
    )
    return construct_constraint(name=name, coefficients=coefficients, rhs=3)


def _max_one_ak_per_room_and_time(
    packed: tuple[tuple[AkId, AkId], TimeslotId, RoomId],
    var: ExportLPVarDicts,
    props: ProblemProperties,
) -> ConstraintItem:
    (ak_id1, ak_id2), timeslot_id, room_id = packed
    coefficients = var_sum(
        var.time[ak_id1][timeslot_id],
        var.time[ak_id2][timeslot_id],
        var.room[ak_id1][room_id],
        var.room[ak_id2][room_id],
    )
    name = _construct_constraint_name(
        "MaxOneAKPerRoomAndTime",
        ak_id1,
        ak_id2,
        timeslot_id,
        room_id,
    )
    return construct_constraint(name=name, coefficients=coefficients, rhs=3)


def _ak_durations(
    ak_id: AkId,
    var: ExportLPVarDicts,
    props: ProblemProperties,
) -> ConstraintItem:
    return construct_constraint(
        name=_construct_constraint_name("AKDuration", ak_id),
        coefficients=var_sum(*var.time[ak_id].values()),
        rhs=props.ak_durations[ak_id],
        sense=const.LpConstraintGE,
    )


def _ak_single_block(
    ak_id: AkId, var: ExportLPVarDicts, props: ProblemProperties
) -> ConstraintItem:
    return construct_constraint(
        name=_construct_constraint_name("AKSingleBlock", ak_id),
        coefficients=var_sum(*var.block[ak_id].values()),
        rhs=1,
    )


def _ak_single_block_per_block(
    packed: tuple[AkId, tuple[BlockId, Block]],
    var: ExportLPVarDicts,
    props: ProblemProperties,
) -> ConstraintItem:
    ak_id, (block_id, block) = packed
    coefficients = [(var.time[ak_id][timeslot_id], 1.0) for timeslot_id in block]
    coefficients.append((var.block[ak_id][block_id], props.ak_durations[ak_id]))
    return construct_constraint(
        name=_construct_constraint_name("AKSingleBlock", ak_id, str(block_id)),
        coefficients=var_sum_coeffs(*coefficients),
        rhs=0.0,
    )


def _room_sizes(
    packed: tuple[RoomId, AkId], var: ExportLPVarDicts, props: ProblemProperties
) -> ConstraintItem | None:
    room_id, ak_id = packed
    if props.ak_num_interested[ak_id] > props.room_capacities[room_id]:
        coefficients = [(v, 1) for v in var.person[ak_id].values()]
        coefficients.append((var.room[ak_id][room_id], props.ak_num_interested[ak_id]))
        return construct_constraint(
            name=_construct_constraint_name("Roomsize", room_id, ak_id),
            coefficients=var_sum_coeffs(*coefficients),
            rhs=props.ak_num_interested[ak_id] + props.room_capacities[room_id],
        )
    return None


def _at_most_one_room_per_ak(
    ak_id: AkId, var: ExportLPVarDicts, props: ProblemProperties
) -> ConstraintItem:
    return construct_constraint(
        name=_construct_constraint_name("AtMostOneRoomPerAK", ak_id),
        coefficients=var_sum(*var.room[ak_id].values()),
        rhs=1,
    )


def _at_least_one_room_per_ak(
    ak_id: AkId, var: ExportLPVarDicts, props: ProblemProperties
) -> ConstraintItem:
    return construct_constraint(
        name=_construct_constraint_name("AtLeastOneRoomPerAK", ak_id),
        coefficients=var_sum(*var.room[ak_id].values()),
        rhs=1,
        sense=const.LpConstraintGE,
    )


def _not_more_people_than_interested(
    ak_id: AkId, var: ExportLPVarDicts, props: ProblemProperties
) -> ConstraintItem:
    # We need this constraint so the Roomsize is correct
    return construct_constraint(
        name=_construct_constraint_name("NotMorePeopleThanInterested", ak_id),
        coefficients=var_sum(*var.person[ak_id].values()),
        rhs=props.ak_num_interested[ak_id],
    )


def _time_impossible_for_person(
    packed: tuple[PersonId, TimeslotId, AkId],
    var: ExportLPVarDicts,
    props: ProblemProperties,
) -> ConstraintItem:
    person_id, timeslot_id, ak_id = packed
    name = _construct_constraint_name(
        "TimePersonVar",
        person_id,
        timeslot_id,
        ak_id,
    )
    coefficients = [
        (var.time[ak_id][timeslot_id], 1),
        (var.person[ak_id][person_id], 1),
    ]
    coefficients.append((var.person_time[person_id][timeslot_id], -1))
    return construct_constraint(
        name=name,
        coefficients=var_sum_coeffs(*coefficients),
        rhs=1,
    )


def _room_impossible_for_person(
    packed: tuple[PersonId, RoomId, AkId],
    var: ExportLPVarDicts,
    props: ProblemProperties,
) -> ConstraintItem:
    person_id, room_id, ak_id = packed
    return construct_constraint(
        name=_construct_constraint_name(
            "RoomImpossibleForPerson", person_id, room_id, ak_id
        ),
        coefficients=var_sum(var.room[ak_id][room_id], var.person[ak_id][person_id]),
        rhs=1,
    )


def _room_for_ak(
    ak_id: AkId, var: ExportLPVarDicts, props: ProblemProperties
) -> ConstraintItem:
    return construct_constraint(
        name=_construct_constraint_name("RoomForAK", ak_id),
        coefficients=var_sum(*var.room[ak_id].values()),
        rhs=1,
        sense=const.LpConstraintGE,
    )


def _time_impossible_for_room(
    packed: tuple[RoomId, TimeslotId, AkId],
    var: ExportLPVarDicts,
    props: ProblemProperties,
) -> ConstraintItem | None:
    room_id, timeslot_id, ak_id = packed

    if props.room_time_constraints[room_id].difference(
        props.fulfilled_time_constraints[timeslot_id]
    ):
        name = _construct_constraint_name(
            "TimeImpossibleForRoom", room_id, timeslot_id, ak_id
        )
        return construct_constraint(
            name=name,
            coefficients=var_sum(
                var.room[ak_id][room_id], var.time[ak_id][timeslot_id]
            ),
            rhs=1,
        )
    return None


def _ak_conflict(
    packed: tuple[TimeslotId, tuple[AkId, AkId]],
    var: ExportLPVarDicts,
    props: ProblemProperties,
) -> ConstraintItem:
    timeslot_id, (ak_a, ak_b) = packed
    return construct_constraint(
        name=_construct_constraint_name("AKConflict", ak_a, ak_b, timeslot_id),
        coefficients=var_sum(var.time[ak_a][timeslot_id], var.time[ak_b][timeslot_id]),
        rhs=1,
    )
