"""Unit tests to check feasibility of the constructed schedules."""

import json
from itertools import product
from pathlib import Path
from typing import TypeVar, cast

import linopy
import linopy.solvers
import numpy as np
import numpy.typing as npt
import pytest
from _pytest.mark import ParameterSet

from akplan import types
from akplan.solve import process_solved_lp, solve_scheduling
from akplan.util import (
    AKData,
    ParticipantData,
    RoomData,
    ScheduleAtom,
    SchedulingInput,
    SolverConfig,
    TimeSlotData,
    default_num_threads,
)

T = TypeVar("T")


def _test_uniqueness(
    lst: list[T],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.intp], bool]:
    arr = np.asarray(lst, dtype=np.int64)
    unique_vals, cnts = np.unique(arr, axis=0, return_counts=True)
    return unique_vals, cnts, not bool(np.abs(cnts - 1).sum())


@pytest.fixture(
    scope="module",
    params=[
        "examples/test_20a_20p_3r_5rc_0.25rc-lam_0.json",
        "examples/test_20a_40p_4r_5rc_0.25rc-lam_0.json",
        "examples/test_20a_100p_4r_5rc_0.25rc-lam_0.json",
        "examples/test_10a_15p_4r_5rc_0.25rc-lam_0.json",
        "examples/test_20a_20p_5r_5rc_0.25rc-lam_3confl_3dep_0.json",
        "examples/test_20a_20p_5r_5rc_0.25rc-lam_5confl_5dep_0.json",
        "examples/test_20a_20p_5r_5rc_0.25rc-lam_10confl_0.json",
        "examples/test_20a_20p_5r_5rc_0.25rc-lam_10dep_0.json",
        "examples/test1.json",
        pytest.param(
            "examples/test_30a_20p_3r_5rc_0.25rc-lam_0.json", marks=pytest.mark.slow
        ),
        pytest.param(
            "examples/test_40a_10p_4r_5rc_0.25rc-lam_0.json", marks=pytest.mark.slow
        ),
        pytest.param(
            "examples/test_40a_70p_4r_10rc_1.00rc-lam_0.json", marks=pytest.mark.slow
        ),
        pytest.param("examples/test2.json", marks=pytest.mark.slow),
    ],
)
def scheduling_input(request: pytest.FixtureRequest) -> SchedulingInput:
    """Load scheduling input from a JSON file."""
    json_file = Path(request.param)
    assert json_file.suffix == ".json"
    with json_file.open("r") as f:
        input_dict = json.load(f)

    return SchedulingInput.from_dict(input_dict)


mus: list[float] = [2, 1, 5]
fast_mu_values = mus[:1]
available_solvers = linopy.solvers.available_solvers + [None]
core_solver_set = {"highs", "gurobi"}
licensed_solvers = {"gurobi"}


scheduling_params: list[ParameterSet] = []
for mu, solver_name in product(mus, available_solvers):
    marks = []
    if solver_name not in core_solver_set:
        marks.extend([pytest.mark.slow, pytest.mark.extensive])
    elif mu not in fast_mu_values:
        marks.append(pytest.mark.slow)

    if solver_name in licensed_solvers:
        marks.append(pytest.mark.licensed)

    scheduling_params.append(pytest.param((mu, solver_name), marks=marks))


scheduling_param_ids: list[str] = []
for param in scheduling_params:
    mu, solver_name = cast(tuple[float, str], param.values[0])
    scheduling_param_ids.append(f"mu={mu}-{solver_name}")


@pytest.fixture(
    scope="module",
    ids=scheduling_param_ids,
    params=scheduling_params,
)
def solved_lp_fixture(
    request: pytest.FixtureRequest, scheduling_input: SchedulingInput
) -> tuple[linopy.Model, types.ExportTuple, SchedulingInput]:
    """Solve an ILP."""
    mu, solver_name = request.param
    solver_config = SolverConfig(
        threads=default_num_threads(),
        time_limit=60,
    )
    scheduling_input.config.mu = mu

    solution_tuple = solve_scheduling(
        scheduling_input,
        solver_config=solver_config,
        solver_name=solver_name,
    )
    assert solution_tuple is not None, "Model is infeasible!"

    return (*solution_tuple, scheduling_input)


@pytest.fixture(scope="module")
def scheduled_aks(
    solved_lp_fixture: tuple[linopy.Model, types.ExportTuple, SchedulingInput],
) -> dict[types.AkId, ScheduleAtom]:
    """Construct a schedule from solved ILP."""
    solved_lp_problem, solution, scheduling_input = solved_lp_fixture

    schedule = process_solved_lp(
        solved_lp_problem, solution, input_data=scheduling_input
    )

    if schedule is None:
        pytest.skip("No LP solution found")

    return schedule


@pytest.fixture(scope="module")
def ak_dict(scheduling_input: SchedulingInput) -> dict[types.AkId, AKData]:
    """Construct dict mapping AK ids to AKs."""
    return {ak.id: ak for ak in scheduling_input.aks}


@pytest.fixture(scope="module")
def participant_dict(
    scheduling_input: SchedulingInput,
) -> dict[types.PersonId, ParticipantData]:
    """Construct dict mapping participant ids to participant."""
    return {
        participant.id: participant for participant in scheduling_input.participants
    }


@pytest.fixture(scope="module")
def room_dict(scheduling_input: SchedulingInput) -> dict[types.RoomId, RoomData]:
    """Construct dict mapping room ids to rooms."""
    return {room.id: room for room in scheduling_input.rooms}


@pytest.fixture(scope="module")
def timeslot_dict(
    scheduling_input: SchedulingInput,
) -> dict[types.TimeslotId, TimeSlotData]:
    """Construct dict mapping timeslot ids to timeslots."""
    return {
        timeslot.id: timeslot
        for block in scheduling_input.timeslot_blocks
        for timeslot in block
    }


@pytest.fixture(scope="module")
def timeslot_blocks(scheduling_input: SchedulingInput) -> list[list[TimeSlotData]]:
    """Timeslot blocks of the scheduling input."""
    return scheduling_input.timeslot_blocks


def test_rooms_not_overbooked(scheduled_aks: dict[types.AkId, ScheduleAtom]) -> None:
    """Test that no room is used more than once at a time."""
    assert _test_uniqueness(
        [
            (ak.room_id, timeslot_id)
            for ak in scheduled_aks.values()
            for timeslot_id in ak.timeslot_ids
        ]
    )[-1]


def test_participant_no_overlapping_timeslot(
    scheduled_aks: dict[types.AkId, ScheduleAtom],
) -> None:
    """Test that no participant visits more than one AK at a time."""
    assert _test_uniqueness(
        [
            (participant_id, timeslot_id)
            for ak in scheduled_aks.values()
            for timeslot_id in ak.timeslot_ids
            for participant_id in ak.participant_ids
        ]
    )[-1]


def test_ak_lengths(
    scheduled_aks: dict[types.AkId, ScheduleAtom], ak_dict: dict[types.AkId, AKData]
) -> None:
    """Test that the scheduled AK length matched the specified one."""
    for ak in scheduled_aks.values():
        timeslots = set(ak.timeslot_ids)
        assert len(ak.timeslot_ids) == len(timeslots)
        assert len(timeslots) == ak_dict[ak.ak_id].duration


def test_room_capacities(
    scheduled_aks: dict[types.AkId, ScheduleAtom],
    room_dict: dict[types.RoomId, RoomData],
) -> None:
    """Test that the room capacity is not exceeded."""
    for ak in scheduled_aks.values():
        participants = set(ak.participant_ids)
        assert len(ak.participant_ids) == len(participants)
        assert ak.room_id is not None
        assert len(participants) <= room_dict[ak.room_id].capacity


def test_timeslots_consecutive(
    scheduled_aks: dict[types.AkId, ScheduleAtom],
    timeslot_blocks: list[list[TimeSlotData]],
) -> None:
    """Test that the scheduled timeslots for an AK are consecutive."""
    for ak in scheduled_aks.values():
        timeslots = [
            (block_idx, timeslot_idx)
            for block_idx, block in enumerate(timeslot_blocks)
            for timeslot_idx, timeslot in enumerate(block)
            if timeslot.id in ak.timeslot_ids
        ]
        timeslots.sort()

        for (prev_block_idx, prev_timeslot_idx), (
            next_block_idx,
            next_timeslot_idx,
        ) in zip(timeslots, timeslots[1:]):
            assert prev_timeslot_idx + 1 == next_timeslot_idx
            assert prev_block_idx == next_block_idx


def test_room_constraints(
    scheduled_aks: dict[types.AkId, ScheduleAtom],
    ak_dict: dict[types.AkId, AKData],
    participant_dict: dict[types.PersonId, ParticipantData],
    room_dict: dict[types.RoomId, RoomData],
) -> None:
    """Test that the room constraints are fulfilled."""
    for ak in scheduled_aks.values():
        assert ak.room_id is not None
        fulfilled_room_constraints = set(
            room_dict[ak.room_id].fulfilled_room_constraints
        )
        room_constraints_ak = set(ak_dict[ak.ak_id].room_constraints)
        if ak.participant_ids:
            room_constraints_participants = set.union(
                *(
                    set(participant_dict[participant_id].room_constraints)
                    for participant_id in ak.participant_ids
                )
            )
        else:
            room_constraints_participants = set()
        assert not room_constraints_ak.difference(fulfilled_room_constraints)
        assert not room_constraints_participants.difference(fulfilled_room_constraints)


def test_time_constraints(
    scheduled_aks: dict[types.AkId, ScheduleAtom],
    ak_dict: dict[types.AkId, AKData],
    participant_dict: dict[types.PersonId, ParticipantData],
    room_dict: dict[types.RoomId, RoomData],
    timeslot_dict: dict[types.TimeslotId, TimeSlotData],
) -> None:
    """Test that the time constraints are fulfilled."""
    for ak in scheduled_aks.values():
        assert ak.room_id is not None
        time_constraints_room = set(room_dict[ak.room_id].time_constraints)
        time_constraints_ak = set(ak_dict[ak.ak_id].time_constraints)

        fullfilled_time_constraints = set(
            timeslot_dict[ak.timeslot_ids[0]].fulfilled_time_constraints
        )
        for timeslot_id in ak.timeslot_ids[1:]:
            fullfilled_time_constraints = fullfilled_time_constraints.intersection(
                set(timeslot_dict[timeslot_id].fulfilled_time_constraints)
            )
        if ak.participant_ids:
            time_constraints_participants = set.union(
                *(
                    set(participant_dict[participant_id].time_constraints)
                    for participant_id in ak.participant_ids
                )
            )
        else:
            time_constraints_participants = set()
        assert not time_constraints_room.difference(fullfilled_time_constraints)
        assert not time_constraints_ak.difference(fullfilled_time_constraints)
        assert not time_constraints_participants.difference(fullfilled_time_constraints)


def test_required(
    scheduled_aks: dict[types.AkId, ScheduleAtom],
    participant_dict: dict[types.PersonId, ParticipantData],
) -> None:
    """Test that the required preferences are fulfilled."""
    for participant_id, participant in participant_dict.items():
        for pref in participant.preferences:
            pref_fulfilled = participant_id in scheduled_aks[pref.ak_id].participant_ids
            # required => pref_fulfilled
            # equivalent to (not required) or pref_fulfilled
            assert not pref.required or pref_fulfilled


def test_conflicts(
    scheduled_aks: dict[types.AkId, ScheduleAtom], ak_dict: dict[types.AkId, AKData]
) -> None:
    """Test that conflicting AKs are not overlapping."""
    for ak_id, ak in ak_dict.items():
        for conflicting_ak in ak.properties.get("conflicts", []):
            ak_timeslots = scheduled_aks[ak_id].timeslot_ids
            conflicting_ak_timeslots = scheduled_aks[conflicting_ak].timeslot_ids
            assert not set(ak_timeslots).intersection(set(conflicting_ak_timeslots))


def test_dependencies(
    scheduled_aks: dict[types.AkId, ScheduleAtom], ak_dict: dict[types.AkId, AKData]
) -> None:
    """Test that AKs do not overlap their dependencies."""
    for ak_id, ak in ak_dict.items():
        for dependent_ak in ak.properties.get("dependencies", []):
            ak_timeslots = scheduled_aks[ak_id].timeslot_ids
            dependent_ak_timeslots = scheduled_aks[dependent_ak].timeslot_ids
            assert max(map(int, dependent_ak_timeslots)) < min(map(int, ak_timeslots))
