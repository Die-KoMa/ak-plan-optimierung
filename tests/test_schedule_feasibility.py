"""Unit tests to check feasibility of the constructed schedules."""

import json
import multiprocessing
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pulp
import pytest

from src.akplan.solve import process_solved_lp, solve_scheduling
from src.akplan.util import (
    AKData,
    ParticipantData,
    RoomData,
    SchedulingInput,
    TimeSlotData,
)


def _test_uniqueness(lst) -> tuple[np.ndarray, np.ndarray, bool]:
    arr = np.asarray(lst)
    unique_vals, cnts = np.unique(arr, axis=0, return_counts=True)
    return unique_vals, cnts, not bool(np.abs(cnts - 1).sum())


@pytest.fixture(
    scope="module",
    params=[
        "examples/test_20a_20p_3r_5rc_0.25rc-lam_0.json",
        "examples/test_20a_40p_4r_5rc_0.25rc-lam_0.json",
        "examples/test_20a_100p_4r_5rc_0.25rc-lam_0.json",
        "examples/test_10a_15p_4r_5rc_0.25rc-lam_0.json",
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
def scheduling_input(request) -> SchedulingInput:
    """Load scheduling input from a JSON file."""
    json_file = Path(request.param)
    assert json_file.suffix == ".json"
    with json_file.open("r") as f:
        input_dict = json.load(f)

    return SchedulingInput.from_dict(input_dict)


mus = [2, 1, 5]
available_solvers = pulp.listSolvers(onlyAvailable=True) + [None]
core_solver_set = {"HiGHS_CMD", "GUROBI"}

fast_scheduled_ak_params = list(product(mus[:1], core_solver_set))

scheduled_aks_params = [
    (
        pytest.param(param_pair)
        if param_pair in fast_scheduled_ak_params
        else (
            pytest.param(param_pair, marks=pytest.mark.slow)
            if param_pair[1] in core_solver_set
            else pytest.param(param_pair, marks=[pytest.mark.slow, pytest.mark.extensive])
        )
    )
    for param_pair in product(mus, available_solvers)
]


@pytest.fixture(
    scope="module",
    ids=[
        f"mu={param.values[0][0]}-{param.values[0][1]}"
        for param in scheduled_aks_params
    ],
    params=scheduled_aks_params,
)
def solved_lp_fixture(request, scheduling_input) -> pulp.LpProblem:
    """Solve an ILP."""
    mu, solver_name = request.param
    solver_kwargs = {}
    if solver_name not in ["GLPK_CMD"]:
        solver_kwargs["threads"] = max(1, multiprocessing.cpu_count() - 1)

    return (
        solve_scheduling(
            scheduling_input,
            mu=mu,
            solver_name=solver_name,
            output_lp_file=None,
            timeLimit=60,
            **solver_kwargs,
        ),
        scheduling_input,
    )


@pytest.fixture(scope="module")
def scheduled_aks(solved_lp_fixture) -> dict[str, dict]:
    """Construct a schedule from solved ILP."""
    solved_lp_problem, scheduling_input = solved_lp_fixture

    aks = process_solved_lp(solved_lp_problem, input_data=scheduling_input)

    if aks is None:
        pytest.skip("No LP solution found")

    return {ak["ak_id"]: ak for ak in aks["scheduled_aks"]}


@pytest.fixture(scope="module")
def ak_dict(scheduling_input: SchedulingInput) -> dict[str, AKData]:
    """Construct dict mapping AK ids to AKs."""
    return {ak.id: ak for ak in scheduling_input.aks}


@pytest.fixture(scope="module")
def participant_dict(scheduling_input: SchedulingInput) -> dict[str, ParticipantData]:
    """Construct dict mapping participant ids to participant."""
    return {
        participant.id: participant for participant in scheduling_input.participants
    }


@pytest.fixture(scope="module")
def room_dict(scheduling_input: SchedulingInput) -> dict[str, RoomData]:
    """Construct dict mapping room ids to rooms."""
    return {room.id: room for room in scheduling_input.rooms}


@pytest.fixture(scope="module")
def timeslot_dict(scheduling_input: SchedulingInput) -> dict[str, TimeSlotData]:
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


def test_rooms_not_overbooked(scheduled_aks) -> None:
    """Test that no room is used more than once at a time."""
    assert _test_uniqueness(
        [
            (ak["room_id"], timeslot_id)
            for ak in scheduled_aks.values()
            for timeslot_id in ak["timeslot_ids"]
        ]
    )[-1]


def test_participant_no_overlapping_timeslot(scheduled_aks) -> None:
    """Test that no participant visits more than one AK at a time."""
    assert _test_uniqueness(
        [
            (participant_id, timeslot_id)
            for ak in scheduled_aks.values()
            for timeslot_id in ak["timeslot_ids"]
            for participant_id in ak["participant_ids"]
        ]
    )[-1]


def test_ak_lengths(scheduled_aks, ak_dict: dict[str, AKData]) -> None:
    """Test that the scheduled AK length matched the specified one."""
    for ak_id, ak in scheduled_aks.items():
        timeslots = set(ak["timeslot_ids"])
        assert len(ak["timeslot_ids"]) == len(timeslots)
        assert len(timeslots) == ak_dict[ak_id].duration


def test_room_capacities(scheduled_aks, room_dict: dict[str, RoomData]) -> None:
    """Test that the room capacity is not exceeded."""
    for ak in scheduled_aks.values():
        participants = set(ak["participant_ids"])
        assert len(ak["participant_ids"]) == len(participants)
        assert len(participants) <= room_dict[ak["room_id"]].capacity


def test_timeslots_consecutive(
    scheduled_aks, timeslot_blocks: list[list[TimeSlotData]]
) -> bool:
    """Test that the scheduled timeslots for an AK are consecutive."""
    for ak in scheduled_aks.values():
        timeslots = [
            (block_idx, timeslot_idx)
            for block_idx, block in enumerate(timeslot_blocks)
            for timeslot_idx, timeslot in enumerate(block)
            if timeslot.id in ak["timeslot_ids"]
        ]
        timeslots.sort()

        for (prev_block_idx, prev_timeslot_idx), (
            next_block_idx,
            next_timeslot_idx,
        ) in zip(timeslots, timeslots[1:]):
            assert prev_timeslot_idx + 1 == next_timeslot_idx
            assert prev_block_idx == next_block_idx


def test_room_constraints(
    scheduled_aks,
    ak_dict: dict[str, AKData],
    participant_dict: dict[str, ParticipantData],
    room_dict: dict[str, RoomData],
) -> None:
    """Test that the room constraints are fulfilled."""
    for ak_id, ak in scheduled_aks.items():
        fulfilled_room_constraints = set(
            room_dict[ak["room_id"]].fulfilled_room_constraints
        )
        room_constraints_ak = set(ak_dict[ak_id].room_constraints)
        room_constraints_participants = set.union(
            *(
                set(participant_dict[participant_id].room_constraints)
                for participant_id in ak["participant_ids"]
            )
        )
        assert not room_constraints_ak.difference(fulfilled_room_constraints)
        assert not room_constraints_participants.difference(fulfilled_room_constraints)


def test_time_constraints(
    scheduled_aks,
    ak_dict: dict[str, AKData],
    participant_dict: dict[str, ParticipantData],
    room_dict: dict[str, RoomData],
    timeslot_dict: dict[str, TimeSlotData],
) -> None:
    """Test that the time constraints are fulfilled."""
    for ak_id, ak in scheduled_aks.items():
        time_constraints_room = set(room_dict[ak["room_id"]].time_constraints)
        time_constraints_ak = set(ak_dict[ak_id].time_constraints)

        fullfilled_time_constraints = set(
            timeslot_dict[ak["timeslot_ids"][0]].fulfilled_time_constraints
        )
        for timeslot_id in ak["timeslot_ids"][1:]:
            fullfilled_time_constraints = fullfilled_time_constraints.intersection(
                set(timeslot_dict[timeslot_id].fulfilled_time_constraints)
            )
        time_constraints_participants = set.union(
            *(
                set(participant_dict[participant_id].time_constraints)
                for participant_id in ak["participant_ids"]
            )
        )
        assert not time_constraints_room.difference(fullfilled_time_constraints)
        assert not time_constraints_ak.difference(fullfilled_time_constraints)
        assert not time_constraints_participants.difference(fullfilled_time_constraints)


def test_required(scheduled_aks, participant_dict: dict[str, ParticipantData]) -> None:
    """Test that the required preferences are fulfilled."""
    for participant_id, participant in participant_dict.items():
        for pref in participant.preferences:
            pref_fulfilled = (
                participant_id in scheduled_aks[pref.ak_id]["participant_ids"]
            )
            # required => pref_fulfilled
            # equivalent to (not required) or pref_fulfilled
            assert not pref.required or pref_fulfilled


def _print_missing_stats(
    scheduled_aks, participant_dict: dict[str, ParticipantData]
) -> None:
    num_weak_misses = defaultdict(int)
    num_strong_misses = defaultdict(int)
    num_weak_prefs = defaultdict(int)
    num_strong_prefs = defaultdict(int)
    for participant_id, participant in participant_dict.items():
        for pref in participant.preferences:
            pref_fulfilled = (
                participant_id in scheduled_aks[pref.ak_id]["participant_ids"]
            )
            if pref.preference_score == 1:
                num_weak_misses[participant_id] += not pref_fulfilled
                num_weak_prefs[participant_id] += 1
            elif pref.preference_score == 2:
                num_strong_misses[participant_id] += not pref_fulfilled
                num_strong_prefs[participant_id] += 1
            elif pref.required and pref.preference_score == -1:
                continue
            else:
                raise ValueError

    # PRINT STATS ABOUT MISSING AKs
    print(f"\n{' ' * 5}=== STATS ON PARTICIPANT PREFERENCE MISSES ===\n")
    max_participant_id_len = max(
        len(participant_id) for participant_id in participant_dict
    )
    print(f"| {' ' * max_participant_id_len} |    WEAK MISSES    |   STRONG MISSES   |")
    print(f"| {'-' * max_participant_id_len} | {'-' * 17} | {'-' * 17} |")
    for participant_id in participant_dict:
        out_lst = ["|", f"{participant_id}", "|"]
        if num_weak_prefs[participant_id] > 0:
            weak_perc = num_weak_misses[participant_id] / num_weak_prefs[participant_id]
            out_lst.extend(
                [
                    f"{num_weak_misses[participant_id]:2d}",
                    "/",
                    f"{num_weak_prefs[participant_id]:2d}",
                    f"({weak_perc*100: 6.2f}%)",
                    "|",
                ]
            )
        else:
            out_lst.extend([f"{0:2d} / {0:2d}", f"\t({0*100: 6.2f}%)", "|"])
        if num_strong_prefs[participant_id] > 0:
            strong_perc = (
                num_strong_misses[participant_id] / num_strong_prefs[participant_id]
            )
            out_lst.extend(
                [
                    f"{num_strong_misses[participant_id]:2d}",
                    "/",
                    f"{num_strong_prefs[participant_id]:2d}",
                    f"({strong_perc*100: 6.2f}%)",
                    "|",
                ]
            )
        else:
            out_lst.extend([f"{0:2d} / {0:2d}", f"({0*100: 6.2f}%)", "|"])
        print(" ".join(out_lst))

    weak_misses_perc = [
        num_weak_misses[participant_id] / num_weak_prefs[participant_id]
        for participant_id in participant_dict
        if num_weak_prefs[participant_id] > 0
    ]
    strong_misses_perc = [
        num_strong_misses[participant_id] / num_strong_prefs[participant_id]
        for participant_id in participant_dict
        if num_strong_prefs[participant_id] > 0
    ]

    import matplotlib.pyplot as plt

    plt.title("Histogram of percentage of preference misses")
    plt.hist(weak_misses_perc, bins=25, alpha=0.7, label="weak prefs")
    plt.hist(strong_misses_perc, bins=25, alpha=0.7, label="strong prefs")
    plt.legend(loc="upper right")
    plt.show()


def _print_ak_stats(
    scheduled_aks,
    ak_dict: dict[str, AKData],
    participant_dict: dict[str, ParticipantData],
    room_dict: dict[str, RoomData],
    timeslot_dict: dict[str, TimeSlotData],
) -> None:
    # PRINT STATS ABOUT MISSING AKs
    print("\n=== AK STATS ===\n")
    out_lst = []
    for ak_id, ak in scheduled_aks.items():
        out_lst.append(
            f"{ak_id}\t room {ak['room_id']}"
            f" timeslots{sorted(ak['timeslot_ids'])}"
            f" - {len(ak['participant_ids'])} paricipants"
        )
    print("\n".join(sorted(out_lst)))


def _print_participant_schedules(
    scheduled_aks,
    ak_dict: dict[str, AKData],
    room_dict: dict[str, RoomData],
    timeslot_dict: dict[str, TimeSlotData],
) -> None:
    print("\n=== PARTICIPANT SCHEDULES ===\n")
    participant_schedules = defaultdict(list)
    for ak_id, ak in scheduled_aks.items():
        for participant_id in ak["participant_ids"]:
            participant_schedules[participant_id].append(ak_id)

    for name, schedule in sorted(participant_schedules.items()):
        print(f"{name}:\t {sorted(schedule)}")
