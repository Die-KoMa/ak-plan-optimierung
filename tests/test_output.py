from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.akplan.pulp_solve import solve_scheduling
from src.akplan.util import (
    AKData,
    ParticipantData,
    PreferenceData,
    RoomData,
    SchedulingInput,
    TimeSlotData,
)


def _test_uniqueness(lst) -> tuple[np.ndarray, np.ndarray, bool]:
    arr = np.asarray(lst)
    unique_vals, cnts = np.unique(arr, axis=0, return_counts=True)
    return unique_vals, cnts, not bool(np.abs(cnts - 1).sum())


@pytest.fixture(scope="module", params=["examples/test1.json"])
def scheduling_input(request) -> SchedulingInput:
    json_file = Path(request.param)
    assert json_file.suffix == ".json"
    with json_file.open("r") as f:
        input_dict = json.load(f)

    return SchedulingInput.from_dict(input_dict)


@pytest.fixture(scope="module", params=[(2, None)])
def scheduled_aks(request, scheduling_input) -> dict[str, dict]:
    aks = solve_scheduling(
        scheduling_input,
        mu=request.param[0],
        solver_name=request.param[1],
        output_lp_file=None,
        output_json_file=None,
        **{"timelimit": 60},
    )["scheduled_aks"]

    return {ak["ak_id"]: ak for ak in aks}


@pytest.fixture
def ak_dict(scheduling_input: SchedulingInput) -> dict[str, AKData]:
    return {ak.id: ak for ak in scheduling_input.aks}


@pytest.fixture
def participant_dict(scheduling_input: SchedulingInput) -> dict[str, ParticipantData]:
    return {
        participant.id: participant for participant in scheduling_input.participants
    }


@pytest.fixture
def room_dict(scheduling_input: SchedulingInput) -> dict[str, RoomData]:
    return {room.id: room for room in scheduling_input.rooms}


@pytest.fixture
def timeslot_dict(scheduling_input: SchedulingInput) -> dict[str, TimeSlotData]:
    return {
        timeslot.id: timeslot
        for block in scheduling_input.timeslot_blocks
        for timeslot in block
    }


@pytest.fixture
def timeslot_blocks(scheduling_input: SchedulingInput) -> list[list[TimeSlotData]]:
    return scheduling_input.timeslot_blocks


def test_rooms_not_overbooked(scheduled_aks) -> None:
    # test no room is used more than once at a time
    assert _test_uniqueness(
        [
            (ak["room_id"], timeslot_id)
            for ak in scheduled_aks.values()
            for timeslot_id in ak["timeslot_ids"]
        ]
    )[-1]


def test_participant_no_overlapping_timeslot(scheduled_aks) -> None:
    # test no participant visits more than once at a time
    assert _test_uniqueness(
        [
            (participant_id, timeslot_id)
            for ak in scheduled_aks.values()
            for timeslot_id in ak["timeslot_ids"]
            for participant_id in ak["participant_ids"]
        ]
    )[-1]


def test_ak_lengths(scheduled_aks, ak_dict: dict[str, AKData]) -> None:
    # test AK length
    for ak_id, ak in scheduled_aks.items():
        timeslots = set(ak["timeslot_ids"])
        assert len(ak["timeslot_ids"]) == len(timeslots)
        assert len(timeslots) == ak_dict[ak_id].duration


def test_room_capacities(scheduled_aks, room_dict: dict[str, RoomData]) -> None:
    # test room capacity not exceeded
    for ak in scheduled_aks.values():
        participants = set(ak["participant_ids"])
        assert len(ak["participant_ids"]) == len(participants)
        assert len(participants) <= room_dict[ak["room_id"]].capacity


def test_timeslots_consecutive(
    scheduled_aks, timeslot_blocks: list[list[TimeSlotData]]
) -> bool:
    # test AK timeslot consecutive
    for ak_id, ak in scheduled_aks.items():
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
    # test room constraints
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
    # test time constraints
    for ak_id, ak in scheduled_aks.items():
        time_constraints_room = set(room_dict[ak["room_id"]].time_constraints)
        time_constraints_ak = set(ak_dict[ak_id].time_constraints)

        fullfilled_time_constraints = None
        for timeslot_id in ak["timeslot_ids"]:
            if fullfilled_time_constraints is None:
                fullfilled_time_constraints = set(
                    timeslot_dict[timeslot_id].fulfilled_time_constraints
                )
            else:
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
    # test required preferences fulfilled
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
            out_lst.extend(
                [
                    f"{num_weak_misses[participant_id]:2d} / {num_weak_prefs[participant_id]:2d}",
                    f"({num_weak_misses[participant_id] / num_weak_prefs[participant_id]*100: 6.2f}%)",
                    "|",
                ]
            )
        else:
            out_lst.extend([f"{0:2d} / {0:2d}", f"\t({0*100: 6.2f}%)", "|"])
        if num_strong_prefs[participant_id] > 0:
            out_lst.extend(
                [
                    f"{num_strong_misses[participant_id]:2d} / {num_strong_prefs[participant_id]:2d}",
                    f"({num_strong_misses[participant_id] / num_strong_prefs[participant_id]*100: 6.2f}%)",
                    "|",
                ]
            )
        else:
            out_lst.extend([f"{0:2d} / {0:2d}", f"({0*100: 6.2f}%)", "|"])
        print(f" ".join(out_lst))

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
        ak_name = ak_dict[ak_id].info["name"]
        room_name = room_dict[ak["room_id"]].info["name"]
        begin = timeslot_dict[ak["timeslot_ids"][0]].info["start"]
        participant_names = sorted(
            [
                participant_dict[participant_id].info["name"]
                for participant_id in ak["participant_ids"]
            ]
        )
        out_lst.append(
            f"{ak['ak_id']}\t room {ak['room_id']} timeslots{sorted(ak['timeslot_ids'])} - {len(ak['participant_ids'])} paricipants"
        )
    print("\n".join(sorted(out_lst)))


def _print_participant_schedules(
    scheduled_aks,
    ak_dict: dict[str, AKData],
    room_dict: dict[str, RoomData],
    timeslot_dict: dict[str, TimeSlotData],
) -> None:
    print(f"\n=== PARTICIPANT SCHEDULES ===\n")
    participant_schedules = defaultdict(list)
    for ak_id, ak in scheduled_aks.items():
        ak_name = ak_dict[ak_id].info["name"]
        room_name = room_dict[ak["room_id"]].info["name"]
        begin = timeslot_dict[ak["timeslot_ids"][0]].info["start"]
        for participant_id in ak["participant_ids"]:
            participant_schedules[participant_id].append(ak_id)

    for name, schedule in sorted(participant_schedules.items()):
        print(f"{name}:\t {sorted(schedule)}")
