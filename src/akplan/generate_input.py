"""Generate test inputs."""
import argparse
import json
from collections import defaultdict

import numpy as np


def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persons", type=int, default=30)
    parser.add_argument("--aks", type=int, default=10)
    parser.add_argument("--rooms", type=int, default=4)
    parser.add_argument("--num_room_constraints", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--room_poisson_mean", type=float, default=1)
    args = parser.parse_args()

    rng = np.random.default_rng(seed=args.seed)

    # we have one hour time slots
    # on Tuesday we go from 8-18, Wednesday from 8-16, Thursday from 8-16, Friday from 8-18
    block_sizes = [10, 8, 8, 10]
    block_properties = [
        ("Dienstag", 10),
        ("Mittwoch", 8),
        ("Donnerstag", 8),
        ("Freitag", 10),
    ]

    possible_room_constraints = ["barrierefrei", "Beamer"]

    # create timeslots:
    list_of_time_blocks = []

    global_timeslot_cnt = 0
    for block_id, (block_label, block_size) in enumerate(block_properties):
        fulfilled_time_constraints = [block_label]
        if block_id == 0:  # die Reso Aks sollen alle am ersten Tag stattfinden
            fulfilled_time_constraints.append("ResoAK")
        list_of_time_blocks.append(
            [
                {
                    "id": str(global_timeslot_cnt + slot_idx),
                    "info": {"start": f"{block_label}, {8 + slot_idx} Uhr"},
                    "fulfilled_time_constraints": fulfilled_time_constraints,
                }
                for slot_idx in range(block_size)
            ]
        )
        global_timeslot_cnt += block_size

    time_slot_dictionary = {
        "info": {"duration": "1 Stunde"},
        "blocks": list_of_time_blocks,
    }

    all_room_constraints = [
        f"room-constraint-{idx}" for idx in range(args.num_room_constraints)
    ]
    # create rooms:
    rooms = [
        {
            "id": str(room_idx),
            "info": {"name": f"room {room_idx}"},
            "capacity": rng.integers(low=10, high=51),
            "fulfilled_room_constraints": rng.choice(
                all_room_constraints,
                size=rng.integers(low=0, high=len(all_room_constraints) + 1),
                replace=False,
            ),
            "time_constraints": [],
        }
        for room_idx in range(args.rooms)
    ]

    # create aks
    room_constraint_arr = rng.choice(
        all_room_constraints,
        replace=False,
        size=np.minimum(
            len(all_room_constraints),
            rng.poisson(lam=args.room_poisson_mean, size=args.aks),
        ),
    )
    reso_ak_arr = rng.choice(2, p=[0.8, 0.2], size=args.aks).astype(bool)
    duration_arr = rng.choice(2, size=args.aks) + 1

    aks = [
        {
            "id": str(idx),
            "duration": duration,
            "properties": [],
            "room_constraints": room_constraints,
            "time_constraints": ["ResoAK"] if is_reso_ak else [],
            "info": {
                "name": f"AK {idx}",
                "head": "N/A",
                "description": "N/A",
                "reso": is_reso_ak,
            },
        }
        for idx, (room_constraints, is_reso_ak, duration) in enumerate(
            zip(room_constraint_arr, reso_ak_arr, duration_arr, strict=True)
        )
    ]

    num_preferences_arr = np.minimum(
        rng.poisson(
            lam=max(10, round(0.2 * args.aks)),
            size=args.persons,
        ),
        args.aks,
    )

    sampled_aks = {
        person_idx: set(rng.choice(args.aks, replace=False, size=num_prefs))
        for person_idx, num_prefs in enumerate(num_preferences_arr)
    }

    # 1. Ignore one half of participants
    required_indices = rng.choice(
        args.persons, size=round(0.5 * args.persons), replace=False
    )
    # 2. For each ak sample the person(s) required for the ak (0.1/0.8/0.1 split)
    num_persons_required = rng.choice(3, size=args.aks, p=[0.1, 0.8, 0.1])

    required_aks = defaultdict(set)
    for ak_idx, num_required in enumerate(num_persons_required):
        persons_required_for_ak = rng.choice(
            required_indices, replace=False, size=num_required
        )
        for person_idx in persons_required_for_ak:
            required_aks[person_idx].add(ak_idx)

    def _calc_preferred_score(person_idx: int, ak_idx: int) -> int:
        if ak_idx in required_aks[person_idx]:
            return -1
        return rng.choice(2) + 1

    # TODO: Generate room & time constraints
    participants = [
        {
            "id": str(person_idx),
            "info": {"name": f"Person {person_idx}"},
            "preferences": [
                {
                    "ak_id": ak_idx,
                    "required": ak_idx in required_aks[person_idx],
                    "preference_score": _calc_preferred_score(person_idx, ak_idx),
                }
                for ak_idx in preferred_aks.union(required_aks[person_idx])
            ],
            "room_constraints": [],
            "time_constraints": [],
        }
        for person_idx, preferred_aks in sampled_aks.items()
    ]

    # create dictionary that we later write into the json-file
    dictionary = {
        "aks": aks,
        "rooms": rooms,
        "participants": participants,
        "timeslots": time_slot_dictionary,
        "info": "DummySet",
    }

    # print(dictionary)
    with open("dummy_set.json", "w") as output_file:
        json.dump(dictionary, output_file)


if __name__ == "__main__":
    generate()
