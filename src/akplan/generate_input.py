"""Generate test inputs."""

import argparse
import json
from collections import defaultdict
from typing import Any

import numpy as np


def generate(
    num_persons: int,
    num_aks: int,
    num_rooms: int,
    num_room_constraints: int,
    seed: int,
    room_poisson_mean: float,
) -> dict[str, Any]:
    """Generate a test input to test scheduling.

    TODO: describe test assumptions

    Args:
        num_persons (int): The number of persons in the example.
        num_aks (int): The number of AKs in the example.
        num_rooms (int): The number of rooms in the example.
        num_room_constraints (int): The number of different possible room constraints
            in the example.
        seed (int): The seed to make the generation reproducible.
        room_poisson_mean (float): The number of room constraints a AK requests is
            modelled with a Poisson distribution with this mean.

    Returns:
        dict: The generated test input as a dict. For a format
            specification, see
            https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format
    """
    rng = np.random.default_rng(seed=seed)

    # we have one hour time slots
    # on Tuesday we go from 8-18, Wednesday from 8-16,
    #    Thursday from 8-16, Friday from 8-18
    block_properties = [
        ("Dienstag", 10),
        ("Mittwoch", 8),
        ("Donnerstag", 8),
        ("Freitag", 10),
    ]

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
                    "fulfilled_time_constraints": list(fulfilled_time_constraints),
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
        f"room-constraint-{idx}" for idx in range(num_room_constraints)
    ]
    # create rooms:
    rooms = [
        {
            "id": str(room_idx),
            "info": {"name": f"room {room_idx}"},
            "capacity": int(rng.integers(low=10, high=51)),
            "fulfilled_room_constraints": list(
                rng.choice(
                    all_room_constraints,
                    size=rng.integers(low=0, high=len(all_room_constraints) + 1),
                    replace=False,
                )
            ),
            "time_constraints": [],
        }
        for room_idx in range(num_rooms)
    ]

    # create aks
    room_constraint_arr = [
        rng.choice(
            all_room_constraints,
            replace=False,
            size=np.minimum(
                len(all_room_constraints),
                rng.poisson(lam=room_poisson_mean),
            ),
        )
        for _ak_idx in range(num_aks)
    ]
    reso_ak_arr = rng.choice(2, p=[0.8, 0.2], size=num_aks).astype(bool)
    duration_arr = rng.choice(2, size=num_aks) + 1

    aks = [
        {
            "id": str(ak_idx),
            "duration": int(duration),
            "properties": {},
            "room_constraints": list(room_constraints),
            "time_constraints": ["ResoAK"] if is_reso_ak else [],
            "info": {
                "name": f"AK {ak_idx}",
                "head": "N/A",
                "description": "N/A",
                "reso": bool(is_reso_ak),
            },
        }
        for ak_idx, (room_constraints, is_reso_ak, duration) in enumerate(
            zip(room_constraint_arr, reso_ak_arr, duration_arr, strict=True)
        )
    ]

    num_preferences_arr = np.minimum(
        rng.poisson(
            lam=max(10, round(0.2 * num_aks)),
            size=num_persons,
        ),
        num_aks,
    )

    sampled_aks = {
        person_idx: set(rng.choice(num_aks, replace=False, size=num_prefs))
        for person_idx, num_prefs in enumerate(num_preferences_arr)
    }

    # 1. Ignore one half of participants
    required_indices = rng.choice(
        num_persons, size=round(0.5 * num_persons), replace=False
    )
    # 2. For each ak sample the person(s) required for the ak (0.1/0.8/0.1 split)
    num_persons_required = rng.choice(3, size=num_aks, p=[0.1, 0.8, 0.1])

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
        return int(rng.choice(2)) + 1

    # TODO: Generate room & time constraints
    participants = [
        {
            "id": str(person_idx),
            "info": {"name": f"Person {person_idx}"},
            "preferences": [
                {
                    "ak_id": str(ak_idx),
                    "required": bool(ak_idx in required_aks[person_idx]),
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
    return {
        "aks": aks,
        "rooms": rooms,
        "participants": participants,
        "timeslots": time_slot_dictionary,
        "info": "DummySet",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--persons", type=int, default=30)
    parser.add_argument("--aks", type=int, default=10)
    parser.add_argument("--rooms", type=int, default=4)
    parser.add_argument("--num_room_constraints", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--room_poisson_mean", type=float, default=1)
    args = parser.parse_args()

    output_dict = generate(
        num_persons=args.persons,
        num_aks=args.aks,
        num_rooms=args.rooms,
        num_room_constraints=args.num_room_constraints,
        seed=args.seed,
        room_poisson_mean=args.room_poisson_mean,
    )

    filename = "_".join(
        [
            "examples/test",
            f"{args.aks}a",
            f"{args.persons}p",
            f"{args.rooms}r",
            f"{args.num_room_constraints}rc",
            f"{args.room_poisson_mean:.2f}rc-lam",
            f"{args.seed}.json",
        ]
    )

    with open(filename, "w") as output_file:
        json.dump(output_dict, output_file, indent=4)

    print(f"Generated {filename}")
