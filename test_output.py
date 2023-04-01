import argparse
from collections import defaultdict
import json

import numpy as np


def main(filename: str):
    with open(filename, "r") as f:
        output_dict = json.load(f)

    input_vals = output_dict["input"]
    ak_dict = {ak["id"]: ak for ak in input_vals["aks"]}
    room_dict = {room["id"]: room for room in input_vals["rooms"]}
    timeslot_dict = {}
    for block_id, block in enumerate(input_vals["timeslots"]["blocks"]):
        for timeslot in block:
            timeslot_dict[timeslot["id"]] = timeslot

    participant_dict = {
        participant["id"]: participant for participant in input_vals["participants"]
    }


    def test_uniqueness(lst) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(lst)
        unique_vals, cnts = np.unique(arr, axis=0, return_counts=True)
        return unique_vals, cnts, np.abs(cnts - 1).sum()

    # test all AKs only occur once
    _unique_ak_ids, cnts, not_unique = test_uniqueness(
        [ak["ak_id"] for ak in output_dict["scheduled_aks"]]
    )
    if not_unique:
        out_str = ", ".join(
            [
                f"AK {ak_id}: {cnt}"
                for ak_id, cnt in zip(
                    _unique_ak_ids[cnts > 1], cnts[cnts > 1], strict=True
                )
            ]
        )
        raise AssertionError(f"AK ids not unique: {out_str}")

    scheduled_aks = {ak["ak_id"]: ak for ak in output_dict["scheduled_aks"]}

    # test no room is used more than once at a time
    _unique_pairs, cnts, not_unique = test_uniqueness(
        [
            (ak["room_id"], timeslot_id)
            for ak in scheduled_aks.values()
            for timeslot_id in ak["timeslot_ids"]
        ]
    )
    if not_unique:
        out_str = ", ".join(
            [
                f"(room {room_id}, timeslot {timeslot_id}): {cnt}"
                for (room_id, timeslot_id), cnt in zip(
                    _unique_pairs[cnts > 1], cnts[cnts > 1], strict=True
                )
            ]
        )
        raise AssertionError(f"Multiple AKs scheduled in a room: {out_str}")

    # test no participant visits more than once at a time
    _unique_pairs, cnts, not_unique = test_uniqueness(
        [
            (participant_id, timeslot_id)
            for ak in scheduled_aks.values()
            for timeslot_id in ak["timeslot_ids"]
            for participant_id in ak["participant_ids"]
        ]
    )
    if not_unique:
        out_str = ", ".join(
            [
                f"(participant {participant_id}, timeslot {timeslot_id}): {cnt}"
                for (participant_id, timeslot_id), cnt in zip(
                    _unique_pairs[cnts > 1], cnts[cnts > 1], strict=True
                )
            ]
        )
        raise AssertionError(f"Multiple AKs scheduled in a room: {out_str}")

    # test AK length
    for ak_id, ak in scheduled_aks.items():
        timeslots = set(ak["timeslot_ids"])
        assert len(timeslots) == len(ak["timeslot_ids"])
        assert len(timeslots) == ak_dict[ak_id]["duration"]

    # test room capacity not exceeded
    for ak in scheduled_aks.values():
        participants = set(ak["participant_ids"])
        assert len(participants) == len(ak["participant_ids"])
        assert len(participants) <= room_dict[ak["room_id"]]["capacity"]

    # test timeslots consectutive
    ## TODO

    # test room constraints
    for ak in scheduled_aks.values():
        fulfilled_room_constraints = set(
            room_dict[ak["room_id"]]["fulfilled_room_constraints"]
        )
        room_constraints_ak = set(ak_dict[ak["ak_id"]]["room_constraints"])
        room_constraints_participants = set.union(
            *(
                set(participant_dict[participant_id]["room_constraints"])
                for participant_id in ak["participant_ids"]
            )
        )
        if room_constraints_ak.difference(fulfilled_room_constraints):
            raise AssertionError("Not all AK room constraints met!")
        if room_constraints_participants.difference(fulfilled_room_constraints):
            raise AssertionError("Not all participant room constraints met!")

    # test time constraints
    for ak in scheduled_aks.values():
        time_constraints_room = set(room_dict[ak["room_id"]]["time_constraints"])
        time_constraints_ak = set(ak_dict[ak["ak_id"]]["time_constraints"])

        fullfilled_time_constraints = None
        for timeslot_id in ak["timeslot_ids"]:
            if fullfilled_time_constraints is None:
                fullfilled_time_constraints = set(
                    timeslot_dict[timeslot_id]["fulfilled_time_constraints"]
                )
            else:
                fullfilled_time_constraints = fullfilled_time_constraints.intersection(
                    set(timeslot_dict[timeslot_id]["fulfilled_time_constraints"])
                )

        time_constraints_participants = set.union(
            *(
                set(participant_dict[participant_id]["time_constraints"])
                for participant_id in ak["participant_ids"]
            )
        )

        if time_constraints_room.difference(fullfilled_time_constraints):
            raise AssertionError("Not all room time constraints met!")
        if time_constraints_ak.difference(fullfilled_time_constraints):
            raise AssertionError("Not all AK time constraints met!")
        if time_constraints_participants.difference(fullfilled_time_constraints):
            raise AssertionError("Not all participant time constraints met!")

    num_weak_misses = defaultdict(int)
    num_strong_misses = defaultdict(int)
    num_weak_prefs = defaultdict(int)
    num_strong_prefs = defaultdict(int)
    # test required preferences fulfilled
    for participant_id, participant in participant_dict.items():
        for pref in participant["preferences"]:
            pref_fulfilled = (
                participant_id in scheduled_aks[pref["ak_id"]]["participant_ids"]
            )
            if pref["required"]:
                if not pref_fulfilled:
                    raise AssertionError("Required AK not fulfillable")
            elif pref["preference_score"] == 1:
                num_weak_misses[participant_id] += not pref_fulfilled
                num_weak_prefs[participant_id] += 1
            elif pref["preference_score"] == 2:
                num_strong_misses[participant_id] += not pref_fulfilled
                num_strong_prefs[participant_id] += 1
            else:
                raise NotImplementedError

    # PRINT STATS ABOUT MISSING AKs
    print("\n=== AK STATS ===\n")
    out_lst = []
    for ak in output_dict["scheduled_aks"]:
        ak_name = ak_dict[ak["ak_id"]]["info"]["name"]
        room_name = room_dict[ak["room_id"]]["info"]["name"]
        begin = timeslot_dict[ak["timeslot_ids"][0]]["info"]["start"]
        participant_names = sorted(
            [
                participant_dict[participant_id]["info"]["name"]
                for participant_id in ak["participant_ids"]
            ]
        )
        out_lst.append(
            f"{ak['ak_id']}\t room {ak['room_id']} timeslots{sorted(ak['timeslot_ids'])} - {len(ak['participant_ids'])} paricipants"
        )
    print("\n".join(sorted(out_lst)))

    # PRINT STATS ABOUT MISSING AKs
    print(f"\n{' ' * 5}=== STATS ON PARTICIPANT PREFERENCE MISSES ===\n")
    max_participant_id_len = max(len(participant_id) for participant_id in participant_dict)
    print(f"| {' ' * max_participant_id_len} |    WEAK MISSES    |   STRONG MISSES   |")
    print(f"| {'-' * max_participant_id_len} | {'-' * 17} | {'-' * 17} |")
    for participant_id in participant_dict:
        out_lst = ["|", f"{participant_id}", "|"]
        if num_weak_prefs[participant_id] > 0:
            out_lst.extend(
                [
                    f"{num_weak_misses[participant_id]:2d} / {num_weak_prefs[participant_id]:2d}",
                    f"({num_weak_misses[participant_id] / num_weak_prefs[participant_id]*100: 6.2f}%)",
                    "|"
                ]
            )
        else:
            out_lst.extend(
                [
                    f"{0:2d} / {0:2d}",
                    f"\t({0*100: 6.2f}%)",
                    "|"
                ]
            )
        if num_strong_prefs[participant_id] > 0:
            out_lst.extend(
                [
                    f"{num_strong_misses[participant_id]:2d} / {num_strong_prefs[participant_id]:2d}",
                    f"({num_strong_misses[participant_id] / num_strong_prefs[participant_id]*100: 6.2f}%)",
                    "|"
                ]
            )
        else:
            out_lst.extend(
                [
                    f"{0:2d} / {0:2d}",
                    f"({0*100: 6.2f}%)",
                    "|"
                ]
            )
        print(f" ".join(out_lst))


    print(f"\n=== PARTICIPANT SCHEDULES ===\n")
    participant_schedules = defaultdict(list)
    for ak in output_dict["scheduled_aks"]:
        ak_name = ak_dict[ak["ak_id"]]["info"]["name"]
        room_name = room_dict[ak["room_id"]]["info"]["name"]
        begin = timeslot_dict[ak["timeslot_ids"][0]]["info"]["start"]
        for participant_id in ak["participant_ids"]:
            participant_schedules[participant_id].append(ak["ak_id"])

    for name, schedule in sorted(participant_schedules.items()):
        print(f"{name}:\t {sorted(schedule)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test Output JSON")
    parser.add_argument("outfile", type=str, default="output.json")
    args = parser.parse_args()
    main(args.outfile)
