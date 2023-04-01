import argparse
import json
import warnings
from collections import defaultdict

import numpy as np


class TestInstance:
    def _test_uniqueness(self, lst) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(lst)
        unique_vals, cnts = np.unique(arr, axis=0, return_counts=True)
        return unique_vals, cnts, np.abs(cnts - 1).sum()

    def __init__(self, filename: str, verbose: bool = True):
        with open(filename, "r") as f:
            output_dict = json.load(f)

        input_vals = output_dict["input"]
        self.ak_dict = {ak["id"]: ak for ak in input_vals["aks"]}
        self.room_dict = {room["id"]: room for room in input_vals["rooms"]}
        self.timeslot_dict = {}
        for block_id, block in enumerate(input_vals["timeslots"]["blocks"]):
            for timeslot in block:
                self.timeslot_dict[timeslot["id"]] = timeslot

        self.participant_dict = {
            participant["id"]: participant for participant in input_vals["participants"]
        }

        self.scheduled_aks = self.check_ak_list(output_dict["scheduled_aks"])

    def check_ak_list(self, ak_list: list[dict]) -> bool:
        # test all AKs only occur once
        _unique_ak_ids, cnts, not_unique = self._test_uniqueness(
            [ak["ak_id"] for ak in ak_list]
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
            raise ValueError(f"AK ids not unique: {out_str}")
        return {ak["ak_id"]: ak for ak in ak_list}

    def run_all_tests(self) -> bool:
        return all(
            [
                self.test_rooms_not_overbooked(),
                self.test_participant_no_overlapping_timeslot(),
                self.test_ak_lengths(),
                self.test_room_capacities(),
                self.test_timeslots_consecutive(),
                self.test_room_constraints(),
                self.test_time_constraints(),
                self.test_required(),
            ]
        )

    def test_rooms_not_overbooked(self) -> bool:
        # test no room is used more than once at a time
        _unique_pairs, cnts, not_unique = self._test_uniqueness(
            [
                (ak["room_id"], timeslot_id)
                for ak in self.scheduled_aks.values()
                for timeslot_id in ak["timeslot_ids"]
            ]
        )
        if not_unique and self.verbose:
            out_str = ", ".join(
                [
                    f"(room {room_id}, timeslot {timeslot_id}): {cnt}"
                    for (room_id, timeslot_id), cnt in zip(
                        _unique_pairs[cnts > 1], cnts[cnts > 1], strict=True
                    )
                ]
            )
            print(f"Multiple AKs scheduled in a room: {out_str}")
        return not bool(not_unique)

    def test_participant_no_overlapping_timeslot(self) -> bool:
        # test no participant visits more than once at a time
        _unique_pairs, cnts, not_unique = self._test_uniqueness(
            [
                (participant_id, timeslot_id)
                for ak in self.scheduled_aks.values()
                for timeslot_id in ak["timeslot_ids"]
                for participant_id in ak["participant_ids"]
            ]
        )
        if not_unique and self.verbose:
            out_str = ", ".join(
                [
                    f"(participant {participant_id}, timeslot {timeslot_id}): {cnt}"
                    for (participant_id, timeslot_id), cnt in zip(
                        _unique_pairs[cnts > 1], cnts[cnts > 1], strict=True
                    )
                ]
            )
            print(f"Multiple AKs scheduled in a room: {out_str}")
        return not bool(not_unique)

    def test_ak_lengths(self) -> bool:
        # test AK length
        for ak_id, ak in self.scheduled_aks.items():
            timeslots = set(ak["timeslot_ids"])
            if not (
                len(ak["timeslot_ids"])
                == len(timeslots)
                == self.ak_dict[ak_id]["duration"]
            ):
                return False
        return True

    def test_room_capacities(self) -> bool:
        # test room capacity not exceeded
        for ak in self.scheduled_aks.values():
            participants = set(ak["participant_ids"])
            if not (
                len(ak["participant_ids"])
                == len(participants)
                <= self.room_dict[ak["room_id"]]["capacity"]
            ):
                return False
        return True

    def test_timeslots_consecutive(self) -> bool:
        # test AK timeslot consecutive
        for ak_id, ak in self.scheduled_aks.items():
            timeslots = [
                (timeslot_id, block_idx, timeslot_idx)
                for block_idx, block in enumerate(self.timeslot_dict)
                for timeslot_idx, timeslot_id in enumerate(block)
                if timeslot_id in ak["timeslot_ids"]
            ]
            timeslots.sort(key=lambda x: x[2])
            for idx, (id, block_idx, timeslot_idx) in enumerate(timeslots):
                if idx == 0:
                    continue
                if (
                    timeslots[idx - 1][2] + 1 != timeslot_idx
                    or timeslots[idx - 1][1] != block_idx
                ):
                    return False
        return True

    def test_room_constraints(self) -> bool:
        # test room constraints
        for ak in self.scheduled_aks.values():
            fulfilled_room_constraints = set(
                self.room_dict[ak["room_id"]]["fulfilled_room_constraints"]
            )
            room_constraints_ak = set(self.ak_dict[ak["ak_id"]]["room_constraints"])
            room_constraints_participants = set.union(
                *(
                    set(self.participant_dict[participant_id]["room_constraints"])
                    for participant_id in ak["participant_ids"]
                )
            )
            if room_constraints_ak.difference(fulfilled_room_constraints):
                if self.verbose:
                    print("Not all AK room constraints met!")
                return False
            if room_constraints_participants.difference(fulfilled_room_constraints):
                if self.verbose:
                    print("Not all participant room constraints met!")
                return False
        return True

    def test_time_constraints(self) -> bool:
        # test time constraints
        for ak in self.scheduled_aks.values():
            time_constraints_room = set(
                self.room_dict[ak["room_id"]]["time_constraints"]
            )
            time_constraints_ak = set(self.ak_dict[ak["ak_id"]]["time_constraints"])

            fullfilled_time_constraints = None
            for timeslot_id in ak["timeslot_ids"]:
                if fullfilled_time_constraints is None:
                    fullfilled_time_constraints = set(
                        self.timeslot_dict[timeslot_id]["fulfilled_time_constraints"]
                    )
                else:
                    fullfilled_time_constraints = (
                        fullfilled_time_constraints.intersection(
                            set(
                                self.timeslot_dict[timeslot_id][
                                    "fulfilled_time_constraints"
                                ]
                            )
                        )
                    )

            time_constraints_participants = set.union(
                *(
                    set(self.participant_dict[participant_id]["time_constraints"])
                    for participant_id in ak["participant_ids"]
                )
            )

            if time_constraints_room.difference(fullfilled_time_constraints):
                if self.verbose:
                    print("Not all room time constraints met!")
                return False
            if time_constraints_ak.difference(fullfilled_time_constraints):
                if self.verbose:
                    print("Not all AK time constraints met!")
                return False
            if time_constraints_participants.difference(fullfilled_time_constraints):
                if self.verbose:
                    print("Not all participant time constraints met!")
                return False
        return True

    def test_required(self) -> bool:
        # test required preferences fulfilled
        for participant_id, participant in self.participant_dict.items():
            for pref in participant["preferences"]:
                pref_fulfilled = (
                    participant_id
                    in self.scheduled_aks[pref["ak_id"]]["participant_ids"]
                )
                if pref["required"] and not pref_fulfilled:
                    return False
        return True

    def print_missing_stats(self) -> None:
        num_weak_misses = defaultdict(int)
        num_strong_misses = defaultdict(int)
        num_weak_prefs = defaultdict(int)
        num_strong_prefs = defaultdict(int)
        for participant_id, participant in self.participant_dict.items():
            for pref in participant["preferences"]:
                pref_fulfilled = (
                    participant_id
                    in self.scheduled_aks[pref["ak_id"]]["participant_ids"]
                )
                if pref["preference_score"] == 1:
                    num_weak_misses[participant_id] += not pref_fulfilled
                    num_weak_prefs[participant_id] += 1
                elif pref["preference_score"] == 2:
                    num_strong_misses[participant_id] += not pref_fulfilled
                    num_strong_prefs[participant_id] += 1
                elif pref["required"] and pref["preference_score"] == -1:
                    continue
                else:
                    raise ValueError

        # PRINT STATS ABOUT MISSING AKs
        print(f"\n{' ' * 5}=== STATS ON PARTICIPANT PREFERENCE MISSES ===\n")
        max_participant_id_len = max(
            len(participant_id) for participant_id in self.participant_dict
        )
        print(
            f"| {' ' * max_participant_id_len} |    WEAK MISSES    |   STRONG MISSES   |"
        )
        print(f"| {'-' * max_participant_id_len} | {'-' * 17} | {'-' * 17} |")
        for participant_id in self.participant_dict:
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
            for participant_id in self.participant_dict
            if num_weak_prefs[participant_id] > 0
        ]
        strong_misses_perc = [
            num_strong_misses[participant_id] / num_strong_prefs[participant_id]
            for participant_id in self.participant_dict
            if num_strong_prefs[participant_id] > 0
        ]

        import matplotlib.pyplot as plt

        plt.title("Histogram of percentage of preference misses")
        plt.hist(weak_misses_perc, bins=25, alpha=0.7, label="weak prefs")
        plt.hist(strong_misses_perc, bins=25, alpha=0.7, label="strong prefs")
        plt.legend(loc="upper right")
        plt.show()

    def print_ak_stats(self) -> None:
        # PRINT STATS ABOUT MISSING AKs
        print("\n=== AK STATS ===\n")
        out_lst = []
        for ak_id, ak in self.scheduled_aks.items():
            ak_name = self.ak_dict[ak_id]["info"]["name"]
            room_name = self.room_dict[ak["room_id"]]["info"]["name"]
            begin = self.timeslot_dict[ak["timeslot_ids"][0]]["info"]["start"]
            participant_names = sorted(
                [
                    self.participant_dict[participant_id]["info"]["name"]
                    for participant_id in ak["participant_ids"]
                ]
            )
            out_lst.append(
                f"{ak['ak_id']}\t room {ak['room_id']} timeslots{sorted(ak['timeslot_ids'])} - {len(ak['participant_ids'])} paricipants"
            )
        print("\n".join(sorted(out_lst)))

    def print_participant_schedules(self) -> None:
        print(f"\n=== PARTICIPANT SCHEDULES ===\n")
        participant_schedules = defaultdict(list)
        for ak_id, ak in self.scheduled_aks.items():
            ak_name = self.ak_dict[ak_id]["info"]["name"]
            room_name = self.room_dict[ak["room_id"]]["info"]["name"]
            begin = self.timeslot_dict[ak["timeslot_ids"][0]]["info"]["start"]
            for participant_id in ak["participant_ids"]:
                participant_schedules[participant_id].append(ak["ak_id"])

        for name, schedule in sorted(participant_schedules.items()):
            print(f"{name}:\t {sorted(schedule)}")


def main(filename: str):
    test_instance = TestInstance(filename)
    assert test_instance.run_all_tests()
    test_instance.print_missing_stats()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test Output JSON")
    parser.add_argument("outfile", type=str, default="output.json")
    args = parser.parse_args()
    main(args.outfile)
