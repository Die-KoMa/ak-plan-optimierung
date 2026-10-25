"""A pretty-printer for the generated schedules."""

import argparse
import json
import logging

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class AK:
    """an AK."""

    id: int
    name: str
    duration: float
    reso: bool


def parse_aks(aks) -> dict[int, AK]:
    """Parse the list of AKs."""
    result = dict()

    for ak in aks:
        result[ak["id"]] = AK(
            id=ak["id"],
            name=ak["info"]["name"],
            duration=ak["duration"],
            reso=ak["info"]["reso"],
        )

    return result


def parse_rooms(rooms) -> dict[int, str]:
    """Parse the list of rooms."""
    result = dict()

    for room in rooms:
        result[room["id"]] = room["info"]["name"]

    return result


def parse_slots(slots) -> dict[int, str]:
    """Parse the list of time slots."""
    result = dict()

    for block in slots:
        for slot in block:
            result[slot["id"]] = slot["info"]["start"]

    return result


def parse_schedule(
    schedule, *, aks, rooms, participants, slots
) -> dict[int, (str, str, str)]:
    """Parse the list of time slots."""
    result = defaultdict(list)

    for item in schedule:
        ak = item["ak_id"]
        room = item["room_id"]
        for slot in item["timeslot_ids"]:
            result[slot] += [(slots[slot], rooms[room], aks[ak].name)]

    return result


def parse_participants(participants) -> dict[int, str]:
    """Parse the list of time slots."""
    result = dict()

    for participant in participants:
        result[participant["id"]] = participant["info"]["name"]

    return result


def pprint_json(path: str) -> None:
    """Pretty-print the schedule at `path`."""
    with open(path) as input_schedule:
        data = json.load(input_schedule)

        aks = parse_aks(data["input"]["aks"])
        rooms = parse_rooms(data["input"]["rooms"])
        participants = parse_participants(data["input"]["participants"])
        info = data["input"]["info"]
        slots = parse_slots(data["input"]["timeslots"]["blocks"])
        schedule = parse_schedule(
            data["scheduled_aks"],
            aks=aks,
            rooms=rooms,
            participants=participants,
            slots=slots,
        )

        for slot in sorted(schedule.keys()):
            print(repr(schedule[slot]))


def main() -> None:
    """Pretty-print the optimised schedule."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loglevel",
        type=str.lower,
        choices=["error", "warning", "info", "debug"],
        default="info",
        help="Select logging level. Defaults to 'info'.",
    )
    parser.add_argument("path", type=str, help="Path of the JSON input file.")

    args = parser.parse_args()

    # set logging level
    numeric_loglevel = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_loglevel, int):
        raise ValueError(f"Invalid log level: {args.loglevel}")
    logging.basicConfig(
        level=numeric_loglevel,
        format="[%(levelname)s] %(asctime)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    pprint_json(args.path)


if __name__ == "__main__":
    main()
