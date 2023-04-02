import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dacite import from_dict


@dataclass(frozen=True)
class AKData:
    id: str
    duration: int
    properties: dict[str, Any]
    room_constraints: list[str]
    time_constraints: list[str]
    info: dict[str, Any]


@dataclass(frozen=True)
class PreferenceData:
    ak_id: str
    required: bool
    preference_score: int


@dataclass(frozen=True)
class ParticipantData:
    id: str
    preferences: list[PreferenceData]
    room_constraints: list[str]
    time_constraints: list[str]
    info: dict[str, Any]


@dataclass(frozen=True)
class RoomData:
    id: str
    capacity: int
    fulfilled_room_constraints: list[str]
    time_constraints: list[str]
    info: dict[str, Any]


@dataclass(frozen=True)
class TimeSlotData:
    id: str
    fulfilled_time_constraints: list[str]
    info: dict[str, Any]


@dataclass(frozen=True)
class SchedulingInput:
    aks: list[AKData]
    participants: list[ParticipantData]
    rooms: list[RoomData]
    timeslot_blocks: list[list[TimeSlotData]]

    @classmethod
    def from_dict(cls, input_dict: str) -> "SchedulingInput":
        aks = [from_dict(data_class=AKData, data=ak) for ak in input_dict["aks"]]
        rooms = [
            from_dict(data_class=RoomData, data=room) for room in input_dict["rooms"]
        ]
        participants = [
            from_dict(data_class=ParticipantData, data=participant)
            for participant in input_dict["participants"]
        ]
        timeslot_blocks = [
            [from_dict(data_class=TimeSlotData, data=timeslot) for timeslot in block]
            for block in input_dict["timeslots"]["blocks"]
        ]

        return cls(
            aks=aks,
            participants=participants,
            rooms=rooms,
            timeslot_blocks=timeslot_blocks,
        )

    def to_dict(self) -> dict:
        raise NotImplementedError
