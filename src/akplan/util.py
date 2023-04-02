import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    def from_json(cls, filename: str) -> "SchedulingInput":
        json_file = Path(filename)
        assert json_file.suffix == ".json"
        with json_file.open("r") as f:
            input_dict = json.load(f)
        input_vals = input_dict["input"]
        ak_dict = {ak["id"]: ak for ak in input_vals["aks"]}
        room_dict = {room["id"]: room for room in input_vals["rooms"]}
        timeslot_dict = {}
        for block_id, block in enumerate(input_vals["timeslots"]["blocks"]):
            for timeslot in block:
                timeslot_dict[timeslot["id"]] = timeslot

        participant_dict = {
            participant["id"]: participant for participant in input_vals["participants"]
        }

        # scheduled_aks = self.check_ak_list(output_dict["scheduled_aks"])

        return cls(
            ak_dict=ak_dict,
            participant_dict=participant_dict,
            room_dict=room_dict,
            timeslot_dict=timeslot_dict,
        )

    def to_dict(self) -> dict:
        raise NotImplementedError
