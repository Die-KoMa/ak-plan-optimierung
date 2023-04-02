import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SchedulingInput:
    ak_dict: dict[str, str | int | list[str] | dict[str, str | bool]]
    participant_dict: dict[
        str, str | dict[str, str], list[dict[str, str | bool | int]] | list[str]
    ]
    room_dict: dict[str, str | int | list[str] | dict[str, str]]
    timeslot_dict: dict[
        str,
        dict[str, str] | list[list[dict[str, str | dict[str, str] | list[str]]]],
    ]

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
