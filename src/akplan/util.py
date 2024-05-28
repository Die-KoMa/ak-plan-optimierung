"""Utility classes and functions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from dacite import from_dict


@dataclass(frozen=True)
class AKData:
    """Dataclass containing the input data of an AK.

    For a specification of the input format, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format

    Args:
        id (str): The unique id of the AK.
        duration (int): The number of consecutive slots needed for the AK.
        properties (dict): A dict containing additional properties of the AK.
        room_constraints (list of str): A list of all room constraints required
            for the AK.
        time_constraints (list of str): A list of all time constraints required
            for the AK.
        info (dict): A dictionary containing additional information about the AK, e.g. a
            human readable name or a description. Not used for the optimization.
    """

    id: str
    duration: int
    properties: dict[str, Any]
    room_constraints: list[str]
    time_constraints: list[str]
    info: dict[str, Any]


@dataclass(frozen=True)
class PreferenceData:
    """Dataclass containing the input data of a preference a participant has.

    For a specification of the input format, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format

    Args:
        ak_id (str): The unique id of the AK the preference object is about.
        required (bool): Whether the participant is required for the AK or not.
        preference_score (int): The score of the preference: not interested (0),
            weakly interested (1), strongly interested (2) or required (-1).
    """

    ak_id: str
    required: bool
    preference_score: int


@dataclass(frozen=True)
class ParticipantData:
    """Dataclass containing the input data of a participant.

    For a specification of the input format, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format

    Args:
        id (str): The unique id of the participant.
        prefereces (list of PreferenceData): A list of preferences.
            AKs not contained in this list are assumed to have zero preference.
        room_constraints (list of str): A list of all room constraints required
            by the participant.
        time_constraints (list of str): A list of all time constraints required
            by the participant.
        info (dict): A dictionary containing additional information about the person,
            e.g. a human readable name. Not used for the optimization.
    """

    id: str
    preferences: list[PreferenceData]
    room_constraints: list[str]
    time_constraints: list[str]
    info: dict[str, Any]


@dataclass(frozen=True)
class RoomData:
    """Dataclass containing the input data of a room.

    For a specification of the input format, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format

    Args:
        id (str): The unique id of the room.
        capacity (int): The capacity of the room. May be -1 in case of unbounded
            room capacity.
        fulfilled_room_constraints (list of str): A list of all room constraints
            fulfilled by the room.
        time_constraints (list of str): A list of all time constraints required
            for the room.
        info (dict): A dictionary containing additional information about the room,
            e.g. a human readable name. Not used for the optimization.
    """

    id: str
    capacity: int
    fulfilled_room_constraints: list[str]
    time_constraints: list[str]
    info: dict[str, Any]


@dataclass(frozen=True)
class TimeSlotData:
    """Dataclass containing the input data of a timeslot.

    For a specification of the input format, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format

    Args:
        id (str): The unique id of the timeslot.
        fulfilled_time_constraints (list of str): A list of all time constraints
            fulfilled by the timeslot.
        info (dict): A dictionary containing additional information about the timeslot,
            e.g. a human readable start time and date. Not used for the optimization.
    """

    id: str
    fulfilled_time_constraints: list[str]
    info: dict[str, Any]


@dataclass(frozen=True)
class SchedulingInput:
    """Dataclass containing the input data of the scheduling problem.

    For a specification of the input format, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format

    Args:
        aks (list of AKData): The AKs to schedule.
        participants (list of ParticipantData): The participants according to whose
            preferences the schedule will be created.
        rooms (list of RoomData): The rooms in which the AKs take place.
        timeslot_info (dict): A dictionary containing additional information about
            the timeslots, e.g. the duration of a slot in hours.
            Not used for the optimization.
        timeslot_blocks (list of lists of TimeSlotData): A lost containing the
            timeslot block. Each block is a list of timeslots in chronological order
        info (dict): A dictionary containing additional information about the input,
            e.g. a human readable name of the conference. Not used for the optimization.
    """

    aks: list[AKData]
    participants: list[ParticipantData]
    rooms: list[RoomData]
    timeslot_info: dict[str, str]
    timeslot_blocks: list[list[TimeSlotData]]
    info: dict[str, str]

    @classmethod
    def from_dict(cls, input_dict: dict[str, Any]) -> SchedulingInput:
        """Create a SchedulingInput object from a dictionary."""
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
            timeslot_info=input_dict["timeslots"]["info"],
            info=input_dict["info"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary."""
        return_dict = {
            "aks": [asdict(ak) for ak in self.aks],
            "rooms": [asdict(room) for room in self.rooms],
            "participants": [asdict(participant) for participant in self.participants],
            "info": self.info,
        }
        blocks = [
            [asdict(timeslot) for timeslot in block] for block in self.timeslot_blocks
        ]
        return_dict["timeslots"] = {"info": self.timeslot_info, "blocks": blocks}
        return return_dict
