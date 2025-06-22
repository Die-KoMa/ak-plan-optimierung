"""Utility classes and functions."""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import asdict, dataclass
from itertools import chain
from typing import Any, Type

from dacite import from_dict
from pulp import LpBinary, LpVariable

VarDict = dict[int, dict[int, LpVariable]]
PartialSolvedVarDict = dict[int, dict[int, int | None]]
SolvedVarDict = dict[int, dict[int, int]]
ConstraintSetDict = dict[int, set[str]]


@dataclass(frozen=True)
class AKData:
    """Dataclass containing the input data of an AK.

    For a specification of the input format, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format

    Args:
        id (int): The unique id of the AK.
        duration (int): The number of consecutive slots needed for the AK.
        properties (dict): A dict containing additional properties of the AK.
        room_constraints (list of str): A list of all room constraints required
            for the AK.
        time_constraints (list of str): A list of all time constraints required
            for the AK.
        info (dict): A dictionary containing additional information about the AK, e.g. a
            human readable name or a description. Not used for the optimization.
    """

    id: int
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
        ak_id (int): The unique id of the AK the preference object is about.
        required (bool): Whether the participant is required for the AK or not.
        preference_score (int): The score of the preference: not interested (0),
            weakly interested (1), strongly interested (2) or required (-1).
    """

    ak_id: int
    required: bool
    preference_score: int


@dataclass(frozen=True)
class ParticipantData:
    """Dataclass containing the input data of a participant.

    For a specification of the input format, see
    https://github.com/Die-KoMa/ak-plan-optimierung/wiki/Input-&-output-format

    Args:
        id (int): The unique id of the participant.
        prefereces (list of PreferenceData): A list of preferences.
            AKs not contained in this list are assumed to have zero preference.
        room_constraints (list of str): A list of all room constraints required
            by the participant.
        time_constraints (list of str): A list of all time constraints required
            by the participant.
        info (dict): A dictionary containing additional information about the person,
            e.g. a human readable name. Not used for the optimization.
    """

    id: int
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
        id (int): The unique id of the room.
        capacity (int): The capacity of the room. May be -1 in case of unbounded
            room capacity.
        fulfilled_room_constraints (list of str): A list of all room constraints
            fulfilled by the room.
        time_constraints (list of str): A list of all time constraints required
            for the room.
        info (dict): A dictionary containing additional information about the room,
            e.g. a human readable name. Not used for the optimization.
    """

    id: int
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
        id (int): The unique id of the timeslot.
        fulfilled_time_constraints (list of str): A list of all time constraints
            fulfilled by the timeslot.
        info (dict): A dictionary containing additional information about the timeslot,
            e.g. a human readable start time and date. Not used for the optimization.
    """

    id: int
    fulfilled_time_constraints: list[str]
    info: dict[str, Any]


@dataclass(frozen=False)
class ScheduleAtom:
    """Dataclass containing one scheduled ak.

    Args:
        ak_id (int): The id of the AK scheduled.
        room_id (int | None): The id of the room, where the AK is scheduled or None
            if the room is not fixed yet.
        timeslot_ids (list of int): The list of timeslots when the AK is scheduled.
        participant_ids (list of int): The list of participants that are meant to go to this AK.
    """

    ak_id: int
    room_id: int | None
    timeslot_ids: list[int]
    participant_ids: list[int]


@dataclass(frozen=False)
class ConfigData:
    """Dataclass containing the config for buildung the ILP and solving it.

    Args:
        mu (float): The weight associated with a strong preference for an AK.
        max_num_timeslots_before_break (int): The maximum number of timeslots any participant
            is planned to go to before a break.
        allow_unscheduled_aks(bool): Whether not scheduling an AK is allowed or not.
        allow_changing_rooms (bool): Whether changing the room for an fixed AK
            is allowed or not.
    """

    mu: float = 2
    max_num_timeslots_before_break: int = 0
    allow_unscheduled_aks: bool = True
    allow_changing_rooms: bool = False


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
    scheduled_aks: list[ScheduleAtom]
    config: ConfigData
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
        scheduled_aks = (
            [
                from_dict(data_class=ScheduleAtom, data=scheduled_ak)
                for scheduled_ak in input_dict["scheduled_aks"]
            ]
            if "scheduled_aks" in input_dict
            else []
        )
        config = (
            from_dict(data_class=ConfigData, data=input_dict["config"])
            if "config" in input_dict
            else ConfigData()
        )

        return cls(
            aks=aks,
            participants=participants,
            rooms=rooms,
            timeslot_blocks=timeslot_blocks,
            timeslot_info=input_dict["timeslots"]["info"],
            scheduled_aks=scheduled_aks,
            config=config,
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


def get_ak_name(input_data: SchedulingInput, ak_id: int) -> str:
    """Get name string for an AK."""
    ak_names = [
        ak.info["name"]
        for ak in input_data.aks
        if ak.id == ak_id and "name" in ak.info.keys()
    ]
    return ", ".join(ak_names)


def process_pref_score(preference_score: int, required: bool, mu: float) -> float:
    """Process the input preference score for the MILP constraints.

    Args:
        preference_score (int): The input score of preference: not interested (0),
            weakly interested (1), strongly interested (2) or required (-1).
        required (bool): Whether the participant is required for the AK or not.
        mu (float): The weight associated with a strong preference for an AK.

    Returns:
        float: The processed preference score: Required AKs are weighted with 0,
        weakly preferred AKs with 1 and strongly preferred AKs with `mu`.

    Raises:
        ValueError: if `preference_score` is not in [0, 1, 2, -1].
    """
    if required or preference_score == -1:
        return 0
    elif preference_score in [0, 1]:
        return preference_score
    elif preference_score == 2:
        return mu
    else:
        raise ValueError(preference_score)


def process_room_cap(room_capacity: int, num_participants: int) -> int:
    """Process the input room capacity for the MILP constraints.

    Args:
        room_capacity (int): The input room capacity: infinite (-1) or actual capacity >=0
        num_participants (int): The total number of participants (needed to model infinity)

    Returns:
        int: The processed room capacity: Rooms with infinite capacity or capacity larger than
        num_participants are set to num_participants. Rooms with a smaller non-negative capacity
        hold their capacity.

    Raises:
        ValueError: if 'room_capacity' < -1
    """
    if room_capacity == -1:
        return num_participants
    if room_capacity >= num_participants:
        return num_participants
    if room_capacity < 0:
        raise ValueError(
            f"Invalid room capacity {room_capacity}. "
            "Room capacity must be non-negative or -1."
        )
    return room_capacity


@dataclass(frozen=True)
class ProblemIds:
    """Dataclass containing the id collections from a problem."""

    ak: set[int]
    room: set[int]
    timeslot: set[int]
    person: set[int]
    block_dict: dict[int, list[int]]
    sorted_timeslot: list[int]
    conflict_pairs: set[tuple[int, int]]

    @staticmethod
    def get_ids(
        input_data: SchedulingInput,
    ) -> tuple[set[int], set[int], set[int], set[int]]:
        """Create id sets from scheduling input."""

        def _retrieve_ids(
            input_iterable: Iterable[
                AKData | ParticipantData | RoomData | TimeSlotData
            ],
        ) -> set[int]:
            return {obj.id for obj in input_iterable}

        ak_ids = _retrieve_ids(input_data.aks)
        participant_ids = _retrieve_ids(input_data.participants)
        room_ids = _retrieve_ids(input_data.rooms)
        timeslot_ids = _retrieve_ids(chain.from_iterable(input_data.timeslot_blocks))
        return ak_ids, participant_ids, room_ids, timeslot_ids

    @classmethod
    def init_from_problem(
        cls: Type["ProblemIds"],
        input_data: SchedulingInput,
    ) -> "ProblemIds":
        """Get problem ids from the input data."""
        ak_ids, person_ids, room_ids, timeslot_ids = cls.get_ids(input_data)
        sorted_timeslot_ids = sorted(timeslot_ids)

        block_dict = {
            block_idx: sorted([timeslot.id for timeslot in block])
            for block_idx, block in enumerate(input_data.timeslot_blocks)
        }

        conflict_pairs: set[tuple[int, int]] = set()
        for ak in input_data.aks:
            conflicting_aks: list[int] = ak.properties.get("conflicts", [])
            depending_aks: list[int] = ak.properties.get("dependencies", [])
            conflict_pairs.update(
                [
                    (
                        (ak.id, other_ak_id)
                        if ak.id < other_ak_id
                        else (other_ak_id, ak.id)
                    )
                    for other_ak_id in conflicting_aks + depending_aks
                ]
            )

        return cls(
            ak=ak_ids,
            room=room_ids,
            timeslot=timeslot_ids,
            person=person_ids,
            block_dict=block_dict,
            sorted_timeslot=sorted_timeslot_ids,
            conflict_pairs=conflict_pairs,
        )


@dataclass(frozen=True)
class ProblemProperties:
    """Dataclass containing derived properties from a problem."""

    room_capacities: dict[int, int]
    ak_durations: dict[int, int]
    weighted_preferences: dict[int, dict[int, float]]
    required_persons: dict[int, set[int]]
    ak_num_interested: dict[int, int]
    participant_time_constraints: ConstraintSetDict
    participant_room_constraints: ConstraintSetDict
    ak_time_constraints: ConstraintSetDict
    ak_room_constraints: ConstraintSetDict
    room_time_constraints: ConstraintSetDict
    fulfilled_time_constraints: ConstraintSetDict
    fulfilled_room_constraints: ConstraintSetDict

    @classmethod
    def init_from_problem(
        cls: Type["ProblemProperties"],
        input_data: SchedulingInput,
        ids: ProblemIds | None = None,
    ) -> "ProblemProperties":
        """Get derived problem properties from the input data."""
        if ids is None:
            ids = ProblemIds.init_from_problem(input_data)

        # Get values needed from the input_dict
        room_capacities = {
            room.id: process_room_cap(room.capacity, len(ids.person))
            for room in input_data.rooms
        }
        ak_durations = {ak.id: ak.duration for ak in input_data.aks}

        # dict of real participants only (without dummy participants)
        # with numerical preferences
        weighted_preferences = {
            person.id: {
                pref.ak_id: process_pref_score(
                    pref.preference_score,
                    pref.required,
                    mu=input_data.config.mu,
                )
                for pref in person.preferences
            }
            for person in input_data.participants
        }
        required_persons = {
            ak_id: {
                person.id
                for person in input_data.participants
                for pref in person.preferences
                if pref.ak_id == ak_id and pref.required
            }
            for ak_id in ids.ak
        }
        for ak_id, persons in required_persons.items():
            if len(persons) == 0:
                print(
                    f"Warning: AK {get_ak_name(input_data, ak_id)} with id {ak_id} "
                    "has no required persons. Who owns this?"
                )
        ak_num_interested = {
            ak_id: len(required_persons[ak_id])
            + sum(
                1
                for person, prefs in weighted_preferences.items()
                if ak_id in prefs.keys() and prefs[ak_id] != 0
            )
            for ak_id in ids.ak
        }

        # Get constraints from input_dict
        participant_time_constraints = {
            participant.id: set(participant.time_constraints)
            for participant in input_data.participants
        }
        participant_room_constraints = {
            participant.id: set(participant.room_constraints)
            for participant in input_data.participants
        }

        ak_time_constraints = {ak.id: set(ak.time_constraints) for ak in input_data.aks}
        ak_room_constraints = {ak.id: set(ak.room_constraints) for ak in input_data.aks}

        room_time_constraints = {
            room.id: set(room.time_constraints) for room in input_data.rooms
        }

        fulfilled_time_constraints = {
            timeslot.id: set(timeslot.fulfilled_time_constraints)
            for block in input_data.timeslot_blocks
            for timeslot in block
        }
        fulfilled_room_constraints = {
            room.id: set(room.fulfilled_room_constraints) for room in input_data.rooms
        }

        return cls(
            room_capacities=room_capacities,
            ak_durations=ak_durations,
            weighted_preferences=weighted_preferences,
            required_persons=required_persons,
            ak_num_interested=ak_num_interested,
            participant_time_constraints=participant_time_constraints,
            participant_room_constraints=participant_room_constraints,
            ak_time_constraints=ak_time_constraints,
            ak_room_constraints=ak_room_constraints,
            room_time_constraints=room_time_constraints,
            fulfilled_time_constraints=fulfilled_time_constraints,
            fulfilled_room_constraints=fulfilled_room_constraints,
        )


@dataclass(frozen=True)
class LPVarDicts:
    """Dataclass containing the decision variable dicts for the LP."""

    room: VarDict
    time: VarDict
    block: VarDict
    person: VarDict
    person_time: VarDict

    def to_export_dict(self) -> dict[str, VarDict]:
        """We only export the variables for room, time, and persons."""
        return {"Room": self.room, "Time": self.time, "Part": self.person}

    @classmethod
    def init_from_ids(
        cls: Type["LPVarDicts"],
        ak_ids: Collection[int],
        room_ids: Collection[int],
        timeslot_ids: Collection[int],
        block_ids: Collection[int],
        person_ids: Collection[int],
    ) -> "LPVarDicts":
        """Initialize decision variables from the problem ids."""
        room_var: VarDict = LpVariable.dicts("Room", (ak_ids, room_ids), cat=LpBinary)
        time_var: VarDict = LpVariable.dicts(
            "Time",
            (ak_ids, timeslot_ids),
            cat=LpBinary,
        )
        block_var: VarDict = LpVariable.dicts(
            "Block", (ak_ids, block_ids), cat=LpBinary
        )
        person_var: VarDict = LpVariable.dicts(
            "Part", (ak_ids, person_ids), cat=LpBinary
        )
        person_time_var: VarDict = LpVariable.dicts(
            "Working",
            (person_ids, timeslot_ids),
            cat=LpBinary,
        )
        return cls(
            room=room_var,
            time=time_var,
            block=block_var,
            person=person_var,
            person_time=person_time_var,
        )


def _construct_constraint_name(name: str, *args: Any) -> str:
    return name + "_" + "_".join(map(str, args))
