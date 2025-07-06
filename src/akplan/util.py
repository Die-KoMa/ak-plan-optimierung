"""Utility classes and functions."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Literal, Type, overload

import numpy as np
import pandas as pd
import xarray as xr
from dacite import from_dict

from . import types

logger = logging.getLogger(__name__)


solvers_supporting_direct_api = ["gurobi", "highs", "mosek"]


@dataclass(frozen=True, order=True)
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

    id: types.AkId
    duration: int
    properties: dict[str, Any]
    room_constraints: list[str]
    time_constraints: list[str]
    info: dict[str, Any]


@dataclass(frozen=True, order=True)
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

    ak_id: types.AkId
    preference_score: int
    required: bool


@dataclass(frozen=True, order=True)
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

    id: types.PersonId
    preferences: list[PreferenceData]
    room_constraints: list[str]
    time_constraints: list[str]
    info: dict[str, Any]


@dataclass(frozen=True, order=True)
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

    id: types.RoomId
    capacity: int
    fulfilled_room_constraints: list[str]
    time_constraints: list[str]
    info: dict[str, Any]


@dataclass(frozen=True, order=True)
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

    id: types.TimeslotId
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

    ak_id: types.AkId
    room_id: types.RoomId | None
    timeslot_ids: np.ndarray
    participant_ids: np.ndarray

    @property
    def _comparison_tuple(self) -> types.ScheduleAtomComparisonTuple:
        self.timeslot_ids.sort()
        self.participant_ids.sort()
        return (
            self.ak_id,
            self.room_id,
            tuple(self.timeslot_ids),
            tuple(self.participant_ids),
        )

    def __lt__(self, other: object) -> bool:
        """Determine lesser than by comparing the `_comparison_tuple`."""
        if not isinstance(other, ScheduleAtom):
            return NotImplemented
        return self._comparison_tuple < other._comparison_tuple

    def __eq__(self, other: object) -> bool:
        """Determine equality by comparing the `_comparison_tuple`."""
        if not isinstance(other, ScheduleAtom):
            return NotImplemented
        return self._comparison_tuple == other._comparison_tuple

    def __hash__(self) -> int:
        """Calculate hash by hashing the `_comparison_tuple`."""
        return hash(self._comparison_tuple)

    def stripped_copy(
        self,
        strip_room: bool = False,
        strip_timeslots: bool = False,
        strip_participants: bool = False,
    ) -> ScheduleAtom:
        """Clone this object and strip selectef fields from the copy."""
        # sort before copy to avoid double effort
        self.timeslot_ids.sort()
        self.participant_ids.sort()
        return ScheduleAtom(
            self.ak_id,
            None if strip_room else self.room_id,
            np.array([]) if strip_timeslots else self.timeslot_ids.copy(),
            np.array([]) if strip_participants else self.participant_ids.copy(),
        )


@dataclass(frozen=False)
class ConfigData:
    """Dataclass containing the config for building the ILP and solving it.

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
        scheduled_aks (list of ScheduleAtom): The schedule atoms of all fixed AKs.
        config (ConfigData): The configuration of the problem.
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
        aks = sorted(from_dict(data_class=AKData, data=ak) for ak in input_dict["aks"])
        rooms = sorted(
            from_dict(data_class=RoomData, data=room) for room in input_dict["rooms"]
        )
        participants = sorted(
            from_dict(data_class=ParticipantData, data=participant)
            for participant in input_dict["participants"]
        )
        timeslot_blocks = [
            sorted(
                from_dict(data_class=TimeSlotData, data=timeslot) for timeslot in block
            )
            for block in input_dict["timeslots"]["blocks"]
        ]
        scheduled_aks = (
            sorted(
                from_dict(data_class=ScheduleAtom, data=scheduled_ak)
                for scheduled_ak in input_dict["scheduled_aks"]
            )
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


def get_ak_name(input_data: SchedulingInput, ak_id: types.AkId) -> str:
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

    ak: pd.Index
    room: pd.Index
    timeslot: pd.Index
    person: pd.Index
    block: pd.Index
    block_dict: dict[types.BlockId, types.Block]

    @staticmethod
    def get_ids(
        input_data: SchedulingInput,
    ) -> tuple[
        list[types.AkId],
        list[types.PersonId],
        list[types.RoomId],
        list[types.TimeslotId],
    ]:
        """Create ids from scheduling input."""

        def _retrieve_ids(
            input_iterable: Iterable[
                AKData | ParticipantData | RoomData | TimeSlotData
            ],
        ) -> list[types.Id]:
            return [obj.id for obj in input_iterable]

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

        block_dict = {
            block_idx: pd.Index(
                [timeslot.id for timeslot in block], name=f"block-{block_idx}"
            )
            for block_idx, block in enumerate(input_data.timeslot_blocks)
        }

        return cls(
            ak=pd.Index(ak_ids, name="ak"),
            room=pd.Index(room_ids, name="room"),
            timeslot=pd.Index(timeslot_ids, name="timeslot"),
            person=pd.Index(person_ids, name="person"),
            block=pd.Index(block_dict.keys(), name="block"),
            block_dict=block_dict,
        )


@dataclass(frozen=True)
class ProblemProperties:
    """Dataclass containing derived properties from a problem."""

    conflict_pairs: set[tuple[types.AkId, types.AkId]]
    dependencies: dict[types.AkId, list[types.AkId]]
    time_constraint: pd.Index[str]
    room_constraint: pd.Index[str]
    room_capacities: xr.DataArray
    ak_durations: xr.DataArray
    preferences: xr.DataArray
    required_persons: xr.DataArray
    ak_num_interested: xr.DataArray
    block_mask: xr.DataArray
    participant_time_constraints: xr.DataArray
    participant_room_constraints: xr.DataArray
    ak_time_constraints: xr.DataArray
    ak_room_constraints: xr.DataArray
    room_time_constraints: xr.DataArray
    fulfilled_time_constraints: xr.DataArray
    fulfilled_room_constraints: xr.DataArray

    @classmethod
    def init_from_problem(
        cls: Type["ProblemProperties"],
        input_data: SchedulingInput,
        ids: ProblemIds | None = None,
    ) -> "ProblemProperties":
        """Get derived problem properties from the input data."""
        if ids is None:
            ids = ProblemIds.init_from_problem(input_data)

        conflict_pairs: set[tuple[types.AkId, types.AkId]] = set()
        dependencies: dict[types.AkId, list[types.AkId]] = {}
        for ak in input_data.aks:
            conflicting_aks: list[types.AkId] = ak.properties.get("conflicts", [])
            depending_aks: list[types.AkId] = ak.properties.get("dependencies", [])
            if depending_aks:
                dependencies[ak.id] = depending_aks
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

        # Get values needed from the input_dict
        room_capacities = xr.DataArray(
            data=[
                process_room_cap(room.capacity, len(ids.person))
                for room in input_data.rooms
            ],
            coords=[ids.room],
        )
        ak_durations = xr.DataArray(
            data=[ak.duration for ak in input_data.aks],
            coords=[ids.ak],
        )

        # xarray of participants with numerical preferences
        preferences = xr.DataArray(0.0, coords=[ids.ak, ids.person])
        required_persons = xr.DataArray(False, coords=[ids.ak, ids.person])
        for person in input_data.participants:
            for pref in person.preferences:
                preferences.loc[pref.ak_id, person.id] = process_pref_score(
                    pref.preference_score,
                    pref.required,
                    mu=input_data.config.mu,
                )
                if pref.required:
                    required_persons.loc[pref.ak_id, person.id] = True

        num_required_per_ak = required_persons.sum("person")
        if (num_required_per_ak == 0).any():
            for ak_id in num_required_per_ak.where(
                num_required_per_ak == 0, drop=True
            ).coords["ak"]:
                logger.warning(
                    "AK %s with id %d has no required persons. Who owns this?",
                    get_ak_name(input_data, ak_id),
                    ak_id.item(),
                )

        ak_num_interested = num_required_per_ak + (preferences != 0).sum("person")

        block_mask = xr.DataArray(data=False, coords=[ids.block, ids.timeslot])
        for block_id, block_lst in ids.block_dict.items():
            block_mask.loc[block_id, block_lst] = True

        # collect all constraint strings
        all_time_constraints = set()
        all_room_constraints = set()
        for participant in input_data.participants:
            all_time_constraints.update(participant.time_constraints)
            all_room_constraints.update(participant.room_constraints)
        for ak in input_data.aks:
            all_time_constraints.update(ak.time_constraints)
            all_room_constraints.update(ak.room_constraints)
        for room in input_data.rooms:
            all_time_constraints.update(room.time_constraints)
            all_room_constraints.update(room.fulfilled_room_constraints)
        for timeslot in chain.from_iterable(input_data.timeslot_blocks):
            all_time_constraints.update(timeslot.fulfilled_time_constraints)

        time_constraint = pd.Index(sorted(all_time_constraints), name="time_constraint")
        room_constraint = pd.Index(sorted(all_room_constraints), name="room_constraint")

        participant_time_constraints = xr.DataArray(
            False, coords=[ids.person, time_constraint]
        )
        participant_room_constraints = xr.DataArray(
            False, coords=[ids.person, room_constraint]
        )
        for person in input_data.participants:
            participant_time_constraints.loc[person.id, person.time_constraints] = True
            participant_room_constraints.loc[person.id, person.room_constraints] = True
        ak_time_constraints = xr.DataArray(False, coords=[ids.ak, time_constraint])
        ak_room_constraints = xr.DataArray(False, coords=[ids.ak, room_constraint])
        for ak in input_data.aks:
            ak_time_constraints.loc[ak.id, ak.time_constraints] = True
            ak_room_constraints.loc[ak.id, ak.room_constraints] = True

        room_time_constraints = xr.DataArray(False, coords=[ids.room, time_constraint])
        fulfilled_room_constraints = xr.DataArray(
            False, coords=[ids.room, room_constraint]
        )
        for room in input_data.rooms:
            room_time_constraints.loc[room.id, room.time_constraints] = True
            fulfilled_room_constraints.loc[room.id, room.fulfilled_room_constraints] = (
                True
            )

        fulfilled_time_constraints = xr.DataArray(
            False, coords=[ids.timeslot, time_constraint]
        )
        for timeslot in chain.from_iterable(input_data.timeslot_blocks):
            fulfilled_time_constraints.loc[
                timeslot.id, timeslot.fulfilled_time_constraints
            ] = True

        return cls(
            conflict_pairs=conflict_pairs,
            dependencies=dependencies,
            room_capacities=room_capacities,
            ak_durations=ak_durations,
            preferences=preferences,
            required_persons=required_persons,
            ak_num_interested=ak_num_interested,
            block_mask=block_mask,
            room_constraint=room_constraint,
            time_constraint=time_constraint,
            participant_time_constraints=participant_time_constraints,
            participant_room_constraints=participant_room_constraints,
            ak_time_constraints=ak_time_constraints,
            ak_room_constraints=ak_room_constraints,
            room_time_constraints=room_time_constraints,
            fulfilled_time_constraints=fulfilled_time_constraints,
            fulfilled_room_constraints=fulfilled_room_constraints,
        )


def _construct_constraint_name(name: str, *args: Any) -> str:
    return name + "_" + "_".join(map(str, args))


@dataclass(frozen=True)
class SolverConfig:
    """Dataclass storing config values for the solver to run."""

    solver_io_api: Literal["direct", "lp", "mps"] = "direct"
    solver_dir: str | None = None
    time_limit: float | None = None
    gap_abs: float | None = None
    gap_rel: float | None = None
    threads: int | None = None
    warmstart_fn: str | Path | None = None

    @property
    def _has_non_none_attr(self) -> bool:
        if (
            self.time_limit is not None
            or self.gap_abs is not None
            or self.gap_rel is not None
            or self.threads is not None
        ):
            return True
        return False

    @overload
    def generate_kwargs(self, solver_name: None) -> types.SolverKwargs: ...

    @overload
    def generate_kwargs(
        self, solver_name: Literal["gurobi"]
    ) -> types.HighsSolverKwargs: ...

    @overload
    def generate_kwargs(
        self, solver_name: Literal["highs"]
    ) -> types.HighsSolverKwargs: ...

    @overload
    def generate_kwargs(self, solver_name: str) -> types.SolverKwargs: ...

    def generate_kwargs(
        self, solver_name: str | None
    ) -> types.GurobiSolverKwargs | types.HighsSolverKwargs | types.SolverKwargs:
        """Export solver config to solver kwargs. Keys differ between solvers."""
        io_api = self.solver_io_api
        if solver_name not in solvers_supporting_direct_api:
            logger.warning(
                "Using 'direct' IO-API for solver %s is not supported. "
                "Changing IO-API to 'lp' instead.",
                solver_name,
            )
            io_api = "lp"

        if solver_name == "highs":
            highs_solver_kwargs: types.HighsSolverKwargs = {}
            if self.solver_io_api is not None:
                highs_solver_kwargs["io_api"] = io_api
            if self.time_limit is not None:
                highs_solver_kwargs["time_limit"] = self.time_limit
            if self.gap_abs is not None:
                highs_solver_kwargs["mip_abs_gap"] = self.gap_abs
            if self.gap_rel is not None:
                highs_solver_kwargs["mip_rel_gap"] = self.gap_rel
            if self.threads is not None:
                highs_solver_kwargs["threads"] = self.threads
            if self.warmstart_fn is not None:
                highs_solver_kwargs["warmstart_fn"] = self.warmstart_fn
            return highs_solver_kwargs
        elif solver_name == "gurobi":
            gurobi_solver_kwargs: types.GurobiSolverKwargs = {}
            if self.solver_io_api is not None:
                gurobi_solver_kwargs["io_api"] = io_api
            if self.time_limit is not None:
                gurobi_solver_kwargs["TimeLimit"] = self.time_limit
            if self.gap_abs is not None:
                gurobi_solver_kwargs["MIPGapAbs"] = self.gap_abs
            if self.gap_rel is not None:
                gurobi_solver_kwargs["MIPGap"] = self.gap_rel
            if self.threads is not None:
                gurobi_solver_kwargs["Threads"] = self.threads
            if self.warmstart_fn is not None:
                gurobi_solver_kwargs["warmstart_fn"] = self.warmstart_fn
            return gurobi_solver_kwargs
        else:
            if self._has_non_none_attr:
                logger.warning(
                    "Exporting CLI args to solver '%s' is not supported. "
                    "The solver is run with its default parameters.",
                    solver_name,
                )
            solver_kwargs: types.SolverKwargs = {}
            if self.solver_io_api is not None:
                solver_kwargs["io_api"] = io_api
            if self.warmstart_fn is not None:
                solver_kwargs["warmstart_fn"] = self.warmstart_fn
            return solver_kwargs


def default_num_threads() -> int:
    """Calculate the default # Threads to use as # CPUs minus 1."""
    try:
        # gives the number of CPUs that the current process is allowed to use
        # might be unequal to `os.cpu_count` which gives the # of logical CPUs
        n_available_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        # on windows / mac, os.sched_getaffinity is not available.
        n_available_cpus = os.cpu_count() or 0
    return max(n_available_cpus - 1, 1)
