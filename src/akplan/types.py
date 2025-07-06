"""Type definitions."""

from pathlib import Path
from typing import Literal, NamedTuple, TypedDict, TypeVar

import pandas as pd
import xarray as xr
from typing_extensions import NotRequired

Id = int
T = TypeVar("T")
IdType = TypeVar("IdType", bound=Id)
IdType2 = TypeVar("IdType2", bound=Id)

RoomId = Id
PersonId = Id
AkId = Id
TimeslotId = Id
BlockId = Id
Block = pd.Index

ScheduleAtomComparisonTuple = tuple[
    AkId,
    RoomId | None,
    tuple[TimeslotId, ...],
    tuple[PersonId, ...],
]


class ExportTuple(NamedTuple):
    """Named tuple containing the LP variables resp. their values."""

    room: xr.DataArray
    time: xr.DataArray
    person: xr.DataArray


SupportedSolver = Literal["gurobi", "highs"]


class SolverKwargs(TypedDict, total=False):
    """Key word arguments to initialize an unsupported linopy solver."""

    warmstart_fn: NotRequired[str | Path | None]
    io_api: Literal["direct", "lp", "mps"]


class GurobiSolverKwargs(SolverKwargs):
    """Key word arguments to initialize the Gurobi linopy solver."""

    TimeLimit: NotRequired[float]  # noqa: N815
    MIPGap: NotRequired[float]  # noqa: N815
    MIPGapAbs: NotRequired[float]  # noqa: N815
    Threads: NotRequired[int]  # noqa: N815


class HighsSolverKwargs(SolverKwargs):
    """Key word arguments to initialize the HiGHS linopy solver."""

    time_limit: NotRequired[float]
    mip_rel_gap: NotRequired[float]
    mip_abs_gap: NotRequired[float]
    threads: NotRequired[int]
