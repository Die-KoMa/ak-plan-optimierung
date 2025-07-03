"""Type definitions."""

from pathlib import Path
from typing import NamedTuple, TypedDict, TypeVar

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


class ExportTuple(NamedTuple):
    """Named tuple containing the LP variables resp. their values."""

    room: xr.DataArray
    time: xr.DataArray
    person: xr.DataArray


class SolverKwargs(TypedDict, total=False):
    """Key word arguments to initialize the linopy solver."""

    # TODO check kwargs names
    timeLimit: NotRequired[float]  # noqa: N815
    gapRel: NotRequired[float]  # noqa: N815
    gapAbs: NotRequired[float]  # noqa: N815
    threads: NotRequired[int]
    warmstart_fn: NotRequired[str | Path | None]
