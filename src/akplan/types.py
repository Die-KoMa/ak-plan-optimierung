"""Type definitions."""

from typing import Generic, NamedTuple, TypeAlias, TypeVar

from pulp import LpVariable

Id = int
T = TypeVar("T")
IdType = TypeVar("IdType", bound=Id)
IdType2 = TypeVar("IdType2", bound=Id)

RoomId = Id
PersonId = Id
AkId = Id
TimeslotId = Id
BlockId = Id
Block = list[TimeslotId]

ConstraintSetDict = dict[IdType, set[str]]
"""Dictionary containing the constraint strings per object id."""

# different values to store for the different solving stages
Var: TypeAlias = LpVariable
PartialSolved = int | None
Solved = int

# nested dictionaries containing either the LP variables (`VarDict`)
# or the values assigned to them by the solver (`PartialSolvedVarDict`, `SolvedVarDict`)
GenericVarDict = dict[IdType, dict[IdType2, T]]
VarDict = GenericVarDict[IdType, IdType2, Var]
PartialSolvedVarDict = GenericVarDict[IdType, IdType2, PartialSolved]
SolvedVarDict = GenericVarDict[IdType, IdType2, Solved]


class ExportTuple(NamedTuple, Generic[T]):
    """Named tuple containing the LP variables resp. their values."""

    room: GenericVarDict[AkId, RoomId, T]
    time: GenericVarDict[AkId, TimeslotId, T]
    person: GenericVarDict[AkId, PersonId, T]
