"""Monkey patches to the linopy library."""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partialmethod, update_wrapper
from typing import TYPE_CHECKING

from linopy.constants import Solution, SolverStatus, TerminationCondition
from linopy.solvers import Solver

if TYPE_CHECKING:
    from linopy.constants import Status


logger = logging.getLogger(__name__)


def monkey_patch(
    cls: type[Solver],
    pass_unpatched_method: bool = False,
) -> Callable:  # type: ignore[type-arg]
    """Decorate to monkey patch solver methods.

    Adapted from `linopy.monkey_patch_xarray`.

    Args:
        cls (type[Solver]): The linopy solver class to patch.
        pass_unpatched_method (bool): Whether to pass the original method
            as an kwarg to the patched method.

    Returns:
        Callable: A decorator which expects the input func to have the name
        of the method to replace. Expects the patch func to have the same
        signature as the method to patch; if `pass_unpatched_method` is True,
        we further assume the keyword argument `unpatched_method`.
    """

    def deco(func: Callable) -> Callable:  # type: ignore[type-arg]
        func_name = func.__name__
        wrapped = getattr(cls, func_name)
        update_wrapper(func, wrapped)
        if pass_unpatched_method:
            func = partialmethod(func, unpatched_method=wrapped)  # type: ignore
        setattr(cls, func_name, func)
        return func

    return deco


@monkey_patch(Solver, pass_unpatched_method=True)  # type: ignore[type-abstract]
def safe_get_solution(
    solver: Solver,
    status: Status,
    func: Callable[[], Solution],
    unpatched_method: Callable[[Solver, Status, Callable[[], Solution]], Solution],
) -> Solution:
    """Patch `safe_get_solution` to read solution after user interrupt."""
    if (
        not status.is_ok
        and status.termination_condition == TerminationCondition.user_interrupt
    ):
        try:
            logger.warning("Termination status user aborted. Trying to parse solution.")
            sol = func()
            status.status = SolverStatus.ok
            logger.warning("Solution parsed successfully.")
            return sol
        except Exception as e:
            logger.error(f"Failed to parse solution: {e}")
            return Solution()

    return unpatched_method(solver, status, func)
