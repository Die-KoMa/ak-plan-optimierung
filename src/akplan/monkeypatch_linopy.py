"""Monkey patches to the linopy library.

This patches the linopy library in two ways:
1. After a user interrupt, we attempt to read a solution
   and if we suceed, we consider the solution status to be ok.
2. We make the HiGHS solver listen to keyboard interrupts.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partialmethod, update_wrapper
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from linopy.constants import Solution, SolverStatus, TerminationCondition
from linopy.solvers import Highs, Solver

if TYPE_CHECKING:
    import highspy
    from linopy.constants import Result, Status
    from linopy.model import Model


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
            logger.error("Failed to parse solution: %s", e)
            return Solution()

    return unpatched_method(solver, status, func)


class _SolverFunc(Protocol):
    """Protocol to type hint the Highs._solve method."""

    def __call__(
        self,
        solver: Highs,
        h: highspy.Highs,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        model: Model | None = None,
        io_api: str | None = None,
        sense: str | None = None,
    ) -> Result: ...


@monkey_patch(Highs, pass_unpatched_method=True)
def _solve(
    solver: Highs,
    h: highspy.Highs,
    solution_fn: Path | None = None,
    log_fn: Path | None = None,
    warmstart_fn: Path | None = None,
    basis_fn: Path | None = None,
    model: Model | None = None,
    io_api: str | None = None,
    sense: str | None = None,
    *,
    unpatched_method: _SolverFunc,
) -> Result:
    """Patch `_solve` to make HiGHS listen for keyboard interrupts."""
    # patch to use the highspy Model's solve function as the entry point
    # `run` falls back to the C version, which skips the python code
    # which allows to listen for keyboard interrupts.
    h.run = h.solve  # type: ignore[method-assign]
    h.HandleKeyboardInterrupt = True
    return unpatched_method(
        solver,
        h,
        solution_fn=solution_fn,
        log_fn=log_fn,
        warmstart_fn=warmstart_fn,
        basis_fn=basis_fn,
        model=model,
        io_api=io_api,
        sense=sense,
    )
