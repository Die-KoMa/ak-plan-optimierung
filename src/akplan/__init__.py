"""Conference scheduling using MILPs."""

from importlib.metadata import version

__version__ = version("akplan")

# Note: The linopy solver base class handles a solver termination
# by user interrupt not as an 'ok' solver state, so the solution values
# are discarded. We patch this by trying to read a solution after a
# user interrupt.
# Note: We also patch linopy's Highs solver class to make the Highs
# solver listen to keyboard interrupts.
import akplan.monkeypatch_linopy  # noqa: F401
