"""optlang and COBRApy integration for Cardinal Optimizer (COPT)."""

import sys

from . import interface
from .interface import Configuration, Constraint, Model, Objective, Variable

__version__ = "0.2.0"


def register_with_optlang():
    """Expose COPT using optlang's conventional ``copt_interface`` name."""
    import optlang

    optlang.copt_interface = interface
    optlang.available_solvers["COPT"] = True
    sys.modules.setdefault("optlang.copt_interface", interface)
    return interface


def register_with_cobra():
    """Register the COPT interface in COBRApy's process-local solver registry.

    Returns the interface module so it can also be assigned directly to
    ``cobra.Model.solver``.
    """
    register_with_optlang()
    try:
        from cobra.util import solver as cobra_solver
    except ImportError as exc:
        raise ImportError("COBRApy is not installed in this environment") from exc
    cobra_solver.solvers["copt"] = interface
    if "copt" not in cobra_solver.qp_solvers:
        cobra_solver.qp_solvers.append("copt")
    global COBRA_REGISTERED
    COBRA_REGISTERED = True
    return interface


def set_copt_solver(model):
    """Register COPT and select it for a COBRApy model."""
    register_with_cobra()
    model.solver = "copt"
    return model


# Register with optlang immediately. COBRApy registration remains lazy so an
# optlang-only program does not pay the cost of importing all of COBRApy.
register_with_optlang()
COBRA_REGISTERED = "cobra.util.solver" in sys.modules
if COBRA_REGISTERED:
    register_with_cobra()
else:
    # Also installs the lazy hook for source/editable installations where the
    # wheel's .pth startup hook is not present.
    import _optlang_copt_bootstrap  # noqa: F401


__all__ = [
    "Configuration",
    "COBRA_REGISTERED",
    "Constraint",
    "Model",
    "Objective",
    "Variable",
    "interface",
    "register_with_cobra",
    "register_with_optlang",
    "set_copt_solver",
]
