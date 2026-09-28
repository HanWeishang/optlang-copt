# optlang-copt

`optlang-copt` is a standalone adapter that makes Cardinal Optimizer (COPT)
available through optlang and COBRApy without modifying either installed
package.

## Supported scope

- Linear programming (LP)
- Mixed-integer linear programming (MILP)
- Linear and convex quadratic objectives (QP/MIQP)
- Variable and constraint addition/removal
- Bound, objective-coefficient, and constraint-coefficient changes
- Primal values, objective values, LP dual values, and reduced costs
- COBRApy FBA, pFBA, FVA, and common reaction-bound/knockout workflows
- COBRApy linear and quadratic MOMA
- COBRApy linear/integer ROOM and loopless linear/MILP workflows
- Python 3.7+ with optlang 1.5.2+ and COPT 8.x

Quadratic constraints (QCP/MIQCP), indicator/SOS constraints, callbacks,
native COPT model serialization, and advanced basis operations are not part of
version 0.2.0.

## Installation

COPT must be licensed on the machine. Install the adapter and its Python
dependencies from PyPI:

```bash
python -m pip install optlang-copt
```

When `coptpy` was installed from the official offline COPT package, use:

```bash
python -m pip install --no-deps optlang-copt
```

## Automatic discovery and COBRApy usage

The wheel installs a Python startup hook scoped to the current environment.
It registers COPT with optlang and COBRApy automatically, so application code
does not need `import optlang_copt`:

```python
import cobra

model = cobra.io.load_model("textbook")
model.solver = "copt"

solution = model.optimize()
print(solution.status, solution.objective_value)
```

To make new COBRApy models use COPT by default in the current process:

```python
import cobra

cobra.Configuration().solver = "copt"
```

This registration does not edit COBRApy files and does not remove Gurobi,
GLPK, or other interfaces. Set `OPTLANG_COPT_AUTO_REGISTER=0` before starting
Python to disable the startup hook. Explicit `import optlang_copt` remains
available when auto-registration is disabled.

### MOMA

Existing COBRApy calls remain unchanged after selecting COPT:

```python
from cobra.flux_analysis import moma, pfba

model.solver = "copt"
reference = pfba(model)

with model:
    model.reactions.PGI.knock_out()
    linear_result = moma(model, solution=reference, linear=True)

with model:
    model.reactions.PGI.knock_out()
    quadratic_result = moma(model, solution=reference, linear=False)
```

Version 0.2.0 rebuilds the native COPT model when COBRApy removes QP
auxiliary variables during context rollback. This avoids stale native
quadratic references in COPT 8.0.6 and allows subsequent FBA, ROOM, or
loopless calculations to continue on the same model.

## Direct optlang usage

```python
import optlang

copt = optlang.copt_interface

x = copt.Variable("x", lb=0, ub=10)
y = copt.Variable("y", lb=0, ub=10)
model = copt.Model()
model.add([x, y, copt.Constraint(x + y, ub=12, name="capacity")])
model.objective = copt.Objective(3*x + 2*y, direction="max")
print(model.optimize(), model.objective.value)
```

## Development test

```bash
pytest -q
```

This project is an independent compatibility layer and is not an official
component of Cardinal Operations, optlang, or COBRApy.
