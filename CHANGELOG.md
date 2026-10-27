# Changelog

## 0.2.1 - 2026-10-27

- Synchronize the variable-to-constraint index after coefficient edits and
  remove LP/MIP variables without rebuilding the native model.
- Clear previous quadratic terms when replacing objectives; safely remove
  current or former QP variables during COBRApy context rollback.
- Preserve configuration and nondefault native COPT parameters, including
  memory limits and relative MIP gaps, across cloning, pickling and QP rebuilds.
- Return numeric missing values (`NaN`) for unavailable attached-model primal
  and objective values rather than exposing COPT's no-solution sentinel.
  Preserve feasible resource-limited incumbents and the actual solver status.
- Add standalone regressions for mutation, MOMA rollback, configuration copies,
  infeasible LP/MIP models and resource-limited solution availability.

## 0.2.0 - 2026-09-28

- Add lazy, installation-time discovery by optlang and COBRApy.
- Add linear and quadratic objective support, including quadratic MOMA.
- Rebuild native COPT state safely when COBRApy rolls back QP auxiliary variables.
- Support FBA, pFBA, FVA, reaction deletion, linear and integer ROOM, and loopless workflows.

## 0.1.2 - 2026-09-28

- Add process-local COBRApy registration with `model.solver = "copt"`.
- Fix pending-variable updates used by MOMA and ROOM.
