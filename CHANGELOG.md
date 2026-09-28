# Changelog

## 0.2.0 - 2026-09-28

- Add lazy, installation-time discovery by optlang and COBRApy.
- Add linear and quadratic objective support, including quadratic MOMA.
- Rebuild native COPT state safely when COBRApy rolls back QP auxiliary variables.
- Support FBA, pFBA, FVA, reaction deletion, linear and integer ROOM, and loopless workflows.

## 0.1.2 - 2026-09-28

- Add process-local COBRApy registration with `model.solver = "copt"`.
- Fix pending-variable updates used by MOMA and ROOM.
