"""Native adapter regressions; no application-side monkey patches are used."""

from copy import deepcopy
import math
import pickle
from types import SimpleNamespace

from coptpy import COPT
import pytest

import optlang_copt
from optlang_copt import Constraint, Model, Objective, Variable


@pytest.mark.parametrize("variable_type", ["continuous", "binary"])
def test_linear_removal_keeps_edited_rows_and_native_limits(variable_type):
    kept = Variable("kept", lb=0, ub=10)
    removed = Variable("removed", lb=0, ub=1, type=variable_type)
    model = Model(variables=[kept, removed])
    row = Constraint(kept, ub=4, name="capacity")
    model.add(row)
    model.objective = Objective(kept, direction="max")
    row.set_linear_coefficients({kept: 2, removed: 3})
    model.objective.set_linear_coefficients({kept: 2, removed: 5})
    native = model.problem
    native.setParam("RelGap", 0.002)
    native.setParam("MemLimit", 2048)
    assert model._variables_to_constraints_mapping["removed"] == {"capacity"}
    assert model.optimize() == "optimal"
    assert model.objective.value == pytest.approx(6)

    model.remove(removed)
    model.update()

    assert model.problem is native
    assert native.getParam("RelGap") == pytest.approx(0.002)
    assert native.getParam("MemLimit") == 2048
    assert list(model.variables) == [kept]
    assert row.expression.free_symbols == {kept}
    assert model.objective.expression.free_symbols == {kept}
    assert model.optimize() == "optimal"
    assert kept.primal == pytest.approx(2)
    assert model.objective.value == pytest.approx(4)


def test_coefficient_removal_updates_constraint_index():
    x, y = Variable("x"), Variable("y")
    model = Model(variables=[x, y])
    row = Constraint(x, ub=4, name="capacity")
    model.add(row)
    model.update()
    row.set_linear_coefficients({y: 3})
    assert model._variables_to_constraints_mapping["y"] == {"capacity"}
    row.set_linear_coefficients({y: 0})
    assert model._variables_to_constraints_mapping["y"] == set()
    assert y not in row.expression.free_symbols


def test_replacing_quadratic_objective_clears_previous_terms():
    x = Variable("x", lb=-10, ub=10)
    model = Model(variables=[x])
    model.objective = Objective((x - 2) ** 2, direction="min")
    assert model.optimize() == "optimal"
    assert x.primal == pytest.approx(2)

    model.objective = Objective((x + 3) ** 2, direction="min")
    assert model.optimize() == "optimal"
    assert x.primal == pytest.approx(-3)

    model.objective = Objective(x, direction="max")
    assert not model.problem.HasQObj
    assert model.optimize() == "optimal"
    assert x.primal == pytest.approx(10)
    assert model.objective.value == pytest.approx(10)


def configured_model():
    x = Variable("x", lb=0, ub=4)
    model = Model(variables=[x])
    model.objective = Objective(x, direction="max")
    model.configuration.timeout = 12
    model.configuration.presolve = False
    model.configuration.lp_method = "dual"
    for name in ("feasibility", "integrality", "optimality"):
        setattr(model.configuration.tolerances, name, 1e-9)
    model.problem.setParam("MemLimit", 2048)
    model.problem.setParam("RelGap", 0.002)
    return model


def assert_configuration(model):
    assert model.configuration.timeout == 12
    assert model.configuration.presolve is False
    assert model.configuration.lp_method == "dual"
    for name in ("feasibility", "integrality", "optimality"):
        assert getattr(model.configuration.tolerances, name) == 1e-9
    assert model.problem.getParam("MemLimit") == 2048
    assert model.problem.getParam("RelGap") == pytest.approx(0.002)


@pytest.mark.parametrize("copy_mode", ["deepcopy", "pickle", "clone_json", "clone_objects"])
def test_copy_preserves_configuration_and_native_parameters(copy_mode):
    model = configured_model()
    if copy_mode == "deepcopy":
        copied = deepcopy(model)
    elif copy_mode == "pickle":
        copied = pickle.loads(pickle.dumps(model))
    else:
        copied = Model.clone(model, use_json=copy_mode == "clone_json")
    assert copied.problem is not model.problem
    assert_configuration(copied)
    assert copied.optimize() == "optimal"
    assert copied.objective.value == pytest.approx(4)


@pytest.mark.parametrize("restore_linear", [False, True])
def test_qp_removal_substitutes_native_row_edits_and_keeps_limits(restore_linear):
    model = configured_model()
    x = model.variables.x
    removed = Variable("removed", lb=0, ub=1)
    model.add(removed)
    row = Constraint(x, ub=4, name="capacity")
    model.add(row)
    model.update()
    # Reproduce an edit bypassing even the adapter's coefficient setter.
    model.problem.setCoeff(row._internal_constraint, removed._internal_variable, 3)
    model.objective = Objective((x - 2) ** 2 + removed ** 2, direction="min")
    assert model.optimize() == "optimal"
    if restore_linear:
        model.objective = Objective(x, direction="max")
        assert not model.problem.HasQObj
    native = model.problem

    model.remove(removed)
    model.update()

    assert model.problem is not native
    assert row.expression.free_symbols == {x}
    assert "removed" not in model.variables
    assert_configuration(model)
    assert model.optimize() == "optimal"
    assert x.primal == pytest.approx(4 if restore_linear else 2)


@pytest.mark.parametrize("variable_type", ["continuous", "binary"])
def test_infeasible_values_are_missing_and_status_is_preserved(variable_type):
    x = Variable("x", lb=0, ub=1, type=variable_type)
    row = Constraint(x, ub=0, name="impossible")
    model = Model(variables=[x], constraints=[row, Constraint(x, lb=1, name="required")])
    model.objective = Objective(x, direction="max")
    model.configuration.presolve = True
    assert model.optimize() == "infeasible"
    assert math.isnan(model.objective.value)
    assert math.isnan(x.primal)
    assert all(math.isnan(value) for value in model.primal_values.values())
    assert math.isnan(row.primal)
    assert model.status == "infeasible"


@pytest.mark.parametrize("integer", [False, True])
@pytest.mark.parametrize("has_solution", [False, True])
def test_time_limit_preserves_only_available_incumbents(integer, has_solution):
    model = Model(variables=[Variable("x", lb=0, ub=1)])
    model.objective = Objective(model.variables.x)
    model.update()
    model._status = "time_limit"
    model.problem = SimpleNamespace(
        status=COPT.TIMEOUT, ismip=integer,
        hasmipsol=has_solution if integer else False,
        haslpsol=has_solution if not integer else False,
        objval=0.5 if has_solution else 1e30,
        getValues=lambda: [0.5] if has_solution else [1e30],
        update=lambda: None,
    )
    if has_solution:
        assert model.objective.value == pytest.approx(0.5)
        assert model._get_primal_values() == [0.5]
    else:
        assert math.isnan(model.objective.value)
        assert math.isnan(model._get_primal_values()[0])
    assert model.status == "time_limit"


@pytest.mark.parametrize("native_status", [COPT.INFEASIBLE, COPT.UNBOUNDED, COPT.INF_OR_UNB])
def test_invalid_status_does_not_expose_stale_solution(native_status):
    model = Model()
    model.problem = SimpleNamespace(status=native_status, ismip=False, haslpsol=True)
    assert not model._has_primal_solution()
    assert math.isnan(model.objective.value)


def test_cobra_configuration_copy_and_infeasible_draft():
    cobra = pytest.importorskip("cobra")
    optlang_copt.register_with_cobra()
    model = cobra.Model("infeasible_draft")
    model.solver = "copt"
    metabolite = cobra.Metabolite("a_c", compartment="c")
    production = cobra.Reaction("PRODUCTION", lower_bound=1, upper_bound=1)
    production.add_metabolites({metabolite: 1})
    model.add_reactions([production])
    model.objective = production
    model.tolerance = 1e-9
    model.solver.configuration.timeout = 12
    model.solver.configuration.presolve = False
    model.solver.configuration.lp_method = "dual"
    model.solver.problem.setParam("MemLimit", 2048)
    model.solver.problem.setParam("RelGap", 0.002)
    copied = model.copy()
    assert copied.tolerance == 1e-9
    assert_configuration(copied.solver)
    with pytest.warns(UserWarning, match="infeasible"):
        solution = copied.optimize()
    assert solution.status == "infeasible"
    assert math.isnan(solution.objective_value)
    assert solution.fluxes.isna().all()
    with pytest.raises(cobra.exceptions.OptimizationError, match="infeasible"):
        copied.optimize(raise_error=True)


def test_cobra_qp_to_lp_context_rollback():
    cobra = pytest.importorskip("cobra")
    optlang_copt.register_with_cobra()
    model = cobra.Model("quadratic_rollback")
    model.solver = "copt"
    a, b = cobra.Metabolite("a_c"), cobra.Metabolite("b_c")
    uptake = cobra.Reaction("UPTAKE", upper_bound=10)
    uptake.add_metabolites({a: 1})
    conversion = cobra.Reaction("CONVERT")
    conversion.add_metabolites({a: -1, b: 1})
    biomass = cobra.Reaction("BIOMASS")
    biomass.add_metabolites({b: -1})
    model.add_reactions([uptake, conversion, biomass])
    model.objective = biomass
    reference = model.optimize()
    assert reference.objective_value == pytest.approx(10)
    original_variables = set(model.solver.variables.keys())
    original_constraints = set(model.solver.constraints.keys())
    model.solver.problem.setParam("MemLimit", 2048)

    with model:
        conversion.upper_bound = 5
        cobra.flux_analysis.add_moma(model, solution=reference, linear=False)
        adjusted = model.optimize()
        assert adjusted.objective_value == pytest.approx(75)
        assert adjusted.fluxes["BIOMASS"] == pytest.approx(5)
        model.objective = Objective(biomass.flux_expression, direction="max")
        assert not model.solver.problem.HasQObj
        assert model.slim_optimize() == pytest.approx(5)

    assert not model.solver.problem.HasQObj
    assert set(model.solver.variables.keys()) == original_variables
    assert set(model.solver.constraints.keys()) == original_constraints
    assert model.solver.problem.getParam("MemLimit") == 2048
    assert model.slim_optimize() == pytest.approx(10)
