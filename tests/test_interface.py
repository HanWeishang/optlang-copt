import math

import optlang_copt
from optlang_copt import Constraint, Model, Objective, Variable


def test_linear_program_and_mutation():
    x = Variable("x", lb=0, ub=10)
    y = Variable("y", lb=0, ub=10)
    constraint = Constraint(x + y, ub=12, name="capacity")
    model = Model(name="lp", variables=[x, y], constraints=[constraint])
    model.objective = Objective(3 * x + 2 * y, direction="max")

    assert model.optimize() == "optimal"
    assert math.isclose(model.objective.value, 34.0)
    assert math.isclose(x.primal, 10.0)
    assert math.isclose(y.primal, 2.0)
    assert constraint.dual is not None

    constraint.set_linear_coefficients({y: 2})
    assert model.optimize() == "optimal"
    assert math.isclose(model.objective.value, 32.0)
    assert math.isclose(y.primal, 1.0)


def test_mixed_integer_program():
    x = Variable("x", type="binary")
    y = Variable("y", type="binary")
    model = Model()
    model.add([x, y, Constraint(2 * x + y, ub=2, name="limit")])
    model.objective = Objective(3 * x + 2 * y, direction="max")

    assert model.optimize() == "optimal"
    assert math.isclose(model.objective.value, 3.0)


def test_convex_quadratic_objective():
    x = Variable("x", lb=-10, ub=10)
    model = Model(variables=[x])
    model.objective = Objective((x - 2) ** 2, direction="min")

    assert model.optimize() == "optimal"
    assert math.isclose(x.primal, 2.0, abs_tol=1e-7)
    assert math.isclose(model.objective.value, 0.0, abs_tol=1e-7)


def test_optlang_registration():
    import optlang

    assert optlang.copt_interface is optlang_copt.interface
    assert optlang.available_solvers["COPT"] is True


def test_cobra_registration_and_fba():
    import cobra
    from cobra.util.solver import solvers
    from cobra.flux_analysis import (
        flux_variability_analysis,
        pfba,
        single_reaction_deletion,
    )

    cobra.Configuration().solver = "glpk"
    assert solvers["copt"] is optlang_copt.interface
    model = cobra.Model("minimal_fba")
    metabolite = cobra.Metabolite("a_c")
    uptake = cobra.Reaction("UPTAKE")
    uptake.bounds = (0, 10)
    uptake.add_metabolites({metabolite: 1})
    biomass = cobra.Reaction("BIOMASS")
    biomass.bounds = (0, 1000)
    biomass.add_metabolites({metabolite: -1})
    model.add_reactions([uptake, biomass])
    model.objective = biomass

    model.solver = "copt"
    solution = model.optimize()
    assert solution.status == "optimal"
    assert math.isclose(solution.objective_value, 10.0)

    parsimonious = pfba(model)
    assert parsimonious.status == "optimal"
    assert math.isclose(parsimonious.fluxes["BIOMASS"], 10.0)

    variability = flux_variability_analysis(model, processes=1)
    assert math.isclose(variability.loc["BIOMASS", "minimum"], 10.0)
    assert math.isclose(variability.loc["BIOMASS", "maximum"], 10.0)

    deletion = single_reaction_deletion(model, processes=1)
    assert set(deletion["status"]) == {"optimal"}
    assert all(abs(value) < 1e-9 for value in deletion["growth"])


def test_cobra_moma_room_and_loopless_workflows():
    import cobra
    from cobra.flux_analysis import add_loopless, loopless_solution, moma, pfba, room

    cobra.Configuration().solver = "glpk"
    model = cobra.Model("branched_paths")
    a = cobra.Metabolite("a_c")
    b = cobra.Metabolite("b_c")
    c = cobra.Metabolite("c_c")

    uptake = cobra.Reaction("UPTAKE")
    uptake.bounds = (0, 10)
    uptake.add_metabolites({a: 1})
    direct = cobra.Reaction("DIRECT")
    direct.bounds = (0, 1000)
    direct.add_metabolites({a: -1, b: 1})
    detour_1 = cobra.Reaction("DETOUR_1")
    detour_1.bounds = (0, 1000)
    detour_1.add_metabolites({a: -1, c: 1})
    detour_2 = cobra.Reaction("DETOUR_2")
    detour_2.bounds = (0, 1000)
    detour_2.add_metabolites({c: -1, b: 1})
    biomass = cobra.Reaction("BIOMASS")
    biomass.bounds = (0, 1000)
    biomass.add_metabolites({b: -1})
    model.add_reactions([uptake, direct, detour_1, detour_2, biomass])
    model.objective = biomass
    optlang_copt.set_copt_solver(model)

    reference = pfba(model)
    assert reference.status == "optimal"

    with model:
        direct.knock_out()
        solution = moma(model, solution=reference, linear=True)
        assert solution.status == "optimal"

    for _ in range(3):
        with model:
            direct.knock_out()
            solution = moma(model, solution=reference, linear=False)
            assert solution.status == "optimal"
            assert model.solver.interface is optlang_copt.interface
        assert model.optimize().status == "optimal"

    for linear in (True, False):
        with model:
            direct.knock_out()
            solution = room(model, solution=reference, linear=linear)
            assert solution.status == "optimal"

    assert loopless_solution(model).status == "optimal"
    with model:
        add_loopless(model)
        assert model.optimize().status == "optimal"
