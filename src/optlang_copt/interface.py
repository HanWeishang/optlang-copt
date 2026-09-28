# Copyright 2026 optlang-copt contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LP, MILP, and convex-QP optlang interface for COPT.

The implementation follows optlang's public solver-interface contract and is
intended primarily for COBRApy workflows. Linear constraints and linear or
quadratic objectives are supported. Indicator constraints, SOS constraints,
quadratic constraints, and callbacks remain outside the compatibility scope.
"""

from __future__ import absolute_import

import coptpy
from coptpy import COPT

from optlang import interface, symbolics
from optlang.expression_parsing import parse_optimization_expression
from optlang.util import inheritdocstring


_COPT_STATUS_TO_STATUS = {
    COPT.UNSTARTED: interface.LOADED,
    COPT.OPTIMAL: interface.OPTIMAL,
    COPT.INFEASIBLE: interface.INFEASIBLE,
    COPT.UNBOUNDED: interface.UNBOUNDED,
    COPT.INF_OR_UNB: interface.INFEASIBLE_OR_UNBOUNDED,
    COPT.NUMERICAL: interface.NUMERIC,
    COPT.NODELIMIT: interface.NODE_LIMIT,
    COPT.IMPRECISE: interface.SUBOPTIMAL,
    COPT.TIMEOUT: interface.TIME_LIMIT,
    COPT.UNFINISHED: interface.ABORTED,
    COPT.INTERRUPTED: interface.INTERRUPTED,
    COPT.ITERLIMIT: interface.ITERATION_LIMIT,
}

_VTYPE_TO_COPT = {
    "continuous": COPT.CONTINUOUS,
    "integer": COPT.INTEGER,
    "binary": COPT.BINARY,
}
_COPT_TO_VTYPE = {value: key for key, value in _VTYPE_TO_COPT.items()}

# COPT currently exposes dual simplex but not a separate primal-simplex mode.
# ``primal`` is accepted for optlang compatibility and maps to dual simplex.
_LP_METHODS = {
    "auto": -1,
    "primal": 1,
    "dual": 1,
    "barrier": 2,
}


def _finite_lower(value):
    return -COPT.INFINITY if value is None else float(value)


def _finite_upper(value):
    return COPT.INFINITY if value is None else float(value)


def _optional_lower(value):
    return None if value <= -COPT.INFINITY / 2 else float(value)


def _optional_upper(value):
    return None if value >= COPT.INFINITY / 2 else float(value)


def _native_objective_to_symbolic(model, native_objective):
    """Translate a native COPT linear or quadratic objective to symbols."""
    terms = []
    if isinstance(native_objective, coptpy.QuadExpr):
        for index in range(native_objective.getSize()):
            native_var1 = native_objective.getVar1(index)
            native_var2 = native_objective.getVar2(index)
            var1 = model._variables[native_var1.name]
            var2 = model._variables[native_var2.name]
            terms.append(
                symbolics.Real(native_objective.getCoeff(index)) * var1 * var2
            )
        linear_expression = native_objective.getLinExpr()
    else:
        linear_expression = native_objective

    for index in range(linear_expression.getSize()):
        native_variable = linear_expression.getVar(index)
        variable = model._variables[native_variable.name]
        terms.append(symbolics.Real(linear_expression.getCoeff(index)) * variable)
    terms.append(symbolics.Real(linear_expression.getConstant()))
    return symbolics.add(terms)


class Variable(interface.Variable, metaclass=inheritdocstring):
    def __init__(self, name, *args, **kwargs):
        super(Variable, self).__init__(name, **kwargs)

    @property
    def _internal_variable(self):
        if getattr(self, "problem", None) is None:
            return None
        return self.problem.problem.getVarByName(self.name)

    @interface.Variable.lb.setter
    def lb(self, value):
        super(Variable, Variable).lb.fset(self, value)
        if getattr(self, "problem", None) is not None:
            self._internal_variable.lb = _finite_lower(value)

    @interface.Variable.ub.setter
    def ub(self, value):
        super(Variable, Variable).ub.fset(self, value)
        if getattr(self, "problem", None) is not None:
            self._internal_variable.ub = _finite_upper(value)

    def set_bounds(self, lb, ub):
        super(Variable, self).set_bounds(lb, ub)
        if getattr(self, "problem", None) is not None:
            internal = self._internal_variable
            internal.lb = _finite_lower(lb)
            internal.ub = _finite_upper(ub)

    @interface.Variable.type.setter
    def type(self, value):
        super(Variable, Variable).type.fset(self, value)
        if getattr(self, "problem", None) is not None:
            self._internal_variable.vtype = _VTYPE_TO_COPT[value]

    def _get_primal(self):
        if self._internal_variable is None:
            return None
        try:
            return self._internal_variable.x
        except coptpy.CoptError:
            return None

    @property
    def dual(self):
        if getattr(self, "problem", None) is None:
            return None
        if self.problem.is_integer:
            raise ValueError("Dual values are not well-defined for integer problems")
        try:
            return self._internal_variable.rc
        except coptpy.CoptError:
            return None

    @interface.Variable.name.setter
    def name(self, value):
        internal = self._internal_variable
        super(Variable, Variable).name.fset(self, value)
        if internal is not None:
            internal.name = value


class Constraint(interface.Constraint, metaclass=inheritdocstring):
    _INDICATOR_CONSTRAINT_SUPPORT = False

    def __init__(self, expression, *args, **kwargs):
        super(Constraint, self).__init__(expression, *args, **kwargs)

    @property
    def _internal_constraint(self):
        if getattr(self, "problem", None) is None:
            return None
        return self.problem.problem.getConstrByName(self.name)

    def set_linear_coefficients(self, coefficients):
        if self.problem is None:
            raise Exception("Can't change coefficients if constraint is not associated with a model.")
        # COBRApy may set coefficients immediately after queueing new
        # variables/constraints (for example in MOMA and loopless models).
        # Materialize optlang's pending objects before looking them up in COPT.
        self.problem.update()
        internal_constraint = self._internal_constraint
        for variable, coefficient in coefficients.items():
            self.problem.problem.setCoeff(
                internal_constraint, variable._internal_variable, float(coefficient)
            )
        self._get_expression()

    def get_linear_coefficients(self, variables):
        if self.problem is None:
            raise Exception("Can't get coefficients from solver if constraint is not in a model")
        internal_constraint = self._internal_constraint
        return {
            variable: self.problem.problem.getCoeff(
                internal_constraint, variable._internal_variable
            )
            for variable in variables
        }

    def _get_expression(self):
        if self.problem is not None:
            row = self.problem.problem.getRow(self._internal_constraint)
            terms = []
            for index in range(row.getSize()):
                internal_variable = row.getVar(index)
                variable = self.problem._variables[internal_variable.name]
                terms.append(symbolics.Real(row.getCoeff(index)) * variable)
            self._expression = symbolics.add(terms)
        return self._expression

    @property
    def problem(self):
        return self._problem

    @problem.setter
    def problem(self, value):
        if value is None and getattr(self, "_problem", None) is not None:
            self._get_expression()
        self._problem = value

    @property
    def primal(self):
        if self.problem is None:
            return None
        try:
            # COPT's Constraint.slack property is the row activity.
            return self._internal_constraint.slack
        except coptpy.CoptError:
            return None

    @property
    def dual(self):
        if self.problem is None:
            return None
        if self.problem.is_integer:
            raise ValueError("Dual values are not well-defined for integer problems")
        try:
            return self._internal_constraint.pi
        except coptpy.CoptError:
            return None

    @interface.Constraint.name.setter
    def name(self, value):
        internal = self._internal_constraint
        super(Constraint, Constraint).name.fset(self, value)
        if internal is not None:
            internal.name = value

    @interface.Constraint.lb.setter
    def lb(self, value):
        super(Constraint, Constraint).lb.fset(self, value)
        if getattr(self, "problem", None) is not None:
            self._internal_constraint.lb = _finite_lower(value)

    @interface.Constraint.ub.setter
    def ub(self, value):
        super(Constraint, Constraint).ub.fset(self, value)
        if getattr(self, "problem", None) is not None:
            self._internal_constraint.ub = _finite_upper(value)

    def __iadd__(self, other):
        if self.problem is None:
            super(Constraint, self).__iadd__(other)
            return self
        model = self.problem
        model._remove_constraint(self)
        super(Constraint, self).__iadd__(other)
        model._add_constraint(self, sloppy=False)
        return self


class Objective(interface.Objective, metaclass=inheritdocstring):
    def __init__(self, expression, sloppy=False, *args, **kwargs):
        super(Objective, self).__init__(expression, *args, sloppy=sloppy, **kwargs)
        self._expression_expired = False
        if not (sloppy or self.is_Linear or self.is_Quadratic):
            raise ValueError(
                "COPT optlang adapter supports linear or quadratic objectives."
            )

    @property
    def value(self):
        if getattr(self, "problem", None) is None:
            return None
        try:
            return self.problem.problem.objval
        except coptpy.CoptError:
            return None

    @interface.Objective.direction.setter
    def direction(self, value):
        super(Objective, Objective).direction.__set__(self, value)
        if getattr(self, "problem", None) is not None:
            self.problem.problem.setObjSense(
                {"min": COPT.MINIMIZE, "max": COPT.MAXIMIZE}[value]
            )

    def set_linear_coefficients(self, coefficients):
        if self.problem is None:
            raise Exception("Can't change coefficients if objective is not associated with a model.")
        # MOMA/ROOM add auxiliary variables and then set their objective
        # coefficients without an explicit model.update() in between.
        self.problem.update()
        for variable, coefficient in coefficients.items():
            variable._internal_variable.obj = float(coefficient)
        self._expression_expired = True

    def get_linear_coefficients(self, variables):
        if self.problem is None:
            raise Exception("Can't get coefficients if objective is not associated with a model.")
        return {variable: variable._internal_variable.obj for variable in variables}

    def _get_expression(self):
        if self.problem is not None and self._expression_expired:
            native_objective = self.problem.problem.getObjective()
            self._expression = _native_objective_to_symbolic(
                self.problem, native_objective
            )
            self._expression_expired = False
        return self._expression


class Configuration(interface.MathematicalProgrammingConfiguration, metaclass=inheritdocstring):
    def __init__(
        self,
        lp_method="auto",
        presolve="auto",
        verbosity=0,
        timeout=None,
        *args,
        **kwargs
    ):
        super(Configuration, self).__init__(*args, **kwargs)
        self.verbosity = verbosity
        self.lp_method = lp_method
        self.presolve = presolve
        self.timeout = timeout
        if "tolerances" in kwargs:
            for key, value in kwargs["tolerances"].items():
                try:
                    setattr(self.tolerances, key, value)
                except AttributeError:
                    pass

    @property
    def lp_method(self):
        method = self.problem.problem.getParam(COPT.Param.LpMethod)
        if method == -1:
            return "auto"
        if method == 1:
            return getattr(self, "_lp_method", "dual")
        if method == 2:
            return "barrier"
        return "auto"

    @lp_method.setter
    def lp_method(self, value):
        if value not in _LP_METHODS:
            raise ValueError("Invalid LP method: %s" % value)
        self._lp_method = value
        if getattr(self, "problem", None) is not None:
            self.problem.problem.setParam(COPT.Param.LpMethod, _LP_METHODS[value])

    @property
    def presolve(self):
        return self._presolve

    @presolve.setter
    def presolve(self, value):
        if value not in (True, False, "auto"):
            raise ValueError("presolve must be True, False, or 'auto'")
        if getattr(self, "problem", None) is not None:
            native_value = {True: 1, False: 0, "auto": -1}[value]
            self.problem.problem.setParam(COPT.Param.Presolve, native_value)
        self._presolve = value

    @property
    def verbosity(self):
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value):
        if getattr(self, "problem", None) is not None:
            self.problem.problem.setParam(COPT.Param.Logging, int(value > 0))
        self._verbosity = value

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        if getattr(self, "problem", None) is not None:
            native_value = COPT.INFINITY if value is None else float(value)
            self.problem.problem.setParam(COPT.Param.TimeLimit, native_value)
        self._timeout = value

    def _get_feasibility(self):
        return self.problem.problem.getParam(COPT.Param.FeasTol)

    def _set_feasibility(self, value):
        self.problem.problem.setParam(COPT.Param.FeasTol, value)

    def _get_optimality(self):
        return self.problem.problem.getParam(COPT.Param.DualTol)

    def _set_optimality(self, value):
        self.problem.problem.setParam(COPT.Param.DualTol, value)

    def _get_integrality(self):
        return self.problem.problem.getParam(COPT.Param.IntTol)

    def _set_integrality(self, value):
        self.problem.problem.setParam(COPT.Param.IntTol, value)

    def _tolerance_functions(self):
        return {
            "feasibility": (self._get_feasibility, self._set_feasibility),
            "optimality": (self._get_optimality, self._set_optimality),
            "integrality": (self._get_integrality, self._set_integrality),
        }

    def __getstate__(self):
        return {
            "presolve": self.presolve,
            "timeout": self.timeout,
            "verbosity": self.verbosity,
            "lp_method": self.lp_method,
            "tolerances": {
                "feasibility": self.tolerances.feasibility,
                "optimality": self.tolerances.optimality,
                "integrality": self.tolerances.integrality,
            },
        }

    def __setstate__(self, state):
        for key, value in state.items():
            if key != "tolerances":
                setattr(self, key, value)
        for key, value in state.get("tolerances", {}).items():
            setattr(self.tolerances, key, value)


class Model(interface.Model):
    def _initialize_problem(self):
        self._environment = coptpy.Envr()
        self.problem = self._environment.createModel(self.name or "optlang-copt")
        self.problem.setParam(COPT.Param.Logging, 0)

    def _initialize_model_from_problem(self, problem):
        if not isinstance(problem, coptpy.Model):
            raise TypeError("Provided problem is not a valid COPT model.")
        self.problem = problem
        self._environment = None
        variables = []
        for native in self.problem.getVars():
            variables.append(
                Variable(
                    native.name,
                    lb=_optional_lower(native.lb),
                    ub=_optional_upper(native.ub),
                    type=_COPT_TO_VTYPE[native.vtype],
                    problem=self,
                )
            )
        super(Model, self)._add_variables(variables)

        constraints = []
        for native in self.problem.getConstrs():
            row = self.problem.getRow(native)
            terms = []
            for index in range(row.getSize()):
                native_variable = row.getVar(index)
                terms.append(
                    symbolics.Real(row.getCoeff(index))
                    * self._variables[native_variable.name]
                )
            constraints.append(
                Constraint(
                    symbolics.add(terms),
                    name=native.name,
                    lb=_optional_lower(native.lb),
                    ub=_optional_upper(native.ub),
                    problem=self,
                )
            )
        super(Model, self)._add_constraints(constraints, sloppy=True)

        native_objective = self.problem.getObjective()
        self._objective = Objective(
            _native_objective_to_symbolic(self, native_objective),
            problem=self,
            direction={COPT.MINIMIZE: "min", COPT.MAXIMIZE: "max"}[
                self.problem.objsense
            ],
        )

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, value):
        super(Model, Model).objective.fset(self, value)
        self._set_native_objective(value.expression, value.direction)
        value.problem = self

    def _set_native_objective(self, expression, direction):
        """Set a COPT linear or quadratic objective from an optlang expression."""
        offset, linear_coefficients, quadratic_coefficients = parse_optimization_expression(
            self.objective, quadratic=True, expression=expression
        )
        terms = [
            float(coefficient) * variable._internal_variable
            for variable, coefficient in linear_coefficients.items()
        ]
        for variables, coefficient in quadratic_coefficients.items():
            native_variables = [
                variable._internal_variable for variable in variables
            ]
            if len(native_variables) == 1:
                native_variables.append(native_variables[0])
            terms.append(
                float(coefficient) * native_variables[0] * native_variables[1]
            )
        native_expression = coptpy.quicksum(terms) + float(offset)
        self.problem.setObjective(
            native_expression,
            {"min": COPT.MINIMIZE, "max": COPT.MAXIMIZE}[direction],
        )

    def _rebuild_native_problem(self):
        """Recreate COPT state from optlang state without dangling QP terms.

        COPT 8.0.6 may retain internal quadratic-objective references when a
        variable is removed. COBRApy context rollback removes the auxiliary
        variables created by quadratic MOMA, so rebuilding is safer than
        mutating that native QP model in place.
        """
        configuration_state = self.configuration.__getstate__()
        constraint_expressions = {
            constraint.name: constraint._expression for constraint in self._constraints
        }
        objective_expression = self.objective._expression
        objective_direction = self.objective.direction

        new_environment = coptpy.Envr()
        new_problem = new_environment.createModel(self.name or "optlang-copt")
        new_problem.setParam(COPT.Param.Logging, 0)
        self._environment = new_environment
        self.problem = new_problem

        for variable in self._variables:
            self.problem.addVar(
                lb=_finite_lower(variable.lb),
                ub=_finite_upper(variable.ub),
                vtype=_VTYPE_TO_COPT[variable.type],
                name=variable.name,
            )
        self.problem.update()

        for constraint in self._constraints:
            expression = constraint_expressions[constraint.name]
            offset, coefficients, quadratic_coefficients = parse_optimization_expression(
                constraint, linear=True, expression=expression
            )
            if quadratic_coefficients:
                raise ValueError("COPT adapter supports linear constraints only.")
            native_expression = coptpy.quicksum(
                [
                    float(coefficient) * variable._internal_variable
                    for variable, coefficient in coefficients.items()
                ]
            )
            lb = _finite_lower(constraint.lb)
            ub = _finite_upper(constraint.ub)
            if offset:
                lb -= float(offset)
                ub -= float(offset)
            self.problem.addBoundConstr(
                native_expression, lb=lb, ub=ub, name=constraint.name
            )
        self.problem.update()
        self._set_native_objective(objective_expression, objective_direction)
        self.configuration.__setstate__(configuration_state)

    def update(self):
        super(Model, self).update(callback=self.problem.update)

    def _optimize(self):
        self.problem.solve()
        return _COPT_STATUS_TO_STATUS.get(self.problem.status, interface.SPECIAL)

    def _set_variable_bounds_on_problem(self, var_lb, var_ub):
        for variable, value in var_lb:
            variable._internal_variable.lb = _finite_lower(value)
        for variable, value in var_ub:
            variable._internal_variable.ub = _finite_upper(value)

    def _add_variables(self, variables):
        super(Model, self)._add_variables(variables)
        for variable in variables:
            self.problem.addVar(
                lb=_finite_lower(variable.lb),
                ub=_finite_upper(variable.ub),
                vtype=_VTYPE_TO_COPT[variable.type],
                name=variable.name,
            )
        self.problem.update()

    def _remove_variables(self, variables):
        # Synchronize symbolic expressions before detaching variables. This is
        # important when coefficients were changed directly on the COPT model.
        self.objective._expression = self.objective.expression
        self.objective._expression_expired = False
        for constraint in self._constraints:
            constraint._expression = constraint.expression
        super(Model, self)._remove_variables(variables)
        self._rebuild_native_problem()

    def _add_constraints(self, constraints, sloppy=False):
        super(Model, self)._add_constraints(constraints, sloppy=sloppy)
        for constraint in constraints:
            # The optlang base class binds the object to this model before the
            # native COPT row exists. Temporarily detach it while parsing the
            # symbolic expression, then attach it after addBoundConstr.
            constraint._problem = None
            if constraint.lb is None and constraint.ub is None:
                raise ValueError("optlang does not support free constraints in the COPT interface.")
            if not constraint.is_Linear:
                raise ValueError("COPT optlang adapter currently supports linear constraints only.")
            offset, coefficients, _ = parse_optimization_expression(
                constraint, linear=True
            )
            native_expression = coptpy.quicksum(
                [
                    float(coefficient) * variable._internal_variable
                    for variable, coefficient in coefficients.items()
                ]
            )
            lb = _finite_lower(constraint.lb)
            ub = _finite_upper(constraint.ub)
            if offset:
                lb -= float(offset)
                ub -= float(offset)
            self.problem.addBoundConstr(
                native_expression, lb=lb, ub=ub, name=constraint.name
            )
            constraint.problem = self
        self.problem.update()

    def _remove_constraints(self, constraints):
        internal_constraints = [
            self.problem.getConstrByName(constraint.name) for constraint in constraints
        ]
        super(Model, self)._remove_constraints(constraints)
        self.problem.remove(internal_constraints)
        self.problem.update()

    def _get_variables_names(self):
        return [variable.name for variable in self.problem.getVars()]

    def _get_constraint_names(self):
        return [constraint.name for constraint in self.problem.getConstrs()]

    def _get_primal_values(self):
        try:
            return list(self.problem.getValues())
        except coptpy.CoptError:
            return [None] * len(self.variables)

    def _get_reduced_costs(self):
        if self.is_integer:
            raise ValueError("Reduced costs are not well-defined for integer problems.")
        return list(self.problem.getRedcosts())

    def _get_shadow_prices(self):
        if self.is_integer:
            raise ValueError("Shadow prices are not well-defined for integer problems.")
        return list(self.problem.getDuals())

    @property
    def is_integer(self):
        return bool(self.problem.ismip)

    @property
    def is_mip(self):
        return bool(self.problem.ismip)
