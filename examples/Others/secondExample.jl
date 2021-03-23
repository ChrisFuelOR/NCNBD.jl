# Copyright (c) 2021 Christian Fuellner <christian.fuellner@kit.edu>

# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
################################################################################

using JuMP
using SDDP
using NCNBD
using Revise
using Gurobi
using GAMS
using SCIP
using Infiltrator


function exampleModel()

    model = SDDP.LinearPolicyGraph(
        stages = 1,
        lower_bound = 0.0,
        optimizer = Gurobi.Optimizer,
        sense = :Min
        #integrality_handler = SDDP.SDDiP()
    ) do subproblem, t

        # DEFINE MILP MODEL
        ########################################################################
        linearizedSubproblem = JuMP.Model()

        # construct a list to store all nonlinear functions
        nonlinearFunctionList = NCNBD.NonlinearFunction[]
        numberOfNonlinearFunctions = 2

        # SET-UP LINEAR PART OF MODEL
        ########################################################################
        for problem in [subproblem, linearizedSubproblem]
            @variable(problem, 0 <= x[i=1:2] <= 4)

            # SET-UP EXPRESSION GRAPH FOR NONLINEARITIES OF MINLP MODEL
            ####################################################################
            # for each nonlinear term, introduce an auxiliary variable
            @variable(problem, nonlinearAux[i=1:numberOfNonlinearFunctions])

            # for each nonlinear constraint, determine the expression graph
            # and replace the nonlinearity by the auxliary variable
            @constraint(problem, actual_nlcon_1, x[2] - nonlinearAux[1] <= 0)
            @constraint(problem, actual_nlcon_2, -x[2] + nonlinearAux[2] <= 0)
        end

        x = subproblem[:x]
        @stageobjective(subproblem, sum(x[i] for i in 1:2))

        x = linearizedSubproblem[:x]
        @objective(linearizedSubproblem, MOI.MIN_SENSE, sum(x[i] for i in 1:2))

        # SET-UP NONLINEARITIES
        ########################################################################
        # define nonlinear functions as user-defined functions
        # (once for evaluation, once for expression building)
        nlf_1_eval = function nonlinear_function_1_eval(y::Float64)
            return y^2
        end

        nlf_1_expr = function nonlinear_function_1_expr(y::JuMP.VariableRef)
            return :($(y)^2)
        end

        nlf_2_eval = function nonlinear_function_2_eval(y::Float64, z::Float64)
            return sqrt(y) + sqrt(z)
        end

        nlf_2_expr = function nonlinear_function_2_expr(y::JuMP.VariableRef, z::JuMP.VariableRef)
            return :(sqrt($(y)) + sqrt($(z)))
        end

        # define nonlinear expressions (once as Julia expression)
        x = subproblem[:x]
        nonlinearexp_1 = nlf_1_expr(x[2])
        nonlinearexp_2 = nlf_2_expr(x[1], x[2])

        # defining nonlinear constraints using auxiliary variables
        nonlinearAux = subproblem[:nonlinearAux]
        add_NL_constraint(subproblem, :($(nonlinearAux[1]) == $(nonlinearexp_1)))
        add_NL_constraint(subproblem, :($(nonlinearAux[2]) == $(nonlinearexp_2)))

        # construct nonlinearFunction objects for both constraints
        x = linearizedSubproblem[:x]
        nonlinearAux = linearizedSubproblem[:nonlinearAux]
        nlf_1 = NCNBD.NonlinearFunction(nlf_1_eval, nlf_1_expr, nonlinearAux[1], [x[2]])
        nlf_2 = NCNBD.NonlinearFunction(nlf_2_eval, nlf_2_expr, nonlinearAux[2], [x[1], x[2]])

        # push both nonlinearFunction objects to list
        push!(nonlinearFunctionList, nlf_1)
        push!(nonlinearFunctionList, nlf_2)

        # no access to model or node yet, so store nonlinearFunctionList
        # and the linearizedSubproblem in ext of subproblem
        # shift it to right location later
        subproblem.ext[:nlFunctions] = nonlinearFunctionList
        subproblem.ext[:linSubproblem] = linearizedSubproblem

    end

    # SET-UP PARAMETERS
    ############################################################################
    appliedSolvers = NCNBD.AppliedSolvers(Gurobi.Optimizer, Gurobi.Optimizer, GAMS.Optimizer)

    epsilon_outerLoop = 0.0
    epsilon_innerLoop = 0.0
    binaryPrecision = [0.5]
    plaPrecision = [0.5]
    sigma = [1.0]

    initialAlgoParameters = NCNBD.InitialAlgoParams(epsilon_outerLoop,
                            epsilon_innerLoop, binaryPrecision, plaPrecision, sigma)
    algoParameters = NCNBD.AlgoParams(epsilon_outerLoop, epsilon_innerLoop,
                                      binaryPrecision, sigma)

    # SET-UP NONLINEARITIES
    ############################################################################
    NCNBD.solve(model, algoParameters, initialAlgoParameters, appliedSolvers,
                iteration_limit = 100, print_level = 0)

end

exampleModel()
