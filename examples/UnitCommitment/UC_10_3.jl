# Copyright (c) 2021 Christian Fuellner <christian.fuellner@kit.edu>

# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
################################################################################

"""
Unit commitment problem with 10 stages and 3 generators
"""

module UC_10_3

export unitCommitment
export unitCommitment_with_parameters

using JuMP
using SDDP
using NCNBD
using Revise
#using Gurobi
using GAMS
#using SCIP
using Infiltrator


struct Generator
    comm_ini::Int
    gen_ini::Float64
    pmax::Float64
    pmin::Float64
    fuel_cost::Float64
    om_cost::Float64
    su_cost::Float64
    sd_cost::Float64
    ramp_up::Float64
    ramp_dw::Float64
    a::Float64
    b::Float64
    c::Float64
end


function unitCommitment()

    # define required tolerances
    epsilon_outerLoop = 1e-2
    epsilon_innerLoop = 1e-2
    lagrangian_atol = 1e-4
    lagrangian_rtol = 1e-4

    # define time and iteration limits
    lagrangian_iteration_limit = 10000
    iteration_limit = 1000
    time_limit = 10800

    # define sigma
    sigma = [0.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0]
    sigma_factor = 2.0

    # define initial approximations
    plaPrecision = [[0.226], [0.564], [0.646]] # apart from one generator always 1/5 of pmax
    binaryPrecisionFactor = 1/7

    # define infiltration level
    infiltrate_state = :none
    # alternatives: :none, :all, :outer, :sigma, :inner, :lagrange, :bellman

    # define regime for initializing duals for Lagrangian relaxation
    dual_initialization_regime = :zeros
    # alternatives: :zeros, :gurobi_relax, :cplex_relax, :cplex_fixed, :cplex_combi

    # define solution method for lagrangian dual
    lagrangian_method = :bundle_level
    # alternatives: :kelley, :bundle_proximal, :bundle_level

    bundle_alpha = 0.5
    bundle_factor = 1.0
    level_factor = 0.2

    # cut selection strategy
    cut_selection = true

    # lagrangian status
    lag_status_regime = :lax
    # alternatives: :rigorous, :lax

    # outer loop strategy
    outer_loop_strategy = :approx

    # used solvers
    solvers = ["CPLEX", "CPLEX", "Baron", "SCIP", "CPLEX"]

    # CALL METHOD WITH PARAMETERS
    ############################################################################
    unitCommitment_with_parameters(
        epsilon_outerLoop=epsilon_outerLoop,
        epsilon_innerLoop=epsilon_innerLoop,
        lagrangian_atol=lagrangian_atol,
        lagrangian_rtol=lagrangian_rtol,
        lagrangian_iteration_limit=lagrangian_iteration_limit,
        iteration_limit=iteration_limit,
        time_limit=time_limit,
        sigma=sigma,
        sigma_factor=sigma_factor,
        plaPrecision=plaPrecision,
        binaryPrecisionFactor=binaryPrecisionFactor,
        infiltrate_state=infiltrate_state,
        dual_initialization_regime=dual_initialization_regime,
        lagrangian_method=lagrangian_method,
        bundle_alpha=bundle_alpha,
        bundle_factor=bundle_factor,
        level_factor=level_factor,
        solvers=solvers,
        cut_selection=cut_selection,
        lag_status_regime=lag_status_regime,
        outer_loop_strategy=outer_loop_strategy,
    )
end


function unitCommitment_with_parameters(;
    epsilon_outerLoop::Float64 = 1e-2,
    epsilon_innerLoop::Float64 = 1e-2,
    lagrangian_atol::Float64 = 1e-4,
    lagrangian_rtol::Float64 = 1e-4,
    lagrangian_iteration_limit::Int = 10000,
    iteration_limit::Int=1000,
    time_limit::Int = 10800,
    sigma::Vector{Float64} = [0.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0],
    sigma_factor::Float64 = 2.0,
    plaPrecision::Array{Array{Float64,1},1} = [[0.226], [0.564], [0.646]], # apart from one generator always 1/5 of pmax
    binaryPrecisionFactor::Float64 = 1/7,
    infiltrate_state::Symbol = :none, # alternatives: :none, :all, :outer, :sigma, :inner, :lagrange, :bellman
    dual_initialization_regime::Symbol = :zeros, # alternatives: :zeros, :gurobi_relax, :cplex_relax, :cplex_fixed, :cplex_combi
    lagrangian_method::Symbol = :kelley, # alternatives: :kelley, :bundle_proximal, :bundle_level
    bundle_alpha::Float64 = 0.5,
    bundle_factor::Float64 = 1.0,
    level_factor::Float64 = 0.2,
    solvers::Vector{String} = ["CPLEX", "CPLEX", "Baron", "SCIP", "CPLEX"],
    cut_selection::Bool = true,
    lag_status_regime::Symbol = :lax,
    outer_loop_strategy::Symbol = :approx,
    )

    # DEFINE MODEL
    ############################################################################
    model = define_10_3()

    # DEFINE SOLVERS
    ############################################################################
    appliedSolvers = NCNBD.AppliedSolvers(solvers[1], solvers[2], solvers[3], solvers[4], solvers[5])

    # DEFINE INITIAL APPROXIMATIONS
    ############################################################################
    binaryPrecision = Dict{Symbol, Float64}()

    for (name, state_comp) in model.nodes[1].ext[:lin_states]
        ub = JuMP.upper_bound(state_comp.out)

        string_name = string(name)
        if occursin("gen", string_name)
            binaryPrecision[name] = binaryPrecisionFactor * ub
        else
            binaryPrecision[name] = 1
        end
    end

    # SET-UP PARAMETER STRUCTS
    ############################################################################
    initialAlgoParameters = NCNBD.InitialAlgoParams(epsilon_outerLoop,
                            epsilon_innerLoop, binaryPrecision, plaPrecision,
                            sigma, sigma_factor, lagrangian_atol,
                            lagrangian_rtol, lagrangian_iteration_limit,
                            dual_initialization_regime, lagrangian_method,
                            bundle_alpha, bundle_factor, level_factor,
                            cut_selection, lag_status_regime,
                            outer_loop_strategy)
    algoParameters = NCNBD.AlgoParams(epsilon_outerLoop, epsilon_innerLoop,
                                      binaryPrecision, sigma, sigma_factor,
                                      infiltrate_state, lagrangian_atol,
                                      lagrangian_rtol, lagrangian_iteration_limit,
                                      dual_initialization_regime,
                                      lagrangian_method, bundle_alpha,
                                      bundle_factor, level_factor,
                                      cut_selection, lag_status_regime,
                                      outer_loop_strategy)

    # SOLVE MODEL
    ############################################################################
    NCNBD.solve(model, algoParameters, initialAlgoParameters, appliedSolvers,
                iteration_limit = iteration_limit, print_level = 2,
                time_limit = time_limit, stopping_rules = [NCNBD.DeterministicStopping()],
                log_file = "C:/Users/cg4102/Documents/julia_logs/UC_10_3_f.log")

    # WRITE LOGS TO FILE
    ############################################################################
    #NCNBD.write_log_to_csv(model, "uc_results.csv", algoParameters)

end


function define_10_3()

    generators = [
        Generator(0, 0.0, 1.13, 0.48, 53.97, 0.0, 171.60, 17.0, 0.28, 0.27, -0.24, 1.02, 0.0),
        Generator(1, 2.2, 2.82, 0.85, 61.59, 0.0, 486.81, 49.0, 0.9, 0.79, -0.3, 1.1, 0.0),
        Generator(0, 0.0, 3.23, 0.84, 54.92, 0.0, 503.34, 50.0, 1.01, 1.00, -0.24, 1.04, 0.0),
    ]
    num_of_generators = size(generators,1)

    # NOTE: no fixed cost, no fixed emission cost, no o&m cost so far
    # NOTE: start-up cost is scaled if less than 24 stages are used, shut-down cost not

    demand_penalty = 5e2
    emission_price = 25

    demand = [3.06 2.91 2.71 2.7 2.73 2.91 3.38 4.01 4.6 4.78]

    num_of_stages = 10

    model = SDDP.LinearPolicyGraph(
        stages = num_of_stages,
        lower_bound = 0.0,
        optimizer = GAMS.Optimizer,
        sense = :Min
    ) do subproblem, t

        # DEFINE LINEARIZED PROBLEM (MILP)
        # ------------------------------------------------------------------
        linearizedSubproblem = JuMP.Model()
        node = subproblem.ext[:sddp_node]
        model = subproblem.ext[:sddp_policy_graph]
        linearizedSubproblem.ext[:sddp_node] = node
        linearizedSubproblem.ext[:sddp_policy_graph] = model
        # place holder for state variable refs of linearized problem
        node.ext[:lin_states] = Dict{Symbol,NCNBD.State{JuMP.VariableRef}}()
        model.ext[:lin_initial_root_state] = Dict{Symbol,Float64}()

        # DEFINE STATE VARIABLES
        # ------------------------------------------------------------------
        JuMP.@variable(
                    subproblem,
                    0.0 <= commit[i = 1:num_of_generators] <= 1.0,
                    SDDP.State,
                    Bin,
                    initial_value = generators[i].comm_ini
                    )
        JuMP.@variable(
                    linearizedSubproblem,
                    0.0 <= commit[i = 1:num_of_generators] <= 1.0,
                    NCNBD.State,
                    Bin,
                    initial_value = generators[i].comm_ini
                    )

        JuMP.@variable(
                    subproblem,
                    0.0 <= gen[i = 1:num_of_generators] <= generators[i].pmax,
                    SDDP.State,
                    initial_value = generators[i].gen_ini
                    )

        JuMP.@variable(
                    linearizedSubproblem,
                    0.0 <= gen[i = 1:num_of_generators] <= generators[i].pmax,
                    NCNBD.State,
                    initial_value = generators[i].gen_ini
                    )

        # DEFINE STAGE t MODEL
        ########################################################################
        # DEFINE STORAGE FOR NONLINEAR DATA
        # ------------------------------------------------------------------
        nonlinearFunctionList = NCNBD.NonlinearFunction[]
        numberOfNonlinearFunctions = num_of_generators

        # DEFINE LINEAR PART OF MODEL
        # ------------------------------------------------------------------
        for problem in [subproblem, linearizedSubproblem]
            gen = problem[:gen]
            commit = problem[:commit]

            # start-up variables
            JuMP.@variable(problem, up[i=1:num_of_generators], Bin)
            JuMP.@variable(problem, down[i=1:num_of_generators], Bin)

            # demand slack
            JuMP.@variable(problem, demand_slack >= 0.0)
            JuMP.@variable(problem, neg_demand_slack >= 0.0)

            # cost variables
            JuMP.@variable(problem, startup_costs[i=1:num_of_generators] >= 0.0)
            JuMP.@variable(problem, shutdown_costs[i=1:num_of_generators] >= 0.0)
            JuMP.@variable(problem, fuel_costs[i=1:num_of_generators] >= 0.0)
            JuMP.@variable(problem, om_costs[i=1:num_of_generators] >= 0.0)
            JuMP.@variable(problem, emission_costs[i=1:num_of_generators] >= 0.0)

            # generation bounds
            JuMP.@constraint(problem, genmin[i=1:num_of_generators], gen[i].out >= commit[i].out * generators[i].pmin)
            JuMP.@constraint(problem, genmax[i=1:num_of_generators], gen[i].out <= commit[i].out * generators[i].pmax)

            # ramping
            # we do not need a case distinction as we defined initial_values
            JuMP.@constraint(problem, rampup[i=1:num_of_generators], gen[i].out - gen[i].in <= generators[i].ramp_up * commit[i].in + generators[i].pmin * (1-commit[i].in))
            JuMP.@constraint(problem, rampdown[i=1:num_of_generators], gen[i].in - gen[i].out <= generators[i].ramp_dw * commit[i].out + generators[i].pmin * (1-commit[i].out))

            # start-up and shut-down
            # we do not need a case distinction as we defined initial_values
            JuMP.@constraint(problem, startup[i=1:num_of_generators], up[i] >= commit[i].out - commit[i].in)
            JuMP.@constraint(problem, shutdown[i=1:num_of_generators], down[i] >= commit[i].in - commit[i].out)

            # load balance
            JuMP.@constraint(problem, load, sum(gen[i].out for i in 1:num_of_generators) + demand_slack - neg_demand_slack == demand[t] )

            # costs
            JuMP.@constraint(problem, startupcost[i=1:num_of_generators], num_of_stages/24 * generators[i].su_cost * up[i] == startup_costs[i])
            JuMP.@constraint(problem, shutdowncost[i=1:num_of_generators], generators[i].sd_cost * down[i] == shutdown_costs[i])
            JuMP.@constraint(problem, fuelcost[i=1:num_of_generators], generators[i].fuel_cost * gen[i].out == fuel_costs[i])
            JuMP.@constraint(problem, omcost[i=1:num_of_generators], generators[i].om_cost * gen[i].out == om_costs[i])

            # DEFINE EXPRESSION GRAPH FOR NONLINEAR CONSTRAINT
            # --------------------------------------------------------------
            JuMP.@variable(problem, emission_aux[1:num_of_generators])
            JuMP.@constraint(problem, emissioncost[i=1:num_of_generators], emission_price * emission_aux[i] == emission_costs[i])
        end

        # DEFINE STAGE OBJECTIVE
        # ------------------------------------------------------------------
        su_costs = subproblem[:startup_costs]
        sd_costs = subproblem[:shutdown_costs]
        f_costs = subproblem[:fuel_costs]
        om_costs = subproblem[:om_costs]
        em_costs = subproblem[:emission_costs]
        demand_slack = subproblem[:demand_slack]
        neg_demand_slack = subproblem[:neg_demand_slack]
        SDDP.@stageobjective(subproblem,
                            sum(su_costs[i] + sd_costs[i] + f_costs[i] + om_costs[i] + em_costs[i] for i in 1:num_of_generators)
                            + demand_slack * demand_penalty + neg_demand_slack * demand_penalty)

        su_costs = linearizedSubproblem[:startup_costs]
        sd_costs = linearizedSubproblem[:shutdown_costs]
        f_costs = linearizedSubproblem[:fuel_costs]
        om_costs = linearizedSubproblem[:om_costs]
        em_costs = linearizedSubproblem[:emission_costs]
        demand_slack = linearizedSubproblem[:demand_slack]
        neg_demand_slack = linearizedSubproblem[:neg_demand_slack]
        NCNBD.@lin_stageobjective(linearizedSubproblem,
                            sum(su_costs[i] + sd_costs[i] + f_costs[i] + om_costs[i] + em_costs[i] for i in 1:num_of_generators)
                            + demand_slack * demand_penalty + neg_demand_slack * demand_penalty)

        # DEFINE NONLINEARITY
        # ------------------------------------------------------------------
        nlf_emission_eval =

        for i in 1:num_of_generators
            # user-defined function for evaluation
            nlf_emission_eval = function nonl_function_eval(y::Float64)
                return generators[i].b * y + generators[i].a * y^2
            end

            # user-defined function for expression building
            nlf_emission_expr = function nonl_function_expr(y::JuMP.VariableRef)
                return :($(generators[i].b) * $(y) + $(generators[i].a) * $(y)^2)
            end

            # define nonlinear expression
            gen = subproblem[:gen][i]
            nonlinear_exp = nlf_emission_expr(gen.out)

            # nonlinear constraint
            aux = subproblem[:emission_aux][i]
            JuMP.add_NL_constraint(subproblem, :($(aux) == $(nonlinear_exp)))

            # define nonlinearFunction struct for PLA
            gen = linearizedSubproblem[:gen][i]
            aux = linearizedSubproblem[:emission_aux][i]

            nlf = NCNBD.NonlinearFunction(nlf_emission_eval, nlf_emission_expr, aux, [gen.out], :noshift, :replace)
            push!(nonlinearFunctionList, nlf)

        end

        # store in ext of subproblem
        subproblem.ext[:nlFunctions] = nonlinearFunctionList
        subproblem.ext[:linSubproblem] = linearizedSubproblem

    end

    return model
end

end
