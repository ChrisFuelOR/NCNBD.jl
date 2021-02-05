module UC_2_5

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
    epsilon_outerLoop = 1e-1
    epsilon_innerLoop = 1e-2
    lagrangian_atol = 1e-8
    lagrangian_rtol = 1e-8

    # define time and iteration limits
    lagrangian_iteration_limit = 1000
    iteration_limit = 1000
    time_limit = 10800

    # define sigma
    sigma = [0.0, 10.0]
    sigma_factor = 5.0

    # define initial approximations
    plaPrecision = [40, 64, 30, 104, 56] # apart from one generator always 1/5 of pmax
    binaryPrecisionFactor = 1/7

    # define infiltration level
    # TODO: Abstract data type
    infiltrate_state = :none
    # alternatives: :none, :all, :outer, :sigma, :inner, :lagrange, :bellman

    # define regime for initializing duals for Lagrangian relaxation
    dual_initialization_regime = :cplex_combi
    # alternatives: :zeros, :gurobi_relax, :cplex_relax, :cplex_fixed, :cplex_combi

    # define solution method for lagrangian dual
    lagrangian_method = :bundle_level
    # alternatives: :kelley, :bundle_proximal, :bundle_level

    bundle_alpha = 0.5
    bundle_factor = 1.0
    level_factor = 0.8

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
        level_factor=level_factor
    )
end


function unitCommitment_with_parameters(;
    epsilon_outerLoop::Float64 = 1e-1,
    epsilon_innerLoop::Float64 = 1e-2,
    lagrangian_atol::Float64 = 1e-8,
    lagrangian_rtol::Float64 = 1e-8,
    lagrangian_iteration_limit::Int = 1000,
    iteration_limit::Int=1000,
    time_limit::Int = 10800,
    sigma::Vector{Float64} = [0.0, 10.0],
    sigma_factor::Float64 = 5.0,
    plaPrecision::Vector{Float64} = [40, 64, 30, 104, 56], # apart from one generator always 1/5 of pmax
    binaryPrecisionFactor::Float64 = 1/7,
    infiltrate_state::Symbol = :none, # alternatives: :none, :all, :outer, :sigma, :inner, :lagrange, :bellman
    dual_initialization_regime::Symbol = :zeros, # alternatives: :zeros, :gurobi_relax, :cplex_relax, :cplex_fixed, :cplex_combi
    lagrangian_method::Symbol = :kelley, # alternatives: :kelley, :bundle_proximal, :bundle_level
    bundle_alpha::Float64 = 0.5,
    bundle_factor::Float64 = 1.0,
    level_factor::Float64 = 0.4,
    )

    # DEFINE MODEL
    ############################################################################
    model = define_2_5()

    # DEFINE SOLVERS
    ############################################################################
    appliedSolvers = NCNBD.AppliedSolvers("Gurobi", "Gurobi", "Baron", "Baron")

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
                            bundle_alpha, bundle_factor, level_factor)
    algoParameters = NCNBD.AlgoParams(epsilon_outerLoop, epsilon_innerLoop,
                                      binaryPrecision, sigma, sigma_factor,
                                      infiltrate_state, lagrangian_atol,
                                      lagrangian_rtol, lagrangian_iteration_limit,
                                      dual_initialization_regime,
                                      lagrangian_method, bundle_alpha,
                                      bundle_factor, level_factor)

    # SOLVE MODEL
    ############################################################################
    NCNBD.solve(model, algoParameters, initialAlgoParameters, appliedSolvers,
                iteration_limit = iteration_limit, print_level = 2,
                time_limit = time_limit, stopping_rules = [NCNBD.DeterministicStopping()],
                log_file = "C:/Users/cg4102/Documents/julia_logs/UC_2_5.log")

    # WRITE LOGS TO FILE
    ############################################################################
    #NCNBD.write_log_to_csv(model, "uc_results.csv", algoParameters)

end


function define_2_5()

    generators = [
        Generator(0, 0.0, 200.0, 40.0, 18.0, 2.0, 42.6, 42.6, 40.0, 40.0, -2.375, 1025.0, 0.0),
        Generator(0, 0.0, 320.0, 64.0, 15.0, 4.0, 50.6, 50.6, 64.0, 64.0, -2.75, 1800.0, 0.0),
        Generator(0, 0.0, 150.0, 30.0, 17.0, 2.0, 57.1, 57.1, 30.0, 30.0, -3.2, 1025.0, 0.0),
        Generator(1, 400.0, 520.0, 104.0, 13.2, 4.0, 47.1, 47.1, 104.0, 104.0, -1.5, 1800.0, 0.0),
        Generator(1, 280.0, 280.0, 56.0, 14.3, 4.0, 56.9, 56.9, 56.0, 56.0, -3, 1800.0, 0.0),
        Generator(0, 0.0, 80.0, 16.0, 40.2, 4.0, 141.5, 141.5, 30.0, 30.0, -6.8, 1200.0, 0.0),
    ]
    num_of_generators = size(generators,1)

    demand_penalty = 5e4
    emission_price = 0.02

    demand = [800 850]

    model = SDDP.LinearPolicyGraph(
        stages = 2,
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
            JuMP.@constraint(problem, rampup[i=1:num_of_generators], gen[i].out - gen[i].in <= generators[i].ramp_up)
            JuMP.@constraint(problem, rampdown[i=1:num_of_generators], gen[i].in - gen[i].out <= generators[i].ramp_dw)

            # start-up and shut-down
            # we do not need a case distinction as we defined initial_values
            JuMP.@constraint(problem, startup[i=1:num_of_generators], up[i] >= commit[i].out - commit[i].in)
            JuMP.@constraint(problem, shutdown[i=1:num_of_generators], down[i] >= commit[i].in - commit[i].out)

            # load balance
            JuMP.@constraint(problem, load, sum(gen[i].out for i in 1:num_of_generators) + demand_slack == demand[t] )

            # costs
            JuMP.@constraint(problem, startupcost[i=1:num_of_generators], generators[i].su_cost * up[i] == startup_costs[i])
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
        SDDP.@stageobjective(subproblem,
                            sum(su_costs[i] + sd_costs[i] + f_costs[i] + om_costs[i] + em_costs[i] for i in 1:num_of_generators)
                            + demand_slack * demand_penalty)

        su_costs = linearizedSubproblem[:startup_costs]
        sd_costs = linearizedSubproblem[:shutdown_costs]
        f_costs = linearizedSubproblem[:fuel_costs]
        om_costs = linearizedSubproblem[:om_costs]
        em_costs = linearizedSubproblem[:emission_costs]
        demand_slack = linearizedSubproblem[:demand_slack]
        NCNBD.@lin_stageobjective(linearizedSubproblem,
                            sum(su_costs[i] + sd_costs[i] + f_costs[i] + om_costs[i] + em_costs[i] for i in 1:num_of_generators)
                            + demand_slack * demand_penalty)

        # DEFINE NONLINEARITY
        # ------------------------------------------------------------------
        # TODO: Add c*commit, but then two-dimensional
        # TODO: Use same function only ones, but insert correct paramters

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

#unitCommitment_2_5()
