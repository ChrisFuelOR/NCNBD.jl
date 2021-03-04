module UC_2_5_Batt

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

struct Battery
    max_charge::Float64
    max_discharge::Float64
    max_level::Float64
    min_level::Float64
    #eff_charge::Float64
    #eff_discharge::Float64
    self_discharge::Float64
    level_ini::Float64
    level_end::Float64
end

function unitCommitment()

    # define required tolerances
    epsilon_outerLoop = 1e-2
    epsilon_innerLoop = 1e-2
    lagrangian_atol = 1e-4
    lagrangian_rtol = 1e-4

    # define time and iteration limits
    lagrangian_iteration_limit = 1000
    iteration_limit = 1000
    time_limit = 10800

    # define sigma
    sigma = [0.0, 1000.0]
    sigma_factor = 2.0

    # define initial approximations
    plaPrecision = [[0.4], [0.64], [0.3], [1.04], [0.56], [0.05, 0.1], [0.05, 0.1]] # apart from one generator always 1/5 of pmax
    binaryPrecisionFactor = 1/7

    # define infiltration level
    # TODO: Abstract data type
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
    level_factor = 0.8

    # cut selection strategy
    cut_selection = true

    # lagrangian status
    lag_status_regime = :lax
    # alternatives: :rigorous, :lax

    # outer loop strategy
    outer_loop_strategy = :approx
    # alternatives: :opt, :approx

    # used solvers
    solvers = ["Gurobi", "Gurobi", "Baron", "Baron", "Gurobi"]

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
        cut_selection=cut_selection,
        lag_status_regime=lag_status_regime,
        outer_loop_strategy=outer_loop_strategy,
    )
end


function unitCommitment_with_parameters(;
    epsilon_outerLoop::Float64 = 1e-3,
    epsilon_innerLoop::Float64 = 1e-3,
    lagrangian_atol::Float64 = 1e-8,
    lagrangian_rtol::Float64 = 1e-8,
    lagrangian_iteration_limit::Int = 1000,
    iteration_limit::Int=1000,
    time_limit::Int = 10800,
    sigma::Vector{Float64} = [0.0, 1000.0],
    sigma_factor::Float64 = 2.0,
    plaPrecision::Array{Vector{Float64},1} = [[0.4], [0.64], [0.3], [1.04], [0.56], [0.05, 0.1], [0.05, 0.1]], # apart from one generator always 1/5 of pmax
    binaryPrecisionFactor::Float64 = 1/7,
    infiltrate_state::Symbol = :none, # alternatives: :none, :all, :outer, :sigma, :inner, :lagrange, :bellman
    dual_initialization_regime::Symbol = :zeros, # alternatives: :zeros, :gurobi_relax, :cplex_relax, :cplex_fixed, :cplex_combi
    lagrangian_method::Symbol = :kelley, # alternatives: :kelley, :bundle_proximal, :bundle_level
    bundle_alpha::Float64 = 0.5,
    bundle_factor::Float64 = 1.0,
    level_factor::Float64 = 0.4,
    solvers::Vector{String} = ["Gurobi", "Gurobi", "Baron", "Baron", "Gurobi"],
    cut_selection::Bool = true,
    lag_status_regime::Symbol = :rigorous,
    outer_loop_strategy::Symbol = :approx,
    )

    # DEFINE MODEL
    ############################################################################
    model = define_2_5()

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
                log_file = "C:/Users/cg4102/Documents/julia_logs/UC_2_5_batt.log")

    # WRITE LOGS TO FILE
    ############################################################################
    #NCNBD.write_log_to_csv(model, "uc_results.csv", algoParameters)

end


function define_2_5()

    generators = [
        Generator(0, 0.0, 2.0, 0.4, 18.0, 2.0, 42.6, 42.6, 0.4, 0.4, -0.34, 1.0, 0.0),
        Generator(0, 0.0, 3.2, 0.64, 15.0, 4.0, 50.6, 50.6, 0.64, 0.64, -0.21, 1.0, 0.0),
        Generator(0, 0.0, 1.5, 0.3, 17.0, 2.0, 57.1, 57.1, 0.3, 0.3, -0.39, 0.95, 0.0),
        Generator(1, 4.0, 5.0, 1.04, 13.2, 4.0, 47.1, 47.1, 1.04, 1.04, -0.14, 1.09, 0.0),
        Generator(1, 2.8, 2.8, 0.56, 14.3, 4.0, 56.9, 56.9, 0.56, 0.56, -0.24, 1.0, 0.0),
    ]
    num_of_generators = size(generators,1)

    batteries = [
        Battery(0.25, 0.25, 0.8, 0.1, 0.05, 0.1, 0.1), #0.9, 0.9
        #Battery(0.4, 0.4, 1.0, 0.1, 0.05, 0.4, 0.4), #0.9, 0.9
    ]
    num_of_batteries = size(batteries,1)

    demand_penalty = 5e2
    emission_price = 2.5

    demand = [8.0 8.5]

    num_of_stages = 2

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

        # additional battery states
        JuMP.@variable(
                    subproblem,
                    batteries[i].min_level <= level[i = 1:num_of_batteries] <= batteries[i].max_level,
                    SDDP.State,
                    initial_value = batteries[i].level_ini
                    )

        JuMP.@variable(
                    linearizedSubproblem,
                    batteries[i].min_level <= level[i = 1:num_of_batteries] <= batteries[i].max_level,
                    NCNBD.State,
                    initial_value = batteries[i].level_ini
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
            level = problem[:level]

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
            #JuMP.@constraint(problem, load, sum(gen[i].out for i in 1:num_of_generators) + demand_slack == demand[t] )

            # costs
            JuMP.@constraint(problem, startupcost[i=1:num_of_generators], generators[i].su_cost * up[i] == startup_costs[i])
            JuMP.@constraint(problem, shutdowncost[i=1:num_of_generators], generators[i].sd_cost * down[i] == shutdown_costs[i])
            JuMP.@constraint(problem, fuelcost[i=1:num_of_generators], generators[i].fuel_cost * gen[i].out == fuel_costs[i])
            JuMP.@constraint(problem, omcost[i=1:num_of_generators], generators[i].om_cost * gen[i].out == om_costs[i])

            # battery variables
            JuMP.@variable(problem, 0.0 <= charge[i=1:num_of_batteries] <= batteries[i].max_charge)
            JuMP.@variable(problem, 0.0 <= discharge[i=1:num_of_batteries] <= batteries[i].max_discharge)
            JuMP.@variable(problem, charging[i=1:num_of_batteries], Bin)
            JuMP.@variable(problem, discharging[i=1:num_of_batteries], Bin)
            JuMP.@variable(problem, 0.6 <= efficiency[i=1:num_of_batteries] <= 1.0)
            JuMP.@variable(problem, batteries[i].min_level / batteries[i].max_level <= soc[i=1:num_of_batteries] <= 1.0)

            # battery constraints
            JuMP.@constraint(problem, charge_eq[i=1:num_of_batteries], charge[i] <= batteries[i].max_charge * charging[i])
            JuMP.@constraint(problem, discharge_eq[i=1:num_of_batteries], discharge[i] <= batteries[i].max_discharge * discharging[i])
            JuMP.@constraint(problem, onlycharge[i=1:num_of_batteries], charging[i] + discharging[i] <= 1)
            JuMP.@constraint(problem, dischargelevel[i=1:num_of_batteries], discharge[i] <= level[i].in + charge[i])
            JuMP.@constraint(problem, soc_eq[i=1:num_of_batteries], soc[i] == level[i].in / batteries[i].max_level)

            if t == num_of_stages
                JuMP.@constraint(problem, end_level[i=1:num_of_batteries], level[i].out == batteries[i].level_end)
            end

            # load balance
            JuMP.@constraint(problem, load,
                sum(gen[i].out for i in 1:num_of_generators)
                + sum(discharge[i] - charge[i] for i in 1:num_of_batteries)
                + demand_slack == demand[t]
                )

            # DEFINE EXPRESSION GRAPH FOR NONLINEAR CONSTRAINT
            # --------------------------------------------------------------
            JuMP.@variable(problem, emission_aux[1:num_of_generators])
            JuMP.@constraint(problem, emissioncost[i=1:num_of_generators], emission_price * emission_aux[i] == emission_costs[i])

            # DEFINE EXPRESSION GRAPH FOR NONLINEAR BATTERY CONSTRAINT
            # --------------------------------------------------------------
            JuMP.@variable(problem, charge_aux[1:num_of_batteries])
            JuMP.@variable(problem, discharge_aux[1:num_of_batteries])
            JuMP.@constraint(problem,
                battery_level[i=1:num_of_batteries],
                level[i].out == (1-batteries[i].self_discharge) * level[i].in + charge_aux[i] - discharge_aux[i]
                )
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

        # DEFINE NONLINEARITY FOR BATTERY
        # ------------------------------------------------------------------

        for i in 1:num_of_batteries
            # user-defined function for evaluation
            nlf_charge_eval = function nonl_charge_eval(x_charge::Float64, x_soc::Float64)
                return x_charge * (1/12 * log(x_soc / (1 + x_soc)) + 1)
            end

            # user-defined function for expression building
            nlf_charge_expr = function nonl_charge_expr(x_charge::JuMP.VariableRef, x_soc::JuMP.VariableRef)
                return :($(x_charge) * (1/12 * log($(x_soc) / (1 + $(x_soc))) + 1))
            end

            # define nonlinear expression
            charge = subproblem[:charge][i]
            soc = subproblem[:soc][i]
            nonlinear_exp_1 = nlf_charge_expr(charge, soc)

            # nonlinear constraint
            aux = subproblem[:charge_aux][i]
            JuMP.add_NL_constraint(subproblem, :($(aux) == $(nonlinear_exp_1)))

            # define nonlinearFunction struct for PLA
            charge = linearizedSubproblem[:charge][i]
            soc = linearizedSubproblem[:soc][i]
            aux = linearizedSubproblem[:charge_aux][i]

            nlf2 = NCNBD.NonlinearFunction(nlf_charge_eval, nlf_charge_expr, aux, [charge, soc], :shiftUp, :keep) # concave, but in equality (or >=)
            push!(nonlinearFunctionList, nlf2)

            ####################################################################

            # user-defined function for evaluation
            nlf_discharge_eval = function nonl_discharge_eval(x_discharge::Float64, x_soc::Float64)
                return x_discharge / (1/12 * log(x_soc / (1 + x_soc)) + 1)
            end

            # user-defined function for expression building
            nlf_discharge_expr = function nonl_discharge_expr(x_discharge::JuMP.VariableRef, x_soc::JuMP.VariableRef)
                return :($(x_discharge) / (1/12 * log($(x_soc) / (1 + $(x_soc))) + 1))
            end

            # define nonlinear expression
            discharge = subproblem[:discharge][i]
            soc = subproblem[:soc][i]
            nonlinear_exp_2 = nlf_discharge_expr(discharge, soc)

            # nonlinear constraint
            aux = subproblem[:discharge_aux][i]
            JuMP.add_NL_constraint(subproblem, :($(aux) == $(nonlinear_exp_2)))

            # define nonlinearFunction struct for PLA
            discharge = linearizedSubproblem[:discharge][i]
            soc = linearizedSubproblem[:soc][i]
            aux = linearizedSubproblem[:discharge_aux][i]

            nlf3 = NCNBD.NonlinearFunction(nlf_discharge_eval, nlf_discharge_expr, aux, [discharge, soc], :shiftDown, :keep) # convex, but in equality (or >=)
            push!(nonlinearFunctionList, nlf3)

        end

        # store in ext of subproblem
        subproblem.ext[:nlFunctions] = nonlinearFunctionList
        subproblem.ext[:linSubproblem] = linearizedSubproblem

    end

    return model
end

end

#unitCommitment_2_2()
