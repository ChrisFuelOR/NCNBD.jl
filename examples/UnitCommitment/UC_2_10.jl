using JuMP
using SDDP
using NCNBD
using Revise
using Gurobi
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


function unitCommitment_2_10()

    generators = [
        Generator(0, 0.0, 2.0, 0.4, 18.0, 2.0, 42.6, 42.6, 0.4, 0.4, -0.34, 1.0, 0.0),
        Generator(0, 0.0, 3.2, 0.64, 15.0, 4.0, 50.6, 50.6, 0.64, 0.64, -0.21, 1.0, 0.0),
        Generator(0, 0.0, 1.5, 0.3, 17.0, 2.0, 57.1, 57.1, 0.3, 0.3, -0.39, 0.95, 0.0),
        Generator(1, 4.0, 5.0, 1.04, 13.2, 4.0, 47.1, 47.1, 1.04, 1.04, -0.14, 1.09, 0.0),
        Generator(1, 2.8, 2.8, 0.56, 14.3, 4.0, 56.9, 56.9, 0.56, 0.56, -0.24, 1.0, 0.0),
        Generator(0, 0.0, 0.8, 0.16, 40.2, 4.0, 141.5, 141.5, 0.3, 0.3, -0.85, 1.0, 0.0),
        Generator(1, 1.2, 1.2, 0.24, 17.1, 2.0, 113.5, 113.5, 0.24, 0.24, -0.53, 0.91, 0.0),
        Generator(1, 1.1, 1.1, 0.22, 17.3, 2.0, 42.6, 42.6, 0.22, 0.22, -0.62, 0.95, 0.0),
        Generator(0, 0.0, 0.8, 0.16, 59.4, 4.0, 50.6, 50.6, 0.16, 0.16, -0.79, 0.95, 0.0),
        Generator(0, 0.0, 0.6, 0.12, 19.5, 2.0, 57.1, 57.1, 0.12, 0.12, -1.13, 1.0, 0.0),
    ]
    num_of_generators = size(generators,1)

    demand_penalty = 5e2
    emission_price = 2.5

    demand = [8.83 9.15]
    # 8.83 9.15 10.10 11.49 12.36 13.31 13.97 14.19 14.55 14.55 14.41 14.19
    # 13.97 13.39 13.68 13.39 12.36 11.05 10.38 9.59 9.22 8.85 9.15 8.34

    model = SDDP.LinearPolicyGraph(
        stages = 2,
        lower_bound = 0.0,
        optimizer = Gurobi.Optimizer,
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

    # SET-UP PARAMETERS
    ############################################################################
    # appliedSolvers = NCNBD.AppliedSolvers(GAMS.Optimizer, GAMS.Optimizer, GAMS.Optimizer, GAMS.Optimizer)
    appliedSolvers = NCNBD.AppliedSolvers("Gurobi", "Gurobi", "Baron", "Baron")

    epsilon_outerLoop = 1e-3
    epsilon_innerLoop = 1e-3

    binaryPrecision = Dict{Symbol, Float64}()

    for (name, state_comp) in model.nodes[1].ext[:lin_states]
        ub = JuMP.upper_bound(state_comp.out)

        string_name = string(name)
        if occursin("gen", string_name)
            binaryPrecision[name] = 1/7 * ub
        else
            binaryPrecision[name] = 1
        end
    end

    plaPrecision = [0.4, 0.64, 0.3, 1.04, 0.56, 0.2, 0.24, 0.22, 0.16, 0.12] # apart from one generator always 1/5 of pmax
    sigma = [0.0, 1.0]
    sigma_factor = 5

    infiltrate_state = :none
    # alternatives: :none, :all, :outer, :sigma, :inner, :lagrange, :bellman

    initialAlgoParameters = NCNBD.InitialAlgoParams(epsilon_outerLoop,
                            epsilon_innerLoop, binaryPrecision, plaPrecision,
                            sigma, sigma_factor)
    algoParameters = NCNBD.AlgoParams(epsilon_outerLoop, epsilon_innerLoop,
                                      binaryPrecision, sigma, sigma_factor,
                                      infiltrate_state)

    # SET-UP NONLINEARITIES
    ############################################################################
    NCNBD.solve(model, algoParameters, initialAlgoParameters, appliedSolvers,
                iteration_limit = 15, print_level = 2,
                time_limit = 7200, stopping_rules = [NCNBD.DeterministicStopping()],
                log_file = "C:/Users/cg4102/Documents/julia_logs/UC_2_10.log")

    # WRITE LOGS TO FILE
    ############################################################################
    NCNBD.write_log_to_csv(model, "uc_results.csv", algoParameters)

end

unitCommitment_2_10()
