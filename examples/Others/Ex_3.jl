module Ex_3

export thirdExample
export thirdExample_with_parameters

using JuMP
using SDDP
using NCNBD
using Revise
#using Gurobi
using GAMS
#using SCIP
using Infiltrator


function thirdExample()

    # define required tolerances
    epsilon_outerLoop = 1e-3
    epsilon_innerLoop = 1e-3 #1e-4
    lagrangian_atol = 1e-8
    lagrangian_rtol = 1e-8

    # define time and iteration limits
    lagrangian_iteration_limit = 1000
    iteration_limit = 1000
    time_limit = 10800

    # define sigma
    sigma = [0.0, 1.0]
    sigma_factor = 5.0

    # define initial approximations
    plaPrecision = [2.0, 4.0]
    binaryPrecisionFactor = 0.5

    # define infiltration level
    infiltrate_state = :none
    # alternatives: :none, :all, :outer, :sigma, :inner, :lagrange, :bellman

    # define regime for initializing duals for Lagrangian relaxation
    dual_initialization_regime = :zeros
    # alternatives: :zeros, :gurobi_relax, :cplex_relax, :cplex_fixed, :cplex_combi

    # define solution method for lagrangian dual
    lagrangian_method = :kelley
    # alternatives: :kelley, :bundle_proximal, :bundle_level

    bundle_alpha = 0.5
    bundle_factor = 1.0
    level_factor = 0.2

    # cut selection strategy
    cut_selection = true

    # used solvers
    solvers = ["CPLEX", "CPLEX", "Baron", "Baron", "CPLEX"]

    # CALL METHOD WITH PARAMETERS
    ############################################################################
    thirdExample_with_parameters(
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
    )
end


function thirdExample_with_parameters(;
    epsilon_outerLoop::Float64 = 1e-3,
    epsilon_innerLoop::Float64 = 1e-3,
    lagrangian_atol::Float64 = 1e-8,
    lagrangian_rtol::Float64 = 1e-8,
    lagrangian_iteration_limit::Int = 1000,
    iteration_limit::Int=1000,
    time_limit::Int = 10800,
    sigma::Vector{Float64} = [0.0, 1.0],
    sigma_factor::Float64 = 5.0,
    plaPrecision::Vector{Float64} = [2.0, 4.0],
    binaryPrecisionFactor::Float64 = 0.5,
    infiltrate_state::Symbol = :none, # alternatives: :none, :all, :outer, :sigma, :inner, :lagrange, :bellman
    dual_initialization_regime::Symbol = :zeros, # alternatives: :zeros, :gurobi_relax, :cplex_relax, :cplex_fixed, :cplex_combi
    lagrangian_method::Symbol = :kelley, # alternatives: :kelley, :bundle_proximal, :bundle_level
    bundle_alpha::Float64 = 0.5,
    bundle_factor::Float64 = 1.0,
    level_factor::Float64 = 0.2,
    solvers::Vector{String} = ["Gurobi", "Gurobi", "Baron", "Baron", "Gurobi"],
    cut_selection::Bool = true,
    )

    # DEFINE MODEL
    ############################################################################
    model = define_thirdExample()

    # DEFINE SOLVERS
    ############################################################################
    appliedSolvers = NCNBD.AppliedSolvers(solvers[1], solvers[2], solvers[3], solvers[4], solvers[5])

    # DEFINE INITIAL APPROXIMATIONS
    ############################################################################
    binaryPrecision = Dict{Symbol, Float64}()

    for (name, state_comp) in model.nodes[1].ext[:lin_states]
        binaryPrecision[name] = binaryPrecisionFactor
    end

    # SET-UP PARAMETER STRUCTS
    ############################################################################
    initialAlgoParameters = NCNBD.InitialAlgoParams(epsilon_outerLoop,
                            epsilon_innerLoop, binaryPrecision, plaPrecision,
                            sigma, sigma_factor, lagrangian_atol,
                            lagrangian_rtol, lagrangian_iteration_limit,
                            dual_initialization_regime, lagrangian_method,
                            bundle_alpha, bundle_factor, level_factor,
                            cut_selection)
    algoParameters = NCNBD.AlgoParams(epsilon_outerLoop, epsilon_innerLoop,
                                      binaryPrecision, sigma, sigma_factor,
                                      infiltrate_state, lagrangian_atol,
                                      lagrangian_rtol, lagrangian_iteration_limit,
                                      dual_initialization_regime,
                                      lagrangian_method, bundle_alpha,
                                      bundle_factor, level_factor,
                                      cut_selection)

    # SOLVE MODEL
    ############################################################################
    NCNBD.solve(model, algoParameters, initialAlgoParameters, appliedSolvers,
                iteration_limit = iteration_limit, print_level = 2,
                time_limit = time_limit, stopping_rules = [NCNBD.DeterministicStopping()],
                log_file = "C:/Users/cg4102/Documents/julia_logs/ThirdEx.log")

    # WRITE LOGS TO FILE
    ############################################################################
    #NCNBD.write_log_to_csv(model, "uc_results.csv", algoParameters)

end


function define_thirdExample()

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
        JuMP.@variable(subproblem, 0.0 <= x <= 2.0, SDDP.State, initial_value = 0)
        JuMP.@variable(linearizedSubproblem, 0.0 <= x <= 2.0, NCNBD.State, initial_value = 0)

        # DEFINE STAGE 1 MODEL
        ########################################################################
        if t == 1

            # DEFINE STORAGE FOR NONLINEAR DATA
            # ------------------------------------------------------------------
            nonlinearFunctionList = NCNBD.NonlinearFunction[]
            numberOfNonlinearFunctions = 1

            # DEFINE LINEAR PART OF MODEL
            # ------------------------------------------------------------------
            for problem in [subproblem, linearizedSubproblem]
                x = problem[:x]
                JuMP.@variable(problem, 0.0 <= y_loc <= 2.0)
                JuMP.@constraint(problem, lin_con, y_loc + 3/2*x.out >= 3 + x.in)

                # DEFINE EXPRESSION GRAPH FOR NONLINEAR CONSTRAINT
                # --------------------------------------------------------------
                JuMP.@variable(problem, nonlinearAux[1:numberOfNonlinearFunctions])
                JuMP.@constraint(problem, actual_nlcon, y_loc - nonlinearAux[1] >= 0)
            end

            # DEFINE STAGE OBJECTIVE
            # ------------------------------------------------------------------
            y_loc = subproblem[:y_loc]
            SDDP.@stageobjective(subproblem, y_loc)

            y_loc = linearizedSubproblem[:y_loc]
            NCNBD.@lin_stageobjective(linearizedSubproblem, y_loc)

            # DEFINE NONLINEARITY
            # ------------------------------------------------------------------
            # user-defined function for evaluation
            nlf_eval = function nonl_function_1_eval(y::Float64)
                return sqrt(y)
            end

            # user-defined function for expression building
            nlf_expr = function nonl_function_1_expr(y::JuMP.VariableRef)
                return :(sqrt($(y)))
            end

            # define nonlinear expression
            x = subproblem[:x]
            nonlinear_exp = nlf_expr(x.out)

            # nonlinear constraint
            nonlinearAux = subproblem[:nonlinearAux]
            JuMP.add_NL_constraint(subproblem, :($(nonlinearAux[1]) == $(nonlinear_exp)))

            # define nonlinearFunction struct for PLA
            x = linearizedSubproblem[:x]
            nonlinearAux = linearizedSubproblem[:nonlinearAux]
            nlf = NCNBD.NonlinearFunction(nlf_eval, nlf_expr, nonlinearAux[1], [x.out], :replace)
            push!(nonlinearFunctionList, nlf)

            # store in ext of subproblem
            subproblem.ext[:nlFunctions] = nonlinearFunctionList
            subproblem.ext[:linSubproblem] = linearizedSubproblem

        # DEFINE STAGE 2 MODEL
        ########################################################################
        else

            # DEFINE STORAGE FOR NONLINEAR DATA
            # ------------------------------------------------------------------
            nonlinearFunctionList = NCNBD.NonlinearFunction[]
            numberOfNonlinearFunctions = 1

            # DEFINE LINEAR PART OF MODEL
            # ------------------------------------------------------------------
            for problem in [subproblem, linearizedSubproblem]
                x = problem[:x]
                JuMP.@variable(problem, 0.0 <= y_loc[i=1:2] <= 4.0)
                JuMP.@constraint(problem, lin_con, sum(y_loc[i] for i in 1:2) == 2 * x.in)
                JuMP.@constraint(problem, x.out == 0)

                # DEFINE EXPRESSION GRAPH FOR NONLINEAR CONSTRAINT
                # --------------------------------------------------------------
                JuMP.@variable(problem, nonlinearAux[1:numberOfNonlinearFunctions])
                JuMP.@constraint(problem, actual_nlcon, y_loc[1] - nonlinearAux[1] >= 0)
            end

            # DEFINE STAGE OBJECTIVE
            # ------------------------------------------------------------------
            y_loc = subproblem[:y_loc]
            SDDP.@stageobjective(subproblem, y_loc[1])

            y_loc = linearizedSubproblem[:y_loc]
            NCNBD.@lin_stageobjective(linearizedSubproblem, y_loc[1])

            # DEFINE NONLINEARITY
            # ------------------------------------------------------------------
            # user-defined function for evaluation
            nlf_eval = function nonl_function_2_eval(y::Float64)
                return y^2
            end

            # user-defined function for expression building
            nlf_expr = function nonl_function_2_expr(y::JuMP.VariableRef)
                return :($(y)^2)
            end

            # define nonlinear expression
            y_loc = subproblem[:y_loc]
            nonlinear_exp = nlf_expr(y_loc[2])

            # nonlinear constraint
            nonlinearAux = subproblem[:nonlinearAux]
            JuMP.add_NL_constraint(subproblem, :($(nonlinearAux[1]) == $(nonlinear_exp)))

            # define nonlinearFunction struct for PLA
            y_loc = linearizedSubproblem[:y_loc]
            nonlinearAux = linearizedSubproblem[:nonlinearAux]
            nlf = NCNBD.NonlinearFunction(nlf_eval, nlf_expr, nonlinearAux[1], [y_loc[2]], :keep)
            push!(nonlinearFunctionList, nlf)

            # store in ext of subproblem
            subproblem.ext[:nlFunctions] = nonlinearFunctionList
            subproblem.ext[:linSubproblem] = linearizedSubproblem

        end
    end

    return model
end

end

#thirdExample()
