using JuMP
using SDDP
using NCNBD
using Revise
using Gurobi
using GAMS
using SCIP
using Infiltrator


function thirdExample()

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
            nlf = NCNBD.NonlinearFunction(nlf_eval, nlf_expr, nonlinearAux[1], [x.out])
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
            nlf = NCNBD.NonlinearFunction(nlf_eval, nlf_expr, nonlinearAux[1], [y_loc[2]])
            push!(nonlinearFunctionList, nlf)

            # store in ext of subproblem
            subproblem.ext[:nlFunctions] = nonlinearFunctionList
            subproblem.ext[:linSubproblem] = linearizedSubproblem

        end
    end

    # SET-UP PARAMETERS
    ############################################################################
    appliedSolvers = NCNBD.AppliedSolvers(Gurobi.Optimizer, Gurobi.Optimizer, SCIP.Optimizer)

    epsilon_outerLoop = 0.001
    epsilon_innerLoop = 0.001
    binaryPrecision = Dict(:x => 0.5)
    plaPrecision = [2, 4]
    sigma = [1.0, 1.0]
    sigma_counter = 5

    #@infiltrate

    initialAlgoParameters = NCNBD.InitialAlgoParams(epsilon_outerLoop,
                            epsilon_innerLoop, binaryPrecision, plaPrecision,
                            sigma, sigma_counter)
    algoParameters = NCNBD.AlgoParams(epsilon_outerLoop, epsilon_innerLoop,
                                      binaryPrecision, sigma, sigma_counter)

    # SET-UP NONLINEARITIES
    ############################################################################
    NCNBD.solve(model, algoParameters, initialAlgoParameters, appliedSolvers,
                iteration_limit = 5, print_level = 1,
                time_limit = 600, stopping_rules = [NCNBD.DeterministicStopping()])

    @infiltrate

    # WRITE LOGS TO FILE
    ############################################################################
    NCNBD.write_log_to_csv(model, "test_results.csv", algoParameters)

end

thirdExample()
