using JuMP
using SDDP
using NCNBD
using Revise
using Gurobi
using GAMS
using SCIP
using Infiltrator


function discontExample()

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
        JuMP.@variable(subproblem, 0.0 <= b <= 2.0, SDDP.State, initial_value = 0)
        JuMP.@variable(linearizedSubproblem, 0.0 <= b <= 2.0, NCNBD.State, initial_value = 0)

        # DEFINE STAGE 1 MODEL
        ########################################################################
        if t == 1

            # DEFINE LINEAR PART OF MODEL
            # ------------------------------------------------------------------
            for problem in [subproblem, linearizedSubproblem]
                b = problem[:b]
                JuMP.@constraint(problem, b.out == 1.1 + b.in)
            end

            # DEFINE STAGE OBJECTIVE
            # ------------------------------------------------------------------
            SDDP.@stageobjective(subproblem, 1)
            NCNBD.@lin_stageobjective(linearizedSubproblem, 1)

            # store in ext of subproblem
            nonlinearFunctionList = NCNBD.NonlinearFunction[]
            subproblem.ext[:nlFunctions] = nonlinearFunctionList
            subproblem.ext[:linSubproblem] = linearizedSubproblem

        # DEFINE STAGE 2 MODEL
        ########################################################################
        else

            # DEFINE LINEAR PART OF MODEL
            # ------------------------------------------------------------------
            for problem in [subproblem, linearizedSubproblem]
                b = problem[:b]
                JuMP.@variable(problem, 0.0 <= x[i=1:4])
                JuMP.set_integer(x[1])
                JuMP.set_integer(x[2])

                JuMP.@constraint(problem, con, 1.25*x[1] - x[2] + 0.5*x[3] + 1/3*x[4] == b.in)
                JuMP.@constraint(problem, b.out == 0)
            end

            # DEFINE STAGE OBJECTIVE
            # ------------------------------------------------------------------
            x = subproblem[:x]
            SDDP.@stageobjective(subproblem, x[1] - 0.75*x[2] + 0.75*x[3] + 2.5*x[4])

            x = linearizedSubproblem[:x]
            NCNBD.@lin_stageobjective(linearizedSubproblem, x[1] - 0.75*x[2] + 0.75*x[3] + 2.5*x[4])

            # store in ext of subproblem
            nonlinearFunctionList = NCNBD.NonlinearFunction[]
            subproblem.ext[:nlFunctions] = nonlinearFunctionList
            subproblem.ext[:linSubproblem] = linearizedSubproblem

        end
    end

    # SET-UP PARAMETERS
    ############################################################################
    appliedSolvers = NCNBD.AppliedSolvers(Gurobi.Optimizer, Gurobi.Optimizer, GAMS.Optimizer)

    epsilon_outerLoop = 0.01
    epsilon_innerLoop = 0.001
    binaryPrecision = Dict(:b => 2/15)
    plaPrecision = [0, 0]
    sigma = [0.0, 1.0]
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
                time_limit = 6000, stopping_rules = [NCNBD.DeterministicStopping()])

    #@infiltrate

    # WRITE LOGS TO FILE
    ############################################################################
    NCNBD.write_log_to_csv(model, "discont_results.csv", algoParameters)

end

discontExample()
