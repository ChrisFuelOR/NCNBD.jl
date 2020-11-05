using JuMP
using SDDP
using NCNBD
using Revise
using Gurobi

function exampleModel()

    model = SDDP.LinearPolicyGraph(
        stages = 1,
        lower_bound = 0.0,
        optimizer = Gurobi.Optimizer,
        sense = :Min
        #integrality_handler = SDDP.SDDiP()
    ) do subproblem, t

        # DEFINE MILP MODEL
        ############################################################################
        linearizedSubproblem = JuMP.Model()

        # construct a list to store all nonlinear functions
        nonlinearFunctionList = NCNBD.NonlinearFunction[]
        numberOfNonlinearFunctions = 2

        # SET-UP LINEAR PART OF MODEL
        ############################################################################
        for problem in [subproblem, linearizedSubproblem]
            @variable(problem, 0 <= x[i=1:2] <= 4)

            # SET-UP EXPRESSION GRAPH FOR NONLINEARITIES OF MINLP MODEL
            ########################################################################
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
        ############################################################################
        # define nonlinear expressions
        nonlinearexp_1(y) = y^2
        nonlinearexp_2(y) = sqrt(y)

        # register nonlinear expressions
        register(subproblem, :nonlinearexp_1, 1, nonlinearexp_1, autodiff=true)
        register(subproblem, :nonlinearexp_2, 1, nonlinearexp_2, autodiff=true)

        # defining nonlinear constraints using auxiliary variables
        nonlinearAux = subproblem[:nonlinearAux]
        x = subproblem[:x]
        @NLconstraint(subproblem, nlcon_1, nonlinearAux[1] == nonlinearexp_1(x[2]))
        @NLconstraint(subproblem, nlcon_2, nonlinearAux[2] == nonlinearexp_2(x[2]))

        # construct nonlinearFunction objects for both constraints
        nlf_1 = NCNBD.NonlinearFunction(nonlinearexp_1, nonlinearAux[1], nlcon_1, [x[2]])
        nlf_2 = NCNBD.NonlinearFunction(nonlinearexp_2, nonlinearAux[2], nlcon_2, [x[2]])

        # push both nonlinearFunction objects to list
        push!(nonlinearFunctionList, nlf_1)
        push!(nonlinearFunctionList, nlf_2)

        model.nodes[t].ext[:nlFunctions] = nonlinearFunctionList

        # TODOS.
        # 1.) Leere Triangulation erzeugen, wie geht es?
        # 2.) Wie kann ich das linearizedSubproblem abspeichern?
        # Auf den node hat man vermutlich noch keinen Zugriff (Zeile 69).
        # Man könnte es in ext subproblem speichern und das dann später switchen,
        # ist aber auch komisch.


    end

    #appliedSolvers = NCNBD.AppliedSolvers(Gurobi.Optimizer, Gurobi.Optimizer, Scip.Optimizer)

end

exampleModel()