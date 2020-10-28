using JuMP
using Delaunay
using Gurobi
using PiecewiseLinearOpt
using NCNBD

function main()

    # (1) SET-UP THE ORIGINAL MINLP MODEL
    ############################################################################
    MINLPmodel = JuMP.Model()
    @variable(MINLPmodel, 0 <= x[i=1:2] <= 4, base_name="x")
    @objective(MINLPmodel, Min, x[1] + x[2])
    set_optimizer(MINLPmodel, Gurobi.Optimizer)

    # nonlinear functions to be used (first defined, then added using expression graph)
    # --------------------------------------------------------------------------
    # first a list is constructed containing all the nonlinear expressions and function structs
    #nonlinearExpressionList = NonlinearExpression[]
    nonlinearFunctionList = NonlinearFunction[]
    numberOfNonlinearFunctions = 2

    # for each nonlinear term to be approximated, an auxiliary variables is introduced
    # should be multidimensional in general
    @variable(MINLPmodel, nonlinearAux[i=1:numberOfNonlinearFunctions], base_name="nonlinearAux")

    # for each nonlinear constraint, the nonlinear term is replaced by the auxiliary variable
    @constraint(MINLPmodel, actualnlcon_1, x[2] - nonlinearAux[1] <= 0)
    @constraint(MINLPmodel, actualnlcon_2, -x[2] + nonlinearAux[2] <= 0)

    # the original (still linear) model is copied
    # note that this is done this early, since nonlinear models cannot be copied in JuMP
    # note that the variables have to be copied as well to get a reference to the new model
    MILPmodel = JuMP.copy(MINLPmodel)
    x_MILP = copy(x, MILPmodel)

    # defining and registering the nonlinear functions
    # registration is required for using user-defined nonlinear functions in JuMP
    nonlinearexp_1(y) = y^2
    nonlinearexp_2(y) = sqrt(y)
    #nonlinearExpressionList[1] = nonlinearexp_1
    #nonlinearExpressionList[2] = nonlinearexp_2

    register(MINLPmodel, :nonlinearexp_1, 1, nonlinearexp_1, autodiff=true)
    register(MINLPmodel, :nonlinearexp_2, 1, nonlinearexp_2, autodiff=true)
    #for i in nonlinearExpressionList:
        #register(MINLPmodel; Symbol("nonlinearexp_$i"), 1, nonlinearExpressionList[i], autodiff=true)

    # defining the nonlinear constraints by setting the auxiliary variable equal to the nonlinear term
    @NLconstraint(MINLPmodel, nlcon_1, nonlinearAux[1]==nonlinearexp_1(x[2]))
    nlf_1 = NonlinearFunction([x_MILP[2]], nonlinearAux[1], nlcon_1, nonlinearexp_1)
    push!(nonlinearFunctionList, nlf_1)

    @NLconstraint(MINLPmodel, nlcon_2, nonlinearAux[2]==nonlinearexp_2(x[2]))
    nlf_2 = NonlinearFunction([x_MILP[2]], nonlinearAux[2], nlcon_2, nonlinearexp_2)
    push!(nonlinearFunctionList, nlf_2)

    # (2) DEFINE NC-NBD parameters
    ############################################################################
    sigma = 1
    initialSimplices = 5
    initialBinaryPrecision = 0.5
    maxcuts = 100

    algoParams = InitialAlgoParams(sigma, initialSimplices, initialBinaryPrecision, maxucts)

    # (3) START THE NC-NBD METHOD
    ############################################################################
    #startNCNBD(MINLPmodel, MILPmodel, nonlinearFunctionList, algoConfig)



    # get information on model
    #num_variables(MINLPmodel) # number of all variables
    #num_constraints(MINLPmodel) # number of (linear) constraints
    #num_nl_constraints(MINLPmodel) # number of nonlinear constraints
    #all_variables(MINLPmodel) # all variables in model
    #quadr
    #quadr(1)
    #model[:x]
    #model[:nlcon]
    #nlcon

end

#function quadr(
#    y::Float64
#)
#    return y^2
#end


main()
