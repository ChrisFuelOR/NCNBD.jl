function startNCNBD(MINLPmodel::JuMP.model, MILPmodel, nonlinearFunctionList, algoConfig)






MILPmodel = constructMILP(MILPmodel, MINLPmodel, nonlinearFunctionList)







for nonlinfunc in nonlinearFunctionList
    # Determine the variable bounds
    # Determine a triangulation
    xgrid = Vector(lower_bound(nlf1.variableList[1]): upper_bound(nlf1.variableList[1]))

    # Determine the function values of all vertices
    ygrid = quadr.(xgrid)

    # Determine the piecewise linear approximation
    z = piecewiselinear(MILPmodel, x_MILP[2], xgrid, ygrid; method=:DLog)

    println(all_variables(MINLPmodel))
    println(MINLPmodel)

    println(all_variables(MILPmodel))
    println(MILPmodel)

    λ


    # Deleting all variables from previous PLA




    # Construct an initial triangulation
        # if the function is one-dimensional only, then we can just insert a breakpoint in the middle
        # otherwise we can use the Delaunay package to derive an initial triangulation
            # output of the Delaunay package is an array for the simplices (mesh.simplices)
            # each row refers to one simplex, so we can easily determine how many simplices there are
            # each entry in a row refers to one vertex
            # we have to evaluate the nonlinfunc at each of the vertices
            # probably we can simply store the function values in an array with the same shape, e.g. ygrid = quadr.(xgrid)
            # if we do not insist to use the generalized incremental model, we could use
            # Huchette's PiecewiseLinearOpt package, which provides different modeling techniques
            # for univariate and bivariate functions
            # however, in this package no relaxation is determined, so this additional part has to be still deldimplemented by myself


    # Set up a piecewise linear function

    # Should this be done directly in the constructor
end
