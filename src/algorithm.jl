# Copyright Christian Füllner (Karlsruhe Institute of Technology) 2020
#
# This source code form is subject to the terms of the Mozilla Public License, v. x.x.
# If a copy of the MPL was not distributed with this file, you can obtain one
# at https://mozilla.org/MPL/x.x.

# This file is inspired by and re-uses parts from the source code of
# SDDiP.jl (lkapelevich),
# SLDP.jl (bfpc)
# and especially SDDP.jl (odow).


"""
    NCNBD.piecewiseLinearRelaxation!(node::SDDP.Node, algoParams::AlgoParams)

Determines a piecewise linear relaxation for all nonlinear functions in node.

"""

function piecewiseLinearRelaxation!(node::SDDP.Node, algoParams::AlgoParams)

    # MILP subproblem
    linearizedSubproblem = node.ext[:linearizedSubproblem]

    # LOOP OVER ALL NONLINEAR FUNCTIONS AND CALL MORE SPECIFIC FUNCTIONS
    ############################################################################
    for nlFunction in node.ext[:nlFunctions]

        # Determine Triangulation
        nlFunction.triangulation = triangulate(nlFunction, node, AlgoParams)
        triang = nlFunction.triangulation

        # Determine Piecewise Linear Approximation
        piecewiseLinearApproximation!(triang, linearizedSubproblem)

        # Shift approximation to obtain a relaxation
        for simplex in triang.simplices
            shiftApproximation(simplex, triang, linearizedSubproblem)
        end
    end

end


"""
    NCNBD.piecewiseLinearRelaxation!(node::SDDP.Node, algoParams::AlgoParams)

Determines a piecewise linear relaxation for all nonlinear functions in node.

"""

function triangulate!(node::SDDP.Node, algoParams::AlgoParams)
    print("triangulate")
end










#for nonlinfunc in nonlinearFunctionList
# Determine the variable bounds
# Determine a triangulation
#xgrid = Vector(lower_bound(nlf1.variableList[1]): upper_bound(nlf1.variableList[1]))

# Determine the function values of all vertices
#ygrid = quadr.(xgrid)

# Determine the piecewise linear approximation
#z = piecewiselinear(MILPmodel, x_MILP[2], xgrid, ygrid; method=:DLog)

#println(all_variables(MINLPmodel))
#println(MINLPmodel)

#println(all_variables(MILPmodel))
#println(MILPmodel)

#λ


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
#end
