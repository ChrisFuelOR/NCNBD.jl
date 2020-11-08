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
    linearizedSubproblem = node.ext[:linSubproblem]

    # LOOP OVER ALL NONLINEAR FUNCTIONS AND CALL MORE SPECIFIC FUNCTIONS
    ############################################################################
    for nlFunction in node.ext[:nlFunctions]

        # Determine Triangulation
        nlFunction.triangulation = triangulate!(nlFunction, node, algoParams)
        triang = nlFunction.triangulation

        # Determine Piecewise Linear Approximation
        #piecewiseLinearApproximation!(triang, linearizedSubproblem)

        # Shift approximation to obtain a relaxation
        #for simplex in triang.simplices
        #    shiftApproximation(simplex, triang, linearizedSubproblem)
        #end
    end

end


"""
    NCNBD.piecewiseLinearRelaxation!(node::SDDP.Node, algoParams::AlgoParams)

Determines a piecewise linear relaxation for all nonlinear functions in node.

"""

function triangulate!(nlFunction::NCNBD.NonlinearFunction, node::SDDP.Node, algoParams::AlgoParams)

    # CHECK FOR ONE- OR TWO-DIMENSIONAL CASE
    ############################################################################
    dimension = size(nlFunction.variablesContained, 1)

    # 1D
    ############################################################################
    if dimension == 1
        # get interval to be considered
        lower_bound = JuMP.lower_bound(variablesContained[1])
        upper_bound = JuMP.upper_bound(variablesContained[1])
        interval_length = upper_bound - lower_bound

        # determine uniform grid based on user-given precision
        # note: if precision is no exact divisor of interval_length, the precision
        # is decreased to obtain a uniform grid
        number_of_simplices = ceil(interval_length / algoParams.plaPrecision)
        simplex_length = interval_length / number_of_simplices

        # pre-allocate storage for simplices and points
        simplices = Array{Int64}(undef, number_of_simplices, 2)
        xgrid = Array{Float64}(undef, number_of_simplices + 1)
        values_grid = Array{Float64}(undef, number_of_simplices + 1)

        # add first values to vectors
        xgrid[1] = lower_bound
        values_grid[1] = nlFunction.nonlinearExpression(xgrid[1])

        # TODO: Maybe it is more efficient to use a dict here, with a
        # vertex index as key and a tuple of point and values as content.
        # At least, in such case, the points and function values were always
        # related to each other.

        # determine simplices
        for simplexIndex = 1 : number_of_simplices
            # add next breakpoint
            xgrid[simplexIndex] = lower_bound + simplexIndex * simplex_length
            # add function value
            values_grid[simplexIndex] = nlFunction.nonlinearExpression(xgrid[simplexIndex])
            # add simplex
            simplices[simplexIndex, :] = [simplexIndex-1, simplexIndex]
        end

        @assert xgrid[number_of_simplices + 1] = upper_bound

        # set up triangulation
        triangulation = Triangulation(xgrid, values_grid, simplices,
        algoParams.plaPrecision, JuMP.VariableRef[], JuMP.ConstraintRef[],
        Float64[], Float64[])

        nlFunction.triangulation = triangulation

    # 2D
    ############################################################################
    elseif dimension == 2
        # get interval to be considered
        lower_bound_1 = JuMP.lower_bound(variablesContained[1])
        upper_bound_1 = JuMP.upper_bound(variablesContained[1])
        interval_length_1 = upper_bound_1 - lower_bound_1

        lower_bound_2 = JuMP.lower_bound(variablesContained[2])
        upper_bound_2 = JuMP.upper_bound(variablesContained[2])
        interval_length_2 = upper_bound_2 - lower_bound_2

        # determine uniform grid based on user-given precision
        # note: if precision is no exact divisor of interval_length, the precision
        # is decreased to obtain a uniform grid
        number_of_points_1 = ceil(interval_length_1 / algoParams.plaPrecision) + 1
        number_of_points_2 = ceil(interval_length_2 / algoParams.plaPrecision) + 1
        number_of_points = number_of_points_1 * number_of_points_2

        # length of grid elements
        length_points_1 = interval_length_1 / (number_of_points_1 - 1)
        length_points_2 = interval_length_2 / (number_of_points_2 - 1)

        # pre-allocate storage for simplices and points
        xgrid = Array{Float64}(undef, number_of_points, 2)
        values_grid = Array{Float64}(undef, number_of_points)

        # determine all points in the grid
        for ind_1 in number_of_points_1
            for ind_2 in number_of_points_2
                coord_1 = lower_bound_1 + (ind_1 - 1) * length_points_1
                coord_2 = lower_bound_2 + (ind_2 - 1) * length_points_2

                ind = ind_2 + (ind_1 - 1) * number_of_points_2
                xgrid[ind, :] = [coord_1 coord_2]
                values_grid = nlFunction.nonlinearExpression(xgrid[ind, 1], xgrid[ind, 2])
            end
        end

        @assert xgrid[number_of_points, 1] = upper_bound_1
        @assert xgrid[number_of_points, 2] = upper_bound_2

        # determine the simplices
        simplices = Delaunay.delaunay(xgrid).simplices

        # set up triangulation
        triangulation = Triangulation(xgrid, values_grid, simplices,
        algoParams.plaPrecision, JuMP.VariableRef[], JuMP.ConstraintRef[],
        Float64[], Float64[])

        nlFunction.triangulation = triangulation

    # OTHER CASES
    ############################################################################
    else
        print("blubb")
        #throw ErrorException("Nonlinearities have to be one- or two-dimensional.")
    end





end










#for nonlinfunc in nonlinearFunctionList
# Determine the variable bounds
# Determine a triangulation


# Determine the function values of all vertices
#ygrid = quadr.(xgrid)

# Determine the piecewise linear approximation
#z = piecewiselinear(MILPmodel, x_MILP[2], xgrid, ygrid; method=:DLog)

#println(all_variables(MINLPmodel))
#println(MINLPmodel)

#println(all_variables(MILPmodel))
#println(MILPmodel)

#λ
