# Copyright Christian Füllner (Karlsruhe Institute of Technology) 2020
#
# This source code form is subject to the terms of the Mozilla Public License, v. x.x.
# If a copy of the MPL was not distributed with this file, you can obtain one
# at https://mozilla.org/MPL/x.x.

# This file is inspired by and re-uses parts from the source code of
# SDDiP.jl (lkapelevich),
# SLDP.jl (bfpc)
# and especially SDDP.jl (odow)
# and piecewiseLinearOpt.jl (joehuchette).


"""
    NCNBD.piecewiseLinearRelaxation!(node::SDDP.Node, algoParams::AlgoParams)

Determines a piecewise linear relaxation for all nonlinear functions in node.

"""

function piecewiseLinearRelaxation!(node::SDDP.Node, plaPrecision::Float64)

    # MILP subproblem
    linearizedSubproblem = node.ext[:linSubproblem]

    # TODO: Alternative: Define dicts for lists for storing variable
    # and constraint references in linearizedSubproblem.ext.
    #linearizedSubproblem.ext[:varReferences]
    #linearizedSubproblem.ext[:constrReferences]

    # LOOP OVER ALL NONLINEAR FUNCTIONS AND CALL MORE SPECIFIC FUNCTIONS
    ############################################################################
    for nlIndex in 1:size(node.ext[:nlFunctions],1)
        # Get nonlinear function
        nlFunction = node.ext[:nlFunctions][nlIndex]

        # Determine Triangulation
        nlFunction.triangulation = triangulate!(nlFunction, node, plaPrecision)

        # Define overestimation/underestimation problem
        estimationProblem = JuMP.Model()

        # Determine Piecewise Linear Approximation
        piecewiseLinearApproximation!(nlIndex, nlFunction.triangulation, linearizedSubproblem, estimationProblem)

        # Shift approximation to obtain a relaxation
        #for simplex in triang.simplices
        #    shiftApproximation(simplex, nlFunction.triangulation, linearizedSubproblem)
        #end
    end

end


"""
    NCNBD.piecewiseLinearRelaxation!(node::SDDP.Node, plaPrecision::Float64)

Determines a piecewise linear relaxation for all nonlinear functions in node.

"""

function triangulate!(nlFunction::NCNBD.NonlinearFunction, node::SDDP.Node, plaPrecision::Float64)

    # CHECK FOR ONE- OR TWO-DIMENSIONAL CASE
    ############################################################################
    dimension = size(nlFunction.variablesContained, 1)

    # 1D
    ############################################################################
    if dimension == 1
        # get interval to be considered
        lower_bound = JuMP.lower_bound(nlFunction.variablesContained[1])
        upper_bound = JuMP.upper_bound(nlFunction.variablesContained[1])
        interval_length = upper_bound - lower_bound

        # determine uniform grid based on user-given precision
        # note: if precision is no exact divisor of interval_length, the precision
        # is decreased to obtain a uniform grid
        number_of_simplices = ceil(Int64, interval_length / plaPrecision)
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
            xgrid[1+simplexIndex] = lower_bound + simplexIndex * simplex_length
            # add function value
            values_grid[1+simplexIndex] = nlFunction.nonlinearExpression(xgrid[simplexIndex])
            # add simplex
            simplices[simplexIndex, :] = [simplexIndex, simplexIndex+1]
        end

        @assert xgrid[number_of_simplices + 1] == upper_bound

        # initialize maxOverestimation and maxUnderestimation
        maxOverestimation = fill(Inf, number_of_simplices)
        maxUnderestimation = fill(Inf, number_of_simplices)

        # set up triangulation
        triangulation = Triangulation(xgrid, values_grid, simplices, plaPrecision, JuMP.VariableRef[], JuMP.ConstraintRef[], maxOverestimation, maxUnderestimation, Dict{Symbol,Any}())

        nlFunction.triangulation = triangulation
        nlFunction.triangulation.ext[:nonlinearFunction] = nlFunction
        nlFunction.triangulation.ext[:node] = node

    # 2D
    ############################################################################
    elseif dimension == 2
        # get interval to be considered
        lower_bound_1 = JuMP.lower_bound(nlFunction.variablesContained[1])
        upper_bound_1 = JuMP.upper_bound(nlFunction.variablesContained[1])
        interval_length_1 = upper_bound_1 - lower_bound_1

        lower_bound_2 = JuMP.lower_bound(nlFunction.variablesContained[2])
        upper_bound_2 = JuMP.upper_bound(nlFunction.variablesContained[2])
        interval_length_2 = upper_bound_2 - lower_bound_2

        # determine uniform grid based on user-given precision
        # note: if precision is no exact divisor of interval_length, the precision
        # is decreased to obtain a uniform grid
        number_of_points_1 = ceil(Int64, interval_length_1 / plaPrecision) + 1
        number_of_points_2 = ceil(Int64, interval_length_2 / plaPrecision) + 1
        number_of_points = number_of_points_1 * number_of_points_2

        # length of grid elements
        length_points_1 = interval_length_1 / (number_of_points_1 - 1)
        length_points_2 = interval_length_2 / (number_of_points_2 - 1)

        # pre-allocate storage for simplices and points
        xgrid = Array{Float64}(undef, number_of_points, 2)
        values_grid = Array{Float64}(undef, number_of_points)

        # determine all points in the grid
        for ind_1 in 1:number_of_points_1
            for ind_2 in 1:number_of_points_2
                coord_1 = lower_bound_1 + (ind_1 - 1) * length_points_1
                coord_2 = lower_bound_2 + (ind_2 - 1) * length_points_2

                ind = ind_2 + (ind_1 - 1) * number_of_points_2
                xgrid[ind, :] = [coord_1 coord_2]
                values_grid[ind] = nlFunction.nonlinearExpression(xgrid[ind, 1], xgrid[ind, 2])
            end
        end

        @assert xgrid[number_of_points, 1] == upper_bound_1
        @assert xgrid[number_of_points, 2] == upper_bound_2

        # determine the simplices
        simplices = Delaunay.delaunay(xgrid).simplices

        # set up triangulation
        triangulation = Triangulation(xgrid, values_grid, simplices, plaPrecision, JuMP.VariableRef[], JuMP.ConstraintRef[], Float64[], Float64[], Dict{Symbol,Any}())

        nlFunction.triangulation = triangulation
        nlFunction.triangulation.ext[:nonlinearFunction] = nlFunction
        nlFunction.triangulation.ext[:node] = node

    # OTHER CASES
    ############################################################################
    else
        Print("TODO: Error message and stopping")
        # throw ErrorException(dimension, "Nonlinearities have to be one- or two-dimensional.")
    end

    return triangulation

end

"""
    NCNBD.piecewiseLinearApproximation!(nlIndex::Int64, triangulation::NCNBD.Triangulation, linSubproblem::JuMP.Model, estimationProblem::JuMP.Model)

Determines an MILP model for the piecewise linear approximation given by the triangulation.
Currently, only allows for a disaggregated logarithmic convex combination model.
Note that also the over-/underestimation problem is set up using these constraints.

"""

function piecewiseLinearApproximation!(nlIndex::Int64, triangulation::NCNBD.Triangulation, linSubproblem::JuMP.Model, estimationProblem::JuMP.Model)

    # GET REQUIRED PARAMETES
    ############################################################################
    number_of_simplices = size(triangulation.simplices, 1)
    dimension = size(triangulation.ext[:nonlinearFunction].variablesContained, 1)

    @assert dimension <= 2
    @assert dimension >= 1

    # ADD VARIABLES
    ############################################################################
    # variables associated with nonlinear function
    if dimension == 1
        x_1 = triangulation.ext[:nonlinearFunction].variablesContained[1]

        # set up variables required for estimationProblem (we cannot use x_1 here)
        x_1_est = JuMP.@variable(estimationProblem, base_name="x_1_est")
        JuMP.set_lower_bound(x_1_est, lower_bound(x_1))
        JuMP.set_upper_bound(x_1_est, upper_bound(x_1))

    elseif dimension == 2
        x_1 = triangulation.ext[:nonlinearFunction].variablesContained[1]
        x_2 = triangulation.ext[:nonlinearFunction].variablesContained[2]

        # set up variables required for estimationProblem (we cannot use x_1 here)
        x_1_est = JuMP.@variable(estimationProblem, base_name="x_1_est")
        JuMP.set_lower_bound(x_1_est, lower_bound(x_1))
        JuMP.set_upper_bound(x_1_est, upper_bound(x_1))

        # set up variables required for estimationProblem (we cannot use x_1 here)
        x_2_est = JuMP.@variable(estimationProblem, base_name="x_2_est")
        JuMP.set_lower_bound(x_2_est, lower_bound(x_2))
        JuMP.set_upper_bound(x_2_est, upper_bound(x_2))
    end

    # TODO: Pre-allocation instead of push. But then I have to determine
    # how many variables/constraints have to be added.

    # variable to encode function value of PLA
    y = JuMP.@variable(linSubproblem, base_name="y_$nlIndex")
    push!(triangulation.plrVariables, y)
    y_est = JuMP.@variable(estimationProblem, base_name="y_est")

    # get auxVariable to encode function value of PLA
    auxVar = triangulation.ext[:nonlinearFunction].auxVariable

    # variable to encode convex combination
    λ = JuMP.@variable(linSubproblem, [i=1:number_of_simplices, j=1:dimension+1], lower_bound=0, upper_bound=1, base_name="λ_$nlIndex")
    append!(triangulation.plrVariables, λ)
    λ_est = JuMP.@variable(estimationProblem, [i=1:number_of_simplices, j=1:dimension+1], lower_bound=0, upper_bound=1, base_name="λ_est")

    # variables for log modeling
    number_log = ceil(Int, log2(number_of_simplices))
    z = JuMP.@variable(linSubproblem, [l=1:number_log], Bin, base_name="z_$nlIndex")
    append!(triangulation.plrVariables, z)
    z_est = JuMP.@variable(estimationProblem, [l=1:number_log], Bin, base_name="z_est")

    # variable for shift
    e = JuMP.@variable(linSubproblem, base_name="e_$nlIndex")
    push!(triangulation.plrVariables, e)

    # ADD CONSTRAINTS
    ############################################################################
    # constraints to identify simplex
    # sum of convex coefficients must be 1
    convexSum = JuMP.@constraint(linSubproblem, sum(λ) == 1)
    push!(triangulation.plrConstraints, convexSum)
    convexSum_est = JuMP.@constraint(estimationProblem, sum(λ_est) == 1)

    # reflected gray codes provide the unique identification of the logarithmic binary encoding
    c = PiecewiseLinearOpt.reflected_gray_codes(number_log)

    # log modeling constraints
    for l in 1:number_log
        exp = JuMP.@expression(linSubproblem, sum(c[i][l] * λ[i,j] for i in 1:number_of_simplices, j in 1:dimension + 1) - z[l])
        logConst1 = JuMP.@constraint(linSubproblem, exp <= 0)
        exp2 = JuMP.@expression(linSubproblem, sum((1-c[i][l])* λ[i,j] for i in 1:number_of_simplices, j in 1:dimension + 1) - 1 + z[l])
        logConst2 = JuMP.@constraint(linSubproblem, exp2 <= 0)
        push!(triangulation.plrConstraints, logConst1)
        push!(triangulation.plrConstraints, logConst2)

        exp_est = JuMP.@expression(estimationProblem, sum(c[i][l] * λ_est[i,j] for i in 1:number_of_simplices, j in 1:dimension + 1) - z_est[l])
        logConst1_est = JuMP.@constraint(estimationProblem, exp_est <= 0)
        exp2_est = JuMP.@expression(estimationProblem, sum((1-c[i][l])* λ_est[i,j] for i in 1:number_of_simplices, j in 1:dimension + 1) - 1 + z_est[l])
        logConst2_est = JuMP.@constraint(estimationProblem, exp2_est <= 0)
    end

    # function value encoding
    yConst = JuMP.@constraint(linSubproblem, sum(λ[i,j] * triangulation.verticeValues[triangulation.simplices[i,j]] for  i in 1:number_of_simplices, j in 1:dimension+1) + e == y )
    push!(triangulation.plrConstraints, yConst)
    yConst_est = JuMP.@constraint(estimationProblem, sum(λ_est[i,j] * triangulation.verticeValues[triangulation.simplices[i,j]] for  i in 1:number_of_simplices, j in 1:dimension+1) == y_est )

    auxVarConst = JuMP.@constraint(linSubproblem, auxVar == y)
    push!(triangulation.plrConstraints, auxVarConst)

    # original variable encoding
    if dimension == 1
        xConst = JuMP.@constraint(linSubproblem, sum(λ[i,j] * triangulation.vertices[triangulation.simplices[i,j]] for  i in 1:number_of_simplices, j in 1:dimension+1) == x_1 )
        push!(triangulation.plrConstraints, xConst)

        xConst_est = JuMP.@constraint(estimationProblem, sum(λ_est[i,j] * triangulation.vertices[triangulation.simplices[i,j]] for  i in 1:number_of_simplices, j in 1:dimension+1) == x_1_est )
    elseif dimension == 2
        xConst1 = JuMP.@constraint(linSubproblem, sum(λ[i,j] * triangulation.vertices[triangulation.simplices[i,j], 1] for  i in 1:number_of_simplices, j in 1:dimension+1) == x_1 )
        push!(triangulation.plrConstraints, xConst1)
        xConst1_est = JuMP.@constraint(estimationProblem, sum(λ_est[i,j] * triangulation.vertices[triangulation.simplices[i,j], 1] for  i in 1:number_of_simplices, j in 1:dimension+1) == x_1_est )

        xConst2 = JuMP.@constraint(linSubproblem, sum(λ[i,j] * triangulation.vertices[triangulation.simplices[i,j], 2] for  i in 1:number_of_simplices, j in 1:dimension+1) == x_2 )
        push!(triangulation.plrConstraints, xConst2)
        xConst2_est = JuMP.@constraint(estimationProblem, sum(λ_est[i,j] * triangulation.vertices[triangulation.simplices[i,j], 2] for  i in 1:number_of_simplices, j in 1:dimension+1) == x_2_est )
        
    end

end
