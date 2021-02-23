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
    NCNBD.piecewiseLinearRelaxation!(node::SDDP.Node, algoParams::AlgoParams, appliedSolvers::AppliedSolvers)

Determines a piecewise linear relaxation for all nonlinear functions in node.

"""

function piecewiseLinearRelaxation!(node::SDDP.Node, plaPrecision::Vector{Float64}, appliedSolvers::NCNBD.AppliedSolvers)

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

        # Get precision
        plaPrecision = plaPrecision[nlIndex]

        # Determine Triangulation
        nlFunction.triangulation = triangulate!(nlFunction, node, plaPrecision)

        # Define overestimation/underestimation problem
        estimationProblem = JuMP.Model()
        set_optimizer(estimationProblem, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>appliedSolvers.NLP, "optcr"=>0.0))
        #set_optimizer(estimationProblem, GAMS.Optimizer)
        #JuMP.set_optimizer_attribute(estimationProblem, "Solver", appliedSolvers.NLP)
        #JuMP.set_optimizer_attribute(estimationProblem, "optcr", 0.0)
        #JuMP.set_silent(estimationProblem)

        # Determine Piecewise Linear Approximation
        piecewiseLinearApproximation!(nlIndex, nlFunction.triangulation, linearizedSubproblem, estimationProblem)
        # Determine number of simplices in triangulation
        number_of_simplices = size(nlFunction.triangulation.simplices, 1)

        @assert nlFunction.refineType == :replace || nlFunction.refineType == :keep
        @assert nlFunction.shift == :shift || nlFunction.shift == :noshift

        # Shift approximation to obtain a relaxation (if required)
        for simplex_index in 1:number_of_simplices
            if nlFunction.shift == :shift
                determineShifts!(simplex_index, nlFunction, estimationProblem, appliedSolvers)
            elseif nlFunction.shift == :noshift
                nlFunction.triangulation.simplices[simplex_index].maxOverestimation = 0
                nlFunction.triangulation.simplices[simplex_index].maxUnderestimation = 0
            end
        end

        # Get dimension
        dimension = size(nlFunction.variablesContained, 1)

        # Add relaxation constraints to linearizedSubproblem
        λ = linearizedSubproblem[:λ]
        e = linearizedSubproblem[:e]

        relax_1 = JuMP.@constraint(linearizedSubproblem, -sum(nlFunction.triangulation.simplices[i].maxOverestimation * sum(λ[i,j] for j in 1:dimension+1) for i in 1:number_of_simplices) <= e)
        relax_2 = JuMP.@constraint(linearizedSubproblem, sum(nlFunction.triangulation.simplices[i].maxUnderestimation * sum(λ[i,j] for j in 1:dimension+1) for i in 1:number_of_simplices) >= e)
        push!(nlFunction.triangulation.plrConstraints, relax_1)
        push!(nlFunction.triangulation.plrConstraints, relax_2)
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

        # pre-allocate storage for simplices
        simplices = Vector{NCNBD.Simplex}(undef, number_of_simplices)

        # add first values to vectors
        xcoord = lower_bound
        func_value = nlFunction.nonlinfunc_eval(xcoord)

        # determine simplices
        for simplexIndex = 1 : number_of_simplices
            # add empty Simplex
            simplices[simplexIndex] = NCNBD.Simplex(Array{Float64,2}(undef, dimension+1, 1), Vector{Float64}(undef, dimension+1), Inf, Inf)

            # add both vertices
            simplices[simplexIndex].vertices[1,1] = xcoord
            xcoord += simplex_length
            simplices[simplexIndex].vertices[2,1] = xcoord

            # add function values
            simplices[simplexIndex].vertice_values[1] = func_value
            func_value =  nlFunction.nonlinfunc_eval(xcoord)
            simplices[simplexIndex].vertice_values[2] = func_value

        end

        @assert isapprox(xcoord, upper_bound, atol=1e-9)

        # set up triangulation
        triangulation = Triangulation(simplices, plaPrecision, JuMP.VariableRef[], JuMP.ConstraintRef[], Dict{Symbol,Any}())

        nlFunction.triangulation = triangulation
        nlFunction.triangulation.ext[:nonlinearFunction] = nlFunction
        nlFunction.triangulation.ext[:node] = node

    # 2D
    ############################################################################
    elseif dimension == 2
        # DETERMINE UNIFORM GRID
        ########################################################################
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
            coord_1 = lower_bound_1 + (ind_1 - 1) * length_points_1
            for ind_2 in 1:number_of_points_2
                coord_2 = lower_bound_2 + (ind_2 - 1) * length_points_2

                ind = ind_2 + (ind_1 - 1) * number_of_points_2
                xgrid[ind, :] = [coord_1 coord_2]
                values_grid[ind] = nlFunction.nonlinfunc_eval(xgrid[ind, 1], xgrid[ind, 2])
            end
        end

        @assert xgrid[number_of_points, 1] == upper_bound_1
        @assert xgrid[number_of_points, 2] == upper_bound_2

        # DETERMINE TRIANGULATION
        ########################################################################
        # determine the simplices using Delaunay package
        simplices_delaunay = Delaunay.delaunay(xgrid).simplices

        # pre-allocate storage for simplices
        simplices = Vector{NCNBD.Simplex}(undef, size(simplices_delaunay,1))

        # CREATE SIMPLICES AND TRIANGULATION IN OUR FORMATS
        ########################################################################
        # create Simplex structs in our format
        for simplex_index in 1:size(simplices_delaunay,1)
            simplices[simplex_index] = NCNBD.Simplex(Array{Float64,2}(undef, dimension+1, 2), Vector{Float64}(undef, dimension+1), Inf, Inf)

            simplex_delaunay = simplices_delaunay[simplex_index, :]
            for i in 1:dimension+1
                simplices[simplex_index].vertices[i, :] = xgrid[simplex_delaunay[i], :]
                simplices[simplex_index].vertice_values[i] = values_grid[simplex_delaunay[i]]
            end

        end

        # set up triangulation
        triangulation = Triangulation(simplices, plaPrecision, JuMP.VariableRef[], JuMP.ConstraintRef[], Dict{Symbol,Any}())

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
        JuMP.set_lower_bound(x_1_est, JuMP.lower_bound(x_1))
        JuMP.set_upper_bound(x_1_est, JuMP.upper_bound(x_1))
        estimationProblem[:x_1_est] = x_1_est

    elseif dimension == 2
        x_1 = triangulation.ext[:nonlinearFunction].variablesContained[1]
        x_2 = triangulation.ext[:nonlinearFunction].variablesContained[2]

        # set up variables required for estimationProblem (we cannot use x_1 here)
        x_1_est = JuMP.@variable(estimationProblem, base_name="x_1_est")
        JuMP.set_lower_bound(x_1_est, JuMP.lower_bound(x_1))
        JuMP.set_upper_bound(x_1_est, JuMP.upper_bound(x_1))
        estimationProblem[:x_1_est] = x_1_est

        # set up variables required for estimationProblem (we cannot use x_1 here)
        x_2_est = JuMP.@variable(estimationProblem, base_name="x_2_est")
        JuMP.set_lower_bound(x_2_est, JuMP.lower_bound(x_2))
        JuMP.set_upper_bound(x_2_est, JuMP.upper_bound(x_2))
        estimationProblem[:x_2_est] = x_2_est
    end

    # variable to encode function value of PLA
    y = JuMP.@variable(linSubproblem, base_name="y_$nlIndex")
    push!(triangulation.plrVariables, y)
    y_est = JuMP.@variable(estimationProblem, base_name="y_est")
    estimationProblem[:y_est] = y_est

    # variable to encode convex combination
    λ = JuMP.@variable(linSubproblem, [i=1:number_of_simplices, j=1:dimension+1], lower_bound=0, upper_bound=1, base_name="λ_$nlIndex")
    append!(triangulation.plrVariables, λ)
    linSubproblem[:λ] = λ
    λ_est = JuMP.@variable(estimationProblem, [i=1:number_of_simplices, j=1:dimension+1], lower_bound=0, upper_bound=1, base_name="λ_est")
    estimationProblem[:λ_est] = λ_est

    # variables for log modeling
    number_log = ceil(Int, log2(number_of_simplices))
    z = JuMP.@variable(linSubproblem, [l=1:number_log], Bin, base_name="z_$nlIndex")
    append!(triangulation.plrVariables, z)
    # note z_est is not binary, as we will fix these values in the estimation problem anyway
    # for the same reason, we do not need bounds 0 and 1 here
    z_est = JuMP.@variable(estimationProblem, [l=1:number_log], base_name="z_est")
    estimationProblem[:z_est] = z_est

    # variable for shift
    e = JuMP.@variable(linSubproblem, base_name="e_$nlIndex")
    linSubproblem[:e] = e
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
    triangulation.ext[:gray_code] = c

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
    yConst = JuMP.@constraint(linSubproblem, sum(λ[i,j] * triangulation.simplices[i].vertice_values[j] for  i in 1:number_of_simplices, j in 1:dimension+1) + e == y )
    push!(triangulation.plrConstraints, yConst)
    yConst_est = JuMP.@constraint(estimationProblem, sum(λ_est[i,j] * triangulation.simplices[i].vertice_values[j] for  i in 1:number_of_simplices, j in 1:dimension+1) == y_est )

    auxVariable = triangulation.ext[:nonlinearFunction].auxVariable
    auxConst = JuMP.@constraint(linSubproblem, auxVariable == y)
    push!(triangulation.plrConstraints, auxConst)

    # original variable encoding
    if dimension == 1
        xConst = JuMP.@constraint(linSubproblem, sum(λ[i,j] * triangulation.simplices[i].vertices[j, 1] for  i in 1:number_of_simplices, j in 1:dimension+1) == x_1 )
        push!(triangulation.plrConstraints, xConst)

        xConst_est = JuMP.@constraint(estimationProblem, sum(λ_est[i,j] * triangulation.simplices[i].vertices[j, 1] for  i in 1:number_of_simplices, j in 1:dimension+1) == x_1_est )
    elseif dimension == 2
        xConst1 = JuMP.@constraint(linSubproblem, sum(λ[i,j] * triangulation.simplices[i].vertices[j, 1] for  i in 1:number_of_simplices, j in 1:dimension+1) == x_1 )
        push!(triangulation.plrConstraints, xConst1)
        xConst1_est = JuMP.@constraint(estimationProblem, sum(λ_est[i,j] * triangulation.simplices[i].vertices[j, 1] for  i in 1:number_of_simplices, j in 1:dimension+1) == x_1_est )

        xConst2 = JuMP.@constraint(linSubproblem, sum(λ[i,j] * triangulation.simplices[i].vertices[j, 2] for  i in 1:number_of_simplices, j in 1:dimension+1) == x_2 )
        push!(triangulation.plrConstraints, xConst2)
        xConst2_est = JuMP.@constraint(estimationProblem, sum(λ_est[i,j] * triangulation.simplices[i].vertices[j, 2] for  i in 1:number_of_simplices, j in 1:dimension+1) == x_2_est )
    end

end


"""
    NCNBD.determineShifts!(simplex_index::Int64, nlfunction::NCNBD.NonlinearFunction,
    estimationProblem::JuMP.Model, appliedSolvers::NCNBD.AppliedSolvers)

Determines an up and down shift based on the maximum overestimation and underestimation errors of the PLA determined before.
Return both shifts in a vector.

"""

function determineShifts!(simplex_index::Int64, nlfunction::NCNBD.NonlinearFunction, estimationProblem::JuMP.Model, appliedSolvers::NCNBD.AppliedSolvers)

    # DEFINE SIMPLEX CONSTRAINTS BY SETTING LOG VARIABLES TO SPECIFIC GRAY CODE
    ############################################################################
    # log number
    number_log = ceil(Int, log2(size(nlfunction.triangulation.simplices, 1)))

    # Get z variable
    z = estimationProblem[:z_est]

    # set values of z to gray_code of given simplex
    for l = 1:number_log
        JuMP.fix(z[l], nlfunction.triangulation.ext[:gray_code][simplex_index][l], force=true)
    end
    # ; force = true

    # INITIALIZATION
    ############################################################################
    # dimension
    dimension = size(nlfunction.variablesContained, 1)
    # required variable representing PLA function value
    y_est = estimationProblem[:y_est]

    # variableRefs to insert into nonlinear expression
    # variablesContained = nlfunction.variablesContained
    # note that we cannot insert nlfunction.variablesContained, since those variables
    # belong to the linearized subproblem. We have to use x_1_est and x_2_est

    if dimension == 1
        x_1 = estimationProblem[:x_1_est]
        nonlinearobj = nlfunction.nonlinfunc_exp(x_1)
    elseif dimension == 2
        x_1 = estimationProblem[:x_1_est]
        x_2 = estimationProblem[:x_2_est]
        nonlinearobj = nlfunction.nonlinfunc_exp(x_1, x_2)
    end

    # DETERMINE AND SOLVE MAXIMUM OVERESTIMATION PROBLEM
    ############################################################################
    # nonlinear expression describing the approximated function
    JuMP.set_NL_objective(estimationProblem, MathOptInterface.MAX_SENSE, :($(y_est) - $(nonlinearobj)))

    #println("###################################################################")
    #println("Simplex: ", simplex_index)
    #println("Max overestimation error")
    #println(estimationProblem)

    JuMP.optimize!(estimationProblem)
    # TODO: Check if globally optimal solution
    overestimation = JuMP.objective_value(estimationProblem)

    #println("optimal x: ", JuMP.value(estimationProblem[:x_1_est]))
    #println("optimal y: ", JuMP.value(estimationProblem[:y_est]))
    #println("optimal value: ", overestimation)

    # x_opt = JuMP.value(estimationProblem[:x_1_est])
    # y_opt = JuMP.value(estimationProblem[:y_est])
    # l_opt_1 = JuMP.value(estimationProblem[:λ_est][simplex_index,1])
    # l_opt_2 = JuMP.value(estimationProblem[:λ_est][simplex_index,2])
    # l_opt_r = JuMP.value(estimationProblem[:λ_est][simplex_index+1,1])
    # println(l_opt_1)
    # println(l_opt_2)
    # println(l_opt_r)

    # vertex_indices = nlfunction.triangulation.simplices[simplex_index, :]
    # x_val = 0
    # y_val = 0
    #
    # for i in 1:dimension+1
    #     vertex_index = vertex_indices[i]
    #     vertex = nlfunction.triangulation.vertices[vertex_index]
    #     vertex_value = nlfunction.triangulation.verticeValues[vertex_index]
    #     x_val += JuMP.value(estimationProblem[:λ_est][simplex_index,i]) * vertex
    #     y_val += JuMP.value(estimationProblem[:λ_est][simplex_index,i]) * vertex_value
    # end
    # println("calculated x: ", x_val)
    # println("calculated y: ", y_val)

    # DETERMINE AND SOLVE MAXIMUM UNDERESTIMATION PROBLEM
    ############################################################################
    JuMP.set_NL_objective(estimationProblem, MathOptInterface.MAX_SENSE, :($(nonlinearobj) - $(y_est)))
    # println("Max underestimation error")
    # println(estimationProblem)


    JuMP.optimize!(estimationProblem)
    # TODO: Check if globally optimal solution
    underestimation = JuMP.objective_value(estimationProblem)

    # println("optimal x: ", JuMP.value(estimationProblem[:x_1_est]))
    # println("optimal y: ", JuMP.value(estimationProblem[:y_est]))
    # println("optimal value: ", underestimation)
    # println("###################################################################")

    # UNFIX VARIABLES AGAIN
    ############################################################################
    for l = 1:number_log
        JuMP.unfix(z[l])
    end

    # STORE ESTIMATION ERRORS
    ############################################################################
    nlfunction.triangulation.simplices[simplex_index].maxOverestimation = overestimation
    nlfunction.triangulation.simplices[simplex_index].maxUnderestimation = underestimation

end

function piecewise_linear_refinement(model::SDDP.PolicyGraph{T}, appliedSolvers::NCNBD.AppliedSolvers) where{T}
    # ITERATE OVER NODES
    ############################################################################
    for (node_index, children) in model.nodes
        node = model.nodes[node_index]

        # MILP subproblem
        linearizedSubproblem = node.ext[:linSubproblem]

        # ITERATE OVER NLFUNCTIONS
        ########################################################################
        for nlIndex in 1:size(node.ext[:nlFunctions],1)
            # Get nonlinear function
            nlFunction = node.ext[:nlFunctions][nlIndex]

            # EXTRACT CURRENT SOLUTION AND LOCALIZE SIMPLEX
            ####################################################################
            # Variables to be considered
            variablesContained = nlFunction.variablesContained

            # Determine dimension
            dimension = size(variablesContained, 1)

            # Optimal component
            if dimension == 1
                optpoint = nlFunction.ext[:optSolution][variablesContained[1]]
            elseif dimension == 2
                optpoint = [nlFunction.ext[:optSolution][variablesContained[1]], nlFunction.ext[:optSolution][variablesContained[2]]]
            end

            # store new simplices
            new_simplex_indices_list = Int64[]
            simplices_to_refine_list = Int64[]

            # Iterate over all simplices of current triangulation
            for simplex_index in 1:size(nlFunction.triangulation.simplices,1)
                simplex = nlFunction.triangulation.simplices[simplex_index]

                # check if the point is located in this simplex
                if dimension == 1
                    interval_check = NCNBD.pointInInterval(simplex.vertices, optpoint)
                elseif dimension == 2
                    interval_check = NCNBD.pointInTriangle(simplex.vertices, optpoint)
                end

                # if yes, then add this simplex to the simplices to be refined
                # (note that we cannot break here, since the optimal point may
                # (be contained in several simplices)
                if interval_check
                    push!(simplices_to_refine_list, simplex_index)
                end
            end

            # Refine all simplices in simplices_to_refine_list
            for simplex_index in simplices_to_refine_list
                # divide simplex by longest edge and construct two new ones
                new_simplex_indices = NCNBD.divide_simplex_by_longest_edge!(simplex_index, nlFunction.triangulation)
                # append to list of new simplices
                append!(new_simplex_indices_list, new_simplex_indices)
                # delete old simplex
                deleteat!(nlFunction.triangulation.simplices, simplex_index)

                # adapt the indices of the new simplices accordingly
                for i in 1:size(new_simplex_indices_list,1)
                    new_index = new_simplex_indices_list[i]
                    if new_index > simplex_index
                        new_simplex_indices_list[i] -= 1
                    end
                end

                # adapt the indices of the remaining simplices to be refined accordingly
                for i in 1:size(simplices_to_refine_list,1)
                    refine_index = simplices_to_refine_list[i]
                    if refine_index > simplex_index
                        simplices_to_refine_list[i] -= 1
                    end
                end
            end

            # DELETE PREVIOUS PIECEWISE LINEAR APPROXIMATION
            ####################################################################
            if nlFunction.refineType == :replace
                delete(linearizedSubproblem, nlFunction.triangulation.plrVariables)
                for constraint in nlFunction.triangulation.plrConstraints
                    delete(linearizedSubproblem, constraint)
                end
            end
            nlFunction.triangulation.plrVariables = JuMP.VariableRef[]
            nlFunction.triangulation.plrConstraints = JuMP.ConstraintRef[]

            # CREATE A NEW PIECEWISE LINEAR APPROXIMATION
            ####################################################################
            # Define overestimation/underestimation problem
            estimationProblem = JuMP.Model()
            set_optimizer(estimationProblem, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>appliedSolvers.NLP, "optcr"=>0.0))
            #set_optimizer(estimationProblem, GAMS.Optimizer)
            #JuMP.set_optimizer_attribute(estimationProblem, "Solver", appliedSolvers.NLP)
            #JuMP.set_optimizer_attribute(estimationProblem, "optcr", 0.0)
            #JuMP.set_silent(estimationProblem)

            piecewiseLinearApproximation!(nlIndex, nlFunction.triangulation, linearizedSubproblem, estimationProblem)

            # DETERMINE ESTIMATION ERRORS
            ####################################################################
            # Note that this is only required for the new simplices here,
            # since the other approximations essentially did not change
            # Shift approximation to obtain a relaxation (if required)
            for simplex_index in new_simplex_indices_list
                if nlFunction.shift == :shift
                    determineShifts!(simplex_index, nlFunction, estimationProblem, appliedSolvers)
                elseif nlFunction.shift == :noshift
                    nlFunction.triangulation.simplices[simplex_index].maxOverestimation = 0
                    nlFunction.triangulation.simplices[simplex_index].maxUnderestimation = 0
                end
            end

            # CREATE RELAXATION
            ####################################################################
            # Get dimension
            dimension = size(nlFunction.variablesContained, 1)

            # Add relaxation constraints to linearizedSubproblem
            λ = linearizedSubproblem[:λ]
            e = linearizedSubproblem[:e]

            number_of_simplices = size(nlFunction.triangulation.simplices, 1)
            relax_1 = JuMP.@constraint(linearizedSubproblem, -sum(nlFunction.triangulation.simplices[i].maxOverestimation * sum(λ[i,j] for j in 1:dimension+1) for i in 1:number_of_simplices) <= e)
            relax_2 = JuMP.@constraint(linearizedSubproblem, sum(nlFunction.triangulation.simplices[i].maxUnderestimation * sum(λ[i,j] for j in 1:dimension+1) for i in 1:number_of_simplices) >= e)
            push!(nlFunction.triangulation.plrConstraints, relax_1)
            push!(nlFunction.triangulation.plrConstraints, relax_2)

        end
    end
end
