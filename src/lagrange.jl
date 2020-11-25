function _kelley(node::SDDP.Node, dual_vars::Vector{Float64}, integrality_handler::SDDP.SDDiP)

    # INITIALIZATION
    ############################################################################
    atol = integrality_handler.atol
    rtol = integrality_handler.rtol
    model = node.ext[:linSubproblem]
    # Assume the model has been solved. Solving the MIP is usually very quick
    # relative to solving for the Lagrangian duals, so we cheat and use the
    # solved model's objective as our bound while searching for the optimal duals
    @assert JuMP.termination_status(model) == MOI.OPTIMAL
    obj = JuMP.objective_value(model)

    for (i, (name, state_comp)) in enumerate(node.ext[:lin_states])
        integrality_handler.old_rhs[i] = JuMP.fix_value(state_comp.in)
        integrality_handler.slacks[i] = state.in - integrality_handler.old_rhs[i]
        JuMP.unfix(state_comp.in)
        JuMP.unset_binary(state_comp.in) # TODO: maybe not required
        JuMP.set_lower_bound(state_comp.in, 0)
        JuMP.set_upper_bound(state_comp.in, 1)
    end

    # SET-UP APPROXIMATION MODEL
    ############################################################################
    # Subgradient at current solution
    subgradients = integrality_handler.subgradients
    # Best multipliers found so far
    best_mult = integrality_handler.best_mult
    # Dual problem has the opposite sense to the primal
    dualsense = (
        JuMP.objective_sense(model) == JuMP.MOI.MIN_SENSE ? JuMP.MOI.MAX_SENSE :
            JuMP.MOI.MIN_SENSE
    )

    # Approximation of Lagrangian dual as a function of the multipliers
    approx_model = JuMP.Model(appliedSolvers.MILP)

    # Objective estimate and Lagrangian duals
    @variables approx_model begin
        θ
        x[1:length(dual_vars)]
    end
    JuMP.@objective(approx_model, dualsense, θ)

    if dualsense == MOI.MIN_SENSE
        JuMP.set_lower_bound(θ, obj)
        (best_actual, f_actual, f_approx) = (Inf, Inf, -Inf)
    else
        JuMP.set_upper_bound(θ, obj)
        (best_actual, f_actual, f_approx) = (-Inf, -Inf, Inf)
    end

    # DETERMINE AND ADD BOUNDS FOR DUALVARIABLES
    ############################################################################
    #TODO: Determine a norm of B (coefficient matrix of binary expansion)
    # We use the maximum norm here!
    B_norm = 1
    dual_bound = algoParams.sigma * B_norm

    for i in 1:length(dual_vars)
        JuMP.set_lower_bound(x[i], -dual_bound)
        JuMP.set_upper_bound(x[i], dual_bound)
    end

    # CUTTING-PLANE METHOD
    ############################################################################
    iter = 0
    while iter < integrality_handler.iteration_limit
        iter += 1

        # SOLVE LAGRANGIAN RELAXATION FOR GIVEN DUAL_VARS
        ########################################################################
        # Evaluate the real function and a subgradient
        f_actual = _solve_Lagrangian_relaxation!(subgradients, node, dual_vars, integrality_handler.slacks)

        # ADD CUTTING PLANE
        ########################################################################
        # Update the model and update best function value so far
        if dualsense == MOI.MIN_SENSE
            JuMP.@constraint(
                approx_model,
                θ >= f_actual + LinearAlgebra.dot(subgradients, x - dual_vars)
            )
            if f_actual <= best_actual
                best_actual = f_actual
                best_mult .= dual_vars
            end
        else
            JuMP.@constraint(
                approx_model,
                θ <= f_actual + LinearAlgebra.dot(subgradients, x - dual_vars)
            )
            if f_actual >= best_actual
                best_actual = f_actual
                best_mult .= dual_vars
            end
        end

        # SOLVE APPROXIMATION MODEL
        ########################################################################
        # Get a bound from the approximate model
        JuMP.optimize!(approx_model)
        @assert JuMP.termination_status(approx_model) == JuMP.MOI.OPTIMAL
        f_approx = JuMP.objective_value(approx_model)

        # CONVERGENCE CHECK AND UPDATE
        ########################################################################
        # More reliable than checking whether subgradient is zero
        if isapprox(best_actual, f_approx, atol = atol, rtol = rtol)
            dual_vars .= best_mult
            if dualsense == JuMP.MOI.MIN_SENSE
                dual_vars .*= -1
            end
            for (i, (name, state_comp)) in enumerate(node.ext[:lin_states])
                #prepare_state_fixing!(node, state_comp)
                JuMP.fix(state_comp.in, integrality_handler.old_rhs[i], force = true)
            end

            return best_actual
        end
        # Next iterate
        dual_vars .= value.(x)
    end
    error("Could not solve for Lagrangian duals. Iteration limit exceeded.")
end


function _solve_Lagrangian_relaxation!(
    subgradients::Vector{Float64},
    node::Node,
    dual_vars::Vector{Float64},
    slacks,
)
    model = node.ext[:linSubproblem]
    old_obj = JuMP.objective_function(model)
    # Set the Lagrangian relaxation of the objective in the primal model
    fact = (JuMP.objective_sense(model) == JuMP.MOI.MIN_SENSE ? 1 : -1)
    new_obj = old_obj + fact * LinearAlgebra.dot(dual_vars, slacks)
    JuMP.set_objective_function(model, new_obj)
    JuMP.optimize!(model)
    lagrangian_obj = JuMP.objective_value(model)

    # Reset old objective, update subgradients using slack values
    JuMP.set_objective_function(model, old_obj)
    subgradients .= fact .* JuMP.value.(slacks)
    return lagrangian_obj
end