function _kelley(
    node::SDDP.Node,
    node_index::Int64,
    obj::Float64,
    dual_vars::Vector{Float64},
    integrality_handler::SDDP.SDDiP,
    algoParams::NCNBD.AlgoParams,
    appliedSolvers::NCNBD.AppliedSolvers,
    dual_bound::Union{Float64,Nothing}
    )

    # INITIALIZATION
    ############################################################################
    atol = integrality_handler.atol
    rtol = integrality_handler.rtol
    model = node.ext[:linSubproblem]
    # Assume the model has been solved. Solving the MIP is usually very quick
    # relative to solving for the Lagrangian duals, so we cheat and use the
    # solved model's objective as our bound while searching for the optimal duals

    # This does not work since the problem has been changed since then
    #assert JuMP.termination_status(model) == MOI.OPTIMAL
    #obj = JuMP.objective_value(model)

    for (i, (name, bin_state)) in enumerate(node.ext[:backward_data][:bin_states])
        integrality_handler.old_rhs[i] = JuMP.fix_value(bin_state)
        integrality_handler.slacks[i] = bin_state - integrality_handler.old_rhs[i]
        JuMP.unfix(bin_state)
        #JuMP.unset_binary(state_comp.in) # TODO: maybe not required
        JuMP.set_lower_bound(bin_state, 0)
        JuMP.set_upper_bound(bin_state, 1)
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
    approx_model = JuMP.Model(Gurobi.Optimizer)
    set_optimizer(approx_model, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>appliedSolvers.MILP, "optcr"=>0.0))

    #JuMP.set_silent(approx_model)

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
        #JuMP.set_upper_bound(θ, 10000.0)

        JuMP.set_upper_bound(θ, obj)
        (best_actual, f_actual, f_approx) = (-Inf, -Inf, Inf)
    end

    # BOUND DUAL VARIABLES IF INTENDED
    ############################################################################
    if !isnothing(dual_bound)
        for i in 1:length(dual_vars)
            JuMP.set_lower_bound(x[i], -dual_bound)
            JuMP.set_upper_bound(x[i], dual_bound)
        end
    end

    # CUTTING-PLANE METHOD
    ############################################################################
    iter = 0
    while iter < integrality_handler.iteration_limit
        iter += 1

        # SOLVE LAGRANGIAN RELAXATION FOR GIVEN DUAL_VARS
        ########################################################################
        # Evaluate the real function and a subgradient
        f_actual = _solve_Lagrangian_relaxation!(subgradients, node, dual_vars, integrality_handler.slacks, :yes)
        @infiltrate algoParams.infiltrate_state in [:all, :lagrange] || model.ext[:sddp_policy_graph].ext[:iteration] >= 5

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

        @infiltrate algoParams.infiltrate_state in [:all, :lagrange] || model.ext[:sddp_policy_graph].ext[:iteration] >= 5

        print("UB: ", f_approx, ", LB: ", f_actual)
        println()

        # CONVERGENCE CHECK AND UPDATE
        ########################################################################
        # More reliable than checking whether subgradient is zero
        if isapprox(best_actual, f_approx, atol = atol, rtol = rtol) || all(subgradients.==0)
            dual_vars .= best_mult
            if dualsense == JuMP.MOI.MIN_SENSE
                dual_vars .*= -1
            end

            for (i, (name, bin_state)) in enumerate(node.ext[:backward_data][:bin_states])
                #prepare_state_fixing!(node, state_comp)
                JuMP.fix(bin_state, integrality_handler.old_rhs[i], force = true)
            end
            return best_actual
        end
        # Next iterate
        @infiltrate algoParams.infiltrate_state in [:all, :lagrange] || model.ext[:sddp_policy_graph].ext[:iteration] >= 5
        dual_vars .= value.(x)
        # can be deleted with the next update of GAMS.jl
        replace!(dual_vars, NaN => 0)
    end
    error("Could not solve for Lagrangian duals. Iteration limit exceeded.")
end


function _solve_Lagrangian_relaxation!(
    subgradients::Vector{Float64},
    node::SDDP.Node,
    dual_vars::Vector{Float64},
    slacks,
    update_subgradients::Symbol, #TODO: Why not boolean?
)
    model = node.ext[:linSubproblem]
    old_obj = JuMP.objective_function(model)
    # Set the Lagrangian relaxation of the objective in the primal model
    fact = (JuMP.objective_sense(model) == JuMP.MOI.MIN_SENSE ? 1 : -1)
    new_obj = old_obj + fact * LinearAlgebra.dot(dual_vars, slacks)
    JuMP.set_objective_function(model, new_obj)
    JuMP.optimize!(model)
    lagrangian_obj = JuMP.objective_value(model)

    if update_subgradients == :yes
        subgradients .= fact .* JuMP.value.(slacks)
    end

    # Reset old objective, update subgradients using slack values
    JuMP.set_objective_function(model, old_obj)

    return lagrangian_obj
end


function initialize_duals(
    node::SDDP.Node,
    linearizedSubproblem::JuMP.Model,
    dual_regime::Symbol,
)

    # Get number of states and create zero vector for duals
    number_of_states = length(node.ext[:backward_data][:bin_states])
    dual_vars_initial = zeros(number_of_states)

    # DUAL REGIME I: USE ZEROS
    ############################################################################
    if dual_regime == :zeros
        # Do nothing, since zeros are already defined

    # DUAL REGIME II: USE LP RELAXATION
    ############################################################################
    elseif dual_regime == :gurobi_relax || dual_regime == :cplex_relax
        # Create LP Relaxation
        undo_relax = JuMP.relax_integrality(linearizedSubproblem);

        # Define appropriate solver
        if dual_regime == :gurobi_relax
            set_optimizer(linearizedSubproblem, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>"Gurobi", "optcr"=>0.0))
        elseif dual_regime == :cplex_relax
            set_optimizer(linearizedSubproblem, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>"CPLEX", "optcr"=>0.0))
        end

        # Solve LP Relaxation
        JuMP.optimize!(linearizedSubproblem)

        # Get dual values (reduced costs) for binary states as initial solution
        for (i, name) in enumerate(keys(node.ext[:backward_data][:bin_states]))
           variable_name = node.ext[:backward_data][:bin_states][name]
           reference_to_constr = FixRef(variable_name)
           dual_vars_initial[i] = JuMP.getdual(reference_to_constr)
        end

        # Undo relaxation
        undo_relax()

    # DUAL REGIME III: USE FIXED MIP MODEL (DUALS ONLY PROVIDED BY CPLEX)
    ############################################################################
    elseif dual_regime == :cplex_fixed
        # Define cplex solver
        set_optimizer(linearizedSubproblem, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>"CPLEX", "optcr"=>0.0))

        # Solve original primal model in binary space
        JuMP.optimize!(linearizedSubproblem)

        # Get dual values (reduced costs) for binary states as initial solution
        for (i, name) in enumerate(keys(node.ext[:backward_data][:bin_states]))
           variable_name = node.ext[:backward_data][:bin_states][name]
           reference_to_constr = FixRef(variable_name)
           dual_vars_initial[i] = JuMP.getdual(reference_to_constr)
        end

    # DUAL REGIME IV: USE COMBINATION FOR CPLEX
    # use fixed MIP model values for continuous original states
    # use LP relaxation values for binary or integer original states
    ############################################################################
    elseif dual_regime == :cplex_combi
        # Define cplex solver
        set_optimizer(linearizedSubproblem, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>"CPLEX", "optcr"=>0.0))

        # Solve original primal model in binary space
        JuMP.optimize!(linearizedSubproblem)

        # Get dual values (reduced costs) for binary states as initial solution
        for (i, name) in enumerate(keys(node.ext[:backward_data][:bin_states]))
           variable_name = node.ext[:backward_data][:bin_states][name]
           reference_to_constr = FixRef(variable_name)
           dual_vars_initial[i] = JuMP.getdual(reference_to_constr)
        end

        # Create LP Relaxation
        undo_relax = JuMP.relax_integrality(linearizedSubproblem);

        # Solve LP Relaxation
        JuMP.optimize!(linearizedSubproblem)

        # Replace dual values for binary and integer original variables
        for (i, name) in enumerate(keys(node.ext[:backward_data][:bin_states]))
           variable_name = node.ext[:backward_data][:bin_states][name]
           reference_to_constr = FixRef(variable_name)

           # associated original state
           # TODO: Why is this stored in backward_data, but also in BinaryState struct itself?
           original_state_sym = node.ext[:backward_data][:bin_x_names][name]
           original_state = node.ext[:lin_states][original_state_sym]

           # if original state is integer or binary, replace dual_vars_initial
           if original_state.info.in.binary || original_state.info.in.integer
               dual_vars_initial[i] = JuMP.getdual(reference_to_constr)
           end
        end

        # Undo relaxation
        undo_relax()
    end

    return dual_vars_initial
end


function _bundle_proximal(
    node::SDDP.Node,
    node_index::Int64,
    obj::Float64,
    dual_vars::Vector{Float64},
    integrality_handler::SDDP.SDDiP,
    algoParams::NCNBD.AlgoParams,
    appliedSolvers::NCNBD.AppliedSolvers,
    dual_bound::Union{Float64,Nothing}
    )

    # INITIALIZATION
    ############################################################################
    atol = integrality_handler.atol # corresponds to deltabar
    rtol = integrality_handler.rtol # corresponds to deltabar
    model = node.ext[:linSubproblem]
    # Assume the model has been solved. Solving the MIP is usually very quick
    # relative to solving for the Lagrangian duals, so we cheat and use the
    # solved model's objective as our bound while searching for the optimal duals

    # initialize bundle parameters
    alpha = algoParams.bundle_alpha
    bundle_factor = algoParams.bundle_factor

    # initialize stability center
    center = deepcopy(dual_vars)

    # This does not work since the problem has been changed since then
    #assert JuMP.termination_status(model) == MOI.OPTIMAL
    #obj = JuMP.objective_value(model)

    for (i, (name, bin_state)) in enumerate(node.ext[:backward_data][:bin_states])
        integrality_handler.old_rhs[i] = JuMP.fix_value(bin_state)
        integrality_handler.slacks[i] = bin_state - integrality_handler.old_rhs[i]
        JuMP.unfix(bin_state)
        #JuMP.unset_binary(state_comp.in) # TODO: maybe not required
        JuMP.set_lower_bound(bin_state, 0)
        JuMP.set_upper_bound(bin_state, 1)
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
    approx_model = JuMP.Model(Gurobi.Optimizer)
    # even if objective is quadratic, it should be possible to use Gurobi
    set_optimizer(approx_model, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>appliedSolvers.MILP, "optcr"=>0.0))

    #JuMP.set_silent(approx_model)

    # Determine sign for bundle regularization term
    fact = (dualsense == JuMP.MOI.MIN_SENSE ? 1 : -1)

    # Define Lagrangian dual multipliers
    @variables approx_model begin
        θ
        x[1:length(dual_vars)]
    end
    # Define objective for bundle method
    JuMP.@objective(approx_model, dualsense, θ + fact * 0.5 / bundle_factor * LinearAlgebra.dot(x-center, x-center))

    if dualsense == MOI.MIN_SENSE
        JuMP.set_lower_bound(θ, obj)
        (best_actual, f_actual, f_approx) = (Inf, Inf, -Inf)
    else
        #JuMP.set_upper_bound(θ, 10000.0)

        JuMP.set_upper_bound(θ, obj)
        (best_actual, f_actual, f_approx) = (-Inf, -Inf, Inf)
    end

    # BOUND DUAL VARIABLES IF INTENDED
    ############################################################################
    if !isnothing(dual_bound)
        for i in 1:length(dual_vars)
            JuMP.set_lower_bound(x[i], -dual_bound)
            JuMP.set_upper_bound(x[i], dual_bound)
        end
    end

    f_stability = 0
    delta = 0

    # CUTTING-PLANE METHOD
    ############################################################################
    iter = 0
    while iter < integrality_handler.iteration_limit
        iter += 1

        # SOLVE LAGRANGIAN RELAXATION FOR GIVEN DUAL_VARS
        ########################################################################
        # Evaluate the real function and a subgradient
        f_actual = _solve_Lagrangian_relaxation!(subgradients, node, dual_vars, integrality_handler.slacks, :yes)

        # ADAPT STABILITY CENTER
        ########################################################################
        # stability center update
        if f_actual - f_stability >= alpha * delta && iter > 1
            # serious step
            center .= value.(x)
            # increase bundle_factor if there is a serious step
            bundle_factor = bundle_factor * 2
            # bundle_factor = bundle_factor * delta / (2 * (delta - (f_actual - f_stability)))
        else
            # null step (actually not required)
            center = center
        end

        # ADD CUTTING PLANE TO APPROX_MODEL
        ########################################################################
        # Update the model and update best function value so far
        if dualsense == MOI.MIN_SENSE
            JuMP.@constraint(
                approx_model,
                θ >= f_actual + LinearAlgebra.dot(subgradients, x - dual_vars)
            )
            if f_stability <= best_actual
                best_actual = f_stability
                best_mult .= center
            end
        else
            JuMP.@constraint(
                approx_model,
                θ <= f_actual + LinearAlgebra.dot(subgradients, x - dual_vars)
            )
            if f_stability >= best_actual
                best_actual = f_stability
                best_mult .= center
            end
        end

        # SOLVE LAGRANGIAN RELAXATION FOR GIVEN STABILITY CENTER
        ########################################################################
        # Make sure that subgradients are not overwritten
        f_stability = _solve_Lagrangian_relaxation!(subgradients, node, center, integrality_handler.slacks, :no)

        # SOLVE APPROXIMATION MODEL
        ########################################################################
        # Get a bound from the approximate model
        JuMP.optimize!(approx_model)
        @assert JuMP.termination_status(approx_model) == JuMP.MOI.OPTIMAL
        f_approx = JuMP.objective_value(approx_model)

        print("UB: ", f_approx, ", LB: ", f_stability)
        println()

        @infiltrate algoParams.infiltrate_state in [:all, :lagrange]

        # DETERMINE DELTA (not directly used for termination, though)
        ########################################################################
        if dualsense == JuMP.MOI.MIN_SENSE
            #delta = f_stability - f_approx
            delta = f_stability - JuMP.value(θ)
        elseif dualsense == JuMP.MOI.MAX_SENSE
            #delta = f_approx - f_stability
            delta = f_stability - JuMP.value(θ)
        end

        # CONVERGENCE CHECK AND UPDATE
        ########################################################################
        # More reliable than checking whether subgradient is zero
        if isapprox(f_stability, f_approx, atol = atol, rtol = rtol) || all(subgradients.==0)
            #TODO: Check if this is correct
            dual_vars .= best_mult
            if dualsense == JuMP.MOI.MIN_SENSE
                dual_vars .*= -1
            end

            for (i, (name, bin_state)) in enumerate(node.ext[:backward_data][:bin_states])
                #prepare_state_fixing!(node, state_comp)
                JuMP.fix(bin_state, integrality_handler.old_rhs[i], force = true)
            end
            return best_actual
        end
        # Next iterate
        dual_vars .= value.(x)
        # can be deleted with the next update of GAMS.jl
        replace!(dual_vars, NaN => 0)

        # Objective function of approx model has to be adapted to new center
        JuMP.@objective(approx_model, dualsense, θ + fact * 0.5 / bundle_factor * LinearAlgebra.dot(x-center, x-center))

        @infiltrate algoParams.infiltrate_state in [:all, :lagrange]

    end
    error("Could not solve for Lagrangian duals. Iteration limit exceeded.")
end


function _bundle_level(
    node::SDDP.Node,
    node_index::Int64,
    obj::Float64,
    dual_vars::Vector{Float64},
    integrality_handler::SDDP.SDDiP,
    algoParams::NCNBD.AlgoParams,
    appliedSolvers::NCNBD.AppliedSolvers,
    dual_bound::Union{Float64,Nothing}
    )

    # INITIALIZATION
    ############################################################################
    atol = integrality_handler.atol # corresponds to deltabar
    rtol = integrality_handler.rtol # corresponds to deltabar
    model = node.ext[:linSubproblem]
    # Assume the model has been solved. Solving the MIP is usually very quick
    # relative to solving for the Lagrangian duals, so we cheat and use the
    # solved model's objective as our bound while searching for the optimal duals

    # initialize bundle parameters
    level_factor = algoParams.level_factor

    # NOTE initialize stability center
    # center = deepcopy(dual_vars)

    # This does not work since the problem has been changed since then
    #assert JuMP.termination_status(model) == MOI.OPTIMAL
    #obj = JuMP.objective_value(model)

    for (i, (name, bin_state)) in enumerate(node.ext[:backward_data][:bin_states])
        integrality_handler.old_rhs[i] = JuMP.fix_value(bin_state)
        integrality_handler.slacks[i] = bin_state - integrality_handler.old_rhs[i]
        JuMP.unfix(bin_state)
        #JuMP.unset_binary(state_comp.in) # TODO: maybe not required
        JuMP.set_lower_bound(bin_state, 0)
        JuMP.set_upper_bound(bin_state, 1)
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
    approx_model = JuMP.Model(Gurobi.Optimizer)
    # even if objective is quadratic, it should be possible to use Gurobi
    set_optimizer(approx_model, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>appliedSolvers.MILP, "optcr"=>0.0))

    #JuMP.set_silent(approx_model)

    # NOTE Determine sign for bundle regularization term
    # fact = (dualsense == JuMP.MOI.MIN_SENSE ? 1 : -1)

    # Define Lagrangian dual multipliers
    @variables approx_model begin
        θ
        x[1:length(dual_vars)]
    end

    if dualsense == MOI.MIN_SENSE
        JuMP.set_lower_bound(θ, obj)
        (best_actual, f_actual, f_approx) = (Inf, Inf, -Inf)
    else
        #JuMP.set_upper_bound(θ, 10000.0)

        JuMP.set_upper_bound(θ, obj)
        (best_actual, f_actual, f_approx) = (-Inf, -Inf, Inf)
    end

    # BOUND DUAL VARIABLES IF INTENDED
    ############################################################################
    if !isnothing(dual_bound)
        for i in 1:length(dual_vars)
            JuMP.set_lower_bound(x[i], -dual_bound)
            JuMP.set_upper_bound(x[i], dual_bound)
        end
    end

    # CUTTING-PLANE METHOD
    ############################################################################
    iter = 0
    while iter < integrality_handler.iteration_limit
        iter += 1

        # SOLVE LAGRANGIAN RELAXATION FOR GIVEN DUAL_VARS
        ########################################################################
        # Evaluate the real function and determine a subgradient
        f_actual = _solve_Lagrangian_relaxation!(subgradients, node, dual_vars, integrality_handler.slacks, :yes)

        # ADD CUTTING PLANE TO APPROX_MODEL
        ########################################################################
        # Update the model and update best function value so far
        if dualsense == MOI.MIN_SENSE
            JuMP.@constraint(
                approx_model,
                θ >= f_actual + LinearAlgebra.dot(subgradients, x - dual_vars)
                # TODO: Reset upper bound to inf?
            )
            if f_actual <= best_actual
                best_actual = f_actual
                best_mult .= dual_vars
            end
        else
            JuMP.@constraint(
                approx_model,
                θ <= f_actual + LinearAlgebra.dot(subgradients, x - dual_vars)
                # TODO: Reset lower boumd to -inf?
            )
            if f_actual >= best_actual
                # bestmult is not simply getvalue.(x), since approx_model may just haven gotten lucky
                # same for best_actual
                best_actual = f_actual
                best_mult .= dual_vars
            end
        end

        # SOLVE APPROXIMATION MODEL
        ########################################################################
        # Define objective for approx_model
        JuMP.@objective(approx_model, dualsense, θ)

        # Get an upper bound from the approximate model
        # (we could actually also use obj here)
        JuMP.optimize!(approx_model)
        @assert JuMP.termination_status(approx_model) == JuMP.MOI.OPTIMAL
        f_approx = JuMP.objective_value(approx_model)

        @infiltrate algoParams.infiltrate_state in [:all, :lagrange]

        # Construct the gap (not directly used for termination, though)
        #gap = abs(best_actual - f_approx)
        gap = abs(best_actual - obj)

        print("UB: ", f_approx, ", LB: ", f_actual, best_actual)
        println()

        # CONVERGENCE CHECK AND UPDATE
        ########################################################################
        # More reliable than checking whether subgradient is zero
        if isapprox(best_actual, f_approx, atol = atol, rtol = rtol) || all(subgradients.==0)
            #TODO: Check if this is correct
            dual_vars .= best_mult
            if dualsense == JuMP.MOI.MIN_SENSE
                dual_vars .*= -1
            end

            for (i, (name, bin_state)) in enumerate(node.ext[:backward_data][:bin_states])
                #prepare_state_fixing!(node, state_comp)
                JuMP.fix(bin_state, integrality_handler.old_rhs[i], force = true)
            end
            return best_actual
        end

        # FORM A NEW LEVEL
        ########################################################################
        if dualsense == :Min
            level = f_approx + gap * level_factor
            #TODO: + atol/10.0 for numerical issues?
            JuMP.setupperbound(θ, level)
        else
            level = f_approx - gap * level_factor
            #TODO: - atol/10.0 for numerical issues?
            JuMP.setlowerbound(θ, level)
        end

        # DETERMINE NEXT ITERATE USING PROXIMAL PROBLEM
        ########################################################################
        # Objective function of approx model has to be adapted to new center
        JuMP.@objective(approx_model, Min, sum((dual_vars[i] - x[i])^2 for i=1:length(dual_vars)))
        JuMP.optimize!(approx_model)
        @assert JuMP.termination_status(approx_model) == JuMP.MOI.OPTIMAL

        # Next iterate
        dual_vars .= value.(x)
        # can be deleted with the next update of GAMS.jl
        replace!(dual_vars, NaN => 0)

        @infiltrate algoParams.infiltrate_state in [:all, :lagrange]

    end
    error("Could not solve for Lagrangian duals. Iteration limit exceeded.")
end
