const NCNBD_TIMER = TimerOutputs.TimerOutput()

function outer_loop_iteration(parallel_scheme::SDDP.Serial, model::SDDP.PolicyGraph{T},
    options::SDDP.Options, algoParams::NCNBD.AlgoParams, appliedSolvers::NCNBD.AppliedSolvers) where {T}

    # CALL THE INNER LOOP AND GET BACK RESULTS IF CONVERGED
    ############################################################################
    # TODO: TO BE IMPLEMENTED
    #TimerOutputs.@timeit NCNBD_TIMER "inner_loop" begin
    #    inner_loop_results = NCNBD.inner_loop(parallel_scheme, model, options, algoParams, appliedSolvers)
    #end

    @infiltrate

    # START AN OUTER LOOP FORWARD PASS
    ############################################################################
    TimerOutputs.@timeit NCNBD_TIMER "outer_loop_solution" begin
        forward_results = NCNBD.outer_loop_forward_pass(model, options, algoParams, appliedSolvers)
    end
    # forward_pass options?
    # TODO: values of which variables to return in optimal solution? Only states or all?

    # DETERMINE AN ALTERNATIVE LOWER BOUND
    ############################################################################
    # TODO: TO BE IMPLEMENTED
    # This can just be determined as the solution of the first stage from forward_results

    # LOGGING RESULTS?
    ############################################################################
    # push!(
    #     options.log,
    #     Log(
    #         length(options.log) + 1,
    #         bound,
    #         forward_trajectory.cumulative_value,
    #         time() - options.start_time,
    #         Distributed.myid(),
    #         model.ext[:total_solves],
    #     ),
    # )

    # CONVERGENCE TEST
    ############################################################################
    # TODO: Here we should use a classical convergence test
    # Later on, also SDDP stopping rules should be considered
    #has_converged, status = convergence_test(model, options.log, options.stopping_rules)

    @infiltrate

    has_converged = false
    status = :Blubb
    cuts = Dict{Symbol, Vector{Float64}}()
    current_sol = forward_trajectory.sampled_states
    lower_bound = 0.0

    @infiltrate

    # RETURN RESULTS
    ############################################################################
    return NCNBD.OuterLoopIterationResult(
        #Distributed.myid(),
        lower_bound,
        forward_trajectory.cumulative_value,
        current_sol,
        has_converged,
        status,
        cuts,
    )
end

function outer_loop_forward_pass(model::SDDP.PolicyGraph{T},
    options::SDDP.Options, algoParams::NCNBD.AlgoParams, appliedSolvers::NCNBD.AppliedSolvers) where {T}

    # SAMPLING AND INITIALIZATION (JUST LIKE IN SDDP)
    ############################################################################
    # First up, sample a scenario. Note that if a cycle is detected, this will
    # return the cycle node as well.
    TimerOutputs.@timeit NCNBD_TIMER "sample_scenario_outer" begin
        scenario_path, terminated_due_to_cycle =
            SDDP.sample_scenario(model, options.sampling_scheme)
    end
    # Storage for the list of outgoing states that we visit on the forward pass.
    sampled_states = Dict{Symbol,Float64}[]
    # Storage for the belief states: partition index and the belief dictionary.
    belief_states = Tuple{Int,Dict{T,Float64}}[]
    current_belief = SDDP.initialize_belief(model)
    # Our initial incoming state.
    incoming_state_value = copy(options.initial_state)
    # A cumulator for the stage-objectives.
    cumulative_value = 0.0
    # Objective state interpolation.
    objective_state_vector, N = SDDP.initialize_objective_state(model[scenario_path[1][1]])
    objective_states = NTuple{N,Float64}[]

    @infiltrate

    # ACTUAL ITERATION
    ############################################################################
    # Iterate down the scenario.
    for (depth, (node_index, noise)) in enumerate(scenario_path)
        node = model[node_index]
        # Objective state interpolation.
        objective_state_vector =
            SDDP.update_objective_state(node.objective_state, objective_state_vector, noise)
        if objective_state_vector !== nothing
            push!(objective_states, objective_state_vector)
        end
        # Update belief state, etc.
        if node.belief_state !== nothing
            belief = node.belief_state::SDDP.BeliefState{T}
            partition_index = belief.partition_index
            current_belief =
                belief.updater(belief.belief, current_belief, partition_index, noise)
            push!(belief_states, (partition_index, copy(current_belief)))
        end
        # ===== Begin: starting state for infinite horizon =====
        starting_states = options.starting_states[node_index]
        if length(starting_states) > 0
            # There is at least one other possible starting state. If our
            # incoming state is more than δ away from the other states, add it
            # as a possible starting state.
            if distance(starting_states, incoming_state_value) >
               options.cycle_discretization_delta
                push!(starting_states, incoming_state_value)
            end
            # TODO(odow):
            # - A better way of randomly sampling a starting state.
            # - Is is bad that we splice! here instead of just sampling? For
            #   convergence it is probably bad, since our list of possible
            #   starting states keeps changing, but from a computational
            #   perspective, we don't want to keep a list of discretized points
            #   in the state-space δ distance apart...
            incoming_state_value = splice!(starting_states, rand(1:length(starting_states)))
        end
        # ===== End: starting state for infinite horizon =====

        # Set optimizer to MINLP optimizer
        set_optimizer(model, appliedSolvers.MINLP)

        # SUBPROBLEM SOLUTION
        ############################################################################
        # Solve the subproblem, note that `require_duals = false`.
        TimerOutputs.@timeit NCNBD_TIMER "solve_subproblem" begin
            subproblem_results = SDDP.solve_subproblem(
                model,
                node,
                incoming_state_value,
                noise,
                scenario_path[1:depth],
                require_duals = false,
            )
        end
        # Cumulate the stage_objective.
        cumulative_value += subproblem_results.stage_objective
        # Set the outgoing state value as the incoming state value for the next
        # node.
        incoming_state_value = copy(subproblem_results.state)
        # Add the outgoing state variable to the list of states we have sampled
        # on this forward pass.
        push!(sampled_states, incoming_state_value)

        @infiltrate

    end
    if terminated_due_to_cycle
        # Get the last node in the scenario.
        final_node_index = scenario_path[end][1]
        # We terminated due to a cycle. Here is the list of possible starting
        # states for that node:
        starting_states = options.starting_states[final_node_index]
        # We also need the incoming state variable to the final node, which is
        # the outgoing state value of the last node:
        incoming_state_value = sampled_states[end]
        # If this incoming state value is more than δ away from another state,
        # add it to the list.
        if distance(starting_states, incoming_state_value) >
           options.cycle_discretization_delta
            push!(starting_states, incoming_state_value)
        end
    end

    # ===== End: drop off starting state if terminated due to cycle =====
    return (
        scenario_path = scenario_path,
        sampled_states = sampled_states,
        objective_states = objective_states,
        belief_states = belief_states,
        cumulative_value = cumulative_value,
    )
end


function inner_loop_iteration(model::SDDP.PolicyGraph{T}, options::SDDP.Options,
    algoParams::NCNBD.AlgoParams, appliedSolvers::NCNBD.AppliedSolvers) where {T}

    # FORWARD PASS
    ############################################################################
    TimerOutputs.@timeit NCNBD_TIMER "forward_pass" begin
        forward_trajectory = NCNBD.inner_loop_forward_pass(model, options, algoParams, appliedSolvers, options.forward_pass)
    end

    # BINARY REFINEMENT
    ############################################################################
    # TODO: To be implemented
    # If the forward pass solution did not change during the last iteration, then
    # increase the binary precision (for all stages?)

    # BACKWARD PASS
    ############################################################################
    # TimerOutputs.@timeit NCNBD_TIMER "backward_pass" begin
    #     cuts = backward_pass(
    #         model,
    #         options,
    #         algoParams,
    #         appliedSolvers,
    #         forward_trajectory.scenario_path,
    #         forward_trajectory.sampled_states,
    #         forward_trajectory.objective_states,
    #         forward_trajectory.belief_states,
    #     )
    # end

    # CALCULATE LOWER BOUND
    ############################################################################
    #TimerOutputs.@timeit NCNBD_TIMER "calculate_bound" begin
    #    bound = calculate_bound(model)
    #end

    # PREPARE LOGGING
    ############################################################################
    # TODO: Should this be done here or in the inner_loop function?
    # Which parts are required for the convergence test?

    # push!(
    #     options.log,
    #     Log(
    #         length(options.log) + 1,
    #         bound,
    #         forward_trajectory.cumulative_value,
    #         time() - options.start_time,
    #         Distributed.myid(),
    #         model.ext[:total_solves],
    #     ),
    # )

    # CHECK IF THE INNER LOOP CONVERGED YET
    ############################################################################
    # TODO: To be implemented
    #has_converged, status = convergence_test(model, options.log, options.stopping_rules)

    return NCNBD.InnerLoopIterationResult(
        #Distributed.myid(),
        lower_bound,
        current_sol,
        forward_trajectory.cumulative_value,
        has_converged,
        status,
        cuts,
    )
end


function inner_loop_forward_pass(model::SDDP.PolicyGraph{T}, options::SDDP.Options,
    algoParams::NCNBD.AlgoParams, appliedSolvers::NCNBD.AppliedSolvers,
    ::SDDP.DefaultForwardPass) where {T}

    # SAMPLING AND INITIALIZATION (JUST LIKE IN SDDP)
    ############################################################################
    # First up, sample a scenario. Note that if a cycle is detected, this will
    # return the cycle node as well.
    TimerOutputs.@timeit SDDP_TIMER "sample_scenario" begin
        scenario_path, terminated_due_to_cycle =
            SDDP.sample_scenario(model, options.sampling_scheme)
    end

    #TODO: Has something here to be adapted such that it relates to the linearizedSubproblem?
    # Storage for the list of outgoing states that we visit on the forward pass.
    sampled_states = Dict{Symbol,Float64}[]
    # Storage for the belief states: partition index and the belief dictionary.
    belief_states = Tuple{Int,Dict{T,Float64}}[]
    current_belief = SDDP.initialize_belief(model)
    # Our initial incoming state.
    incoming_state_value = copy(model.ext[:lin_initial_root_state])
    # A cumulator for the stage-objectives.
    cumulative_value = 0.0
    # Objective state interpolation.
    objective_state_vector, N = SDDP.initialize_objective_state(model[scenario_path[1][1]])
    objective_states = NTuple{N,Float64}[]

    # ACTUAL ITERATION
    ########################################################################
    # Iterate down the scenario.
    for (depth, (node_index, noise)) in enumerate(scenario_path)
        node = model[node_index]

        # Objective state interpolation.
        objective_state_vector =
            SDDP.update_objective_state(node.objective_state, objective_state_vector, noise)
        if objective_state_vector !== nothing
            push!(objective_states, objective_state_vector)
        end
        # Update belief state, etc.
        if node.belief_state !== nothing
            belief = node.belief_state::SDDP.BeliefState{T}
            partition_index = belief.partition_index
            current_belief =
                belief.updater(belief.belief, current_belief, partition_index, noise)
            push!(belief_states, (partition_index, copy(current_belief)))
        end
        # ===== Begin: starting state for infinite horizon =====
        starting_states = options.starting_states[node_index]
        if length(starting_states) > 0
            # There is at least one other possible starting state. If our
            # incoming state is more than δ away from the other states, add it
            # as a possible starting state.
            if distance(starting_states, incoming_state_value) >
               options.cycle_discretization_delta
                push!(starting_states, incoming_state_value)
            end
            # TODO(odow):
            # - A better way of randomly sampling a starting state.
            # - Is is bad that we splice! here instead of just sampling? For
            #   convergence it is probably bad, since our list of possible
            #   starting states keeps changing, but from a computational
            #   perspective, we don't want to keep a list of discretized points
            #   in the state-space δ distance apart...
            incoming_state_value = splice!(starting_states, rand(1:length(starting_states)))
        end
        # ===== End: starting state for infinite horizon =====

        # Set optimizer to MINLP optimizer
        set_optimizer(model, appliedSolvers.MILP)

        # REGULARIZE SUBPROBLEM
        ############################################################################
        linearizedSubproblem = node.ext[:linearizedSubproblem]
        # storage for regularization data
        node.ext[:regularization_data] = Dict{Symbol,Any}
        # sigma for this stage
        sigma = algoParams.sigma[node_index]
        regularize_subproblem!(node, linearizedSubproblem, sigma)

        # SUBPROBLEM SOLUTION
        ############################################################################
        # Solve the subproblem, note that `require_duals = false`.
        TimerOutputs.@timeit NCNBD_TIMER "solve_subproblem" begin
            subproblem_results = solve_subproblem(
                model,
                node,
                incoming_state_value,
                noise,
                scenario_path[1:depth],
                require_duals = false,
            )
        end
        # Cumulate the stage_objective.
        cumulative_value += subproblem_results.stage_objective
        # Set the outgoing state value as the incoming state value for the next
        # node.
        incoming_state_value = copy(subproblem_results.state)
        # Add the outgoing state variable to the list of states we have sampled
        # on this forward pass.
        push!(sampled_states, incoming_state_value)

        # DE-REGULARIZE SUBPROBLEM
        ############################################################################
        deregularize_subproblem!(node, linearizedSubproblem)

    end
    if terminated_due_to_cycle
        # Get the last node in the scenario.
        final_node_index = scenario_path[end][1]
        # We terminated due to a cycle. Here is the list of possible starting
        # states for that node:
        starting_states = options.starting_states[final_node_index]
        # We also need the incoming state variable to the final node, which is
        # the outgoing state value of the last node:
        incoming_state_value = sampled_states[end]
        # If this incoming state value is more than δ away from another state,
        # add it to the list.
        if distance(starting_states, incoming_state_value) >
           options.cycle_discretization_delta
            push!(starting_states, incoming_state_value)
        end
    end
    # ===== End: drop off starting state if terminated due to cycle =====
    return (
        scenario_path = scenario_path,
        sampled_states = sampled_states,
        objective_states = objective_states,
        belief_states = belief_states,
        cumulative_value = cumulative_value,
    )
end


function regularize_subproblem!(node::SDDP.Node, linearizedSubproblem::JuMP.Model, sigma::Float64)

    #NOTE: The copy constraint is not modeled explicitly here. Instead,
    # the state variable is unfixed and takes the role of z in our paper.
    # It is then subtracted from the fixed value to obtain the so called slack.
    # TODO: Check if this should be changed later. We manage to avoid introducing
    # an additional variable here, but it becomes a bit confusing. Moreover,
    # this is maybe not possible in the backward pass.

    reg_data = node.ext[:regularization_data]
    reg_data[:fixed_state_value] = Float64[]
    reg_data[:reg_variables] = JuMP.VariableRef[]
    reg_data[:reg_constraints] = JuMP.ConstraintRef[]

    number_of_states = 0

    # UNFIX THE STATE VARIABLES
    ############################################################################
    for (i, (name, state)) in enumerate(node.states)
        reg_data[:fixed_state_value][i] = JuMP.fix_value(state.in)
        reg_data[:slacks][i] = reg_data[:fixed_state_value][i] - state.in
        JuMP.unfix(state.in)
        JuMP.set_lower_bound(state.in, LB)
        JuMP.set_upper_bound(state.in, UB)
        # TODO: Lower and upper bounds have to be stored in the beginning
        # Also required for binary expansion determination
        # Also required for constraints below
        number_of_states = i
    end

    # STORE ORIGINAL OBJECTIVE FUNCTION
    ############################################################################
    reg_data[:old_objective] = JuMP.objective_function(linearizedSubproblem)

    # DEFINE NEW VARIABLES, CONSTRAINTS AND OBJECTIVE
    ############################################################################
    # These variables and constraints are used to define the norm of the slack as a MILP
    # So far, the modeling approach by Bernardo da Costa is used
    # TODO: Maybe change to more efficient approach
    # TODO: Check if signs are correct for max and min problem
    slack = reg_data[:slacks]

    # Variable for objective
    v = JuMP.@variable(linearizedSubproblem)
    push!(reg_data[:reg_variables], v)

    # Get sign for regularization term
    fact = (JuMP.objective_sense(model) == JuMP.MOI.MIN_SENSE ? 1 : -1)

    # New objective
    new_obj = old_obj + sigma * v
    JuMP.set_objective_function(linearizedSubproblem, new_obj)

    # Variables
    slack_plus = JuMP.@variable(linearizedSubproblem, [i=1:number_of_states])
    slack_minus = JuMP.@variable(linearizedSubproblem, [i=1:number_of_states])
    JuMP.set_lower_bound(slack_plus, 0)
    JuMP.set_lower_bound(slack_minus, 0)

    aux_bin = JuMP.@variable(linearizedSubproblem, [i=1:number_of_states])
    push!(reg_data[:reg_variables], slack_plus)
    push!(reg_data[:reg_variables], slack_minus)
    push!(reg_data[:reg_variables], aux_bin, Bin)

    # Constraints
    const_plus = JuMP.@variable(linearizedSubproblem, [i=1:number_of_states], slack_plus[i] <= z[i] * UB[i])
    const_minus = JuMP.@variable(linearizedSubproblem, [i=1:number_of_states], slack_minus[i] <= (1-z[i]) * UB[i])
    push!(reg_data[:reg_constraints], const_plus)
    push!(reg_data[:reg_constraints], const_minus)

    plus_minus = JuMP.@constraint(linearizedSubproblem, [i=1:number_of_states], slack[i] == slack_plus[i] - slack_minus[i])
    push!(reg_data[:reg_constraints], plus_minus)

    const_norm = JuMP.@constraint(linearizedSubproblem, v >= sum(slack_plus[i] + slack_minus[i] for i in 1:number_of_states))
    push!(reg_data[:reg_constraints], const_norm)

end

function deregularize_subproblem!(node::SDDP.Node, linearizedSubproblem::JuMP.Model)

    reg_data = node.ext[:regularization_data]

    # FIX THE STATE VARIABLES
    ############################################################################
    for (i, (name, state)) in enumerate(node.states)
        JuMP.fix(state.in, reg_data[:fixed_state_value][i])
    end

    # REPLACE THE NEW BY THE OLD OBJECTIVE
    ############################################################################
    JuMP.set_objective_function(linearizedSubproblem, reg_data[:old_objective])

    # DELETE ALL REGULARIZATION-BASED VARIABLES AND CONSTRAINTS
    ############################################################################
    delete(linearizedSubproblem, reg_data[:reg_variables])
    delete(linearizedSubproblem, reg_data[:reg_constraints])

end


# Internal function: solve the subproblem associated with node given the
# incoming state variables state and realization of the stagewise-independent
# noise term noise. If require_duals=true, also return the dual variables
# associated with the fixed constraint of the incoming state variables.
function solve_subproblem(
    model::SDDP.PolicyGraph{T},
    node::SDDP.Node{T},
    state::Dict{Symbol,Float64},
    noise,
    scenario_path::Vector{Tuple{T,S}};
    require_duals::Bool,
) where {T,S}

    linearizedSubproblem = node.ext[:linearizedSubproblem]

    # Parameterize the model. First, fix the value of the incoming state
    # variables. Then parameterize the model depending on `noise`. Finally,
    # set the objective.
    set_incoming_state(node, state)
    parameterize(node, noise)

    # pre_optimize_ret = if node.pre_optimize_hook !== nothing
    #     node.pre_optimize_hook(model, node, state, noise, scenario_path, require_duals)
    # else
    #     nothing
    # end

    JuMP.optimize!(linearizedSubproblem)

    if haskey(model.ext, :total_solves)
        model.ext[:total_solves] += 1
    else
        model.ext[:total_solves] = 1
    end

    if JuMP.primal_status(node.subproblem) != JuMP.MOI.FEASIBLE_POINT
        SDDP.attempt_numerical_recovery(node)
    end

    state = get_outgoing_state(node)
    stage_objective = stage_objective_value(node.stage_objective)
    objective = JuMP.objective_value(node.subproblem)

    # If require_duals = true, check for dual feasibility and return a dict with
    # the dual on the fixed constraint associated with each incoming state
    # variable. If require_duals=false, return an empty dictionary for
    # type-stability.
    dual_values = if require_duals
        get_dual_variables(node, node.integrality_handler)
    else
        Dict{Symbol,Float64}()
    end

    if node.post_optimize_hook !== nothing
        node.post_optimize_hook(pre_optimize_ret)
    end

    return (
        state = state,
        duals = dual_values,
        objective = objective,
        stage_objective = stage_objective,
    )
end
