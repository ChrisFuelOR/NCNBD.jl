const NCNBD_TIMER = TimerOutputs.TimerOutput()

function outer_loop_iteration(parallel_scheme::SDDP.Serial, model::SDDP.PolicyGraph{T},
    options::SDDP.Options, algoParams::NCNBD.AlgoParams, appliedSolvers::NCNBD.AppliedSolvers) where {T}

    # CALL THE INNER LOOP AND GET BACK RESULTS IF CONVERGED
    ############################################################################
    TimerOutputs.@timeit NCNBD_TIMER "inner_loop" begin
        inner_loop_results = NCNBD.inner_loop(parallel_scheme, model, options, algoParams, appliedSolvers)
    end

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

    has_converged = false
    status = :Blubb
    cuts = Dict{Symbol, Vector{Float64}}()
    current_sol = forward_results.sampled_states
    lower_bound = 0.0

    @infiltrate

    # RETURN RESULTS
    ############################################################################
    return NCNBD.OuterLoopIterationResult(
        #Distributed.myid(),
        lower_bound,
        forward_results.cumulative_value,
        current_sol,
        has_converged,
        status
        #cuts,
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
                incoming_state_value, # no State struct!
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
    TimerOutputs.@timeit NCNBD_TIMER "backward_pass" begin
        cuts = NCNBD.inner_loop_backward_pass(
            model,
            options,
            algoParams,
            appliedSolvers,
            forward_trajectory.scenario_path,
            forward_trajectory.sampled_states,
            forward_trajectory.objective_states,
            forward_trajectory.belief_states,
        )
    end
    @infiltrate

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

    has_converged = true
    status = :Blubb
    cuts = Dict{Symbol, Vector{Float64}}()
    current_sol = forward_trajectory.sampled_states
    lower_bound = 0.0

    return NCNBD.InnerLoopIterationResult(
        #Distributed.myid(),
        lower_bound,
        forward_trajectory.cumulative_value,
        current_sol,
        has_converged,
        status,
        #cuts,
    )
end


function inner_loop_forward_pass(model::SDDP.PolicyGraph{T}, options::SDDP.Options,
    algoParams::NCNBD.AlgoParams, appliedSolvers::NCNBD.AppliedSolvers,
    ::SDDP.DefaultForwardPass) where {T}

    # SAMPLING AND INITIALIZATION (JUST LIKE IN SDDP)
    ############################################################################
    # First up, sample a scenario. Note that if a cycle is detected, this will
    # return the cycle node as well.
    TimerOutputs.@timeit NCNBD_TIMER "sample_scenario" begin
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

        # Set sigma for regularization
        sigma = algoParams.sigma[node_index]

        # Set optimizer to MILP optimizer
        linearizedSubproblem = node.ext[:linSubproblem]
        #set_optimizer(linearizedSubproblem, appliedSolvers.MILP)
        set_optimizer(linearizedSubproblem, GAMS.Optimizer)
        JuMP.set_optimizer_attribute(linearizedSubproblem, "Solver", "Gurobi")

        # SUBPROBLEM SOLUTION
        ############################################################################
        # Solve the subproblem, note that `require_duals = false`.
        TimerOutputs.@timeit NCNBD_TIMER "solve_subproblem" begin
            subproblem_results = solve_subproblem(
                model,
                node,
                incoming_state_value, # only values, no State struct!
                noise,
                scenario_path[1:depth],
                sigma,
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


# Internal function: solve the subproblem associated with node given the
# incoming state variables state and realization of the stagewise-independent
# noise term noise. If require_duals=true, also return the dual variables
# associated with the fixed constraint of the incoming state variables.
function solve_subproblem(
    model::SDDP.PolicyGraph{T},
    node::SDDP.Node{T},
    state::Dict{Symbol,Float64},
    noise,
    scenario_path::Vector{Tuple{T,S}},
    sigma::Float64;
    require_duals::Bool,
) where {T,S}

    # MODEL PARAMETRIZATION (-> LINEARIZED SUBPROBLEM!)
    ############################################################################
    linearizedSubproblem = node.ext[:linSubproblem]

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

    # REGULARIZE SUBPROBLEM
    ############################################################################
    # storage for regularization data
    node.ext[:regularization_data] = Dict{Symbol,Any}()

    regularize_subproblem!(node, linearizedSubproblem, sigma)

    # SOLUTION
    ############################################################################
    JuMP.optimize!(linearizedSubproblem)

    if haskey(model.ext, :total_solves)
        model.ext[:total_solves] += 1
    else
        model.ext[:total_solves] = 1
    end

    # if JuMP.primal_status(node.ext[:linSubproblem]) != JuMP.MOI.FEASIBLE_POINT
    #     SDDP.attempt_numerical_recovery(node)
    # end

    state = get_outgoing_state(node)
    stage_objective = JuMP.value(node.ext[:lin_stage_objective])
    #stage_objective_value(node.ext[:lin_stage_objective])
    objective = JuMP.objective_value(node.ext[:linSubproblem])

    # If require_duals = true, check for dual feasibility and return a dict with
    # the dual on the fixed constraint associated with each incoming state
    # variable. If require_duals=false, return an empty dictionary for
    # type-stability.
    dual_values = if require_duals
        get_dual_variables(node, node.integrality_handler)
    else
        Dict{Symbol,Float64}()
    end

    # if node.post_optimize_hook !== nothing
    #     node.post_optimize_hook(pre_optimize_ret)
    # end

    # STORE RESULTS FOR NL-CONSTRAINT VARIABLES FOR PLA REFINEMENT
    ############################################################################
    number_of_nonlinearities = size(node.ext[:nlFunctions], 1)

    for i = 1:number_of_nonlinearities
        nlFunction = node.ext[:nlFunctions][i]
        dimension = size(nlFunction.variablesContained, 1)

        nlFunction.ext[:optSolution] = Dict{JuMP.VariableRef, Float64}()

        for j = 1:dimension
            variableReference = nlFunction.variablesContained[j]
            nlFunction.ext[:optSolution][variableReference] = JuMP.value(variableReference)
        end
    end

    # DE-REGULARIZE SUBPROBLEM
    ############################################################################
    deregularize_subproblem!(node, linearizedSubproblem)

    return (
        state = state,
        duals = dual_values,
        objective = objective,
        stage_objective = stage_objective,
    )
end

# TODO: actually not required
# Requires node.subproblem to have been solved with DualStatus == FeasiblePoint
function get_dual_variables(node::SDDP.Node, ::SDDP.ContinuousRelaxation)
    # Note: due to JuMP's dual convention, we need to flip the sign for
    # maximization problems.
    dual_values = Dict{Symbol,Float64}()
    if JuMP.dual_status(node.ext[:linSubproblem]) != JuMP.MOI.FEASIBLE_POINT
        SDDP.write_subproblem_to_file(node, "linSubproblem.mof.json", throw_error = true)
    end
    dual_sign = JuMP.objective_sense(node.ext[:linSubproblem]) == MOI.MIN_SENSE ? 1.0 : -1.0
    for (name, state_comp) in node.ext[:lin_states]
        ref = JuMP.FixRef(state_comp.in)
        dual_values[name] = dual_sign * JuMP.dual(ref)
    end
    return dual_values
end


function inner_loop_backward_pass(
    model::SDDP.PolicyGraph{T},
    options::SDDP.Options,
    algoParams::NCNBD.AlgoParams,
    appliedSolvers::NCNBD.AppliedSolvers,
    scenario_path::Vector{Tuple{T,NoiseType}},
    sampled_states::Vector{Dict{Symbol,Float64}},
    objective_states::Vector{NTuple{N,Float64}},
    belief_states::Vector{Tuple{Int,Dict{T,Float64}}}) where {T,NoiseType,N}

    cuts = Dict{T,Vector{Any}}(index => Any[] for index in keys(model.nodes))

    for index = length(scenario_path):-1:1
        outgoing_state = sampled_states[index]
        objective_state = get(objective_states, index, nothing)
        partition_index, belief_state = get(belief_states, index, (0, nothing))
        items = BackwardPassItems(T, SDDP.Noise)
        if belief_state !== nothing
            # # Update the cost-to-go function for partially observable model.
            #print("blubb")
            # for (node_index, belief) in belief_state
            #     belief == 0.0 && continue
            #     solve_all_children(
            #         model,
            #         model[node_index],
            #         items,
            #         belief,
            #         belief_state,
            #         objective_state,
            #         outgoing_state,
            #         options.backward_sampling_scheme,
            #         scenario_path[1:index],
            #     )
            # end
            # # We need to refine our estimate at all nodes in the partition.
            # for node_index in model.belief_partition[partition_index]
            #     node = model[node_index]
            #     # Update belief state, etc.
            #     current_belief = node.belief_state::BeliefState{T}
            #     for (idx, belief) in belief_state
            #         current_belief.belief[idx] = belief
            #     end
            #     new_cuts = refine_bellman_function(
            #         model,
            #         node,
            #         node.bellman_function,
            #         options.risk_measures[node_index],
            #         outgoing_state,
            #         items.duals,
            #         items.supports,
            #         items.probability .* items.belief,
            #         items.objectives,
            #     )
            #     push!(cuts[node_index], new_cuts)
            #end
        else
            node_index, _ = scenario_path[index]
            node = model[node_index]
            if length(node.children) == 0
                continue
            end

            # Dict to store values of binary approximation of the state
            # Note that we could also retrieve this from the actual trial point
            # (outgoing_state) or from its approximation via binexpand. However,
            # this collection is not only important to obtain the correct values,
            # but also to store them together with the symbol/name of the variable.
            node.ext[:binary_state_values] = Dict{Symbol, Float64}()

            # SOLVE ALL CHILDREN PROBLEM
            ####################################################################
            solve_all_children(
                model,
                node,
                node_index,
                items,
                1.0,
                belief_state,
                objective_state,
                outgoing_state,
                options.backward_sampling_scheme,
                scenario_path[1:index],
                algoParams,
                appliedSolvers
            )

            # RECONSTRUCT USED TRIAL POINTS IN BACKWARD PASS
            ####################################################################
            used_trial_points = Dict{Symbol,Float64}()
            epsilon = algoParams.binaryPrecision[node_index]
            for (name, value) in outgoing_state
                state_comp = node.ext[:lin_states][name]
                (approx_state_value, )  = determine_used_trial_states(state_comp, value, epsilon)
                used_trial_points[name] = approx_state_value
            end

            @infiltrate
            # REFINE BELLMAN FUNCTION BY ADDING CUTS
            ####################################################################
            new_cuts = refine_bellman_function(
                model,
                node,
                node_index,
                node.bellman_function,
                options.risk_measures[node_index],
                used_trial_points,
                items.bin_state_values,
                items.duals,
                items.supports,
                items.probability,
                items.objectives,
                algoParams
            )
            # push!(cuts[node_index], new_cuts)
            # if options.refine_at_similar_nodes
            #     # Refine the bellman function at other nodes with the same
            #     # children, e.g., in the same stage of a Markovian policy graph.
            #     for other_index in options.similar_children[node_index]
            #         copied_probability = similar(items.probability)
            #         other_node = model[other_index]
            #         for (idx, child_index) in enumerate(items.nodes)
            #             copied_probability[idx] =
            #                 get(options.Φ, (other_index, child_index), 0.0) *
            #                 items.supports[idx].probability
            #         end
            #         new_cuts = refine_bellman_function(
            #             model,
            #             other_node,
            #             other_node.bellman_function,
            #             options.risk_measures[other_index],
            #             outgoing_state,
            #             items.duals,
            #             items.supports,
            #             copied_probability,
            #             items.objectives,
            #         )
            #         push!(cuts[other_index], new_cuts)
            #     end
            # end
        end
    end
    return cuts
end


function solve_all_children(
    model::SDDP.PolicyGraph{T},
    node::SDDP.Node{T},
    node_index::Int64,
    items::BackwardPassItems,
    belief::Float64,
    belief_state,
    objective_state,
    outgoing_state::Dict{Symbol,Float64},
    backward_sampling_scheme::SDDP.AbstractBackwardSamplingScheme,
    scenario_path,
    algoParams::NCNBD.AlgoParams,
    appliedSolvers::NCNBD.AppliedSolvers
) where {T}
    length_scenario_path = length(scenario_path)
    for child in node.children
        if isapprox(child.probability, 0.0, atol = 1e-6)
            continue
        end
        child_node = model[child.term]
        for noise in SDDP.sample_backward_noise_terms(backward_sampling_scheme, child_node)
            if length(scenario_path) == length_scenario_path
                push!(scenario_path, (child.term, noise.term))
            else
                scenario_path[end] = (child.term, noise.term)
            end
            if haskey(items.cached_solutions, (child.term, noise.term))
                sol_index = items.cached_solutions[(child.term, noise.term)]
                push!(items.duals, items.duals[sol_index])
                push!(items.supports, items.supports[sol_index])
                push!(items.nodes, child_node.index)
                push!(items.probability, items.probability[sol_index])
                push!(items.objectives, items.objectives[sol_index])
                push!(items.belief, belief)
                push!(items.bin_state_values, items.bin_state_values[sol_index])
            else
                # Update belief state, etc.
                if belief_state !== nothing
                    current_belief = child_node.belief_state::SDDP.BeliefState{T}
                    current_belief.updater(
                        current_belief.belief,
                        belief_state,
                        current_belief.partition_index,
                        noise.term,
                    )
                end
                if objective_state !== nothing
                    SDDP.update_objective_state(
                        child_node.objective_state,
                        objective_state,
                        noise.term,
                    )
                end
                TimerOutputs.@timeit NCNBD_TIMER "solve_bw_subproblem" begin
                    subproblem_results = solve_subproblem_backward(
                        model,
                        child_node,
                        node_index+1,
                        outgoing_state,
                        noise.term,
                        scenario_path,
                        require_duals = true, #TODO: Delete (also in forward pass)
                        algoParams,
                        appliedSolvers
                    )
                end
                push!(items.duals, subproblem_results.duals)
                push!(items.supports, noise)
                push!(items.nodes, child_node.index)
                push!(items.probability, child.probability * noise.probability)
                push!(items.objectives, subproblem_results.objective)
                push!(items.belief, belief)
                push!(items.bin_state_values, subproblem_results.bin_state_values)
                items.cached_solutions[(child.term, noise.term)] = length(items.duals)
                #TODO: Maybe add binary precision
            end
        end
    end
    if length(scenario_path) == length_scenario_path
        # No-op. There weren't any children to solve.
    else
        # Drop the last element (i.e., the one we added).
        pop!(scenario_path)
    end
end


# Internal function: solve the subproblem associated with node given the
# incoming state variables state and realization of the stagewise-independent
# noise term noise. If require_duals=true, also return the dual variables
# associated with the fixed constraint of the incoming state variables.
function solve_subproblem_backward(
    model::SDDP.PolicyGraph{T},
    node::SDDP.Node{T},
    node_index::Int64,
    state::Dict{Symbol,Float64},
    noise,
    scenario_path::Vector{Tuple{T,S}},
    algoParams::NCNBD.AlgoParams,
    appliedSolvers::NCNBD.AppliedSolvers;
    require_duals::Bool
) where {T,S}

    # MODEL PARAMETRIZATION (-> LINEARIZED SUBPROBLEM!)
    ############################################################################
    linearizedSubproblem = node.ext[:linSubproblem]

    # storage for backward pass data
    node.ext[:backward_data] = Dict{Symbol,Any}()

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

    # BACKWARD PASS PREPARATION
    ############################################################################
    # prepare_backward_pass!(node, linearizedSubproblem, binaryPrecision)
    # Also adapt solver here
    changeToBinarySpace!(node, linearizedSubproblem, state, algoParams.binaryPrecision[node_index])

    # PRIMAL SOLUTION
    ############################################################################
    JuMP.optimize!(linearizedSubproblem)
    #@infiltrate

    if haskey(model.ext, :total_solves)
        model.ext[:total_solves] += 1
    else
        model.ext[:total_solves] = 1
    end

    # if JuMP.primal_status(node.subproblem) != JuMP.MOI.FEASIBLE_POINT
    #     attempt_numerical_recovery(node)
    # end

    objective = JuMP.objective_value(linearizedSubproblem)

    # DUAL SOLUTION
    ############################################################################
    # If require_duals = true, check for dual feasibility and return a dict with
    # the dual on the fixed constraint associated with each incoming state
    # variable. If require_duals=false, return an empty dictionary for
    # type-stability.
    if require_duals
        lagrangian_results = get_dual_variables_backward(node, node_index, algoParams, appliedSolvers)
        dual_values = lagrangian_results.dual_values
        bin_state_valus = lagrangian.results.dual_values
    else
        dual_values = Dict{Symbol,Float64}()
        bin_state_values = Dict{Symbol,Float64}()
    end

    # if node.post_optimize_hook !== nothing
    #     node.post_optimize_hook(pre_optimize_ret)
    # end

    #TODO: REGAIN ORIGINAL MODEL
    ############################################################################
    changeToOriginalSpace!(node, linearizedSubproblem, state)
    return (
        duals = dual_values,
        bin_state_values = bin_state_values,
        objective = objective
    )
end


function get_dual_variables_backward(
    node::SDDP.Node,
    node_index::Int64,
    algoParams::NCNBD.AlgoParams,
    appliedSolvers::NCNBD.AppliedSolvers)

    # storages for return of dual values and binary state values (trial point)
    dual_values = Dict{Symbol,Float64}()
    bin_state_values = Dict{Symbol, Float64}()

    # TODO implement smart choice for initial duals
    number_of_states = length(node.ext[:backward_data][:bin_states])
    dual_vars = zeros(number_of_states)
    solver_obj = JuMP.objective_value(node.ext[:linSubproblem])

    # Create an SDDiP integrality_handler here to store the Lagrangian dual information
    #TODO: Store tolerances in algoParams
    integrality_handler = SDDP.SDDiP(iteration_limit = 100, atol = 1e-8, rtol = 1e-8)
    integrality_handler = SDDP.update_integrality_handler!(integrality_handler, appliedSolvers.MILP, number_of_states)
    node.ext[:lagrange] = integrality_handler

    try
        kelley_obj = _kelley(node, node_index, dual_vars, integrality_handler, algoParams, appliedSolvers)::Float64
        @assert isapprox(solver_obj, kelley_obj, atol = 1e-8, rtol = 1e-8)
    catch e
        SDDP.write_subproblem_to_file(node, "subproblem.mof.json", throw_error = false)
        rethrow(e)
    end

    for (i, name) in enumerate(keys(node.ext[:backward_data][:bin_states]))
        # TODO (maybe) change dual signs inside kelley to match LP duals
        dual_values[name] = -dual_vars[i]
        bin_state_values[name] = integrality_handler.old_rhs[i]
    end

    return (
        dual_values=dual_values,
        bin_state_values=bin_state_values
    )
end
