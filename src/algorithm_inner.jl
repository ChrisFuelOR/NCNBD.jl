# The functions
# > "inner_loop_iteration",
# > "inner_loop_forward_pass",
# > "solve_subproblem_forward_inner",
# > "solve_all_children",
# > "solve_subproblem_backward",
# > "get_dual_variables_backward",
# > "calculate_bound",
# > "solve_first_stage_problem"
# > "inner_loop_forward_sigma_test"
# > "solve_subproblem_sigma_test"
# are derived from similar named functions (iteration, forward_pass, backward_pass,
# solve_all_children, solve_subproblem, get_dual_variables, calculate_bound,
# solve_first_stage_problem) in the 'SDDP.jl' package by
# Oscar Dowson and released under the Mozilla Public License 2.0.
# The reproduced function and other functions in this file are also released
# under Mozilla Public License 2.0

# Copyright (c) 2021 Christian Fuellner <christian.fuellner@kit.edu>
# Copyright (c) 2021 Oscar Dowson <o.dowson@gmail.com>

# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
################################################################################

"""
Executing an inner loop iteration for NC-NBD.
"""
function inner_loop_iteration(
    model::SDDP.PolicyGraph{T},
    options::NCNBD.Options,
    algoParams::NCNBD.AlgoParams,
    appliedSolvers::NCNBD.AppliedSolvers,
    previousSolution::Union{Vector{Dict{Symbol,Float64}},Nothing},
    boundCheck::Bool,
    sigma_increased::Bool
    ) where {T}

    # ITERATION COUNTER
    ############################################################################
    if haskey(model.ext, :iteration)
        model.ext[:iteration] += 1
    else
        model.ext[:iteration] = 1
    end

    # FORWARD PASS
    ############################################################################
    TimerOutputs.@timeit NCNBD_TIMER "forward_pass" begin
        forward_trajectory = NCNBD.inner_loop_forward_pass(model, options, algoParams, appliedSolvers, options.forward_pass)
    end

    #@infiltrate model.ext[:iteration] == 13

    # BINARY REFINEMENT
    ############################################################################
    # If the forward pass solution did not change during the last iteration, then
    # increase the binary precision (for all stages)
    solutionCheck = true
    binaryRefinement = :none

    if !isnothing(previousSolution)
        TimerOutputs.@timeit NCNBD_TIMER "bin_refinement" begin
            solutionCheck = NCNBD.binary_refinement_check(model, previousSolution, forward_trajectory.sampled_states, solutionCheck)
            if solutionCheck || boundCheck
                # Increase binary precision such that K = K + 1
                binaryRefinement = NCNBD.binary_refinement(model, algoParams, binaryRefinement)
            end
        end
    end
    boundCheck = true

    @infiltrate algoParams.infiltrate_state in [:all, :inner]

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

    # CALCULATE LOWER BOUND
    ############################################################################
    TimerOutputs.@timeit NCNBD_TIMER "calculate_bound" begin
        first_stage_results = calculate_bound(model)
    end
    bound = first_stage_results.bound
    subproblem_size = first_stage_results.problem_size[1]

    # CHECK IF BEST KNOWN SOLUTION HAS BEEN IMPROVED
    ############################################################################
    if model.objective_sense == JuMP.MOI.MIN_SENSE
        if forward_trajectory.cumulative_value < model.ext[:best_inner_loop_objective] || sigma_increased
            # udpate best upper bound
            model.ext[:best_inner_loop_objective] = forward_trajectory.cumulative_value
            # update best point so far
            model.ext[:best_inner_loop_point] = forward_trajectory.sampled_states
        end
    else
        if forward_trajectory.cumulative_value > model.ext[:best_inner_loop_objective] || sigma_increased
            # udpate best lower bound
            model.ext[:best_inner_loop_objective] = forward_trajectory.cumulative_value
            # update best point so far
            model.ext[:best_inner_loop_point] = forward_trajectory.sampled_states
        end
    end

    # PREPARE LOGGING
    ############################################################################
    model.ext[:total_cuts] = 0
    model.ext[:active_cuts] = 0

    for (node_index, children) in model.nodes
        node = model.nodes[node_index]

        if length(node.children) == 0
            continue
        end
        model.ext[:total_cuts] += node.ext[:total_cuts]
        model.ext[:active_cuts] += node.ext[:active_cuts]
    end

    push!(
         options.log_inner,
         Log(
             model.ext[:outer_iteration],
             model.ext[:iteration], #length(options.log) + 1,
             bound,
             model.ext[:best_inner_loop_objective],
             forward_trajectory.cumulative_value,
             forward_trajectory.sampled_states,
             time() - options.start_time,
             #Distributed.myid(),
             #model.ext[:total_solves],
             #algoParams.sigma,
             #algoParams.binaryPrecision,
             sigma_increased,
             binaryRefinement,
             subproblem_size,
             algoParams.epsilon_innerLoop,
             model.ext[:lag_iterations],
             model.ext[:lag_status],
             model.ext[:total_cuts],
             model.ext[:active_cuts],
         ),
     )

    # CHECK IF THE INNER LOOP CONVERGED YET
    ############################################################################
    has_converged, status = convergence_test(model, options.log_inner, options.stopping_rules, :inner)

    @infiltrate algoParams.infiltrate_state in [:all, :inner]

    return NCNBD.InnerLoopIterationResult(
        #Distributed.myid(),
        bound,
        model.ext[:best_inner_loop_objective],
        model.ext[:best_inner_loop_point],
        forward_trajectory.scenario_path,
        has_converged,
        status,
        cuts,
    )
end


"""
Executing the forward pass of an inner loop iteration for NC-NBD.
"""
function inner_loop_forward_pass(model::SDDP.PolicyGraph{T}, options::NCNBD.Options,
    algoParams::NCNBD.AlgoParams, appliedSolvers::NCNBD.AppliedSolvers,
    ::SDDP.DefaultForwardPass) where {T}

    # SAMPLING AND INITIALIZATION (JUST LIKE IN SDDP)
    ############################################################################
    # First up, sample a scenario. Note that if a cycle is detected, this will
    # return the cycle node as well.
    #TimerOutputs.@timeit NCNBD_TIMER "sample_scenario" begin
    scenario_path, terminated_due_to_cycle =
            SDDP.sample_scenario(model, options.sampling_scheme)
    #end

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

        if appliedSolvers.MILP == "CPLEX"
            set_optimizer(linearizedSubproblem, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>appliedSolvers.MILP, "optcr"=>0.0, "numericalemphasis"=>0))
        elseif appliedSolvers.MILP == "Gurobi"
            set_optimizer(linearizedSubproblem, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>appliedSolvers.MILP, "optcr"=>0.0, "NumericFocus"=>1))
        else
            set_optimizer(linearizedSubproblem, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>appliedSolvers.MILP, "optcr"=>0.0))
        end

        # SUBPROBLEM SOLUTION
        ############################################################################
        # Solve the subproblem, note that `require_duals = false`.
        TimerOutputs.@timeit NCNBD_TIMER "solve_FP" begin
            subproblem_results = solve_subproblem_forward_inner(
                model,
                node,
                node_index,
                incoming_state_value, # only values, no State struct!
                noise,
                scenario_path[1:depth],
                sigma,
                algoParams.infiltrate_state,
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


"""
Solving the subproblem within the forward pass of an inner loop of NC-NBD.
"""
# Internal function: solve the subproblem associated with node given the
# incoming state variables state and realization of the stagewise-independent
# noise term noise. If require_duals=true, also return the dual variables
# associated with the fixed constraint of the incoming state variables.
function solve_subproblem_forward_inner(
    model::SDDP.PolicyGraph{T},
    node::SDDP.Node{T},
    node_index::Int64,
    state::Dict{Symbol,Float64},
    noise,
    scenario_path::Vector{Tuple{T,S}},
    sigma::Float64,
    infiltrate_state::Symbol;
    require_duals::Bool,
) where {T,S}
    #TODO: We can actually delete the duals part here

    # MODEL PARAMETRIZATION (-> LINEARIZED SUBPROBLEM!)
    ############################################################################
    linearizedSubproblem = node.ext[:linSubproblem]

    # Parameterize the model. First, fix the value of the incoming state
    # variables. Then parameterize the model depending on `noise`. Finally,
    # set the objective.
    set_incoming_lin_state(node, state)
    parameterize_lin(node, noise)

    # pre_optimize_ret = if node.pre_optimize_hook !== nothing
    #     node.pre_optimize_hook(model, node, state, noise, scenario_path, require_duals)
    # else
    #     nothing
    # end

    # REGULARIZE SUBPROBLEM
    ############################################################################
    if node_index > 1
        # storage for regularization data
        node.ext[:regularization_data] = Dict{Symbol,Any}()

        regularize_subproblem!(node, linearizedSubproblem, sigma)
    end

    # SOLUTION
    ############################################################################
    @infiltrate infiltrate_state in [:all, :inner]
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
    objective = JuMP.objective_value(node.ext[:linSubproblem])
    stage_objective = objective - JuMP.value(bellman_term(node.ext[:lin_bellman_function]))
    @infiltrate infiltrate_state in [:all, :inner]

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

    # DE-REGULARIZE SUBPROBLEM
    ############################################################################
    if node_index > 1
        deregularize_subproblem!(node, linearizedSubproblem)
    end

    return (
        state = state,
        duals = dual_values,
        objective = objective,
        stage_objective = stage_objective,
    )
end


"""
Executing the backward pass of an inner loop of NC-NBD.
"""
function inner_loop_backward_pass(
    model::SDDP.PolicyGraph{T},
    options::NCNBD.Options,
    algoParams::NCNBD.AlgoParams,
    appliedSolvers::NCNBD.AppliedSolvers,
    scenario_path::Vector{Tuple{T,NoiseType}},
    sampled_states::Vector{Dict{Symbol,Float64}},
    objective_states::Vector{NTuple{N,Float64}},
    belief_states::Vector{Tuple{Int,Dict{T,Float64}}}) where {T,NoiseType,N}

    cuts = Dict{T,Vector{Any}}(index => Any[] for index in keys(model.nodes))

    model.ext[:lag_iterations] = Int[]
    model.ext[:lag_status] = Symbol[]

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
            #epsilon = algoParams.binaryPrecision[node_index]
            for (name, value) in outgoing_state
                state_comp = node.ext[:lin_states][name]
                epsilon = algoParams.binaryPrecision[name]
                (approx_state_value, )  = determine_used_trial_states(state_comp, value, epsilon)
                used_trial_points[name] = approx_state_value
            end

            @infiltrate algoParams.infiltrate_state in [:all, :inner]

            # REFINE BELLMAN FUNCTION BY ADDING CUTS
            ####################################################################
            TimerOutputs.@timeit NCNBD_TIMER "update_bellman" begin
                new_cuts = refine_bellman_function(
                    model,
                    node,
                    node_index,
                    node.bellman_function,
                    options.risk_measures[node_index],
                    outgoing_state,
                    used_trial_points,
                    items.bin_state,
                    items.duals,
                    items.supports,
                    items.probability,
                    items.objectives,
                    algoParams,
                    appliedSolvers
                )
            end
            push!(cuts[node_index], new_cuts)
            push!(model.ext[:lag_iterations], sum(items.lag_iterations))
            #TODO: Has to be adapted for stochastic case
            push!(model.ext[:lag_status], items.lag_status[1])

            #TODO: No cut-sharing allowed so far
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


"""
Solving all children within the backward pass.
"""
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
                push!(items.bin_state, items.bin_state[sol_index])
                push!(items.lag_iterations, items.lag_iterations[sol_index])
                push!(items.lag_status, items.lag_status[sol_index])
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
                TimerOutputs.@timeit NCNBD_TIMER "solve_BP" begin
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
                push!(items.bin_state, subproblem_results.bin_state)
                push!(items.lag_iterations, subproblem_results.iterations)
                push!(items.lag_status, subproblem_results.lag_status)
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


"""
Solving the backward pass problem for one specific child
"""
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
    set_incoming_lin_state(node, state)
    parameterize(node, noise)

    # pre_optimize_ret = if node.pre_optimize_hook !== nothing
    #     node.pre_optimize_hook(model, node, state, noise, scenario_path, require_duals)
    # else
    #     nothing
    # end

    # BACKWARD PASS PREPARATION
    ############################################################################
    @infiltrate algoParams.infiltrate_state in [:all, :inner]

    # Also adapt solver here
    TimerOutputs.@timeit NCNBD_TIMER "space_change" begin
        changeToBinarySpace!(node, linearizedSubproblem, state, algoParams.binaryPrecision)
    end

    # INITIALIZE DUALS
    ############################################################################
    TimerOutputs.@timeit NCNBD_TIMER "dual_initialization" begin
        dual_vars_initial = initialize_duals(node, linearizedSubproblem, algoParams.dual_initialization_regime)
    end

    # reset solver as it may have been changed
    if appliedSolvers.MILP == "CPLEX"
        set_optimizer(linearizedSubproblem, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>appliedSolvers.MILP, "optcr"=>0.0, "numericalemphasis"=>0))
    elseif appliedSolvers.MILP == "Gurobi"
        set_optimizer(linearizedSubproblem, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>appliedSolvers.MILP, "optcr"=>0.0, "NumericFocus"=>1))
    else
        set_optimizer(linearizedSubproblem, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>appliedSolvers.MILP, "optcr"=>0.0))
    end

    # REGULARIZE ALSO FOR BACKWARD PASS (FOR PRIMAL SOLUTION TO BOUND LAGRANGIAN DUAL)
    ############################################################################
    @infiltrate algoParams.infiltrate_state in [:all, :inner]

    node.ext[:regularization_data] = Dict{Symbol,Any}()
    regularize_backward!(node, linearizedSubproblem, algoParams.sigma[node_index])

    # PRIMAL SOLUTION
    ############################################################################
    TimerOutputs.@timeit NCNBD_TIMER "solve_primal_reg" begin
        JuMP.optimize!(linearizedSubproblem)
    end

    if haskey(model.ext, :total_solves)
        model.ext[:total_solves] += 1
    else
        model.ext[:total_solves] = 1
    end

    # if JuMP.primal_status(node.subproblem) != JuMP.MOI.FEASIBLE_POINT
    #     attempt_numerical_recovery(node)
    # end

    solver_obj = JuMP.objective_value(linearizedSubproblem)
    @assert JuMP.termination_status(linearizedSubproblem) == MOI.OPTIMAL

    @infiltrate algoParams.infiltrate_state in [:all, :inner]

    # PREPARE ACTUAL BACKWARD PASS METHOD BY DEREGULARIZATION
    ############################################################################
    deregularize_backward!(node, linearizedSubproblem)

    # DUAL SOLUTION
    ############################################################################
    # If require_duals = true, check for dual feasibility and return a dict with
    # the dual on the fixed constraint associated with each incoming state
    # variable. If require_duals=false, return an empty dictionary for
    # type-stability.
    if require_duals
        TimerOutputs.@timeit NCNBD_TIMER "solve_lagrange" begin
            lagrangian_results = get_dual_variables_backward(node, node_index, solver_obj, algoParams, appliedSolvers, dual_vars_initial)
        end
        dual_values = lagrangian_results.dual_values
        bin_state = lagrangian_results.bin_state
        objective = lagrangian_results.intercept
        iterations = lagrangian_results.iterations
        lag_status = lagrangian_results.lag_status
    else
        #NOTE: not required in my implementation
        dual_values = Dict{Symbol,Float64}()
        bin_state = Dict{Symbol,BinaryState}()
        objective = solver_obj
        iterations = 0
        lag_status = :none
    end

    @infiltrate algoParams.infiltrate_state in [:all, :inner]

    # if node.post_optimize_hook !== nothing
    #     node.post_optimize_hook(pre_optimize_ret)
    # end

    #TODO: REGAIN ORIGINAL MODEL
    ############################################################################
    TimerOutputs.@timeit NCNBD_TIMER "space_change" begin
        changeToOriginalSpace!(node, linearizedSubproblem, state)
    end

    return (
        duals = dual_values,
        bin_state = bin_state,
        objective = objective,
        iterations = iterations,
        lag_status = lag_status,
    )
end


"""
Calling the Lagrangian dual solution method and determining dual variables
required to construct a cut
"""
function get_dual_variables_backward(
    node::SDDP.Node,
    node_index::Int64,
    solver_obj::Float64,
    algoParams::NCNBD.AlgoParams,
    appliedSolvers::NCNBD.AppliedSolvers,
    dual_vars_initial::Vector{Float64}
    )

    # storages for return of dual values and binary state values (trial point)
    dual_values = Dict{Symbol,Float64}()
    bin_state = Dict{Symbol, BinaryState}()

    # TODO implement smart choice for initial duals
    number_of_states = length(node.ext[:backward_data][:bin_states])
    # dual_vars = zeros(number_of_states)
    #solver_obj = JuMP.objective_value(node.ext[:linSubproblem])
    dual_vars = dual_vars_initial

    lag_obj = 0
    lag_iterations = 0
    lag_status = :none

    # Create an SDDiP integrality_handler here to store the Lagrangian dual information
    #TODO: Store tolerances in algoParams
    integrality_handler = SDDP.SDDiP(iteration_limit = algoParams.lagrangian_iteration_limit, atol = algoParams.lagrangian_atol, rtol = algoParams.lagrangian_rtol)
    integrality_handler = SDDP.update_integrality_handler!(integrality_handler, appliedSolvers.MILP, number_of_states)
    node.ext[:lagrange] = integrality_handler

    # DETERMINE AND ADD BOUNDS FOR DUALVARIABLES
    ############################################################################
    #TODO: Determine a norm of B (coefficient matrix of binary expansion)
    # We use the column sum norm here
    # But instead of calculating it exactly, we can also use the maximum
    # upper bound of all state variables as a bound

    B_norm_bound = 0
    for (name, state_comp) in node.ext[:lin_states]
        if state_comp.info.in.upper_bound > B_norm_bound
            B_norm_bound = state_comp.info.in.upper_bound
        end
    end
    dual_bound = algoParams.sigma[node_index] * B_norm_bound

    @infiltrate algoParams.infiltrate_state in [:all, :inner, :lagrange] #|| node.ext[:linSubproblem].ext[:sddp_policy_graph].ext[:iteration] == 12
    try
        # SOLUTION WITHOUT BOUNDED DUAL VARIABLES (BETTER TO OBTAIN BASIC SOLUTIONS)
        ########################################################################
        if algoParams.lagrangian_method == :kelley
            results = _kelley(node, node_index, solver_obj, dual_vars, integrality_handler, algoParams, appliedSolvers, nothing)
            lag_obj = results.lag_obj
            lag_iterations = results.iterations
            lag_status = results.lag_status
        elseif algoParams.lagrangian_method == :bundle_level
            results = _bundle_level(node, node_index, solver_obj, dual_vars, integrality_handler, algoParams, appliedSolvers, nothing)
            lag_obj = results.lag_obj
            lag_iterations = results.iterations
            lag_status = results.lag_status
        end

        # OPTIMAL VALUE CHECKS
        ########################################################################
        if algoParams.lag_status_regime == :rigorous
            if lag_status == :conv
                error("Lagrangian dual converged to value < solver_obj.")
            elseif lag_status == :sub
                error("Lagrangian dual had subgradients zero without LB=UB.")
            elseif lag_status == :iter
                error("Solving Lagrangian dual exceeded iteration limit.")
            end

        elseif algoParams.lag_status_regime == :lax
            # all cuts will be used as they are valid even though not necessarily tight
        end

        # DUAL VARIABLE BOUND CHECK
        ########################################################################
        # if one of the dual variables exceeds the bounds (e.g. in case of an
        # discontinuous value function), use bounded version of Kelley's method
        boundCheck = true
        for dual_var in dual_vars
            if dual_var > dual_bound
                boundCheck = false
            end
        end

        @infiltrate algoParams.infiltrate_state in [:all, :inner, :lagrange]

    catch e
        SDDP.write_subproblem_to_file(node, "subproblem.mof.json", throw_error = false)
        rethrow(e)
    end

    # SET DUAL VARIABLES AND STATES CORRECTLY FOR RETURN
    ############################################################################
    for (i, name) in enumerate(keys(node.ext[:backward_data][:bin_states]))
        # TODO (maybe) change dual signs inside kelley to match LP duals
        dual_values[name] = -dual_vars[i]

        value = integrality_handler.old_rhs[i]
        x_name = node.ext[:backward_data][:bin_x_names][name]
        k = node.ext[:backward_data][:bin_k][name]
        bin_state[name] = BinaryState(value, x_name, k)
    end

    return (
        dual_values=dual_values,
        bin_state=bin_state,
        intercept=lag_obj,
        iterations=lag_iterations,
        lag_status=lag_status,
    )
end


"""
Calculate the lower bound (if minimizing, otherwise upper bound) of the problem
model at the point state.
"""
function calculate_bound(
    model::SDDP.PolicyGraph{T},
    root_state::Dict{Symbol,Float64} = model.initial_root_state;
    risk_measure = SDDP.Expectation(),
) where {T}

    # Note that here all children of the root node are solved, since the root
    # node is not node 1, but node 0.
    # In our case, this means that only stage 1 problem is solved again,
    # using the updated Bellman function from the backward pass.
    # NOTE: We could also implement this in our case such that only
    # the linearizedSubproblem of the first stage is solved and
    # the bound is returned.

    # Initialization.
    noise_supports = Any[]
    probabilities = Float64[]
    objectives = Float64[]
    problem_size = Dict{Symbol,Int64}[]
    current_belief = SDDP.initialize_belief(model)

    # Solve all problems that are children of the root node.
    for child in model.root_children
        if isapprox(child.probability, 0.0, atol = 1e-6)
            continue
        end
        node = model[child.term]
        for noise in node.noise_terms
            if node.objective_state !== nothing
                SDDP.update_objective_state(
                    node.objective_state,
                    node.objective_state.initial_value,
                    noise.term,
                )
            end
            # Update belief state, etc.
            if node.belief_state !== nothing
                belief = node.belief_state::SDDP.BeliefState{T}
                partition_index = belief.partition_index
                belief.updater(belief.belief, current_belief, partition_index, noise.term)
            end
            subproblem_results = solve_first_stage_problem(
                model,
                node,
                root_state,
                noise.term,
                Tuple{T,Any}[(child.term, noise.term)],
                require_duals = false,
            )
            push!(objectives, subproblem_results.objective)
            push!(probabilities, child.probability * noise.probability)
            push!(noise_supports, noise.term)
            push!(problem_size, subproblem_results.problem_size)
        end
    end
    # Now compute the risk-adjusted probability measure:
    risk_adjusted_probability = similar(probabilities)
    offset = SDDP.adjust_probability(
        risk_measure,
        risk_adjusted_probability,
        probabilities,
        noise_supports,
        objectives,
        model.objective_sense == MOI.MIN_SENSE,
    )
    # Finally, calculate the risk-adjusted value.
    return (bound = sum(obj * prob for (obj, prob) in zip(objectives, risk_adjusted_probability)) +
           offset, problem_size = problem_size)
end


"""
Solving the first-stage problem to determine a lower bound
"""
function solve_first_stage_problem(
    model::SDDP.PolicyGraph{T},
    node::SDDP.Node{T},
    state::Dict{Symbol,Float64},
    noise,
    scenario_path::Vector{Tuple{T,S}};
    require_duals::Bool,
) where {T,S}
    #TODO: We can actually delete the duals part here

    # MODEL PARAMETRIZATION (-> LINEARIZED SUBPROBLEM!)
    ############################################################################
    linearizedSubproblem = node.ext[:linSubproblem]

    # Parameterize the model. First, fix the value of the incoming state
    # variables. Then parameterize the model depending on `noise`. Finally,
    # set the objective.
    set_incoming_lin_state(node, state)
    parameterize_lin(node, noise)

    # pre_optimize_ret = if node.pre_optimize_hook !== nothing
    #     node.pre_optimize_hook(model, node, state, noise, scenario_path, require_duals)
    # else
    #     nothing
    # end

    # SOLUTION
    ############################################################################
    #set_optimizer(linearizedSubproblem, optimizer_with_attributes(Gurobi.Optimizer))
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

    # DETERMINE THE PROBLEM SIZE
    ############################################################################
    problem_size = Dict{Symbol,Int64}()
    problem_size[:total_var] = size(JuMP.all_variables(linearizedSubproblem),1)
    problem_size[:bin_var] = JuMP.num_constraints(linearizedSubproblem, VariableRef, MOI.ZeroOne)
    problem_size[:int_var] = JuMP.num_constraints(linearizedSubproblem, VariableRef, MOI.Integer)
    problem_size[:total_con] = JuMP.num_constraints(linearizedSubproblem, GenericAffExpr{Float64,VariableRef}, MOI.LessThan{Float64})
                                + JuMP.num_constraints(linearizedSubproblem, GenericAffExpr{Float64,VariableRef}, MOI.GreaterThan{Float64})
                                + JuMP.num_constraints(linearizedSubproblem, GenericAffExpr{Float64,VariableRef}, MOI.EqualTo{Float64})

    return (
        state = state,
        duals = dual_values,
        objective = objective,
        stage_objective = stage_objective,
        problem_size = problem_size
    )
end


"""
Executing a forward pass to check if parameter sigma is chosen sufficiently
large in NC-NBD.
"""
function inner_loop_forward_sigma_test(
    model::SDDP.PolicyGraph{T}, options::NCNBD.Options,
    algoParams::NCNBD.AlgoParams, appliedSolvers::NCNBD.AppliedSolvers,
    scenario_path::Vector{Tuple{T,S}}, ::SDDP.DefaultForwardPass,
    sigma_increased::Bool) where {T,S}

    # INITIALIZATION (NO SAMPLING HERE!)
    ############################################################################
    # scenario path is given as an argument

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

            incoming_state_value = splice!(starting_states, rand(1:length(starting_states)))
        end
        # ===== End: starting state for infinite horizon =====

        # Set optimizer to MILP optimizer
        linearizedSubproblem = node.ext[:linSubproblem]
        if appliedSolvers.MILP == "CPLEX"
            set_optimizer(linearizedSubproblem, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>appliedSolvers.MILP, "optcr"=>0.0, "numericalemphasis"=>1))
        elseif appliedSolvers.MILP == "Gurobi"
            set_optimizer(linearizedSubproblem, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>appliedSolvers.MILP, "optcr"=>0.0, "NumericFocus"=>1))
        else
            set_optimizer(linearizedSubproblem, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>appliedSolvers.MILP, "optcr"=>0.0))
        end

        # SOLVE REGULARIZED PROBLEM
        ############################################################################
        # This has to be done in this sigma test again, since the current upper bound
        # may not be the best upper bound and thus, just comparing the bounds is not
        # sufficient. Moreover, we do it in this loop to not increase sigma for
        # all stages, but only where it is needed

        # Solve the subproblem, note that `require_duals = false`.
        TimerOutputs.@timeit NCNBD_TIMER "solve_sigma_test_reg" begin
            reg_results = solve_subproblem_forward_inner(
                model,
                node,
                node_index,
                incoming_state_value, # only values, no State struct!
                noise,
                scenario_path[1:depth],
                algoParams.sigma[node_index],
                algoParams.infiltrate_state,
                require_duals = false,
            )
        end

        #@infiltrate

        # SOLVE NON-REGULARIZED PROBLEM
        ########################################################################
        # Solve the subproblem, note that `require_duals = false`.
        TimerOutputs.@timeit NCNBD_TIMER "solve_sigma_test" begin
            non_reg_results = solve_subproblem_sigma_test(
                model,
                node,
                incoming_state_value, # only values, no State struct!
                noise,
                scenario_path[1:depth],
                algoParams.sigma[node_index],
                algoParams.infiltrate_state,
                require_duals = false,
            )
        end

        # COMPARE SOLUTIONS
        ########################################################################
        if !isapprox(reg_results.objective, non_reg_results.objective)
            # if stage objectives are not approximately equal, then the
            # regularization is not exact and sigma should be increased
            algoParams.sigma[node_index] = algoParams.sigma[node_index] * algoParams.sigma_factor
            # marker if new inner loop iteration should be started
            # instead of heading to outer loop
            sigma_increased = true
        end

        # Cumulate the stage_objective. (NOTE: not really required anymore)
        cumulative_value += non_reg_results.stage_objective
        # Set the outgoing state value as the incoming state value for the next
        # node.
        # NOTE: We use the states determined by the regularized problem here
        #incoming_state_value = copy(subproblem_results.state)
        incoming_state_value = copy(reg_results.state)
        # Add the outgoing state variable to the list of states we have sampled
        # on this forward pass.
        push!(sampled_states, incoming_state_value)

    end
    # if terminated_due_to_cycle
    #     # Get the last node in the scenario.
    #     final_node_index = scenario_path[end][1]
    #     # We terminated due to a cycle. Here is the list of possible starting
    #     # states for that node:
    #     starting_states = options.starting_states[final_node_index]
    #     # We also need the incoming state variable to the final node, which is
    #     # the outgoing state value of the last node:
    #     incoming_state_value = sampled_states[end]
    #     # If this incoming state value is more than δ away from another state,
    #     # add it to the list.
    #     if distance(starting_states, incoming_state_value) >
    #        options.cycle_discretization_delta
    #         push!(starting_states, incoming_state_value)
    #     end
    # end

    # ===== End: drop off starting state if terminated due to cycle =====
    return (
        scenario_path = scenario_path,
        sampled_states = sampled_states,
        objective_states = objective_states,
        belief_states = belief_states,
        cumulative_value = cumulative_value,
        sigma_increased = sigma_increased,
    )
end


"""
Solving the subproblems for the sigma test.
"""
function solve_subproblem_sigma_test(
    model::SDDP.PolicyGraph{T},
    node::SDDP.Node{T},
    state::Dict{Symbol,Float64},
    noise,
    scenario_path::Vector{Tuple{T,S}},
    sigma::Float64,
    infiltrate_state::Symbol;
    require_duals::Bool,
) where {T,S}
    #TODO: We can actually delete the duals part here

    # MODEL PARAMETRIZATION (-> LINEARIZED SUBPROBLEM!)
    ############################################################################
    linearizedSubproblem = node.ext[:linSubproblem]

    # Parameterize the model. First, fix the value of the incoming state
    # variables. Then parameterize the model depending on `noise`. Finally,
    # set the objective.
    set_incoming_lin_state(node, state)
    parameterize_lin(node, noise)

    # pre_optimize_ret = if node.pre_optimize_hook !== nothing
    #     node.pre_optimize_hook(model, node, state, noise, scenario_path, require_duals)
    # else
    #     nothing
    # end

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
    objective = JuMP.objective_value(node.ext[:linSubproblem])
    stage_objective = objective - JuMP.value(bellman_term(node.ext[:lin_bellman_function])) #JuMP.value(node.ext[:lin_stage_objective])

    #@infiltrate model.ext[:iteration] == 14

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

    return (
        state = state,
        duals = dual_values,
        objective = objective,
        stage_objective = stage_objective,
    )
end
