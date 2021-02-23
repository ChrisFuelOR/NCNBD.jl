const NCNBD_TIMER = TimerOutputs.TimerOutput()

function outer_loop_iteration(parallel_scheme::SDDP.Serial, model::SDDP.PolicyGraph{T},
    options::NCNBD.Options, algoParams::NCNBD.AlgoParams, appliedSolvers::NCNBD.AppliedSolvers) where {T}

    # ITERATION COUNTER
    ############################################################################
    if haskey(model.ext, :outer_iteration)
        model.ext[:outer_iteration] += 1
    else
        model.ext[:outer_iteration] = 1
    end

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

    # CHECK IF BEST KNOWN SOLUTION HAS BEEN IMPROVED
    ############################################################################
    if model.objective_sense == JuMP.MOI.MIN_SENSE
        if forward_results.cumulative_value < model.ext[:best_outer_loop_objective]
            # udpate best upper bound
            model.ext[:best_outer_loop_objective] = forward_results.cumulative_value
            # update best point so far
            model.ext[:best_outer_loop_point] = forward_results.sampled_states
        end
    else
        if forward_trajectory.cumulative_value > model.ext[:best_outer_loop_objective]
            # udpate best lower bound
            model.ext[:best_outer_loop_objective] = forward_results.cumulative_value
            # update best point so far
            model.ext[:best_outer_loop_point] = forward_results.sampled_states
        end
    end

    # LOGGING RESULTS?
    ############################################################################

    @infiltrate algoParams.infiltrate_state in [:all, :outer]
    push!(
        options.log_outer,
        Log(
            model.ext[:outer_iteration], #length(options.log) + 1,
            nothing,
            #inner_loop_results.lower_bound,
            forward_results.first_stage_objective,
            model.ext[:best_outer_loop_objective],
            forward_results.cumulative_value,
            forward_results.sampled_states,
            time() - options.start_time,
            #Distributed.myid(),
            #model.ext[:total_solves],
            #algoParams.sigma,
            #algoParams.binaryPrecision,
            nothing,
            nothing,
            nothing,
            algoParams.epsilon_outerLoop,
            nothing,
            model.ext[:total_cuts],
            model.ext[:active_cuts],
        ),
    )

    # CONVERGENCE TEST
    ############################################################################
    has_converged, status = convergence_test(model, options.log_outer, options.stopping_rules, :outer)

    @infiltrate algoParams.infiltrate_state in [:all, :outer]

    # RETURN RESULTS
    ############################################################################
    return NCNBD.OuterLoopIterationResult(
        #Distributed.myid(),
        #inner_loop_results.lower_bound,
        forward_results.first_stage_objective,
        forward_results.cumulative_value,
        forward_results.sampled_states,
        has_converged,
        status
        #cuts,
    )
end

function outer_loop_forward_pass(model::SDDP.PolicyGraph{T},
    options::NCNBD.Options, algoParams::NCNBD.AlgoParams, appliedSolvers::NCNBD.AppliedSolvers) where {T}

    # SAMPLING AND INITIALIZATION (JUST LIKE IN SDDP)
    ############################################################################
    # First up, sample a scenario. Note that if a cycle is detected, this will
    # return the cycle node as well.
    #TimerOutputs.@timeit NCNBD_TIMER "sample_scenario_outer" begin
    scenario_path, terminated_due_to_cycle =
            SDDP.sample_scenario(model, options.sampling_scheme)
    #end
    # Storage for the list of outgoing states that we visit on the forward pass.
    sampled_states = Dict{Symbol,Float64}[]
    # Storage for the belief states: partition index and the belief dictionary.
    belief_states = Tuple{Int,Dict{T,Float64}}[]
    current_belief = SDDP.initialize_belief(model)
    # Our initial incoming state.
    incoming_state_value = copy(options.initial_state)
    # A cumulator for the stage-objectives.
    cumulative_value = 0.0
    # First-stage optimal value
    first_stage_objective = 0.0
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
        set_optimizer(node.subproblem, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>appliedSolvers.MINLP, "optcr"=>0.0))
        #set_optimizer(node.subproblem, GAMS.Optimizer)
        #JuMP.set_optimizer_attribute(node.subproblem, "Solver", appliedSolvers.MINLP)
        #JuMP.set_optimizer_attribute(node.subproblem, "optcr", 0.0)

        # SUBPROBLEM SOLUTION
        ############################################################################
        # Solve the subproblem, note that `require_duals = false`.
        TimerOutputs.@timeit NCNBD_TIMER "solve_subproblem" begin
            subproblem_results = solve_subproblem_forward_outer(
                model,
                node,
                incoming_state_value, # no State struct!
                noise,
                scenario_path[1:depth],
                algoParams.infiltrate_state,
                require_duals = false,
            )
        end
        # Cumulate the stage_objective.
        cumulative_value += subproblem_results.stage_objective
        # Determine the first stage objective
        if node_index == 1
            first_stage_objective = subproblem_results.objective
        end
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
        first_stage_objective = first_stage_objective,
    )
end


function solve_subproblem_forward_outer(
    model::SDDP.PolicyGraph{T},
    node::SDDP.Node{T},
    state::Dict{Symbol,Float64},
    noise,
    scenario_path::Vector{Tuple{T,S}},
    infiltrate_state::Symbol;
    require_duals::Bool,
) where {T,S}
    #TODO: We can actually delete the duals part here

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

    # SOLVE THE MILP TO OBTAIN A BOUND ON THE MINLP
    ############################################################################
    set_incoming_lin_state(node, state)
    parameterize_lin(node, noise)
    linearizedSubproblem = node.ext[:linSubproblem]
    JuMP.optimize!(linearizedSubproblem)
    bound_value = JuMP.objective_value(node.ext[:linSubproblem])

    # BOUND THE MINLP OPTIMAL VALUE
    # This way, a lot of branch-and-cut nodes can be pruned early on
    ############################################################################
    if model.ext[:outer_iteration] == 1
        if model.objective_sense == JuMP.MOI.MIN_SENSE
            JuMP.@constraint(node.subproblem, bound_constr, JuMP.objective_function(node.subproblem) >= bound_value)
        else
            JuMP.@constraint(node.subproblem, bound_constr, JuMP.objective_function(node.subproblem) <= bound_value)
    else
        JuMP.set_normalized_rhs(bound_constr, bound_value)
    end

    # SOLVE THE MINLP
    ############################################################################
    JuMP.optimize!(node.subproblem)

    #if JuMP.primal_status(node.subproblem) != JuMP.MOI.FEASIBLE_POINT
    #    SDDP.attempt_numerical_recovery(node)
    #end

    state = SDDP.get_outgoing_state(node)
    stage_objective = SDDP.stage_objective_value(node.stage_objective)
    objective = JuMP.objective_value(node.subproblem)

    @infiltrate infiltrate_state in [:all, :outer]

    # If require_duals = true, check for dual feasibility and return a dict with
    # the dual on the fixed constraint associated with each incoming state
    # variable. If require_duals=false, return an empty dictionary for
    # type-stability.
    dual_values = if require_duals
        SDDP.get_dual_variables(node, node.integrality_handler)
    else
        Dict{Symbol,Float64}()
    end

    # if node.post_optimize_hook !== nothing
    #     node.post_optimize_hook(pre_optimize_ret)
    # end

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
