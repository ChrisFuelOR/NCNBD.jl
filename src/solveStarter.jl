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
    NCNBD.solve(model::PolicyGraph, algoParams::AlgoParams; kwargs...)

Solves the `model`. If used for stochastic multistage programs, this could
be used the same way as SDDP.train.

Keyword arguments:

 - `iteration_limit::Int`: number of iterations to conduct before termination.

 - `time_limit::Float64`: number of seconds to train before termination.

 - `stopping_rules`: a vector of [`SDDP.AbstractStoppingRule`](@ref)s or
    "deterministic". Defaults to `deterministic`.

 - `print_level::Int`: control the level of printing to the screen. Defaults to
    `1`. Set to `0` to disable all printing.

 - `log_file::String`: filepath at which to write a log of the training progress.
    Defaults to `SDDP.log`.

 - `log_frequency::Int`: control the frequency with which the logging is
    outputted (iterations/log). Defaults to `1`.

 - `run_numerical_stability_report::Bool`: generate (and print) a numerical stability
    report prior to solve. Defaults to `true`.

 - `refine_at_similar_nodes::Bool`: if SDDP can detect that two nodes have the
    same children, it can cheaply add a cut discovered at one to the other. In
    almost all cases this should be set to `true`.

 - `cut_deletion_minimum::Int`: the minimum number of cuts to cache before
    deleting  cuts from the subproblem. The impact on performance is solver
    specific; however, smaller values result in smaller subproblems (and therefore
    quicker solves), at the expense of more time spent performing cut selection.

 - `risk_measure`: the risk measure to use at each node. Defaults to [`Expectation`](@ref).

 - `sampling_scheme`: a sampling scheme to use on the forward pass of the
    algorithm. Defaults to [`InSampleMonteCarlo`](@ref).

 - `backward_sampling_scheme`: a backward pass sampling scheme to use on the
    backward pass of the algorithm. Defaults to `CompleteSampler`.

 - `cut_type`: choose between `SDDP.SINGLE_CUT` and `SDDP.MULTI_CUT` versions of SDDP.

 - `dashboard::Bool`: open a visualization of the training over time. Defaults
    to `false`.

 - `parallel_scheme::AbstractParallelScheme`: specify a scheme for solving in parallel.
    Defaults to `Serial()`.

 - `forward_pass::AbstractForwardPass`: specify a scheme to use for the forward passes.

There is also a special option for infinite horizon problems

 - `cycle_discretization_delta`: the maximum distance between states allowed on
    the forward pass. This is for advanced users only and needs to be used in
    conjunction with a different `sampling_scheme`.
"""

function solve(
    model::SDDP.PolicyGraph,
    algoParams::NCNBD.AlgoParams;
    iteration_limit::Union{Int,Nothing} = nothing,
    time_limit::Union{Real,Nothing} = nothing,
    print_level::Int = 1,
    log_file::String = "SDDP.log",
    log_frequency::Int = 1,
    run_numerical_stability_report::Bool = true,
    stopping_rules = SDDP.AbstractStoppingRule[],
    risk_measure = SDDP.Expectation(),
    sampling_scheme = SDDP.InSampleMonteCarlo(),
    cut_type = SDDP.SINGLE_CUT,
    cycle_discretization_delta::Float64 = 0.0,
    refine_at_similar_nodes::Bool = true,
    cut_deletion_minimum::Int = 1,
    backward_sampling_scheme::SDDP.AbstractBackwardSamplingScheme = SDDP.CompleteSampler(),
    dashboard::Bool = false,
    parallel_scheme::SDDP.AbstractParallelScheme = SDDP.Serial(),
    forward_pass::SDDP.AbstractForwardPass = SDDP.DefaultForwardPass(),
)

    # INITIALIZATION (AS IN SDDP)
    ############################################################################
    # Reset the TimerOutput.
    TimerOutputs.reset_timer!(NCNBD_TIMER)
    log_file_handle = open(log_file, "a")
    log = Log[]

    if print_level > 0
        print_helper(print_banner, log_file_handle)
    end

    if run_numerical_stability_report
        report =
            sprint(io -> numerical_stability_report(io, model, print = print_level > 0))
        print_helper(print, log_file_handle, report)
    end

    if print_level > 0
        print_helper(io -> println(io, "Solver: ", parallel_scheme, "\n"), log_file_handle)
        print_helper(print_iteration_header, log_file_handle)
    end

    # Convert the vector to an AbstractStoppingRule. Otherwise if the user gives
    # something like stopping_rules = [SDDP.IterationLimit(100)], the vector
    # will be concretely typed and we can't add a TimeLimit.
    stopping_rules = convert(Vector{AbstractStoppingRule}, stopping_rules)
    # Add the limits as stopping rules. An IterationLimit or TimeLimit may
    # already exist in stopping_rules, but that doesn't matter.
    if iteration_limit !== nothing
        push!(stopping_rules, IterationLimit(iteration_limit))
    end
    if time_limit !== nothing
        push!(stopping_rules, TimeLimit(time_limit))
    end
    if length(stopping_rules) == 0
        @warn(
            "You haven't specified a stopping rule! You can only terminate " *
            "the call to NCNBD.solve via a keyboard interrupt ([CTRL+C])."
        )
    end

    # Update the nodes with the selected cut type (SINGLE_CUT or MULTI_CUT)
    # and the cut deletion minimum.
    if cut_deletion_minimum < 0
        cut_deletion_minimum = typemax(Int)
    end
    for (key, node) in model.nodes
        node.bellman_function.cut_type = cut_type
        node.bellman_function.global_theta.cut_oracle.deletion_minimum =
            cut_deletion_minimum
        for oracle in node.bellman_function.local_thetas
            oracle.cut_oracle.deletion_minimum = cut_deletion_minimum
        end
    end

    # Perform relaxations required by integrality_handler
    # binaries, integers =
    #    relax_integrality(model, last(first(model.nodes)).integrality_handler)

    dashboard_callback = if dashboard
        launch_dashboard()
    else
        (::Any, ::Any) -> nothing
    end

    sddpOptions = Options(
        model,
        model.initial_root_state,
        sampling_scheme,
        backward_sampling_scheme,
        risk_measure,
        cycle_discretization_delta,
        refine_at_similar_nodes,
        stopping_rules,
        dashboard_callback,
        print_level,
        time(),
        log,
        log_file_handle,
        log_frequency,
        forward_pass,
    )

    # MODEL CHECK
    ############################################################################
    # count integer variables and nonlinear functions in the model
    counter_integer_variables = 0
    counter_nonlinear_functions = 0

    for (node_index, children = in model.nodes)
        node = model.nodes[node_index]
        m = node.subproblem

        for x in JuMP.all_variables(m)
            if JuMP.is_binary(x) or JuMP.is_integer(x)
                counter_integer_variables += 1
            end
        end

        if isempty(subproblem.ext[:nlFunctions])
        else
            counter_nonlinear_functions += 1
        end
    end

    # Depending on this, different solve functions are called.

    status = :not_solved
    try
        if counter_nonlinear_functions > 0
            # call ordinary NC-NBD with outer and inner loop
            status = solve_ncnbd(parallel_scheme, model, options, algoParams)
        elseif counter_integer_variables > 0
            # call NC-NBD with only inner loop
            status = solve_ncnbd_inner(parallel_scheme, model, options, algoParams)
        else
            # call SDDP
            status = SDDP.solve_sddp(parallel_scheme, model, options)
    catch ex
        if isa(ex, InterruptException)
            status = :interrupted
            interrupt(parallel_scheme)
        else
            close(log_file_handle)
            rethrow(ex)
        end
    finally
        # # Remember to reset any relaxed integralities.
        # enforce_integrality(binaries, integers)
        # # And close the dashboard callback if necessary.
        # dashboard_callback(nothing, true)
    end
    # TODO: Print/Log also initialAlgoParams
    # training_results = TrainingResults(status, log)
    # model.most_recent_training_results = training_results
    # if print_level > 0
    #     print_helper(print_footer, log_file_handle, training_results)
    #     if print_level > 1
    #         print_helper(TimerOutputs.print_timer, log_file_handle, SDDP_TIMER)
    #         # Annoyingly, TimerOutputs doesn't end the print section with `\n`,
    #         # so we do it here.
    #         print_helper(println, log_file_handle)
    #     end
    # end
    # close(log_file_handle)
    return
end

function solve_ncnbd(::SDDP.Serial, model::SDDP.PolicyGraph{T}, options::Options, algoParams::algoParams) where {T}

    # SHIFT LINEARIZED SUBPROBLEM AND NONLINEAR FUNCTION LIST TO NODES
    ############################################################################
    for (node_index, children) in model.nodes
        node = model.nodes[node_index]

        node.ext[:nlFunctions] = subproblem.ext[:nlFunctions]
        subproblem.ext[:nlFunctions] = Nothing

        node.ext[:linSubproblem] = subproblem.ext[:linSubproblem]
        subproblem.ext[:linSubproblem] = Nothing

        node.ext[:linSubproblem].ext[:sddp_node] = node
        node.ext[:linSubproblem].ext[:sddp_policy_graph] = model

    end

    # INITIALIZE PIECEWISE LINEAR RELAXATION
    ############################################################################
    for (node_index, children) in model.nodes
        node = model.nodes[node_index]

        # determines a piecewise linear relaxation for all nonlinear functions
        # in this node
        piecewiseLinearRelaxation!(node, algoParams)

    end

    # CALL ACTUAL SOLUTION PROCEDURE
    ############################################################################
    status = master_loop_ncbd(parallel_scheme, model, options, algoParams)
    return status

end

function solve_ncnbd_inner(::SDDP.Serial, model::SDDP.PolicyGraph{T}, options::Options, algoParams::algoParams) where {T}

    status = master_loop_ncbd_inner(parallel_scheme, model, options, algoParams)
    return status

end

function solve_sddp(::SDDP.Serial, model::SDDP.PolicyGraph{T}, options::Options) where {T}

    status = SDDP.master_loop(parallel_scheme, model, options)
    return status

end

function master_loop_ncbd(::SDDP.Serial, model::SDDP.PolicyGraph{T}, options::Options, algoParams::algoParams) where {T}
    while true
        result = outer_loop(model, options)
        log_iteration(options)
        if result.has_converged
            return result.status
        end
    end
end

function master_loop_ncbd_inner(::SDDP.Serial, model::SDDP.PolicyGraph{T}, options::Options, algoParams::algoParams) where {T}
    while true
        result = inner_loop(model, options)
        log_iteration(options)
        if result.has_converged
            return result.status
        end
    end
end