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
    NCNBD.solve(model::PolicyGraph, algoParams::AlgoParams, initialAlgoParams:InitialAlgoParams,
    appliedSolvers::AppliedSolvers ; kwargs...)

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
    algoParams::NCNBD.AlgoParams,
    initialAlgoParams::NCNBD.InitialAlgoParams,
    appliedSolvers::NCNBD.AppliedSolvers;
    iteration_limit::Union{Int,Nothing} = nothing,
    time_limit::Union{Real,Nothing} = nothing,
    print_level::Int = 1,
    log_file::String = "NCNBD.log",
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
    log_inner = Log[]
    log_outer = Log[]

    if print_level > 0
        print_helper(print_banner, log_file_handle)
    end

    if print_level > 1
        print_helper(print_parameters, log_file_handle, initialAlgoParams, appliedSolvers)
    end

    # if run_numerical_stability_report
    #     report =
    #         sprint(io -> SDDP.numerical_stability_report(io, model, print = print_level > 0))
    #     print_helper(print, log_file_handle, report)
    # end

    if print_level > 0
        print_helper(io -> println(io, "Solver: ", parallel_scheme, "\n"), log_file_handle)
        print_helper(print_iteration_header, log_file_handle)
    end

    # Convert the vector to an AbstractStoppingRule. Otherwise if the user gives
    # something like stopping_rules = [SDDP.IterationLimit(100)], the vector
    # will be concretely typed and we can't add a TimeLimit.
    stopping_rules = convert(Vector{SDDP.AbstractStoppingRule}, stopping_rules)
    # Add the limits as stopping rules. An IterationLimit or TimeLimit may
    # already exist in stopping_rules, but that doesn't matter.
    if iteration_limit !== nothing
        push!(stopping_rules, SDDP.IterationLimit(iteration_limit))
    end
    if time_limit !== nothing
        push!(stopping_rules, SDDP.TimeLimit(time_limit))
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
        log_inner,
        log_outer,
        log_file_handle,
        log_frequency,
        forward_pass,
    )

    # MODEL CHECK
    ############################################################################
    # count integer variables and nonlinear functions in the model
    counter_integer_variables = 0
    counter_nonlinear_functions = 0

    for (node_index, children) in model.nodes
        node = model.nodes[node_index]
        m = node.subproblem

        for x in JuMP.all_variables(m)
            if JuMP.is_binary(x) || JuMP.is_integer(x)
                counter_integer_variables += 1
            end
        end

        if isempty(node.subproblem.ext[:nlFunctions])
        else
            counter_nonlinear_functions += 1
        end
    end

    # Depending on this, different solve functions are called.

    status = :not_solved
    try
        if counter_nonlinear_functions >= 0 #TODO later
            # call ordinary NC-NBD with outer and inner loop
            status = solve_ncnbd(parallel_scheme, model, sddpOptions, algoParams, initialAlgoParams, appliedSolvers)
        elseif counter_integer_variables > 0
            # call NC-NBD with only inner loop
            status = solve_ncnbd_inner(parallel_scheme, model, sddpOptions, algoParams, appliedSolvers)
        else
            # call SDDP
            status = SDDP.solve_sddp(parallel_scheme, model, sddpOptions)
        end
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
    ncnbd_results = Results(status, log_inner, log_outer)
    model.ext[:results] = ncnbd_results
    if print_level > 0
        print_helper(print_footer, log_file_handle, ncnbd_results)
        if print_level > 1
            print_helper(TimerOutputs.print_timer, log_file_handle, NCNBD_TIMER)
            # Annoyingly, TimerOutputs doesn't end the print section with `\n`,
            # so we do it here.
            print_helper(println, log_file_handle)
        end
    end
    close(log_file_handle)
    return
end

"""
    NCNBD.solve_ncnbd(parallel_scheme::SDDP.Serial, model::PolicyGraph, options:SDDP.Options,
    algoParams::AlgoParams, initialAlgoParams:InitialAlgoParams, appliedSolvers::AppliedSolvers ; kwargs...)

Solves the `model` using NCNBD in a serial scheme.
"""

function solve_ncnbd(parallel_scheme::SDDP.Serial, model::SDDP.PolicyGraph{T},
    options::NCNBD.Options, algoParams::NCNBD.AlgoParams,
    initialAlgoParams::NCNBD.InitialAlgoParams, appliedSolvers::NCNBD.AppliedSolvers) where {T}

    # SET UP LINEARIZED SUBPROBLEM DATA
    ############################################################################
    for (node_index, children) in model.nodes
        node = model.nodes[node_index]

        node.ext[:nlFunctions] = node.subproblem.ext[:nlFunctions]
        node.subproblem.ext[:nlFunctions] = nothing

        node.ext[:linSubproblem] = node.subproblem.ext[:linSubproblem]
        node.subproblem.ext[:linSubproblem] = nothing

        # Set info for x_in (taking bounds, binary, integer info from previous stage's x_out)
        #-----------------------------------------------------------------------
        if node_index > 1

            for (i, (name, state)) in enumerate(node.ext[:lin_states])
                # Get correct state_info
                state_info = model.nodes[node_index-1].ext[:lin_states][name].info.out

                if state_info.has_lb
                    JuMP.set_lower_bound(state.in, state_info.lower_bound)
                end
                if state_info.has_ub
                    JuMP.set_upper_bound(state.in, state_info.upper_bound)
                end
                if state_info.binary
                    JuMP.set_binary(state.in)
                elseif state_info.integer
                    JuMP.set_integer(state.in)
                end

                # Store info to reset it later
                state.info.in = state_info
            end
        end

        # Set objective sense
        JuMP.set_objective_sense(node.ext[:linSubproblem], model.objective_sense)

        # Initialize Bellman function
        node.ext[:lin_bellman_function] = nothing
        if node_index != model.root_node

            if model.objective_sense == MOI.MIN_SENSE
                lower_bound = JuMP.lower_bound(node.bellman_function.global_theta.theta)
                upper_bound = Inf
            elseif model.objective_sense == MOI.MAX_SENSE
                upper_bound = JuMP.upper_bound(node.bellman_function.global_theta.theta)
                lower_bound = -Inf
            end

            cut_type = node.bellman_function.cut_type
            deletion_minimum = node.bellman_function.global_theta.cut_oracle.deletion_minimum

            bellman_function = BellmanFunction(lower_bound = lower_bound, upper_bound = upper_bound)
            node.ext[:lin_bellman_function] = initialize_bellman_function_MILP(bellman_function, model, node)
            node.ext[:lin_bellman_function].cut_type = cut_type
            node.ext[:lin_bellman_function].global_theta.cut_oracle.deletion_minimum = deletion_minimum
            for oracle in node.ext[:lin_bellman_function].local_thetas
                oracle.cut_oracle.deletion_minimum = deletion_minimum
                #oracle.cut_oracle.deletion_minimum = node.oracle.cut_oracle.deletion_minimum
            end

            # also re-initialize the existing value function such that nonlinear cuts are used
            # fortunately, node.bellman_function requires no specific type
            node.bellman_function = initialize_bellman_function_MINLP(bellman_function, model, node)
            node.bellman_function.cut_type = cut_type
            node.bellman_function.global_theta.cut_oracle.deletion_minimum = deletion_minimum
            for oracle in node.bellman_function.local_thetas
                oracle.cut_oracle.deletion_minimum = deletion_minimum
            end

            #JuMP.set_silent(node.subproblem)
            #JuMP.set_silent(node.ext[:linSubproblem])

        end
    end

    @infiltrate algoParams.infiltrate_state == :all

    # INITIALIZE PIECEWISE LINEAR RELAXATION
    ############################################################################
    for (node_index, children) in model.nodes
        node = model.nodes[node_index]

        # get corresponding pla precision (node_index is stage for non-Markovian policy graphs)
        plaPrecision = initialAlgoParams.plaPrecision[node_index]

        # determines a piecewise linear relaxation for all nonlinear functions
        # in this node
        TimerOutputs.@timeit NCNBD_TIMER "initialize_PLR" begin
            NCNBD.piecewiseLinearRelaxation!(node, plaPrecision, appliedSolvers)
        end
    end

    @infiltrate algoParams.infiltrate_state == :all

    # CALL ACTUAL SOLUTION PROCEDURE
    ############################################################################
    status = master_loop_ncnbd(parallel_scheme, model, options, algoParams, appliedSolvers)
    return status

end

"""
    NCNBD.solve_ncnbd_inner(parallel_scheme::SDDP.Serial, model::PolicyGraph, options:SDDP.Options,
    algoParams::AlgoParams, initialAlgoParams:InitialAlgoParams, appliedSolvers::AppliedSolvers ; kwargs...)

Solves the `model` (MILP) using the inner loop of NCNBD in a serial scheme.
"""

function solve_ncnbd_inner(parallel_scheme::SDDP.Serial, model::SDDP.PolicyGraph{T},
    options::NCNBD.Options, algoParams::NCNBD.AlgoParams, appliedSolvers::NCNBD.AppliedSolvers) where {T}

    status = master_loop_ncnbd_inner(parallel_scheme, model, options, algoParams, appliedSolvers)

    return status

    # TODO: Hier muss man linearizedSubproblem = subproblem setzen.
    # TODO: Hier muss man node.ext[:lin_states] auf Node.states setzen.
    # TODO: Hier muss man model.ext[:lin_initial_root_state] auf model.initial_root_state setzen.
    # Dann ist sichergestellt, dass in der inner_loop auch subproblem anstelle
    # von linearizedSubproblem richtig gelöst wird
end

"""
    NCNBD.solve_sddp(parallel_scheme::SDDP.Serial, model::PolicyGraph, options:SDDP.Options)

Solves the `model` (LP) using SDDP in a serial scheme.
"""

function solve_sddp(parallel_scheme::SDDP.Serial, model::SDDP.PolicyGraph{T}, options::NCNBD.Options) where {T}

    status = SDDP.master_loop(parallel_scheme, model, options)
    return status
    #create sddp options from ncnbd options

end

"""
    NCNBD.master_loop_ncbd(parallel_scheme::SDDP.Serial, model::SDDP.PolicyGraph{T},
        options::SDDP.Options, algoParams::NCNBD.AlgoParams, appliedSolvers::NCNBD.AppliedSolvers)

Outer loop function of NCNBD.
"""

function master_loop_ncnbd(parallel_scheme::SDDP.Serial, model::SDDP.PolicyGraph{T},
    options::NCNBD.Options, algoParams::NCNBD.AlgoParams, appliedSolvers::NCNBD.AppliedSolvers) where {T}

    #previousSolution = nothing

    # INITIALIZE BEST KNOWN POINT AND OBJECTIVE VALUE FOR INNER LOOP
    ############################################################################
    model.ext[:best_outer_loop_objective] = model.objective_sense == JuMP.MOI.MIN_SENSE ? Inf : -Inf
    model.ext[:best_outer_loop_point] = Vector{Dict{Symbol,Float64}}()

    while true
        TimerOutputs.@timeit NCNBD_TIMER "outer_loop" begin
            result_outer = outer_loop_iteration(parallel_scheme, model, options, algoParams, appliedSolvers)
        end

        @infiltrate algoParams.infiltrate_state in [:all, :outer]

        log_iteration(options, options.log_outer)
        if result_outer.has_converged
            return result_outer.status
        end

       # Piecewise linear refinement
       TimerOutputs.@timeit NCNBD_TIMER "refine_PLR" begin
            NCNBD.piecewise_linear_refinement(model, appliedSolvers)
       end

    end
end

"""
    NCNBD.master_loop_ncbd(parallel_scheme::SDDP.Serial, model::SDDP.PolicyGraph{T},
        options::SDDP.Options, algoParams::NCNBD.AlgoParams, appliedSolvers::NCNBD.AppliedSolvers)

Inner loop function of NCNBD, if only inner loop is used.
"""

function master_loop_ncnbd_inner(parallel_scheme::SDDP.Serial, model::SDDP.PolicyGraph{T},
    options::NCNBD.Options, algoParams::NCNBD.AlgoParams, appliedSolvers::NCNBD.AppliedSolvers) where {T}

    previousSolution = nothing

    #TODO: Store best solution so far

    while true
        # start an inner loop
        result_inner = inner_loop_iteration(model, options, algoParams, appliedSolvers, previousSolution)
        # logging
        log_iteration(options, options.log_inner)
        @infiltrate algoParams.infiltrate_state in [:all, :sigma]
        if result_inner.has_converged
            sigma_test_results = inner_loop_forward_sigma_test(model, options, algoParams, appliedSolvers, result_inner.scenario_path, options.forward_pass)

            upper_bound_non_reg = sigma_test_results.cumulative_value
            upper_bound_reg = result_inner.upper_bound

            @infiltrate algoParams.infiltrate_state in [:all, :sigma]

            if isapprox(upper_bound_non_reg, upper_bound_reg)
                # by solving the regularized problem, approximately the real MILP has been solved

                # update information for MINLP
                result_inner.upper_bound = upper_bound_non_reg
                result_inner.current_sol = sigma_test_results.sampled_states

                # return all results here to keep them accessible in outer pass
                return result_inner

            else
                # increase sigma
                algoParams.sigma = algoParams.sigma * algoParams.sigma_factor

            end
            # return all results here to keep them accessible in outer pass
            # return result_inner
        else
            if result_inner.upper_bound < result_inner.lower_bound
                # increase sigma
                algoParams.sigma = algoParams.sigma * algoParams.sigma_factor
            end

        end

        previousSolution = result_inner.current_sol
    end
end

"""
    NCNBD.master_loop_ncbd(parallel_scheme::SDDP.Serial, model::SDDP.PolicyGraph{T},
        options::SDDP.Options, algoParams::NCNBD.AlgoParams, appliedSolvers::NCNBD.AppliedSolvers)

Inner loop function of NCNBD, if also outer loop is used.
"""

function inner_loop(parallel_scheme::SDDP.Serial, model::SDDP.PolicyGraph{T},
    options::NCNBD.Options, algoParams::NCNBD.AlgoParams, appliedSolvers::NCNBD.AppliedSolvers) where {T}

    previousSolution = nothing
    sigma_increased = false

    # INITIALIZE BEST KNOWN POINT AND OBJECTIVE VALUE FOR INNER LOOP
    ############################################################################
    model.ext[:best_inner_loop_objective] = model.objective_sense == JuMP.MOI.MIN_SENSE ? Inf : -Inf
    model.ext[:best_inner_loop_point] = Vector{Dict{Symbol,Float64}}()

    # ACTUAL LOOP
    ############################################################################
    while true
        # start an inner loop
        result_inner = inner_loop_iteration(model, options, algoParams, appliedSolvers, previousSolution, sigma_increased)
        # logging
        log_iteration(options, options.log_inner)

        @infiltrate algoParams.infiltrate_state in [:all, :sigma]

        if result_inner.has_converged

            TimerOutputs.@timeit NCNBD_TIMER "sigma_test" begin
                sigma_test_results = inner_loop_forward_sigma_test(model, options, algoParams, appliedSolvers, result_inner.scenario_path, options.forward_pass)
            end

            upper_bound_non_reg = sigma_test_results.cumulative_value
            upper_bound_reg = result_inner.upper_bound

            @infiltrate algoParams.infiltrate_state in [:all, :sigma]

            if isapprox(upper_bound_non_reg, upper_bound_reg)
                # by solving the regularized problem, approximately the real MILP has been solved
                # we do not need an epsilon tolerance here, because the values should be exactly equal for a sufficiently high sigma
                # WARNING: isapprox should not be used with 0 as comparison

                # update information for MINLP
                result_inner.upper_bound = upper_bound_non_reg
                result_inner.current_sol = sigma_test_results.sampled_states
                sigma_increased = false

                # return all results here to keep them accessible in outer pass
                return result_inner

            else
                # increase sigma
                algoParams.sigma = algoParams.sigma * algoParams.sigma_factor
                sigma_increased = true

            end
            # return all results here to keep them accessible in outer pass
            # return result_inner
        else
            if result_inner.upper_bound - result_inner.lower_bound < - algoParams.epsilon_innerLoop * 0.1
                # increase sigma
                algoParams.sigma = algoParams.sigma * algoParams.sigma_factor
                sigma_increased = true
            else
                sigma_increased = false
            end

        end

        previousSolution = result_inner.current_sol
    end
end
