struct Log
    outer_iteration::Int
    iteration::Union{Int,Nothing}
    lower_bound::Float64
    #best_upper_bound::Float64
    best_upper_bound::Float64
    upper_bound::Float64
    current_state::Vector{Dict{Symbol,Float64}}
    #simulation_value::Float64
    time::Float64
    #pid::Int
    #total_solves::Int
    #sigma::Vector{Float64}
    #binaryPrecision::Dict{Symbol,Float64}
    sigma_increased::Union{Bool,Nothing}
    bin_refinement::Union{Symbol,Nothing}
    subproblem_size::Union{Dict{Symbol,Int64},Nothing}
    opt_tolerance::Float64
    lag_iterations::Union{Vector{Int},Nothing}
    total_cuts::Int
    active_cuts::Int
end


# Internal struct: storage for SDDP options and cached data. Users shouldn't
# interact with this directly.
struct Options{T}
    # The initial state to start from the root node.
    initial_state::Dict{Symbol,Float64}
    # The sampling scheme to use on the forward pass.
    sampling_scheme::SDDP.AbstractSamplingScheme
    backward_sampling_scheme::SDDP.AbstractBackwardSamplingScheme
    # Storage for the set of possible sampling states at each node. We only use
    # this if there is a cycle in the policy graph.
    starting_states::Dict{T,Vector{Dict{Symbol,Float64}}}
    # Risk measure to use at each node.
    risk_measures::Dict{T,SDDP.AbstractRiskMeasure}
    # The delta by which to check if a state is close to a previously sampled
    # state.
    cycle_discretization_delta::Float64
    # Flag to add cuts to similar nodes.
    refine_at_similar_nodes::Bool
    # The node transition matrix.
    Φ::Dict{Tuple{T,T},Float64}
    # A list of nodes that contain a subset of the children of node i.
    similar_children::Dict{T,Vector{T}}
    stopping_rules::Vector{SDDP.AbstractStoppingRule}
    dashboard_callback::Function
    print_level::Int
    start_time::Float64
    log_inner::Vector{Log}
    log_outer::Vector{Log}
    log_file_handle
    log_frequency::Int
    forward_pass::SDDP.AbstractForwardPass

    # Internal function: users should never construct this themselves.
    function Options(
        model::SDDP.PolicyGraph{T},
        initial_state::Dict{Symbol,Float64},
        sampling_scheme::SDDP.AbstractSamplingScheme,
        backward_sampling_scheme::SDDP.AbstractBackwardSamplingScheme,
        risk_measures,
        cycle_discretization_delta::Float64,
        refine_at_similar_nodes::Bool,
        stopping_rules::Vector{SDDP.AbstractStoppingRule},
        dashboard_callback::Function,
        print_level::Int,
        start_time::Float64,
        log_inner::Vector{Log},
        log_outer::Vector{Log},
        log_file_handle,
        log_frequency::Int,
        forward_pass::SDDP.AbstractForwardPass,
    ) where {T}
        return new{T}(
            initial_state,
            sampling_scheme,
            backward_sampling_scheme,
            SDDP.to_nodal_form(model, x -> Dict{Symbol,Float64}[]),
            SDDP.to_nodal_form(model, risk_measures),
            cycle_discretization_delta,
            refine_at_similar_nodes,
            SDDP.build_Φ(model),
            SDDP.get_same_children(model),
            stopping_rules,
            dashboard_callback,
            print_level,
            start_time,
            log_inner,
            log_outer,
            log_file_handle,
            log_frequency,
            forward_pass,
        )
    end
end

struct Results
    status::Symbol
    log_inner::Vector{Log}
    log_outer::Vector{Log}
end

function print_helper(f, io, args...)
    f(stdout, args...)
    f(io, args...)
end

function print_banner(io)
    println(
        io,
        "--------------------------------------------------------------------------------",
    )
    println(io, "                      NCNBD.jl (c) Christian Füllner, 2020")
    println(io, "re-uses code from     SDDP.jl (c) Oscar Dowson, 2017-20")
    println(io)
    flush(io)
end

function print_parameters(io, initialAlgoParams::NCNBD.InitialAlgoParams, appliedSolvers::NCNBD.AppliedSolvers)

    # Printing the time
    println(io, Dates.now())

    # Printint the file name
    print(io, "calling ")
    print(io, @__FILE__)
    println(io)
    println(io)

    # Printing the parameters used
    println(io, Printf.@sprintf("outer loop optimality tolerance: %1.4e", initialAlgoParams.epsilon_outerLoop))
    println(io, Printf.@sprintf("inner loop optimality tolerance: %1.4e", initialAlgoParams.epsilon_innerLoop))
    println(io, "Initial binary precision:")
    println(io, initialAlgoParams.binaryPrecision)
    println(io, "Initial PLA precision:")
    println(io, initialAlgoParams.plaPrecision)
    println(io, "Initial sigma:")
    println(io, initialAlgoParams.sigma)
    println(io, "Lagrangian atol:")
    println(io, initialAlgoParams.lagrangian_atol)
    println(io, "Lagrangian rtol:")
    println(io, initialAlgoParams.lagrangian_rtol)
    println(io, "Dual initialization:")
    println(io, initialAlgoParams.dual_initialization_regime)
    println(io, "Lagrangian method:")
    println(io, initialAlgoParams.lagrangian_method)
    if initialAlgoParams.lagrangian_method == :bundle_level
        println(io, "Level parameter:")
        println(io, initialAlgoParams.level_factor)
    end

    println(io, "Used solvers:")
    println(io, "LP:", appliedSolvers.LP)
    println(io, "MILP:", appliedSolvers.MILP)
    println(io, "MINLP:", appliedSolvers.MINLP)
    println(io, "NLP:", appliedSolvers.NLP)
    println(io, "Lagrange:", appliedSolvers.Lagrange)

    # println(io, "Sigma factor:")
    # println(io, initialAlgoParams.sigma_factor)
    flush(io)
end


function print_iteration_header(io)
    println(
        io,
        " Outer_Iteration   Inner_Iteration   Upper Bound    Best Upper Bound     Lower Bound     Time (s)         sigma_ref    bin_ref     tot_var     bin_var     int_var       con       cuts   active     Lag iterations      ",
    )
    flush(io)
end

function print_iteration(io, log::Log)
    print(io, lpad(Printf.@sprintf("%5d", log.outer_iteration), 15))
    print(io, "   ")
    if !isnothing(log.iteration)
       print(io, lpad(Printf.@sprintf("%5d", log.iteration), 15))
    else
       print(io, lpad(Printf.@sprintf(""), 15))
    end
    print(io, "   ")
    print(io, lpad(Printf.@sprintf("%1.6e", log.upper_bound), 13))
    print(io, "   ")
    print(io, lpad(Printf.@sprintf("%1.6e", log.best_upper_bound), 16))
    print(io, "   ")
    print(io, lpad(Printf.@sprintf("%1.6e", log.lower_bound), 13))
    print(io, "   ")
    print(io, lpad(Printf.@sprintf("%1.6e", log.time), 13))
    print(io, "   ")
    if !isnothing(log.sigma_increased)
    	print(io, Printf.@sprintf("%9s", log.sigma_increased ? "true" : "false"))
    else
   	    print(io, lpad(Printf.@sprintf(""), 9))
    end
    print(io, "   ")
    if !isnothing(log.bin_refinement)
    	#print(io, Printf.@sprintf("%9s", log.bin_refinement ? "true" : "false"))
        print(io, Printf.@sprintf("%9s", log.bin_refinement))
    else
   	    print(io, lpad(Printf.@sprintf(""), 9))
    end
    print(io, "   ")
    if !isnothing(log.subproblem_size)
       	print(io, Printf.@sprintf("%9d", log.subproblem_size[:total_var]))
        print(io, "   ")
       	print(io, Printf.@sprintf("%9d", log.subproblem_size[:bin_var]))
        print(io, "   ")
       	print(io, Printf.@sprintf("%9d", log.subproblem_size[:int_var]))
        print(io, "   ")
       	print(io, Printf.@sprintf("%9d", log.subproblem_size[:total_con]))
    else
        print(io, lpad(Printf.@sprintf(""), 36))
    end
    print(io, "   ")
    print(io, lpad(Printf.@sprintf("%5d", log.total_cuts), 7))
    print(io, "   ")
    print(io, lpad(Printf.@sprintf("%5d", log.active_cuts), 7))
    print(io, "   ")

    if !isnothing(log.lag_iterations)
        print(io, log.lag_iterations)
    else
        print(io, lpad(Printf.@sprintf(""), 19))
    end
    print(io, "   ")

    println(io)
    flush(io)
end


function print_footer(io, training_results)
    println(io, "\nTerminating NCNBD with status: $(training_results.status)")
    println(
        io,
        "------------------------------------------------------------------------------",
    )
    flush(io)
end

function log_iteration(options, log)
    options.dashboard_callback(log[end], false)
    if options.print_level > 0 && mod(length(log), options.log_frequency) == 0
        print_helper(print_iteration, options.log_file_handle, log[end])
    end
end


"""
    write_log_to_csv(model::PolicyGraph, filename::String)

Write the log of the most recent training to a csv for post-analysis.

Assumes that the model has been trained via [`NCNBD.solve`](@ref).
"""
function write_log_to_csv(model::SDDP.PolicyGraph, filename::String, algoParams::NCNBD.AlgoParams)
    if model.ext[:results] === nothing
        error("Unable to write the log to file because the model has not been solved yet.")
    end
    open(filename, "w") do io
        for log in model.ext[:results].log_outer

            println(io, "OUTER LOOP ITERATION ", log.outer_iteration)
            println(io, "################################################################################")

            for log_inner in model.ext[:results].log_inner
                if log_inner.outer_iteration == log.outer_iteration
                    println(io, "solving inner loop problem")
                    println(io)
                    #println(io, "binary precision: " )
                    # should be the same for all nodes
                    #for (name, state) in model.nodes[1].ext[:lin_states]
                    #    println(io, "state: ", string(name), " ", algoParams.binaryPrecision[name])
                    #end
                    #println(io)
                    #println(io, "sigma: " )
                    #for i in 1:size(log.sigma, 1)
                    #    println(io, "stage $i :", " ", log.sigma[i])
                    #end
                    #println(io)

                    println(io, "inner_iteration, inner_lower_bound, inner_upper_bound, time")
                    println(
                        io,
                        @sprintf("%.6f",log_inner.iteration),
                        ", ",
                        @sprintf("%.6f",log_inner.lower_bound),
                        ", ",
                        @sprintf("%.6f",log_inner.upper_bound),
                        ", ",
                        @sprintf("%.6f",log_inner.time),
                    )

                    println(io)
                    println(io, "current solution: " )
                    # should be the same for all nodes
                    for i in 1:size(log.current_state, 1)
                        println(io, "Stage $i :")
                        for (name, state) in model.nodes[i].ext[:lin_states]
                            println(io, "state: ", string(name), " ", @sprintf("%.6f", log.current_state[i][name]))
                        end
                    end

                    println(io)

                end

                println(io, "--------------------------------------------------------------------------------")

            end
            println(io, "################################################################################")

            println(io, "solving outer loop problem")
            println(io)

            println(io, "outer_iteration, lower_bound, upper_bound, time")
            println(
                io,
                @sprintf("%.6f",log.outer_iteration),
                ", ",
                @sprintf("%.6f",log.lower_bound),
                ", ",
                @sprintf("%.6f",log.upper_bound),
                ", ",
                @sprintf("%.6f",log.time),
            )

            println(io)
            println(io, "current solution: " )
            # should be the same for all nodes
            for i in 1:size(log.current_state, 1)
                println(io, "Stage $i :")
                for (name, state) in model.nodes[i].states
                    println(io, "state: ", string(name), " ", @sprintf("%.6f", log.current_state[i][name]))
                end
            end

            println(io, "######################################################")
        end

    end
end


function print_lagrange_header(io)
    println(
        io,
        " Iteration     f_approx    best_actual     f_actual  ",
    )
    flush(io)
end

function print_lag_iteration(io, iter::Int, f_approx::Float64, best_actual::Float64, f_actual::Float64)
    print(io, lpad(Printf.@sprintf("%5d", iter), 15))
    print(io, "   ")
    print(io, lpad(Printf.@sprintf("%1.10e", f_approx), 13))
    print(io, "   ")
    print(io, lpad(Printf.@sprintf("%1.10e", best_actual), 13))
    print(io, "   ")
    print(io, lpad(Printf.@sprintf("%1.10e", f_actual), 13))

    println(io)
    flush(io)
end
