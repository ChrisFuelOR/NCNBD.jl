struct Log
    outer_iteration::Int
    iteration::Union{Int,Nothing}
    lower_bound::Float64
    upper_bound::Float64
    current_state::Vector{Dict{Symbol,Float64}}
    #simulation_value::Float64
    time::Float64
    #pid::Int
    #total_solves::Int
    sigma::Vector{Float64}
    binaryPrecision::Dict{Symbol,Float64}
    opt_tolerance::Float64
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
end

function print_iteration_header(io)
    println(
        io,
        " Outer_Iteration   Inner_Iteration    Upper Bound     Lower Bound     Time (s)   ",
    )
end

print_value(x::Real) = lpad(Printf.@sprintf("%1.6e", x), 13)
print_value(x::Int) = Printf.@sprintf("%9d", x)
print_value(x::Nothing) = Print("")

function print_iteration(io, log::Log)
    print(io, print_value(log.outer_iteration))
    print(io, "                  ", print_value(log.iteration))
    print(io, "                        ", print_value(log.upper_bound))
    print(io, "   ", print_value(log.lower_bound))
    #print(io, "  ", print_value(log.current_state[1][:x]))
    print(io, "   ", print_value(log.time))
    # print(io, "  ", print_value(log.pid))
    # print(io, "  ", print_value(log.total_solves))
    #print(io, "  ", print_value(log.binaryPrecision))
    println(io)
end

function print_footer(io, training_results)
    println(io, "\nTerminating NCNBD with status: $(training_results.status)")
    println(
        io,
        "------------------------------------------------------------------------------",
    )
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
                    println(io, "binary precision: " )
                    # should be the same for all nodes
                    for (name, state) in model.nodes[1].ext[:lin_states]
                        println(io, "state: ", string(name), " ", algoParams.binaryPrecision[name])
                    end
                    println(io)
                    println(io, "sigma: " )
                    for i in 1:size(log.sigma, 1)
                        println(io, "stage $i :", " ", log.sigma[i])
                    end
                    println(io)

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
