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
