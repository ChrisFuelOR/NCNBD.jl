function initialize_bellman_function(
    factory::SDDP.InstanceFactory{SDDP.BellmanFunction},
    model::SDDP.PolicyGraph{T},
    node::SDDP.Node{T},
) where {T}
    lower_bound, upper_bound, deletion_minimum, cut_type = -Inf, Inf, 0, SDDP.SINGLE_CUT
    if length(factory.args) > 0
        error("Positional arguments $(factory.args) ignored in BellmanFunction.")
    end
    for (kw, value) in factory.kwargs
        if kw == :lower_bound
            lower_bound = value
        elseif kw == :upper_bound
            upper_bound = value
        elseif kw == :deletion_minimum
            deletion_minimum = value
        elseif kw == :cut_type
            cut_type = value
        else
            error("Keyword $(kw) not recognised as argument to BellmanFunction.")
        end
    end
    if lower_bound == -Inf && upper_bound == Inf
        error("You must specify a finite bound on the cost-to-go term.")
    end
    if length(node.children) == 0
        lower_bound = upper_bound = 0.0
    end
    Θᴳ = JuMP.@variable(node.ext[:linSubproblem], base_name="Θᴳ")
    lower_bound > -Inf && JuMP.set_lower_bound(Θᴳ, lower_bound)
    upper_bound < Inf && JuMP.set_upper_bound(Θᴳ, upper_bound)
    # Initialize bounds for the objective states. If objective_state==nothing,
    # this check will be skipped by dispatch.
    SDDP._add_initial_bounds(node.objective_state, Θᴳ)
    x′ = Dict(key => var.out for (key, var) in node.ext[:lin_states])
    obj_μ = node.objective_state !== nothing ? node.objective_state.μ : nothing
    belief_μ = node.belief_state !== nothing ? node.belief_state.μ : nothing
    return SDDP.BellmanFunction(
        SDDP.ConvexApproximation(Θᴳ, x′, obj_μ, belief_μ, deletion_minimum),
        SDDP.ConvexApproximation[],
        cut_type,
        Set{Vector{Float64}}(),
    )
end
