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


# Could also be shifted to SDDP.jl, since it overwrites an existing function,
# but with additional arguments. Therefore, both methods can be distinguished.
function refine_bellman_function(
    model::SDDP.PolicyGraph{T},
    node::SDDP.Node{T},
    node_index::Int64,
    bellman_function::SDDP.BellmanFunction,
    risk_measure::SDDP.AbstractRiskMeasure,
    outgoing_state::Dict{Symbol,Float64},
    dual_variables::Vector{Dict{Symbol,Float64}},
    noise_supports::Vector,
    nominal_probability::Vector{Float64},
    objective_realizations::Vector{Float64},
    algoParams::NCNBD.AlgoParams
) where {T}
    # Sanity checks.
    @assert length(dual_variables) ==
    length(noise_supports) ==
    length(nominal_probability) ==
    length(objective_realizations)
    # Preliminaries that are common to all cut types.
    risk_adjusted_probability = similar(nominal_probability)
    offset = SDDP.adjust_probability(
        risk_measure,
        risk_adjusted_probability,
        nominal_probability,
        noise_supports,
        objective_realizations,
        model.objective_sense == MOI.MIN_SENSE,
    )
    # The meat of the function.
    if bellman_function.cut_type == SDDP.SINGLE_CUT
        return _add_average_cut(
            node,
            node_index,
            outgoing_state,
            risk_adjusted_probability,
            objective_realizations,
            dual_variables,
            offset,
            algoParams,
        )
    else  # Add a multi-cut
        @assert bellman_function.cut_type == SDDP.MULTI_CUT
        #TODO: Not implemented so far
        # SDDP._add_locals_if_necessary(node, bellman_function, length(dual_variables))
        # return _add_multi_cut(
        #     node,
        #     outgoing_state,
        #     risk_adjusted_probability,
        #     objective_realizations,
        #     dual_variables,
        #     offset,
        # )
    end
end


# Could also be shifted to SDDP.jl, since it overwrites an existing function,
# but with additional arguments. Therefore, both methods can be distinguished.
function _add_average_cut(
    node::SDDP.Node,
    node_index::Int64,
    outgoing_state::Dict{Symbol,Float64},
    risk_adjusted_probability::Vector{Float64},
    objective_realizations::Vector{Float64},
    dual_variables::Vector{Dict{Symbol,Float64}},
    offset::Float64,
    algoParams::NCNBD.AlgoParams
)

    @infiltrate

    N = length(risk_adjusted_probability)
    @assert N == length(objective_realizations) == length(dual_variables)
    # Calculate the expected intercept and dual variables with respect to the
    # risk-adjusted probability distribution.
    πᵏ = Dict(key => 0.0 for key in keys(outgoing_state))
    θᵏ = offset
    for i = 1:length(objective_realizations)
        p = risk_adjusted_probability[i]
        θᵏ += p * objective_realizations[i]
        for (key, dual) in dual_variables[i]
            πᵏ[key] += p * dual
        end
    end
    # Now add the average-cut to the subproblem. We include the objective-state
    # component μᵀy and the belief state (if it exists).
    obj_y = node.objective_state === nothing ? nothing : node.objective_state.state
    belief_y = node.belief_state === nothing ? nothing : node.belief_state.belief

    epsilon = algoParams.binaryPrecision[node_index]
    sigma = algoParams.sigma[node_index]
    _add_cut(node, node.bellman_function.global_theta, node.ext[:lin_bellman_function].global_theta, θᵏ, πᵏ, outgoing_state, obj_y, belief_y, epsilon, sigma)
    return (theta = θᵏ, pi = πᵏ, x = outgoing_state, obj_y = obj_y, belief_y = belief_y)
end


# Add the cut to the model and the convex approximation.
function _add_cut(
    node::SDDP.Node,
    V::SDDP.ConvexApproximation,
    V_lin::SDDP.ConvexApproximation,
    θᵏ::Float64,
    πᵏ::Dict{Symbol,Float64},
    xᵏ::Dict{Symbol,Float64},
    obj_y::Union{Nothing,NTuple{N,Float64}},
    belief_y::Union{Nothing,Dict{T,Float64}};
    cut_selection::Bool = true,
    epsilon::Float64,
    sigma::Float64
) where {N,T}

    # CORRECT INTERCEPT
    ############################################################################
    for (key, x) in xᵏ
        θᵏ -= πᵏ[key] * xᵏ[key]
    end

    # CONSTRUCT NONLINEAR CUT STRUCT
    ############################################################################
    cut = NonlinearCut(θᵏ, πᵏ, xᵏ, binaryPrecision, JuMP.VariableRef[], JuMP.ConstraintRef[], JuMP.VariableRef[], JuMP.ConstraintRef[], obj_y, belief_y, 1)

    # ADD CUT PROJECTION TO BOTH MODELS (MINLP AND MILP)
    ############################################################################
    add_cut_constraint_to_lin_model(node, V_lin, cut, sigma, epsilon)

    #add_cut_constraint_to_model(node, V, cut, sigma, epsilon)

    if cut_selection
        _cut_selection_update(V, cut, xᵏ)
        _cut_selection_update(V_lin, cut, xᵏ)
    else
        push!(V.cut_oracle.cuts, cut)
        push!(V_lin.cut_oracle.cuts, cut)
    end
    return
end


function add_cut_constraint_to_lin_model(
    node::SDDP.Node,
    V::SDDP.ConvexApproximation,
    cut::NonlinearCut,
    sigma::Float64,
    binaryPrecision::Float64
    )

    model = JuMP.owner_model(V.theta)
    @assert model == node.ext[:linSubproblem]

    # In gamma, all lambda (or here gamma) variables are stored,
    # such that they can be multiplied with the cut_coefficients which relate
    # to the lambdas, but cannot be matched with the original states anymore
    # as they are written into one vector
    # All other constraints and variables are introduced per state component
    gamma = JuMP.VariableRef[]
    duals_so_far = 0

    # determine maximum U
    Umax = 0
    for (i, (name, state_comp)) in enumerate(node.ext[:lin_states])
        if state_comp.info.out.upper_bound > Umax
            Umax = state_comp.info.out.upper_bound
        end
    end

    for (i, (name, state_comp)) in enumerate(node.ext[:lin_states])
        if state_comp.info.out.binary

            for (i, x) in V.states
                push!(gamma, x)
            end

        else
            if !isfinite(state_comp.info.out.upper_bound) || !state_comp.info.out.has_ub
            error("When using SDDiP, state variables require an upper bound.")
            end

            if state_comp.info.out.integer
                K = SDDP._bitsrequired(state_comp.info.out.upper_bound)
                epsilon = 1
            elseif state_comp.info.out.binary
                K = SDDP._bitsrequired(round(Int, state_comp.info.out.upper_bound / binaryPrecision))
                epsilon = binaryPrecision
            end

            # ADD VARIABLES FOR CUT PROJECTION
            ####################################################################
            #TODO: Can we enumerate the cuts here as well?
            ν = JuMP.@variable(model, [k in 1:K], lower_bound=0, base_name = "ν_" * string(i))
            μ = JuMP.@variable(model, [k in 1:K], lower_bound=0, base_name = "μ_" * string(i))
            η = JuMP.@variable(model, base_name = "η_" * string(i))
            γ = JuMP.@variable(model, [k in 1:K], lower_bound=0, upper_bound=1, base_name = "γ_" * string(i))
            w = JuMP.@variable(model, [k in 1:K], binary=true, base_name = "w_" * string(i))
            u = JuMP.@variable(model, [k in 1:K], binary=true, base_name = "u_" * string(i))
            append!(gamma, γ)

            #Store those variables!
            append!(cut.cutVariables_lin, ν)
            append!(cut.cutVariables_lin, μ)
            push!(cut.cutVariables_lin, η)
            append!(cut.cutVariables_lin, γ)
            append!(cut.cutVariables_lin, w)
            append!(cut.cutVariables_lin, u)

            # ADD BINARY EXPANSION CONSTRAINT
            ####################################################################
            binary_constraint = JuMP.@constraint(
                model,
                state_comp.out == SDDP.bincontract([γ[k] for k = 1:K], epsilon)
            )
            push!(cut.cutConstraints_lin, binary_constraint)

            # ADD KKT CONSTRAINT
            ####################################################################
            kkt_constraints = JuMP.@constraint(
                model,
                [k=1:K],
                cut.coefficients[duals_so_far+k] - ν[k] + μ[k] + 2^(k-1) * epsilon == η
            )
            append!(cut.cutConstraints_lin, kkt_constraints)

            # ADD BIG M CONSTRAINTS
            ####################################################################
            bigM = 2 * sigma * Umax

            bigM_11_constraints = JuMP.@constraint(
                model,
                [k=1:K],
                γ[k] <= w[k]
            )
            append!(cut.cutConstraints_lin, bigM_11_constraints)

            bigM_12_constraints = JuMP.@constraint(
                model,
                [k=1:K],
                ν[k] <= bigM * (1-w[k])
            )
            append!(cut.cutConstraints_lin, bigM_12_constraints)

            bigM_21_constraints = JuMP.@constraint(
                model,
                [k=1:K],
                1 - γ[k] <= u[k]
            )
            append!(cut.cutConstraints_lin, bigM_21_constraints)

            bigM_22_constraints = JuMP.@constraint(
                model,
                [k=1:K],
                μ[k] <= bigM * (1-u[k])
            )
            append!(cut.cutConstraints_lin, bigM_22_constraints)

            duals_so_far += K
        end
    end

    # ADD ORIGINAL CUT DESCRIPTION
    ####################################################################
    yᵀμ = JuMP.AffExpr(0.0)
    # if V.objective_states !== nothing
    #     for (y, μ) in zip(cut.obj_y, V.objective_states)
    #         JuMP.add_to_expression!(yᵀμ, y, μ)
    #     end
    # end
    # if V.belief_states !== nothing
    #     for (k, μ) in V.belief_states
    #         JuMP.add_to_expression!(yᵀμ, cut.belief_y[k], μ)
    #     end
    # end

    @assert size(gamma, 1) == collect(values(cut.coefficients)) == duals_so_far
    number_of_duals = size(gamma, 1)

    expr = @expression(
        model,
        V.theta + yᵀμ - sum(cut.coefficients[j] * gammas[j]  for j=1:number_of_duals)
    )

    cut.constraint_ref = if JuMP.objective_sense(model) == MOI.MIN_SENSE
        @constraint(model, expr >= cut.intercept)
    else
        @constraint(model, expr <= cut.intercept)
    end

    return

end


# function add_cut_constraint_to_model(
#     node::SDDP.Node,
#     V::SDDP.ConvexApproximation,
#     cut::NonlinearCut,
#     sigma::Float64,
#     binaryPrecision::Float64
#     )
#
#     model = JuMP.owner_model(V.theta)
#     @assert model == node.subproblem
#
#     gamma = JuMP.VariableRef[]
#     duals_so_far = 0
#
#     # determine maximum U
#     Umax = 0
#     for (i, (name, state_comp)) in enumerate(node.states)
#         if state_comp.info.out.upper_bound > Umax
#             Umax = state_comp.info.out.upper_bound
#         end
#     end
#
#     for (i, (name, state_comp)) in enumerate(node.states)
#         if state_comp.info.out.binary
#
#             for (i, x) in V.states
#                 push!(gamma, x)
#             end
#
#         else
#             if !isfinite(state_comp.info.out.upper_bound) || !state_comp.info.out.has_ub
#             error("When using SDDiP, state variables require an upper bound.")
#             end
#
#             if state_comp.info.out.integer
#                 K = SDDP._bitsrequired(state_comp.info.out.upper_bound)
#                 epsilon = 1
#             elseif state_comp.info.out.binary
#                 K = SDDP._bitsrequired(round(Int, state_comp.info.out.upper_bound / binaryPrecision))
#                 epsilon = binaryPrecision
#             end
#
#             # ADD VARIABLES FOR CUT PROJECTION
#             ####################################################################
#             #TODO: Can we enumerate the cuts here as well?
#             ν = JuMP.@variable(model, [k in 1:K], lower_bound=0, base_name = "ν_" * string(i))
#             μ = JuMP.@variable(model, [k in 1:K], lower_bound=0, base_name = "μ_" * string(i))
#             η = JuMP.@variable(model, base_name = "η_" * string(i))
#             γ = JuMP.@variable(model, [k in 1:K], lower_bound=0, upper_bound=1, base_name = "γ_" * string(i))
#             w = JuMP.@variable(model, [k in 1:K], binary=true, base_name = "w_" * string(i))
#             u = JuMP.@variable(model, [k in 1:K], binary=true, base_name = "u_" * string(i))
#             append!(gamma, γ)
#
#             #Store those variables!
#             append!(cut.cutVariables_lin, ν)
#             append!(cut.cutVariables_lin, μ)
#             push!(cut.cutVariables_lin, η)
#             append!(cut.cutVariables_lin, γ)
#             append!(cut.cutVariables_lin, w)
#             append!(cut.cutVariables_lin, u)
#
#             # ADD BINARY EXPANSION CONSTRAINT
#             ####################################################################
#             binary_constraint = JuMP.@constraint(
#                 model,
#                 state_comp.out == SDDP.bincontract([γ[k] for k = 1:K], epsilon)
#             )
#             push!(cut.cutConstraints_lin, binary_constraint)
#
#             # ADD KKT CONSTRAINT
#             ####################################################################
#             kkt_constraints = JuMP.@constraint(
#                 model,
#                 [k=1:K],
#                 cut.coefficients[duals_so_far+k] - ν[k] + μ[k] + 2^(k-1) * epsilon == η
#             )
#             append!(cut.cutConstraints_lin, kkt_constraints)
#
#             # ADD BIG M CONSTRAINTS
#             ####################################################################
#             bigM = 2 * sigma * Umax
#
#             bigM_11_constraints = JuMP.@constraint(
#                 model,
#                 [k=1:K],
#                 γ[k] <= w[k]
#             )
#             append!(cut.cutConstraints_lin, bigM_11_constraints)
#
#             bigM_12_constraints = JuMP.@constraint(
#                 model,
#                 [k=1:K],
#                 ν[k] <= bigM * (1-w[k])
#             )
#             append!(cut.cutConstraints_lin, bigM_12_constraints)
#
#             bigM_21_constraints = JuMP.@constraint(
#                 model,
#                 [k=1:K],
#                 1 - γ[k] <= u[k]
#             )
#             append!(cut.cutConstraints_lin, bigM_21_constraints)
#
#             bigM_22_constraints = JuMP.@constraint(
#                 model,
#                 [k=1:K],
#                 μ[k] <= bigM * (1-u[k])
#             )
#             append!(cut.cutConstraints_lin, bigM_22_constraints)
#
#             duals_so_far += K
#         end
#     end
#
#     # ADD ORIGINAL CUT DESCRIPTION
#     ####################################################################
#     yᵀμ = JuMP.AffExpr(0.0)
#     # if V.objective_states !== nothing
#     #     for (y, μ) in zip(cut.obj_y, V.objective_states)
#     #         JuMP.add_to_expression!(yᵀμ, y, μ)
#     #     end
#     # end
#     # if V.belief_states !== nothing
#     #     for (k, μ) in V.belief_states
#     #         JuMP.add_to_expression!(yᵀμ, cut.belief_y[k], μ)
#     #     end
#     # end
#
#     @assert size(gamma, 1) = collect(values(cut.coefficients)) = duals_so_far
#     number_of_duals = size(gamma, 1)
#
#     expr = @expression(
#         model,
#         V.theta + yᵀμ - sum(cut.coefficients[j] * gammas[j]  for j=1:number_of_duals)
#     )
#
#     cut.constraint_ref = if JuMP.objective_sense(model) == MOI.MIN_SENSE
#         @constraint(model, expr >= cut.intercept)
#     else
#         @constraint(model, expr <= cut.intercept)
#     end
#
#     return
#
# end
