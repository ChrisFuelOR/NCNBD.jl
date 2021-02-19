mutable struct LevelOneOracle
    cuts::Vector{NCNBD.NonlinearCut}
    states::Vector{SDDP.SampledState}
    cuts_to_be_deleted::Vector{NCNBD.NonlinearCut}
    deletion_minimum::Int
    function LevelOneOracle(deletion_minimum)
        return new(NCNBD.NonlinearCut[], SDDP.SampledState[], NCNBD.NonlinearCut[], deletion_minimum)
    end
end

mutable struct NonConvexApproximation
    theta::JuMP.VariableRef
    states::Dict{Symbol,JuMP.VariableRef}
    objective_states::Union{Nothing,NTuple{N,JuMP.VariableRef} where {N}}
    belief_states::Union{Nothing,Dict{T,JuMP.VariableRef} where {T}}
    cut_oracle::LevelOneOracle
    function NonConvexApproximation(
        theta::JuMP.VariableRef,
        states::Dict{Symbol,JuMP.VariableRef},
        objective_states,
        belief_states,
        deletion_minimum::Int,
    )
        return new(
            theta,
            states,
            objective_states,
            belief_states,
            LevelOneOracle(deletion_minimum),
        )
    end
end

mutable struct BellmanFunction <: SDDP.AbstractBellmanFunction
    global_theta::NonConvexApproximation
    local_thetas::Vector{NonConvexApproximation}
    cut_type::SDDP.CutType
    # Cuts defining the dual representation of the risk measure.
    risk_set_cuts::Set{Vector{Float64}}
end

function BellmanFunction(;
    lower_bound = -Inf,
    upper_bound = Inf,
    deletion_minimum::Int = 1,
    cut_type::SDDP.CutType = SDDP.MULTI_CUT,
)
    return SDDP.InstanceFactory{BellmanFunction}(
        lower_bound = lower_bound,
        upper_bound = upper_bound,
        deletion_minimum = deletion_minimum,
        cut_type = cut_type,
    )
end

function bellman_term(bellman_function::NCNBD.BellmanFunction)
    return bellman_function.global_theta.theta
end

function initialize_bellman_function_MILP(
    factory::SDDP.InstanceFactory{BellmanFunction},
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
    Θᴳ = JuMP.@variable(node.ext[:linSubproblem], Θ, base_name="Θᴳ")
    lower_bound > -Inf && JuMP.set_lower_bound(Θᴳ, lower_bound)
    upper_bound < Inf && JuMP.set_upper_bound(Θᴳ, upper_bound)
    # Initialize bounds for the objective states. If objective_state==nothing,
    # this check will be skipped by dispatch.
    SDDP._add_initial_bounds(node.objective_state, Θᴳ)
    x′ = Dict(key => var.out for (key, var) in node.ext[:lin_states])
    obj_μ = node.objective_state !== nothing ? node.objective_state.μ : nothing
    belief_μ = node.belief_state !== nothing ? node.belief_state.μ : nothing
    return BellmanFunction(
        NonConvexApproximation(Θᴳ, x′, obj_μ, belief_μ, deletion_minimum),
        NonConvexApproximation[],
        cut_type,
        Set{Vector{Float64}}(),
    )
end


function initialize_bellman_function_MINLP(
    factory::SDDP.InstanceFactory{BellmanFunction},
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
    Θᴳ = JuMP.@variable(node.subproblem, base_name="Θᴳ")
    lower_bound > -Inf && JuMP.set_lower_bound(Θᴳ, lower_bound)
    upper_bound < Inf && JuMP.set_upper_bound(Θᴳ, upper_bound)
    # Initialize bounds for the objective states. If objective_state==nothing,
    # this check will be skipped by dispatch.
    SDDP._add_initial_bounds(node.objective_state, Θᴳ)
    x′ = Dict(key => var.out for (key, var) in node.states)
    obj_μ = node.objective_state !== nothing ? node.objective_state.μ : nothing
    belief_μ = node.belief_state !== nothing ? node.belief_state.μ : nothing
    return BellmanFunction(
        NonConvexApproximation(Θᴳ, x′, obj_μ, belief_μ, deletion_minimum),
        NonConvexApproximation[],
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
    bellman_function::BellmanFunction,
    risk_measure::SDDP.AbstractRiskMeasure,
    trial_points::Dict{Symbol,Float64},
    used_trial_points::Dict{Symbol,Float64},
    bin_states::Vector{Dict{Symbol,BinaryState}},
    dual_variables::Vector{Dict{Symbol,Float64}},
    noise_supports::Vector,
    nominal_probability::Vector{Float64},
    objective_realizations::Vector{Float64},
    algoParams::NCNBD.AlgoParams,
    appliedSolvers::NCNBD.AppliedSolvers,
) where {T}
    # Sanity checks.
    @assert length(dual_variables) ==
    length(noise_supports) ==
    length(nominal_probability) ==
    length(objective_realizations) ==
    length(bin_states)
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
            trial_points,
            used_trial_points,
            bin_states,
            risk_adjusted_probability,
            objective_realizations,
            dual_variables,
            offset,
            algoParams,
            model.ext[:iteration],
            appliedSolvers
        )
    else  # Add a multi-cut
        @assert bellman_function.cut_type == SDDP.MULTI_CUT
        #TODO: Not implemented so far
        # SDDP._add_locals_if_necessary(node, bellman_function, length(dual_variables))
        # return _add_multi_cut(
        #     node,
        #     used_trial_points,
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
    trial_points::Dict{Symbol,Float64},
    used_trial_points::Dict{Symbol,Float64},
    bin_states::Vector{Dict{Symbol,BinaryState}},
    risk_adjusted_probability::Vector{Float64},
    objective_realizations::Vector{Float64},
    dual_variables::Vector{Dict{Symbol,Float64}},
    offset::Float64,
    algoParams::NCNBD.AlgoParams,
    iteration::Int64,
    appliedSolvers::NCNBD.AppliedSolvers,
)

    N = length(risk_adjusted_probability)
    @assert N == length(objective_realizations) == length(dual_variables) == length(bin_states)
    bin_states = bin_states[1]

    # Calculate the expected intercept and dual variables with respect to the
    # risk-adjusted probability distribution.
    πᵏ = Dict(key => 0.0 for key in keys(bin_states))
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

    # As cuts are created for the value function of the following state,
    # we need the parameters for this stage
    sigma = algoParams.sigma[node_index+1]

    _add_cut(
        node,
        node.bellman_function.global_theta,
        node.ext[:lin_bellman_function].global_theta,
        θᵏ,
        πᵏ,
        bin_states,
        trial_points,
        used_trial_points,
        obj_y,
        belief_y,
        algoParams.binaryPrecision,
        sigma,
        iteration,
        algoParams.infiltrate_state,
        appliedSolvers,
        cut_selection = algoParams.cut_selection
    )

    return (theta = θᵏ, pi = πᵏ, λ = bin_states, obj_y = obj_y, belief_y = belief_y)
end


# Add the cut to the model and the convex approximation.
function _add_cut(
    node::SDDP.Node,
    V::NonConvexApproximation,
    V_lin::NonConvexApproximation,
    θᵏ::Float64,
    πᵏ::Dict{Symbol,Float64},
    λᵏ::Dict{Symbol,BinaryState},
    trial_points::Dict{Symbol,Float64},
    xᵏ::Dict{Symbol,Float64},
    obj_y::Union{Nothing,NTuple{N,Float64}},
    belief_y::Union{Nothing,Dict{T,Float64}},
    binaryPrecision::Dict{Symbol,Float64},
    sigma::Float64,
    iteration::Int64,
    infiltrate_state::Symbol,
    appliedSolvers::NCNBD.AppliedSolvers;
    cut_selection::Bool = false
) where {N,T}

    # CORRECT INTERCEPT
    ############################################################################
    for (key, λ) in λᵏ
        θᵏ -= πᵏ[key] * λᵏ[key].value
    end
    @infiltrate infiltrate_state in [:bellman]

    # CONSTRUCT NONLINEAR CUT STRUCT
    ############################################################################
    #TODO: Should we add λᵏ? Actually, this information is not required.
    cut = NonlinearCut(θᵏ, πᵏ, xᵏ, λᵏ, binaryPrecision, sigma, JuMP.VariableRef[], JuMP.ConstraintRef[],
                       JuMP.VariableRef[], JuMP.ConstraintRef[], obj_y, belief_y, 1, iteration)

    # ADD CUT PROJECTION TO BOTH MODELS (MINLP AND MILP)
    ############################################################################
    add_cut_constraints_to_models(node, V, V_lin, cut, infiltrate_state)

    if cut_selection
        NCNBD._cut_selection_update(node, V, V_lin, cut, xᵏ, trial_points, appliedSolvers, infiltrate_state)
    else
        push!(V.cut_oracle.cuts, cut)
        push!(V_lin.cut_oracle.cuts, cut)
    end

    return
end


function add_cut_constraints_to_models(
    node::SDDP.Node,
    V::NonConvexApproximation,
    V_lin::NonConvexApproximation,
    cut::NonlinearCut,
    #λᵏ::Dict{Symbol,Float64},
    infiltrate_state::Symbol,
    )

    model = JuMP.owner_model(V.theta)
    @assert model == node.subproblem

    model_lin = JuMP.owner_model(V_lin.theta)
    @assert model_lin == node.ext[:linSubproblem]

    # In gamma, all lambda (or here gamma) variables are stored,
    # such that they can be multiplied with the cut_coefficients (the coefficients
    # relate to the lambdas, but cannot be directly matched with the original states anymore
    # as they are written into one vector)
    # All other constraints and variables are introduced per state component
    gamma = JuMP.VariableRef[]
    gamma_lin = JuMP.VariableRef[]
    allCoefficients = Float64[]
    allCoefficients_lin = Float64[]

    duals_so_far = 0
    duals_lin_so_far = 0

    # NOTE: Next steps are only done based on lin_states,
    # since normal SDDP states do not have an info argument.

    # Determine maximum U for Big M constant
    Umax = 0
    for (i, (name, state_comp)) in enumerate(node.ext[:lin_states])
        if state_comp.info.out.upper_bound > Umax
            Umax = state_comp.info.out.upper_bound
        end
    end

    # ADD NEW VARIABLES AND CONSTRAINTS FOR CUT PROJECTION
    ############################################################################
    for (i, (name, state_comp)) in enumerate(node.ext[:lin_states])
        if state_comp.info.out.binary

            # No cut projection required in binary state
            # Store states in gamma though for general cut expression
            push!(gamma, V.states[name])
            duals_so_far += 1

            push!(gamma_lin, V_lin.states[name])
            duals_lin_so_far += 1

            # Determine correct coefficient and add it to allCoefficients
            # Maybe possible with where-statement
            relatedCoefficient = 0.0
            for (i, (bin_name, value)) in enumerate(cut.coefficients)
                if cut.binary_state[bin_name].x_name == name
                    relatedCoefficient = cut.coefficients[bin_name]
                end
            end
            push!(allCoefficients, relatedCoefficient)
            push!(allCoefficients_lin, relatedCoefficient)

        else
            if !isfinite(state_comp.info.out.upper_bound) || !state_comp.info.out.has_ub
            error("When using SDDiP, state variables require an upper bound.")
            end

            if state_comp.info.out.integer
                K = SDDP._bitsrequired(state_comp.info.out.upper_bound)
                epsilon = 1
            else
                epsilon = cut.binary_precision[name]
                K = SDDP._bitsrequired(round(Int, state_comp.info.out.upper_bound / epsilon))
            end

            # Call function to add projection constraints and variables to
            # model and model_Lin
            duals_so_far = add_cut_projection_to_model!(
                                model,
                                V.states[name],
                                name,
                                cut.coefficients,
                                cut.binary_state,
                                gamma,
                                allCoefficients,
                                cut.iteration,
                                K,
                                epsilon,
                                cut.sigma,
                                cut.cutVariables,
                                cut.cutConstraints,
                                i,
                                duals_so_far,
                                Umax,
                                infiltrate_state)
            duals_lin_so_far = add_cut_projection_to_model!(
                                model_lin,
                                V_lin.states[name],
                                name,
                                cut.coefficients,
                                cut.binary_state,
                                gamma_lin,
                                allCoefficients_lin,
                                cut.iteration,
                                K,
                                epsilon,
                                cut.sigma,
                                cut.cutVariables_lin,
                                cut.cutConstraints_lin,
                                i,
                                duals_lin_so_far,
                                Umax,
                                infiltrate_state)

        end
    end

    # ADD ORIGINAL CUT DESCRIPTION
    ############################################################################
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

    @infiltrate infiltrate_state in [:bellman]

    @assert (size(gamma, 1) == size(collect(values(cut.coefficients)), 1)
                           == duals_so_far
                           == duals_lin_so_far
                           == size(gamma_lin, 1)
                           == size(allCoefficients, 1)
                           == size(allCoefficients_lin, 1)
                           )
                           # == size(collect(values(λᵏ)), 1)
    number_of_duals = size(gamma, 1)

    # TO ORIGINAL MODEL
    ############################################################################
    expr = @expression(
        model,
        V.theta + yᵀμ - sum(allCoefficients[j] * gamma[j]  for j=1:number_of_duals)
    )

    constraint_ref = if JuMP.objective_sense(model) == MOI.MIN_SENSE
        @constraint(model, expr >= cut.intercept)
    else
        @constraint(model, expr <= cut.intercept)
    end
    push!(cut.cutConstraints, constraint_ref)

    # TO LINEAR MODEL
    ############################################################################
    expr_lin = @expression(
        model_lin,
        V_lin.theta + yᵀμ - sum(allCoefficients[j] * gamma_lin[j]  for j=1:number_of_duals)
    )

    constraint_ref_lin = if JuMP.objective_sense(model_lin) == MOI.MIN_SENSE
        @constraint(model_lin, expr_lin >= cut.intercept)
    else
        @constraint(model_lin, expr_lin <= cut.intercept)
    end
    push!(cut.cutConstraints_lin, constraint_ref_lin)

    return

end


function add_cut_projection_to_model!(
    model::JuMP.Model,
    state_comp::JuMP.VariableRef,
    state_name::Symbol,
    coefficients::Dict{Symbol,Float64},
    binary_state::Dict{Symbol,BinaryState},
    gamma::Vector{JuMP.VariableRef},
    allCoefficients::Vector{Float64},
    iteration::Int64,
    K::Int64,
    epsilon::Float64,
    sigma::Float64,
    cutVariables::Vector{JuMP.VariableRef},
    cutConstraints::Vector{JuMP.ConstraintRef},
    i::Int64,
    duals_so_far::Int64,
    Umax::Float64,
    infiltrate_state::Symbol,
    )

    # ADD VARIABLES FOR CUT PROJECTION
    ####################################################################
    ν = JuMP.@variable(model, [k in 1:K], lower_bound=0, base_name = "ν_" * string(i) * "_it" * string(iteration))
    μ = JuMP.@variable(model, [k in 1:K], lower_bound=0, base_name = "μ_" * string(i) * "_it" * string(iteration))
    η = JuMP.@variable(model, base_name = "η_" * string(i) * "_it" * string(iteration))
    γ = JuMP.@variable(model, [k in 1:K], lower_bound=0, upper_bound=1, base_name = "γ_" * string(i) * "_it" * string(iteration))
    w = JuMP.@variable(model, [k in 1:K], binary=true, base_name = "w_" * string(i) * "_it" * string(iteration))
    u = JuMP.@variable(model, [k in 1:K], binary=true, base_name = "u_" * string(i) * "_it" * string(iteration))
    append!(gamma, γ)

    #Store those variables!
    append!(cutVariables, ν)
    append!(cutVariables, μ)
    push!(cutVariables, η)
    #append!(cutVariables, γ)
    append!(cutVariables, w)
    append!(cutVariables, u)

    # ADD BINARY EXPANSION CONSTRAINT
    ####################################################################
    binary_constraint = JuMP.@constraint(
        model,
        state_comp == SDDP.bincontract([γ[k] for k = 1:K], epsilon)
    )
    push!(cutConstraints, binary_constraint)

    # ADD KKT CONSTRAINT
    ####################################################################
    relatedCoefficients = Vector{Float64}(undef, K)

    for (i, (name, value)) in enumerate(coefficients)
        if binary_state[name].x_name == state_name
            index = binary_state[name].k
            relatedCoefficients[index] = coefficients[name]
        end
    end
    append!(allCoefficients, relatedCoefficients)

    kkt_constraints = JuMP.@constraint(
        model,
        [k=1:K],
        -relatedCoefficients[k] - ν[k] + μ[k] + 2^(k-1) * epsilon * η == 0
    )
    append!(cutConstraints, kkt_constraints)

    # ADD BIG M CONSTRAINTS
    ####################################################################
    #bigM = 2 * sigma * Umax

    # new method for Big-M, since earlier method was probably not correct
    # bigM = 0
    # for k in 1:K
    #     candidate = Umax * (sigma + abs(relatedCoefficients[k]) / (2^(k-1) * epsilon))
    #     if bigM < candidate
    #         bigM = candidate
    #     end
    # end
    bigM = sigma

    @infiltrate infiltrate_state in [:bellman]

    bigM_11_constraints = JuMP.@constraint(
        model,
        [k=1:K],
        γ[k] <= w[k]
    )
    append!(cutConstraints, bigM_11_constraints)

    bigM_12_constraints = JuMP.@constraint(
        model,
        [k=1:K],
        ν[k] <= bigM * (1-w[k])
    )
    append!(cutConstraints, bigM_12_constraints)

    bigM_21_constraints = JuMP.@constraint(
        model,
        [k=1:K],
        1 - γ[k] <= u[k]
    )
    append!(cutConstraints, bigM_21_constraints)

    bigM_22_constraints = JuMP.@constraint(
        model,
        [k=1:K],
        μ[k] <= bigM * (1-u[k])
    )
    append!(cutConstraints, bigM_22_constraints)

    duals_so_far += K

    return duals_so_far

end


# Internal function: update the Level-One datastructures inside `bellman_function`.
function _cut_selection_update(
    node::SDDP.Node,
    V::NCNBD.NonConvexApproximation,
    V_lin::NCNBD.NonConvexApproximation,
    cut::NCNBD.NonlinearCut,
    anchor_state::Dict{Symbol,Float64},
    trial_state::Dict{Symbol,Float64},
    appliedSolvers::NCNBD.AppliedSolvers,
    infiltrate_state::Symbol,
)
    # if cut.obj_y !== nothing || cut.belief_y !== nothing
    #     # Skip cut selection if belief or objective states present.
    #     push!(V.cut_oracle.cuts, cut)
    #     return
    # end

    # GET MODEL INFORMATION
    ############################################################################
    model = JuMP.owner_model(V.theta)
    model_lin = JuMP.owner_model(V_lin.theta)
    is_minimization = JuMP.objective_sense(model) == MOI.MIN_SENSE
    oracle = V.cut_oracle
    oracle_lin = V_lin.cut_oracle

    # GET TRIAL STATE AND BINARY STATE
    ############################################################################
    sampled_state_anchor = SDDP.SampledState(anchor_state, cut, _eval_height(node, cut, anchor_state, appliedSolvers))
    sampled_state_trial = SDDP.SampledState(trial_state, cut, _eval_height(node, cut, trial_state, appliedSolvers))

    # NOTE: By considering both type of states, we have way more states than
    # cuts, so we may not eliminiate that many cuts.
    # On the other hand, only considering the trial points is not sufficient,
    # because it may lead to creating the same cuts over and over again if the
    # binary approximation is not refined.

    # LOOP THROUGH PREVIOUSLY VISITED STATES (ANCHOR OR TRIAL STATES)
    ############################################################################
    # Loop through previously sampled states and compare the height of the most recent cut
    # against the current best. If this new cut is an improvement, store this one instead.
    for old_state in oracle.states
        height = _eval_height(node, cut, old_state.state, appliedSolvers)
        if SDDP._dominates(height, old_state.best_objective, is_minimization)
            old_state.dominating_cut.non_dominated_count -= 1
            cut.non_dominated_count += 1
            old_state.dominating_cut = cut
            old_state.best_objective = height
        end
    end
    push!(oracle.states, sampled_state_anchor)
    push!(oracle.states, sampled_state_trial)

    # LOOP THROUGH PREVIOUSLY VISITED STATES (ANCHOR OR TRIAL STATES)
    ############################################################################
    # Now loop through previously discovered cuts and compare their height at
    # `sampled_state`. If a cut is an improvement, add it to a queue to be added.
    for old_cut in oracle.cuts
        if old_cut.constraint_ref !== nothing
            # We only care about cuts not currently in the model.
            continue
        end

        # For anchor state (is this required? the cuts should be tight here and
        # we also do not have a stochastic program)
        height = _eval_height(node, old_cut, sampled_state_anchor, appliedSolvers)
        if SDDP._dominates(height, sampled_state_anchor.best_objective, is_minimization)
            sampled_state_anchor.dominating_cut.non_dominated_count -= 1
            old_cut.non_dominated_count += 1
            sampled_state_anchor.dominating_cut = old_cut
            sampled_state_anchor.best_objective = height
            add_cut_constraint_to_model(V, old_cut)
        end

        # For trial state
        height = _eval_height(node, old_cut, sampled_state_trial, appliedSolvers)
        if SDDP._dominates(height, sampled_state_trial.best_objective, is_minimization)
            sampled_state_trial.dominating_cut.non_dominated_count -= 1
            old_cut.non_dominated_count += 1
            sampled_state_trial.dominating_cut = old_cut
            sampled_state_trial.best_objective = height
            add_cut_constraints_to_models(node, V, V_lin, old_cut, infiltrate_state)
        end
    end
    push!(oracle.cuts, cut)

    # DETERMINE CUTS TO BE DELETED
    ############################################################################
    for cut in V.cut_oracle.cuts
        if cut.non_dominated_count < 1
            if cut.constraint_ref !== nothing
                push!(oracle.cuts_to_be_deleted, cut)
            end
        end
    end

    # DELETE CUTS FOR V AND V_LIN
    ############################################################################
    if length(oracle.cuts_to_be_deleted) >= oracle.deletion_minimum
        for cut in oracle.cuts_to_be_deleted
            # MINLP model
            for variable_ref in cut.cutVariables
                JuMP.delete(model, cut.variable_ref)
            end
            for constraint_ref in cut.cutConstraints
                JuMP.delete(model, cut.constraint_ref)
            end
            cut.cutVariables = nothing
            cut.cutConstraints = nothing

            # MILP model
            for variable_ref in cut.cutVariables_lin
                JuMP.delete(model_lin, cut.variable_ref)
            end
            for constraint_ref in cut.cutConstraints_lin
                JuMP.delete(model_lin, cut.constraint_ref)
            end
            cut.cutVariables_lin = nothing
            cut.cutConstraints_lin = nothing

            cut.non_dominated_count = 0
        end
    end
    empty!(oracle.cuts_to_be_deleted)
    return
end


# Internal function: calculate the height of `cut` evaluated at `state`.
function _eval_height(node::SDDP.Node, cut::Cut, states::Dict{Symbol,Float64}, appliedSolvers::NCNBD.AppliedSolvers)

    # Create a new JuMP model to evaluate the height of a non-convex cut
    model = JuMP.Model()
    JuMP.set_optimizer(model, optimizer_with_attributes(GAMS.Optimizer, "Solver"=>appliedSolvers.LP, "optcr"=>0.0))

    # Storages for coefficients and binary states
    binary_state_storage = JuMP.VariableRef[]
    allCoefficients = Float64[]
    binary_variables_so_far = 0

    for (i, (state_name, value)) in enumerate(states)
        # Get actual state from state_name
        state_comp = node.ext[:lin_states][state_name]

        if state_comp.info.out.binary
            # BINARY CASE
            ####################################################################
            # introduce one binary variable to the model
            binary_var = JuMP.@variable(model)
            push!(binary_state_storage, binary_var)
            binary_variables_so_far += 1

            # introduce binary expansion constraint to the model
            binary_constraint = JuMP.@constraint(model, binary_var == value)
            #TODO: Alternatively, we can just fix this variable

            # determine the correct cut coefficient
            relatedCoefficient = 0.0
            for (bin_name, value) in cut.coefficients
                if cut.binary_state[bin_name].x_name == state_name
                    relatedCoefficient = cut.coefficients[bin_name]
                end
            end
            push!(allCoefficients, relatedCoefficient)

        else
            if !isfinite(state_comp.info.out.upper_bound) || !state_comp.info.out.has_ub
            error("When using SDDiP, state variables require an upper bound.")
            end

            # INTEGER OR CONTINUOUS CASE
            ####################################################################
            # Get K and epsilon
            if state_comp.info.out.integer
                K = SDDP._bitsrequired(state_comp.info.out.upper_bound)
                epsilon = 1
            else
                epsilon = cut.binary_precision[name]
                K = SDDP._bitsrequired(round(Int, state_comp.info.out.upper_bound / epsilon))
            end

            # introduce binary variables to the model
            binary_var = JuMP.@variable(model, [k in 1:K], lower_bound=0, upper_bound=1)
            append!(binary_state_storage, binary_var)
            binary_variables_so_far += K

            # introduce binary expansion constraint to the model
            binary_constraint = JuMP.@constraint(model, SDDP.bincontract([binary_var[k] for k=1:K], epsilon) == value)

            # determine the correct cut coefficient
            relatedCoefficients = Vector{Float64}(undef, K)
            for (bin_name, value) in cut.coefficients
                if cut.binary_state[bin_name].x_name == state_name
                    index = cut.binary_state[bin_name].k
                    relatedCoefficients[index] = cut.coefficients[bin_name]
                end
            end
            append!(allCoefficients, relatedCoefficients)

        end
    end

    @assert(size(allCoefficients), 1) == size(binary_state_storage, 1)
                == binary_variables_so_far
                == size(collect(values(cut.coefficients)),1)

    # ADD OBJECTIVE TO THE MODEL
    ####################################################################
    objective_sense_stage = JuMP.objective_sense(node.ext[:lin_subproblem])
    eval_sense = (
        objective_sense_stage == JuMP.MOI.MIN_SENSE ? JuMP.MOI.MAX_SENSE : JuMP.MOI.MIN_SENSE
    )

    JuMP.@objective(
        model, eval_sense,
        cut.intercept + sum(allCoefficients[j] * binary_state_storage[j] for j=1:binary_variables_so_far)
    )

    # SOLVE MODEL AND RETURN SOLUTION
    ####################################################################
    JuMP.optimize!(model)
    height = JuMP.objective_value(model)
    return height


end
