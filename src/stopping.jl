# ======================= Deterministic Stopping Rule ====================== #

"""
    DeterministicStopping()

Terminate the algorithm once optimality is reached.
"""
mutable struct DeterministicStopping <: SDDP.AbstractStoppingRule
end

stopping_rule_status(::DeterministicStopping) = :DeterministicStopping

function convergence_test(graph::SDDP.PolicyGraph, log::Vector{Log}, rule::DeterministicStopping, loop::Symbol)

    bool_1 = abs(log[end].upper_bound - log[end].lower_bound)/abs(max(log[end].upper_bound, log[end].lower_bound)) <= log[end].opt_tolerance
    #bool_1 = log[end].upper_bound - log[end].lower_bound <= log[end].opt_tolerance
    bool_2 = log[end].upper_bound - log[end].lower_bound >= -1e-4

    return bool_1 && bool_2
end

# ======================= Iteration Limit Stopping Rule ====================== #
stopping_rule_status(::SDDP.IterationLimit) = :iteration_limit

function convergence_test(graph::SDDP.PolicyGraph, log::Vector{Log}, rule::SDDP.IterationLimit, loop::Symbol)
    if loop == :inner
        return false
    elseif loop == :outer
        return log[end].outer_iteration >= rule.limit
    end
end

# ========================= Time Limit Stopping Rule ========================= #
stopping_rule_status(::SDDP.TimeLimit) = :time_limit

function convergence_test(graph::SDDP.PolicyGraph, log::Vector{Log}, rule::SDDP.TimeLimit, loop::Symbol)
    return log[end].time >= rule.limit
end


function convergence_test(
    graph::SDDP.PolicyGraph,
    log::Vector{Log},
    stopping_rules::Vector{SDDP.AbstractStoppingRule},
    loop::Symbol
)
    for stopping_rule in stopping_rules
        if convergence_test(graph, log, stopping_rule, loop)
            return true, stopping_rule_status(stopping_rule)
        end
    end
    return false, :not_solved
end
