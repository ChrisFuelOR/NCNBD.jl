struct State{T}
    # The incoming state variable.
    in::T
    # The outgoing state variable.
    out::T
    # The lower bound
    lb::Float64
    # The upper bound
    ub::Float64

    # function State(in::T, out::T) where {T}
    #     return new{T}(
    #         in,
    #         out,
    #         -Inf,
    #         Inf
    #     )
    # end
end

struct StateInfo
    in::JuMP.VariableInfo
    out::JuMP.VariableInfo
    initial_value::Float64
    kwargs
end

function setup_state(
    subproblem::JuMP.Model,
    state::State,
    state_info::StateInfo,
    name::String,
    ::SDDP.ContinuousRelaxation
)
    node = SDDP.get_node(subproblem)
    sym_name = Symbol(name)
    @assert !haskey(node.ext[:lin_states], sym_name)  # JuMP prevents duplicate names.
    node.ext[:lin_states][sym_name] = state
    graph = SDDP.get_policy_graph(subproblem)
    graph.ext[:lin_initial_root_state][sym_name] = state_info.initial_value
    return
end

# Internal function: set the incoming state variables of node to the values
# contained in state.
function set_incoming_state(node::SDDP.Node, state::Dict{Symbol,Float64})
    for (state_name, value) in state
        JuMP.fix(node.ext[:lin_states][state_name].in, value, force=true)
    end
    return
end

# Internal function: get the values of the outgoing state variables in node.
# Requires node.subproblem to have been solved with PrimalStatus ==
# FeasiblePoint.
function get_outgoing_state(node::SDDP.Node)
    values = Dict{Symbol,Float64}()
    for (name, state) in node.ext[:lin_states]
        # To fix some cases of numerical infeasiblities, if the outgoing value
        # is outside its bounds, project the value back onto the bounds. There
        # is a pretty large (×5) penalty associated with this check because it
        # typically requires a call to the solver. It is worth reducing
        # infeasibilities though.
        outgoing_value = JuMP.value(state.out)
        if JuMP.has_upper_bound(state.out)
            current_bound = JuMP.upper_bound(state.out)
            if current_bound < outgoing_value
                outgoing_value = current_bound
            end
        end
        if JuMP.has_lower_bound(state.out)
            current_bound = JuMP.lower_bound(state.out)
            if current_bound > outgoing_value
                outgoing_value = current_bound
            end
        end
        values[name] = outgoing_value
    end
    return values
end
