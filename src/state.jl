# must be mutable in my case, as in_part has to be set later
mutable struct StateInfo
    in::JuMP.VariableInfo
    out::JuMP.VariableInfo
    initial_value::Float64
    kwargs
end

struct State{T}
    # The incoming state variable.
    in::T
    # The outgoing state variable.
    out::T
    # StateInfo
    info::StateInfo
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
function set_incoming_lin_state(node::SDDP.Node, state::Dict{Symbol,Float64})
    for (state_name, value) in state

        # TODO: Check if required
        prepare_state_fixing!(node, state_name)

        # Fix value (bounds are automatically deleted by force argument)
        JuMP.fix(node.ext[:lin_states][state_name].in, value, force=true)
    end
    return
end


# Internal function: set the incoming state variables of node to the values
# contained in state.
function set_incoming_state(node::SDDP.Node, state::Dict{Symbol,Float64})
    for (state_name, value) in state

        # TODO: Check if required
        prepare_state_fixing!(node, state_name)

        # Fix value (bounds are automatically deleted by force argument)
        JuMP.fix(node.states[state_name].in, value, force=true)
    end
    return
end

# Internal function: get the values of the outgoing state variables in node.
# Requires node.subproblem to have been solved with PrimalStatus ==
# FeasiblePoint.
function get_outgoing_state(node::SDDP.Node)
    values = Dict{Symbol,Float64}()
    for (name, state_comp) in node.ext[:lin_states]
        # To fix some cases of numerical infeasiblities, if the outgoing value
        # is outside its bounds, project the value back onto the bounds. There
        # is a pretty large (×5) penalty associated with this check because it
        # typically requires a call to the solver. It is worth reducing
        # infeasibilities though.
        outgoing_value = JuMP.value(state_comp.out)
        if JuMP.has_upper_bound(state_comp.out)
            current_bound = JuMP.upper_bound(state_comp.out)
            if current_bound < outgoing_value
                outgoing_value = current_bound
            end
        end
        if JuMP.has_lower_bound(state_comp.out)
            current_bound = JuMP.lower_bound(state_comp.out)
            if current_bound > outgoing_value
                outgoing_value = current_bound
            end
        end
        values[name] = outgoing_value
    end
    return values
end

# Delete binary and integer type of state variables, since I once had some
# problems with fixing working properly then.
# May not be required, though.
# Bounds are not reset, since this can be done automatically using
# force=true when fixing.
function prepare_state_fixing!(node::SDDP.Node, state_name::Symbol)

    if JuMP.is_binary(node.ext[:lin_states][state_name].in)
        JuMP.unset_binary(node.ext[:lin_states][state_name].in)
    elseif JuMP.is_integer(node.ext[:lin_states][state_name].in)
        JuMP.unset_integer(node.ext[:lin_states][state_name].in)
    end
end

function prepare_state_fixing!(node::SDDP.Node, state::State)

    if JuMP.is_binary(state.in)
        JuMP.unset_binary(state.in)
    elseif JuMP.is_integer(state.in)
        JuMP.unset_integer(state.in)
    end
end

function prepare_state_fixing_binary!(node::SDDP.Node, state::JuMP.VariableRef)

    if JuMP.is_binary(state)
        JuMP.unset_binary(state)
    elseif JuMP.is_integer(state)
        JuMP.unset_integer(state)
    end
end

# Reset binary and integer type of state variables.
# Reset bounds.
# May not be required, though.
function follow_state_unfixing!(state::State)

    if state.info.in.has_lb
        JuMP.set_lower_bound(state.in, state.info.in.lower_bound)
    end
    if state.info.in.has_ub
        JuMP.set_upper_bound(state.in,  state.info.in.upper_bound)
    end
    if state.info.in.binary
        JuMP.set_binary(state.in)
    elseif state.info.in.integer
        JuMP.set_integer(state.in)
    end

end

function follow_state_unfixing_binary!(state::JuMP.VariableRef)

    JuMP.set_lower_bound(state, 0)
    JuMP.set_upper_bound(state, 1)

end

struct BinaryState
    value::Float64
    x_name::Symbol # name of original state it is related to
    k::Int64 # index and exponent
end
