function JuMP.build_variable(
    _error::Function,
    info::JuMP.VariableInfo,
    ::Type{NCNBD.State};
    initial_value = NaN,
    kwargs...,
)
    if isnan(initial_value)
        _error(
            "When creating a state variable, you must set the " *
            "`initial_value` keyword to the value of the state variable at" *
            " the root node.",
        )
    end
    return StateInfo(
        JuMP.VariableInfo(
            false,
            NaN,  # lower bound
            false,
            NaN,  # upper bound
            false,
            NaN,  # fixed value
            false,
            NaN,  # start value
            false,
            false, # binary and integer
        ),
        info,
        initial_value,
        kwargs,
    )
end

function JuMP.add_variable(problem::JuMP.Model, state_info::StateInfo, name::String)
    state = State(
        JuMP.add_variable(problem, JuMP.ScalarVariable(state_info.in), name * "_in"),
        JuMP.add_variable(problem, JuMP.ScalarVariable(state_info.out), name * "_out"),
    )
    integrality_handler = SDDP.get_integrality_handler(problem)
    setup_state(problem, state, state_info, name, integrality_handler)
    return state
end

JuMP.variable_type(model::JuMP.Model, ::Type{State}) = State

function JuMP.value(state::State{JuMP.VariableRef})
    return State(JuMP.value(state.in), JuMP.value(state.out))
end
