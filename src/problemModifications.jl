function regularize_subproblem!(node::SDDP.Node, linearizedSubproblem::JuMP.Model, sigma::Float64)

    #NOTE: The copy constraint is not modeled explicitly here. Instead,
    # the state variable is unfixed and takes the role of z in our paper.
    # It is then subtracted from the fixed value to obtain the so called slack.

    reg_data = node.ext[:regularization_data]
    reg_data[:fixed_state_value] = Dict{Symbol,Float64}()
    reg_data[:slacks] = Any[]
    reg_data[:reg_variables] = JuMP.VariableRef[]
    reg_data[:reg_constraints] = JuMP.ConstraintRef[]

    number_of_states = 0

    # UNFIX THE STATE VARIABLES
    ############################################################################
    for (i, (name, state_comp)) in enumerate(node.ext[:lin_states])
        reg_data[:fixed_state_value][name] = JuMP.fix_value(state_comp.in)
        push!(reg_data[:slacks], reg_data[:fixed_state_value][name] - state_comp.in)
        JuMP.unfix(state_comp.in)

        #TODO: Check if required
        follow_state_unfixing!(state_comp)

        number_of_states = i
    end

    # STORE ORIGINAL OBJECTIVE FUNCTION
    ############################################################################
    old_obj = reg_data[:old_objective] = JuMP.objective_function(linearizedSubproblem)

    # DEFINE NEW VARIABLES, CONSTRAINTS AND OBJECTIVE
    ############################################################################
    # These variables and constraints are used to define the norm of the slack as a MILP
    # Using the lifting approach without binary requirements
    slack = reg_data[:slacks]

    # Variable for objective
    v = JuMP.@variable(linearizedSubproblem, reg_v, base_name = "reg_v")
    push!(reg_data[:reg_variables], v)

    # Get sign for regularization term
    fact = (JuMP.objective_sense(linearizedSubproblem) == JuMP.MOI.MIN_SENSE ? 1 : -1)

    # New objective
    new_obj = old_obj + fact * sigma * v
    JuMP.set_objective_function(linearizedSubproblem, new_obj)

    # Variables
    alpha = JuMP.@variable(linearizedSubproblem, alpha[i=1:number_of_states], base_name = "alpha")
    append!(reg_data[:reg_variables], alpha)

    # Constraints
    const_plus = JuMP.@constraint(linearizedSubproblem, [i=1:number_of_states], -alpha[i] <= slack[i])
    const_minus = JuMP.@constraint(linearizedSubproblem, [i=1:number_of_states], slack[i] <= alpha[i])
    append!(reg_data[:reg_constraints], const_plus)
    append!(reg_data[:reg_constraints], const_minus)

    const_norm = JuMP.@constraint(linearizedSubproblem, v >= sum(alpha[i] for i in 1:number_of_states))
    push!(reg_data[:reg_constraints], const_norm)

end

function deregularize_subproblem!(node::SDDP.Node, linearizedSubproblem::JuMP.Model, state::Dict{Symbol,Float64})

    reg_data = node.ext[:regularization_data]

    # FIX THE STATE VARIABLES
    ############################################################################
    for (i, (name, state_comp)) in enumerate(node.ext[:lin_states])
        #TODO: Check if required
        prepare_state_fixing!(node, state_comp)

        # assert that value stored in the forward pass equals state required now
        @assert reg_data[:fixed_state_value][name] == state[name]

        JuMP.fix(state_comp.in, reg_data[:fixed_state_value][name], force=true)
    end

    # REPLACE THE NEW BY THE OLD OBJECTIVE
    ############################################################################
    JuMP.set_objective_function(linearizedSubproblem, reg_data[:old_objective])

    # DELETE ALL REGULARIZATION-BASED VARIABLES AND CONSTRAINTS
    ############################################################################
    delete(linearizedSubproblem, reg_data[:reg_variables])
    @infiltrate
    for constraint in reg_data[:reg_constraints]
        delete(linearizedSubproblem, constraint)
    end

    delete!(node.ext, :regularization_data)

end


function regularize_backward!(node::SDDP.Node, linearizedSubproblem::JuMP.Model, sigma::Float64, approx_state::Dict{Symbol,Float64})

    number_of_states = size(collect(values(approx_state)), 1)
    reg_data = node.ext[:regularization_data]

    # DELETE EXISTING REGULARIZATION CONSTRAINTS AND SLACKS
    ############################################################################
    for constraint in reg_data[:reg_constraints]
        delete(linearizedSubproblem, constraint)
    end
    delete!(reg_data, :reg_constraints)
    reg_data[:reg_constraints] = JuMP.ConstraintRef[]
    reg_data[:slacks] = Any[]

    # DETERMINE NEW SLACKS
    ############################################################################
    for (i, (name, state_comp)) in enumerate(node.ext[:lin_states])
        @infiltrate
        push!(reg_data[:slacks], approx_state[name] - state_comp.in)
    end
    @infiltrate

    # DEFINE NEW CONSTRAINTS
    ############################################################################
    slack = reg_data[:slacks]

    # Get existing regularization variables
    v = linearizedSubproblem[:reg_v]
    alpha = linearizedSubproblem[:alpha]

    # Define new constraints
    const_plus = JuMP.@constraint(linearizedSubproblem, [i=1:number_of_states], -alpha[i] <= slack[i])
    const_minus = JuMP.@constraint(linearizedSubproblem, [i=1:number_of_states], slack[i] <= alpha[i])
    append!(reg_data[:reg_constraints], const_plus)
    append!(reg_data[:reg_constraints], const_minus)

    const_norm = JuMP.@constraint(linearizedSubproblem, v >= sum(alpha[i] for i in 1:number_of_states))
    push!(reg_data[:reg_constraints], const_norm)
    @infiltrate

end


function changeToBinarySpace!(
    node::SDDP.Node,
    linearizedSubproblem::JuMP.Model,
    state::Dict{Symbol,Float64},
    binaryPrecision::Dict{Symbol,Float64}
    )

    #NOTE: The copy constraint is not modeled explicitly here. Instead,
    # the state variable is unfixed and takes the role of z in our paper.
    # It is then subtracted from the fixed value to obtain the so called slack.

    bw_data = node.ext[:backward_data]
    bw_data[:fixed_state_value] = Dict{Symbol,Float64}()
    #bw_data[:bin_variables] = JuMP.VariableRef[]
    bw_data[:bin_constraints] = JuMP.ConstraintRef[]
    bw_data[:bin_states] = Dict{Symbol,JuMP.VariableRef}()

    number_of_states = 0

    # PREPARE NEW STATE VARIABLES
    ############################################################################
    for (state_name, value) in state
        # Get actual state from state_name
        state_comp = node.ext[:lin_states][state_name]

        # Save fixed value of state
        fixed_value = JuMP.fix_value(state_comp.in)
        bw_data[:fixed_state_value][state_name] = fixed_value
        epsilon = binaryPrecision[state_name]

        # Set up state for backward pass using binary approximation
        setup_state_backward(linearizedSubproblem, state_comp, state_name, epsilon, bw_data)
    end

    return
end


function setup_state_backward(
    subproblem::JuMP.Model,
    state_comp::State,
    state_name::Symbol,
    binaryPrecision::Float64,
    bw_data::Dict{Symbol,Any}
)

    # Get name of state variable in String representation
    name = JuMP.name(state_comp.in)

    if state_comp.info.in.binary
        #-----------------------------------------------------------------------
        # In this case, the variable must not be unfixed, no new binary variables
        # or constraints have to be introduced.

        # INTRODUCE ONE NEW BINARY VARIABLE TO THE PROBLEM
        ####################################################################
        # Still helpful, as later binary_vars is used in Lagrangian dual.
        # If we would not introduce a new variable, but just store the state
        # itself per reference in binary_vars, then it would be deleted later.

        binary_var = JuMP.@variable(
            subproblem,
            base_name = "bin_" * name,
        )
        subproblem[:binary_vars] = binary_var
        # store in list for later access and deletion
        sym_name = Symbol(JuMP.name(binary_var))
        bw_data[:bin_states][sym_name] = binary_var

        # INTRODUCE BINARY EXPANSION CONSTRAINT TO THE PROBLEM
        ####################################################################
        binary_constraint = JuMP.@constraint(subproblem, state_comp.in == binary_var)
        # store in list for later access and deletion
        push!(bw_data[:bin_constraints], binary_constraint)

        # FIX NEW VARIABLE
        ####################################################################
        # Get fixed values from fixed value of original state
        fixed_binary_value = JuMP.fix_value(state_comp.in)
        # Fix binary variables
        #JuMP.unset_binary(binary_var.in)
        JuMP.fix(binary_var, fixed_binary_value)

        # UNFIX ORIGINAL STATE
        ####################################################################
        # Unfix the original state
        JuMP.unfix(state_comp.in)

        #TODO: Check if required
        follow_state_unfixing!(state_comp)

        # DETERMINE BINARY APPROXIMATION STATE IN ORIGINAL COORDINATES
        ####################################################################
        #SDDP.bincontract([JuMP.fix_value(binary_vars[i]) for i = 1:num_vars])

    else
        if !isfinite(state_comp.info.in.upper_bound) || !state_comp.info.in.has_ub
            error("When using SDDiP, state variables require an upper bound.")
        end

        if state_comp.info.in.integer
            #-------------------------------------------------------------------

            # INTRODUCE BINARY VARIABLES TO THE PROBLEM
            ####################################################################
            #NOTE: We do not need to define them as binary,
            #as they are either fixed or relaxed to [0, 1] anyway
            #NOTE: We do not need to define them as state variables,
            #as only the _in-parts would be required anyway.
            #Moreover, no Binary/Integer or bound information has to be stored.

            num_vars = SDDP._bitsrequired(state_comp.info.in.upper_bound)

            binary_vars = JuMP.@variable(
                subproblem,
                [i in 1:num_vars],
                base_name = "bin_" * name,
            )
            subproblem[:binary_vars] = binary_vars
            # store in list for later access and deletion
            for i in 1:num_vars
               #push!(bw_data[:bin_variables], binary_vars[i])
               sym_name = Symbol(JuMP.name(binary_vars[i]))
               bw_data[:bin_states][sym_name] = binary_vars[i]
           end

            # INTRODUCE BINARY EXPANSION CONSTRAINT TO THE PROBLEM
            ####################################################################
            binary_constraint = JuMP.@constraint(
                subproblem,
                state_comp.in == SDDP.bincontract([binary_vars[i] for i = 1:num_vars])
            )
            # store in list for later access and deletion
            push!(bw_data[:bin_constraints], binary_constraint)

            # FIX NEW VARIABLES
            ####################################################################
            # Get fixed values from fixed value of original state
            fixed_binary_values = SDDP.binexpand(bw_data[:fixed_state_value][state_name], state_comp.info.in.upper_bound)
            # Fix binary variables
            for i = 1:num_vars
                #JuMP.unset_binary(binary_vars[i].in)
                JuMP.fix(binary_vars[i], fixed_binary_values[i])
            end

            # UNFIX ORIGINAL STATE
            ####################################################################
            # Unfix the original state
            JuMP.unfix(state_comp.in)

            #TODO: Check if required
            follow_state_unfixing!(state_comp)

            # DETERMINE BINARY APPROXIMATION STATE IN ORIGINAL COORDINATES
            ####################################################################
            #SDDP.bincontract([JuMP.fix_value(binary_vars[i]) for i = 1:num_vars])

        else
            #-------------------------------------------------------------------
            epsilon = binaryPrecision

            # INTRODUCE BINARY VARIABLES TO THE PROBLEM
            ####################################################################
            #NOTE: We do not need to define them as binary,
            #as they are either fixed or relaxed to [0, 1] anyway
            #NOTE: We do not need to define them as state variables,
            #as only the _in-parts would be required anyway.
            #Moreover, no Binary/Integer or bound information has to be stored.

            num_vars = SDDP._bitsrequired(round(Int, state_comp.info.in.upper_bound / epsilon))

            binary_vars = JuMP.@variable(
                subproblem,
                [i in 1:num_vars],
                base_name = "bin_" * name,
            )

            subproblem[:binary_vars] = binary_vars
            # store in list for later access and deletion
            for i in 1:num_vars
                #push!(bw_data[:bin_variables], binary_vars[i])
                sym_name = Symbol(JuMP.name(binary_vars[i]))
                # Store binary state reference for later
                bw_data[:bin_states][sym_name] = binary_vars[i]
            end
            subproblem[:binary_vars] = binary_vars

            # INTRODUCE BINARY EXPANSION CONSTRAINT TO THE PROBLEM
            ####################################################################
            binary_constraint = JuMP.@constraint(
                subproblem,
                state_comp.in == SDDP.bincontract([binary_vars[i] for i = 1:num_vars], epsilon)
            )

            # store in list for later access and deletion
            push!(bw_data[:bin_constraints], binary_constraint)

            # FIX NEW VARIABLES
            ####################################################################
            # Get fixed values from fixed value of original state
            fixed_binary_values = SDDP.binexpand(bw_data[:fixed_state_value][state_name], state_comp.info.in.upper_bound, epsilon)
            # Fix binary variables
            for i = 1:num_vars
                #JuMP.unset_binary(binary_vars[i].in)
                JuMP.fix(binary_vars[i], fixed_binary_values[i])
            end

            # UNFIX ORIGINAL STATE
            ####################################################################
            # Unfix the original state
            JuMP.unfix(state_comp.in)

            #TODO: Check if required
            follow_state_unfixing!(state_comp)

            # DETERMINE BINARY APPROXIMATION STATE IN ORIGINAL COORDINATES
            ####################################################################
            #SDDP.bincontract([JuMP.fix_value(binary_vars[i]) for i = 1:num_vars], epsilon)
        end
    end
    return
end


function changeToOriginalSpace!(
    node::SDDP.Node,
    linearizedSubproblem::JuMP.Model,
    state::Dict{Symbol,Float64}
    )

    bw_data = node.ext[:backward_data]

    # FIX THE STATE VARIABLES AGAIN
    ############################################################################
    for (state_name, value) in state
        state_comp = node.ext[:lin_states][state_name]
        JuMP.delete_lower_bound(state_comp.in)
        JuMP.delete_upper_bound(state_comp.in)

        # unset binary or integer type
        # TODO: probably not requried but I had problems with fixing such variables once
        if JuMP.is_binary(state_comp.in)
            JuMP.unset_binary(state_comp.in)
        elseif JuMP.is_integer(state_comp.in)
            JuMP.unset_integer(state_comp.in)
        end

        JuMP.fix(state_comp.in, bw_data[:fixed_state_value][state_name])
    end

    # REPLACE THE NEW BY THE OLD OBJECTIVE
    ############################################################################
    # Maybe already done in Kelley's method

    # DELETE ALL BINARY SPACE BASED VARIABLES AND CONSTRAINTS
    ############################################################################
    #delete(linearizedSubproblem, bw_data[:bin_variables])
    for (name, bin_state) in bw_data[:bin_states]
        JuMP.delete(linearizedSubproblem, bin_state)
    end

    for constraint in bw_data[:bin_constraints]
        JuMP.delete(linearizedSubproblem, constraint)
    end

    delete!(node.ext, :backward_data)

end


function determine_used_trial_states(
    state_comp::State,
    state_value::Float64,
    binaryPrecision::Float64,
)

    if state_comp.info.out.binary
        fixed_binary_values = state_value
        approx_state_value = state_value
    else
        if !isfinite(state_comp.info.out.upper_bound) || !state_comp.info.out.has_ub
            error("When using SDDiP, state variables require an upper bound.")
        end

        if state_comp.info.out.integer
            fixed_binary_values = SDDP.binexpand(state_value, state_comp.info.out.upper_bound)
            approx_state_value = SDDP.bincontract(fixed_binary_values)
        else
            fixed_binary_values = SDDP.binexpand(state_value, state_comp.info.out.upper_bound, binaryPrecision)
            approx_state_value = SDDP.bincontract(fixed_binary_values, binaryPrecision)
        end
    end
    return approx_state_value, fixed_binary_values
end
