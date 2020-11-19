macro lin_stageobjective(subproblem, expr)
    code = quote
        set_lin_stage_objective(
            $(esc(subproblem)),
            $(Expr(
                :macrocall,
                Symbol("@expression"),
                :LineNumber,
                esc(subproblem),
                esc(expr),
            )),
        )
    end
    return code
end

function set_lin_stage_objective(subproblem::JuMP.Model, stage_objective)
    node = SDDP.get_node(subproblem)
    node.ext[:lin_stage_objective] = stage_objective
    node.ext[:lin_stage_objective_set] = false
    return
end

# Internal function: set the objective of node to the stage objective, plus the
# cost/value-to-go term.
function set_lin_objective(subproblem::JuMP.Model)
    node = SDDP.get_node(subproblem)
    objective_state_component = SDDP.get_objective_state_component(node)
    belief_state_component = SDDP.get_belief_state_component(node)
    if objective_state_component != JuMP.AffExpr(0.0) ||
       belief_state_component != JuMP.AffExpr(0.0)
        node.ext[:lin_stage_objective_set] = false
    end
    if !node.ext[:lin_stage_objective_set]
        JuMP.set_objective(
            subproblem,
            JuMP.objective_sense(subproblem),
            @expression(
                subproblem,
                node.ext[:lin_stage_objective] +
                SDDP.bellman_term(node.ext[:lin_bellman_function])
            )
        )
    end
    node.ext[:lin_stage_objective_set] = true
    return
end

function parameterize(node::SDDP.Node, noise)
    node.parameterize(noise)
    # set objective function and Bellman function for MILP
    set_lin_objective(node.ext[:linSubproblem])
    return
end
