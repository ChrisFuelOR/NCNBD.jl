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
    node = get_node(subproblem)
    node.ext[:stage_objective] = stage_objective
    node.ext[:stage_objective_set] = false
    return
end

# Internal function: set the objective of node to the stage objective, plus the
# cost/value-to-go term.
function set_lin_objective(subproblem::JuMP.Model)
    node = get_node(subproblem)
    objective_state_component = get_objective_state_component(node)
    belief_state_component = get_belief_state_component(node)
    if objective_state_component != JuMP.AffExpr(0.0) ||
       belief_state_component != JuMP.AffExpr(0.0)
        node.ext[:stage_objective_set] = false
    end
    if !node.ext[:stage_objective_set]
        JuMP.set_objective(
            subproblem,
            JuMP.objective_sense(subproblem),
            @expression(
                subproblem
                node.stage_objective +
                objective_state_component +
                belief_state_component +
                bellman_term(node.bellman_function)
            )
        )
    end
    node.ext[:stage_objective_set] = true
    return
end
