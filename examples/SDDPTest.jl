using SDDP, GLPK

model = SDDP.LinearPolicyGraph(
    stages = 1,
    sense = :Max,
    upper_bound = 500.0,
    optimizer = GLPK.Optimizer
) do subproblem, t
    # Define the state variable.
    @variable(subproblem, 0 <= volume <= 200, SDDP.State, initial_value = 200)
    # Define the control variables.
    @variables(subproblem, begin
        thermal_generation >= 0
        hydro_generation   >= 0
        hydro_spill        >= 0
    end)
    # Define the constraints
    @constraints(subproblem, begin
        volume.out == volume.in - hydro_generation - hydro_spill
        thermal_generation + hydro_generation == 150.0
    end)
    # Define the objective for each stage `t`. Note that we can use `t` as an
    # index for t = 1, 2, 3.
    fuel_cost = [10.0]
    @stageobjective(subproblem, fuel_cost[t] * thermal_generation)
end

println(model.nodes[1].subproblem)

println(model.nodes[1].stage_objective)

SDDP.train(model; iteration_limit = 10)
