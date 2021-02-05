using GAMS
using JuMP
using BenchmarkTools

dual_vars = zeros(8)

approx_model = JuMP.Model(GAMS.Optimizer)
JuMP.set_optimizer_attribute(approx_model, "Solver", "Gurobi")

@variables approx_model begin
    θ
    x[1:length(dual_vars)]
end
JuMP.@objective(approx_model, JuMP.MOI.MAX_SENSE, θ)

JuMP.@constraint(
    approx_model,
    θ - 0.65 * x[3] + x[4] - x[5] + x[6] + x[7] <= 36
)

JuMP.optimize!(approx_model)
dual_vars .= value.(x)

println()
println(JuMP.value(x[1]))
println(JuMP.value(x[2]))
println(JuMP.value(x[3]))
println(JuMP.value(x[4]))
println(JuMP.value(x[5]))
println(JuMP.value(x[6]))
println(JuMP.value(x[7]))
println(JuMP.value(x[8]))

println(dual_vars)

#@btime dual_vars[findall(isnan.(dual_vars))] .= 0.0
@btime replace!(dual_vars, NaN => 0)

println(dual_vars)

println(@__FILE__)
