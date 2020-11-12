using JuMP
using SDDP
using NCNBD
using Revise
using Gurobi
using SCIP
using GAMS

function exampleModelTest()

    # normal nonlinear constraint
    ############################################################################
    # storing: nonlinear expression cannot be stored for later purposes
    # evaluating: nonlinear expression cannot be evaluated later
    # solvers: solvers can handle the constraint
    # optimal value: 1.0, optimal point: [1.0, 1.0]
    # => NOT SUFFICIENT FOR MY PURPOSE
    ############################################################################
    # problem = JuMP.Model(GAMS.Optimizer)
    # JuMP.set_optimizer_attribute(problem, "Solver", "Ipopt")
    # @variable(problem, 0 <= x[i=1:2] <= 1)
    # @objective(problem, Max, x[2])
    # @constraint(problem, x[2] - x[1] <= 0)
    # @NLconstraint(problem, x[1]*x[2] <= 1)
    # print(problem)
    # JuMP.optimize!(problem)
    # println()
    # println("optimal value: ", objective_value(problem))
    # println("optimal point: ", value.(x))
    # println()

    # mrapo julialang example.
    # nonlinear expression as user-defined function
    ############################################################################
    # storing: nonlinear expression can be stored for later purposes
    # evaluating: nonlinear expression can be evaluated later
    # solvers: solvers cannot handle the constraint
    # LoadError: unrecognized operation (nonlinearexp)
    # => NOT SUFFICIENT FOR MY PURPOSE
    ############################################################################
    # problem = JuMP.Model(GAMS.Optimizer)
    # JuMP.set_optimizer_attribute(problem, "Solver", "Ipopt")
    # @variable(problem, 0 <= x[i=1:2] <= 1)
    # @objective(problem, Max, x[2])
    # @constraint(problem, x[2] - x[1] <= 0)
    # nonlinearexp(x) = x[1]*x[2]
    # register(problem, :nonlinearexp, 2, nonlinearexp, autodiff=true)
    # add_NL_constraint(problem, :(nonlinearexp($(x...)) <= 1))
    # JuMP.optimize!(problem)
    # println()
    # println("optimal value: ", objective_value(problem))
    # println("optimal point: ", value.(x))
    # println()

    # odow julialang example.
    # nonlinear expression as user-defined function
    ############################################################################
    # storing: nonlinear expression can be stored for later purposes
    # evaluating: nonlinear expression can be evaluated later
    # solvers: solvers cannot handle the constraint
    # LoadError: unrecognized operation (nonlinearexp)
    # => NOT SUFFICIENT FOR MY PURPOSE
    ############################################################################
    # problem = JuMP.Model(GAMS.Optimizer)
    # JuMP.set_optimizer_attribute(problem, "Solver", "Ipopt")
    # @variable(problem, 0 <= x[i=1:2] <= 1)
    # @objective(problem, Max, x[2])
    # @constraint(problem, x[2] - x[1] <= 0)
    # nonlinearexp(x...) = x[1]*x[2]
    # register(problem, :nonlinearexp, 2, nonlinearexp, autodiff=true)
    # @NLconstraint(problem, nonlinearexp(x[1], x[2]) <= 1)
    # JuMP.optimize!(problem)
    # println()
    # println("optimal value: ", objective_value(problem))
    # println("optimal point: ", value.(x))
    # println()

    # my own example.
    # nonlinear expression as user-defined function
    ############################################################################
    # storing: nonlinear expression can be stored for later purposes
    # evaluating: nonlinear expression can be evaluated later
    # solvers: solvers cannot handle the constraint
    # LoadError: unrecognized operation (nonlinearexp)
    # => NOT SUFFICIENT FOR MY PURPOSE
    ############################################################################
    # problem = JuMP.Model(GAMS.Optimizer)
    # JuMP.set_optimizer_attribute(problem, "Solver", "Ipopt")
    # @variable(problem, 0 <= x[i=1:2] <= 1)
    # @objective(problem, Max, x[2])
    # @constraint(problem, x[2] - x[1] <= 0)
    # nonlinearexp(x,y) = x*y
    # register(problem, :nonlinearexp, 2, nonlinearexp, autodiff=true)
    # @NLconstraint(problem, nonlinearexp(x[1], x[2]) <= 1)
    # print(problem)
    # JuMP.optimize!(problem)
    # println()
    # println("optimal value: ", objective_value(problem))
    # println("optimal point: ", value.(x))
    # println()

    # nonlinear expression as Julia expression
    ############################################################################
    # storing: nonlinear expression can be stored for later purposes
    # evaluating: nonlinear expression cannot be evaluated due to inequality
    # solvers: solvers can handle the constraint
    # optimal value: 1.0, optimal point: [1.0, 1.0]
    # => NOT SUFFICIENT FOR MY PURPOSE
    ############################################################################
    # nonlinear expression can be stored and solvers can handle it
    # However, it cannot be evaluated as it is an inequality
    # problem = JuMP.Model(GAMS.Optimizer)
    # JuMP.set_optimizer_attribute(problem, "Solver", "Ipopt")
    # @variable(problem, 0 <= x[i=1:2] <= 1)
    # @objective(problem, Max, x[2])
    # @constraint(problem, x[2] - x[1] <= 0)
    # expressionStored = :($(x[1])*$(x[2]) <= 1)
    # JuMP.add_NL_constraint(problem, expressionStored)
    # print(problem)
    # JuMP.optimize!(problem)
    # println()
    # println("optimal value: ", objective_value(problem))
    # println("optimal point: ", value.(x))
    # println()

    # # nonlinear expression as combined expression
    ############################################################################
    # storing: nonlinear expression can be stored for later purposes
    # evaluating: nonlinear expression cannot be evaluated
    # (at least I don't know how to in the multidimensional case)
    # solvers: solvers can handle the constraint
    # optimal value: 1.0, optimal point: [1.0, 1.0]
    # => NOT SUFFICIENT FOR MY PURPOSE
    ############################################################################
    # By splits into several expressions, we also obtain an evaluatable one.
    # However, I do not know how to evaluate this in the multidimensional case.
    # I tried one version (do not remember which), which worked, but only
    # using x..., i.e. all components of x, and not only the ones used in the term.
    # Moreover, many of those x... approaches did not work, so I did not manage
    # to figure out the right one again.
    ############################################################################
    # problem = JuMP.Model(GAMS.Optimizer)
    # JuMP.set_optimizer_attribute(problem, "Solver", "Ipopt")
    # @variable(problem, 0 <= x[i=1:2] <= 1)
    # @objective(problem, Max, x[2])
    # @constraint(problem, x[2] - x[1] <= 0)
    # express1 = :($(x[1])*$(x[2]))
    # express2 = :($(express1) <= 1)
    # add_NL_constraint(problem, express2)
    # print(problem)
    # optimize!(problem)
    # println()
    # println("optimal value: ", objective_value(problem))
    # println("optimal point: ", value.(x))
    # println()

    # # nonlinear expression as expression and related user-defined function
    ############################################################################
    # storing: nonlinear expression can be stored for later purposes
    # evaluating: nonlinear expression cannot be evaluated
    # (at least I don't know how to in the multidimensional case)
    # solvers: solvers can handle the constraint
    # optimal value: 1.0, optimal point: [1.0, 1.0]
    # => NOT SUFFICIENT FOR MY PURPOSE
    ############################################################################
    # problem = JuMP.Model(GAMS.Optimizer)
    # JuMP.set_optimizer_attribute(problem, "Solver", "Ipopt")
    # @variable(problem, 0 <= x[i=1:2] <= 1)
    # @objective(problem, Max, x[2])
    # @constraint(problem, x[2] - x[1] <= 0)
    # express1 = :($(x[1])*$(x[2]))
    # express2 = :($(express1) <= 1)
    # @eval expressFunction(x) = $express1
    # add_NL_constraint(problem, express2)
    # print(problem)
    # optimize!(problem)
    # println()
    # println("optimal value: ", objective_value(problem))
    # println("optimal point: ", value.(x))
    # println()
    ############################################################################
    # println(expressFunction(1)) -> JUST RETURNS THE EXPRESSION
    # println(expressFunction(1,2)) -> ERROR
    # println(expressFunction([1 2])) -> JUST RETURNS THE EXPRESSION
    # The problem may be that eval always runs in the global scope.

    # # nonlinear expression as expression and independent user-defined function
    ############################################################################
    # storing: nonlinear expression can be stored for later purposes (as expression)
    # evaluating: nonlinear expression can be evaluated (as user-defined function)
    # solvers: solvers can handle the constraint (as expression)
    # optimal value: 1.0, optimal point: [1.0, 1.0]
    # => SUFFICIENT FOR MY PURPOSE BUT VERY CHUNKY
    # NOT SUFFICIENT, SINCE THE EXPRESSION CANNOT BE RE-USED IN ANOTHER
    # MODEL AS x[1] AND x[2] ARE BOUND TO THIS MODEL
    ############################################################################
    # problem = JuMP.Model(GAMS.Optimizer)
    # JuMP.set_optimizer_attribute(problem, "Solver", "Ipopt")
    # @variable(problem, 0 <= x[i=1:2] <= 1)
    # @objective(problem, Max, x[2])
    # @constraint(problem, x[2] - x[1] <= 0)
    # express1 = :($(x[1])*$(x[2]))
    # express2 = :($(express1) <= 1)
    # add_NL_constraint(problem, express2)
    # print(problem)
    # optimize!(problem)
    # println()
    # println("optimal value: ", objective_value(problem))
    # println("optimal point: ", value.(x))
    # println()

    problem = JuMP.Model(GAMS.Optimizer)
    JuMP.set_optimizer_attribute(problem, "Solver", "Ipopt")
    @variable(problem, 0 <= x[i=1:2] <= 1)
    @objective(problem, Max, x[2])
    @constraint(problem, x[2] - x[1] <= 0)
    #register(problem, :nonlinearexp, 2, nonlinearexp, autodiff=true)
    println()
    express1 = :($(x[1])*$(x[2]))
    express2 = :($(express1) <= 1)
    JuMP.add_NL_constraint(problem, express2)

    function christian(y::JuMP.VariableRef)
        return :(sqrt($(y)))
    end
    a = function christian(y::Float64)
        return sqrt(y)
    end

    function christian2(x::JuMP.VariableRef, y::JuMP.VariableRef)
        return :($(x)*$(y))
    end
    function christian2(x::Float64, y::Float64)
        return x*y
    end

    expi = christian(x[1])
    expo = christian(x[2])

    println(expi)
    println(typeof(expi))
    println(expo)
    println(typeof(expo))

    problem2 = JuMP.Model(GAMS.Optimizer)
    JuMP.set_optimizer_attribute(problem2, "Solver", "Ipopt")
    @variable(problem2, 0 <= y[i=1:2] <= 1)

    expa = christian(y[1])
    println(expa)
    println(typeof(expa))
    println(christian(2.0))

    expe = christian2(x[1], x[2])
    println(expe)
    println(typeof(expe))

    println(a(3.0))

    b = NCNBD.solve_ncnbd
    println(b)
    b(2)

    #println(express2)
    #println(typeof(express2))
    #println(problem)
    #nlexp2 = :($(christian(x[1])) <= 1)
    #nlexp3 = :($(nlexp2))
    #println(nlexp3)
    #println(typeof(nlexp3))

    add_NL_constraint(problem, :($(expi) <= 1))
    add_NL_constraint(problem, :($(expe) <= 1))
    #print(problem)
    add_NL_constraint(problem2, :($(expa) <= 1))
    #print(problem2)
    #JuMP.optimize!(problem)
    println()
    #println("optimal value: ", objective_value(problem))
    #println("optimal point: ", value.(x))
    println()
#nlcon_4 = add_NL_constraint(subproblem, :($(Expr(:call, :nonlinearexp_2, x[2], x[1])) == $(nonlinearAux[2]) ))

end


exampleModelTest()
