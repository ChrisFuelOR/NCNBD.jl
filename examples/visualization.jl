using JuMP
using GLPK
using LinearAlgebra

# Visualization
function visualize_cut(c, p, B, x_lb, x_ub , e)
    # define variable dimensions
    dim_c = length(c)
    dim_p = length(p)
    n_row_B = size(B)[1]
    n_col_B = size(B)[2]
    dim_x = n_row_B

    # if dim > 2 the cut can't be visualized
    if dim_p > 2
        @assert dim_p > 2
    end

    # define x
    x = fill(0.0, dim_x)

    # discretize around the middle of the area
    for i in 1:dim_x

        diff = x_ub[i] - x_lb[i]

        if diff > 1
            x[i] = round(x_lb[i] + 0.5*diff, digits=1)
        else
            # find the right digit
            current_digit = get_digit(diff)
            x[i] = round(x_lb[i] + 0.5*diff, digits=current_digit + 1)
        end
    end

    # solve the projection problem
    solution = solve_sub(c, p, B, e, x)

    # print the problem in a .txt file
    # t.b.d
    println(solution)
end

function solve_sub(c, p, B, e, x)

    # solve a problem instance

    # define dimensions
    dim_c = length(c)
    dim_p = length(p)
    n_row_B = size(B, 1)
    n_col_B = size(B, 2)
    dim_x = length(x)

    # define the optimization problem
    opt_problem = Model(GLPK.Optimizer)
    #println("Object created...")
    @variable(opt_problem, 0 <= lambda[1:dim_p])
    #println("Lambda defined...")
    @objective(opt_problem, Max, dot(fill(1, dim_c), c) + dot(p, lambda))
    #println("Objective defined...")
    @constraint(opt_problem, B*lambda .== x)
    @constraint(opt_problem, lambda .<= e)
    #println("Constraints defined...")

    # solve the optimization problem
    optimize!(opt_problem)

    lambda_opt = value.(lambda)
    objective_opt = objective_value(opt_problem)

    # return optimal vector and objective function value
    return lambda_opt, objective_opt
end

function get_digit(x)
    digit = 0
    var = abs(x)

    # in this case we run in to an infinite loop
    if var == 0
        @assert var = 0
    end

    while true
        if var > 1
            break
        end
        var = var*10
        digit += 1
    end
    return digit
end


# test the example
global c = [4]
global p = [2 -3 -6.5]
global B = [5/7 10/7 20/7]
global e = [1 1 1]
global x = [0 1]

#solve_sub(c, p, B, e, x)
visualize_cut(c, p, B, [1 4], [3 3.5] , e)
