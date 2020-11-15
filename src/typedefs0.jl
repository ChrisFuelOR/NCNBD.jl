# Copyright Christian Füllner (Karlsruhe Institute of Technology) 2020
#
# This source code form is subject to the terms of the Mozilla Public License, v. x.x.
# If a copy of the MPL was not distributed with this file, you can obtain one
# at https://mozilla.org/MPL/x.x.

# This file is inspired by and re-uses parts from the source code of
# SDDiP.jl (lkapelevich),
# SLDP.jl (bfpc)
# and SDDP.jl (odow).

import JuMP
import Revise
#import SDDP

# Struct for algorithmic parameters
# ------------------------------------------------------------------------------------------------------------------
# epsilon_outerLoop: optimality tolerance for outer loop
# epsilon_innerLoop: optimality tolerance for inner loop
# binaryPrecision: epsilons for latest/current binary expansion (better use vector instead of dict?)
# plaPrecision: precision of the initial PLA (same for all stages, but can differ between nonlinearities so far)
# sigma: parameters used to obtain the regularized problem (better vector?)
# ------------------------------------------------------------------------------------------------------------------
# possible extensions:
# maxcuts ::  Int64 # maximum number of cuts to be stored (for storage efficiency)
# dropcuts ::  Int64 # number of cuts dropped so far
# what about number of constraints and variables in the model?
# what about differences in the binary expansion of different components?

# Mutable struct for algorithmic parameters that may change during the iterations
# Vector{Float64} or Dict{Int64, Float64}?
mutable struct AlgoParams
    epsilon_outerLoop :: Float64 # optimality tolerance for outer loop
    epsilon_innerLoop :: Float64 # optimality tolerance for inner loop
    binaryPrecision :: Vector{Float64} # Epsilons for latest/current binary expansion (better vector?)
    sigma :: Vector{Float64} # parameters used to obtain the regularized problem (better vector?)
end

# Struct for initial algorithmic parameters that remain fixed and characterize a model run
struct InitialAlgoParams
    epsilon_outerLoop :: Float64
    epsilon_innerLoop :: Float64
    binaryPrecision :: Vector{Float64}
    plaPrecision :: Vector{Float64}
    sigma :: Vector{Float64}
end

# struct for Triangulation
# better to use dicts instead of vectors without index?
mutable struct Triangulation
    vertices :: Union{Vector{Float64}, Array{Float64,2}}
    verticeValues :: Vector{Float64}
    simplices :: Array{Int64, 2}
    precision :: Float64
    plrVariables :: Vector{JuMP.VariableRef}
    plrConstraints :: Vector{JuMP.ConstraintRef}
    maxOverestimation :: Vector{Float64}
    maxUnderestimation :: Vector{Float64}
    # An extension dictionary.
    ext::Dict{Symbol,Any}
end

# struct for nonlinear functions
# Specify first argument to type of user-defined function?
mutable struct NonlinearFunction
    nonlinfunc_eval :: Any # for evaluation
    nonlinfunc_exp :: Any # for constraint building
    auxVariable :: JuMP.VariableRef # for definition of (PLA) constraints
    # refToNonlinearConstraint :: JuMP.ConstraintRef # just for allocation # not used anymore and does not work for add_NL_constraint
    variablesContained :: Vector{JuMP.VariableRef} # for getting bounds for Triangulation
    triangulation :: Union{Triangulation, Nothing} # to store related Triangulation

    function NonlinearFunction(
        nonlinfunc_eval::Any, # Function
        nonlinfunc_exp::Any, #Function
        auxVariable::JuMP.VariableRef,
        #refToNonlinearConstraint::JuMP.ConstraintRef,
        variablesContained::Vector{JuMP.VariableRef}
         )
        return new(
            nonlinfunc_eval,
            nonlinfunc_exp,
            auxVariable,
            #refToNonlinearConstraint,
            variablesContained,
            nothing
        )
    end


end


# struct for solvers to be used (maybe mutable)
struct AppliedSolvers
    LP :: Any
    MILP :: Any
    MINLP :: Any
end

# Struct to store information on a nonlinear cut
struct NonlinearCut
    intercept   ::  Float64 # intercept of the cut (Lagrangian function value)
    gradient    ::  Vector{Float64} # optimal dual variables in binary space
    trialPoint  ::  Vector{Float64} # point at which this cut was created
    binaryNum   ::  Int64 # number of binary variables at moment of creation
    binaryEps   ::  Float64 # binary precision at moment of creation
end
    # what about differences in the binary expansion of all components?
    # do we need to store the trial point also in binary? I think not because
    # we can always convert it. Do we really need binaryNum and binaryEps?



#nlf1 = NonlinearFunction(variables_nfl1, nonlinearAux, nlcon, quadr)

#struct PiecewiseLinearRelaxation
#    triangulation
#    simplices
#    errorToShift        ::  #Must be related to triangulation simplices (ggf. in PWL)
#end