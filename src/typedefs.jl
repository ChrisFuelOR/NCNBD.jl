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

# struct for Simplex
mutable struct Simplex
    vertices :: Array{Float64,2}
    vertice_values :: Vector{Float64}
    maxOverestimation :: Float64
    maxUnderestimation :: Float64
end

# struct for Triangulation
# better to use dicts instead of vectors without index?
mutable struct Triangulation
    simplices :: Vector{Simplex}
    precision :: Float64
    plrVariables :: Vector{JuMP.VariableRef}
    plrConstraints :: Vector{JuMP.ConstraintRef}
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
    ext::Dict{Symbol,Any} # required to store solutions later

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
            nothing,
            Dict{Symbol,Any}()
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
    coefficients  ::  Dict{Symbol,Float64} # optimal dual variables in binary space
    trial_state  ::  Dict{Symbol,Float64} # point at which this cut was created
    binary_precision :: Float64 # binary precision at moment of creation
    cutVariables :: Vector{JuMP.VariableRef}
    cutConstraints :: Vector{JuMP.ConstraintRef}
    cutVariables_lin :: Vector{JuMP.VariableRef}
    cutConstraints_lin :: Vector{JuMP.ConstraintRef}
    obj_y::Union{Nothing,NTuple{N,Float64} where {N}} # SDDP
    belief_y::Union{Nothing,Dict{T,Float64} where {T}} # SDDP
    non_dominated_count::Int # SDDP
end
    # TODO: Do we need to store the trial point also in binary form?
    # I think not because we can always determine it using trial_state
    # and binary_precision.
    # TODO: If the binary precision may be different for all components,
    # then we have to adapt this.
    # TODO: Should we also store the expression of this cut in binary space?

# struct for outer loop iteration results
struct OuterLoopIterationResult#{T}
    # pid
    lower_bound :: Float64 # here the inner or outer loop lower bound can be used
    upper_bound :: Float64 # should be renamed as cumulative_value as in SDDP if we solve stochastic problems
    current_sol :: Array{Dict{Symbol,Float64},1} #Vector{Dict{Symbol, Float64}}
    has_converged :: Bool
    status :: Symbol # solution status (i.e. number of iterations)
    #nonlinearCuts :: Dict{T, Vector{Any}} # only required for logging, binary explanation
    # however, then also binary precision / K should be stored for interpretability
end

# struct for inner loop iteration results
struct InnerLoopIterationResult#{T}
    # pid
    lower_bound :: Float64
    upper_bound :: Float64 # should be renamed as cumulative_value as in SDDP if we solve stochastic problems
    current_sol :: Array{Dict{Symbol,Float64},1} #Vector{Dict{Symbol, Float64}} # current solution of state variables (also required for binary refinement)
    has_converged :: Bool
    status :: Symbol # solution status (i.e. number of iterations)
    #nonlinearCuts :: Dict{T, Vector{Any}} # only required for logging, binary explanation
    # however, then also binary precision / K should be stored for interpretability
end
