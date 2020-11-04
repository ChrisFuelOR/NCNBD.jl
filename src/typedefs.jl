# Copyright Christian Füllner (Karlsruhe Institute of Technology) 2020
#
# This source code form is subject to the terms of the Mozilla Public License, v. x.x.
# If a copy of the MPL was not distributed with this file, you can obtain one
# at https://mozilla.org/MPL/x.x.

# This file is inspired by source code from SDDiP.jl (lkapelevich), SLDP.jl
# (bfpc) and SDDP.jl (odow).

import JuMP
import Revise
#import SDDP

# Struct for algorithmic parameters that may change during the iterations
# Note that some parameters can change between stages and are thus arrays.
mutable struct AlgoParams
    sigma       ::  Array{Float64,1} # parameters used to obtain the regularized problem
    binaryNum   ::  Array{Int64} # Numbers of binary variables of latest binary expansion
    binaryEps   ::  Array{Float64,1} # Epsilons for latest/current binary expansion
    maxcuts     ::  Int64 # maximum number of cuts to be stored (for storage efficiency)
    dropcuts    ::  Int64 # number of cuts dropped so far
end
    # what about number of constraints and variables in the model?
    # check whether mutable is required
    # what about differences in the binary expansion of all components?

# Struct for initial algorithmic parameters that remain fixed to characterize a model run.
struct InitialAlgoParams
    sigma       ::  Float64 # parameters used to obtain the regularized problem (array later for stages)
    initialSimplices :: Int64 # number of simplices for the PLAs used in the beginning (could also be an array)
    initialBinaryPrecision :: Float64 # initial binary precision for all binary expansions (could also be an array)
    maxcuts     ::  Int64 # maximum number of cuts to be stored (for storage efficiency)
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


# Do I need matrix B explicitly?

# struct for nonlinear functions used in the GAMS model
#struct NonlinearFunction{F<:Function}
struct NonlinearFunction
    variableList::Vector{JuMP.VariableRef} #Array{Vector{JuMP.variable},1}
    auxVariable::JuMP.VariableRef
    constraintRef::JuMP.ConstraintRef  #check in SDDP
    func::Any
#    #triangulation       ::  #Triangulation
#    #piecewiseLinearApp  ::  #PiecewiseLinearApproximationStructure

# Constructor
    #NonlinearFunction{F}(variableList, auxVariable, constraintRef, func) where {F<:Function} = new(variableList, auxVariable, constraintRef, func)

    #function NonlinearFunction(;
#        variableList::Vector{JuMP.VariableRef},
#        auxVariable::JuMP.VariableRef,
#        constraintRef::JuMP.ConstraintRef,
#        func::F
#    ){F<:Function}
#        return new{typeof{func}}(
#            variableList, auxVariable, constraintRef, func
#        )
#    end
end


#nlf1 = NonlinearFunction(variables_nfl1, nonlinearAux, nlcon, quadr)

#struct PiecewiseLinearRelaxation
#    triangulation
#    simplices
#    errorToShift        ::  #Must be related to triangulation simplices (ggf. in PWL)
#end
