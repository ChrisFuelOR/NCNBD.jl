using JuMP
using SDDP
using NCNBD
using Revise
using GAMS
#using SCIP
using Infiltrator

include("UC_1_10.jl")
#import .UC_1_10
include("UC_2_2.jl")
#import .UC_1_10
include("UC_2_5.jl")
#import .UC_1_10
include("UC_2_10.jl")
#import .UC_1_10
include("UC_5_5.jl")
#import .UC_1_10

function start_instances()

    # INSTANCE DEFINITIONS
    parameter_sets = [
                      [1e-8, 1e-8, :none, :kelley, 0.0],
                      [1e-8, 1e-8, :none, :bundle_level, 0.4],
                      [1e-8, 1e-8, :none, :bundle_level, 0.2],
                      [1e-8, 1e-8, :none, :bundle_level, 0.6],
                      [1e-8, 1e-8, :cplex_combi, :kelley, 0.0],
                      [1e-8, 1e-8, :cplex_combi, :bundle_level, 0.4],
                      [1e-8, 1e-8, :cplex_combi, :bundle_level, 0.2],
                      [1e-8, 1e-8, :cplex_combi, :bundle_level, 0.6],
                      [1e-4, 1e-4, :none, :kelley, 0.0],
                      [1e-4, 1e-4, :none, :bundle_level, 0.4],
                      [1e-4, 1e-4, :none, :bundle_level, 0.2],
                      [1e-4, 1e-4, :none, :bundle_level, 0.6],
                      [1e-4, 1e-4, :cplex_combi, :kelley, 0.0],
                      [1e-4, 1e-4, :cplex_combi, :bundle_level, 0.4],
                      [1e-4, 1e-4, :cplex_combi, :bundle_level, 0.2],
                      [1e-4, 1e-4, :cplex_combi, :bundle_level, 0.6],
                      ]

    for parameter_set in parameter_sets

        lagrangian_atol = parameter_set[1]
        lagrangian_rtol = parameter_set[2]
        dual_initialization_regime = parameter_set[3]
        lagrangian_method = parameter_set[4]
        level_factor = parameter_set[5]

        UC_1_10.unitCommitment_1_10_with_parameters(
            lagrangian_atol=lagrangian_atol, lagrangian_rtol=lagrangian_rtol,
            dual_initialization_regime=dual_initialization_regime,
            lagrangian_method=lagrangian_method, level_factor=level_factor
        )
        UC_2_2.unitCommitment_2_2_with_parameters(
            lagrangian_atol=lagrangian_atol, lagrangian_rtol=lagrangian_rtol,
            dual_initialization_regime=dual_initialization_regime,
            lagrangian_method=lagrangian_method, level_factor=level_factor
        )
        UC_2_5.unitCommitment_2_5_with_parameters(
            lagrangian_atol=lagrangian_atol, lagrangian_rtol=lagrangian_rtol,
            dual_initialization_regime=dual_initialization_regime,
            lagrangian_method=lagrangian_method, level_factor=level_factor
        )
        UC_2_10.unitCommitment_2_10_with_parameters(
            lagrangian_atol=lagrangian_atol, lagrangian_rtol=lagrangian_rtol,
            dual_initialization_regime=dual_initialization_regime,
            lagrangian_method=lagrangian_method, level_factor=level_factor
        )
        UC_5_5.unitCommitment_5_5_with_parameters(
            lagrangian_atol=lagrangian_atol, lagrangian_rtol=lagrangian_rtol,
            dual_initialization_regime=dual_initialization_regime,
            lagrangian_method=lagrangian_method, level_factor=level_factor
        )

    end
end

start_instances()
