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
include("UC_3_5.jl")
include("UC_3_10.jl")

function start_instances()

    # INSTANCE DEFINITIONS
    parameter_sets = [
                      #[UC_1_10, 1e-8, 1e-8, :none, :kelley, 0.0, 1e-2, 1e-2],

                      #[UC_2_2, 1e-8, 1e-8, :none, :kelley, 0.0, 1e-2, 1e-2],
                      #[UC_2_2, 1e-8, 1e-8, :none, :bundle_level, 0.2, 1e-2, 1e-2],
                      #[UC_2_2, 1e-8, 1e-8, :cplex_combi, :kelley, 0.0, 1e-2, 1e-2],
                      #[UC_2_2, 1e-8, 1e-8, :cplex_combi, :bundle_level, 0.2, 1e-2, 1e-2],
                      #[UC_2_2, 1e-4, 1e-4, :none, :kelley, 0.0, 1e-2, 1e-2],
                      #[UC_2_2, 1e-4, 1e-4, :none, :bundle_level, 0.2, 1e-2, 1e-2],

                      #[UC_2_5, 1e-8, 1e-8, :none, :kelley, 0.0, 1e-2, 1e-2],
                      #[UC_2_5, 1e-8, 1e-8, :none, :bundle_level, 0.2, 1e-2, 1e-2],
                      # [UC_2_5, 1e-8, 1e-8, :cplex_combi, :kelley, 0.0],
                      # [UC_2_5, 1e-8, 1e-8, :cplex_combi, :bundle_level, 0.2],
                      #[UC_2_5, 1e-4, 1e-4, :none, :kelley, 0.0, 1e-2, 1e-2],
                      #[UC_2_5, 1e-4, 1e-4, :none, :bundle_level, 0.2, 1e-2, 1e-2],

                      #[UC_2_10, 1e-8, 1e-8, :none, :kelley, 0.0, 1e-2, 1e-2],
                      #[UC_2_10, 1e-8, 1e-8, :none, :bundle_level, 0.2, 1e-2, 1e-2],
                      #[UC_2_10, 1e-4, 1e-4, :none, :kelley, 0.0, 1e-2, 1e-2],
                      #[UC_2_10, 1e-4, 1e-4, :none, :bundle_level, 0.2, 1e-2, 1e-2],

                      #[UC_3_5, 1e-8, 1e-8, :none, :kelley, 0.0, 1e-2, 1e-2],
                      #[UC_3_5, 1e-8, 1e-8, :none, :bundle_level, 0.2, 1e-2, 1e-2],
                      #[UC_3_5, 1e-4, 1e-4, :none, :kelley, 0.0, 1e-2, 1e-2],
                      #[UC_3_5, 1e-4, 1e-4, :none, :bundle_level, 0.2, 1e-2, 1e-2],

                      #[UC_3_5, 1e-8, 1e-8, :none, :kelley, 0.0, 1e-2, 1e-2],
                      #[UC_3_5, 1e-8, 1e-8, :none, :bundle_level, 0.2, 1e-2, 1e-2],
                      [UC_3_10, 1e-4, 1e-4, :none, :kelley, 0.0, 1e-2, 1e-2],
                      [UC_3_10, 1e-4, 1e-4, :none, :bundle_level, 0.2, 1e-2, 1e-2],

                      #[UC_5_5, 1e-8, 1e-8, :none, :kelley, 0.0, 1e-2, 1e-2],
                      #[UC_5_5, 1e-8, 1e-8, :none, :bundle_level, 0.2, 1e-2, 1e-2],
                      #[UC_5_5, 1e-4, 1e-4, :none, :bundle_level, 0.2, 1e-2, 1e-2],
                      #[UC_5_5, 1e-4, 1e-4, :none, :kelley, 0.0, 1e-2, 1e-2],
                      #[UC_5_5, 1e-4, 1e-4, :none, :bundle_level, 0.2, 1e-2, 1e-2],
                      # [UC_5_5, 1e-8, 1e-8, :cplex_combi, :kelley, 0.0],
                      # [UC_5_5, 1e-8, 1e-8, :cplex_combi, :bundle_level, 0.2],
                      # [UC_5_5, 1e-4, 1e-4, :none, :kelley, 0.0],
                      # [UC_5_5, 1e-4, 1e-4, :none, :bundle_level, 0.2],
                      ]

    for parameter_set in parameter_sets

        module_name = parameter_set[1]
        lagrangian_atol = parameter_set[2]
        lagrangian_rtol = parameter_set[3]
        dual_initialization_regime = parameter_set[4]
        lagrangian_method = parameter_set[5]
        level_factor = parameter_set[6]
        rgap_outer_loop = parameter_set[7]
        rgap_inner_loop = parameter_set[8]

        # used solvers
        solvers = ["CPLEX", "CPLEX", "Baron", "Baron", "CPLEX"]

        module_name.unitCommitment_with_parameters(
            lagrangian_atol=lagrangian_atol, lagrangian_rtol=lagrangian_rtol,
            dual_initialization_regime=dual_initialization_regime,
            lagrangian_method=lagrangian_method, level_factor=level_factor,
            solvers=solvers, epsilon_outerLoop = rgap_outer_loop,
            epsilon_innerLoop = rgap_inner_loop
        )

    end
end

start_instances()
