mutable struct LevelOneOracle
    cuts::Union{Vector{SDDP.Cut}, Vector{NCNBD.NonlinearCut}}
    states::Vector{SDDP.SampledState}
    cuts_to_be_deleted::Union{Vector{SDDP.Cut}, Vector{NCNBD.NonlinearCut}}
    deletion_minimum::Int
    function LevelOneOracle(deletion_minimum)
        return new(SDDP.Cut[], SDDP.SampledState[], SDDP.Cut[], deletion_minimum)
    end
end
