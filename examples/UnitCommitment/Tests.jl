module Tests

import JuMP
import SDDP
import NCNBD
import Revise
import Gurobi
import GAMS
using Infiltrator

include("InstanceStarter.jl")
include("UC_1_10.jl")
include("UC_2_2.jl")
include("UC_2_5.jl")
include("UC_2_10.jl")
include("UC_5_5.jl")

end
