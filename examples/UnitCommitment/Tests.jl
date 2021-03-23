# Copyright (c) 2021 Christian Fuellner <christian.fuellner@kit.edu>

# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
################################################################################

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
