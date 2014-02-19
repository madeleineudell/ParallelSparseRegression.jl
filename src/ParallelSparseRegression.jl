module ParallelSparseRegression

import IterativeSolvers: lsqr!, Adivtype
using ParallelSparseMatMul

include("admm.jl")
include("prox.jl")
include("regression.jl")

end # module
