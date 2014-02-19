module ParallelSparseRegression

import IterativeSolvers: lsqr, lsqr!
using ParallelSparseMatMul

include("admm.jl")
include("prox.jl")
include("regression.jl")

end # module
