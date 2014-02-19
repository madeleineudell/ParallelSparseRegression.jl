module ParallelSparseRegression

using IterativeSolvers
using ParallelSparseMatMul

include("admm.jl")
include("prox.jl")
include("regression.jl")

end # module
