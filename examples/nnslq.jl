require("src/ParallelSparseRegression.jl")
using ParallelSparseRegression

m,n,p = 20,4,.5
A = sprand(m,n,p)
x0 = Base.shmem_randn(n)
# m,n,p = 6,4,.5
# A = speye(m,n)
# x0 = [-1, 1, -1, 1]
b = A*x0
rho = 1
quiet = false
maxiters = true

params = Params(rho,quiet,30)
z = nnlsq(A,b; params=params)

# compute true solution using cvxpy
# XXX ... to do

println("Norm of Az-b is $(norm(A*z-b))")
println("Norm of Ax-b is $(norm(A*x0-b))")
xp = prox_pos(x0)
println("Norm of A(x)_+ -b is $(norm(A*xp-b))")