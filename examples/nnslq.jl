using ParallelSparseRegression

m,n,p = 100,20,.1
A = sprand(m,n,p)
x0 = Base.shmem_randn(n)
b = A*x0

params = ((:rho, 1), (:quiet, false), (:maxiters, 100))

z = nnlsq(A,b; params...)

println("Norm of Az-b is $(norm(A*z-b))")
println("Norm of Ax-b is $(norm(A*x0-b))")
xp = max(x0,0)
println("Norm of A(x)_+ -b is $(norm(A*xp-b))")
