using ParallelSparseRegression

m,n,p = 100,20,.1
A = sprand(m,n,p)
x0 = Base.shmem_randn(n)
b = A*x0
lambda = 1

params = ((:rho,1), (:quiet, false), (:maxiters,100))

# Non-negative least squares
z_nnlsq = nnlsq(A,b; params...)

# Non-negative least squares
z_lasso = lasso(A,b,lambda; params...)

# Non-negative least squares
z_ridge = ridge(A,b,lambda; params...)

# Non-negative least squares
z_elasticnet = elastic_net(A,b,lambda,lambda; params...)

# XXX to do: compare answers to cvxpy
# using PyCall
# @pyimport cvxpy
