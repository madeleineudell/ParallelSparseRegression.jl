# ParallelSparseRegression

[![Build Status](https://travis-ci.org/madeleineudell/ParallelSparseRegression.jl.png)](https://travis-ci.org/madeleineudell/ParallelSparseRegression.jl)

A Julia library for parallel sparse regression using shared memory.
This library intends to implement solvers for regression problems
including least squares, ridge regression, lasso, non-negative least squares,
and elastic net.
It will also contain fast methods to obtain regularization paths.

Using the (Alternating Direction Method of Multipliers)[http://www.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf],
all of these problems can be reduced to computing the prox of each term in the objective.
We rely on the fact that the prox of each term in the objective
of these regression problems can be efficiently computed in parallel.

# Installation

To install, just open a Julia prompt and call

    Pkg.clone("git@github.com:madeleineudell/ParallelSparseRegression.jl.git")
	
# Usage

Before you begin, initialize all the processes you want to participate in multiplying by your matrix.
You'll suffer decreased performance if you add more processes 
than you have hyperthreads on your shared-memory computer.

    addprocs(3)
    using ParallelSparseRegression
    
We will solve a sparse non-negative least squares problem.

    m,n,p = 100,20,.1
    A = sprand(m,n,p)
    x0 = Base.shmem_randn(n)
    b = A*x0
    rho = 1
    quiet = false
    maxiters = 100

    params = Params(rho,quiet,maxiters)
    z = nnlsq(A,b; params=params)

We can verify the solution obtained is better than merely thresholding
the entries of the least squares solution to be positive.

    println("Norm of Az-b is $(norm(A*z-b))")
    xp = max(x0,0)
    println("Norm of A(x)_+ -b is $(norm(A*xp-b))")
  
