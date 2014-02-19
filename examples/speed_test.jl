addprocs(int(ARGS[1]))
using ParallelSparseRegression

function speed_test(memory=:shared)
	m,n,p = 10,5,.5
	A = sprand(m,n,p)
	x0 = Base.shmem_randn(n)
	b = A*x0
	rho = 1
	lambda = 1
	quiet = false
	maxiters = 100
	params = Params(rho,quiet,maxiters)
	println("Making proxs")
	@time proxs = [make_prox_l1(lambda, params.rho),make_prox_lsq(A, b, params.rho; memory=memory)]
	# println("Evaluate each prox 5 times")
	# x0.s += .1*randn(n)
	# for prox in proxs
	# 	@time begin for i=1:5
	# 		prox(x0)
	# 	end end
	# end
	@time admm_consensus(proxs,size(A,2); params=params)
end


println("Speed test using $(nprocs()) processors")
if length(ARGS)>1 && ARGS[2] == "local"
	memory=:local
elseif ARGS[2] == "shared"
	memory=:shared
end
speed_test(memory)

 
