using ParallelSparseRegression

function speed_test(memory=:shared)
	m,n,p = 100000,100000,.0001
	A = sprand(m,n,p)
	x0 = Base.shmem_randn(n)
	b = A*x0
	rho = 1
	lambda = 1
	params = ((:rho, rho), (:quiet, false), (:maxiters, 20))
	println("Making proxs")
	@time proxs = [make_prox_l1(lambda, rho),make_prox_lsq(A, b, rho; memory=memory)]
	# println("Evaluate each prox 5 times")
	# x0.s += .1*randn(n)
	# for prox in proxs
	# 	@time begin for i=1:5
	# 		prox(x0)
	# 	end end
	# end
	@time admm_consensus(proxs,size(A,2); params...)
end

