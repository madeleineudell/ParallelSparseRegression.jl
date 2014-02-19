export prox_pos, make_prox_l1, make_prox_l2, make_prox_lsq

# the prox of the indicator of the positive orthant is thresholding negative values to 0.
function prox_pos(z)
    max(z,0)
end

### The make_prox_* arguments close over rho; the user can change rho when creating the proxs, but not at runtime during the ADMM iterations

# return a function my_prox_l1 that computes the solution to 
# minimize \lambda \|x\|_1 + \rho/2 \|x-z\|_2^2
# as a function of z
# the solution is just soft thresholding
function make_prox_l1(lambda, rho)
    delta = rho/lambda
    function my_prox_l1(z)
        max(z - delta,0)
    end
end

# return a function my_prox_l2 that computes the solution to 
# minimize \lambda/2 \|x\|_2^2 + \rho/2 \|x-z\|_2^2
# as a function of z
function make_prox_l2(lambda, rho)
    alpha = rho/(rho+lambda)
    function my_prox_l2(z)
        alpha*z
    end
end

# return a function my_prox_lsq that computes the solution to 
# minimize \|Ax - b\|_2^2 + \rho/2 \|x-z\|_2^2
# as a function of z
# using shared memory to parallelize matrix vector products
function make_prox_lsq(A,b,rho)    
    C = share(sparse([A, rho*eye(size(A,2))]))
    function my_prox_lsq(z)
        d = [b, z]
        zp, ch = lsqr(C, d; maxiter = 5)
        return zp
    end
end