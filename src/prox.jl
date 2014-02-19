export prox_pos, make_shared_prox_lsq

function prox_pos(z)
    # XXX parallelize for shared z
    max(z,0)
end

# return a function my_prox_lsq that computes the solution to 
# minimize \|Ax - b\|_2^2 + \rho/2 \|x-z\|_2^2
# as a function of rho and z
# using shared memory to parallelize matrix vector products
function make_shared_prox_lsq(A,b,rho)    
    C = share(sparse([A, rho*eye(size(A,2))]))
    # the argument rho is not used in my_prox_pos, so changing it at runtime won't work
    function my_prox_lsq(z)
        d = [b, z]
        zp, ch = lsqr(C, d; maxiter = 5)
        return zp
    end
    return my_prox_lsq
end