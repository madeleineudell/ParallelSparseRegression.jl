export prox_pos!, make_prox_l1, make_prox_l2, make_prox_lsq

# the prox of the indicator of the positive orthant is thresholding negative values to 0.
# function prox_pos!(z::SharedArray)
#     @parallel for i=1:length(z)
#         z[i] = max(z[i],0)
#     end
# end

function prox_pos!(z::SharedArray)
    for i=1:length(z)
        z[i] = max(z[i],0)
    end
end

### The make_prox_* arguments close over rho; the user can change rho when creating the proxs, but not at runtime during the ADMM iterations

# return a function my_prox_l1 that computes the solution to 
# minimize \lambda \|x\|_1 + \rho/2 \|x-z\|_2^2
# as a function of z
# the solution is just soft thresholding
function make_prox_l1(lambda, rho)
    delta = rho/lambda
    function my_prox_l1!(z)
        @parallel for i=1:length(z)
            z[i] = max(z[i]-delta,0)
        end
    end
end

# return a function my_prox_l2 that computes the solution to 
# minimize \lambda/2 \|x\|_2^2 + \rho/2 \|x-z\|_2^2
# as a function of z
function make_prox_l2(lambda, rho)
    alpha = rho/(rho+lambda)
    function my_prox_l2!(z)
        @parallel for i=1:length(z)
            z[i] *= alpha
        end
    end
end

# return a function my_prox_lsq that computes the solution to 
# minimize \|Ax - b\|_2^2 + \rho/2 \|x-z\|_2^2
# as a function of z
function make_prox_lsq(A,b,rho; memory=:shared)    
    C = sparse([A, rho*eye(size(A,2))])
    T = Adivtype(A,b)
    m,n = size(C)
    if memory == :shared
        C = operator(C)
        temp_vars = SharedArray(T,m),SharedArray(T,n),SharedArray(T,m),SharedArray(T,n)
        d = SharedArray(T,m)
    elseif memory == :distributed
        error("Distributed memory least squares is not yet implemented")
    elseif memory == :local
        C = C
        temp_vars = Array(T,m),Array(T,n),Array(T,m),Array(T,n)
        d = Array(T,m)
        else
        error("$memory memory is not implemented. Try using memory=:shared or memory=:local.")
    end
    m1 = length(b)
    d[1:m1] = b
    function my_prox_lsq!(z::SharedArray)
        for i=n
            d[m1+i] = z[i]
        end
        lsqr!(z, C, d; temp_vars=temp_vars, maxiter=5)
    end
end