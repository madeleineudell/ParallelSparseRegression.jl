export nnlsq, lasso, ridge, elastic_net

# admm_consensus accepts z0 (initial guess of solution) 
# and parameters are passed as optional keyword arguments
function nnlsq(A,b; rho=1, kwargs...)
    return admm_consensus([prox_pos!,make_prox_lsq(A, b, rho)],size(A,2); rho=rho, kwargs...)
end

function lasso(A,b,lambda; rho=1, kwargs...)
    return admm_consensus([make_prox_l1(lambda, rho),make_prox_lsq(A, b, rho)],size(A,2); rho=rho, kwargs...)
end

function ridge(A,b,lambda; rho=1, kwargs...)
    return admm_consensus([make_prox_l2(lambda, rho),make_prox_lsq(A, b, rho)],size(A,2); rho=rho, kwargs...)
end

function elastic_net(A,b,lambda1,lambda2; rho=1, kwargs...)
    return admm_consensus([make_prox_l1(lambda1, rho),make_prox_l2(lambda2, rho),make_prox_lsq(A, b, rho)],size(A,2); rho=rho, kwargs...)
end
