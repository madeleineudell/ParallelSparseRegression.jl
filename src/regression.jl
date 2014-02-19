export nnlsq, lasso, ridge, elastic_net

# admm_consensus accepts z0 (initial guess of solution) 
# and params as optional keyword arguments
function nnlsq(A,b; kwargs...)
    return admm_consensus([prox_pos,make_prox_lsq(A, b, params.rho)],size(A,2); kwargs...)
end

function lasso(A,b,lambda; kwargs...)
    return admm_consensus([make_prox_l1(lambda, params.rho),make_prox_lsq(A, b, params.rho)],size(A,2); kwargs...)
end

function ridge(A,b,lambda; kwargs...)
    return admm_consensus([make_prox_l2(lambda, params.rho),make_prox_lsq(A, b, params.rho)],size(A,2); kwargs...)
end

function elastic_net(A,b,lambda1,lambda2; kwargs...)
    return admm_consensus([make_prox_l1(lambda1, params.rho),make_prox_l2(lambda2, params.rho),make_prox_lsq(A, b, params.rho)],size(A,2); kwargs...)
end