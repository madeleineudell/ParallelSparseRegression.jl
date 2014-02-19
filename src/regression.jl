export nnlsq

function nnlsq(A,b; params=Params(), kwargs...)
    return admm_consensus([prox_pos,make_shared_prox_lsq(A, b, params.rho)],size(A,2); kwargs...)
end