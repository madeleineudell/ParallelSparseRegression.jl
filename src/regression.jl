export nnlsq

function nnlsq(A,b; kwargs...)
    return admm_consensus([prox_pos,make_shared_prox_lsq(A, b, params.rho)],n; kwargs...)
end