export admm, admm_consensus

function admm(prox_f, prox_g, A; rho=1, quiet=false, ABSTOL=1e-4, RELTOL=1e-2, maxiters=1000, AA=nothing, F=nothing)
# Generic graph projection splitting solver. Solve
#
#  minimize   f(y) + g(x)
#  subject to y = Ax
# 
# given prox_f, prox_g, and A.
#
# Optional arguments: 
#           AA --- the square of A (ie, A*A' or A'*A, whichever is smaller). Only used if A is dense.
#           F  --- a factorization of A. 
#                   If A is sparse, F is a factorization of [ speye(n) A' ; A -speye(m) ];
#                   If A is dense, F is a factorization of eye(n) + AA;

    tic();

    m, n = size(A);
    xx = zeros(n,1);   xz = zeros(n,1);   xt = zeros(n,1);
    yx = zeros(m,1);   yz = zeros(m,1);   yt = zeros(m,1);
    zx = zeros(n+m,1); zz = zeros(n+m,1); zt = zeros(n+m,1);

    if AA == nothing && ~issparse(A)
        if m < n
            AA = A*A';
        else
            AA = A'*A;
        end
    end

    sqrtn = sqrt(n);

    if ~quiet
        @printf("iter :\t%8s\t%8s\t%8s\t%8s\n", "r", "eps_pri", "s", "eps_dual");
    end

    for iter = 1:1000
        yx = prox_f(yz - yt, rho);
        xx = prox_g(xz - xt, rho);
        zx = [xx;yx];
        zzprev = copy(zz)

        if issparse(A)
            zz, F = project_graph(zx + zt, A, [], F);
        else
            zz, F = project_graph(zx + zt, A, AA, F);
        end
        xz, yz = zz[1:n], zz[n+1:end]

        # termination checks
        eps_pri  = sqrtn*ABSTOL + RELTOL*max(norm(zx), norm(zz));
        eps_dual = sqrtn*ABSTOL + RELTOL*norm(rho*zt);
        prires = norm(zx - zz);
        duares = rho*norm(zz - zzprev);

        if ~quiet && (iter == 1 || mod(iter,10) == 0)
            @printf("%4d :\t%.2e\t%.2e\t%.2e\t%.2e\n", iter, prires, eps_pri, duares, eps_dual);
        end

        if iter > 2 && prires < eps_pri && duares < eps_dual
            if ~quiet
                @printf("total iterations: %d\n", iter);
                toc()
            break;
            end
        end

        xt = xt + xx - xz;
        yt = yt + yx - yz;
        zt = [xt;yt];
    end

    yt = yt*rho;
    return xx
end

function project_graph(v, A, AA=[], F = nothing)
# Project v onto the graph of A. 
# Supports factorization caching and both dense/sparse A.

    m,n = size(A);
    c = v[1:n];
    d = v[n+1:end];

    if issparse(A)
        if F == nothing
            K = [ speye(n) A' ; A -speye(m) ];
            F = cholfact(K);
        end

        # indexing ranges isn't supported for sparse matrices, so we make it full
        z = full(F \ sparse([ c + A'*d ; zeros(m,1) ]));

        return z,F
    else
        if m <= n
            if F == nothing
                F = cholfact(eye(m) + AA);
            end
            y = F \ (A*c + AA*d);
            x = c + A'*(d - y);
        else
            if F == nothing
                F = cholfact(eye(n) + AA);
            end
            x = F \ (c + A'*d);
            y = A*x;
        end

        return [x;y], F
    end

end

function admm_consensus(proxs, n; rho=1, quiet=false, ABSTOL=1e-4, RELTOL=1e-2, maxiters=1000, AA=nothing, F=nothing, z= nothing)
# Generic consensus solver. Solve
#
#  minimize   f_1(x_1) + ... + f_m(x_m)
#  subject to z = x_i, i = 1,...,m
# 
#  given proxs = [prox_f1, ..., prox_fm], where z \in \reals^n

#  here we may parallelize inside the proxs, but don't compute the proxs in parallel

    tic();

    m = length(proxs)
    sqrtm = sqrt(m)
    if z == nothing
        T = Float64
        z = SharedArray(T,n)    
    else
        T = eltype(z)
    end
    xs = [SharedArray(T,n) for i=1:m] 
    ys = [SharedArray(T,n) for i=1:m] 
    zprev = copy(z)

    sqrtn = sqrt(n);

    if ~quiet
        @printf("iter :\t%8s\t%8s\t%8s\t%8s\n", "r", "eps_pri", "s", "eps_dual");
    end

    for iter = 1:maxiters
        for i=1:m
            xs[i][:] = z - ys[i]
            proxs[i](xs[i])
        end

        zprev[:] = z
        z[:] = mean(xs)+mean(ys)

        # termination checks
        eps_pri  = sqrtn*ABSTOL + RELTOL*max(sum([norm(x) for x in xs]), norm(z));
        eps_dual = sqrtn*ABSTOL + RELTOL*rho*sum([norm(y) for y in ys]);
        prires = sum([norm(x-z) for x in xs]);
        duares = rho*norm(z - zprev);

        if ~quiet && (iter == 1 || mod(iter,10) == 0)
            @printf("%4d :\t%.2e\t%.2e\t%.2e\t%.2e\n", iter, prires, eps_pri, duares, eps_dual);
        end

        if iter > 2 && prires < eps_pri && duares < eps_dual
            if ~quiet
                @printf("total iterations: %d\n", iter);
                toc()
            break;
            end
        end

        for i=1:m
            ys[i][:] += xs[i] - z
        end 
    end

    return z
end
