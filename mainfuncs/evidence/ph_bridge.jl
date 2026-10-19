"""Conditional Gaussian coordinates for bridge proposals, with exact Jacobian.
The Cox Hessian is used only to fit the proposal; density evaluations always
use the original, exact Cox likelihood and prior. This removes the shrinkage
funnel and the narrow zero-sum direction without changing model evidence.
"""
function _ph_bridge_transform(model,training; posterior=true)
    p,c=model.p,model.c
    d=length(model.names_internal)-size(ph_time_design(p,c,model.scope),2)-c
    indices=vcat(collect(1:p),collect(d+1:length(model.names_internal)))
    center=vec(mean(training[:,indices];dims=1));k=length(indices)
    T=ph_time_design(p,c,model.scope)
    function score(v)
        _,gb,gs=time_cox_loglikelihood_and_score(model.time_design,view(v,1:p+c),
            T*view(v,p+c+1:k))
        return vcat(gb,T'*gs)
    end
    H=zeros(k,k);linear=zeros(k)
    if posterior
        for j in 1:k
            h=1e-4*max(1.0,abs(center[j]));a=copy(center);b=copy(center);a[j]+=h;b[j]-=h
            H[:,j].=-(score(a).-score(b))./(2h)
        end
        H=(H+H')/2
        # Numerical Hessian roundoff affects only the auxiliary proposal.
        eig=eigen(Symmetric(H));H=eig.vectors*Diagonal(max.(eig.values,0.0))*eig.vectors'
        linear=H*center+score(center)
    end
    sum_variance=model.constraint==:prox ? model.sumlambda*p : _default_soft_center_scale(p)^2
    function conditional(u,g)
        scales=if model.method=="cox-rhs"
            raw=u[p+1].+u[p+2:2p+1];slab=u[2p+2]/2
            exp.([clamp(min(x,slab)-log1p(exp(-2abs(x-slab)))/2,-230.,14.) for x in raw])
        else
            alpha=exp(clamp(u[p+1],-230.,14.))
            radius=alpha.*[g[j] ? model.prior.slab_scale : model.prior.spike_scale for j in 1:p].*
                sqrt(2/((p+1)*(p+2)))
            hypot.(radius,sqrt(get(model.kwargs,:prox_l1_lambda,1e-3)))
        end
        precision=copy(H)
        for j in 1:p;precision[j,j]+=max(inv(scales[j]^2),1e-8);end
        for j in p+1:p+c;precision[j,j]+=inv(3*2.5^2);end
        for j in p+c+1:k;precision[j,j]+=inv(model.time_prior_scale^2);end
        precision[1:p,1:p].+=inv(sum_variance)
        F=cholesky(Symmetric(precision))
        return F,F\linear
    end
    function forward(u,g)
        F,mu=conditional(u,g);v=copy(u)
        v[indices].=F.U*(u[indices].-mu)
        return v
    end
    function inverse(v,g)
        F,mu=conditional(v,g);u=copy(v)
        u[indices].=mu.+F.U\v[indices]
        return u,-sum(log,diag(F.U))
    end
    return (;forward,inverse,indices)
end

"""Bridge normalizer in proposal coordinates fitted to training halves only."""
function ph_bridge_sampling(model,sample;posterior=true,seed=1)
    training=Int[]
    for chain in unique(sample.chain_labels)
        ids=findall(==(chain),sample.chain_labels)
        append!(training,ids[1:length(ids)÷2])
    end
    map=_ph_bridge_transform(model,sample.draws[training,:];posterior)
    transformed=reduce(vcat,[permutedims(map.forward(view(sample.draws,i,:),view(sample.indicators,i,:)))
        for i in axes(sample.draws,1)])
    function target(v,g)
        u,jacobian=map.inverse(v,g)
        return model.logprior(u,g)+(posterior ? model.loglikelihood(u,g) : 0.0)+jacobian
    end
    return bridge_sampling(target,transformed;indicators=sample.indicators,chain_labels=sample.chain_labels,seed)
end
