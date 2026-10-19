function _rhs_constrained_normal(rng,scales,sum_variance)
    variance=sum(abs2,scales)
    z=randn(rng,length(scales))
    variance==0 && return scales.*z
    shrink=1-sqrt(sum_variance/(sum_variance+variance))
    return scales.*(z.-(shrink*dot(scales,z)/variance).*scales)
end

"""Independent draws from the actual zero-sum-penalized Cox-RHS prior.
Given shrinkage scales s, integrating the Gaussian beta prior times the penalty
gives w=sqrt(a/(a+sum(s²)))≤1. Rejecting independent hyperprior draws with this
probability, then drawing beta from its Gaussian conditional, is exact.
The soft-center normal-density constant cancels in rejection probabilities and
is retained in the auxiliary normalizer estimate. No population resampling is used.
"""
function sample_ph_rhs_prior(model;n_samples=10000,chains=4,seed=1,max_proposals=10000n_samples*chains)
    model.method=="cox-rhs" && !model.prior.sample_pseudo_sigma ||
        throw(ArgumentError("independent PH prior sampling requires Cox-RHS with fixed pseudo scale"))
    n=n_samples*chains;p=model.p;c=model.c
    prior=model.prior
    tau_scale=_global_tau_cauchy_scale(_resolved_tau_scale(prior,model.data;pseudo_sigma=prior.pseudo_sigma))
    a=model.constraint==:prox ? model.sumlambda*p : _default_soft_center_scale(p)^2
    rng=MersenneTwister(seed);draws=zeros(n,length(model.names_internal));accepted=0;proposed=0
    q=size(ph_time_design(p,c,model.scope),2)
    while accepted<n
        proposed+=1
        proposed<=max_proposals || error("RHS prior rejection budget exhausted; increase max_proposals")
        tau=abs(rand(rng,Cauchy(0,tau_scale)))
        local_scales=abs.(rand(rng,Cauchy(0,prior.local_scale),p))
        c2=rand(rng,InverseGamma(prior.slab_df/2,prior.slab_df*prior.slab_s2/2))
        tau>0 && all(>(0),local_scales) && isfinite(tau) && all(isfinite,local_scales) && isfinite(c2) || continue
        scales=exp.([_regularized_horseshoe_beta_logscale(tau,l,c2) for l in local_scales])
        rand(rng)<sqrt(a/(a+sum(abs2,scales))) || continue
        accepted+=1
        draws[accepted,1:p].=_rhs_constrained_normal(rng,scales,a)
        draws[accepted,p+1]=log(tau)
        draws[accepted,p+2:2p+1].=log.(local_scales)
        draws[accepted,2p+2]=log(c2)
        for j in 1:c;draws[accepted,2p+2+j]=2.5rand(rng,TDist(3));end
        q>0 && (draws[accepted,end-q+1:end].=model.time_prior_scale.*randn(rng,q))
    end
    probability=(n-1)/(proposed-1)
    constant=model.constraint==:soft_center ? -log(2pi*a)/2 : 0.0
    return (draws=draws,indicators=falses(n,0),chain_labels=repeat(1:chains;inner=n_samples),
        transitions=DataFrame(),sampling_method="independent hyperprior rejection and conditional Gaussian",
        prior_rejection_log_normalizer=constant+log(probability),
        prior_rejection_log_mcse=sqrt((1-probability)/n),prior_proposals=proposed)
end
