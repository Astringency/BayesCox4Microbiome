"""Radius coordinates for the SSLASSO prior-only target.

The contrast scales combine the alpha-dependent epigraph width and the Moreau
smoothing width. The coefficient-sum direction is scaled by its conditional
Gaussian approximation. The scale map follows the
current indicator pattern. Indicator updates in
these coordinates use the full transformed density, including the exact
Jacobian, so their stationary distribution remains the original joint prior.
"""
struct CoxRadiusPriorTarget{T<:CoxTimeVaryingTarget} <: AbstractCoxTarget
    base::T
    p::Int
    sum_variance::Float64
end
function CoxRadiusPriorTarget(base::CoxTimeVaryingTarget)
    base.likelihood_weight==0 || throw(ArgumentError("radius coordinates require a prior-only target"))
    micro=base.base isa CoxClinicalTarget ? base.base.base : base.base
    micro isa CoxProxSSLassoTarget || throw(ArgumentError("radius coordinates require SSLASSO"))
    p=length(micro.gamma)
    a=micro.constraint==:prox ? micro.prox_sum0_lambda*p : _default_soft_center_scale(p)^2
    return CoxRadiusPriorTarget(base,p,a)
end
state_names(t::CoxRadiusPriorTarget)=vcat(["radius_z[$j]" for j in 1:t.p],state_names(t.base)[t.p+1:end])
sample_state_names(t::CoxRadiusPriorTarget)=sample_state_names(t.base)
function _radius_scales(t,u)
    micro=t.base.base isa CoxClinicalTarget ? t.base.base.base : t.base.base
    scales=[g ? micro.prior.slab_scale : micro.prior.spike_scale for g in micro.gamma].*
        sqrt(2/((t.p+1)*(t.p+2)))
    eta=u[t.p+1];loglambda=log(micro.prox_l1_lambda)
    logradius=eta.+log.(scales)
    logscales=[_logaddexp(2x,loglambda)/2 for x in logradius]
    # The Moreau envelope has Gaussian-width tails even when alpha is small.
    # This is only a coordinate map; the exact prior and Jacobian are retained.
    scales=exp.(logscales)
    kappa=exp.(2 .* (logradius.-logscales))
    unit=scales/norm(scales)
    ratio=log(sum(abs2,scales)/t.sum_variance)
    correction=max(ratio,0)+log1p(exp(-abs(ratio)))
    shrink=exp(-correction/2)
    jac=sum(logscales)-correction/2
    return scales,shrink,jac,_logistic(-ratio),unit,kappa
end
function sample_state(t::CoxRadiusPriorTarget,u::AbstractVector)
    scales,shrink,_,_,unit=_radius_scales(t,u)
    z=view(u,1:t.p);parallel=unit.*dot(unit,z);v=copy(u)
    v[1:t.p].=scales.*((z.-parallel).+shrink.*parallel)
    return v
end
function _radius_internal(t::CoxRadiusPriorTarget,v::AbstractVector)
    scales,shrink,_,_,unit=_radius_scales(t,v)
    y=view(v,1:t.p)./scales;parallel=unit.*dot(unit,y);u=copy(v)
    u[1:t.p].=(y.-parallel).+parallel./shrink
    return u
end

# Evaluate the prior directly: subtracting two huge Cox likelihoods is both
# unnecessary and numerically inaccurate in the far tails of a prior draw.
function _ssl_ph_prior_and_gradient!(g,t::CoxTimeVaryingTarget,u)
    fill!(g,0.0)
    clinical=t.base isa CoxClinicalTarget
    micro=clinical ? t.base.base : t.base
    p=length(micro.gamma);d=p+1
    lp,_=_ssl_prox_prior_logdensity_and_gradient!(view(g,1:d),view(u,1:d),
        micro.prior,micro.gamma,micro.theta[],micro.work;
        prox_l1_lambda=micro.prox_l1_lambda,alpha_prior_shape=micro.alpha_prior_shape,
        alpha_prior_scale=micro.alpha_prior_scale,alpha_prior_kind=micro.alpha_prior_kind,
        prox_root_tol=micro.prox_root_tol,prox_root_maxiter=micro.prox_root_maxiter)
    beta=view(u,1:p)
    lp+=_constraint_logpenalty(beta;constraint=micro.constraint,prox_sum0_lambda=micro.prox_sum0_lambda)
    _constraint_gradient_penalty!(view(g,1:p),beta;constraint=micro.constraint,prox_sum0_lambda=micro.prox_sum0_lambda)
    if clinical
        for j in 1:length(t.base.clinical_names)
            x=u[d+j];df=t.base.prior_df;scale=t.base.prior_scale
            lp+=logpdf(TDist(df),x/scale)-log(scale)
            g[d+j]=-(df+1)*x/(df*scale^2+x^2)
        end
        d+=length(t.base.clinical_names)
    end
    for j in d+1:length(u)
        lp+=logpdf(Normal(0,t.slope_scale),u[j]);g[j]=-u[j]/t.slope_scale^2
    end
    return lp,g
end
function logdensity_and_gradient!(g::AbstractVector{Float64},t::CoxRadiusPriorTarget,u::AbstractVector)
    v=sample_state(t,u);scales,shrink,jac,fraction,unit,kappa=_radius_scales(t,u)
    if !all(isfinite,v) || !isfinite(jac)
        fill!(g,0.0);return -Inf,g
    end
    lp,_=_ssl_ph_prior_and_gradient!(g,t.base,v)
    beta_gradient=copy(view(g,1:t.p));z=view(u,1:t.p);parallel=unit.*dot(unit,z)
    average=dot(kappa,abs2.(unit))
    unit_derivative=(kappa.-average).*unit
    shrink_derivative=-shrink*(1-fraction)*average
    rotated=(z.-parallel).+shrink.*parallel
    rotated_derivative=shrink_derivative.*parallel.+(shrink-1).*
        (unit_derivative.*dot(unit,z).+unit.*dot(unit_derivative,z))
    beta_derivative=scales.*(kappa.*rotated.+rotated_derivative)
    g[t.p+1]+=dot(beta_gradient,beta_derivative)+sum(kappa)-(1-fraction)*average
    weighted=scales.*beta_gradient;projection=unit.*dot(unit,weighted)
    g[1:t.p].=(weighted.-projection).+shrink.*projection
    return lp+jac,g
end
logdensity(t::CoxRadiusPriorTarget,u::AbstractVector)=first(logdensity_and_gradient!(zeros(length(u)),t,u))
gradient!(g::AbstractVector{Float64},t::CoxRadiusPriorTarget,u::AbstractVector)=last(logdensity_and_gradient!(g,t,u))
sample_logdensity(t::CoxRadiusPriorTarget,saved::AbstractVector,internal::AbstractVector)=
    first(_ssl_ph_prior_and_gradient!(zeros(length(saved)),t.base,saved))
