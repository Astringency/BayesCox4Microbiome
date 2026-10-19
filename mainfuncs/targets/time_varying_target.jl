"""Replace a static Cox likelihood by a time-varying likelihood, or remove it
for sampling the same model's normalized prior. Static priors remain unchanged.
"""
struct CoxTimeVaryingTarget{T<:AbstractCoxTarget} <: AbstractCoxTarget
    base::T
    design::TimeCoxDesign
    slope_design::Matrix{Float64}
    slope_scale::Float64
    likelihood_weight::Float64
    coefficient_indices::Vector{Int}
    static_data::CoxPreparedData
    static_work::CoxGradientWorkspace
end
function CoxTimeVaryingTarget(base::AbstractCoxTarget,design::TimeCoxDesign,slope_design;
    slope_scale=0.5,likelihood_weight=1.0)
    slope_scale>0 && 0<=likelihood_weight<=1 || throw(ArgumentError("invalid time target controls"))
    size(slope_design,1)==size(design.X,2) || throw(DimensionMismatch("time slope basis"))
    indices=findall(n -> startswith(n,"beta[") || startswith(n,"clinical["),sample_state_names(base))
    length(indices)==size(design.X,2) || throw(DimensionMismatch("static coefficients"))
    static_data=base isa CoxClinicalTarget ? base.joint_data : base.data
    return CoxTimeVaryingTarget(base,design,Matrix{Float64}(slope_design),
        Float64(slope_scale),Float64(likelihood_weight),indices,static_data,CoxGradientWorkspace(static_data))
end
_time_names(t)= ["time_contrast[$j]" for j in axes(t.slope_design,2)]
state_names(t::CoxTimeVaryingTarget)=vcat(state_names(t.base),_time_names(t))
sample_state_names(t::CoxTimeVaryingTarget)=vcat(sample_state_names(t.base),_time_names(t))
initial_state(t::CoxTimeVaryingTarget,rng::AbstractRNG=Random.default_rng())=
    vcat(t.base isa CoxPosteriorTarget ? initial_state(t.base) : initial_state(t.base,rng),
        zeros(size(t.slope_design,2)))
function sample_state(t::CoxTimeVaryingTarget,u::AbstractVector)
    d=dimension(t.base)
    return vcat(sample_state(t.base,view(u,1:d)),u[d+1:end])
end

# Pull a coefficient score back through the existing noncentered RHS map.
function _add_coefficient_score!(g,base::AbstractCoxTarget,u,score)
    p=size(base.data.z,2)
    if base isa CoxNoncenteredPosteriorTarget
        beta=view(sample_state(base,u),1:p)
        for j in 1:p
            raw_logscale=u[p+1]+u[p+1+j]
            if base.prior isa RegularizedHorseshoePrior
                slab_fraction=_logistic(2raw_logscale-u[2p+2]);raw_fraction=1-slab_fraction
                logscale=min(raw_logscale,u[2p+2]/2)-log1p(exp(-2abs(raw_logscale-u[2p+2]/2)))/2
                g[2p+2]+=score[j]*beta[j]*slab_fraction/2
            else
                raw_fraction,logscale=1.0,raw_logscale
            end
            g[j]+=score[j]*exp(logscale)
            v=score[j]*beta[j]*raw_fraction
            g[p+1]+=v;g[p+1+j]+=v
        end
    else
        g[1:p].+=score
    end
    return g
end
function _add_coefficient_score!(g,base::CoxClinicalTarget,u,score)
    d=dimension(base.base);p=size(base.base.data.z,2)
    _add_coefficient_score!(view(g,1:d),base.base,view(u,1:d),view(score,1:p))
    g[d+1:end].+=score[p+1:end]
    return g
end
function logdensity_and_gradient!(g::AbstractVector{Float64},t::CoxTimeVaryingTarget,u::AbstractVector)
    d=dimension(t.base);base_u=view(u,1:d);time_u=view(u,d+1:length(u))
    lp,_=logdensity_and_gradient!(view(g,1:d),t.base,base_u)
    beta=sample_state(t.base,base_u)[t.coefficient_indices]
    if !isfinite(lp) || !all(isfinite,time_u)
        fill!(g,0.0);return -Inf,g
    end
    gm=zeros(length(beta))
    lm=cox_logpartiallikelihood_and_score_gradient!(gm,t.static_data,beta,t.static_work)
    lj,gj,gs=t.likelihood_weight==0 ? (0.0,zeros(length(beta)),zeros(length(beta))) :
        time_cox_loglikelihood_and_score(t.design,beta,t.slope_design*time_u)
    if !isfinite(lm) || !isfinite(lj)
        fill!(g,0.0);return -Inf,g
    end
    _add_coefficient_score!(view(g,1:d),t.base,base_u,t.likelihood_weight.*gj.-gm)
    g[d+1:end].=t.likelihood_weight.*(t.slope_design'*gs).-time_u./t.slope_scale^2
    lp+=t.likelihood_weight*lj-lm
    for v in time_u;lp+=logpdf(Normal(0,t.slope_scale),v);end
    return lp,g
end
logdensity(t::CoxTimeVaryingTarget,u::AbstractVector)=first(logdensity_and_gradient!(zeros(length(u)),t,u))
gradient!(g::AbstractVector{Float64},t::CoxTimeVaryingTarget,u::AbstractVector)=last(logdensity_and_gradient!(g,t,u))
function sample_logdensity(t::CoxTimeVaryingTarget,saved::AbstractVector,internal::AbstractVector)
    d=dimension(t.base);ds=length(sample_state_names(t.base))
    jacobian=sample_logdensity(t.base,view(saved,1:ds),view(internal,1:d))-logdensity(t.base,view(internal,1:d))
    return logdensity(t,internal)+jacobian
end
