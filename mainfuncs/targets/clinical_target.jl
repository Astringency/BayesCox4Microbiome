"""Joint Cox likelihood with clinical coefficients outside the taxon prior/constraint.

The base target retains its original taxon prior and parameterization. Replacing
its likelihood by the joint likelihood avoids treating clinical columns as
compositional taxa. Each chain owns its target and gradient workspaces.
"""
struct CoxClinicalTarget{T<:AbstractCoxTarget} <: AbstractCoxTarget
    base::T
    joint_data::CoxPreparedData
    clinical_names::Vector{String}
    prior_df::Float64
    prior_scale::Float64
    micro_work::CoxGradientWorkspace
    joint_work::CoxGradientWorkspace
end

function joint_cox_data(data::CoxPreparedData, clinical::AbstractMatrix)
    size(clinical, 1) == size(data.z, 1) || throw(DimensionMismatch("clinical row count"))
    all(isfinite, clinical) || throw(ArgumentError("clinical values must be finite"))
    z = hcat(data.z, clinical)
    # This object is for likelihood evaluation only; no composition transform.
    return CoxPreparedData(z, data.time, data.event, data.risk, data.tie_method,
        :none, 0.0, false, false, zeros(size(z, 2)), ones(size(z, 2)))
end

function CoxClinicalTarget(base::AbstractCoxTarget, clinical::AbstractMatrix;
    names = ["clinical[$j]" for j in axes(clinical, 2)], prior_df = 3.0, prior_scale = 2.5)
    base isa Union{CoxNoncenteredPosteriorTarget,CoxPosteriorTarget,CoxProxSSLassoTarget} ||
        throw(ArgumentError("clinical adjustment supports explicit HS/RHS and Prox-SSLASSO targets"))
    base.approx === nothing || throw(ArgumentError("clinical adjustment requires the exact Cox likelihood"))
    size(clinical, 2) == length(names) > 0 || throw(DimensionMismatch("clinical column names"))
    prior_df > 0 && prior_scale > 0 || throw(ArgumentError("clinical prior df and scale must be positive"))
    joint = joint_cox_data(base.data, clinical)
    return CoxClinicalTarget(base, joint, String.(names), Float64(prior_df), Float64(prior_scale),
        CoxGradientWorkspace(base.data), CoxGradientWorkspace(joint))
end

state_names(t::CoxClinicalTarget) = vcat(state_names(t.base), t.clinical_names)
sample_state_names(t::CoxClinicalTarget) = vcat(sample_state_names(t.base), t.clinical_names)
function initial_state(t::CoxClinicalTarget, rng::AbstractRNG = Random.default_rng())
    u = t.base isa CoxPosteriorTarget ? initial_state(t.base) : initial_state(t.base, rng)
    return vcat(u, zeros(length(t.clinical_names)))
end
function sample_state(t::CoxClinicalTarget, u::AbstractVector)
    d = dimension(t.base)
    return vcat(sample_state(t.base, view(u, 1:d)), u[d+1:end])
end

function _clinical_correction(t::CoxClinicalTarget, beta, delta)
    all(isfinite, beta) && all(isfinite, delta) || return -Inf
    lm = cox_logpartiallikelihood_and_score_gradient!(zeros(length(beta)), t.base.data, beta, t.micro_work)
    lj = cox_logpartiallikelihood_and_score_gradient!(zeros(length(beta)+length(delta)), t.joint_data,
        vcat(beta, delta), t.joint_work)
    return lj - lm + _clinical_logprior(t, delta)
end
function _clinical_logprior(t::CoxClinicalTarget, delta)
    lp = 0.0
    for x in delta
        lp += logpdf(TDist(t.prior_df), x / t.prior_scale) - log(t.prior_scale)
    end
    return lp
end

function logdensity(t::CoxClinicalTarget, u::AbstractVector)
    return first(logdensity_and_gradient!(zeros(length(u)), t, u))
end

function logdensity_and_gradient!(g::AbstractVector{Float64}, t::CoxClinicalTarget, u::AbstractVector)
    d, p = dimension(t.base), size(t.base.data.z, 2)
    micro, delta = view(u, 1:d), view(u, d+1:length(u))
    beta = view(sample_state(t.base, micro), 1:p)
    lp, _ = logdensity_and_gradient!(view(g, 1:d), t.base, micro)
    if !isfinite(lp) || !all(isfinite, beta) || !all(isfinite, delta)
        fill!(g, 0.0)
        return -Inf, g
    end
    gm, gj = zeros(p), zeros(size(t.joint_data.z, 2))
    lm = cox_logpartiallikelihood_and_score_gradient!(gm, t.base.data, beta, t.micro_work)
    lj = cox_logpartiallikelihood_and_score_gradient!(gj, t.joint_data, vcat(beta, delta), t.joint_work)
    if !isfinite(lm) || !isfinite(lj)
        fill!(g, 0.0)
        return -Inf, g
    end
    db = view(gj, 1:p) .- gm
    if t.base isa CoxNoncenteredPosteriorTarget
        for j in 1:p
            raw_logscale = micro[p+1] + micro[p+1+j]
            if t.base.prior isa RegularizedHorseshoePrior
                slab_fraction = _logistic(2raw_logscale - micro[2p+2])
                raw_fraction = 1 - slab_fraction
                logscale = min(raw_logscale, micro[2p+2]/2) -
                    log1p(exp(-2abs(raw_logscale-micro[2p+2]/2)))/2
                g[2p+2] += db[j] * beta[j] * slab_fraction / 2
            else
                raw_fraction, logscale = 1.0, raw_logscale
            end
            g[j] += db[j] * exp(logscale)
            v = db[j] * beta[j] * raw_fraction
            g[p+1] += v
            g[p+1+j] += v
        end
    else
        g[1:p] .+= db
    end
    for j in eachindex(delta)
        x = delta[j]
        g[d+j] = gj[p+j] - (t.prior_df+1) * x / (t.prior_df*t.prior_scale^2+x^2)
    end
    return lp + lj - lm + _clinical_logprior(t, delta), g
end
gradient!(g::AbstractVector{Float64}, t::CoxClinicalTarget, u::AbstractVector) =
    last(logdensity_and_gradient!(g, t, u))
function sample_logdensity(t::CoxClinicalTarget, saved::AbstractVector, internal::AbstractVector)
    d, p = dimension(t.base), size(t.base.data.z, 2)
    ds = length(sample_state_names(t.base))
    return sample_logdensity(t.base, view(saved, 1:ds), view(internal, 1:d)) +
        _clinical_correction(t, view(saved, 1:p), view(saved, ds+1:length(saved)))
end

function posterior_clinical_draws(result::CoxSamplerResult)
    result.samples isa Union{MHSamples,MultiMHSamples} ||
        return zeros(size(posterior_beta_draws(result), 1), 0)
    samples = result.samples isa MultiMHSamples ? result.samples.chains : [result.samples]
    idx = findall(startswith("clinical["), first(samples).names)
    return reduce(vcat, [s.draws[:, idx] for s in samples])
end
