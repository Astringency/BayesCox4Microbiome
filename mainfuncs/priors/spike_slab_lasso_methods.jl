function _validate_prior(prior::SpikeSlabLassoPrior)
    _check_positive(prior.spike_scale, "spike_scale")
    _check_positive(prior.slab_scale, "slab_scale")
    prior.spike_scale < prior.slab_scale ||
        throw(ArgumentError("spike_scale must be smaller than slab_scale."))
    _check_positive(prior.theta_a, "theta_a")
    _check_positive(prior.theta_b, "theta_b")
    return prior
end

function _state_names(::SpikeSlabLassoPrior, p::Integer; kwargs...)
    return vcat(
        ["beta[$j]" for j in 1:p],
        ["log_tau2_spike[$j]" for j in 1:p],
        ["log_tau2_slab[$j]" for j in 1:p],
        ["logit_theta"],
    )
end

function initial_state(
    prior::SpikeSlabLassoPrior,
    p::Integer,
    n_events::Integer;
    beta0 = zeros(Float64, p),
    kwargs...,
)
    _validate_prior(prior)
    length(beta0) == p || throw(DimensionMismatch("length(beta0) must be p."))
    return vcat(
        collect(Float64, beta0),
        fill(log(2 * prior.spike_scale^2), p),
        fill(log(2 * prior.slab_scale^2), p),
        [0.0],
    )
end

function _ssl_tau2_rate(scale::Real)
    return inv(2 * scale^2)
end

function _ssl_tau2_logpdf(tau2::Real, scale::Real)
    tau2 > 0 || return -Inf
    rate = _ssl_tau2_rate(scale)
    return log(rate) - rate * tau2
end

function _ssl_component_logpdf(beta, theta, log_tau2_spike, log_tau2_slab)
    log_spike = log1p(-theta) + _normal_zero_logpdf_from_logscale(beta, 0.5 * log_tau2_spike)
    log_slab = log(theta) + _normal_zero_logpdf_from_logscale(beta, 0.5 * log_tau2_slab)
    norm = _logsumexp2(log_spike, log_slab)
    return log_spike, log_slab, norm
end

function _prior_logposterior(
    u::AbstractVector,
    data::CoxPreparedData,
    prior::SpikeSlabLassoPrior;
    kwargs...,
)
    p = size(data.z, 2)
    length(u) == 3p + 1 || throw(DimensionMismatch("wrong state length for SpikeSlabLassoPrior."))
    beta = view(u, 1:p)
    log_tau2_spike = view(u, (p + 1):(2p))
    log_tau2_slab = view(u, (2p + 1):(3p))
    logit_theta = u[3p + 1]
    theta = _logistic(logit_theta)

    lp = logpdf(Beta(prior.theta_a, prior.theta_b), theta) + log(theta) + log1p(-theta)
    for j in 1:p
        tau2_spike = exp(log_tau2_spike[j])
        tau2_slab = exp(log_tau2_slab[j])
        lp += _ssl_tau2_logpdf(tau2_spike, prior.spike_scale) + log_tau2_spike[j]
        lp += _ssl_tau2_logpdf(tau2_slab, prior.slab_scale) + log_tau2_slab[j]
        _, _, norm = _ssl_component_logpdf(beta[j], theta, log_tau2_spike[j], log_tau2_slab[j])
        lp += norm
    end
    return lp
end

function _prior_logposterior_and_gradient!(
    grad::AbstractVector{Float64},
    u::AbstractVector,
    data::CoxPreparedData,
    prior::SpikeSlabLassoPrior;
    kwargs...,
)
    p = size(data.z, 2)
    length(u) == 3p + 1 || throw(DimensionMismatch("wrong state length for SpikeSlabLassoPrior."))
    length(grad) == length(u) || throw(DimensionMismatch("gradient length does not match state."))
    beta = view(u, 1:p)
    log_tau2_spike = view(u, (p + 1):(2p))
    log_tau2_slab = view(u, (2p + 1):(3p))
    logit_theta = Float64(u[3p + 1])
    theta = _logistic(logit_theta)

    lp = logpdf(Beta(prior.theta_a, prior.theta_b), theta) + log(theta) + log1p(-theta)
    grad[3p + 1] += prior.theta_a * (1.0 - theta) - prior.theta_b * theta
    @inbounds for j in 1:p
        b = Float64(beta[j])
        lt_spike = Float64(log_tau2_spike[j])
        lt_slab = Float64(log_tau2_slab[j])
        tau2_spike = exp(lt_spike)
        tau2_slab = exp(lt_slab)

        lp += _ssl_tau2_logpdf(tau2_spike, prior.spike_scale) + lt_spike
        lp += _ssl_tau2_logpdf(tau2_slab, prior.slab_scale) + lt_slab
        log_spike, log_slab, norm = _ssl_component_logpdf(b, theta, lt_spike, lt_slab)
        prob_spike = exp(log_spike - norm)
        prob_slab = exp(log_slab - norm)
        lp += norm
        grad[j] -= b * (prob_spike / tau2_spike + prob_slab / tau2_slab)
        grad[p + j] += 1.0 - _ssl_tau2_rate(prior.spike_scale) * tau2_spike +
            prob_spike * (-0.5 + 0.5 * b^2 / tau2_spike)
        grad[2p + j] += 1.0 - _ssl_tau2_rate(prior.slab_scale) * tau2_slab +
            prob_slab * (-0.5 + 0.5 * b^2 / tau2_slab)
        grad[3p + 1] += prob_slab - theta
    end
    return lp
end
