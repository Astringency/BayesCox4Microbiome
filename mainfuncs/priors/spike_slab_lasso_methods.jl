function _validate_prior(prior::SpikeSlabLassoPrior)
    _check_positive(prior.spike_scale, "spike_scale")
    _check_positive(prior.slab_scale, "slab_scale")
    prior.spike_scale < prior.slab_scale ||
        throw(ArgumentError("spike_scale must be smaller than slab_scale."))
    _check_positive(prior.theta_a, "theta_a")
    _check_positive(prior.theta_b, "theta_b")
    return prior
end

function _state_names(::SpikeSlabLassoPrior, p::Integer)
    return vcat(["beta[$j]" for j in 1:p], ["logit_theta"])
end

function initial_state(
    prior::SpikeSlabLassoPrior,
    p::Integer,
    n_events::Integer;
    beta0 = zeros(Float64, p),
)
    _validate_prior(prior)
    length(beta0) == p || throw(DimensionMismatch("length(beta0) must be p."))
    return vcat(collect(Float64, beta0), [0.0])
end

function _ssl_marginal_logpdf(beta, theta, spike_scale, slab_scale)
    return _logsumexp2(
        log1p(-theta) + _laplace_logpdf(beta, spike_scale),
        log(theta) + _laplace_logpdf(beta, slab_scale),
    )
end

function _prior_logposterior(
    u::AbstractVector,
    data::CoxPreparedData,
    prior::SpikeSlabLassoPrior;
    car_weights = nothing,
    car_precision::Real = 1.0,
)
    p = size(data.z, 2)
    length(u) == p + 1 || throw(DimensionMismatch("wrong state length for SpikeSlabLassoPrior."))
    beta = view(u, 1:p)
    logit_theta = u[p + 1]
    theta = _logistic(logit_theta)

    lp = logpdf(Beta(prior.theta_a, prior.theta_b), theta) + log(theta) + log1p(-theta)
    for j in 1:p
        lp += _ssl_marginal_logpdf(beta[j], theta, prior.spike_scale, prior.slab_scale)
    end
    return lp
end

function _prior_logposterior_and_gradient!(
    grad::AbstractVector{Float64},
    u::AbstractVector,
    data::CoxPreparedData,
    prior::SpikeSlabLassoPrior;
    car_weights = nothing,
    car_precision::Real = 1.0,
)
    p = size(data.z, 2)
    length(u) == p + 1 || throw(DimensionMismatch("wrong state length for SpikeSlabLassoPrior."))
    length(grad) == length(u) || throw(DimensionMismatch("gradient length does not match state."))
    beta = view(u, 1:p)
    logit_theta = Float64(u[p + 1])
    theta = _logistic(logit_theta)

    lp = logpdf(Beta(prior.theta_a, prior.theta_b), theta) + log(theta) + log1p(-theta)
    grad[p + 1] += prior.theta_a * (1.0 - theta) - prior.theta_b * theta
    @inbounds for j in 1:p
        b = Float64(beta[j])
        log_spike = log1p(-theta) + _laplace_logpdf(b, prior.spike_scale)
        log_slab = log(theta) + _laplace_logpdf(b, prior.slab_scale)
        norm = _logsumexp2(log_spike, log_slab)
        prob_spike = exp(log_spike - norm)
        prob_slab = exp(log_slab - norm)
        lp += norm
        sgn = b > 0 ? 1.0 : b < 0 ? -1.0 : 0.0
        grad[j] -= sgn * (prob_spike / prior.spike_scale + prob_slab / prior.slab_scale)
        grad[p + 1] += prob_slab - theta
    end
    return lp
end
