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
