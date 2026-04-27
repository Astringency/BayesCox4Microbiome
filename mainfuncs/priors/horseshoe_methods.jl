function _validate_prior(prior::HorseshoePrior)
    _check_positive(prior.tau_scale, "tau_scale")
    _check_positive(prior.p0, "p0")
    _check_positive(prior.local_scale, "local_scale")
    return prior
end

function _resolved_tau_scale(prior::HorseshoePrior, p::Integer, n_events::Integer)
    return prior.use_cox_tau0 ? cox_tau0(p, prior.p0, n_events) : prior.tau_scale
end

function _state_names(::HorseshoePrior, p::Integer)
    return vcat(["beta[$j]" for j in 1:p], ["log_tau"], ["log_lambda[$j]" for j in 1:p])
end

function initial_state(
    prior::HorseshoePrior,
    p::Integer,
    n_events::Integer;
    beta0 = zeros(Float64, p),
)
    _validate_prior(prior)
    length(beta0) == p || throw(DimensionMismatch("length(beta0) must be p."))
    tau0 = _resolved_tau_scale(prior, p, n_events)
    return vcat(collect(Float64, beta0), [log(tau0)], zeros(p))
end

function _prior_logposterior(
    u::AbstractVector,
    data::CoxPreparedData,
    prior::HorseshoePrior;
    car_weights = nothing,
    car_precision::Real = 1.0,
)
    p = size(data.z, 2)
    length(u) == 2p + 1 || throw(DimensionMismatch("wrong state length for HorseshoePrior."))
    beta = view(u, 1:p)
    log_tau = u[p + 1]
    log_lambda = view(u, (p + 2):(2p + 1))
    tau = exp(log_tau)
    lambda = exp.(log_lambda)
    tau_scale = _resolved_tau_scale(prior, p, count(!=(0), data.event))

    lp = _halfcauchy_logpdf(tau, tau_scale) + log_tau
    for j in 1:p
        lp += _halfcauchy_logpdf(lambda[j], prior.local_scale) + log_lambda[j]
        lp += logpdf(Normal(0, tau * lambda[j]), beta[j])
    end
    lp += _car_logprior(log_lambda, car_weights, car_precision)
    return lp
end
