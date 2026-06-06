function _validate_prior(prior::BayesianLassoPrior)
    if prior.lambda !== nothing
        _check_positive(prior.lambda, "lambda")
    end
    _check_positive(prior.lambda2_shape, "lambda2_shape")
    _check_positive(prior.lambda2_rate, "lambda2_rate")
    return prior
end

function _state_names(prior::BayesianLassoPrior, p::Integer; kwargs...)
    names = vcat(["beta[$j]" for j in 1:p], ["log_tau2[$j]" for j in 1:p])
    if prior.lambda === nothing
        push!(names, "log_lambda2")
    end
    return names
end

function initial_state(
    prior::BayesianLassoPrior,
    p::Integer,
    n_events::Integer; rng::AbstractRNG = Random.default_rng(), init::Symbol = :zero,
    beta0 = init == :rand ? _default_unconstrained_init(rng, p) : zeros(Float64, p),
    kwargs...,
)
    _validate_prior(prior)
    length(beta0) == p || throw(DimensionMismatch("length(beta0) must be p."))
    if prior.lambda === nothing
        lambda2_init = prior.lambda2_shape / prior.lambda2_rate
        tau2_init = 2 / lambda2_init
        return vcat(collect(Float64, beta0), init == :rand ? _default_unconstrained_init(rng, p) : fill(log(tau2_init), p), [log(lambda2_init)])
    else
        tau2_init = 2 / prior.lambda^2
        return vcat(collect(Float64, beta0), init == :rand ? _default_unconstrained_init(rng, p) : fill(log(tau2_init), p))
    end
end

function _prior_logposterior(
    u::AbstractVector,
    data::CoxPreparedData,
    prior::BayesianLassoPrior;
    kwargs...,
)
    p = size(data.z, 2)
    expected = prior.lambda === nothing ? 2p + 1 : 2p
    length(u) == expected || throw(DimensionMismatch("wrong state length for BayesianLassoPrior."))
    beta = view(u, 1:p)
    log_tau2 = view(u, (p + 1):(2p))

    if prior.lambda === nothing
        log_lambda2 = u[2p + 1]
        lambda2 = exp(log_lambda2)
        lp = logpdf(Gamma(prior.lambda2_shape, 1 / prior.lambda2_rate), lambda2) + log_lambda2
    else
        lambda2 = prior.lambda^2
        lp = 0.0
    end

    for j in 1:p
        tau2 = exp(log_tau2[j])
        lp += logpdf(Gamma(1.0, 2 / lambda2), tau2) + log_tau2[j]
        lp += logpdf(Normal(0, sqrt(tau2)), beta[j])
    end
    return lp
end

function _prior_logposterior_and_gradient!(
    grad::AbstractVector{Float64},
    u::AbstractVector,
    data::CoxPreparedData,
    prior::BayesianLassoPrior;
    kwargs...,
)
    p = size(data.z, 2)
    expected = prior.lambda === nothing ? 2p + 1 : 2p
    length(u) == expected || throw(DimensionMismatch("wrong state length for BayesianLassoPrior."))
    length(grad) == length(u) || throw(DimensionMismatch("gradient length does not match state."))
    beta = view(u, 1:p)
    log_tau2 = view(u, (p + 1):(2p))

    if prior.lambda === nothing
        log_lambda2 = Float64(u[2p + 1])
        lambda2 = exp(log_lambda2)
        lp = logpdf(Gamma(prior.lambda2_shape, 1 / prior.lambda2_rate), lambda2) + log_lambda2
        grad[2p + 1] += prior.lambda2_shape - prior.lambda2_rate * lambda2
    else
        lambda2 = prior.lambda^2
        lp = 0.0
    end

    @inbounds for j in 1:p
        lt = Float64(log_tau2[j])
        tau2 = exp(lt)
        lp += logpdf(Gamma(1.0, 2 / lambda2), tau2) + lt
        lp += logpdf(Normal(0, sqrt(tau2)), beta[j])
        inv_tau2 = inv(tau2)
        beta2_inv = Float64(beta[j])^2 * inv_tau2
        grad[j] -= Float64(beta[j]) * inv_tau2
        grad[p + j] += 0.5 - 0.5 * lambda2 * tau2 + 0.5 * beta2_inv
        if prior.lambda === nothing
            grad[2p + 1] += 1.0 - 0.5 * lambda2 * tau2
        end
    end
    return lp
end
