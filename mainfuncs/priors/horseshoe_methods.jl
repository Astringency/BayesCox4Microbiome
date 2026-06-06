function _validate_prior(prior::HorseshoePrior)
    _check_positive(prior.tau_scale, "tau_scale")
    _check_positive(prior.p0, "p0")
    _check_positive(prior.local_scale, "local_scale")
    return prior
end

function _resolved_tau_scale(prior::HorseshoePrior, p::Integer, n_events::Integer)
    return prior.use_cox_tau0 ? prior.tau_scale * cox_tau0(p, prior.p0, n_events) : prior.tau_scale
end

function _state_names(
    ::HorseshoePrior,
    p::Integer;
    sample_car_strength::Bool = false,
    kwargs...,
)
    names = vcat(["beta[$j]" for j in 1:p], ["log_tau"], ["log_lambda[$j]" for j in 1:p])
    sample_car_strength && push!(names, "logit_car_kappa")
    return names
end

function initial_state(
    prior::HorseshoePrior,
    p::Integer,
    n_events::Integer; rng::AbstractRNG = Random.default_rng(), init::Symbol = :zero,
    beta0 = init == :rand ? _default_unconstrained_init(rng, p) : zeros(Float64, p),
    car_weights = nothing,
    sample_car_strength::Bool = false,
    car_strength_prior_a::Real = 1.0,
    car_strength_prior_b::Real = 1.0,
    car_kappa_init::Real = 0.5,
    kwargs...,
)
    _validate_prior(prior)
    length(beta0) == p || throw(DimensionMismatch("length(beta0) must be p."))
    if sample_car_strength
        _validate_adaptive_car_options(
            car_weights,
            car_strength_prior_a,
            car_strength_prior_b,
            car_kappa_init,
        )
    end
    tau0 = _resolved_tau_scale(prior, p, n_events)
    init_vec = vcat(collect(Float64, beta0), [log(tau0)], init == :rand ? _default_unconstrained_init(rng, p) : zeros(p))
    sample_car_strength && push!(init_vec, _logit_from_probability(car_kappa_init, "car_kappa_init"))
    return init_vec
end

function _prior_logposterior(
    u::AbstractVector,
    data::CoxPreparedData,
    prior::HorseshoePrior;
    car_weights = nothing,
    car_precision::Real = 1.0,
    sample_car_strength::Bool = false,
    car_strength_prior_a::Real = 1.0,
    car_strength_prior_b::Real = 1.0,
    car_degree_scale::Symbol = :mean_active,
    car_kappa_init::Real = 0.5,
)
    p = size(data.z, 2)
    expected = 2p + 1 + (sample_car_strength ? 1 : 0)
    length(u) == expected || throw(DimensionMismatch("wrong state length for HorseshoePrior."))
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
    if sample_car_strength
        prepared_weights = _prepare_car_weights(car_weights, p)
        _validate_adaptive_car_options(
            prepared_weights,
            car_strength_prior_a,
            car_strength_prior_b,
            car_kappa_init,
        )
        lp += _adaptive_car_logprior(
            log_lambda,
            prepared_weights,
            u[end];
            car_strength_prior_a = car_strength_prior_a,
            car_strength_prior_b = car_strength_prior_b,
            car_degree_scale = car_degree_scale,
        )
    else
        lp += _car_logprior(log_lambda, car_weights, car_precision)
    end
    return lp
end

function _prior_logposterior_and_gradient!(
    grad::AbstractVector{Float64},
    u::AbstractVector,
    data::CoxPreparedData,
    prior::HorseshoePrior;
    car_weights = nothing,
    car_precision::Real = 1.0,
    sample_car_strength::Bool = false,
    car_strength_prior_a::Real = 1.0,
    car_strength_prior_b::Real = 1.0,
    car_degree_scale::Symbol = :mean_active,
    car_kappa_init::Real = 0.5,
)
    p = size(data.z, 2)
    expected = 2p + 1 + (sample_car_strength ? 1 : 0)
    length(u) == expected || throw(DimensionMismatch("wrong state length for HorseshoePrior."))
    length(grad) == length(u) || throw(DimensionMismatch("gradient length does not match state."))
    beta = view(u, 1:p)
    log_tau = Float64(u[p + 1])
    log_lambda = view(u, (p + 2):(2p + 1))
    tau = exp(log_tau)
    tau_scale = _resolved_tau_scale(prior, p, count(!=(0), data.event))

    lp = _halfcauchy_logpdf(tau, tau_scale) + log_tau
    grad[p + 1] += _halfcauchy_logx_gradient(log_tau, tau_scale)
    @inbounds for j in 1:p
        logλ = Float64(log_lambda[j])
        λ = exp(logλ)
        lp += _halfcauchy_logpdf(λ, prior.local_scale) + logλ
        inv_scale2 = exp(-2.0 * (log_tau + logλ))
        beta2_inv = Float64(beta[j])^2 * inv_scale2
        lp += logpdf(Normal(0, tau * λ), beta[j])
        grad[j] -= Float64(beta[j]) * inv_scale2
        normal_logscale_grad = -1.0 + beta2_inv
        grad[p + 1] += normal_logscale_grad
        grad[p + 1 + j] += _halfcauchy_logx_gradient(logλ, prior.local_scale) + normal_logscale_grad
    end
    if sample_car_strength
        prepared_weights = _prepare_car_weights(car_weights, p)
        _validate_adaptive_car_options(
            prepared_weights,
            car_strength_prior_a,
            car_strength_prior_b,
            car_kappa_init,
        )
        logit_car_kappa = Float64(u[end])
        _, rho, _ = _adaptive_car_precision(
            logit_car_kappa,
            prepared_weights;
            car_degree_scale = car_degree_scale,
        )
        lp += _adaptive_car_logprior(
            log_lambda,
            prepared_weights,
            logit_car_kappa;
            car_strength_prior_a = car_strength_prior_a,
            car_strength_prior_b = car_strength_prior_b,
            car_degree_scale = car_degree_scale,
        )
        _car_logprior_gradient!(
            view(grad, (p + 2):(2p + 1)),
            log_lambda,
            prepared_weights,
            rho,
        )
        grad[end] += _adaptive_car_logit_gradient(
            log_lambda,
            prepared_weights,
            logit_car_kappa;
            car_strength_prior_a = car_strength_prior_a,
            car_strength_prior_b = car_strength_prior_b,
            car_degree_scale = car_degree_scale,
        )
    else
        lp += _car_logprior(log_lambda, car_weights, car_precision)
        _car_logprior_gradient!(
            view(grad, (p + 2):(2p + 1)),
            log_lambda,
            car_weights,
            car_precision,
        )
    end
    return lp
end
