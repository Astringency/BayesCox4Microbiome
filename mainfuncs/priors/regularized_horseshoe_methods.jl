function _validate_prior(prior::RegularizedHorseshoePrior)
    _check_positive(prior.tau_scale, "tau_scale")
    _check_positive(prior.p0, "p0")
    _check_positive(prior.local_scale, "local_scale")
    _check_positive(prior.slab_df, "slab_df")
    _check_positive(prior.slab_s2, "slab_s2")
    prior.tau_formula in (:auto, :fixed, :linear, :cox) ||
        throw(ArgumentError("tau_formula must be :auto, :fixed, :linear, or :cox."))
    if prior.pseudo_sigma !== nothing
        _check_positive(prior.pseudo_sigma, "pseudo_sigma")
    end
    _check_positive(prior.pseudo_sigma_df, "pseudo_sigma_df")
    _check_positive(prior.pseudo_sigma_scale, "pseudo_sigma_scale")
    return prior
end

function _resolved_tau_scale(
    prior::RegularizedHorseshoePrior,
    p::Integer,
    n_events::Integer;
    n_obs::Union{Nothing,Integer} = nothing,
)
    formula = _resolved_tau_formula(prior)
    default_sigma = prior.sample_pseudo_sigma ? sqrt(prior.pseudo_sigma_scale) : 1.0
    n_linear = n_obs === nothing ? n_events : n_obs
    return formula == :cox ? prior.tau_scale * cox_tau0(p, prior.p0, n_events) :
           formula == :linear ? prior.tau_scale * linear_tau0(p, prior.p0, n_linear, something(prior.pseudo_sigma, default_sigma)) :
           prior.tau_scale
end

function _resolved_tau_scale(
    prior::RegularizedHorseshoePrior,
    data::CoxPreparedData;
    pseudo_sigma = prior.pseudo_sigma,
)
    p = size(data.z, 2)
    formula = _resolved_tau_formula(prior)
    return formula == :cox ? prior.tau_scale * cox_tau0(p, prior.p0, count(!=(0), data.event)) :
           formula == :linear ? prior.tau_scale * linear_tau0(p, prior.p0, size(data.z, 1), something(pseudo_sigma, 1.0)) :
           prior.tau_scale
end

_resolved_tau_formula(prior::RegularizedHorseshoePrior) =
    prior.tau_formula == :auto ? (prior.use_cox_tau0 ? :cox : :fixed) : prior.tau_formula

function _state_names(
    prior::RegularizedHorseshoePrior,
    p::Integer;
    sample_car_strength::Bool = false,
    kwargs...,
)
    names = vcat(
        ["beta[$j]" for j in 1:p],
        ["log_tau"],
        ["log_lambda[$j]" for j in 1:p],
        ["log_c2"],
    )
    if prior.sample_pseudo_sigma
        push!(names, "log_pseudo_sigma2")
    end
    sample_car_strength && push!(names, "logit_car_kappa")
    return names
end

function initial_state(
    prior::RegularizedHorseshoePrior,
    p::Integer,
    n_events::Integer; rng::AbstractRNG = Random.default_rng(), init::Symbol = :zero,
    n_obs::Union{Nothing,Integer} = nothing,
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
    tau0 = _resolved_tau_scale(prior, p, n_events; n_obs = n_obs)
    init_vec = vcat(collect(Float64, beta0), [log(_global_tau_cauchy_scale(tau0))], init == :rand ? _default_unconstrained_init(rng, p) : zeros(p), [log(prior.slab_s2)])
    if prior.sample_pseudo_sigma
        pseudo_sigma2_init = prior.pseudo_sigma === nothing ? prior.pseudo_sigma_scale : prior.pseudo_sigma^2
        push!(init_vec, log(pseudo_sigma2_init))
    end
    sample_car_strength && push!(init_vec, _logit_from_probability(car_kappa_init, "car_kappa_init"))
    return init_vec
end

function _prior_logposterior(
    u::AbstractVector,
    data::CoxPreparedData,
    prior::RegularizedHorseshoePrior;
    car_weights = nothing,
    car_precision::Real = 1.0,
    sample_car_strength::Bool = false,
    car_strength_prior_a::Real = 1.0,
    car_strength_prior_b::Real = 1.0,
    car_degree_scale::Symbol = :mean_active,
    car_kappa_init::Real = 0.5,
)
    p = size(data.z, 2)
    expected = (prior.sample_pseudo_sigma ? 2p + 3 : 2p + 2) + (sample_car_strength ? 1 : 0)
    length(u) == expected ||
        throw(DimensionMismatch("wrong state length for RegularizedHorseshoePrior."))
    beta = view(u, 1:p)
    log_tau = u[p + 1]
    log_lambda = view(u, (p + 2):(2p + 1))
    log_c2 = u[2p + 2]
    log_pseudo_sigma2 = prior.sample_pseudo_sigma ? u[2p + 3] : nothing
    tau = exp(log_tau)
    lambda = exp.(log_lambda)
    c2 = exp(log_c2)
    pseudo_sigma2 = prior.sample_pseudo_sigma ? exp(clamp(log_pseudo_sigma2, -20, 20)) : nothing
    pseudo_sigma = prior.sample_pseudo_sigma ? sqrt(pseudo_sigma2) : prior.pseudo_sigma
    tau_cauchy_scale = _global_tau_cauchy_scale(_resolved_tau_scale(prior, data; pseudo_sigma = pseudo_sigma))

    lp = _halfcauchy_logpdf(tau, tau_cauchy_scale) + log_tau
    if prior.sample_pseudo_sigma
        lp += _log_positive_t_pdf(
            pseudo_sigma2,
            prior.pseudo_sigma_df,
            prior.pseudo_sigma_scale,
        ) + log_pseudo_sigma2
    end
    for j in 1:p
        lp += _halfcauchy_logpdf(lambda[j], prior.local_scale) + log_lambda[j]
        log_beta_scale = _regularized_horseshoe_beta_logscale(tau, lambda[j], c2)
        lp += _normal_zero_logpdf_from_logscale(beta[j], log_beta_scale)
    end
    lp += logpdf(InverseGamma(prior.slab_df / 2, prior.slab_df * prior.slab_s2 / 2), c2) + log_c2
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
    prior::RegularizedHorseshoePrior;
    car_weights = nothing,
    car_precision::Real = 1.0,
    sample_car_strength::Bool = false,
    car_strength_prior_a::Real = 1.0,
    car_strength_prior_b::Real = 1.0,
    car_degree_scale::Symbol = :mean_active,
    car_kappa_init::Real = 0.5,
)
    p = size(data.z, 2)
    expected = (prior.sample_pseudo_sigma ? 2p + 3 : 2p + 2) + (sample_car_strength ? 1 : 0)
    length(u) == expected ||
        throw(DimensionMismatch("wrong state length for RegularizedHorseshoePrior."))
    length(grad) == length(u) || throw(DimensionMismatch("gradient length does not match state."))
    beta = view(u, 1:p)
    log_tau = Float64(u[p + 1])
    log_lambda = view(u, (p + 2):(2p + 1))
    log_c2 = Float64(u[2p + 2])
    log_pseudo_sigma2 = prior.sample_pseudo_sigma ? Float64(u[2p + 3]) : 0.0
    tau = exp(log_tau)
    c2 = exp(log_c2)
    pseudo_sigma2 = nothing
    pseudo_sigma = prior.pseudo_sigma
    pseudo_sigma2_deriv = 0.0
    if prior.sample_pseudo_sigma
        pseudo_sigma2, pseudo_sigma2_deriv =
            _clamped_exp_with_derivative(log_pseudo_sigma2, -20.0, 20.0)
        pseudo_sigma = sqrt(pseudo_sigma2)
    end
    tau_cauchy_scale = _global_tau_cauchy_scale(_resolved_tau_scale(prior, data; pseudo_sigma = pseudo_sigma))

    lp = _halfcauchy_logpdf(tau, tau_cauchy_scale) + log_tau
    grad[p + 1] += _halfcauchy_logx_gradient(log_tau, tau_cauchy_scale)

    if prior.sample_pseudo_sigma
        lp += _log_positive_t_pdf(
            pseudo_sigma2,
            prior.pseudo_sigma_df,
            prior.pseudo_sigma_scale,
        ) + log_pseudo_sigma2
        grad[2p + 3] += 1.0 + pseudo_sigma2_deriv *
            _positive_t_logpdf_derivative(
                pseudo_sigma2,
                prior.pseudo_sigma_df,
                prior.pseudo_sigma_scale,
            )
        if _resolved_tau_formula(prior) == :linear
            grad[2p + 3] += (0.5 * pseudo_sigma2_deriv / pseudo_sigma2) *
                _halfcauchy_logscale_gradient(log_tau, tau_cauchy_scale)
        end
    end

    @inbounds for j in 1:p
        logλ = Float64(log_lambda[j])
        λ = exp(logλ)
        lp += _halfcauchy_logpdf(λ, prior.local_scale) + logλ
        log_beta_scale = _regularized_horseshoe_beta_logscale(tau, λ, c2)
        inv_scale2 = exp(-2.0 * log_beta_scale)
        beta2_inv = Float64(beta[j])^2 * inv_scale2
        lp += _normal_zero_logpdf_from_logscale(beta[j], log_beta_scale)
        normal_logscale_grad = -1.0 + beta2_inv
        a_over_sum = _logistic(2.0 * (log_tau + logλ) - log_c2)
        c_over_sum = 1.0 - a_over_sum

        grad[j] -= Float64(beta[j]) * inv_scale2
        grad[p + 1] += normal_logscale_grad * c_over_sum
        grad[p + 1 + j] += _halfcauchy_logx_gradient(logλ, prior.local_scale) +
            normal_logscale_grad * c_over_sum
        grad[2p + 2] += normal_logscale_grad * 0.5 * a_over_sum
    end

    slab_a = prior.slab_df / 2
    slab_b = prior.slab_df * prior.slab_s2 / 2
    lp += logpdf(InverseGamma(slab_a, slab_b), c2) + log_c2
    grad[2p + 2] += -slab_a + slab_b / c2

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
