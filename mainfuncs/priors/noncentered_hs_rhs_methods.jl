_supports_noncentered(::AbstractCoxPrior) = false
_supports_noncentered(::HorseshoePrior) = true
_supports_noncentered(::RegularizedHorseshoePrior) = true

function _check_noncentered_supported(prior::AbstractCoxPrior)
    _supports_noncentered(prior) ||
        throw(ArgumentError("noncentered=true is only supported for Horseshoe and RegularizedHorseshoe priors."))
    return prior
end

_noncentered_state_names(prior::AbstractCoxPrior, p::Integer; kwargs...) =
    _state_names(prior, p; kwargs...)

function _noncentered_state_names(
    ::HorseshoePrior,
    p::Integer;
    sample_car_strength::Bool = false,
    kwargs...,
)
    names = vcat(["beta_raw[$j]" for j in 1:p], ["log_tau"], ["log_lambda[$j]" for j in 1:p])
    sample_car_strength && push!(names, "logit_car_kappa")
    return names
end

function _noncentered_state_names(
    prior::RegularizedHorseshoePrior,
    p::Integer;
    sample_car_strength::Bool = false,
    kwargs...,
)
    names = vcat(
        ["beta_raw[$j]" for j in 1:p],
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

function _noncentered_initial_state(
    prior::HorseshoePrior,
    p::Integer,
    n_events::Integer;
    n_obs::Union{Nothing,Integer} = nothing,
    beta0 = zeros(Float64, p),
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
    log_tau = log(_resolved_tau_scale(prior, p, n_events))
    log_lambda = zeros(Float64, p)
    beta_raw = collect(Float64, beta0) ./ exp(log_tau)
    init = vcat(beta_raw, [log_tau], log_lambda)
    sample_car_strength && push!(init, _logit_from_probability(car_kappa_init, "car_kappa_init"))
    return init
end

function _noncentered_initial_state(
    prior::RegularizedHorseshoePrior,
    p::Integer,
    n_events::Integer;
    n_obs::Union{Nothing,Integer} = nothing,
    beta0 = zeros(Float64, p),
    car_weights = nothing,
    sample_car_strength::Bool = false,
    car_strength_prior_a::Real = 1.0,
    car_strength_prior_b::Real = 1.0,
    car_kappa_init::Real = 0.5,
    kwargs...,
)
    centered = initial_state(
        prior,
        p,
        n_events;
        n_obs = n_obs,
        beta0 = zeros(Float64, p),
        car_weights = car_weights,
        sample_car_strength = sample_car_strength,
        car_strength_prior_a = car_strength_prior_a,
        car_strength_prior_b = car_strength_prior_b,
        car_kappa_init = car_kappa_init,
    )
    log_tau = centered[p + 1]
    log_lambda = view(centered, (p + 2):(2p + 1))
    log_c2 = centered[2p + 2]
    c2 = exp(log_c2)
    beta_raw = Vector{Float64}(undef, p)
    @inbounds for j in 1:p
        log_scale = _regularized_horseshoe_beta_logscale(exp(log_tau), exp(log_lambda[j]), c2)
        beta_raw[j] = Float64(beta0[j]) * exp(-log_scale)
    end
    centered[1:p] .= beta_raw
    return centered
end

function _noncentered_beta!(beta::AbstractVector, u::AbstractVector, ::HorseshoePrior)
    p = length(beta)
    length(u) in (2p + 1, 2p + 2) ||
        throw(DimensionMismatch("wrong state length for noncentered HorseshoePrior."))
    beta_raw = view(u, 1:p)
    log_tau = u[p + 1]
    log_lambda = view(u, (p + 2):(2p + 1))
    @inbounds for j in 1:p
        beta[j] = beta_raw[j] * exp(log_tau + log_lambda[j])
    end
    return beta
end

function _noncentered_beta!(beta::AbstractVector, u::AbstractVector, prior::RegularizedHorseshoePrior)
    p = length(beta)
    expected = prior.sample_pseudo_sigma ? 2p + 3 : 2p + 2
    length(u) in (expected, expected + 1) ||
        throw(DimensionMismatch("wrong state length for noncentered RegularizedHorseshoePrior."))
    beta_raw = view(u, 1:p)
    log_tau = u[p + 1]
    log_lambda = view(u, (p + 2):(2p + 1))
    c2 = exp(u[2p + 2])
    @inbounds for j in 1:p
        log_scale = _regularized_horseshoe_beta_logscale(exp(log_tau), exp(log_lambda[j]), c2)
        beta[j] = beta_raw[j] * exp(log_scale)
    end
    return beta
end

function _centered_state_from_noncentered(
    prior::Union{HorseshoePrior,RegularizedHorseshoePrior},
    data::CoxPreparedData,
    u::AbstractVector,
)
    p = size(data.z, 2)
    centered = collect(Float64, u)
    _noncentered_beta!(view(centered, 1:p), u, prior)
    return centered
end

function _noncentered_prior_logposterior(
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
    length(u) == expected || throw(DimensionMismatch("wrong state length for noncentered HorseshoePrior."))
    beta_raw = view(u, 1:p)
    log_tau = u[p + 1]
    log_lambda = view(u, (p + 2):(2p + 1))
    tau_scale = _resolved_tau_scale(prior, p, count(!=(0), data.event))

    lp = _halfcauchy_logpdf(exp(log_tau), tau_scale) + log_tau
    @inbounds for j in 1:p
        logλ = log_lambda[j]
        lp += _halfcauchy_logpdf(exp(logλ), prior.local_scale) + logλ
        lp += _std_normal_logpdf(beta_raw[j])
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

function _noncentered_prior_logposterior(
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
    length(u) == expected || throw(DimensionMismatch("wrong state length for noncentered RegularizedHorseshoePrior."))
    beta_raw = view(u, 1:p)
    log_tau = u[p + 1]
    log_lambda = view(u, (p + 2):(2p + 1))
    log_c2 = u[2p + 2]
    log_pseudo_sigma2 = prior.sample_pseudo_sigma ? u[2p + 3] : nothing
    pseudo_sigma2 = prior.sample_pseudo_sigma ? exp(clamp(log_pseudo_sigma2, -20.0, 20.0)) : nothing
    pseudo_sigma = prior.sample_pseudo_sigma ? sqrt(pseudo_sigma2) : prior.pseudo_sigma
    tau_cauchy_scale = _global_tau_cauchy_scale(_resolved_tau_scale(prior, data; pseudo_sigma = pseudo_sigma))

    lp = _halfcauchy_logpdf(exp(log_tau), tau_cauchy_scale) + log_tau
    if prior.sample_pseudo_sigma
        lp += _log_positive_t_pdf(
            pseudo_sigma2,
            prior.pseudo_sigma_df,
            prior.pseudo_sigma_scale,
        ) + log_pseudo_sigma2
    end
    @inbounds for j in 1:p
        logλ = log_lambda[j]
        lp += _halfcauchy_logpdf(exp(logλ), prior.local_scale) + logλ
        lp += _std_normal_logpdf(beta_raw[j])
    end
    c2 = exp(log_c2)
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

function noncentered_logposterior(
    u::AbstractVector,
    data::CoxPreparedData,
    prior::Union{HorseshoePrior,RegularizedHorseshoePrior};
    constraint::Symbol = :soft_center,
    prox_sum0_lambda::Real = 1e-3,
    car_weights = nothing,
    use_car::Bool = car_weights !== nothing,
    car_precision::Real = 1.0,
    sample_car_strength::Bool = false,
    car_strength_prior_a::Real = 1.0,
    car_strength_prior_b::Real = 1.0,
    car_degree_scale::Symbol = :mean_active,
    car_kappa_init::Real = 0.5,
)
    _validate_prior(prior)
    p = size(data.z, 2)
    beta = Vector{eltype(u)}(undef, p)
    _noncentered_beta!(beta, u, prior)
    resolved_constraint = _resolve_constraint(constraint)
    lp = cox_logpartiallikelihood(data, beta)
    lp += _constraint_logpenalty(
        beta;
        constraint = resolved_constraint,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    lp += _noncentered_prior_logposterior(
        u,
        data,
        prior;
        car_weights = use_car ? car_weights : nothing,
        car_precision = car_precision,
        sample_car_strength = sample_car_strength,
        car_strength_prior_a = car_strength_prior_a,
        car_strength_prior_b = car_strength_prior_b,
        car_degree_scale = car_degree_scale,
        car_kappa_init = car_kappa_init,
    )
    return lp
end

function _noncentered_logposterior_and_gradient!(
    grad::AbstractVector{Float64},
    u::AbstractVector,
    data::CoxPreparedData,
    prior::HorseshoePrior,
    work::CoxGradientWorkspace;
    constraint::Symbol = :soft_center,
    prox_sum0_lambda::Real = 1e-3,
    car_weights = nothing,
    use_car::Bool = car_weights !== nothing,
    car_precision::Real = 1.0,
    sample_car_strength::Bool = false,
    car_strength_prior_a::Real = 1.0,
    car_strength_prior_b::Real = 1.0,
    car_degree_scale::Symbol = :mean_active,
    car_kappa_init::Real = 0.5,
)
    p = size(data.z, 2)
    expected = 2p + 1 + (sample_car_strength ? 1 : 0)
    length(u) == expected || throw(DimensionMismatch("wrong state length for noncentered HorseshoePrior."))
    length(grad) == length(u) || throw(DimensionMismatch("gradient length does not match state."))
    beta_raw = view(u, 1:p)
    log_tau = Float64(u[p + 1])
    log_lambda = view(u, (p + 2):(2p + 1))
    beta = Vector{Float64}(undef, p)
    _noncentered_beta!(beta, u, prior)

    fill!(grad, 0.0)
    lp = cox_logpartiallikelihood_and_score_gradient!(view(grad, 1:p), data, beta, work)
    resolved_constraint = _resolve_constraint(constraint)
    lp += _constraint_logpenalty(
        beta;
        constraint = resolved_constraint,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    _constraint_gradient_penalty!(
        view(grad, 1:p),
        beta;
        constraint = resolved_constraint,
        prox_sum0_lambda = prox_sum0_lambda,
    )

    tau_scale = _resolved_tau_scale(prior, p, count(!=(0), data.event))
    lp += _halfcauchy_logpdf(exp(log_tau), tau_scale) + log_tau
    grad[p + 1] = _halfcauchy_logx_gradient(log_tau, tau_scale)
    @inbounds for j in 1:p
        gbeta = grad[j]
        z = Float64(beta_raw[j])
        grad[j] = gbeta * exp(log_tau + Float64(log_lambda[j])) - z
        common = gbeta * beta[j]
        grad[p + 1] += common
        logλ = Float64(log_lambda[j])
        lp += _halfcauchy_logpdf(exp(logλ), prior.local_scale) + logλ + _std_normal_logpdf(z)
        grad[p + 1 + j] += _halfcauchy_logx_gradient(logλ, prior.local_scale) + common
    end
    if sample_car_strength
        prepared_weights = _prepare_car_weights(use_car ? car_weights : nothing, p)
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
        lp += _car_logprior(
            log_lambda,
            use_car ? car_weights : nothing,
            car_precision,
        )
        _car_logprior_gradient!(
            view(grad, (p + 2):(2p + 1)),
            log_lambda,
            use_car ? car_weights : nothing,
            car_precision,
        )
    end
    return lp, grad
end

function _noncentered_logposterior_and_gradient!(
    grad::AbstractVector{Float64},
    u::AbstractVector,
    data::CoxPreparedData,
    prior::RegularizedHorseshoePrior,
    work::CoxGradientWorkspace;
    constraint::Symbol = :soft_center,
    prox_sum0_lambda::Real = 1e-3,
    car_weights = nothing,
    use_car::Bool = car_weights !== nothing,
    car_precision::Real = 1.0,
    sample_car_strength::Bool = false,
    car_strength_prior_a::Real = 1.0,
    car_strength_prior_b::Real = 1.0,
    car_degree_scale::Symbol = :mean_active,
    car_kappa_init::Real = 0.5,
)
    p = size(data.z, 2)
    expected = (prior.sample_pseudo_sigma ? 2p + 3 : 2p + 2) + (sample_car_strength ? 1 : 0)
    length(u) == expected || throw(DimensionMismatch("wrong state length for noncentered RegularizedHorseshoePrior."))
    length(grad) == length(u) || throw(DimensionMismatch("gradient length does not match state."))
    beta_raw = view(u, 1:p)
    log_tau = Float64(u[p + 1])
    log_lambda = view(u, (p + 2):(2p + 1))
    log_c2 = Float64(u[2p + 2])
    c2 = exp(log_c2)
    log_pseudo_sigma2 = prior.sample_pseudo_sigma ? Float64(u[2p + 3]) : 0.0
    pseudo_sigma2 = nothing
    pseudo_sigma = prior.pseudo_sigma
    pseudo_sigma2_deriv = 0.0
    if prior.sample_pseudo_sigma
        pseudo_sigma2, pseudo_sigma2_deriv =
            _clamped_exp_with_derivative(log_pseudo_sigma2, -20.0, 20.0)
        pseudo_sigma = sqrt(pseudo_sigma2)
    end
    tau_cauchy_scale = _global_tau_cauchy_scale(_resolved_tau_scale(prior, data; pseudo_sigma = pseudo_sigma))
    beta = Vector{Float64}(undef, p)
    _noncentered_beta!(beta, u, prior)

    fill!(grad, 0.0)
    lp = cox_logpartiallikelihood_and_score_gradient!(view(grad, 1:p), data, beta, work)
    resolved_constraint = _resolve_constraint(constraint)
    lp += _constraint_logpenalty(
        beta;
        constraint = resolved_constraint,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    _constraint_gradient_penalty!(
        view(grad, 1:p),
        beta;
        constraint = resolved_constraint,
        prox_sum0_lambda = prox_sum0_lambda,
    )

    lp += _halfcauchy_logpdf(exp(log_tau), tau_cauchy_scale) + log_tau
    grad[p + 1] = _halfcauchy_logx_gradient(log_tau, tau_cauchy_scale)

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
        gbeta = grad[j]
        z = Float64(beta_raw[j])
        logλ = Float64(log_lambda[j])
        log_scale = _regularized_horseshoe_beta_logscale(exp(log_tau), exp(logλ), c2)
        grad[j] = gbeta * exp(log_scale) - z
        common = gbeta * beta[j]
        a_over_sum = _logistic(2.0 * (log_tau + logλ) - log_c2)
        c_over_sum = 1.0 - a_over_sum

        lp += _halfcauchy_logpdf(exp(logλ), prior.local_scale) + logλ + _std_normal_logpdf(z)
        grad[p + 1] += common * c_over_sum
        grad[p + 1 + j] += _halfcauchy_logx_gradient(logλ, prior.local_scale) +
            common * c_over_sum
        grad[2p + 2] += common * 0.5 * a_over_sum
    end

    slab_a = prior.slab_df / 2
    slab_b = prior.slab_df * prior.slab_s2 / 2
    lp += logpdf(InverseGamma(slab_a, slab_b), c2) + log_c2
    grad[2p + 2] += -slab_a + slab_b / c2

    if sample_car_strength
        prepared_weights = _prepare_car_weights(use_car ? car_weights : nothing, p)
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
        lp += _car_logprior(
            log_lambda,
            use_car ? car_weights : nothing,
            car_precision,
        )
        _car_logprior_gradient!(
            view(grad, (p + 2):(2p + 1)),
            log_lambda,
            use_car ? car_weights : nothing,
            car_precision,
        )
    end
    return lp, grad
end
