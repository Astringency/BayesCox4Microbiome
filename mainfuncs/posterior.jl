function logposterior(
    u::AbstractVector,
    data::CoxPreparedData,
    prior::AbstractCoxPrior;
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
    beta = view(u, 1:p)
    lp = cox_logpartiallikelihood(data, beta)
    resolved_constraint = _resolve_constraint(constraint)
    lp += _constraint_logpenalty(
        beta;
        constraint = resolved_constraint,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    lp += _prior_logposterior(
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

function logposterior_and_gradient!(
    grad::AbstractVector{Float64},
    u::AbstractVector,
    data::CoxPreparedData,
    prior::AbstractCoxPrior,
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
    _validate_prior(prior)
    p = size(data.z, 2)
    length(u) == length(grad) || throw(DimensionMismatch("gradient length does not match state."))
    length(u) >= p || throw(DimensionMismatch("state is shorter than beta dimension."))
    beta = view(u, 1:p)
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
    lp += _prior_logposterior_and_gradient!(
        grad,
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
    return lp, grad
end

mutable struct ProxGradientWorkspace
    cox::CoxGradientWorkspace
    prox_beta::Vector{Float64}
    weights::Vector{Float64}
end

function ProxGradientWorkspace(data::CoxPreparedData)
    p = size(data.z, 2)
    return ProxGradientWorkspace(CoxGradientWorkspace(data), Vector{Float64}(undef, p), Vector{Float64}(undef, p))
end

function _lasso_prox_logposterior(
    u::AbstractVector,
    data::CoxPreparedData,
    prior::BayesianLassoPrior;
    constraint::Symbol,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_scale::Real,
    alpha_prior_kind = :invgamma,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    p = size(data.z, 2)
    beta = view(u, 1:p)
    lp = cox_logpartiallikelihood(data, beta)
    lp += _constraint_logpenalty(
        beta;
        constraint = constraint,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    lp += _lasso_prox_prior_logdensity(
        u,
        p;
        prox_l1_lambda = prox_l1_lambda,
        alpha_prior_shape = alpha_prior_shape,
        alpha_prior_scale = alpha_prior_scale,
        alpha_prior_kind = alpha_prior_kind,
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    return lp
end

function _lasso_prox_gradient(
    u::AbstractVector,
    data::CoxPreparedData;
    constraint::Symbol,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_scale::Real,
    alpha_prior_kind = :invgamma,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    grad = zeros(Float64, length(u))
    work = ProxGradientWorkspace(data)
    return _lasso_prox_gradient!(
        grad,
        u,
        data,
        work;
        constraint = constraint,
        prox_sum0_lambda = prox_sum0_lambda,
        prox_l1_lambda = prox_l1_lambda,
        alpha_prior_shape = alpha_prior_shape,
        alpha_prior_scale = alpha_prior_scale,
        alpha_prior_kind = alpha_prior_kind,
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
end

function _lasso_prox_gradient!(
    grad::AbstractVector{Float64},
    u::AbstractVector,
    data::CoxPreparedData,
    work::ProxGradientWorkspace;
    constraint::Symbol,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_scale::Real,
    alpha_prior_kind = :invgamma,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    p = size(data.z, 2)
    length(u) == p + 1 || throw(DimensionMismatch("u length must be p + 1 for prox LASSO."))
    length(grad) == p + 1 || throw(DimensionMismatch("grad length must be p + 1 for prox LASSO."))
    beta = view(u, 1:p)
    cox_score_gradient!(view(grad, 1:p), data, beta, work.cox)
    grad[p + 1] = 0.0
    _constraint_gradient_penalty!(
        view(grad, 1:p),
        beta;
        constraint = constraint,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    _lasso_prox_prior_gradient!(
        grad,
        u,
        p,
        work;
        prox_l1_lambda = prox_l1_lambda,
        alpha_prior_shape = alpha_prior_shape,
        alpha_prior_scale = alpha_prior_scale,
        alpha_prior_kind = alpha_prior_kind,
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    return grad
end

function _lasso_prox_logposterior_and_gradient!(
    grad::AbstractVector{Float64},
    u::AbstractVector,
    data::CoxPreparedData,
    work::ProxGradientWorkspace;
    constraint::Symbol,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_scale::Real,
    alpha_prior_kind = :invgamma,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    p = size(data.z, 2)
    length(u) == p + 1 || throw(DimensionMismatch("u length must be p + 1 for prox LASSO."))
    length(grad) == p + 1 || throw(DimensionMismatch("grad length must be p + 1 for prox LASSO."))
    beta = view(u, 1:p)
    lp = cox_logpartiallikelihood_and_score_gradient!(view(grad, 1:p), data, beta, work.cox)
    grad[p + 1] = 0.0
    lp += _constraint_logpenalty(
        beta;
        constraint = constraint,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    _constraint_gradient_penalty!(
        view(grad, 1:p),
        beta;
        constraint = constraint,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    prior_lp, _ = _lasso_prox_prior_logdensity_and_gradient!(
        grad,
        u,
        p,
        work;
        prox_l1_lambda = prox_l1_lambda,
        alpha_prior_shape = alpha_prior_shape,
        alpha_prior_scale = alpha_prior_scale,
        alpha_prior_kind = alpha_prior_kind,
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    return lp + prior_lp, grad
end

function _ssl_prox_logposterior(
    u::AbstractVector,
    data::CoxPreparedData,
    prior::SpikeSlabLassoPrior,
    gamma::AbstractVector{Bool},
    theta::Real;
    constraint::Symbol,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_scale::Real,
    alpha_prior_kind = :invgamma,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    p = size(data.z, 2)
    length(gamma) == p || throw(DimensionMismatch("gamma length must match beta dimension."))
    beta = view(u, 1:p)
    lp = cox_logpartiallikelihood(data, beta)
    lp += _constraint_logpenalty(
        beta;
        constraint = constraint,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    lp += _ssl_prox_prior_logdensity(
        u,
        prior,
        gamma,
        theta;
        prox_l1_lambda = prox_l1_lambda,
        alpha_prior_shape = alpha_prior_shape,
        alpha_prior_scale = alpha_prior_scale,
        alpha_prior_kind = alpha_prior_kind,
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    return lp
end

function _ssl_prox_gradient(
    u::AbstractVector,
    data::CoxPreparedData,
    prior::SpikeSlabLassoPrior,
    gamma::AbstractVector{Bool};
    constraint::Symbol,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_scale::Real,
    alpha_prior_kind = :invgamma,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    grad = zeros(Float64, length(u))
    work = ProxGradientWorkspace(data)
    return _ssl_prox_gradient!(
        grad,
        u,
        data,
        prior,
        gamma,
        work;
        constraint = constraint,
        prox_sum0_lambda = prox_sum0_lambda,
        prox_l1_lambda = prox_l1_lambda,
        alpha_prior_shape = alpha_prior_shape,
        alpha_prior_scale = alpha_prior_scale,
        alpha_prior_kind = alpha_prior_kind,
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
end

function _ssl_prox_gradient!(
    grad::AbstractVector{Float64},
    u::AbstractVector,
    data::CoxPreparedData,
    prior::SpikeSlabLassoPrior,
    gamma::AbstractVector{Bool},
    work::ProxGradientWorkspace;
    constraint::Symbol,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_scale::Real,
    alpha_prior_kind = :invgamma,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    p = size(data.z, 2)
    length(gamma) == p || throw(DimensionMismatch("gamma length must match beta dimension."))
    length(u) == p + 1 || throw(DimensionMismatch("u length must be p + 1 for prox SSLASSO."))
    length(grad) == p + 1 || throw(DimensionMismatch("grad length must be p + 1 for prox SSLASSO."))
    beta = view(u, 1:p)
    cox_score_gradient!(view(grad, 1:p), data, beta, work.cox)
    grad[p + 1] = 0.0
    _constraint_gradient_penalty!(
        view(grad, 1:p),
        beta;
        constraint = constraint,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    _ssl_prox_prior_gradient!(
        grad,
        u,
        prior,
        gamma,
        work;
        prox_l1_lambda = prox_l1_lambda,
        alpha_prior_shape = alpha_prior_shape,
        alpha_prior_scale = alpha_prior_scale,
        alpha_prior_kind = alpha_prior_kind,
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    return grad
end

function _ssl_prox_logposterior_and_gradient!(
    grad::AbstractVector{Float64},
    u::AbstractVector,
    data::CoxPreparedData,
    prior::SpikeSlabLassoPrior,
    gamma::AbstractVector{Bool},
    theta::Real,
    work::ProxGradientWorkspace;
    constraint::Symbol,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_scale::Real,
    alpha_prior_kind = :invgamma,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    p = size(data.z, 2)
    length(gamma) == p || throw(DimensionMismatch("gamma length must match beta dimension."))
    length(u) == p + 1 || throw(DimensionMismatch("u length must be p + 1 for prox SSLASSO."))
    length(grad) == p + 1 || throw(DimensionMismatch("grad length must be p + 1 for prox SSLASSO."))
    beta = view(u, 1:p)
    lp = cox_logpartiallikelihood_and_score_gradient!(view(grad, 1:p), data, beta, work.cox)
    grad[p + 1] = 0.0
    lp += _constraint_logpenalty(
        beta;
        constraint = constraint,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    _constraint_gradient_penalty!(
        view(grad, 1:p),
        beta;
        constraint = constraint,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    prior_lp, _ = _ssl_prox_prior_logdensity_and_gradient!(
        grad,
        u,
        prior,
        gamma,
        theta,
        work;
        prox_l1_lambda = prox_l1_lambda,
        alpha_prior_shape = alpha_prior_shape,
        alpha_prior_scale = alpha_prior_scale,
        alpha_prior_kind = alpha_prior_kind,
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    return lp + prior_lp, grad
end
