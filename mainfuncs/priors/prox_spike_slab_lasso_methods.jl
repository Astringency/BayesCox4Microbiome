function _ssl_weights(prior::SpikeSlabLassoPrior, gamma::AbstractVector{Bool})
    weights = Vector{Float64}(undef, length(gamma))
    return _ssl_weights!(weights, prior, gamma)
end

function _ssl_weights!(
    weights::AbstractVector{Float64},
    prior::SpikeSlabLassoPrior,
    gamma::AbstractVector{Bool},
)
    length(weights) == length(gamma) ||
        throw(DimensionMismatch("weights length must match gamma."))
    spike_weight = inv(prior.spike_scale)
    slab_weight = inv(prior.slab_scale)
    @inbounds for j in eachindex(gamma)
        weights[j] = gamma[j] ? slab_weight : spike_weight
    end
    return weights
end

function _ssl_prox_prior_logdensity(
    u::AbstractVector,
    prior::SpikeSlabLassoPrior,
    gamma::AbstractVector{Bool},
    theta::Real;
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_scale::Real,
    alpha_prior_kind = :invgamma,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    p = length(gamma)
    length(u) == p + 1 || throw(DimensionMismatch("u length must be p + 1 for prox SSLASSO."))
    beta = view(u, 1:p)
    eta = u[p + 1]
    alpha = exp(clamp(eta, -700.0, 700.0))
    weights = _ssl_weights(prior, gamma)
    lp = -weighted_l1_moreau(beta, alpha, weights, prox_l1_lambda; prox_root_tol = prox_root_tol, prox_root_maxiter = prox_root_maxiter)
    lp += Float64(p) * eta
    lp += _alpha_prior_logdensity_eta(
        eta;
        alpha_prior_shape = alpha_prior_shape,
        alpha_prior_scale = alpha_prior_scale,
        alpha_prior_kind = alpha_prior_kind,
    ) + sum(log, weights)
    theta_clamped = clamp(theta, eps(Float64), 1 - eps(Float64))
    lp += sum(gamma) * log(theta_clamped) + (length(gamma) - sum(gamma)) * log1p(-theta_clamped)
    return lp
end

function _ssl_prox_prior_gradient(
    u::AbstractVector,
    prior::SpikeSlabLassoPrior,
    gamma::AbstractVector{Bool};
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_scale::Real,
    alpha_prior_kind = :invgamma,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    grad = zeros(Float64, length(u))
    work = _prox_prior_workspace(length(gamma))
    return _ssl_prox_prior_gradient!(
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
end

function _ssl_prox_prior_gradient!(
    grad::AbstractVector{Float64},
    u::AbstractVector,
    prior::SpikeSlabLassoPrior,
    gamma::AbstractVector{Bool},
    work;
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_scale::Real,
    alpha_prior_kind = :invgamma,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    p = length(gamma)
    length(u) == p + 1 || throw(DimensionMismatch("u length must be p + 1 for prox SSLASSO."))
    length(grad) == p + 1 || throw(DimensionMismatch("grad length must be p + 1 for prox SSLASSO."))
    beta = view(u, 1:p)
    eta = u[p + 1]
    alpha, alpha_deriv = _clamped_exp_with_derivative(eta, -700.0, 700.0)
    weights = _ssl_weights!(work.weights, prior, gamma)
    lambda_l1 = _check_positive(prox_l1_lambda, "prox_l1_lambda")
    prox = project_weighted_l1_epigraph!(
        work.prox_beta,
        beta,
        alpha,
        weights;
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    @inbounds for j in 1:p
        grad[j] -= (Float64(beta[j]) - prox.beta[j]) / lambda_l1
    end
    grad[p + 1] += Float64(p) + _alpha_prior_gradient_eta(
        eta;
        alpha_prior_shape = alpha_prior_shape,
        alpha_prior_scale = alpha_prior_scale,
        alpha_prior_kind = alpha_prior_kind,
    ) -
        alpha_deriv * (alpha - prox.alpha) / lambda_l1
    return grad
end

function _ssl_prox_prior_logdensity_and_gradient!(
    grad::AbstractVector{Float64},
    u::AbstractVector,
    prior::SpikeSlabLassoPrior,
    gamma::AbstractVector{Bool},
    theta::Real,
    work;
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_scale::Real,
    alpha_prior_kind = :invgamma,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    p = length(gamma)
    length(u) == p + 1 || throw(DimensionMismatch("u length must be p + 1 for prox SSLASSO."))
    length(grad) == p + 1 || throw(DimensionMismatch("grad length must be p + 1 for prox SSLASSO."))
    beta = view(u, 1:p)
    eta = u[p + 1]
    alpha, alpha_deriv = _clamped_exp_with_derivative(eta, -700.0, 700.0)
    weights = _ssl_weights!(work.weights, prior, gamma)
    lambda_l1 = _check_positive(prox_l1_lambda, "prox_l1_lambda")
    prox = project_weighted_l1_epigraph!(
        work.prox_beta,
        beta,
        alpha,
        weights;
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    dist2 = (Float64(alpha) - Float64(prox.alpha))^2
    @inbounds for j in 1:p
        residual = Float64(beta[j]) - Float64(prox.beta[j])
        dist2 += residual^2
        grad[j] -= residual / lambda_l1
    end
    lp = -0.5 * dist2 / lambda_l1
    lp += Float64(p) * eta
    lp += _alpha_prior_logdensity_eta(
        eta;
        alpha_prior_shape = alpha_prior_shape,
        alpha_prior_scale = alpha_prior_scale,
        alpha_prior_kind = alpha_prior_kind,
    ) + sum(log, weights)
    theta_clamped = clamp(theta, eps(Float64), 1 - eps(Float64))
    lp += sum(gamma) * log(theta_clamped) + (length(gamma) - sum(gamma)) * log1p(-theta_clamped)
    grad[p + 1] += Float64(p) + _alpha_prior_gradient_eta(
        eta;
        alpha_prior_shape = alpha_prior_shape,
        alpha_prior_scale = alpha_prior_scale,
        alpha_prior_kind = alpha_prior_kind,
    ) -
        alpha_deriv * (Float64(alpha) - Float64(prox.alpha)) / lambda_l1
    return lp, grad
end
