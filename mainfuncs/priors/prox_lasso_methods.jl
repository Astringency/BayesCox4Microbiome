function _prox_prior_workspace(p::Integer)
    p > 0 || throw(ArgumentError("p must be positive."))
    return (prox_beta = Vector{Float64}(undef, p), weights = Vector{Float64}(undef, p))
end

function _lasso_prox_prior_logdensity(
    u::AbstractVector,
    p::Integer;
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_scale::Real,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    length(u) == p + 1 || throw(DimensionMismatch("u length must be p + 1 for prox LASSO."))
    beta = view(u, 1:p)
    eta = u[p + 1]
    alpha = exp(clamp(eta, -700.0, 700.0))
    lp = -l1_moreau(beta, alpha, prox_l1_lambda; prox_root_tol = prox_root_tol, prox_root_maxiter = prox_root_maxiter)
    lp += -Float64(alpha_prior_scale) * exp(-eta) - Float64(alpha_prior_shape) * eta
    return lp
end

function _lasso_prox_prior_gradient(
    u::AbstractVector,
    p::Integer;
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_scale::Real,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    grad = zeros(Float64, length(u))
    work = _prox_prior_workspace(p)
    return _lasso_prox_prior_gradient!(
        grad,
        u,
        p,
        work;
        prox_l1_lambda = prox_l1_lambda,
        alpha_prior_shape = alpha_prior_shape,
        alpha_prior_scale = alpha_prior_scale,
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
end

function _lasso_prox_prior_gradient!(
    grad::AbstractVector{Float64},
    u::AbstractVector,
    p::Integer,
    work;
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_scale::Real,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    length(u) == p + 1 || throw(DimensionMismatch("u length must be p + 1 for prox LASSO."))
    length(grad) == p + 1 || throw(DimensionMismatch("grad length must be p + 1 for prox LASSO."))
    beta = view(u, 1:p)
    eta = u[p + 1]
    alpha, alpha_deriv = _clamped_exp_with_derivative(eta, -700.0, 700.0)
    lambda_l1 = _check_positive(prox_l1_lambda, "prox_l1_lambda")
    prox = project_l1_epigraph!(
        work.prox_beta,
        beta,
        alpha;
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    @inbounds for j in 1:p
        grad[j] -= (Float64(beta[j]) - prox.beta[j]) / lambda_l1
    end
    grad[p + 1] += Float64(alpha_prior_scale) * exp(-eta) - Float64(alpha_prior_shape) -
        alpha_deriv * (alpha - prox.alpha) / lambda_l1
    return grad
end

function _lasso_prox_prior_logdensity_and_gradient!(
    grad::AbstractVector{Float64},
    u::AbstractVector,
    p::Integer,
    work;
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_scale::Real,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    length(u) == p + 1 || throw(DimensionMismatch("u length must be p + 1 for prox LASSO."))
    length(grad) == p + 1 || throw(DimensionMismatch("grad length must be p + 1 for prox LASSO."))
    beta = view(u, 1:p)
    eta = u[p + 1]
    alpha, alpha_deriv = _clamped_exp_with_derivative(eta, -700.0, 700.0)
    lambda_l1 = _check_positive(prox_l1_lambda, "prox_l1_lambda")
    prox = project_l1_epigraph!(
        work.prox_beta,
        beta,
        alpha;
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
    lp += -Float64(alpha_prior_scale) * exp(-eta) - Float64(alpha_prior_shape) * eta
    grad[p + 1] += Float64(alpha_prior_scale) * exp(-eta) - Float64(alpha_prior_shape) -
        alpha_deriv * (Float64(alpha) - Float64(prox.alpha)) / lambda_l1
    return lp, grad
end
