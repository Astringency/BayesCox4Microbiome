project_sum0(beta::AbstractVector) = collect(beta) .- mean(beta)

function sum0_moreau(beta::AbstractVector, lambda_sum0::Real)
    lambda = _check_positive(lambda_sum0, "lambda_sum0")
    return 0.5 * sum(beta)^2 / (lambda * length(beta))
end

function sum0_moreau_gradient(beta::AbstractVector, lambda_sum0::Real)
    lambda = _check_positive(lambda_sum0, "lambda_sum0")
    return fill(sum(beta) / (lambda * length(beta)), length(beta))
end

function _soft_threshold(x::Real, threshold::Real)
    ax = abs(x)
    return ax <= threshold ? zero(x) : sign(x) * (ax - threshold)
end

function _l1_projection_root(
    beta::AbstractVector,
    alpha::Real;
    weights::Union{Nothing,AbstractVector} = nothing,
    prox_root_tol::Real = 1e-10,
    prox_root_maxiter::Integer = 100,
)
    alpha >= 0 || throw(ArgumentError("alpha must be nonnegative for l1 epigraph projection."))
    tol = _check_positive(prox_root_tol, "prox_root_tol")
    prox_root_maxiter > 0 || throw(ArgumentError("prox_root_maxiter must be positive."))
    if weights === nothing
        l1 = sum(abs, beta)
        l1 <= alpha && return 0.0
        hi = maximum(abs, beta)
        f = nu -> sum(max(abs(b) - nu, 0.0) for b in beta) - nu - alpha
    else
        length(weights) == length(beta) || throw(DimensionMismatch("weights length must match beta."))
        all(>(0), weights) || throw(ArgumentError("weights must be positive."))
        l1 = sum(weights[j] * abs(beta[j]) for j in eachindex(beta))
        l1 <= alpha && return 0.0
        hi = maximum(abs(beta[j]) / weights[j] for j in eachindex(beta))
        f = nu -> sum(weights[j] * max(abs(beta[j]) - nu * weights[j], 0.0) for j in eachindex(beta)) - nu - alpha
    end
    lo = 0.0
    for _ in 1:prox_root_maxiter
        mid = 0.5 * (lo + hi)
        val = f(mid)
        abs(val) <= tol && return mid
        if val > 0
            lo = mid
        else
            hi = mid
        end
    end
    return 0.5 * (lo + hi)
end

function project_l1_epigraph!(
    prox_beta::AbstractVector{Float64},
    beta::AbstractVector,
    alpha::Real;
    prox_root_tol::Real = 1e-10,
    prox_root_maxiter::Integer = 100,
)
    length(prox_beta) == length(beta) ||
        throw(DimensionMismatch("prox_beta length must match beta."))
    nu = _l1_projection_root(
        beta,
        alpha;
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    if nu == 0.0 && sum(abs, beta) <= alpha
        copyto!(prox_beta, beta)
        return (beta = prox_beta, alpha = Float64(alpha), nu = 0.0)
    end
    @inbounds for j in eachindex(beta)
        prox_beta[j] = _soft_threshold(beta[j], nu)
    end
    return (beta = prox_beta, alpha = Float64(alpha) + nu, nu = nu)
end

function project_l1_epigraph(
    beta::AbstractVector,
    alpha::Real;
    prox_root_tol::Real = 1e-10,
    prox_root_maxiter::Integer = 100,
)
    nu = _l1_projection_root(
        beta,
        alpha;
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    nu == 0.0 && sum(abs, beta) <= alpha && return (beta = collect(beta), alpha = alpha, nu = 0.0)
    prox_beta = [_soft_threshold(b, nu) for b in beta]
    return (beta = prox_beta, alpha = alpha + nu, nu = nu)
end

function project_weighted_l1_epigraph!(
    prox_beta::AbstractVector{Float64},
    beta::AbstractVector,
    alpha::Real,
    weights::AbstractVector;
    prox_root_tol::Real = 1e-10,
    prox_root_maxiter::Integer = 100,
)
    length(prox_beta) == length(beta) ||
        throw(DimensionMismatch("prox_beta length must match beta."))
    nu = _l1_projection_root(
        beta,
        alpha;
        weights = weights,
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    if nu == 0.0 && sum(weights[j] * abs(beta[j]) for j in eachindex(beta)) <= alpha
        copyto!(prox_beta, beta)
        return (beta = prox_beta, alpha = Float64(alpha), nu = 0.0)
    end
    @inbounds for j in eachindex(beta)
        prox_beta[j] = _soft_threshold(beta[j], nu * weights[j])
    end
    return (beta = prox_beta, alpha = Float64(alpha) + nu, nu = nu)
end

function project_weighted_l1_epigraph(
    beta::AbstractVector,
    alpha::Real,
    weights::AbstractVector;
    prox_root_tol::Real = 1e-10,
    prox_root_maxiter::Integer = 100,
)
    nu = _l1_projection_root(
        beta,
        alpha;
        weights = weights,
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    if nu == 0.0 && sum(weights[j] * abs(beta[j]) for j in eachindex(beta)) <= alpha
        return (beta = collect(beta), alpha = alpha, nu = 0.0)
    end
    prox_beta = [_soft_threshold(beta[j], nu * weights[j]) for j in eachindex(beta)]
    return (beta = prox_beta, alpha = alpha + nu, nu = nu)
end

function l1_moreau(
    beta::AbstractVector,
    alpha::Real,
    lambda_l1::Real;
    prox_root_tol::Real = 1e-10,
    prox_root_maxiter::Integer = 100,
)
    lambda = _check_positive(lambda_l1, "lambda_l1")
    prox = project_l1_epigraph(beta, alpha; prox_root_tol = prox_root_tol, prox_root_maxiter = prox_root_maxiter)
    dist2 = sum((beta[j] - prox.beta[j])^2 for j in eachindex(beta)) + (alpha - prox.alpha)^2
    return 0.5 * dist2 / lambda
end

function weighted_l1_moreau(
    beta::AbstractVector,
    alpha::Real,
    weights::AbstractVector,
    lambda_l1::Real;
    prox_root_tol::Real = 1e-10,
    prox_root_maxiter::Integer = 100,
)
    lambda = _check_positive(lambda_l1, "lambda_l1")
    prox = project_weighted_l1_epigraph(
        beta,
        alpha,
        weights;
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    dist2 = sum((beta[j] - prox.beta[j])^2 for j in eachindex(beta)) + (alpha - prox.alpha)^2
    return 0.5 * dist2 / lambda
end
