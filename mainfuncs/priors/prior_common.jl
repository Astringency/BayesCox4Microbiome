function _check_positive(x::Real, name::AbstractString)
    x > 0 || throw(ArgumentError("$name must be positive."))
    return Float64(x)
end

function _check_nonnegative(x::Real, name::AbstractString)
    x >= 0 || throw(ArgumentError("$name must be nonnegative."))
    return Float64(x)
end

function _validate_prior(prior::AbstractCoxPrior)
    throw(ArgumentError("unsupported prior type: $(typeof(prior))"))
end

_is_proximal_l1_prior(::AbstractCoxPrior) = false
_is_proximal_l1_prior(prior::BayesianLassoPrior) = prior.proximal
_is_proximal_l1_prior(prior::SpikeSlabLassoPrior) = prior.proximal

function cox_tau0(p::Integer, p0::Real, n_events::Integer)
    p > 1 || throw(ArgumentError("p must be larger than one."))
    p0f = Float64(p0)
    0 < p0f < p || throw(ArgumentError("p0 must be in (0, p)."))
    n_events > 0 || throw(ArgumentError("n_events must be positive."))
    return (p0f / (p - p0f)) / sqrt(float(n_events))
end

function linear_tau0(p::Integer, p0::Real, n::Integer, pseudo_sigma::Real)
    p > 1 || throw(ArgumentError("p must be larger than one."))
    p0f = Float64(p0)
    0 < p0f < p || throw(ArgumentError("p0 must be in (0, p)."))
    n > 0 || throw(ArgumentError("n must be positive."))
    pseudo_sigma > 0 || throw(ArgumentError("pseudo_sigma must be positive."))
    return (p0f / (p - p0f)) * pseudo_sigma / sqrt(float(n))
end

function _global_tau_cauchy_scale(tau0::Real)
    tau0 > 0 || throw(ArgumentError("tau0 must be positive."))
    return tau0
end

function _positive_t_distribution(df::Real, scale::Real)
    _check_positive(df, "pseudo_sigma_df")
    sc = _check_positive(scale, "pseudo_sigma_scale")
    return truncated(LocationScale(0.0, sc, TDist(df)), 0.0, Inf)
end

function _log_positive_t_pdf(x::Real, df::Real, scale::Real)
    x > 0 || return -Inf
    return logpdf(_positive_t_distribution(df, scale), x)
end

function _car_distance_metric(distance)
    method = distance isa Symbol ? distance : Symbol(lowercase(String(distance)))
    method in (:bray, :jaccard, :jsd) ||
        throw(ArgumentError("car distance must be :bray, :jaccard, or :jsd."))
    return method == :bray ? BrayCurtis() : method == :jaccard ? Jaccard() : JSDivergence()
end

function _validate_car_input(X::AbstractMatrix{<:Real})
    x = Matrix{Float64}(X)
    n, p = size(x)
    n > 0 || throw(ArgumentError("X must contain at least one row."))
    p > 1 || throw(ArgumentError("CAR weights require at least two taxa columns."))
    all(isfinite, x) || throw(ArgumentError("X must be finite."))
    all(>=(0), x) || throw(ArgumentError("CAR distance input must be nonnegative."))
    return x
end

"""
    car_distance_matrix(X; distance=:bray)

Compute pairwise taxa distances from an `n` by `p` abundance matrix `X`.
Columns are treated as taxa and rows as samples. Supported distances are
`:bray`, `:jaccard`, and `:jsd`.
"""
function car_distance_matrix(X::AbstractMatrix{<:Real}; distance = :bray)
    x = _validate_car_input(X)
    metric = _car_distance_metric(distance)
    return Matrix{Float64}(pairwise(metric, x; dims = 2))
end

"""
    car_similarity_matrix(D)

Convert a square distance matrix to the centered similarity matrix used by
the IAR/CAR construction: `-0.5 * J * D.^2 * J`.
"""
function car_similarity_matrix(D::AbstractMatrix{<:Real})
    size(D, 1) == size(D, 2) || throw(DimensionMismatch("D must be a square matrix."))
    p = size(D, 1)
    p > 1 || throw(ArgumentError("D must describe at least two taxa."))
    d = Matrix{Float64}(D)
    all(isfinite, d) || throw(ArgumentError("D must be finite."))
    all(>=(0), d) || throw(ArgumentError("D must be nonnegative."))
    J = I - fill(1 / p, p, p)
    return -0.5 .* (J * (d .^ 2) * J)
end

"""
    car_weight_matrix(X; distance=:bray, threshold_quantile=0.75)

Build the symmetric CAR weight matrix from an abundance matrix by following
the BayesCox IAR construction: compute taxa distances, double-center squared
distances into a similarity matrix, then keep edges whose `abs(similarity)` is
above `threshold_quantile`.
"""
function car_weight_matrix(
    X::AbstractMatrix{<:Real};
    distance = :bray,
    threshold_quantile::Real = 0.75,
)
    0 <= threshold_quantile <= 1 ||
        throw(ArgumentError("threshold_quantile must be in [0, 1]."))
    D = car_distance_matrix(X; distance = distance)
    K = car_similarity_matrix(D)
    cutoff = quantile(abs.(vec(K)), Float64(threshold_quantile))
    p = size(K, 1)
    W = zeros(Float64, p, p)
    @inbounds for j in 2:p
        for i in 1:(j - 1)
            w = abs(K[i, j])
            if w > cutoff
                W[i, j] = w
                W[j, i] = w
            end
        end
    end
    return W
end

function _car_weight_edges(weights::AbstractMatrix{<:Real})
    size(weights, 1) == size(weights, 2) ||
        throw(DimensionMismatch("car_weights must be a square matrix."))
    p = size(weights, 1)
    ii = Int[]
    jj = Int[]
    ww = Float64[]
    @inbounds for j in 2:p
        for i in 1:(j - 1)
            w = Float64(weights[i, j])
            if w != 0
                push!(ii, i)
                push!(jj, j)
                push!(ww, w)
            end
        end
    end
    return CARWeightEdges(ii, jj, ww, p)
end

function _prepare_car_weights(weights, p::Integer)
    weights === nothing && return nothing
    if weights isa CARWeightEdges
        weights.p == p || throw(DimensionMismatch("car_weights must have length p."))
        return weights
    elseif weights isa AbstractMatrix
        size(weights, 1) == p && size(weights, 2) == p ||
            throw(DimensionMismatch("car_weights must be a p by p matrix."))
        return _car_weight_edges(weights)
    else
        throw(ArgumentError("car_weights must be a matrix or CARWeightEdges."))
    end
end

function _halfcauchy_logpdf(x, scale)
    x > 0 || return -Inf
    return log(2) - log(pi) - log(scale) - log1p((x / scale)^2)
end

function _halfcauchy_logx_gradient(log_x::Real, scale::Real)
    scale > 0 || return NaN
    return 1.0 - 2.0 * _logistic(2.0 * (Float64(log_x) - log(Float64(scale))))
end

function _halfcauchy_logscale_gradient(log_x::Real, scale::Real)
    scale > 0 || return NaN
    return -1.0 + 2.0 * _logistic(2.0 * (Float64(log_x) - log(Float64(scale))))
end

function _laplace_logpdf(x, scale)
    scale > 0 || return -Inf
    return -log(2 * scale) - abs(x) / scale
end

function _logsumexp2(a, b)
    m = max(a, b)
    return m + log(exp(a - m) + exp(b - m))
end

const _LOG2PI = log(2pi)

function _std_normal_logpdf(x)
    return -0.5 * _LOG2PI - 0.5 * x^2
end

function _normal_zero_logpdf_from_logscale(x, log_scale)
    z = x * exp(-log_scale)
    return -0.5 * _LOG2PI - log_scale - 0.5 * z^2
end

function _regularized_horseshoe_beta_logscale(tau, lambda, c2)
    log_raw = log(tau) + log(lambda)
    log_slab = 0.5 * log(c2)
    lo = min(log_raw, log_slab)
    hi = max(log_raw, log_slab)
    return lo - 0.5 * log1p(exp(2 * (lo - hi)))
end

function _logistic(x)
    return x >= 0 ? inv(1 + exp(-x)) : begin
        y = exp(x)
        y / (1 + y)
    end
end

function _clamped_exp_with_derivative(x::Real, lo::Real, hi::Real)
    xc = clamp(Float64(x), Float64(lo), Float64(hi))
    y = exp(xc)
    dy = (Float64(lo) < Float64(x) < Float64(hi)) ? y : 0.0
    return y, dy
end

function _positive_t_logpdf_derivative(x::Real, df::Real, scale::Real)
    x > 0 || return NaN
    ν = Float64(df)
    s = Float64(scale)
    return -(ν + 1.0) * Float64(x) / (ν * s^2 + Float64(x)^2)
end

function _car_logprior(loglambda::AbstractVector, weights, precision::Real)
    weights === nothing && return 0.0
    return _car_logprior(loglambda, _prepare_car_weights(weights, length(loglambda)), precision)
end

function _car_logprior(loglambda::AbstractVector, weights::CARWeightEdges, precision::Real)
    p = length(loglambda)
    weights.p == p || throw(DimensionMismatch("car_weights must have length p."))
    _check_positive(precision, "car_precision")
    lp = zero(first(loglambda))
    scale = 0.5 * precision
    @inbounds for k in eachindex(weights.w)
        d = loglambda[weights.i[k]] - loglambda[weights.j[k]]
        lp -= scale * weights.w[k] * d^2
    end
    return lp
end

function _car_logprior_gradient!(
    grad::AbstractVector{Float64},
    loglambda::AbstractVector,
    weights,
    precision::Real,
)
    weights === nothing && return grad
    return _car_logprior_gradient!(
        grad,
        loglambda,
        _prepare_car_weights(weights, length(loglambda)),
        precision,
    )
end

function _car_logprior_gradient!(
    grad::AbstractVector{Float64},
    loglambda::AbstractVector,
    weights::CARWeightEdges,
    precision::Real,
)
    p = length(loglambda)
    length(grad) == p || throw(DimensionMismatch("CAR gradient length must match loglambda."))
    weights.p == p || throw(DimensionMismatch("car_weights must have length p."))
    prec = _check_positive(precision, "car_precision")
    @inbounds for k in eachindex(weights.w)
        i = weights.i[k]
        j = weights.j[k]
        d = Float64(loglambda[i]) - Float64(loglambda[j])
        v = prec * weights.w[k] * d
        grad[i] -= v
        grad[j] += v
    end
    return grad
end

function _soft_center_logpdf(beta::AbstractVector, scale::Real)
    scale > 0 || return 0.0
    return logpdf(Normal(0, scale), sum(beta))
end

_default_soft_center_scale(p::Integer) = 0.001 * p

function _resolve_constraint(constraint::Symbol)
    constraint in (:soft_center, :prox, :none) ||
        throw(ArgumentError("constraint must be :soft_center, :prox, or :none."))
    return constraint
end

function _constraint_logpenalty(
    beta::AbstractVector;
    constraint::Symbol,
    prox_sum0_lambda::Real,
)
    if constraint == :soft_center
        return _soft_center_logpdf(beta, _default_soft_center_scale(length(beta)))
    elseif constraint == :prox
        lambda_sum0 = _check_positive(prox_sum0_lambda, "prox_sum0_lambda")
        s = sum(beta)
        return -0.5 * s^2 / (lambda_sum0 * length(beta))
    elseif constraint == :none
        return 0.0
    end
    throw(ArgumentError("constraint must be :soft_center, :prox, or :none."))
end

function _constraint_gradient_penalty!(
    grad::AbstractVector,
    beta::AbstractVector;
    constraint::Symbol,
    prox_sum0_lambda::Real,
)
    if constraint == :soft_center
        scale = _default_soft_center_scale(length(beta))
        grad .-= sum(beta) / scale^2
    elseif constraint == :prox
        lambda_sum0 = _check_positive(prox_sum0_lambda, "prox_sum0_lambda")
        grad .-= sum(beta) / (lambda_sum0 * length(beta))
    elseif constraint != :none
        throw(ArgumentError("constraint must be :soft_center, :prox, or :none."))
    end
    return grad
end

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
