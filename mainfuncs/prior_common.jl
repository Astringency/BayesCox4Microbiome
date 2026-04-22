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

function _laplace_logpdf(x, scale)
    scale > 0 || return -Inf
    return -log(2 * scale) - abs(x) / scale
end

function _logsumexp2(a, b)
    m = max(a, b)
    return m + log(exp(a - m) + exp(b - m))
end

function _logistic(x)
    return x >= 0 ? inv(1 + exp(-x)) : begin
        y = exp(x)
        y / (1 + y)
    end
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

function _soft_center_logpdf(beta::AbstractVector, scale::Real)
    scale > 0 || return 0.0
    return logpdf(Normal(0, scale), sum(beta))
end

_default_soft_center_scale(p::Integer) = 0.001 * p
