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

function _car_logprior(
    loglambda::AbstractVector,
    weights,
    precision::Real,
)
    weights === nothing && return 0.0
    return _car_logprior(
        loglambda,
        _prepare_car_weights(weights, length(loglambda)),
        precision,
    )
end

"""
    car_weight_matrix(X; distance=:bray, threshold_quantile=0.99)

Build the symmetric CAR weight matrix from an abundance matrix by following
the BayesCox IAR construction: compute taxa distances, double-center squared
distances into a similarity matrix, then keep edges whose `abs(similarity)` is
above `threshold_quantile`.
"""
function car_weight_matrix(
    X::AbstractMatrix{<:Real};
    distance = :bray,
    threshold_quantile::Real = 0.99,
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

function _car_weighted_degrees(weights::CARWeightEdges)
    degrees = zeros(Float64, weights.p)
    @inbounds for k in eachindex(weights.w)
        w = weights.w[k]
        degrees[weights.i[k]] += w
        degrees[weights.j[k]] += w
    end
    return degrees
end

function _car_degree_scale(
    weights::Union{Nothing,CARWeightEdges};
    method::Symbol = :mean_active,
)
    weights === nothing && return 1.0
    degrees = _car_weighted_degrees(weights)
    active = degrees[degrees .> 0.0]
    isempty(active) && return 1.0
    if method == :mean_active
        return mean(active)
    elseif method == :median_active
        return median(active)
    elseif method == :q90_active
        return quantile(active, 0.9)
    end
    throw(ArgumentError("car_degree_scale must be :mean_active, :median_active, or :q90_active."))
end

function _car_rank(weights::CARWeightEdges)
    parent = collect(1:weights.p)
    active = falses(weights.p)

    function find_root!(x::Int)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        return x
    end

    positive_edges = 0
    @inbounds for k in eachindex(weights.w)
        weights.w[k] > 0.0 || continue
        positive_edges += 1
        i = weights.i[k]
        j = weights.j[k]
        active[i] = true
        active[j] = true
        ri = find_root!(i)
        rj = find_root!(j)
        ri == rj || (parent[rj] = ri)
    end
    positive_edges == 0 && return 0
    roots = Set{Int}()
    active_count = 0
    for i in 1:weights.p
        active[i] || continue
        active_count += 1
        push!(roots, find_root!(i))
    end
    return active_count - length(roots)
end

function _car_quadratic_stat(loglambda::AbstractVector, weights)
    weights === nothing && return zero(first(loglambda))
    return _car_quadratic_stat(loglambda, _prepare_car_weights(weights, length(loglambda)))
end

function _car_quadratic_stat(loglambda::AbstractVector, weights::CARWeightEdges)
    weights.p == length(loglambda) || throw(DimensionMismatch("car_weights must have length p."))
    s = zero(first(loglambda))
    @inbounds for k in eachindex(weights.w)
        d = loglambda[weights.i[k]] - loglambda[weights.j[k]]
        s += weights.w[k] * d^2
    end
    return s
end

function _validate_adaptive_car_options(
    weights,
    car_strength_prior_a::Real,
    car_strength_prior_b::Real,
    car_kappa_init::Real,
)
    weights === nothing &&
        throw(ArgumentError("sample_car_strength=true requires car_weights or use_car=true with a CAR graph."))
    _validate_car_strength_prior(car_strength_prior_a, car_strength_prior_b)
    _logit_from_probability(car_kappa_init, "car_kappa_init")
    return nothing
end

function _adaptive_car_precision(
    logit_car_kappa,
    weights::CARWeightEdges;
    car_degree_scale::Symbol = :mean_active,
)
    dbar = _car_degree_scale(weights; method = car_degree_scale)
    kappa = _logistic(logit_car_kappa)
    rho = kappa / dbar
    return kappa, rho, dbar
end

function _adaptive_car_logprior(
    loglambda::AbstractVector,
    weights::CARWeightEdges,
    logit_car_kappa;
    car_strength_prior_a::Real = 1.0,
    car_strength_prior_b::Real = 1.0,
    car_degree_scale::Symbol = :mean_active,
)
    _, rho, dbar = _adaptive_car_precision(
        logit_car_kappa,
        weights;
        car_degree_scale = car_degree_scale,
    )
    rank = _car_rank(weights)
    s = _car_quadratic_stat(loglambda, weights)
    lp = -0.5 * rho * s
    if rank > 0
        lp += 0.5 * rank * (_log_logistic(logit_car_kappa) - log(dbar))
    end
    lp += _beta_logit_logpdf(logit_car_kappa, car_strength_prior_a, car_strength_prior_b)
    return lp
end

function _adaptive_car_logit_gradient(
    loglambda::AbstractVector,
    weights::CARWeightEdges,
    logit_car_kappa::Real;
    car_strength_prior_a::Real = 1.0,
    car_strength_prior_b::Real = 1.0,
    car_degree_scale::Symbol = :mean_active,
)
    kappa, _, dbar = _adaptive_car_precision(
        Float64(logit_car_kappa),
        weights;
        car_degree_scale = car_degree_scale,
    )
    rank = _car_rank(weights)
    s = Float64(_car_quadratic_stat(loglambda, weights))
    car_grad = 0.5 * rank * (1.0 - kappa) -
        0.5 * s * kappa * (1.0 - kappa) / dbar
    return car_grad + _beta_logit_gradient(logit_car_kappa, car_strength_prior_a, car_strength_prior_b)
end

function _car_logprior(
    loglambda::AbstractVector,
    weights::CARWeightEdges,
    precision::Real,
)
    p = length(loglambda)
    weights.p == p || throw(DimensionMismatch("car_weights must have length p."))
    precision > 0 || throw(ArgumentError("car_precision must be positive."))
    ρ = precision
    lp = zero(first(loglambda) * ρ)

    scale = 0.5 * ρ
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
    precision == 0 && return grad
    ρ = _check_positive(precision, "car_precision")

    @inbounds for k in eachindex(weights.w)
        i = weights.i[k]
        j = weights.j[k]
        d = Float64(loglambda[i]) - Float64(loglambda[j])
        v = ρ * weights.w[k] * d
        grad[i] -= v
        grad[j] += v
    end

    return grad
end

