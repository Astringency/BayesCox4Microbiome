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
    p = length(loglambda)
    size(weights, 1) == p && size(weights, 2) == p ||
        throw(DimensionMismatch("car_weights must be a p by p matrix."))
    _check_positive(precision, "car_precision")
    lp = zero(first(loglambda))
    @inbounds for j in 2:p
        for i in 1:(j - 1)
            w = weights[i, j]
            if w != 0
                d = loglambda[i] - loglambda[j]
                lp -= 0.5 * precision * w * d^2
            end
        end
    end
    return lp
end

function _soft_center_logpdf(beta::AbstractVector, scale::Real)
    scale > 0 || return 0.0
    return logpdf(Normal(0, scale), sum(beta))
end

_default_soft_center_scale(p::Integer) = 0.001 * p
