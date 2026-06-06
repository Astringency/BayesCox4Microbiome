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

function _logit_from_probability(x::Real, name::AbstractString)
    0 < x < 1 || throw(ArgumentError("$name must be in (0, 1)."))
    xf = Float64(x)
    return log(xf) - log1p(-xf)
end

function _validate_car_strength_prior(a::Real, b::Real)
    return _check_positive(a, "car_strength_prior_a"),
        _check_positive(b, "car_strength_prior_b")
end

function _log1pexp(x::Real)
    xf = Float64(x)
    if xf > 0
        return xf + log1p(exp(-xf))
    else
        return log1p(exp(xf))
    end
end

_log_logistic(x::Real) = -_log1pexp(-Float64(x))

function _beta_logit_logpdf(logit_x, a::Real, b::Real)
    af, bf = _validate_car_strength_prior(a, b)
    eta = Float64(logit_x)
    return af * eta - (af + bf) * _log1pexp(eta) -
        Distributions.SpecialFunctions.logbeta(af, bf)
end

function _beta_logit_gradient(logit_x::Real, a::Real, b::Real)
    af, bf = _validate_car_strength_prior(a, b)
    x = _logistic(Float64(logit_x))
    return af * (1.0 - x) - bf * x
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


"""
    _default_unconstrained_init(rng::AbstractRNG, n::Integer)

Default unconstrained parameter initialization when a specific RNG is provided:
draw `n` independent samples from Uniform(-2.0, 2.0). Used when `init = :rand`.
"""
_default_unconstrained_init(rng::AbstractRNG, n::Integer) = 4.0 .* rand(rng, Float64, n) .- 2.0

"""
    _default_unconstrained_init(n::Integer)

Default unconstrained parameter initialization using the global RNG:
draw `n` independent samples from Uniform(-2.0, 2.0). Used when `init = :rand`.
"""
_default_unconstrained_init(n::Integer) = _default_unconstrained_init(Random.default_rng(), n)
