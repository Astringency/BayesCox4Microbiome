function _linear_predictor(z::AbstractMatrix, beta::AbstractVector)
    n, p = size(z)
    length(beta) == p || throw(DimensionMismatch("length(beta) must match size(z, 2)."))
    first_value = z[1, 1] * beta[1]
    eta = Vector{typeof(first_value)}(undef, n)
    for i in 1:n
        s = zero(first_value)
        @inbounds for j in 1:p
            s += z[i, j] * beta[j]
        end
        eta[i] = s
    end
    return eta
end

function _logaddexp(a, b)
    if a == -Inf
        return b
    elseif b == -Inf
        return a
    else
        m = max(a, b)
        return m + log(exp(a - m) + exp(b - m))
    end
end

function _logdiffexp(a, b)
    b == -Inf && return a
    b < a || throw(ArgumentError("log difference requires the second argument to be smaller."))
    return a + log1p(-exp(b - a))
end

function _risk_logdenominators(eta::AbstractVector, risk::CoxRiskSet)
    g = length(risk.group_start)
    den = Vector{typeof(first(eta))}(undef, g)
    acc = -Inf
    for k in 1:g
        @inbounds for pos in risk.group_start[k]:risk.group_stop[k]
            acc = _logaddexp(acc, eta[risk.order[pos]])
        end
        den[k] = acc
    end
    return den
end

function cox_logpartiallikelihood(
    z::AbstractMatrix,
    time::AbstractVector{<:Real},
    event::AbstractVector,
    beta::AbstractVector,
    ;
    tie_method::Symbol = :efron,
)
    risk = _build_riskset(time)
    return cox_logpartiallikelihood(z, [Int(v) for v in event], risk, beta; tie_method = tie_method)
end

function cox_logpartiallikelihood(data::CoxPreparedData, beta::AbstractVector)
    return cox_logpartiallikelihood(data.z, data.event, data.risk, beta; tie_method = data.tie_method)
end

function cox_logpartiallikelihood(
    z::AbstractMatrix,
    event::AbstractVector{<:Integer},
    risk::CoxRiskSet,
    beta::AbstractVector,
    ;
    tie_method::Symbol = :efron,
)
    tie = _check_cox_tie_method(tie_method)
    eta = _linear_predictor(z, beta)
    lp = zero(first(eta))
    if tie == :breslow
        den = _risk_logdenominators(eta, risk)
        @inbounds for i in eachindex(event)
            if event[i] != 0
                lp += eta[i] - den[risk.obs_to_group[i]]
            end
        end
    else
        den = _risk_logdenominators(eta, risk)
        @inbounds for k in eachindex(risk.group_start)
            logevent = -Inf
            d = 0
            event_eta_sum = zero(first(eta))
            for pos in risk.group_start[k]:risk.group_stop[k]
                i = risk.order[pos]
                if event[i] != 0
                    d += 1
                    event_eta_sum += eta[i]
                    logevent = _logaddexp(logevent, eta[i])
                end
            end
            if d > 0
                lp += event_eta_sum
                if d == 1
                    lp -= den[k]
                else
                    for r in 0:(d - 1)
                        lp -= _logdiffexp(den[k], log(r / d) + logevent)
                    end
                end
            end
        end
    end
    return lp
end

function cox_loglik_events(data::CoxPreparedData, beta::AbstractVector)
    eta = _linear_predictor(data.z, beta)
    out = Vector{typeof(first(eta))}(undef, length(data.event))
    fill!(out, zero(first(eta)))
    if data.tie_method == :breslow
        den = _risk_logdenominators(eta, data.risk)
        @inbounds for i in eachindex(data.event)
            out[i] = data.event[i] == 0 ? zero(first(eta)) :
                eta[i] - den[data.risk.obs_to_group[i]]
        end
    else
        den = _risk_logdenominators(eta, data.risk)
        @inbounds for k in eachindex(data.risk.group_start)
            logevent = -Inf
            d = 0
            for pos in data.risk.group_start[k]:data.risk.group_stop[k]
                i = data.risk.order[pos]
                if data.event[i] != 0
                    d += 1
                    logevent = _logaddexp(logevent, eta[i])
                end
            end
            if d == 1
                for pos in data.risk.group_start[k]:data.risk.group_stop[k]
                    i = data.risk.order[pos]
                    data.event[i] != 0 && (out[i] = eta[i] - den[k])
                end
            elseif d > 1
                r = 0
                for pos in data.risk.group_start[k]:data.risk.group_stop[k]
                    i = data.risk.order[pos]
                    if data.event[i] != 0
                        out[i] = eta[i] - _logdiffexp(den[k], log(r / d) + logevent)
                        r += 1
                    end
                end
            end
        end
    end
    return out
end

mutable struct CoxGradientWorkspace
    eta::Vector{Float64}
    s1::Vector{Float64}
    group_mean::Matrix{Float64}
end

function CoxGradientWorkspace(n::Integer, p::Integer, n_groups::Integer)
    n >= 0 || throw(ArgumentError("n must be nonnegative."))
    p > 0 || throw(ArgumentError("p must be positive."))
    n_groups >= 0 || throw(ArgumentError("n_groups must be nonnegative."))
    return CoxGradientWorkspace(
        Vector{Float64}(undef, n),
        zeros(Float64, p),
        Matrix{Float64}(undef, n_groups, p),
    )
end

CoxGradientWorkspace(data::CoxPreparedData) =
    CoxGradientWorkspace(size(data.z, 1), size(data.z, 2), length(data.risk.group_start))

function cox_score_gradient(
    z::AbstractMatrix,
    time::AbstractVector{<:Real},
    event::AbstractVector,
    beta::AbstractVector,
    ;
    tie_method::Symbol = :efron,
)
    risk = _build_riskset(time)
    event_int = [Int(v) for v in event]
    work = CoxGradientWorkspace(size(z, 1), size(z, 2), length(risk.group_start))
    grad = zeros(Float64, size(z, 2))
    return cox_score_gradient!(grad, z, event_int, risk, beta, work; tie_method = tie_method)
end

function cox_score_gradient(data::CoxPreparedData, beta::AbstractVector)
    work = CoxGradientWorkspace(data)
    grad = zeros(Float64, size(data.z, 2))
    return cox_score_gradient!(grad, data, beta, work)
end

function cox_score_gradient(
    z::AbstractMatrix,
    event::AbstractVector{<:Integer},
    risk::CoxRiskSet,
    beta::AbstractVector,
    ;
    tie_method::Symbol = :efron,
)
    work = CoxGradientWorkspace(size(z, 1), size(z, 2), length(risk.group_start))
    grad = zeros(Float64, size(z, 2))
    return cox_score_gradient!(grad, z, event, risk, beta, work; tie_method = tie_method)
end

function cox_score_gradient!(
    grad::AbstractVector{Float64},
    data::CoxPreparedData,
    beta::AbstractVector,
    work::CoxGradientWorkspace,
)
    return cox_score_gradient!(grad, data.z, data.event, data.risk, beta, work; tie_method = data.tie_method)
end

function cox_score_gradient!(
    grad::AbstractVector{Float64},
    z::AbstractMatrix,
    event::AbstractVector{<:Integer},
    risk::CoxRiskSet,
    beta::AbstractVector,
    work::CoxGradientWorkspace,
    ;
    tie_method::Symbol = :efron,
)
    cox_logpartiallikelihood_and_score_gradient!(grad, z, event, risk, beta, work; tie_method = tie_method)
    return grad
end

function cox_logpartiallikelihood_and_score_gradient!(
    grad::AbstractVector{Float64},
    data::CoxPreparedData,
    beta::AbstractVector,
    work::CoxGradientWorkspace,
)
    return cox_logpartiallikelihood_and_score_gradient!(
        grad,
        data.z,
        data.event,
        data.risk,
        beta,
        work;
        tie_method = data.tie_method,
    )
end

function cox_logpartiallikelihood_and_score_gradient!(
    grad::AbstractVector{Float64},
    z::AbstractMatrix,
    event::AbstractVector{<:Integer},
    risk::CoxRiskSet,
    beta::AbstractVector,
    work::CoxGradientWorkspace,
    ;
    tie_method::Symbol = :efron,
)
    tie = _check_cox_tie_method(tie_method)
    n, p = size(z)
    length(beta) == p || throw(DimensionMismatch("length(beta) must match size(z, 2)."))
    length(event) == n || throw(DimensionMismatch("length(event) must match size(z, 1)."))
    length(grad) == p || throw(DimensionMismatch("length(grad) must match size(z, 2)."))
    length(work.eta) == n || throw(DimensionMismatch("workspace eta length must match size(z, 1)."))
    length(work.s1) == p || throw(DimensionMismatch("workspace s1 length must match size(z, 2)."))
    size(work.group_mean, 1) == length(risk.group_start) ||
        throw(DimensionMismatch("workspace group_mean rows must match number of risk groups."))
    size(work.group_mean, 2) == p ||
        throw(DimensionMismatch("workspace group_mean columns must match size(z, 2)."))

    eta = work.eta
    fill!(grad, 0.0)
    fill!(work.s1, 0.0)
    lp = 0.0
    @inbounds for i in 1:n
        s = 0.0
        for j in 1:p
            s += Float64(z[i, j]) * Float64(beta[j])
        end
        eta[i] = s
    end
    if !all(isfinite, eta)
        fill!(grad, NaN)
        return -Inf
    end

    s0 = 0.0
    max_eta = -Inf
    @inbounds for k in eachindex(risk.group_start)
        for pos in risk.group_start[k]:risk.group_stop[k]
            i = risk.order[pos]
            if eta[i] > max_eta
                scale = isfinite(max_eta) ? exp(max_eta - eta[i]) : 0.0
                s0 *= scale
                for j in 1:p
                    work.s1[j] *= scale
                end
                max_eta = eta[i]
            end
            w = exp(eta[i] - max_eta)
            s0 += w
            for j in 1:p
                work.s1[j] += w * Float64(z[i, j])
            end
        end
        if !(s0 > 0) || !isfinite(s0)
            fill!(grad, NaN)
            return -Inf
        end
        if tie == :breslow
            logden = max_eta + log(s0)
            for j in 1:p
                work.group_mean[k, j] = work.s1[j] / s0
            end
            for pos in risk.group_start[k]:risk.group_stop[k]
                i = risk.order[pos]
                if event[i] != 0
                    lp += eta[i] - logden
                    for j in 1:p
                        grad[j] += Float64(z[i, j]) - work.group_mean[k, j]
                    end
                end
            end
        else
            d = 0
            e0 = 0.0
            fill!(view(work.group_mean, k, :), 0.0)
            for pos in risk.group_start[k]:risk.group_stop[k]
                i = risk.order[pos]
                if event[i] != 0
                    d += 1
                    lp += eta[i]
                    w = exp(eta[i] - max_eta)
                    e0 += w
                    for j in 1:p
                        grad[j] += Float64(z[i, j])
                        work.group_mean[k, j] += w * Float64(z[i, j])
                    end
                end
            end
            if d > 0
                if d == 1
                    lp -= max_eta + log(s0)
                    for j in 1:p
                        grad[j] -= work.s1[j] / s0
                    end
                else
                    for r in 0:(d - 1)
                        frac = r / d
                        denom = s0 - frac * e0
                        if !(denom > 0) || !isfinite(denom)
                            fill!(grad, NaN)
                            return -Inf
                        end
                        lp -= max_eta + log(denom)
                        for j in 1:p
                            grad[j] -= (work.s1[j] - frac * work.group_mean[k, j]) / denom
                        end
                    end
                end
            end
        end
    end
    return lp
end

# ============================================================================
# Taylor approximation of Cox PH partial likelihood
# ============================================================================

"""
    CoxApproximation

Holds the precomputed expansion point for a second-order Taylor
approximation of the Cox PH log-partial likelihood.

The Taylor expansion at `beta0` is:
    l(β) ≈ l(β₀) + S(β₀)'(β - β₀) - ½(β - β₀)' I(β₀)(β - β₀)

where S is the score (gradient) and I is the observed information
(I = -d²l/dβ², positive semidefinite at the MLE).

# Fields
- `beta0::Vector{Float64}`: expansion point
- `loglik0::Float64`: log-partial likelihood at beta0, l(β₀)
- `grad0::Vector{Float64}`: score (gradient) at beta0, S(β₀)
- `obs_info::Matrix{Float64}`: observed information at beta0, I(β₀) = -d²l/dβ²
- `diff::Vector{Float64}`: workspace for β - β₀
- `point::Symbol`: expansion point type — `:zero`, `:PMLE`, or `:previous`
"""
mutable struct CoxApproximation
    beta0::Vector{Float64}
    loglik0::Float64
    grad0::Vector{Float64}
    obs_info::Matrix{Float64}
    diff::Vector{Float64}
    point::Symbol
end

"""
    CoxApproximation(data::CoxPreparedData, point::Symbol = :zero)

Create an uninitialized `CoxApproximation` workspace sized for `data`.
Call `build_cox_approximation!` to populate it.
"""
function CoxApproximation(data::CoxPreparedData, point::Symbol = :zero)
    p = size(data.z, 2)
    return CoxApproximation(
        Vector{Float64}(undef, p),
        NaN,
        Vector{Float64}(undef, p),
        Matrix{Float64}(undef, p, p),
        Vector{Float64}(undef, p),
        point,
    )
end

"""
    build_cox_approximation!(approx::CoxApproximation, data::CoxPreparedData,
                              beta0::AbstractVector, point::Symbol)

Compute and store the log-likelihood, gradient, and observed information at `beta0`,
populating all fields of `approx`.
"""
function build_cox_approximation!(
    approx::CoxApproximation,
    data::CoxPreparedData,
    beta0::AbstractVector,
    point::Symbol = approx.point,
)
    p = size(data.z, 2)
    length(beta0) == p || throw(DimensionMismatch("beta0 length must match p."))
    copyto!(approx.beta0, beta0)
    approx.loglik0 = cox_logpartiallikelihood(data, beta0)
    cox_score_gradient!(approx.grad0, data, beta0, CoxGradientWorkspace(data))
    cox_observed_information_beta!(approx.obs_info, data, beta0)
    approx.point = point
    return approx
end

"""
    cox_logpartiallikelihood_approx(beta::AbstractVector, approx::CoxApproximation)

Evaluate the second-order Taylor expansion of the Cox log-partial likelihood at `beta`,
using the precomputed expansion point in `approx`.

Formula: l(β) ≈ l(β₀) + S(β₀)'(β - β₀) - ½(β - β₀)' I(β₀)(β - β₀)
"""
function cox_logpartiallikelihood_approx(
    beta::AbstractVector,
    approx::CoxApproximation,
)
    p = length(beta)
    T = promote_type(eltype(beta), Float64)
    lin = zero(T)
    quad = zero(T)
    @inbounds for j in 1:p
        dj = T(beta[j]) - approx.beta0[j]
        lin += approx.grad0[j] * dj
        s = zero(T)
        for i in 1:p
            s += approx.obs_info[j, i] * (T(beta[i]) - approx.beta0[i])
        end
        quad += dj * s
    end
    return T(approx.loglik0) + lin - quad / 2
end

function cox_logpartiallikelihood_approx(
    data::CoxPreparedData,
    beta::AbstractVector,
    approx::CoxApproximation,
)
    return cox_logpartiallikelihood_approx(beta, approx)
end

"""
    cox_approx_score_gradient!(grad::AbstractVector{Float64}, beta::AbstractVector,
                                approx::CoxApproximation)

Compute the gradient of the second-order Taylor expansion at `beta`.
Formula: ∇l(β) ≈ S(β₀) - I(β₀)(β - β₀)
"""
function cox_approx_score_gradient!(
    grad::AbstractVector{Float64},
    beta::AbstractVector,
    approx::CoxApproximation,
)
    p = length(grad)
    length(beta) == p || throw(DimensionMismatch("beta length must match grad."))
    @inbounds for i in 1:p
        s = 0.0
        for j in 1:p
            s += approx.obs_info[i, j] * (Float64(beta[j]) - approx.beta0[j])
        end
        grad[i] = approx.grad0[i] - s
    end
    return grad
end

function cox_approx_score_gradient!(
    grad::AbstractVector{Float64},
    data::CoxPreparedData,
    beta::AbstractVector,
    approx::CoxApproximation,
)
    return cox_approx_score_gradient!(grad, beta, approx)
end

"""
    cox_approx_loglik_and_gradient!(grad::AbstractVector{Float64},
                                     beta::AbstractVector, approx::CoxApproximation)

Compute both the approximate log-likelihood and its gradient at `beta`.
"""
function cox_approx_loglik_and_gradient!(
    grad::AbstractVector{Float64},
    beta::AbstractVector,
    approx::CoxApproximation,
)
    p = length(grad)
    length(beta) == p || throw(DimensionMismatch("beta length must match grad."))
    lin = 0.0
    quad = 0.0
    @inbounds for j in 1:p
        dj = Float64(beta[j]) - approx.beta0[j]
        lin += approx.grad0[j] * dj
        s = 0.0
        for i in 1:p
            s += approx.obs_info[j, i] * (Float64(beta[i]) - approx.beta0[i])
        end
        grad[j] = approx.grad0[j] - s
        quad += dj * s
    end
    return approx.loglik0 + lin - 0.5 * quad
end

function cox_approx_loglik_and_gradient!(
    grad::AbstractVector{Float64},
    data::CoxPreparedData,
    beta::AbstractVector,
    approx::CoxApproximation,
)
    return cox_approx_loglik_and_gradient!(grad, beta, approx)
end
