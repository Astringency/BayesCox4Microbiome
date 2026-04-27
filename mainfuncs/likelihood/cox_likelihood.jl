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
)
    risk = _build_riskset(time)
    return cox_logpartiallikelihood(z, [Int(v) for v in event], risk, beta)
end

function cox_logpartiallikelihood(data::CoxPreparedData, beta::AbstractVector)
    return cox_logpartiallikelihood(data.z, data.event, data.risk, beta)
end

function cox_logpartiallikelihood(
    z::AbstractMatrix,
    event::AbstractVector{<:Integer},
    risk::CoxRiskSet,
    beta::AbstractVector,
)
    eta = _linear_predictor(z, beta)
    den = _risk_logdenominators(eta, risk)
    lp = zero(first(eta))
    @inbounds for i in eachindex(event)
        if event[i] != 0
            lp += eta[i] - den[risk.obs_to_group[i]]
        end
    end
    return lp
end

function cox_loglik_events(data::CoxPreparedData, beta::AbstractVector)
    eta = _linear_predictor(data.z, beta)
    den = _risk_logdenominators(eta, data.risk)
    out = Vector{typeof(first(eta))}(undef, length(data.event))
    @inbounds for i in eachindex(data.event)
        out[i] = data.event[i] == 0 ? zero(first(eta)) :
            eta[i] - den[data.risk.obs_to_group[i]]
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
)
    risk = _build_riskset(time)
    event_int = [Int(v) for v in event]
    work = CoxGradientWorkspace(size(z, 1), size(z, 2), length(risk.group_start))
    grad = zeros(Float64, size(z, 2))
    return cox_score_gradient!(grad, z, event_int, risk, beta, work)
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
)
    work = CoxGradientWorkspace(size(z, 1), size(z, 2), length(risk.group_start))
    grad = zeros(Float64, size(z, 2))
    return cox_score_gradient!(grad, z, event, risk, beta, work)
end

function cox_score_gradient!(
    grad::AbstractVector{Float64},
    data::CoxPreparedData,
    beta::AbstractVector,
    work::CoxGradientWorkspace,
)
    return cox_score_gradient!(grad, data.z, data.event, data.risk, beta, work)
end

function cox_score_gradient!(
    grad::AbstractVector{Float64},
    z::AbstractMatrix,
    event::AbstractVector{<:Integer},
    risk::CoxRiskSet,
    beta::AbstractVector,
    work::CoxGradientWorkspace,
)
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
    max_eta = -Inf
    @inbounds for i in 1:n
        s = 0.0
        for j in 1:p
            s += Float64(z[i, j]) * Float64(beta[j])
        end
        eta[i] = s
        max_eta = max(max_eta, s)
    end

    s0 = 0.0
    @inbounds for k in eachindex(risk.group_start)
        for pos in risk.group_start[k]:risk.group_stop[k]
            i = risk.order[pos]
            w = exp(eta[i] - max_eta)
            s0 += w
            for j in 1:p
                work.s1[j] += w * Float64(z[i, j])
            end
        end
        s0 > 0 || throw(ArgumentError("encountered an empty or degenerate Cox risk set."))
        for j in 1:p
            work.group_mean[k, j] = work.s1[j] / s0
        end
        for pos in risk.group_start[k]:risk.group_stop[k]
            i = risk.order[pos]
            if event[i] != 0
                for j in 1:p
                    grad[j] += Float64(z[i, j]) - work.group_mean[k, j]
                end
            end
        end
    end
    return grad
end
