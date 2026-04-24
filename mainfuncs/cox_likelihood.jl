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

function cox_score_gradient(
    z::AbstractMatrix,
    time::AbstractVector{<:Real},
    event::AbstractVector,
    beta::AbstractVector,
)
    risk = _build_riskset(time)
    return cox_score_gradient(z, [Int(v) for v in event], risk, beta)
end

function cox_score_gradient(data::CoxPreparedData, beta::AbstractVector)
    return cox_score_gradient(data.z, data.event, data.risk, beta)
end

function cox_score_gradient(
    z::AbstractMatrix,
    event::AbstractVector{<:Integer},
    risk::CoxRiskSet,
    beta::AbstractVector,
)
    n, p = size(z)
    length(beta) == p || throw(DimensionMismatch("length(beta) must match size(z, 2)."))
    length(event) == n || throw(DimensionMismatch("length(event) must match size(z, 1)."))

    eta = _linear_predictor(z, beta)
    group_mean = Matrix{Float64}(undef, length(risk.group_start), p)
    @inbounds for k in eachindex(risk.group_start)
        stop_pos = risk.group_stop[k]
        max_eta = -Inf
        for pos in 1:stop_pos
            max_eta = max(max_eta, Float64(eta[risk.order[pos]]))
        end
        denom = 0.0
        fill!(view(group_mean, k, :), 0.0)
        for pos in 1:stop_pos
            i = risk.order[pos]
            w = exp(Float64(eta[i]) - max_eta)
            denom += w
            for j in 1:p
                group_mean[k, j] += w * Float64(z[i, j])
            end
        end
        denom > 0 || throw(ArgumentError("encountered an empty or degenerate Cox risk set."))
        for j in 1:p
            group_mean[k, j] /= denom
        end
    end

    grad = zeros(Float64, p)
    @inbounds for i in 1:n
        if event[i] != 0
            k = risk.obs_to_group[i]
            for j in 1:p
                grad[j] += Float64(z[i, j]) - group_mean[k, j]
            end
        end
    end
    return grad
end
