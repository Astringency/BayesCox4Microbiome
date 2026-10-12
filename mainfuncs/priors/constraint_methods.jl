function _soft_center_logpdf(beta::AbstractVector, scale::Real)
    scale > 0 || return 0.0
    return logpdf(Normal(0, scale), sum(beta))
end

_default_soft_center_scale(p::Integer) = 0.001 * p

function _resolve_constraint(constraint::Symbol)
    constraint in (:soft_center, :prox, :none, :exact) ||
        throw(ArgumentError("constraint must be :soft_center, :prox, :none, or :exact."))
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
        return -0.5 * sum(beta)^2 / (lambda_sum0 * length(beta))
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

