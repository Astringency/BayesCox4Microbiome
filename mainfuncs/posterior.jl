function logposterior(
    u::AbstractVector,
    data::CoxPreparedData,
    prior::AbstractCoxPrior;
    constraint::Symbol = :soft_center,
    soft_center = nothing,
    soft_center_scale::Union{Nothing,Real} = nothing,
    prox_sum0_lambda::Real = 1e-3,
    car_weights = nothing,
    use_car::Bool = car_weights !== nothing,
    car_precision::Real = 1.0,
)
    _validate_prior(prior)
    p = size(data.z, 2)
    beta = view(u, 1:p)
    lp = cox_logpartiallikelihood(data, beta)
    resolved_constraint = _resolve_constraint(constraint, soft_center)
    scale = soft_center_scale === nothing ? _default_soft_center_scale(p) : Float64(soft_center_scale)
    lp += _constraint_logpenalty(
        beta;
        constraint = resolved_constraint,
        soft_center_scale = scale,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    lp += _prior_logposterior(
        u,
        data,
        prior;
        car_weights = use_car ? car_weights : nothing,
        car_precision = car_precision,
    )
    return lp
end

function logposterior_and_gradient!(
    grad::AbstractVector{Float64},
    u::AbstractVector,
    data::CoxPreparedData,
    prior::AbstractCoxPrior,
    work::CoxGradientWorkspace;
    constraint::Symbol = :soft_center,
    soft_center = nothing,
    soft_center_scale::Union{Nothing,Real} = nothing,
    prox_sum0_lambda::Real = 1e-3,
    car_weights = nothing,
    use_car::Bool = car_weights !== nothing,
    car_precision::Real = 1.0,
)
    _validate_prior(prior)
    p = size(data.z, 2)
    length(u) == length(grad) || throw(DimensionMismatch("gradient length does not match state."))
    length(u) >= p || throw(DimensionMismatch("state is shorter than beta dimension."))
    beta = view(u, 1:p)
    fill!(grad, 0.0)
    lp = cox_logpartiallikelihood_and_score_gradient!(view(grad, 1:p), data, beta, work)
    resolved_constraint = _resolve_constraint(constraint, soft_center)
    scale = soft_center_scale === nothing ? _default_soft_center_scale(p) : Float64(soft_center_scale)
    lp += _constraint_logpenalty(
        beta;
        constraint = resolved_constraint,
        soft_center_scale = scale,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    _constraint_gradient_penalty!(
        view(grad, 1:p),
        beta;
        constraint = resolved_constraint,
        soft_center_scale = scale,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    lp += _prior_logposterior_and_gradient!(
        grad,
        u,
        data,
        prior;
        car_weights = use_car ? car_weights : nothing,
        car_precision = car_precision,
    )
    return lp, grad
end
