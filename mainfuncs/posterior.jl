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
