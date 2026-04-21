function logposterior(
    u::AbstractVector,
    data::CoxPreparedData,
    prior::AbstractCoxPrior;
    soft_center::Bool = true,
    soft_center_scale::Union{Nothing,Real} = nothing,
    car_weights = nothing,
    car_precision::Real = 1.0,
)
    _validate_prior(prior)
    p = size(data.z, 2)
    beta = view(u, 1:p)
    lp = cox_logpartiallikelihood(data, beta)
    if soft_center
        scale = soft_center_scale === nothing ? _default_soft_center_scale(p) : Float64(soft_center_scale)
        lp += _soft_center_logpdf(beta, scale)
    end
    lp += _prior_logposterior(
        u,
        data,
        prior;
        car_weights = car_weights,
        car_precision = car_precision,
    )
    return lp
end
