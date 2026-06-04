abstract type AbstractCoxTarget end

struct CoxProxLassoTarget <: AbstractCoxTarget
    data::CoxPreparedData
    prior::BayesianLassoPrior
    constraint::Symbol
    prox_sum0_lambda::Float64
    prox_l1_lambda::Float64
    alpha_prior_shape::Float64
    alpha_prior_scale::Float64
    alpha_prior_kind::Symbol
    prox_root_tol::Float64
    prox_root_maxiter::Int
    work::ProxGradientWorkspace
end

function CoxProxLassoTarget(
    data::CoxPreparedData,
    prior::BayesianLassoPrior;
    constraint::Symbol,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_scale::Real,
    alpha_prior_kind = :invgamma,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    return CoxProxLassoTarget(
        data,
        prior,
        constraint,
        Float64(prox_sum0_lambda),
        Float64(prox_l1_lambda),
        Float64(alpha_prior_shape),
        Float64(alpha_prior_scale),
        _normalize_alpha_prior_kind(alpha_prior_kind),
        Float64(prox_root_tol),
        Int(prox_root_maxiter),
        ProxGradientWorkspace(data),
    )
end

struct CoxProxSSLassoTarget <: AbstractCoxTarget
    data::CoxPreparedData
    prior::SpikeSlabLassoPrior
    gamma::Vector{Bool}
    theta::Base.RefValue{Float64}
    constraint::Symbol
    prox_sum0_lambda::Float64
    prox_l1_lambda::Float64
    alpha_prior_shape::Float64
    alpha_prior_scale::Float64
    alpha_prior_kind::Symbol
    prox_root_tol::Float64
    prox_root_maxiter::Int
    work::ProxGradientWorkspace
end

struct CoxNoncenteredPosteriorTarget{P<:Union{HorseshoePrior,RegularizedHorseshoePrior}} <: AbstractCoxTarget
    data::CoxPreparedData
    prior::P
    constraint::Symbol
    prox_sum0_lambda::Float64
    car_weights
    use_car::Bool
    car_precision::Float64
    sample_car_strength::Bool
    car_strength_prior_a::Float64
    car_strength_prior_b::Float64
    car_degree_scale::Symbol
    car_kappa_init::Float64
    names::Vector{String}
    output_names::Vector{String}
    work::CoxGradientWorkspace
end

function CoxNoncenteredPosteriorTarget(
    data::CoxPreparedData,
    prior::Union{HorseshoePrior,RegularizedHorseshoePrior};
    constraint::Symbol,
    prox_sum0_lambda::Real,
    car_weights = nothing,
    use_car::Bool = car_weights !== nothing,
    car_precision::Real = 1.0,
    sample_car_strength::Bool = false,
    car_strength_prior_a::Real = 1.0,
    car_strength_prior_b::Real = 1.0,
    car_degree_scale::Symbol = :mean_active,
    car_kappa_init::Real = 0.5,
)
    p = size(data.z, 2)
    effective_car_weights = use_car ? car_weights : nothing
    if sample_car_strength
        _validate_adaptive_car_options(
            effective_car_weights,
            car_strength_prior_a,
            car_strength_prior_b,
            car_kappa_init,
        )
    end
    return CoxNoncenteredPosteriorTarget(
        data,
        prior,
        constraint,
        Float64(prox_sum0_lambda),
        car_weights,
        use_car,
        Float64(car_precision),
        sample_car_strength,
        Float64(car_strength_prior_a),
        Float64(car_strength_prior_b),
        car_degree_scale,
        Float64(car_kappa_init),
        _noncentered_state_names(prior, p; sample_car_strength = sample_car_strength),
        _state_names(prior, p; sample_car_strength = sample_car_strength),
        CoxGradientWorkspace(data),
    )
end

function CoxProxSSLassoTarget(
    data::CoxPreparedData,
    prior::SpikeSlabLassoPrior,
    gamma::AbstractVector{Bool},
    theta::Base.RefValue{Float64};
    constraint::Symbol,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_scale::Real,
    alpha_prior_kind = :invgamma,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    gamma_ref = gamma isa Vector{Bool} ? gamma : collect(Bool, gamma)
    return CoxProxSSLassoTarget(
        data,
        prior,
        gamma_ref,
        theta,
        constraint,
        Float64(prox_sum0_lambda),
        Float64(prox_l1_lambda),
        Float64(alpha_prior_shape),
        Float64(alpha_prior_scale),
        _normalize_alpha_prior_kind(alpha_prior_kind),
        Float64(prox_root_tol),
        Int(prox_root_maxiter),
        ProxGradientWorkspace(data),
    )
end

dimension(target::AbstractCoxTarget) = length(state_names(target))
dimension(target::CoxNoncenteredPosteriorTarget) = length(target.names)

initial_state(target::Union{CoxProxLassoTarget,CoxProxSSLassoTarget}) =
    vcat(zeros(Float64, size(target.data.z, 2)), [0.0])

function initial_state(target::CoxNoncenteredPosteriorTarget)
    p = size(target.data.z, 2)
    n_events = count(!=(0), target.data.event)
    if target.prior isa RegularizedHorseshoePrior
        return _noncentered_initial_state(
            target.prior,
            p,
            n_events;
            n_obs = size(target.data.z, 1),
            car_weights = target.use_car ? target.car_weights : nothing,
            sample_car_strength = target.sample_car_strength,
            car_strength_prior_a = target.car_strength_prior_a,
            car_strength_prior_b = target.car_strength_prior_b,
            car_kappa_init = target.car_kappa_init,
        )
    else
        return _noncentered_initial_state(
            target.prior,
            p,
            n_events;
            car_weights = target.use_car ? target.car_weights : nothing,
            sample_car_strength = target.sample_car_strength,
            car_strength_prior_a = target.car_strength_prior_a,
            car_strength_prior_b = target.car_strength_prior_b,
            car_kappa_init = target.car_kappa_init,
        )
    end
end

state_names(target::Union{CoxProxLassoTarget,CoxProxSSLassoTarget}) =
    vcat(["beta[$j]" for j in 1:size(target.data.z, 2)], ["eta"])
state_names(target::CoxNoncenteredPosteriorTarget) = target.names
sample_state_names(target::AbstractCoxTarget) = state_names(target)
sample_state(target::AbstractCoxTarget, u::AbstractVector) = collect(Float64, u)
sample_logdensity(target::AbstractCoxTarget, saved::AbstractVector, internal::AbstractVector) =
    logdensity(target, internal)

sample_state_names(target::CoxNoncenteredPosteriorTarget) = target.output_names
sample_state(target::CoxNoncenteredPosteriorTarget, u::AbstractVector) =
    _centered_state_from_noncentered(target.prior, target.data, u)
sample_logdensity(target::CoxNoncenteredPosteriorTarget, saved::AbstractVector, internal::AbstractVector) =
    logposterior(
        saved,
        target.data,
        target.prior;
        constraint = target.constraint,
        prox_sum0_lambda = target.prox_sum0_lambda,
        car_weights = target.car_weights,
        use_car = target.use_car,
        car_precision = target.car_precision,
        sample_car_strength = target.sample_car_strength,
        car_strength_prior_a = target.car_strength_prior_a,
        car_strength_prior_b = target.car_strength_prior_b,
        car_degree_scale = target.car_degree_scale,
        car_kappa_init = target.car_kappa_init,
    )

function logdensity(target::CoxProxLassoTarget, u::AbstractVector)
    return _lasso_prox_logposterior(
        u,
        target.data,
        target.prior;
        constraint = target.constraint,
        prox_sum0_lambda = target.prox_sum0_lambda,
        prox_l1_lambda = target.prox_l1_lambda,
        alpha_prior_shape = target.alpha_prior_shape,
        alpha_prior_scale = target.alpha_prior_scale,
        alpha_prior_kind = target.alpha_prior_kind,
        prox_root_tol = target.prox_root_tol,
        prox_root_maxiter = target.prox_root_maxiter,
    )
end

function logdensity(target::CoxProxSSLassoTarget, u::AbstractVector)
    return _ssl_prox_logposterior(
        u,
        target.data,
        target.prior,
        target.gamma,
        target.theta[];
        constraint = target.constraint,
        prox_sum0_lambda = target.prox_sum0_lambda,
        prox_l1_lambda = target.prox_l1_lambda,
        alpha_prior_shape = target.alpha_prior_shape,
        alpha_prior_scale = target.alpha_prior_scale,
        alpha_prior_kind = target.alpha_prior_kind,
        prox_root_tol = target.prox_root_tol,
        prox_root_maxiter = target.prox_root_maxiter,
    )
end

function logdensity(target::CoxNoncenteredPosteriorTarget, u::AbstractVector)
    return noncentered_logposterior(
        u,
        target.data,
        target.prior;
        constraint = target.constraint,
        prox_sum0_lambda = target.prox_sum0_lambda,
        car_weights = target.car_weights,
        use_car = target.use_car,
        car_precision = target.car_precision,
        sample_car_strength = target.sample_car_strength,
        car_strength_prior_a = target.car_strength_prior_a,
        car_strength_prior_b = target.car_strength_prior_b,
        car_degree_scale = target.car_degree_scale,
        car_kappa_init = target.car_kappa_init,
    )
end

function gradient!(grad::AbstractVector{Float64}, target::CoxProxLassoTarget, u::AbstractVector)
    return _lasso_prox_gradient!(
        grad,
        u,
        target.data,
        target.work;
        constraint = target.constraint,
        prox_sum0_lambda = target.prox_sum0_lambda,
        prox_l1_lambda = target.prox_l1_lambda,
        alpha_prior_shape = target.alpha_prior_shape,
        alpha_prior_scale = target.alpha_prior_scale,
        alpha_prior_kind = target.alpha_prior_kind,
        prox_root_tol = target.prox_root_tol,
        prox_root_maxiter = target.prox_root_maxiter,
    )
end

function gradient!(grad::AbstractVector{Float64}, target::CoxProxSSLassoTarget, u::AbstractVector)
    return _ssl_prox_gradient!(
        grad,
        u,
        target.data,
        target.prior,
        target.gamma,
        target.work;
        constraint = target.constraint,
        prox_sum0_lambda = target.prox_sum0_lambda,
        prox_l1_lambda = target.prox_l1_lambda,
        alpha_prior_shape = target.alpha_prior_shape,
        alpha_prior_scale = target.alpha_prior_scale,
        alpha_prior_kind = target.alpha_prior_kind,
        prox_root_tol = target.prox_root_tol,
        prox_root_maxiter = target.prox_root_maxiter,
    )
end

function gradient!(grad::AbstractVector{Float64}, target::CoxNoncenteredPosteriorTarget, u::AbstractVector)
    _noncentered_logposterior_and_gradient!(
        grad,
        u,
        target.data,
        target.prior,
        target.work;
        constraint = target.constraint,
        prox_sum0_lambda = target.prox_sum0_lambda,
        car_weights = target.car_weights,
        use_car = target.use_car,
        car_precision = target.car_precision,
        sample_car_strength = target.sample_car_strength,
        car_strength_prior_a = target.car_strength_prior_a,
        car_strength_prior_b = target.car_strength_prior_b,
        car_degree_scale = target.car_degree_scale,
        car_kappa_init = target.car_kappa_init,
    )
    return grad
end

function logdensity_and_gradient!(
    grad::AbstractVector{Float64},
    target::CoxProxLassoTarget,
    u::AbstractVector,
)
    return _lasso_prox_logposterior_and_gradient!(
        grad,
        u,
        target.data,
        target.work;
        constraint = target.constraint,
        prox_sum0_lambda = target.prox_sum0_lambda,
        prox_l1_lambda = target.prox_l1_lambda,
        alpha_prior_shape = target.alpha_prior_shape,
        alpha_prior_scale = target.alpha_prior_scale,
        alpha_prior_kind = target.alpha_prior_kind,
        prox_root_tol = target.prox_root_tol,
        prox_root_maxiter = target.prox_root_maxiter,
    )
end

function logdensity_and_gradient!(
    grad::AbstractVector{Float64},
    target::CoxNoncenteredPosteriorTarget,
    u::AbstractVector,
)
    return _noncentered_logposterior_and_gradient!(
        grad,
        u,
        target.data,
        target.prior,
        target.work;
        constraint = target.constraint,
        prox_sum0_lambda = target.prox_sum0_lambda,
        car_weights = target.car_weights,
        use_car = target.use_car,
        car_precision = target.car_precision,
        sample_car_strength = target.sample_car_strength,
        car_strength_prior_a = target.car_strength_prior_a,
        car_strength_prior_b = target.car_strength_prior_b,
        car_degree_scale = target.car_degree_scale,
        car_kappa_init = target.car_kappa_init,
    )
end

function logdensity_and_gradient!(
    grad::AbstractVector{Float64},
    target::CoxProxSSLassoTarget,
    u::AbstractVector,
)
    return _ssl_prox_logposterior_and_gradient!(
        grad,
        u,
        target.data,
        target.prior,
        target.gamma,
        target.theta[],
        target.work;
        constraint = target.constraint,
        prox_sum0_lambda = target.prox_sum0_lambda,
        prox_l1_lambda = target.prox_l1_lambda,
        alpha_prior_shape = target.alpha_prior_shape,
        alpha_prior_scale = target.alpha_prior_scale,
        alpha_prior_kind = target.alpha_prior_kind,
        prox_root_tol = target.prox_root_tol,
        prox_root_maxiter = target.prox_root_maxiter,
    )
end

function logdensity_and_gradient!(
    grad::AbstractVector{Float64},
    target::AbstractCoxTarget,
    u::AbstractVector,
)
    lp = logdensity(target, u)
    gradient!(grad, target, u)
    return lp, grad
end

LogDensityProblems.dimension(target::AbstractCoxTarget) = dimension(target)
LogDensityProblems.capabilities(::Type{<:AbstractCoxTarget}) =
    LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(target::AbstractCoxTarget, u::AbstractVector) =
    logdensity(target, u)

function LogDensityProblems.logdensity_and_gradient(
    target::AbstractCoxTarget,
    u::AbstractVector,
)
    grad = Vector{Float64}(undef, dimension(target))
    lp, grad = logdensity_and_gradient!(grad, target, u)
    return lp, grad
end
