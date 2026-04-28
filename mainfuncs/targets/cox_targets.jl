abstract type AbstractCoxTarget end

struct CoxProxLassoTarget <: AbstractCoxTarget
    data::CoxPreparedData
    prior::BayesianLassoPrior
    constraint::Symbol
    soft_center_scale::Float64
    prox_sum0_lambda::Float64
    prox_l1_lambda::Float64
    alpha_prior_shape::Float64
    alpha_prior_rate::Float64
    prox_root_tol::Float64
    prox_root_maxiter::Int
    work::ProxGradientWorkspace
end

function CoxProxLassoTarget(
    data::CoxPreparedData,
    prior::BayesianLassoPrior;
    constraint::Symbol,
    soft_center_scale::Real,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_rate::Real,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    return CoxProxLassoTarget(
        data,
        prior,
        constraint,
        Float64(soft_center_scale),
        Float64(prox_sum0_lambda),
        Float64(prox_l1_lambda),
        Float64(alpha_prior_shape),
        Float64(alpha_prior_rate),
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
    soft_center_scale::Float64
    prox_sum0_lambda::Float64
    prox_l1_lambda::Float64
    alpha_prior_shape::Float64
    alpha_prior_rate::Float64
    prox_root_tol::Float64
    prox_root_maxiter::Int
    work::ProxGradientWorkspace
end

function CoxProxSSLassoTarget(
    data::CoxPreparedData,
    prior::SpikeSlabLassoPrior,
    gamma::AbstractVector{Bool},
    theta::Base.RefValue{Float64};
    constraint::Symbol,
    soft_center_scale::Real,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_rate::Real,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    return CoxProxSSLassoTarget(
        data,
        prior,
        collect(Bool, gamma),
        theta,
        constraint,
        Float64(soft_center_scale),
        Float64(prox_sum0_lambda),
        Float64(prox_l1_lambda),
        Float64(alpha_prior_shape),
        Float64(alpha_prior_rate),
        Float64(prox_root_tol),
        Int(prox_root_maxiter),
        ProxGradientWorkspace(data),
    )
end

dimension(target::AbstractCoxTarget) = length(state_names(target))

initial_state(target::Union{CoxProxLassoTarget,CoxProxSSLassoTarget}) =
    vcat(zeros(Float64, size(target.data.z, 2)), [0.0])

state_names(target::Union{CoxProxLassoTarget,CoxProxSSLassoTarget}) =
    vcat(["beta[$j]" for j in 1:size(target.data.z, 2)], ["eta"])

function logdensity(target::CoxProxLassoTarget, u::AbstractVector)
    return _lasso_prox_logposterior(
        u,
        target.data,
        target.prior;
        constraint = target.constraint,
        soft_center_scale = target.soft_center_scale,
        prox_sum0_lambda = target.prox_sum0_lambda,
        prox_l1_lambda = target.prox_l1_lambda,
        alpha_prior_shape = target.alpha_prior_shape,
        alpha_prior_rate = target.alpha_prior_rate,
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
        soft_center_scale = target.soft_center_scale,
        prox_sum0_lambda = target.prox_sum0_lambda,
        prox_l1_lambda = target.prox_l1_lambda,
        alpha_prior_shape = target.alpha_prior_shape,
        alpha_prior_rate = target.alpha_prior_rate,
        prox_root_tol = target.prox_root_tol,
        prox_root_maxiter = target.prox_root_maxiter,
    )
end

function gradient!(grad::AbstractVector{Float64}, target::CoxProxLassoTarget, u::AbstractVector)
    return _lasso_prox_gradient!(
        grad,
        u,
        target.data,
        target.work;
        constraint = target.constraint,
        soft_center_scale = target.soft_center_scale,
        prox_sum0_lambda = target.prox_sum0_lambda,
        prox_l1_lambda = target.prox_l1_lambda,
        alpha_prior_shape = target.alpha_prior_shape,
        alpha_prior_rate = target.alpha_prior_rate,
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
        soft_center_scale = target.soft_center_scale,
        prox_sum0_lambda = target.prox_sum0_lambda,
        prox_l1_lambda = target.prox_l1_lambda,
        alpha_prior_shape = target.alpha_prior_shape,
        alpha_prior_rate = target.alpha_prior_rate,
        prox_root_tol = target.prox_root_tol,
        prox_root_maxiter = target.prox_root_maxiter,
    )
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
        soft_center_scale = target.soft_center_scale,
        prox_sum0_lambda = target.prox_sum0_lambda,
        prox_l1_lambda = target.prox_l1_lambda,
        alpha_prior_shape = target.alpha_prior_shape,
        alpha_prior_rate = target.alpha_prior_rate,
        prox_root_tol = target.prox_root_tol,
        prox_root_maxiter = target.prox_root_maxiter,
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
        soft_center_scale = target.soft_center_scale,
        prox_sum0_lambda = target.prox_sum0_lambda,
        prox_l1_lambda = target.prox_l1_lambda,
        alpha_prior_shape = target.alpha_prior_shape,
        alpha_prior_rate = target.alpha_prior_rate,
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
