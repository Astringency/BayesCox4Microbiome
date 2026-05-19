#!/usr/bin/env julia

# Diagnostic GLM analogue for the Cox simulation anomaly.
#
# This script follows the simulation design in docs/zhang2024BCGLM.pdf:
# logistic-normal compositional X, eta = log(X) * beta, nonzero beta sets at
# the paper's indices, Gaussian outcomes, and balanced binary outcomes obtained
# by median dichotomization of the latent continuous outcome. It reuses the
# package's existing prior definitions and explicit NUTS machinery, replacing
# only the Cox partial likelihood by a GLM likelihood.

using Pkg

function _activate_repo_project()
    project = abspath(joinpath(@__DIR__, "..", "Project.toml"))
    Base.active_project() == project || Pkg.activate(dirname(project))
end

_activate_repo_project()

using BayesCox4Micro
using CSV
using DataFrames
using Distributions
using LinearAlgebra
using MCMCChains
using Random
using Statistics

const B = BayesCox4Micro

function parse_cli_args(args = ARGS)
    out = Dict{String,String}()
    i = 1
    while i <= length(args)
        item = args[i]
        if startswith(item, "--")
            key = item[3:end]
            if occursin("=", key)
                k, v = split(key, "="; limit = 2)
                out[k] = v
            elseif i < length(args) && !startswith(args[i + 1], "--")
                out[key] = args[i + 1]
                i += 1
            else
                out[key] = "true"
            end
        end
        i += 1
    end
    return out
end

arg(args, key, default::String) = get(args, key, default)
arg(args, key, default::Int) = parse(Int, get(args, key, string(default)))
arg(args, key, default::Float64) = parse(Float64, get(args, key, string(default)))
arg(args, key, default::Bool) = lowercase(get(args, key, string(default))) in ("true", "1", "yes", "y")
function optional_float_arg(args, keys...)
    for key in keys
        haskey(args, key) && return parse(Float64, args[key])
    end
    return nothing
end
split_strings(s::AbstractString) = [strip(x) for x in split(s, ","; keepempty = false)]
split_ints(s::AbstractString) = parse.(Int, split_strings(s))
split_float_or_truth(s::AbstractString, truth::Int) =
    [lowercase(x) == "truth" ? Float64(truth) : parse(Float64, x) for x in split_strings(s)]

ensure_dir(path::AbstractString) = (mkpath(path); path)

function logistic_stable(x::Real)
    xf = Float64(x)
    if xf >= 0
        z = exp(-xf)
        return inv(1 + z)
    else
        z = exp(xf)
        return z / (1 + z)
    end
end

function log1pexp_stable(x::Real)
    xf = Float64(x)
    return xf > 0 ? xf + log1p(exp(-xf)) : log1p(exp(xf))
end

function auc_score(score::AbstractVector, y::AbstractVector)
    pos = findall(==(1.0), Float64.(y))
    neg = findall(==(0.0), Float64.(y))
    (isempty(pos) || isempty(neg)) && return NaN
    wins = 0.0
    total = length(pos) * length(neg)
    @inbounds for i in pos, j in neg
        wins += score[i] > score[j] ? 1.0 : score[i] == score[j] ? 0.5 : 0.0
    end
    return wins / total
end

struct BCGLMData
    family::Symbol
    X::Matrix{Float64}
    z::Matrix{Float64}
    y::Vector{Float64}
    beta_true::Vector{Float64}
    nonzero_index::Vector{Int}
    eta_true::Vector{Float64}
    sigma::Float64
    binary_mode::Symbol
end

mutable struct GLMWork
    eta::Vector{Float64}
    mu::Vector{Float64}
end

GLMWork(n::Integer) = GLMWork(Vector{Float64}(undef, n), Vector{Float64}(undef, n))

struct GLMPosteriorTarget <: B.AbstractCoxTarget
    data::BCGLMData
    fake_cox_data::B.CoxPreparedData
    prior::B.AbstractCoxPrior
    constraint::Symbol
    soft_center_scale::Float64
    prox_sum0_lambda::Float64
    car_weights
    use_car::Bool
    car_precision::Float64
    names::Vector{String}
    work::GLMWork
end

mutable struct GLMProxWork
    glm::GLMWork
    prox_beta::Vector{Float64}
    weights::Vector{Float64}
end

GLMProxWork(n::Integer, p::Integer) =
    GLMProxWork(GLMWork(n), zeros(Float64, p), ones(Float64, p))

struct GLMProxLassoTarget <: B.AbstractCoxTarget
    data::BCGLMData
    prior::B.BayesianLassoPrior
    constraint::Symbol
    soft_center_scale::Float64
    prox_sum0_lambda::Float64
    prox_l1_lambda::Float64
    alpha_prior_shape::Float64
    alpha_prior_rate::Float64
    prox_root_tol::Float64
    prox_root_maxiter::Int
    names::Vector{String}
    work::GLMProxWork
end

struct GLMProxSSLassoTarget <: B.AbstractCoxTarget
    data::BCGLMData
    prior::B.SpikeSlabLassoPrior
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
    names::Vector{String}
    work::GLMProxWork
end

function true_beta_zhang(p::Integer, nonzero::Integer)
    nonzero in (6, 12, 24) ||
        throw(ArgumentError("nonzero must be one of 6, 12, or 24 for the Zhang BCGLM design."))
    vals12 = [2.08, -1.41, -1.39, -1.15, 2.12, 2.51, 1.31, -1.95, -1.86, 1.93, -1.34, -0.85]
    vals6 = [2.08, 1.31, -1.39, -1.86, 2.12, -2.26]
    paper_idx =
        nonzero == 6 ? [16 + 2i for i in 1:6] :
        nonzero == 12 ? [16 + 2i for i in 1:12] :
        vcat([16 + 2i for i in 1:12], [66 + 2i for i in 1:12])
    values = nonzero == 6 ? vals6 : nonzero == 12 ? vals12 : vcat(vals12, vals12)
    idx =
        maximum(paper_idx) <= p ? paper_idx :
        unique(round.(Int, range(2, p - 1; length = nonzero)))
    length(idx) == nonzero ||
        throw(ArgumentError("p=$p is too small to place $nonzero unique nonzero coefficients."))
    beta = zeros(Float64, p)
    beta[idx] .= values
    abs(sum(beta)) < 1e-10 ||
        throw(AssertionError("Zhang beta set should satisfy sum(beta)=0; got $(sum(beta))."))
    return beta, collect(idx)
end

function _true_block_cov(idx::AbstractVector{Int})
    k = length(idx)
    Σ = Matrix{Float64}(I, k, k)
    @inbounds for a in 1:k, b in 1:k
        if a != b
            Σ[a, b] = max(0.0, 0.75 - 0.015 * abs(idx[a] - idx[b]))
        end
    end
    jitter = 1e-8
    while true
        try
            return cholesky(Symmetric(Σ + jitter * I)).L
        catch
            jitter *= 10
            jitter <= 1.0 ||
                throw(ArgumentError("could not make true-predictor covariance positive definite."))
        end
    end
end

function simulate_logistic_normal_X(rng::AbstractRNG, n::Integer, p::Integer, idx::AbstractVector{Int})
    U = randn(rng, n, p)
    k = length(idx)
    θ = fill(log(0.5 * p), k)
    L = _true_block_cov(idx)
    @inbounds for i in 1:n
        U[i, idx] .= θ .+ L * randn(rng, k)
    end
    X = similar(U)
    @inbounds for i in 1:n
        row = view(U, i, :)
        m = maximum(row)
        denom = 0.0
        for j in 1:p
            X[i, j] = exp(U[i, j] - m)
            denom += X[i, j]
        end
        for j in 1:p
            X[i, j] /= denom
        end
    end
    return X
end

function simulate_bcglm_data(;
    family::Symbol,
    n::Integer,
    p::Integer,
    nonzero::Integer,
    seed::Integer,
    sigma::Real,
    binary_mode::Symbol,
)
    rng = MersenneTwister(seed)
    beta, idx = true_beta_zhang(p, nonzero)
    X = simulate_logistic_normal_X(rng, n, p, idx)
    z = log.(X)
    eta = z * beta
    σ = Float64(sigma)
    latent = eta .+ σ .* randn(rng, n)
    y =
        family == :gaussian ? latent :
        family == :logistic && binary_mode == :median ? Float64.(latent .>= median(latent)) :
        family == :logistic && binary_mode == :bernoulli ? Float64.(rand(rng, n) .< logistic_stable.(eta)) :
        throw(ArgumentError("family must be gaussian or logistic; binary_mode must be median or bernoulli."))
    return BCGLMData(family, X, z, y, beta, idx, eta, σ, binary_mode)
end

function fake_cox_data(data::BCGLMData)
    n = size(data.X, 1)
    return B.prepare_cox_data(
        data.X,
        collect(1.0:n),
        ones(Int, n);
        transform = :log,
        pseudocount = 0.0,
        closure = false,
        standardize = false,
    )
end

function glm_loglik(data::BCGLMData, beta::AbstractVector)
    eta = data.z * beta
    if data.family == :gaussian
        invσ2 = inv(data.sigma^2)
        return -0.5 * length(data.y) * log(2π * data.sigma^2) -
            0.5 * invσ2 * sum(abs2, data.y .- eta)
    elseif data.family == :logistic
        ll = 0.0
        @inbounds for i in eachindex(eta)
            ll += data.y[i] * eta[i] - log1pexp_stable(eta[i])
        end
        return ll
    end
    throw(ArgumentError("unsupported GLM family: $(data.family)"))
end

function glm_loglik_and_gradient!(
    grad_beta::AbstractVector{Float64},
    data::BCGLMData,
    beta::AbstractVector,
    work::GLMWork,
)
    mul!(work.eta, data.z, beta)
    fill!(grad_beta, 0.0)
    if data.family == :gaussian
        invσ2 = inv(data.sigma^2)
        ll = -0.5 * length(data.y) * log(2π * data.sigma^2)
        @inbounds for i in eachindex(work.eta)
            r = data.y[i] - work.eta[i]
            ll -= 0.5 * invσ2 * r^2
            work.mu[i] = r * invσ2
        end
        mul!(grad_beta, transpose(data.z), work.mu)
        return ll
    elseif data.family == :logistic
        ll = 0.0
        @inbounds for i in eachindex(work.eta)
            η = work.eta[i]
            pr = logistic_stable(η)
            ll += data.y[i] * η - log1pexp_stable(η)
            work.mu[i] = data.y[i] - pr
        end
        mul!(grad_beta, transpose(data.z), work.mu)
        return ll
    end
    throw(ArgumentError("unsupported GLM family: $(data.family)"))
end

function make_prior(
    name::AbstractString,
    p0::Real,
    family::Symbol,
    sigma::Real;
    hs_tau_scale::Real = 1.0,
    hs_local_scale::Real = 1.0,
    rhs_tau_scale::Real = 1.0,
    rhs_local_scale::Real = 1.0,
    rhs_slab_df::Real = 4.0,
    rhs_slab_s2::Real = 2.0,
    rhs_pseudo_sigma_df::Real = 3.0,
    rhs_pseudo_sigma_scale::Real = 10.0,
    lasso_lambda = nothing,
    lasso_lambda2_shape::Real = 1.0,
    lasso_lambda2_rate::Real = 0.1,
    lasso_tau2_shape::Real = 1.0,
    sslasso_spike_scale::Real = 0.04,
    sslasso_slab_scale::Real = 0.5,
    sslasso_theta_a::Real = 1.0,
    sslasso_theta_b::Real = 1.0,
)
    key = lowercase(replace(name, "_" => "-", " " => "-"))
    if key in ("hs", "horseshoe", "bcglm-hs")
        return B.HorseshoePrior(
            tau_scale = Float64(hs_tau_scale),
            use_cox_tau0 = false,
            p0 = Float64(p0),
            local_scale = Float64(hs_local_scale),
        )
    elseif key in ("cox-hs", "coxhs")
        return B.HorseshoePrior(
            tau_scale = Float64(hs_tau_scale),
            use_cox_tau0 = true,
            p0 = Float64(p0),
            local_scale = Float64(hs_local_scale),
        )
    elseif key in ("rhs", "regularized-horseshoe", "bcglm-rhs")
        return B.RegularizedHorseshoePrior(
            tau_scale = Float64(rhs_tau_scale),
            use_cox_tau0 = false,
            p0 = Float64(p0),
            local_scale = Float64(rhs_local_scale),
            tau_formula = :linear,
            pseudo_sigma = family == :gaussian ? Float64(sigma) : 1.0,
            sample_pseudo_sigma = false,
            slab_df = Float64(rhs_slab_df),
            slab_s2 = Float64(rhs_slab_s2),
        )
    elseif key in ("new-rhs", "newrhs")
        return B.RegularizedHorseshoePrior(
            tau_scale = Float64(rhs_tau_scale),
            use_cox_tau0 = false,
            p0 = Float64(p0),
            local_scale = Float64(rhs_local_scale),
            tau_formula = :linear,
            pseudo_sigma = 1.0,
            sample_pseudo_sigma = false,
            slab_df = Float64(rhs_slab_df),
            slab_s2 = Float64(rhs_slab_s2),
        )
    elseif key in ("est-rhs", "estrhs")
        return B.RegularizedHorseshoePrior(
            tau_scale = Float64(rhs_tau_scale),
            use_cox_tau0 = false,
            p0 = Float64(p0),
            local_scale = Float64(rhs_local_scale),
            tau_formula = :linear,
            pseudo_sigma = family == :gaussian ? Float64(sigma) : 1.0,
            sample_pseudo_sigma = false,
            slab_df = Float64(rhs_slab_df),
            slab_s2 = Float64(rhs_slab_s2),
        )
    elseif key in ("rhs-sigma", "rhs-sample-sigma")
        return B.RegularizedHorseshoePrior(
            tau_scale = Float64(rhs_tau_scale),
            use_cox_tau0 = false,
            p0 = Float64(p0),
            local_scale = Float64(rhs_local_scale),
            tau_formula = :linear,
            pseudo_sigma = nothing,
            sample_pseudo_sigma = true,
            pseudo_sigma_df = Float64(rhs_pseudo_sigma_df),
            pseudo_sigma_scale = Float64(rhs_pseudo_sigma_scale),
            slab_df = Float64(rhs_slab_df),
            slab_s2 = Float64(rhs_slab_s2),
        )
    elseif key in ("cox-rhs", "coxrhs")
        return B.RegularizedHorseshoePrior(
            tau_scale = Float64(rhs_tau_scale),
            use_cox_tau0 = true,
            p0 = Float64(p0),
            local_scale = Float64(rhs_local_scale),
            tau_formula = :cox,
            slab_df = Float64(rhs_slab_df),
            slab_s2 = Float64(rhs_slab_s2),
        )
    elseif key in ("lasso", "bayesian-lasso")
        return B.BayesianLassoPrior(
            lambda = lasso_lambda === nothing ? nothing : Float64(lasso_lambda),
            lambda2_shape = Float64(lasso_lambda2_shape),
            lambda2_rate = Float64(lasso_lambda2_rate),
            tau2_shape = Float64(lasso_tau2_shape),
        )
    elseif key in ("prox-lasso", "proximal-lasso")
        return B.BayesianLassoPrior(
            lambda = lasso_lambda === nothing ? nothing : Float64(lasso_lambda),
            lambda2_shape = Float64(lasso_lambda2_shape),
            lambda2_rate = Float64(lasso_lambda2_rate),
            tau2_shape = Float64(lasso_tau2_shape),
            proximal = true,
        )
    elseif key in ("sslasso", "spike-slab", "spike-and-slab")
        return B.SpikeSlabLassoPrior(
            spike_scale = Float64(sslasso_spike_scale),
            slab_scale = Float64(sslasso_slab_scale),
            theta_a = Float64(sslasso_theta_a),
            theta_b = Float64(sslasso_theta_b),
        )
    elseif key in ("prox-sslasso", "prox-ssl", "proximal-sslasso")
        return B.SpikeSlabLassoPrior(
            spike_scale = Float64(sslasso_spike_scale),
            slab_scale = Float64(sslasso_slab_scale),
            theta_a = Float64(sslasso_theta_a),
            theta_b = Float64(sslasso_theta_b),
            proximal = true,
        )
    end
    throw(ArgumentError("unsupported prior for GLM diagnostic: $name"))
end

function GLMPosteriorTarget(
    data::BCGLMData,
    prior::B.AbstractCoxPrior;
    constraint::Symbol,
    soft_center_scale::Union{Nothing,Real},
    prox_sum0_lambda::Real,
    car_weights = nothing,
    use_car::Bool = car_weights !== nothing,
    car_precision::Real = 1.0,
)
    p = size(data.z, 2)
    fake = fake_cox_data(data)
    return GLMPosteriorTarget(
        data,
        fake,
        prior,
        constraint,
        soft_center_scale === nothing ? B._default_soft_center_scale(p) : Float64(soft_center_scale),
        Float64(prox_sum0_lambda),
        car_weights,
        use_car,
        Float64(car_precision),
        B._state_names(prior, p),
        GLMWork(size(data.z, 1)),
    )
end

function _prox_names(p::Integer)
    return vcat(["beta[$j]" for j in 1:p], ["eta"])
end

function GLMProxLassoTarget(
    data::BCGLMData,
    prior::B.BayesianLassoPrior;
    constraint::Symbol,
    soft_center_scale::Real,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_rate::Real,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    p = size(data.z, 2)
    return GLMProxLassoTarget(
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
        _prox_names(p),
        GLMProxWork(size(data.z, 1), p),
    )
end

function GLMProxSSLassoTarget(
    data::BCGLMData,
    prior::B.SpikeSlabLassoPrior,
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
    p = size(data.z, 2)
    return GLMProxSSLassoTarget(
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
        _prox_names(p),
        GLMProxWork(size(data.z, 1), p),
    )
end

B.dimension(target::GLMPosteriorTarget) = length(target.names)
B.state_names(target::GLMPosteriorTarget) = target.names
B.dimension(target::Union{GLMProxLassoTarget,GLMProxSSLassoTarget}) = length(target.names)
B.state_names(target::Union{GLMProxLassoTarget,GLMProxSSLassoTarget}) = target.names
B.initial_state(target::Union{GLMProxLassoTarget,GLMProxSSLassoTarget}) =
    vcat(zeros(Float64, size(target.data.z, 2)), [0.0])

function B.initial_state(target::GLMPosteriorTarget)
    p = size(target.data.z, 2)
    n = size(target.data.z, 1)
    if target.prior isa B.RegularizedHorseshoePrior
        return B.initial_state(target.prior, p, n; n_obs = n)
    else
        return B.initial_state(target.prior, p, n)
    end
end

function B.logdensity(target::GLMPosteriorTarget, u::AbstractVector)
    p = size(target.data.z, 2)
    beta = view(u, 1:p)
    lp = glm_loglik(target.data, beta)
    lp += B._constraint_logpenalty(
        beta;
        constraint = target.constraint,
        soft_center_scale = target.soft_center_scale,
        prox_sum0_lambda = target.prox_sum0_lambda,
    )
    lp += B._prior_logposterior(
        u,
        target.fake_cox_data,
        target.prior;
        car_weights = target.use_car ? target.car_weights : nothing,
        car_precision = target.car_precision,
    )
    return lp
end

function B.gradient!(grad::AbstractVector{Float64}, target::GLMPosteriorTarget, u::AbstractVector)
    p = size(target.data.z, 2)
    fill!(grad, 0.0)
    beta = view(u, 1:p)
    glm_loglik_and_gradient!(view(grad, 1:p), target.data, beta, target.work)
    B._constraint_gradient_penalty!(
        view(grad, 1:p),
        beta;
        constraint = target.constraint,
        soft_center_scale = target.soft_center_scale,
        prox_sum0_lambda = target.prox_sum0_lambda,
    )
    B._prior_logposterior_and_gradient!(
        grad,
        u,
        target.fake_cox_data,
        target.prior;
        car_weights = target.use_car ? target.car_weights : nothing,
        car_precision = target.car_precision,
    )
    return grad
end

function B.logdensity_and_gradient!(
    grad::AbstractVector{Float64},
    target::GLMPosteriorTarget,
    u::AbstractVector,
)
    p = size(target.data.z, 2)
    fill!(grad, 0.0)
    beta = view(u, 1:p)
    lp = glm_loglik_and_gradient!(view(grad, 1:p), target.data, beta, target.work)
    lp += B._constraint_logpenalty(
        beta;
        constraint = target.constraint,
        soft_center_scale = target.soft_center_scale,
        prox_sum0_lambda = target.prox_sum0_lambda,
    )
    B._constraint_gradient_penalty!(
        view(grad, 1:p),
        beta;
        constraint = target.constraint,
        soft_center_scale = target.soft_center_scale,
        prox_sum0_lambda = target.prox_sum0_lambda,
    )
    lp += B._prior_logposterior_and_gradient!(
        grad,
        u,
        target.fake_cox_data,
        target.prior;
        car_weights = target.use_car ? target.car_weights : nothing,
        car_precision = target.car_precision,
    )
    return lp, grad
end

function prox_alpha_hyperparameters(prior::B.AbstractCoxPrior, alpha_prior_shape, alpha_prior_rate)
    shape = Float64(something(alpha_prior_shape, prior isa B.BayesianLassoPrior ? prior.lambda2_shape : 1.0))
    rate = Float64(something(alpha_prior_rate, prior isa B.BayesianLassoPrior ? prior.lambda2_rate : 1.0))
    B._check_positive(shape, "alpha_prior_shape")
    B._check_positive(rate, "alpha_prior_rate")
    return shape, rate
end

function glm_prox_lasso_logposterior(
    u::AbstractVector,
    data::BCGLMData;
    constraint::Symbol,
    soft_center_scale::Real,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_rate::Real,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    p = size(data.z, 2)
    length(u) == p + 1 || throw(DimensionMismatch("state length must be p + 1 for prox lasso."))
    beta = view(u, 1:p)
    eta = Float64(u[p + 1])
    alpha = exp(clamp(eta, -700.0, 700.0))
    lp = glm_loglik(data, beta)
    lp += B._constraint_logpenalty(
        beta;
        constraint = constraint,
        soft_center_scale = soft_center_scale,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    lp -= B.l1_moreau(
        beta,
        alpha,
        prox_l1_lambda;
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    lp += -Float64(alpha_prior_rate) * exp(-eta) - Float64(alpha_prior_shape) * eta
    return lp
end

function glm_prox_lasso_logposterior_and_gradient!(
    grad::AbstractVector{Float64},
    u::AbstractVector,
    data::BCGLMData,
    work::GLMProxWork;
    constraint::Symbol,
    soft_center_scale::Real,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_rate::Real,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    p = size(data.z, 2)
    length(u) == p + 1 || throw(DimensionMismatch("state length must be p + 1 for prox lasso."))
    length(grad) == p + 1 || throw(DimensionMismatch("gradient length must be p + 1 for prox lasso."))
    fill!(grad, 0.0)
    beta = view(u, 1:p)
    eta = Float64(u[p + 1])
    alpha, alpha_deriv = B._clamped_exp_with_derivative(eta, -700.0, 700.0)
    lambda_l1 = B._check_positive(prox_l1_lambda, "prox_l1_lambda")
    prox = B.project_l1_epigraph!(
        work.prox_beta,
        beta,
        alpha;
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    lp = glm_loglik_and_gradient!(view(grad, 1:p), data, beta, work.glm)
    lp += B._constraint_logpenalty(
        beta;
        constraint = constraint,
        soft_center_scale = soft_center_scale,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    B._constraint_gradient_penalty!(
        view(grad, 1:p),
        beta;
        constraint = constraint,
        soft_center_scale = soft_center_scale,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    dist2 = (Float64(alpha) - Float64(prox.alpha))^2
    @inbounds for j in 1:p
        residual = Float64(beta[j]) - Float64(prox.beta[j])
        dist2 += residual^2
        grad[j] -= residual / lambda_l1
    end
    lp -= 0.5 * dist2 / lambda_l1
    lp += -Float64(alpha_prior_rate) * exp(-eta) - Float64(alpha_prior_shape) * eta
    grad[p + 1] = Float64(alpha_prior_rate) * exp(-eta) - Float64(alpha_prior_shape) -
        alpha_deriv * (Float64(alpha) - Float64(prox.alpha)) / lambda_l1
    return lp, grad
end

function glm_prox_ssl_logposterior(
    u::AbstractVector,
    data::BCGLMData,
    prior::B.SpikeSlabLassoPrior,
    gamma::AbstractVector{Bool},
    theta::Real;
    constraint::Symbol,
    soft_center_scale::Real,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_rate::Real,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    p = size(data.z, 2)
    length(u) == p + 1 || throw(DimensionMismatch("state length must be p + 1 for prox SSLASSO."))
    beta = view(u, 1:p)
    eta = Float64(u[p + 1])
    alpha = exp(clamp(eta, -700.0, 700.0))
    weights = B._ssl_weights(prior, gamma)
    lp = glm_loglik(data, beta)
    lp += B._constraint_logpenalty(
        beta;
        constraint = constraint,
        soft_center_scale = soft_center_scale,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    lp -= B.weighted_l1_moreau(
        beta,
        alpha,
        weights,
        prox_l1_lambda;
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    lp += -Float64(alpha_prior_rate) * exp(-eta) - Float64(alpha_prior_shape) * eta
    theta_clamped = clamp(theta, eps(Float64), 1 - eps(Float64))
    lp += sum(gamma) * log(theta_clamped) + (length(gamma) - sum(gamma)) * log1p(-theta_clamped)
    return lp
end

function glm_prox_ssl_logposterior_and_gradient!(
    grad::AbstractVector{Float64},
    u::AbstractVector,
    data::BCGLMData,
    prior::B.SpikeSlabLassoPrior,
    gamma::AbstractVector{Bool},
    theta::Real,
    work::GLMProxWork;
    constraint::Symbol,
    soft_center_scale::Real,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_rate::Real,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    p = size(data.z, 2)
    length(u) == p + 1 || throw(DimensionMismatch("state length must be p + 1 for prox SSLASSO."))
    length(grad) == p + 1 || throw(DimensionMismatch("gradient length must be p + 1 for prox SSLASSO."))
    fill!(grad, 0.0)
    beta = view(u, 1:p)
    eta = Float64(u[p + 1])
    alpha, alpha_deriv = B._clamped_exp_with_derivative(eta, -700.0, 700.0)
    weights = B._ssl_weights!(work.weights, prior, gamma)
    lambda_l1 = B._check_positive(prox_l1_lambda, "prox_l1_lambda")
    prox = B.project_weighted_l1_epigraph!(
        work.prox_beta,
        beta,
        alpha,
        weights;
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    lp = glm_loglik_and_gradient!(view(grad, 1:p), data, beta, work.glm)
    lp += B._constraint_logpenalty(
        beta;
        constraint = constraint,
        soft_center_scale = soft_center_scale,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    B._constraint_gradient_penalty!(
        view(grad, 1:p),
        beta;
        constraint = constraint,
        soft_center_scale = soft_center_scale,
        prox_sum0_lambda = prox_sum0_lambda,
    )
    dist2 = (Float64(alpha) - Float64(prox.alpha))^2
    @inbounds for j in 1:p
        residual = Float64(beta[j]) - Float64(prox.beta[j])
        dist2 += residual^2
        grad[j] -= residual / lambda_l1
    end
    lp -= 0.5 * dist2 / lambda_l1
    lp += -Float64(alpha_prior_rate) * exp(-eta) - Float64(alpha_prior_shape) * eta
    theta_clamped = clamp(theta, eps(Float64), 1 - eps(Float64))
    lp += sum(gamma) * log(theta_clamped) + (length(gamma) - sum(gamma)) * log1p(-theta_clamped)
    grad[p + 1] = Float64(alpha_prior_rate) * exp(-eta) - Float64(alpha_prior_shape) -
        alpha_deriv * (Float64(alpha) - Float64(prox.alpha)) / lambda_l1
    return lp, grad
end

B.logdensity(target::GLMProxLassoTarget, u::AbstractVector) =
    glm_prox_lasso_logposterior(
        u,
        target.data;
        constraint = target.constraint,
        soft_center_scale = target.soft_center_scale,
        prox_sum0_lambda = target.prox_sum0_lambda,
        prox_l1_lambda = target.prox_l1_lambda,
        alpha_prior_shape = target.alpha_prior_shape,
        alpha_prior_rate = target.alpha_prior_rate,
        prox_root_tol = target.prox_root_tol,
        prox_root_maxiter = target.prox_root_maxiter,
    )

B.logdensity(target::GLMProxSSLassoTarget, u::AbstractVector) =
    glm_prox_ssl_logposterior(
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

function B.logdensity_and_gradient!(
    grad::AbstractVector{Float64},
    target::GLMProxLassoTarget,
    u::AbstractVector,
)
    return glm_prox_lasso_logposterior_and_gradient!(
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

function B.logdensity_and_gradient!(
    grad::AbstractVector{Float64},
    target::GLMProxSSLassoTarget,
    u::AbstractVector,
)
    return glm_prox_ssl_logposterior_and_gradient!(
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

B.gradient!(
    grad::AbstractVector{Float64},
    target::Union{GLMProxLassoTarget,GLMProxSSLassoTarget},
    u::AbstractVector,
) = (B.logdensity_and_gradient!(grad, target, u); grad)

function sample_glm_ssl_prox_nuts_gibbs_one(
    data::BCGLMData,
    prior::B.SpikeSlabLassoPrior;
    n_samples::Integer,
    warmup::Integer,
    rng::AbstractRNG,
    target_accept::Real,
    max_depth::Integer,
    constraint::Symbol,
    soft_center_scale::Real,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_rate::Real,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
    progress::Bool,
    progress_desc::AbstractString = "GLM SSLASSO prox NUTS-Gibbs",
)
    p = size(data.z, 2)
    gamma = trues(p)
    theta = 0.5
    theta_ref = Ref(theta)
    weights = B._ssl_weights(prior, gamma)
    spike_weight = inv(prior.spike_scale)
    slab_weight = inv(prior.slab_scale)
    total_iter = warmup + n_samples
    names = vcat(["beta[$j]" for j in 1:p], ["eta"], ["gamma[$j]" for j in 1:p], ["theta"])
    draws = Matrix{Float64}(undef, n_samples, length(names))
    lps = Vector{Float64}(undef, n_samples)
    u = vcat(zeros(Float64, p), [0.0])
    saved = 0
    accept_sum = 0.0
    accept_count = 0

    for iter in 1:total_iter
        theta_ref[] = theta
        target = GLMProxSSLassoTarget(
            data,
            prior,
            gamma,
            theta_ref;
            constraint = constraint,
            soft_center_scale = soft_center_scale,
            prox_sum0_lambda = prox_sum0_lambda,
            prox_l1_lambda = prox_l1_lambda,
            alpha_prior_shape = alpha_prior_shape,
            alpha_prior_rate = alpha_prior_rate,
            prox_root_tol = prox_root_tol,
            prox_root_maxiter = prox_root_maxiter,
        )
        block = B._sample_explicit_nuts_one(
            target;
            n_samples = 1,
            warmup = 0,
            rng = rng,
            target_accept = target_accept,
            max_depth = max_depth,
            progress = false,
            progress_desc = progress_desc,
            initial = u,
        )
        @views u .= block.draws[1, :]
        if isfinite(block.acceptance_rate)
            accept_sum += block.acceptance_rate
            accept_count += 1
        end

        beta = view(u, 1:p)
        alpha = exp(clamp(u[p + 1], -700.0, 700.0))
        theta_clamped = clamp(theta, eps(Float64), 1 - eps(Float64))
        @inbounds for j in 1:p
            gamma[j] = false
            weights[j] = spike_weight
            h0 = B.weighted_l1_moreau(
                beta,
                alpha,
                weights,
                prox_l1_lambda;
                prox_root_tol = prox_root_tol,
                prox_root_maxiter = prox_root_maxiter,
            )
            gamma[j] = true
            weights[j] = slab_weight
            h1 = B.weighted_l1_moreau(
                beta,
                alpha,
                weights,
                prox_l1_lambda;
                prox_root_tol = prox_root_tol,
                prox_root_maxiter = prox_root_maxiter,
            )
            logit_p = log(theta_clamped) - log1p(-theta_clamped) - (h1 - h0)
            gamma[j] = rand(rng) < logistic_stable(logit_p)
            weights[j] = gamma[j] ? slab_weight : spike_weight
        end
        theta = rand(rng, Beta(prior.theta_a + count(gamma), prior.theta_b + p - count(gamma)))

        if iter > warmup
            saved += 1
            @views draws[saved, 1:p] .= beta
            draws[saved, p + 1] = u[p + 1]
            @views draws[saved, (p + 2):(2p + 1)] .= Float64.(gamma)
            draws[saved, 2p + 2] = theta
            lps[saved] = glm_prox_ssl_logposterior(
                u,
                data,
                prior,
                gamma,
                theta;
                constraint = constraint,
                soft_center_scale = soft_center_scale,
                prox_sum0_lambda = prox_sum0_lambda,
                prox_l1_lambda = prox_l1_lambda,
                alpha_prior_shape = alpha_prior_shape,
                alpha_prior_rate = alpha_prior_rate,
                prox_root_tol = prox_root_tol,
                prox_root_maxiter = prox_root_maxiter,
            )
        end
        progress && saved > 0 && saved % max(1, n_samples ÷ 10) == 0 &&
            @info "$progress_desc sample $saved/$n_samples"
    end

    return B.MHSamples(
        draws,
        lps,
        names,
        accept_count == 0 ? NaN : accept_sum / accept_count,
        NaN,
        fill(NaN, length(names)),
    )
end

function samples_to_array(samples::Vector{B.MHSamples})
    n_iter, n_param = size(samples[1].draws)
    arr = Array{Float64}(undef, n_iter, n_param + 1, length(samples))
    @inbounds for c in eachindex(samples)
        arr[:, 1:n_param, c] .= samples[c].draws
        arr[:, n_param + 1, c] .= samples[c].logposterior
    end
    return arr
end

function beta_draws(samples::Vector{B.MHSamples})
    names0 = samples[1].names
    p = count(startswith.(names0, "beta["))
    idx = [findfirst(==("beta[$j]"), names0) for j in 1:p]
    any(isnothing, idx) && throw(ArgumentError("missing ordered beta columns in samples."))
    return reduce(vcat, [s.draws[:, Int.(idx)] for s in samples])
end

function ess_rhat(samples::Vector{B.MHSamples})
    arr = samples_to_array(samples)
    names0 = Symbol.(vcat(samples[1].names, ["logposterior"]))
    chains = Chains(arr, names0)
    summ = DataFrame(summarystats(chains))
    keep = startswith.(string.(summ.parameters), "beta[")
    beta_summ = summ[keep, :]
    ess_col =
        :ess_bulk in propertynames(beta_summ) ? :ess_bulk :
        :ess in propertynames(beta_summ) ? :ess : nothing
    rhat_col = :rhat in propertynames(beta_summ) ? :rhat : nothing
    ess = ess_col === nothing ? Float64[] : Float64.(collect(skipmissing(beta_summ[!, ess_col])))
    rhat = rhat_col === nothing ? Float64[] : Float64.(collect(skipmissing(beta_summ[!, rhat_col])))
    return (
        Rhat_max = isempty(rhat) ? NaN : maximum(rhat),
        ESS_bulk_min = isempty(ess) ? NaN : minimum(ess),
    )
end

function posterior_interval_counts(draws::AbstractMatrix, beta_true::AbstractVector)
    lower = vec(mapslices(x -> quantile(x, 0.025), draws; dims = 1))
    upper = vec(mapslices(x -> quantile(x, 0.975), draws; dims = 1))
    truth = beta_true .!= 0
    selected = (lower .> 0) .| (upper .< 0)
    return (
        FP = count((.!truth) .& selected),
        FN = count(truth .& .!selected),
    )
end

function predictive_metrics(data::BCGLMData, beta_hat::AbstractVector)
    eta_hat = data.z * beta_hat
    if data.family == :gaussian
        mse_y = mean((data.y .- eta_hat) .^ 2)
        deviance = -2 * glm_loglik(data, beta_hat)
        return (pred_mse = mse_y, deviance = deviance, accuracy = NaN, auc = NaN)
    elseif data.family == :logistic
        prob = logistic_stable.(eta_hat)
        deviance = -2 * glm_loglik(data, beta_hat)
        pred = Float64.(prob .>= 0.5)
        return (
            pred_mse = mean((data.y .- prob) .^ 2),
            deviance = deviance,
            accuracy = mean(pred .== data.y),
            auc = auc_score(eta_hat, data.y),
        )
    end
    throw(ArgumentError("unsupported family: $(data.family)"))
end

function sample_one_fit(;
    train::BCGLMData,
    prior,
    samples::Integer,
    warmup::Integer,
    chains::Integer,
    seed::Integer,
    target_accept::Real,
    max_depth::Integer,
    progress::Bool,
    constraint::Symbol,
    soft_center_scale,
    prox_sum0_lambda::Real,
    prox_l1_lambda::Real,
    alpha_prior_shape,
    alpha_prior_rate,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
    use_car::Bool,
    car_distance::Symbol,
    car_weight_quantile::Real,
)
    car_weights =
        use_car ? B._car_weight_edges(B.car_weight_matrix(
            train.X;
            distance = car_distance,
            threshold_quantile = car_weight_quantile,
        )) : nothing
    p = size(train.z, 2)
    soft_scale = soft_center_scale === nothing ? B._default_soft_center_scale(p) : Float64(soft_center_scale)
    if prior isa B.SpikeSlabLassoPrior && prior.proximal
        a_shape, a_rate = prox_alpha_hyperparameters(prior, alpha_prior_shape, alpha_prior_rate)
        out = B.MHSamples[]
        for c in 1:chains
            push!(
                out,
                sample_glm_ssl_prox_nuts_gibbs_one(
                    train,
                    prior;
                    n_samples = samples,
                    warmup = warmup,
                    rng = MersenneTwister(seed + 10_007c),
                    target_accept = target_accept,
                    max_depth = max_depth,
                    constraint = constraint,
                    soft_center_scale = soft_scale,
                    prox_sum0_lambda = prox_sum0_lambda,
                    prox_l1_lambda = prox_l1_lambda,
                    alpha_prior_shape = a_shape,
                    alpha_prior_rate = a_rate,
                    prox_root_tol = prox_root_tol,
                    prox_root_maxiter = prox_root_maxiter,
                    progress = progress,
                    progress_desc = "BCGLM SSLASSO prox NUTS-Gibbs chain $c/$chains",
                ),
            )
        end
        return out
    end
    target =
        prior isa B.BayesianLassoPrior && prior.proximal ? begin
            a_shape, a_rate = prox_alpha_hyperparameters(prior, alpha_prior_shape, alpha_prior_rate)
            GLMProxLassoTarget(
                train,
                prior;
                constraint = constraint,
                soft_center_scale = soft_scale,
                prox_sum0_lambda = prox_sum0_lambda,
                prox_l1_lambda = prox_l1_lambda,
                alpha_prior_shape = a_shape,
                alpha_prior_rate = a_rate,
                prox_root_tol = prox_root_tol,
                prox_root_maxiter = prox_root_maxiter,
            )
        end :
        GLMPosteriorTarget(
            train,
            prior;
            constraint = constraint,
            soft_center_scale = soft_center_scale,
            prox_sum0_lambda = prox_sum0_lambda,
            car_weights = car_weights,
            use_car = use_car,
        )
    out = B.MHSamples[]
    for c in 1:chains
        push!(
            out,
            B._sample_explicit_nuts_one(
                target;
                n_samples = samples,
                warmup = warmup,
                rng = MersenneTwister(seed + 10_007c),
                target_accept = target_accept,
                max_depth = max_depth,
                progress = progress,
                progress_desc = "BCGLM explicit NUTS chain $c/$chains",
            ),
        )
    end
    return out
end

function summarize_fit(;
    train::BCGLMData,
    valid::BCGLMData,
    posterior::Vector{B.MHSamples},
    family::Symbol,
    prior_name::AbstractString,
    n::Integer,
    p::Integer,
    nonzero::Integer,
    p0::Real,
    rep::Integer,
    seed::Integer,
    samples::Integer,
    warmup::Integer,
    chains::Integer,
    use_car::Bool,
    tau_scale_mode::AbstractString,
)
    draws = beta_draws(posterior)
    beta_hat = vec(mean(draws; dims = 1))
    beta_true = train.beta_true
    nz = beta_true .!= 0
    intervals = posterior_interval_counts(draws, beta_true)
    pred_train = predictive_metrics(train, beta_hat)
    pred_valid = predictive_metrics(valid, beta_hat)
    er = ess_rhat(posterior)
    ll_true = glm_loglik(train, beta_true)
    ll_zero = glm_loglik(train, zeros(p))
    return merge(
        (
            family = String(family),
            prior = String(prior_name),
            n = n,
            p = p,
            nonzero = nonzero,
            p0 = Float64(p0),
            rep = rep,
            seed = seed,
            samples = samples,
            warmup = warmup,
            chains = chains,
            use_car = use_car,
            tau_scale_mode = String(tau_scale_mode),
            y_mean = mean(train.y),
            ll_true_minus_zero = ll_true - ll_zero,
            coef_mse = mean((beta_hat .- beta_true) .^ 2),
            coef_mse_nonzero = mean((beta_hat[nz] .- beta_true[nz]) .^ 2),
            coef_mse_zero = mean((beta_hat[.!nz] .- beta_true[.!nz]) .^ 2),
            beta_hat_norm = norm(beta_hat),
            beta_true_norm = norm(beta_true),
            sum_beta_hat = sum(beta_hat),
            acceptance_rate = mean(s.acceptance_rate for s in posterior),
            train_pred_mse = pred_train.pred_mse,
            train_deviance = pred_train.deviance,
            train_accuracy = pred_train.accuracy,
            train_auc = pred_train.auc,
            valid_pred_mse = pred_valid.pred_mse,
            valid_deviance = pred_valid.deviance,
            valid_accuracy = pred_valid.accuracy,
            valid_auc = pred_valid.auc,
        ),
        intervals,
        er,
    )
end

function main()
    args = parse_cli_args()
    outdir = ensure_dir(arg(args, "outdir", joinpath("results", "check_bcglm_compositional_mse")))
    families = Symbol.(split_strings(arg(args, "families", "gaussian,logistic")))
    ps = split_ints(arg(args, "p", "100"))
    ns = split_ints(arg(args, "n", "200"))
    nonzeros = split_ints(arg(args, "nonzero", "6,12"))
    priors = split_strings(arg(args, "priors", "hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso,prox-lasso,prox-sslasso"))
    reps = arg(args, "reps", 1)
    samples = arg(args, "samples", 300)
    warmup = arg(args, "warmup", 150)
    chains = arg(args, "chains", 2)
    seed0 = arg(args, "seed", 20260516)
    sigma = arg(args, "sigma", 1.6)
    binary_mode = Symbol(arg(args, "binary-mode", "median"))
    p0_mode = arg(args, "p0", "truth")
    target_accept = arg(args, "target-accept", 0.8)
    max_depth = arg(args, "max-depth", 8)
    progress = arg(args, "progress", false)
    constraint = Symbol(arg(args, "constraint", "soft_center"))
    soft_center_scale_arg = arg(args, "soft-center-scale", "auto")
    soft_center_scale = lowercase(soft_center_scale_arg) == "auto" ? nothing : parse(Float64, soft_center_scale_arg)
    prox_sum0_lambda = arg(args, "prox-sum0-lambda", 1e-3)
    prox_l1_lambda = arg(args, "prox-l1-lambda", 0.1)
    alpha_prior_shape = optional_float_arg(args, "alpha-prior-shape", "prox-alpha-prior-shape")
    alpha_prior_rate = optional_float_arg(args, "alpha-prior-rate", "prox-alpha-prior-rate")
    prox_root_tol = arg(args, "prox-root-tol", 1e-10)
    prox_root_maxiter = arg(args, "prox-root-maxiter", 100)
    use_cars = [lowercase(x) in ("true", "1", "yes", "y") for x in split_strings(arg(args, "use-car", "false"))]
    car_distance = Symbol(arg(args, "car-distance", "bray"))
    car_weight_quantile = arg(args, "car-weight-quantile", 0.75)
    prior_hyperparams = (
        hs_tau_scale = something(optional_float_arg(args, "hs-tau-scale"), 1.0),
        hs_local_scale = something(optional_float_arg(args, "hs-local-scale"), 1.0),
        rhs_tau_scale = something(optional_float_arg(args, "rhs-tau-scale"), 1.0),
        rhs_local_scale = something(optional_float_arg(args, "rhs-local-scale"), 1.0),
        rhs_slab_df = something(optional_float_arg(args, "rhs-slab-df", "rhs-c2-df"), 4.0),
        rhs_slab_s2 = something(optional_float_arg(args, "rhs-slab-s2", "rhs-c2-s2"), 2.0),
        rhs_pseudo_sigma_df = something(optional_float_arg(args, "rhs-pseudo-sigma-df"), 3.0),
        rhs_pseudo_sigma_scale = something(optional_float_arg(args, "rhs-pseudo-sigma-scale"), 10.0),
        lasso_lambda = optional_float_arg(args, "lasso-lambda"),
        lasso_lambda2_shape = something(optional_float_arg(args, "lasso-lambda2-shape"), 1.0),
        lasso_lambda2_rate = something(optional_float_arg(args, "lasso-lambda2-rate"), 0.1),
        lasso_tau2_shape = something(optional_float_arg(args, "lasso-tau2-shape", "lasso-local-shape"), 1.0),
        sslasso_spike_scale = something(optional_float_arg(args, "sslasso-spike-scale", "sslasso-s0"), 0.04),
        sslasso_slab_scale = something(optional_float_arg(args, "sslasso-slab-scale", "sslasso-s1"), 0.5),
        sslasso_theta_a = something(optional_float_arg(args, "sslasso-theta-a"), 1.0),
        sslasso_theta_b = something(optional_float_arg(args, "sslasso-theta-b"), 1.0),
    )
    prior_hypermeta = merge(
        prior_hyperparams,
        (
            lasso_lambda = prior_hyperparams.lasso_lambda === nothing ? missing : prior_hyperparams.lasso_lambda,
            prox_l1_lambda = prox_l1_lambda,
            alpha_prior_shape = alpha_prior_shape === nothing ? missing : alpha_prior_shape,
            alpha_prior_rate = alpha_prior_rate === nothing ? missing : alpha_prior_rate,
        ),
    )
    haskey(args, "tau-scale-mode") &&
        throw(ArgumentError("--tau-scale-mode was removed; HS/RHS tau scale is fixed at tau0."))
    tau_mode = "tau0"

    rows = DataFrame()
    for family in families, n in ns, p in ps, nonzero in nonzeros
        nonzero <= p || continue
        p0s = split_float_or_truth(p0_mode, nonzero)
        for rep in 1:reps
            seed = seed0 + 100_000rep + 1000p + 10nonzero + (family == :gaussian ? 1 : 2)
            train = simulate_bcglm_data(
                family = family,
                n = n,
                p = p,
                nonzero = nonzero,
                seed = seed,
                sigma = sigma,
                binary_mode = binary_mode,
            )
            valid = simulate_bcglm_data(
                family = family,
                n = n,
                p = p,
                nonzero = nonzero,
                seed = seed + 50_000_000,
                sigma = sigma,
                binary_mode = binary_mode,
            )
            CSV.write(
                joinpath(outdir, "truth_$(family)_n$(n)_p$(p)_nz$(nonzero)_rep$(rep).csv"),
                DataFrame(variable = ["beta[$j]" for j in 1:p], true_beta = train.beta_true, nonzero = train.beta_true .!= 0),
            )
            for p0 in p0s, use_car in use_cars, prior_name in priors
                prior = make_prior(prior_name, p0, family, sigma; prior_hyperparams...)
                posterior = sample_one_fit(
                    train = train,
                    prior = prior,
                    samples = samples,
                    warmup = warmup,
                    chains = chains,
                    seed = seed + round(Int, 100p0),
                    target_accept = target_accept,
                    max_depth = max_depth,
                    progress = progress,
                    constraint = constraint,
                    soft_center_scale = soft_center_scale,
                    prox_sum0_lambda = prox_sum0_lambda,
                    prox_l1_lambda = prox_l1_lambda,
                    alpha_prior_shape = alpha_prior_shape,
                    alpha_prior_rate = alpha_prior_rate,
                    prox_root_tol = prox_root_tol,
                    prox_root_maxiter = prox_root_maxiter,
                    use_car = use_car,
                    car_distance = car_distance,
                    car_weight_quantile = car_weight_quantile,
                )
                row = summarize_fit(
                    train = train,
                    valid = valid,
                    posterior = posterior,
                    family = family,
                    prior_name = prior_name,
                    n = n,
                    p = p,
                    nonzero = nonzero,
                    p0 = p0,
                    rep = rep,
                    seed = seed,
                    samples = samples,
                    warmup = warmup,
                    chains = chains,
                    use_car = use_car,
                    tau_scale_mode = tau_mode,
                )
                row = merge(row, prior_hypermeta)
                push!(rows, row; cols = :union)
                CSV.write(joinpath(outdir, "bcglm_mse_metrics.csv"), rows)
                println(
                    "BCGLM row family=$family n=$n p=$p nz=$nonzero prior=$prior_name " *
                    "p0=$p0 car=$use_car tau=$tau_mode coef_mse=$(row.coef_mse)",
                )
            end
        end
    end

    if !isempty(rows)
        hyper_cols = [k for k in keys(prior_hypermeta) if k in propertynames(rows)]
        group_cols = vcat([:family, :prior, :n, :p, :nonzero, :p0, :use_car, :tau_scale_mode], hyper_cols)
        summary = combine(
            groupby(rows, group_cols),
            :coef_mse => mean => :coef_mse_mean,
            :coef_mse => std => :coef_mse_sd,
            :coef_mse_nonzero => mean => :coef_mse_nonzero_mean,
            :coef_mse_zero => mean => :coef_mse_zero_mean,
            :valid_pred_mse => mean => :valid_pred_mse_mean,
            :valid_auc => mean => :valid_auc_mean,
            :FP => mean => :FP_mean,
            :FN => mean => :FN_mean,
            :Rhat_max => maximum => :Rhat_max,
            :ESS_bulk_min => minimum => :ESS_bulk_min,
        )
        CSV.write(joinpath(outdir, "bcglm_mse_summary.csv"), summary)
        show(summary; allrows = true, allcols = true)
        println()
    end
    println("BCGLM compositional MSE diagnostics written to $(abspath(outdir))")
end

main()
