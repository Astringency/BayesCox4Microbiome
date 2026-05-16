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

function make_prior(name::AbstractString, p0::Real, family::Symbol, sigma::Real)
    key = lowercase(replace(name, "_" => "-", " " => "-"))
    if key in ("rhs", "regularized-horseshoe", "bcglm-rhs")
        return B.RegularizedHorseshoePrior(
            use_cox_tau0 = false,
            p0 = Float64(p0),
            tau_formula = :linear,
            pseudo_sigma = family == :gaussian ? Float64(sigma) : 1.0,
            sample_pseudo_sigma = false,
            slab_df = 4.0,
            slab_s2 = 2.0,
        )
    elseif key in ("rhs-sigma", "rhs-sample-sigma")
        return B.RegularizedHorseshoePrior(
            use_cox_tau0 = false,
            p0 = Float64(p0),
            tau_formula = :linear,
            pseudo_sigma = nothing,
            sample_pseudo_sigma = true,
            pseudo_sigma_df = 3.0,
            pseudo_sigma_scale = 10.0,
            slab_df = 4.0,
            slab_s2 = 2.0,
        )
    elseif key in ("cox-rhs", "coxrhs")
        return B.RegularizedHorseshoePrior(
            use_cox_tau0 = true,
            p0 = Float64(p0),
            tau_formula = :cox,
            slab_df = 4.0,
            slab_s2 = 2.0,
        )
    elseif key in ("lasso", "bayesian-lasso")
        return B.BayesianLassoPrior(lambda = nothing, lambda2_shape = 1.0, lambda2_rate = 0.1)
    elseif key in ("sslasso", "spike-slab", "spike-and-slab")
        return B.SpikeSlabLassoPrior(spike_scale = 0.04, slab_scale = 0.5)
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

B.dimension(target::GLMPosteriorTarget) = length(target.names)
B.state_names(target::GLMPosteriorTarget) = target.names

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
    target = GLMPosteriorTarget(
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
    priors = split_strings(arg(args, "priors", "rhs,lasso,sslasso"))
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
    use_cars = [lowercase(x) in ("true", "1", "yes", "y") for x in split_strings(arg(args, "use-car", "false"))]
    car_distance = Symbol(arg(args, "car-distance", "bray"))
    car_weight_quantile = arg(args, "car-weight-quantile", 0.75)
    tau_modes = split_strings(arg(args, "tau-scale-mode", get(ENV, "BAYESCOX_TAU_CAUCHY_SCALE", "tau0_squared")))

    rows = DataFrame()
    for tau_mode in tau_modes
        old_tau = get(ENV, "BAYESCOX_TAU_CAUCHY_SCALE", nothing)
        ENV["BAYESCOX_TAU_CAUCHY_SCALE"] = tau_mode
        try
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
                        prior = make_prior(prior_name, p0, family, sigma)
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
                        push!(rows, row; cols = :union)
                        CSV.write(joinpath(outdir, "bcglm_mse_metrics.csv"), rows)
                        println(
                            "BCGLM row family=$family n=$n p=$p nz=$nonzero prior=$prior_name " *
                            "p0=$p0 car=$use_car tau=$tau_mode coef_mse=$(row.coef_mse)",
                        )
                    end
                end
            end
        finally
            if old_tau === nothing
                delete!(ENV, "BAYESCOX_TAU_CAUCHY_SCALE")
            else
                ENV["BAYESCOX_TAU_CAUCHY_SCALE"] = old_tau
            end
        end
    end

    if !isempty(rows)
        group_cols = [:family, :prior, :n, :p, :nonzero, :p0, :use_car, :tau_scale_mode]
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
