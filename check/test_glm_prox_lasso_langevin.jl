#!/usr/bin/env julia

using Pkg

function _activate_repo_project()
    project = abspath(joinpath(@__DIR__, "..", "Project.toml"))
    active = Base.active_project()
    active !== project && Pkg.activate(dirname(project))
end

_activate_repo_project()

using BayesCox4Micro
using CSV
using DataFrames
using LinearAlgebra
using MCMCChains
using Plots
using Random
using Statistics

const B = BayesCox4Micro

function parse_cli_args(args = ARGS)
    out = Dict{String,String}()
    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--")
            key = arg[3:end]
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

split_floats(s::AbstractString) = parse.(Float64, strip.(split(s, ","; keepempty = false)))

ensure_dir(path::AbstractString) = (mkpath(path); path)

safe_label(label::AbstractString) =
    replace(lowercase(String(label)), r"[^a-z0-9_.-]+" => "_")

function log1pexp_stable(x::Real)
    xf = Float64(x)
    return xf > 0 ? xf + log1p(exp(-xf)) : log1p(exp(xf))
end

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

struct LogisticGLMData
    x::Matrix{Float64}
    y::Vector{Float64}
    beta_true::Vector{Float64}
end

mutable struct GLMProxWorkspace
    eta::Vector{Float64}
    prob::Vector{Float64}
    prox_beta::Vector{Float64}
end

function GLMProxWorkspace(n::Integer, p::Integer)
    return GLMProxWorkspace(
        Vector{Float64}(undef, n),
        Vector{Float64}(undef, n),
        Vector{Float64}(undef, p),
    )
end

function standardize_columns(x::AbstractMatrix)
    z = Matrix{Float64}(x)
    mu = vec(mean(z; dims = 1))
    sigma = vec(std(z; dims = 1))
    sigma .= ifelse.(sigma .> 0, sigma, 1.0)
    z .= (z .- mu') ./ sigma'
    return z
end

function simulate_logistic_glm(; n::Integer, p::Integer, nonzero::Integer, seed::Integer, beta_scale::Real)
    rng = MersenneTwister(seed)
    x = standardize_columns(randn(rng, n, p))
    beta = zeros(Float64, p)
    base = [1.2, -1.0, 0.85, -0.7, 0.55, -0.45, 0.35, -0.3]
    for j in 1:min(nonzero, p)
        beta[j] = beta_scale * (j <= length(base) ? base[j] : (-1)^j * 0.25)
    end
    prob = logistic_stable.(x * beta)
    y = Float64.(rand(rng, n) .< prob)
    return LogisticGLMData(x, y, beta)
end

function glm_loglik(data::LogisticGLMData, beta::AbstractVector)
    eta = data.x * beta
    ll = zero(eltype(eta))
    @inbounds for i in eachindex(eta)
        ll += data.y[i] * eta[i] - log1pexp_stable(eta[i])
    end
    return Float64(ll)
end

function glm_score_gradient!(grad::AbstractVector{Float64}, data::LogisticGLMData, beta::AbstractVector, work::GLMProxWorkspace)
    n, p = size(data.x)
    length(beta) == p || throw(DimensionMismatch("beta length must match number of columns."))
    length(grad) == p || throw(DimensionMismatch("gradient length must match beta."))
    mul!(work.eta, data.x, beta)
    @inbounds for i in 1:n
        work.prob[i] = logistic_stable(work.eta[i])
    end
    fill!(grad, 0.0)
    @inbounds for j in 1:p
        s = 0.0
        for i in 1:n
            s += data.x[i, j] * (data.y[i] - work.prob[i])
        end
        grad[j] = s
    end
    return grad
end

function prox_lasso_logpost(
    u::AbstractVector,
    data::LogisticGLMData;
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_rate::Real,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    p = size(data.x, 2)
    length(u) == p + 1 || throw(DimensionMismatch("state length must be p + 1."))
    beta = view(u, 1:p)
    eta_alpha = Float64(u[p + 1])
    alpha = exp(clamp(eta_alpha, -700.0, 700.0))
    lp = glm_loglik(data, beta)
    lp -= B.l1_moreau(
        beta,
        alpha,
        prox_l1_lambda;
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    lp += -Float64(alpha_prior_rate) * exp(-eta_alpha) - Float64(alpha_prior_shape) * eta_alpha
    return lp
end

function prox_lasso_gradient!(
    grad::AbstractVector{Float64},
    u::AbstractVector,
    data::LogisticGLMData,
    work::GLMProxWorkspace;
    prox_l1_lambda::Real,
    alpha_prior_shape::Real,
    alpha_prior_rate::Real,
    prox_root_tol::Real,
    prox_root_maxiter::Integer,
)
    p = size(data.x, 2)
    length(u) == p + 1 || throw(DimensionMismatch("state length must be p + 1."))
    length(grad) == p + 1 || throw(DimensionMismatch("gradient length must be p + 1."))
    beta = view(u, 1:p)
    eta_alpha = Float64(u[p + 1])
    alpha, alpha_deriv = B._clamped_exp_with_derivative(eta_alpha, -700.0, 700.0)
    lambda_l1 = Float64(prox_l1_lambda)
    lambda_l1 > 0 || throw(ArgumentError("prox_l1_lambda must be positive."))
    prox = B.project_l1_epigraph!(
        work.prox_beta,
        beta,
        alpha;
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    glm_score_gradient!(view(grad, 1:p), data, beta, work)
    @inbounds for j in 1:p
        grad[j] -= (Float64(beta[j]) - prox.beta[j]) / lambda_l1
    end
    grad[p + 1] = Float64(alpha_prior_rate) * exp(-eta_alpha) - Float64(alpha_prior_shape) -
        alpha_deriv * (Float64(alpha) - Float64(prox.alpha)) / lambda_l1
    return grad
end

function prox_lasso_gradient(u::AbstractVector, data::LogisticGLMData; kwargs...)
    work = GLMProxWorkspace(size(data.x, 1), size(data.x, 2))
    grad = zeros(Float64, length(u))
    return prox_lasso_gradient!(grad, u, data, work; kwargs...)
end

function chains_array(chains)
    arr = Array(getfield(chains, :value))
    ndims(arr) == 2 && (arr = reshape(arr, size(arr, 1), size(arr, 2), 1))
    return arr
end

function chains_from_samples(samples::Vector{MHSamples})
    n_iter, n_param = size(samples[1].draws)
    n_chain = length(samples)
    arr = Array{Float64}(undef, n_iter, n_param + 1, n_chain)
    for c in 1:n_chain
        arr[:, 1:n_param, c] .= samples[c].draws
        arr[:, n_param + 1, c] .= samples[c].logposterior
    end
    return Chains(arr, Symbol.(vcat(samples[1].names, ["logposterior"])))
end

function lag_autocorr(x::AbstractVector, lag::Integer)
    n = length(x)
    lag <= 0 && return 1.0
    n <= lag && return NaN
    mu = mean(x)
    den = sum(abs2, x .- mu)
    den <= eps(Float64) && return NaN
    return sum((view(x, 1:(n - lag)) .- mu) .* (view(x, (lag + 1):n) .- mu)) / den
end

function ess_rhat(chains; prefix::AbstractString = "beta[")
    summary = DataFrame(summarystats(chains))
    keep = startswith.(string.(summary.parameters), prefix)
    beta_summary = summary[keep, :]
    ess_col =
        :ess_bulk in propertynames(beta_summary) ? :ess_bulk :
        :ess in propertynames(beta_summary) ? :ess : nothing
    rhat_col = :rhat in propertynames(beta_summary) ? :rhat : nothing
    ess = ess_col === nothing ? Float64[] : Float64.(collect(skipmissing(beta_summary[!, ess_col])))
    rhat = rhat_col === nothing ? Float64[] : Float64.(collect(skipmissing(beta_summary[!, rhat_col])))
    return (
        ess_bulk_min = isempty(ess) ? NaN : minimum(ess),
        ess_bulk_median = isempty(ess) ? NaN : median(ess),
        rhat_max = isempty(rhat) ? NaN : maximum(rhat),
        rhat_median = isempty(rhat) ? NaN : median(rhat),
    )
end

function predictive_metrics(data::LogisticGLMData, beta_mean::AbstractVector)
    prob = logistic_stable.(data.x * beta_mean)
    pred = prob .>= 0.5
    y_bool = data.y .>= 0.5
    acc = mean(pred .== y_bool)
    loglik = glm_loglik(data, beta_mean)
    return (accuracy = acc, loglik_at_mean = loglik)
end

function summarize_run(label::AbstractString, sampler::Symbol, step::Real, samples::Vector{MHSamples}, data::LogisticGLMData)
    chains = chains_from_samples(samples)
    arr = chains_array(chains)
    p = size(data.x, 2)
    beta_arr = arr[:, 1:p, :]
    beta_vals = reshape(permutedims(beta_arr, (1, 3, 2)), :, p)
    beta_mean = vec(mean(beta_vals; dims = 1))
    beta_sd = vec(std(beta_vals; dims = 1))
    acf1 = [lag_autocorr(view(arr, :, j, c), 1) for j in 1:p for c in axes(arr, 3)]
    acf10 = [lag_autocorr(view(arr, :, j, c), 10) for j in 1:p for c in axes(arr, 3)]
    finite_fraction = count(isfinite, beta_arr) / length(beta_arr)
    er = ess_rhat(chains)
    pred = predictive_metrics(data, beta_mean)
    accept = sampler == :mala ? mean(s.acceptance_rate for s in samples) : NaN
    warmup_accept = sampler == :mala ? mean(s.warmup_acceptance_rate for s in samples) : NaN
    steps = reduce(vcat, [s.final_step for s in samples])
    return merge(
        (
            label = String(label),
            sampler = String(sampler),
            step_size = Float64(step),
            chains = length(samples),
            samples_per_chain = size(samples[1].draws, 1),
            p = p,
            acceptance_rate = accept,
            warmup_acceptance_rate = warmup_accept,
            final_step_min = minimum(steps),
            final_step_median = median(steps),
            final_step_max = maximum(steps),
            beta_mean_abs_error = mean(abs.(beta_mean .- data.beta_true)),
            beta_max_abs_error = maximum(abs.(beta_mean .- data.beta_true)),
            beta_mean_l1 = sum(abs, beta_mean),
            beta_sd_median = median(beta_sd),
            lag1_acf_mean = mean(filter(isfinite, acf1)),
            lag10_acf_mean = mean(filter(isfinite, acf10)),
            finite_fraction = finite_fraction,
            has_nan_or_inf = finite_fraction < 1.0,
            max_abs_beta = maximum(abs, beta_arr),
            logposterior_first_mean = mean(arr[1, p + 2, :]),
            logposterior_last_mean = mean(arr[end, p + 2, :]),
        ),
        er,
        pred,
    )
end

function plot_run(label::AbstractString, chains, outdir::AbstractString; max_beta::Integer, max_lag::Integer)
    ensure_dir(outdir)
    arr = chains_array(chains)
    nms = string.(names(chains))
    p = count(startswith.(nms, "beta["))
    k = min(p, max_beta)
    slug = safe_label(label)

    trace = plot(title = "$label trace", xlabel = "iteration", ylabel = "value", size = (1000, 520))
    for j in 1:k, c in axes(arr, 3)
        plot!(trace, arr[:, j, c]; label = c == 1 ? "beta[$j]" : "", linewidth = 0.8, alpha = 0.75)
    end
    savefig(trace, joinpath(outdir, "$(slug)_trace.png"))

    lp_idx = p + 2
    lp = plot(title = "$label log posterior", xlabel = "iteration", ylabel = "log posterior", size = (1000, 420))
    for c in axes(arr, 3)
        plot!(lp, arr[:, lp_idx, c]; label = "chain $c", linewidth = 0.9)
    end
    savefig(lp, joinpath(outdir, "$(slug)_logposterior.png"))

    lags = collect(0:max_lag)
    acf_plot = plot(title = "$label ACF", xlabel = "lag", ylabel = "autocorrelation", size = (900, 480))
    for j in 1:k
        vals = [mean(filter(isfinite, [lag_autocorr(view(arr, :, j, c), lag) for c in axes(arr, 3)])) for lag in lags]
        plot!(acf_plot, lags, vals; label = "beta[$j]", linewidth = 1.3)
    end
    savefig(acf_plot, joinpath(outdir, "$(slug)_acf.png"))
    return nothing
end

function finite_difference_gradient(f::Function, u::AbstractVector{Float64}; step::Real = 1e-5)
    grad = similar(u)
    work = copy(u)
    h = Float64(step)
    for j in eachindex(u)
        old = work[j]
        work[j] = old + h
        fp = f(work)
        work[j] = old - h
        fm = f(work)
        work[j] = old
        grad[j] = (fp - fm) / (2h)
    end
    return grad
end

function gradient_check(data::LogisticGLMData; prox_l1_lambda, alpha_prior_shape, alpha_prior_rate, prox_root_tol, prox_root_maxiter)
    rng = MersenneTwister(9001)
    p = size(data.x, 2)
    u = vcat(0.05 .* randn(rng, p), [log(1.0)])
    work = GLMProxWorkspace(size(data.x, 1), p)
    grad = zeros(Float64, p + 1)
    prox_lasso_gradient!(
        grad,
        u,
        data,
        work;
        prox_l1_lambda = prox_l1_lambda,
        alpha_prior_shape = alpha_prior_shape,
        alpha_prior_rate = alpha_prior_rate,
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    logpost = x -> prox_lasso_logpost(
            x,
            data;
            prox_l1_lambda = prox_l1_lambda,
            alpha_prior_shape = alpha_prior_shape,
            alpha_prior_rate = alpha_prior_rate,
            prox_root_tol = prox_root_tol,
            prox_root_maxiter = prox_root_maxiter,
        )
    grad_fd = finite_difference_gradient(logpost, u)
    max_abs = maximum(abs.(grad .- grad_fd))
    return DataFrame(
        max_abs_gradient_error = [max_abs],
        relative_gradient_error = [max_abs / max(1.0, maximum(abs.(grad_fd)))],
        logposterior = [prox_lasso_logpost(
            u,
            data;
            prox_l1_lambda = prox_l1_lambda,
            alpha_prior_shape = alpha_prior_shape,
            alpha_prior_rate = alpha_prior_rate,
            prox_root_tol = prox_root_tol,
            prox_root_maxiter = prox_root_maxiter,
        )],
        gradient_norm = [norm(grad)],
    )
end

function main()
    args = parse_cli_args()
    outdir = ensure_dir(arg(args, "outdir", joinpath("results", "glm_prox_lasso_langevin")))
    n = arg(args, "n", 300)
    p = arg(args, "p", 12)
    nonzero = arg(args, "nonzero", 4)
    samples = arg(args, "samples", 3000)
    warmup = arg(args, "warmup", 1500)
    thin = arg(args, "thin", 1)
    chains = arg(args, "chains", 4)
    seed = arg(args, "seed", 20260509)
    beta_scale = arg(args, "beta-scale", 1.0)
    prox_l1_lambda = arg(args, "prox-l1-lambda", 0.01)
    alpha_prior_shape = arg(args, "alpha-prior-shape", 1.0)
    alpha_prior_rate = arg(args, "alpha-prior-rate", 1.0)
    prox_root_tol = arg(args, "prox-root-tol", 1e-10)
    prox_root_maxiter = arg(args, "prox-root-maxiter", 100)
    mala_steps = split_floats(arg(args, "mala-steps", "0.0005,0.001,0.002,0.005"))
    ula_steps = split_floats(arg(args, "ula-steps", "0.0002,0.0005,0.001,0.002"))
    mala_target_accept = arg(args, "mala-target-accept", 0.574)
    mala_adapt_every = arg(args, "mala-adapt-every", 50)
    max_lag = arg(args, "max-lag", 50)
    max_plot_beta = arg(args, "max-plot-beta", 6)

    data = simulate_logistic_glm(n = n, p = p, nonzero = nonzero, seed = seed, beta_scale = beta_scale)
    CSV.write(
        joinpath(outdir, "data_summary.csv"),
        DataFrame(
            n = [n],
            p = [p],
            nonzero = [nonzero],
            y_mean = [mean(data.y)],
            prox_l1_lambda = [prox_l1_lambda],
            alpha_prior_shape = [alpha_prior_shape],
            alpha_prior_rate = [alpha_prior_rate],
        ),
    )
    CSV.write(joinpath(outdir, "truth.csv"), DataFrame(parameter = ["beta[$j]" for j in 1:p], value = data.beta_true))

    gc = gradient_check(
        data;
        prox_l1_lambda = prox_l1_lambda,
        alpha_prior_shape = alpha_prior_shape,
        alpha_prior_rate = alpha_prior_rate,
        prox_root_tol = prox_root_tol,
        prox_root_maxiter = prox_root_maxiter,
    )
    CSV.write(joinpath(outdir, "gradient_check.csv"), gc)

    rows = DataFrame()
    names0 = vcat(["beta[$j]" for j in 1:p], ["eta"])
    for sampler in (:mala, :ula)
        steps = sampler == :mala ? mala_steps : ula_steps
        for step in steps
            label = "glm_prox_lasso_$(sampler)_step_$(step)"
            chain_samples = MHSamples[]
            for c in 1:chains
                rng = MersenneTwister(seed + 1009c + round(Int, 1_000_000step))
                work = GLMProxWorkspace(n, p)
                logpost = u -> prox_lasso_logpost(
                    u,
                    data;
                    prox_l1_lambda = prox_l1_lambda,
                    alpha_prior_shape = alpha_prior_shape,
                    alpha_prior_rate = alpha_prior_rate,
                    prox_root_tol = prox_root_tol,
                    prox_root_maxiter = prox_root_maxiter,
                )
                gradient! = (grad, u) -> prox_lasso_gradient!(
                    grad,
                    u,
                    data,
                    work;
                    prox_l1_lambda = prox_l1_lambda,
                    alpha_prior_shape = alpha_prior_shape,
                    alpha_prior_rate = alpha_prior_rate,
                    prox_root_tol = prox_root_tol,
                    prox_root_maxiter = prox_root_maxiter,
                )
                gradient = u -> prox_lasso_gradient(
                    u,
                    data;
                    prox_l1_lambda = prox_l1_lambda,
                    alpha_prior_shape = alpha_prior_shape,
                    alpha_prior_rate = alpha_prior_rate,
                    prox_root_tol = prox_root_tol,
                    prox_root_maxiter = prox_root_maxiter,
                )
                init = vcat(zeros(Float64, p), [log(1.0)])
                sample =
                    sampler == :mala ?
                    B._run_full_mala(
                        logpost,
                        gradient,
                        init,
                        names0;
                        n_samples = samples,
                        warmup = warmup,
                        thin = thin,
                        rng = rng,
                        mala_step_size = step,
                        mala_target_accept = mala_target_accept,
                        mala_adapt_every = mala_adapt_every,
                        mala_min_step = step / 100,
                        mala_max_step = step * 100,
                        progress = false,
                        gradient! = gradient!,
                    ) :
                    B._run_full_ula(
                        logpost,
                        gradient,
                        init,
                        names0;
                        n_samples = samples,
                        warmup = warmup,
                        thin = thin,
                        rng = rng,
                        mala_step_size = step,
                        progress = false,
                        gradient! = gradient!,
                    )
                push!(chain_samples, sample)
            end
            chains_obj = chains_from_samples(chain_samples)
            row = summarize_run(label, sampler, step, chain_samples, data)
            push!(rows, row; cols = :union)
            CSV.write(joinpath(outdir, "metrics.csv"), rows)
            CSV.write(joinpath(outdir, "$(safe_label(label))_summary.csv"), DataFrame(summarystats(chains_obj)))
            plot_run(label, chains_obj, outdir; max_beta = max_plot_beta, max_lag = max_lag)
        end
    end

    println("GLM prox-lasso Langevin diagnostics written to $(abspath(outdir))")
    println("Gradient max abs error = ", gc.max_abs_gradient_error[1])
    show(rows; allrows = true, allcols = true)
    println()
end

main()
