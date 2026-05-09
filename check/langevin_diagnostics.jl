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
using ForwardDiff
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

function split_symbols(s::AbstractString)
    return Symbol.(strip.(split(s, ","; keepempty = false)))
end

function split_floats(s::AbstractString)
    return parse.(Float64, strip.(split(s, ","; keepempty = false)))
end

function split_strings(s::AbstractString)
    return strip.(split(s, ","; keepempty = false))
end

function safe_label(label::AbstractString)
    return replace(lowercase(String(label)), r"[^a-z0-9_.-]+" => "_")
end

function ensure_dir(path::AbstractString)
    mkpath(path)
    return path
end

normal_logpost(x) = -0.5 * sum(abs2, x)

function normal_gradient!(grad::AbstractVector{Float64}, x::AbstractVector)
    length(grad) == length(x) || throw(DimensionMismatch("normal gradient length mismatch."))
    @inbounds for j in eachindex(x)
        grad[j] = -Float64(x[j])
    end
    return grad
end

normal_gradient(x) = -collect(Float64, x)

function _chains_from_draws(draws_by_chain::Vector{Matrix{Float64}}, lps_by_chain::Vector{Vector{Float64}}, names::Vector{String})
    n_iter, n_param = size(draws_by_chain[1])
    n_chain = length(draws_by_chain)
    arr = Array{Float64}(undef, n_iter, n_param + 1, n_chain)
    for c in 1:n_chain
        size(draws_by_chain[c]) == (n_iter, n_param) ||
            throw(DimensionMismatch("all chains must have identical draw shapes."))
        length(lps_by_chain[c]) == n_iter ||
            throw(DimensionMismatch("log posterior length must match draws."))
        arr[:, 1:n_param, c] .= draws_by_chain[c]
        arr[:, n_param + 1, c] .= lps_by_chain[c]
    end
    return Chains(arr, Symbol.(vcat(names, ["logposterior"])))
end

function samples_to_chains(samples::Vector{MHSamples})
    draws = [s.draws for s in samples]
    lps = [s.logposterior for s in samples]
    return _chains_from_draws(draws, lps, samples[1].names)
end

function result_to_chain_object(result::CoxSamplerResult)
    return result_to_chains(result)
end

function _chains_array(chains)
    arr = Array(getfield(chains, :value))
    ndims(arr) == 2 && (arr = reshape(arr, size(arr, 1), size(arr, 2), 1))
    return arr
end

function finite_fraction(arr)
    values = vec(arr)
    isempty(values) && return NaN
    return count(isfinite, values) / length(values)
end

function lag_autocorr(x::AbstractVector, lag::Integer)
    n = length(x)
    lag <= 0 && return 1.0
    n <= lag && return NaN
    x1 = view(x, 1:(n - lag))
    x2 = view(x, (lag + 1):n)
    mu = mean(x)
    den = sum(abs2, x .- mu)
    den <= eps(Float64) && return NaN
    return sum((x1 .- mu) .* (x2 .- mu)) / den
end

function mean_lag_acf(arr::Array{Float64,3}, param_idx::AbstractVector{Int}, lag::Integer)
    vals = Float64[]
    for c in axes(arr, 3), j in param_idx
        ac = lag_autocorr(view(arr, :, j, c), lag)
        isfinite(ac) && push!(vals, ac)
    end
    return isempty(vals) ? NaN : mean(vals)
end

function min_bulk_ess_and_rhat(chains, param_prefix::AbstractString)
    summary = DataFrame(summarystats(chains))
    if hasproperty(summary, :parameters)
        keep = startswith.(string.(summary.parameters), param_prefix)
        summary = summary[keep, :]
    end
    ess_col =
        :ess_bulk in propertynames(summary) ? :ess_bulk :
        :ess in propertynames(summary) ? :ess : nothing
    rhat_col = :rhat in propertynames(summary) ? :rhat : nothing
    ess = ess_col === nothing ? Float64[] : Float64.(collect(skipmissing(summary[!, ess_col])))
    rhat = rhat_col === nothing ? Float64[] : Float64.(collect(skipmissing(summary[!, rhat_col])))
    return (
        ess_bulk_min = isempty(ess) ? NaN : minimum(ess),
        ess_bulk_median = isempty(ess) ? NaN : median(ess),
        rhat_max = isempty(rhat) ? NaN : maximum(rhat),
        rhat_median = isempty(rhat) ? NaN : median(rhat),
    )
end

function acceptance_rate(samples)
    if samples isa MHSamples
        return samples.acceptance_rate
    elseif samples isa MultiMHSamples
        rates = [s.acceptance_rate for s in samples.chains]
        return all(isnan, rates) ? NaN : mean(rates)
    end
    return NaN
end

function warmup_acceptance_rate(samples)
    if samples isa MHSamples
        return samples.warmup_acceptance_rate
    elseif samples isa MultiMHSamples
        rates = [s.warmup_acceptance_rate for s in samples.chains]
        return all(isnan, rates) ? NaN : mean(rates)
    end
    return NaN
end

function final_step_summary(samples)
    if samples isa MHSamples
        steps = samples.final_step
    elseif samples isa MultiMHSamples
        steps = reduce(vcat, [s.final_step for s in samples.chains])
    else
        return (final_step_min = NaN, final_step_median = NaN, final_step_max = NaN)
    end
    return (
        final_step_min = minimum(steps),
        final_step_median = median(steps),
        final_step_max = maximum(steps),
    )
end

function selected_param_indices(chains, prefix::AbstractString, max_params::Integer)
    nms = string.(names(chains))
    idx = findall(name -> startswith(name, prefix), nms)
    isempty(idx) && (idx = collect(1:min(max_params, length(nms))))
    return idx[1:min(length(idx), max_params)]
end

function plot_trace_and_acf(chains, outdir::AbstractString, label::AbstractString; prefix::AbstractString, max_params::Integer, max_lag::Integer)
    ensure_dir(outdir)
    arr = _chains_array(chains)
    nms = string.(names(chains))
    idx = selected_param_indices(chains, prefix, max_params)
    slug = safe_label(label)

    trace = plot(title = "$label trace", xlabel = "iteration", ylabel = "value", size = (1000, 520))
    for j in idx
        for c in axes(arr, 3)
            plot!(trace, arr[:, j, c]; label = c == 1 ? nms[j] : "", linewidth = 0.8, alpha = 0.75)
        end
    end
    savefig(trace, joinpath(outdir, "$(slug)_trace.png"))

    lp_idx = findfirst(==("logposterior"), nms)
    if lp_idx !== nothing
        lp_plot = plot(title = "$label log posterior", xlabel = "iteration", ylabel = "log posterior", size = (1000, 420))
        for c in axes(arr, 3)
            plot!(lp_plot, arr[:, lp_idx, c]; label = "chain $c", linewidth = 0.9)
        end
        savefig(lp_plot, joinpath(outdir, "$(slug)_logposterior.png"))
    end

    acf_plot = plot(title = "$label ACF", xlabel = "lag", ylabel = "autocorrelation", size = (900, 480))
    lags = collect(0:max_lag)
    for j in idx
        ac = [mean(skipmissing([lag_autocorr(view(arr, :, j, c), lag) for c in axes(arr, 3)])) for lag in lags]
        plot!(acf_plot, lags, ac; label = nms[j], linewidth = 1.4)
    end
    savefig(acf_plot, joinpath(outdir, "$(slug)_acf.png"))
    return nothing
end

function summarize_chains(chains; experiment::AbstractString, sampler::Symbol, step::Real, thin::Integer, target_mean = nothing, target_var = nothing, param_prefix::AbstractString = "x[")
    arr = _chains_array(chains)
    nms = string.(names(chains))
    param_idx = findall(name -> startswith(name, param_prefix), nms)
    isempty(param_idx) && (param_idx = collect(1:(size(arr, 2) - 1)))
    vals = reshape(permutedims(arr[:, param_idx, :], (1, 3, 2)), :, length(param_idx))
    means = vec(mean(vals; dims = 1))
    vars = vec(var(vals; dims = 1))
    ess_rhat = min_bulk_ess_and_rhat(chains, param_prefix)
    lp_idx = findfirst(==("logposterior"), nms)
    lp_values = lp_idx === nothing ? Float64[] : vec(arr[:, lp_idx, :])
    ref_mean_error =
        target_mean === nothing ? NaN :
        maximum(abs.(means .- collect(Float64, target_mean)[1:length(means)]))
    ref_var_error =
        target_var === nothing ? NaN :
        maximum(abs.(vars .- collect(Float64, target_var)[1:length(vars)]))
    return merge(
        (
            experiment = String(experiment),
            sampler = String(sampler),
            step_size = Float64(step),
            thin = thin,
            n_iter = size(arr, 1),
            chains = size(arr, 3),
            n_params = length(param_idx),
            max_abs_mean = maximum(abs.(means)),
            mean_abs_mean = mean(abs.(means)),
            mean_variance = mean(vars),
            min_variance = minimum(vars),
            max_variance = maximum(vars),
            reference_max_abs_mean_error = ref_mean_error,
            reference_max_abs_variance_error = ref_var_error,
            lag1_acf_mean = mean_lag_acf(arr, param_idx, 1),
            lag10_acf_mean = mean_lag_acf(arr, param_idx, 10),
            finite_fraction = finite_fraction(arr),
            has_nan_or_inf = finite_fraction(arr) < 1.0,
            max_abs_state = maximum(abs, arr[:, param_idx, :]),
            logposterior_first_mean = isempty(lp_values) ? NaN : mean(arr[1, lp_idx, :]),
            logposterior_last_mean = isempty(lp_values) ? NaN : mean(arr[end, lp_idx, :]),
            logposterior_min = isempty(lp_values) ? NaN : minimum(lp_values),
            logposterior_max = isempty(lp_values) ? NaN : maximum(lp_values),
        ),
        ess_rhat,
    )
end

function run_normal_scan(args, outdir::AbstractString)
    d = arg(args, "normal-dim", 10)
    n_samples = arg(args, "normal-samples", arg(args, "samples", 4000))
    warmup = arg(args, "normal-warmup", arg(args, "warmup", 1000))
    chains = arg(args, "chains", 4)
    thin = arg(args, "normal-thin", arg(args, "thin", 1))
    seed = arg(args, "seed", 1)
    mala_steps = split_floats(arg(args, "normal-mala-steps", "0.25,0.5,0.75"))
    ula_steps = split_floats(arg(args, "normal-ula-steps", "0.01,0.05,0.1,0.2"))
    max_lag = arg(args, "max-lag", 50)
    rows = DataFrame()
    names0 = ["x[$j]" for j in 1:d]
    normal_dir = ensure_dir(joinpath(outdir, "normal"))
    for sampler in (:mala, :ula)
        steps = sampler == :mala ? mala_steps : ula_steps
        for step in steps
            samples = MHSamples[]
            for c in 1:chains
                rng = MersenneTwister(seed + 10007c + round(Int, 1_000_000step))
                init = zeros(Float64, d)
                s =
                    sampler == :mala ?
                    B._run_full_mala(
                        normal_logpost,
                        normal_gradient,
                        init,
                        names0;
                        n_samples = n_samples,
                        warmup = warmup,
                        thin = thin,
                        rng = rng,
                        mala_step_size = step,
                        mala_target_accept = 0.574,
                        mala_adapt_every = 50,
                        mala_min_step = step / 100,
                        mala_max_step = step * 100,
                        progress = false,
                        gradient! = normal_gradient!,
                    ) :
                    B._run_full_ula(
                        normal_logpost,
                        normal_gradient,
                        init,
                        names0;
                        n_samples = n_samples,
                        warmup = warmup,
                        thin = thin,
                        rng = rng,
                        mala_step_size = step,
                        progress = false,
                        gradient! = normal_gradient!,
                    )
                push!(samples, s)
            end
            chains_obj = samples_to_chains(samples)
            theoretical_var = sampler == :ula ? fill(2 / (2 - step), d) : ones(d)
            row = merge(
                summarize_chains(
                    chains_obj;
                    experiment = "normal",
                    sampler = sampler,
                    step = step,
                    thin = thin,
                    target_mean = zeros(d),
                    target_var = theoretical_var,
                    param_prefix = "x[",
                ),
                (
                    acceptance_rate = sampler == :mala ? mean(s.acceptance_rate for s in samples) : NaN,
                    warmup_acceptance_rate = sampler == :mala ? mean(s.warmup_acceptance_rate for s in samples) : NaN,
                    final_step_min = minimum(minimum(s.final_step) for s in samples),
                    final_step_median = median(reduce(vcat, [s.final_step for s in samples])),
                    final_step_max = maximum(maximum(s.final_step) for s in samples),
                    target_variance = sampler == :ula ? 2 / (2 - step) : 1.0,
                ),
            )
            push!(rows, row; cols = :union)
            plot_trace_and_acf(chains_obj, normal_dir, "normal_$(sampler)_step_$(step)"; prefix = "x[", max_params = 4, max_lag = max_lag)
        end
    end
    CSV.write(joinpath(normal_dir, "metrics.csv"), rows)
    return rows
end

function simulate_cox_data(; n::Integer, p::Integer, nonzero::Integer, seed::Integer, event_fraction::Real)
    rng = MersenneTwister(seed)
    x_raw = exp.(0.35 .* randn(rng, n, p))
    z = Matrix{Float64}(x_raw)
    mu = vec(mean(z; dims = 1))
    sigma = vec(std(z; dims = 1))
    sigma .= ifelse.(sigma .> 0, sigma, 1.0)
    z .= (z .- mu') ./ sigma'
    beta = zeros(Float64, p)
    base = [0.9, -0.75, 0.6, -0.5, 0.42, -0.35, 0.3, -0.25]
    for j in 1:min(nonzero, p)
        beta[j] = j <= length(base) ? base[j] : (-1)^j * 0.2
    end
    eta = z * beta
    t_event = randexp(rng, n) ./ exp.(eta)
    censor_time = quantile(t_event, event_fraction)
    time = min.(t_event, censor_time)
    event = Int.(t_event .<= censor_time)
    data = prepare_cox_data(x_raw, time, event; transform = :none, standardize = true)
    return data, beta, x_raw, time, event
end

function cox_gaussian_logpost(data::CoxPreparedData, beta::AbstractVector, prior_sd::Real)
    return cox_logpartiallikelihood(data, beta) - 0.5 * sum(abs2, beta) / prior_sd^2
end

function cox_gaussian_gradient!(grad::AbstractVector{Float64}, data::CoxPreparedData, beta::AbstractVector, prior_sd::Real, work::CoxGradientWorkspace)
    cox_score_gradient!(grad, data, beta, work)
    @inbounds for j in eachindex(grad)
        grad[j] -= Float64(beta[j]) / prior_sd^2
    end
    return grad
end

function write_cox_gradient_check(data::CoxPreparedData, prior_sd::Real, beta0::AbstractVector, outdir::AbstractString)
    work = CoxGradientWorkspace(data)
    grad = zeros(Float64, length(beta0))
    cox_gaussian_gradient!(grad, data, beta0, prior_sd, work)
    grad_ad = ForwardDiff.gradient(b -> cox_gaussian_logpost(data, b, prior_sd), beta0)
    like_grad = cox_score_gradient(data, beta0)
    prior_grad = -beta0 ./ prior_sd^2
    row = DataFrame(
        max_abs_gradient_error = [maximum(abs.(grad .- grad_ad))],
        relative_gradient_error = [maximum(abs.(grad .- grad_ad)) / max(1.0, maximum(abs.(grad_ad)))],
        logposterior = [cox_gaussian_logpost(data, beta0, prior_sd)],
        likelihood_gradient_norm = [norm(like_grad)],
        prior_gradient_norm = [norm(prior_grad)],
        posterior_gradient_norm = [norm(grad)],
        max_abs_likelihood_gradient = [maximum(abs.(like_grad))],
        max_abs_prior_gradient = [maximum(abs.(prior_grad))],
    )
    CSV.write(joinpath(outdir, "cox_gaussian_gradient_check.csv"), row)
    return row
end

function run_cox_gaussian_scan(args, outdir::AbstractString)
    n = arg(args, "cox-n", 120)
    p = arg(args, "cox-p", 12)
    nonzero = arg(args, "cox-nonzero", 4)
    n_samples = arg(args, "cox-samples", arg(args, "samples", 2500))
    warmup = arg(args, "cox-warmup", arg(args, "warmup", 1000))
    chains = arg(args, "chains", 4)
    thin = arg(args, "cox-thin", arg(args, "thin", 1))
    prior_sd = arg(args, "prior-sd", 2.0)
    seed = arg(args, "seed", 1)
    event_fraction = arg(args, "event-fraction", 0.70)
    mala_steps = split_floats(arg(args, "cox-mala-steps", "0.0005,0.001,0.002,0.005"))
    ula_steps = split_floats(arg(args, "cox-ula-steps", "0.0002,0.0005,0.001"))
    max_lag = arg(args, "max-lag", 50)
    cox_dir = ensure_dir(joinpath(outdir, "cox_gaussian"))
    data, beta_true, x_raw, time, event = simulate_cox_data(n = n, p = p, nonzero = nonzero, seed = seed, event_fraction = event_fraction)
    CSV.write(
        joinpath(cox_dir, "data_summary.csv"),
        DataFrame(n = [n], p = [p], nonzero = [nonzero], events = [sum(event)], event_fraction = [mean(event)], prior_sd = [prior_sd]),
    )
    CSV.write(joinpath(cox_dir, "truth.csv"), DataFrame(parameter = ["beta[$j]" for j in 1:p], value = beta_true))
    write_cox_gradient_check(data, prior_sd, zeros(p), cox_dir)
    names0 = ["beta[$j]" for j in 1:p]
    rows = DataFrame()
    for sampler in (:mala, :ula)
        steps = sampler == :mala ? mala_steps : ula_steps
        for step in steps
            samples = MHSamples[]
            for c in 1:chains
                rng = MersenneTwister(seed + 20011c + round(Int, 1_000_000step))
                work = CoxGradientWorkspace(data)
                logpost = b -> cox_gaussian_logpost(data, b, prior_sd)
                grad! = (g, b) -> cox_gaussian_gradient!(g, data, b, prior_sd, work)
                grad = b -> begin
                    g = zeros(Float64, length(b))
                    grad!(g, b)
                    g
                end
                s =
                    sampler == :mala ?
                    B._run_full_mala(
                        logpost,
                        grad,
                        zeros(Float64, p),
                        names0;
                        n_samples = n_samples,
                        warmup = warmup,
                        thin = thin,
                        rng = rng,
                        mala_step_size = step,
                        mala_target_accept = 0.574,
                        mala_adapt_every = 50,
                        mala_min_step = step / 100,
                        mala_max_step = step * 100,
                        progress = false,
                        gradient! = grad!,
                    ) :
                    B._run_full_ula(
                        logpost,
                        grad,
                        zeros(Float64, p),
                        names0;
                        n_samples = n_samples,
                        warmup = warmup,
                        thin = thin,
                        rng = rng,
                        mala_step_size = step,
                        progress = false,
                        gradient! = grad!,
                    )
                push!(samples, s)
            end
            chains_obj = samples_to_chains(samples)
            row = merge(
                summarize_chains(
                    chains_obj;
                    experiment = "cox_gaussian",
                    sampler = sampler,
                    step = step,
                    thin = thin,
                    target_mean = nothing,
                    target_var = nothing,
                    param_prefix = "beta[",
                ),
                (
                    acceptance_rate = sampler == :mala ? mean(s.acceptance_rate for s in samples) : NaN,
                    warmup_acceptance_rate = sampler == :mala ? mean(s.warmup_acceptance_rate for s in samples) : NaN,
                    final_step_min = minimum(minimum(s.final_step) for s in samples),
                    final_step_median = median(reduce(vcat, [s.final_step for s in samples])),
                    final_step_max = maximum(maximum(s.final_step) for s in samples),
                    mean_abs_error_to_truth = mean(abs.(vec(mean(reshape(permutedims(_chains_array(chains_obj)[:, 1:p, :], (1, 3, 2)), :, p); dims = 1)) .- beta_true)),
                ),
            )
            push!(rows, row; cols = :union)
            plot_trace_and_acf(chains_obj, cox_dir, "cox_gaussian_$(sampler)_step_$(step)"; prefix = "beta[", max_params = 6, max_lag = max_lag)
        end
    end
    CSV.write(joinpath(cox_dir, "metrics.csv"), rows)
    return rows
end

function beta_draws_by_chain(result::CoxSamplerResult)
    if result.samples isa MHSamples
        p = size(result.data.z, 2)
        return reshape(result.samples.draws[:, 1:p], size(result.samples.draws, 1), p, 1)
    elseif result.samples isa MultiMHSamples
        p = size(result.data.z, 2)
        n_iter = size(result.samples.chains[1].draws, 1)
        n_chain = length(result.samples.chains)
        arr = Array{Float64}(undef, n_iter, p, n_chain)
        for c in 1:n_chain
            arr[:, :, c] .= result.samples.chains[c].draws[:, 1:p]
        end
        return arr
    end
    chains = result_to_chains(result)
    arr = _chains_array(chains)
    idx = findall(name -> startswith(string(name), "beta["), names(chains))
    return arr[:, idx, :]
end

function run_lasso_scan(args, outdir::AbstractString)
    n = arg(args, "lasso-n", arg(args, "cox-n", 120))
    p = arg(args, "lasso-p", arg(args, "cox-p", 12))
    nonzero = arg(args, "lasso-nonzero", arg(args, "cox-nonzero", 4))
    n_samples = arg(args, "lasso-samples", arg(args, "samples", 2000))
    warmup = arg(args, "lasso-warmup", arg(args, "warmup", 1000))
    chains = arg(args, "chains", 4)
    thin = arg(args, "lasso-thin", arg(args, "thin", 1))
    seed = arg(args, "seed", 1)
    event_fraction = arg(args, "event-fraction", 0.70)
    steps = split_floats(arg(args, "lasso-steps", "0.0002,0.0005,0.001"))
    lambdas = split_floats(arg(args, "lasso-lambdas", "0.25,1.0"))
    priors = split_strings(arg(args, "lasso-priors", "gaussian-lasso,prox-lasso"))
    samplers = split_symbols(arg(args, "lasso-samplers", "mala,ula"))
    prox_l1_lambda = arg(args, "prox-l1-lambda", 0.01)
    prox_sum0_lambda = arg(args, "prox-sum0-lambda", 0.01)
    max_lag = arg(args, "max-lag", 50)
    lasso_dir = ensure_dir(joinpath(outdir, "cox_lasso"))
    data, beta_true, x_raw, time, event = simulate_cox_data(n = n, p = p, nonzero = nonzero, seed = seed, event_fraction = event_fraction)
    CSV.write(
        joinpath(lasso_dir, "data_summary.csv"),
        DataFrame(n = [n], p = [p], nonzero = [nonzero], events = [sum(event)], event_fraction = [mean(event)]),
    )
    CSV.write(joinpath(lasso_dir, "truth.csv"), DataFrame(parameter = ["beta[$j]" for j in 1:p], value = beta_true))
    rows = DataFrame()
    for prior_name in priors, lambda in lambdas, sampler in samplers, step in steps
        constraint = prior_name == "prox-lasso" ? :prox : :soft_center
        prior = BayesianLassoPrior(lambda = lambda)
        label = "$(prior_name)_lambda_$(lambda)_$(sampler)_step_$(step)"
        row = try
            case_seed = seed + Int(mod(hash(label), UInt(100000)))
            result = sample_bayes_cox(
                data;
                prior = prior,
                sampler = sampler,
                n_samples = n_samples,
                warmup = warmup,
                thin = thin,
                chains = chains,
                multichains = true,
                seed = case_seed,
                use_car = false,
                constraint = constraint,
                prox_sum0_lambda = prox_sum0_lambda,
                prox_l1_lambda = prox_l1_lambda,
                mala_step_size = step,
                mala_target_accept = 0.574,
                mala_adapt_every = 50,
                mala_min_step = step / 100,
                mala_max_step = step * 100,
                progress = false,
            )
            chains_obj = result_to_chain_object(result)
            beta_arr = beta_draws_by_chain(result)
            beta_vals = reshape(permutedims(beta_arr, (1, 3, 2)), :, p)
            merged = merge(
                summarize_chains(
                    chains_obj;
                    experiment = "cox_lasso",
                    sampler = sampler,
                    step = step,
                    thin = thin,
                    target_mean = nothing,
                    target_var = nothing,
                    param_prefix = "beta[",
                ),
                (
                    prior = prior_name,
                    lambda = lambda,
                    constraint = String(constraint),
                    prox_l1_lambda = prox_l1_lambda,
                    prox_sum0_lambda = prox_sum0_lambda,
                    status = "ok",
                    acceptance_rate = acceptance_rate(result.samples),
                    warmup_acceptance_rate = warmup_acceptance_rate(result.samples),
                    mean_abs_error_to_truth = mean(abs.(vec(mean(beta_vals; dims = 1)) .- beta_true)),
                ),
                final_step_summary(result.samples),
            )
            plot_trace_and_acf(chains_obj, lasso_dir, label; prefix = "beta[", max_params = 6, max_lag = max_lag)
            merged
        catch err
            (
                experiment = "cox_lasso",
                sampler = String(sampler),
                step_size = step,
                thin = thin,
                prior = prior_name,
                lambda = lambda,
                constraint = String(constraint),
                prox_l1_lambda = prox_l1_lambda,
                prox_sum0_lambda = prox_sum0_lambda,
                status = "error",
                error = sprint(showerror, err),
            )
        end
        push!(rows, row; cols = :union)
        CSV.write(joinpath(lasso_dir, "metrics.csv"), rows)
    end
    return rows
end

function main()
    args = parse_cli_args()
    outdir = ensure_dir(arg(args, "outdir", joinpath("results", "langevin_diagnostics")))
    mode = Symbol(arg(args, "mode", "all"))
    mode in (:all, :normal, :cox_gaussian, :lasso) ||
        throw(ArgumentError("--mode must be one of all, normal, cox_gaussian, lasso."))
    rows = DataFrame()
    if mode in (:all, :normal)
        append!(rows, run_normal_scan(args, outdir); cols = :union)
    end
    if mode in (:all, :cox_gaussian)
        append!(rows, run_cox_gaussian_scan(args, outdir); cols = :union)
    end
    if mode in (:all, :lasso)
        append!(rows, run_lasso_scan(args, outdir); cols = :union)
    end
    CSV.write(joinpath(outdir, "metrics_all.csv"), rows)
    println("Wrote Langevin diagnostics to $(abspath(outdir))")
    show(rows; allrows = true, allcols = true)
    println()
end

main()
