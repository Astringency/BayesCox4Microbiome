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
using Distributions
using LinearAlgebra
using MCMCChains
using Plots
using Random
using Statistics
using StatsPlots
using Turing

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

function split_symbols(s::AbstractString)
    return Symbol.(strip.(split(s, ",")))
end

function _standardize_columns(x::AbstractMatrix)
    z = Matrix{Float64}(x)
    mu = vec(mean(z; dims = 1))
    sigma = vec(std(z; dims = 1))
    sigma .= ifelse.(sigma .> 0, sigma, 1.0)
    z .= (z .- mu') ./ sigma'
    return z, mu, sigma
end

function _truth_beta(p::Integer, nonzero::Integer)
    base = [1.35, -1.15, 0.95, -0.85, 0.75, -0.65, 0.58, -0.52, 0.46, -0.40, 0.35, -0.30]
    beta = zeros(Float64, p)
    k = min(p, nonzero)
    for j in 1:k
        beta[j] = j <= length(base) ? base[j] : (-1.0)^j * max(0.15, 0.9 - 0.04j)
    end
    return beta
end

function _event_times(rng::AbstractRNG, eta::AbstractVector, baseline::Symbol)
    n = length(eta)
    e = randexp(rng, n)
    rate = exp.(eta)
    if baseline == :weibull
        shape = 2.0
        scale = 50.0
        return scale .* (e ./ rate) .^ (1 / shape)
    elseif baseline == :exponential
        base_rate = 0.02
        return e ./ (base_rate .* rate)
    elseif baseline == :gompertz
        a = 0.02
        b = 0.03
        return log.(1 .+ b .* e ./ (a .* rate)) ./ b
    else
        throw(ArgumentError("baseline must be one of weibull, exponential, gompertz"))
    end
end

function simulate_cox_data(; n::Integer, p::Integer, nonzero::Integer, baseline::Symbol, seed::Integer, event_fraction::Real)
    rng = MersenneTwister(seed)
    x_raw = exp.(0.45 .* randn(rng, n, p))
    z, _, _ = _standardize_columns(x_raw)
    beta_true = _truth_beta(p, nonzero)
    eta = z * beta_true
    t_event = _event_times(rng, eta, baseline)
    censor_time = quantile(t_event, event_fraction)
    time = min.(t_event, censor_time)
    event = Int.(t_event .<= censor_time)
    data = prepare_cox_data(x_raw, time, event; transform = :none, standardize = true)
    return data, beta_true, x_raw, time, event
end

function _cox_normal_logtarget(data::CoxPreparedData, beta::AbstractVector, prior_sd::Real)
    return cox_logpartiallikelihood(data, beta) - 0.5 * sum(abs2, beta) / prior_sd^2
end

function _cox_normal_gradient(data::CoxPreparedData, beta::AbstractVector, prior_sd::Real)
    return cox_score_gradient(data, beta) .- beta ./ prior_sd^2
end

function _mh_chain(rng::AbstractRNG, data::CoxPreparedData, n_samples::Integer, warmup::Integer, step_size::Real, prior_sd::Real)
    p = size(data.z, 2)
    total = warmup + n_samples
    beta = zeros(Float64, p)
    lp = _cox_normal_logtarget(data, beta, prior_sd)
    draws = Matrix{Float64}(undef, n_samples, p)
    accepted = 0
    saved = 0
    for iter in 1:total
        proposal = beta .+ step_size .* randn(rng, p)
        prop_lp = _cox_normal_logtarget(data, proposal, prior_sd)
        if log(rand(rng)) < prop_lp - lp
            beta = proposal
            lp = prop_lp
            iter > warmup && (accepted += 1)
        end
        if iter > warmup
            saved += 1
            draws[saved, :] .= beta
        end
    end
    return draws, accepted / n_samples
end

function _mala_logq(y::AbstractVector, x::AbstractVector, grad_x::AbstractVector, step_size::Real)
    diff = y .- x .- step_size .* grad_x
    return -0.25 / step_size * sum(abs2, diff)
end

function _mala_chain(rng::AbstractRNG, data::CoxPreparedData, n_samples::Integer, warmup::Integer, step_size::Real, prior_sd::Real)
    p = size(data.z, 2)
    total = warmup + n_samples
    beta = zeros(Float64, p)
    lp = _cox_normal_logtarget(data, beta, prior_sd)
    draws = Matrix{Float64}(undef, n_samples, p)
    accepted = 0
    saved = 0
    for iter in 1:total
        grad = _cox_normal_gradient(data, beta, prior_sd)
        proposal = beta .+ step_size .* grad .+ sqrt(2 * step_size) .* randn(rng, p)
        prop_lp = _cox_normal_logtarget(data, proposal, prior_sd)
        prop_grad = _cox_normal_gradient(data, proposal, prior_sd)
        log_alpha = prop_lp + _mala_logq(beta, proposal, prop_grad, step_size) -
            lp - _mala_logq(proposal, beta, grad, step_size)
        if log(rand(rng)) < log_alpha
            beta = proposal
            lp = prop_lp
            iter > warmup && (accepted += 1)
        end
        if iter > warmup
            saved += 1
            draws[saved, :] .= beta
        end
    end
    return draws, accepted / n_samples
end

function _coordinate_mwg_chain(rng::AbstractRNG, data::CoxPreparedData, n_samples::Integer, warmup::Integer, step_size::Real, prior_sd::Real)
    p = size(data.z, 2)
    total = warmup + n_samples
    beta = zeros(Float64, p)
    lp = _cox_normal_logtarget(data, beta, prior_sd)
    draws = Matrix{Float64}(undef, n_samples, p)
    accepted = 0
    proposals = 0
    saved = 0
    for iter in 1:total
        for j in 1:p
            proposal = copy(beta)
            proposal[j] += step_size * randn(rng)
            prop_lp = _cox_normal_logtarget(data, proposal, prior_sd)
            if log(rand(rng)) < prop_lp - lp
                beta = proposal
                lp = prop_lp
                iter > warmup && (accepted += 1)
            end
            iter > warmup && (proposals += 1)
        end
        if iter > warmup
            saved += 1
            draws[saved, :] .= beta
        end
    end
    return draws, accepted / proposals
end

@model function _cox_normal_turing_model(data, prior_sd, p)
    beta ~ MvNormal(zeros(p), prior_sd^2 * Matrix{Float64}(I, p, p))
    Turing.@addlogprob! cox_logpartiallikelihood(data, beta)
end

function _chains_from_draws(draws_by_chain::Vector{Matrix{Float64}}, names)
    n_iter, n_param = size(draws_by_chain[1])
    arr = Array{Float64}(undef, n_iter, n_param, length(draws_by_chain))
    for (c, draws) in enumerate(draws_by_chain)
        size(draws) == (n_iter, n_param) ||
            throw(DimensionMismatch("all chains must have the same draw matrix shape."))
        arr[:, :, c] .= draws
    end
    return Chains(arr, Symbol.(names))
end

function run_sampler(
    sampler::Symbol,
    data::CoxPreparedData;
    n_samples::Integer,
    warmup::Integer,
    chains::Integer,
    seed::Integer,
    prior_sd::Real,
    mh_step::Real,
    mala_step::Real,
    gibbs_step::Real,
)
    p = size(data.z, 2)
    names = ["beta[$j]" for j in 1:p]
    if sampler == :nuts
        rng = MersenneTwister(seed)
        model = _cox_normal_turing_model(data, prior_sd, p)
        return Turing.sample(rng, model, Turing.NUTS(warmup, 0.8), Turing.MCMCThreads(), n_samples, chains; progress = false), missing
    end
    draws_by_chain = Matrix{Float64}[]
    accept = Float64[]
    for c in 1:chains
        rng = MersenneTwister(seed + 1009c)
        draws, rate =
            sampler == :mh ? _mh_chain(rng, data, n_samples, warmup, mh_step, prior_sd) :
            sampler == :mala ? _mala_chain(rng, data, n_samples, warmup, mala_step, prior_sd) :
            sampler == :gibbs ? _coordinate_mwg_chain(rng, data, n_samples, warmup, gibbs_step, prior_sd) :
            throw(ArgumentError("sampler must be one of nuts,mh,mala,gibbs"))
        push!(draws_by_chain, draws)
        push!(accept, rate)
    end
    return _chains_from_draws(draws_by_chain, names), accept
end

function _chains_array(chains)
    arr = Array(getfield(chains, :value))
    ndims(arr) == 2 && (arr = reshape(arr, size(arr, 1), size(arr, 2), 1))
    ndims(arr) == 3 || throw(ArgumentError("expected a 2D or 3D Chains value array, got $(ndims(arr)) dimensions."))
    return arr
end

function _beta_parameter_indices(chains)
    idx = findall(name -> startswith(string(name), "beta["), names(chains))
    isempty(idx) && throw(ArgumentError("Chains object does not contain beta[...] parameters."))
    return idx
end

function chains_to_wide(chains)
    arr = _chains_array(chains)
    n_iter, n_param, n_chain = size(arr)
    param_names = string.(names(chains))
    df = DataFrame(chain = Int[], iter = Int[], draw = Int[])
    for name in param_names
        df[!, Symbol(name)] = Float64[]
    end
    for c in 1:n_chain, i in 1:n_iter
        row = Dict{Symbol,Any}(:chain => c, :iter => i, :draw => (c - 1) * n_iter + i)
        for j in 1:n_param
            row[Symbol(param_names[j])] = arr[i, j, c]
        end
        push!(df, row; cols = :union)
    end
    return df
end

function _safe_savefig(plot_obj, path::AbstractString)
    if lowercase(splitext(path)[2]) == ".eps"
        open(path, "w") do io
            show(io, MIME("application/postscript"), plot_obj)
        end
    else
        savefig(plot_obj, path)
    end
    return path
end

function _summary_metrics(summary::DataFrame, chains, beta_true::AbstractVector, accept)
    beta_summary =
        hasproperty(summary, :parameters) ?
        summary[startswith.(string.(summary.parameters), "beta["), :] :
        summary
    rhat = hasproperty(beta_summary, :rhat) ? collect(skipmissing(beta_summary.rhat)) : Float64[]
    ess = hasproperty(beta_summary, :ess_bulk) ? collect(skipmissing(beta_summary.ess_bulk)) : Float64[]
    arr = _chains_array(chains)
    beta_idx = _beta_parameter_indices(chains)
    beta_arr = arr[:, beta_idx, :]
    beta_mean = vec(mean(reshape(permutedims(beta_arr, (1, 3, 2)), :, length(beta_idx)); dims = 1))
    beta_truth = beta_true[1:length(beta_idx)]
    return (
        mean_abs_error = mean(abs.(beta_mean .- beta_truth)),
        max_abs_error = maximum(abs.(beta_mean .- beta_truth)),
        rhat_max = isempty(rhat) ? missing : maximum(rhat),
        rhat_median = isempty(rhat) ? missing : median(rhat),
        ess_bulk_min = isempty(ess) ? missing : minimum(ess),
        ess_bulk_median = isempty(ess) ? missing : median(ess),
        acceptance_rate = accept === missing ? missing : mean(accept),
    )
end

function write_diagnostics(chains, output_dir::AbstractString, sampler::Symbol, accept, beta_true; max_lag::Integer, formats)
    mkpath(output_dir)
    summary = DataFrame(MCMCChains.summarystats(chains))
    qs = DataFrame(MCMCChains.quantile(chains; q = [0.025, 0.5, 0.975]))
    CSV.write(joinpath(output_dir, "diagnostics_summary.csv"), summary)
    CSV.write(joinpath(output_dir, "quantiles.csv"), qs)
    CSV.write(joinpath(output_dir, "samples_wide.csv"), chains_to_wide(chains))
    CSV.write(joinpath(output_dir, "truth.csv"), DataFrame(parameter = ["beta[$j]" for j in eachindex(beta_true)], value = beta_true))
    metadata = DataFrame(
        sampler = [String(sampler)],
        chains = [size(_chains_array(chains), 3)],
        acceptance_rate = [accept === missing ? missing : mean(accept)],
        acceptance_rate_by_chain = [accept === missing ? missing : join(round.(accept; digits = 4), ",")],
    )
    CSV.write(joinpath(output_dir, "metadata.csv"), metadata)

    selected_names = Symbol.(first(string.(names(chains)), min(8, length(names(chains)))))
    selected = chains[:, selected_names, :]
    plot_specs = (
        "traceplot" => (seriestype = :traceplot,),
        "density" => (seriestype = :mixeddensity,),
        "acf" => (seriestype = :autocorplot, maxlag = max_lag),
    )
    for (stem, kwargs) in plot_specs
        plt = plot(selected; kwargs...)
        for fmt in formats
            _safe_savefig(plt, joinpath(output_dir, "$stem.$fmt"))
        end
    end

    if hasproperty(summary, :rhat)
        rhat = collect(skipmissing(summary.rhat))
        if !isempty(rhat)
            p1 = scatter(1:length(rhat), Float64.(rhat); xlabel = "parameter index", ylabel = "R-hat", label = "", title = "R-hat")
            hline!(p1, [1.01]; label = "1.01")
            p2 = histogram(Float64.(rhat); xlabel = "R-hat", ylabel = "count", label = "", title = "R-hat histogram")
            vline!(p2, [1.01]; label = "1.01")
            for fmt in formats
                _safe_savefig(p1, joinpath(output_dir, "rhat_scatter.$fmt"))
                _safe_savefig(p2, joinpath(output_dir, "rhat_histogram.$fmt"))
            end
        end
    end
    return summary
end

function main()
    args = parse_cli_args()
    n = arg(args, "n", 400)
    p = arg(args, "p", 50)
    nonzero = arg(args, "nonzero", 12)
    baseline = Symbol(arg(args, "baseline", "weibull"))
    n_samples = arg(args, "samples", 1000)
    warmup = arg(args, "warmup", 500)
    chains = arg(args, "chains", 4)
    seed = arg(args, "seed", 1)
    prior_sd = arg(args, "prior-sd", 2.5)
    mh_step = arg(args, "mh-step", 0.04)
    mala_step = arg(args, "mala-step", 1e-3)
    gibbs_step = arg(args, "gibbs-step", 0.08)
    event_fraction = arg(args, "event-fraction", 0.70)
    max_lag = arg(args, "max-lag", 50)
    outdir = arg(args, "outdir", joinpath("results", "cox_normal_prior_sampler_test"))
    samplers = split_symbols(arg(args, "samplers", "nuts,mh,mala,gibbs"))
    formats = strip.(split(arg(args, "format", "png"), ","))

    data, beta_true, _, time, event = simulate_cox_data(
        n = n,
        p = p,
        nonzero = nonzero,
        baseline = baseline,
        seed = seed,
        event_fraction = event_fraction,
    )

    mkpath(outdir)
    CSV.write(
        joinpath(outdir, "data_summary.csv"),
        DataFrame(
            n = [n],
            p = [p],
            nonzero = [nonzero],
            baseline = [String(baseline)],
            number_of_events = [sum(event)],
            event_fraction = [mean(event)],
            median_time = [median(time)],
            prior_sd = [prior_sd],
        ),
    )

    rows = DataFrame()
    for sampler in samplers
        println("Running $sampler on CoxPH normal-prior target: n=$n, p=$p, events=$(sum(event))...")
        chains_obj, accept = run_sampler(
            sampler,
            data;
            n_samples = n_samples,
            warmup = warmup,
            chains = chains,
            seed = seed,
            prior_sd = prior_sd,
            mh_step = mh_step,
            mala_step = mala_step,
            gibbs_step = gibbs_step,
        )
        summary = write_diagnostics(
            chains_obj,
            joinpath(outdir, String(sampler)),
            sampler,
            accept,
            beta_true;
            max_lag = max_lag,
            formats = formats,
        )
        metrics = _summary_metrics(summary, chains_obj, beta_true, accept)
        push!(
            rows,
            merge(
                (
                    sampler = String(sampler),
                    n = n,
                    p = p,
                    nonzero = nonzero,
                    baseline = String(baseline),
                    number_of_events = sum(event),
                    samples = n_samples,
                    warmup = warmup,
                    chains = chains,
                    prior_sd = prior_sd,
                ),
                metrics,
            );
            cols = :union,
        )
    end
    CSV.write(joinpath(outdir, "sampler_comparison.csv"), rows)
    println("CoxPH normal-prior sampler test written to $(abspath(outdir))")
    println(rows)
end

main()
