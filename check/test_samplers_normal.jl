#!/usr/bin/env julia

using Pkg

function _activate_repo_project()
    project = abspath(joinpath(@__DIR__, "..", "Project.toml"))
    active = Base.active_project()
    active !== project && Pkg.activate(dirname(project))
end

_activate_repo_project()

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

function _normal_logdensity(x::AbstractVector)
    return -0.5 * sum(abs2, x)
end

function _normal_gradient(x::AbstractVector)
    return -x
end

function _mh_normal_chain(rng::AbstractRNG, dim::Integer, n_samples::Integer, warmup::Integer, step_size::Real)
    total = warmup + n_samples
    x = randn(rng, dim)
    lp = _normal_logdensity(x)
    draws = Matrix{Float64}(undef, n_samples, dim)
    accepted = 0
    saved = 0
    for iter in 1:total
        proposal = x .+ step_size .* randn(rng, dim)
        prop_lp = _normal_logdensity(proposal)
        if log(rand(rng)) < prop_lp - lp
            x = proposal
            lp = prop_lp
            iter > warmup && (accepted += 1)
        end
        if iter > warmup
            saved += 1
            draws[saved, :] .= x
        end
    end
    return draws, accepted / n_samples
end

function _mala_logq(y::AbstractVector, x::AbstractVector, grad_x::AbstractVector, step_size::Real)
    diff = y .- x .- step_size .* grad_x
    return -0.25 / step_size * sum(abs2, diff)
end

function _mala_normal_chain(rng::AbstractRNG, dim::Integer, n_samples::Integer, warmup::Integer, step_size::Real)
    total = warmup + n_samples
    x = randn(rng, dim)
    lp = _normal_logdensity(x)
    draws = Matrix{Float64}(undef, n_samples, dim)
    accepted = 0
    saved = 0
    for iter in 1:total
        grad_x = _normal_gradient(x)
        proposal = x .+ step_size .* grad_x .+ sqrt(2 * step_size) .* randn(rng, dim)
        prop_lp = _normal_logdensity(proposal)
        grad_prop = _normal_gradient(proposal)
        log_alpha = prop_lp + _mala_logq(x, proposal, grad_prop, step_size) -
            lp - _mala_logq(proposal, x, grad_x, step_size)
        if log(rand(rng)) < log_alpha
            x = proposal
            lp = prop_lp
            iter > warmup && (accepted += 1)
        end
        if iter > warmup
            saved += 1
            draws[saved, :] .= x
        end
    end
    return draws, accepted / n_samples
end

function _gibbs_normal_chain(rng::AbstractRNG, dim::Integer, n_samples::Integer, warmup::Integer)
    total = warmup + n_samples
    draws = Matrix{Float64}(undef, n_samples, dim)
    saved = 0
    for iter in 1:total
        x = randn(rng, dim)
        if iter > warmup
            saved += 1
            draws[saved, :] .= x
        end
    end
    return draws, 1.0
end

@model function _normal_turing_model(dim)
    x ~ MvNormal(zeros(dim), I)
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

function run_sampler(sampler::Symbol; dim::Integer, n_samples::Integer, warmup::Integer, chains::Integer, seed::Integer, mh_step::Real, mala_step::Real)
    names = ["x[$j]" for j in 1:dim]
    if sampler == :nuts
        model = _normal_turing_model(dim)
        rng = MersenneTwister(seed)
        return Turing.sample(rng, model, Turing.NUTS(warmup, 0.8), Turing.MCMCThreads(), n_samples, chains; progress = false), missing
    end
    draws_by_chain = Matrix{Float64}[]
    accept = Float64[]
    for c in 1:chains
        rng = MersenneTwister(seed + 1009c)
        draws, rate =
            sampler == :mh ? _mh_normal_chain(rng, dim, n_samples, warmup, mh_step) :
            sampler == :mala ? _mala_normal_chain(rng, dim, n_samples, warmup, mala_step) :
            sampler == :gibbs ? _gibbs_normal_chain(rng, dim, n_samples, warmup) :
            throw(ArgumentError("sampler must be one of nuts,mh,mala,gibbs"))
        push!(draws_by_chain, draws)
        push!(accept, rate)
    end
    return _chains_from_draws(draws_by_chain, names), accept
end

function chains_to_wide(chains)
    arr = Array(getfield(chains, :value))
    ndims(arr) == 2 && (arr = reshape(arr, size(arr, 1), size(arr, 2), 1))
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

function write_diagnostics(chains, output_dir::AbstractString, sampler::Symbol, accept; max_lag::Integer, formats)
    mkpath(output_dir)
    summary = DataFrame(MCMCChains.summarystats(chains))
    qs = DataFrame(MCMCChains.quantile(chains; q = [0.025, 0.5, 0.975]))
    CSV.write(joinpath(output_dir, "diagnostics_summary.csv"), summary)
    CSV.write(joinpath(output_dir, "quantiles.csv"), qs)
    CSV.write(joinpath(output_dir, "samples_wide.csv"), chains_to_wide(chains))
    metadata = DataFrame(
        sampler = [String(sampler)],
        chains = [size(Array(getfield(chains, :value)), 3)],
        acceptance_rate = [accept === missing ? missing : mean(accept)],
        acceptance_rate_by_chain = [accept === missing ? missing : join(round.(accept; digits = 4), ",")],
    )
    CSV.write(joinpath(output_dir, "metadata.csv"), metadata)

    selected = chains[:, Symbol.(first(string.(names(chains)), min(5, length(names(chains))))), :]
    plot_specs = (
        "traceplot" => (seriestype = :traceplot,),
        "density" => (seriestype = :mixeddensity,),
        "acf" => (seriestype = :autocorplot, maxlag = max_lag),
    )
    for (stem, kwargs) in plot_specs
        p = plot(selected; kwargs...)
        for fmt in formats
            _safe_savefig(p, joinpath(output_dir, "$stem.$fmt"))
        end
    end

    if :rhat in propertynames(summary)
        rhat = summary.rhat
        params = string.(summary.parameters)
        p1 = scatter(1:length(rhat), Float64.(rhat); xlabel = "parameter index", ylabel = "R-hat", label = "", title = "R-hat")
        hline!(p1, [1.01]; label = "1.01")
        p2 = histogram(Float64.(rhat); xlabel = "R-hat", ylabel = "count", label = "", title = "R-hat histogram")
        vline!(p2, [1.01]; label = "1.01")
        for fmt in formats
            _safe_savefig(p1, joinpath(output_dir, "rhat_scatter.$fmt"))
            _safe_savefig(p2, joinpath(output_dir, "rhat_histogram.$fmt"))
        end
    end
    return summary
end

function main()
    args = parse_cli_args()
    dim = arg(args, "dim", 5)
    n_samples = arg(args, "samples", 1000)
    warmup = arg(args, "warmup", 500)
    chains = arg(args, "chains", 4)
    seed = arg(args, "seed", 1)
    mh_step = arg(args, "mh-step", 0.7)
    mala_step = arg(args, "mala-step", 0.5)
    max_lag = arg(args, "max-lag", 50)
    outdir = arg(args, "outdir", joinpath("results", "normal_sampler_test"))
    samplers = split_symbols(arg(args, "samplers", "nuts,mh,mala,gibbs"))
    formats = strip.(split(arg(args, "format", "png"), ","))

    mkpath(outdir)
    rows = DataFrame()
    for sampler in samplers
        println("Running $sampler on $dim-dimensional standard normal target...")
        chains_obj, accept = run_sampler(
            sampler;
            dim = dim,
            n_samples = n_samples,
            warmup = warmup,
            chains = chains,
            seed = seed,
            mh_step = mh_step,
            mala_step = mala_step,
        )
        summary = write_diagnostics(chains_obj, joinpath(outdir, String(sampler)), sampler, accept; max_lag = max_lag, formats = formats)
        rhat = collect(skipmissing(summary.rhat))
        ess = collect(skipmissing(summary.ess_bulk))
        push!(
            rows,
            (
                sampler = String(sampler),
                max_abs_mean = maximum(abs.(summary.mean)),
                median_sd = median(summary.std),
                rhat_max = maximum(rhat),
                rhat_median = median(rhat),
                ess_bulk_min = minimum(ess),
                ess_bulk_median = median(ess),
                acceptance_rate = accept === missing ? missing : mean(accept),
            );
            cols = :union,
        )
    end
    CSV.write(joinpath(outdir, "sampler_comparison.csv"), rows)
    println("Normal sampler test written to $(abspath(outdir))")
    println(rows)
end

main()
