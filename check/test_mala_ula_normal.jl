using Pkg

function activate_repo()
    project = abspath(joinpath(@__DIR__, "..", "Project.toml"))
    Pkg.activate(dirname(project))
end

activate_repo()

using BayesCox4Micro
using DataFrames
using MCMCChains
using Random
using Statistics
using LinearAlgebra
using CSV
using Plots

function simulation_data(n::Int, p::Int; seed::Int = 2027)
    rng = MersenneTwister(seed)

    X = rand(rng, n, p) .+ 0.1

    beta_true = zeros(p)
    k = min(4, p)
    beta_true[1:k] .= range(0.6, -0.4; length = k)

    risk = X * beta_true

    time = exp.(-0.2 .* risk) .* (rand(rng, n) .* 10 .+ 1.0)

    event = Int.(rand(rng, n) .< 0.75)
    event[1] = 1

    return X, time, event, beta_true
end

function acceptance_rate(result)
    if result.samples isa MHSamples
        return result.samples.acceptance_rate
    elseif result.samples isa MultiMHSamples
        return mean(s.acceptance_rate for s in result.samples.chains)
    else
        return NaN
    end
end

function final_step(result)
    if result.samples isa MHSamples
        return result.samples.final_step
    elseif result.samples isa MultiMHSamples
        return [s.final_step for s in result.samples.chains]
    else
        return missing
    end
end

function summarize_result(label::String, result, beta_true)
    draws = posterior_beta_draws(result)
    chn = result_to_chains(result)

    stats = DataFrame(summarystats(chn))

    ess_col =
        :ess_bulk in propertynames(stats) ? :ess_bulk :
        :ess in propertynames(stats) ? :ess :
        nothing

    rhat_col = :rhat in propertynames(stats) ? :rhat : nothing

    ess_min =
        ess_col === nothing ? NaN :
        minimum(skipmissing(stats[!, ess_col]))

    rhat_max =
        rhat_col === nothing ? NaN :
        maximum(skipmissing(stats[!, rhat_col]))

    beta_mean = vec(mean(draws; dims = 1))
    mean_abs_error = mean(abs.(beta_mean .- beta_true))

    logp = posterior_logposterior_draws(result)

    println()
    println("========== $label ==========")
    println("draws size          = ", size(draws))
    println("acceptance rate     = ", acceptance_rate(result))
    println("final step          = ", final_step(result))
    println("ESS min             = ", ess_min)
    println("Rhat max            = ", rhat_max)
    println("mean abs beta error = ", mean_abs_error)
    println("mean(beta_mean)     = ", mean(beta_mean))
    println("max |beta_mean|     = ", maximum(abs.(beta_mean)))
    println("logpost first/last  = ", first(logp), " -> ", last(logp))

    return (
        sampler = label,
        acceptance_rate = acceptance_rate(result),
        ess_min = ess_min,
        rhat_max = rhat_max,
        mean_abs_beta_error = mean_abs_error,
        mean_beta_mean = mean(beta_mean),
        max_abs_beta_mean = maximum(abs.(beta_mean)),
        logpost_first = first(logp),
        logpost_last = last(logp),
    )
end

function safe_filename(label::String)
    return replace(label, r"[^A-Za-z0-9_.-]" => "_")
end

function plot_sampler_result(label::String, result, beta_true, outdir::String)
    mkpath(outdir)

    draws = posterior_beta_draws(result)
    beta_mean = vec(mean(draws; dims = 1))
    beta_sd = vec(std(draws; dims = 1))
    p = length(beta_true)

    file_label = safe_filename(label)

    plt_beta = plot(
        1:p,
        beta_true;
        label = "true beta",
        marker = :circle,
        linewidth = 2,
        xlabel = "coefficient index",
        ylabel = "beta",
        title = "$label: true beta vs posterior mean",
        size = (900, 500),
    )

    plot!(
        plt_beta,
        1:p,
        beta_mean;
        ribbon = beta_sd,
        label = "posterior mean ± sd",
        marker = :diamond,
        linewidth = 2,
    )

    savefig(plt_beta, joinpath(outdir, "$(file_label)_beta.png"))

    logp = posterior_logposterior_draws(result)

    plt_logp = plot(
        eachindex(logp),
        logp;
        label = "",
        xlabel = "draw",
        ylabel = "log posterior",
        title = "$label: log posterior trace",
        size = (900, 500),
    )

    savefig(plt_logp, joinpath(outdir, "$(file_label)_logposterior.png"))
end

function plot_summary_metrics(rows::DataFrame, outdir::String)
    mkpath(outdir)

    metrics = [
        :acceptance_rate,
        :ess_min,
        :rhat_max,
        :mean_abs_beta_error,
        :mean_beta_mean,
        :max_abs_beta_mean,
        :logpost_last,
    ]

    labels = String.(rows.sampler)

    for metric in metrics
        metric_name = String(metric)

        if !(metric_name in names(rows))
            continue
        end

        vals = Float64[]

        for x in rows[!, metric]
            try
                push!(vals, Float64(x))
            catch
                push!(vals, NaN)
            end
        end

        if !any(isfinite.(vals))
            continue
        end

        plt = bar(
            labels,
            vals;
            legend = false,
            xlabel = "sampler",
            ylabel = metric_name,
            title = "Sampler comparison: $metric_name",
            xrotation = 45,
            size = (1000, 500),
        )

        savefig(plt, joinpath(outdir, "$(metric_name).png"))
    end
end

function run_one_sampler(sampler::Symbol, X, time, event; seed::Int, step::Float64)
    common = (
        prior = BayesianLassoPrior(),
        constraint = :prox,
        n_samples = 5000,
        warmup = 5000,
        chains = 4,
        progress = false,
        seed = seed,
        transform = :none,
        prox_sum0_lambda = 1e-3,
        prox_l1_lambda = 1e-3,
    )

    if sampler == :nuts_explicit
        return sample_bayes_cox(
            X,
            time,
            event;
            common...,
            sampler = :nuts,
            nuts_backend = :explicit,
            parallel_chains = true,
        )
    elseif sampler in (:mala, :ula)
        return sample_bayes_cox(
            X,
            time,
            event;
            common...,
            sampler = sampler,
            multichains = true,
            mala_step_size = step,
            mala_min_step = step / 100,
            mala_max_step = step * 100,
            mala_target_accept = 0.574,
        )
    else
        error("unsupported sampler: $sampler")
    end
end

function main()
    X, time, event, beta_true = simulation_data(120, 8; seed = 42)

    println("n = ", size(X, 1), ", p = ", size(X, 2), ", events = ", sum(event))
    println("beta_true = ", beta_true)

    outdir = joinpath("results", "current_mala_ula_test")
    mkpath(outdir)

    rows = DataFrame()

    nuts_label = "nuts_explicit"
    nuts = run_one_sampler(:nuts_explicit, X, time, event; seed = 101, step = 1e-6)
    push!(rows, summarize_result(nuts_label, nuts, beta_true); cols = :union)
    plot_sampler_result(nuts_label, nuts, beta_true, outdir)

    for step in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
        mala_label = "mala_step_$step"
        mala = run_one_sampler(
            :mala,
            X,
            time,
            event;
            seed = 200 + round(Int, -log10(step)),
            step = step,
        )
        push!(rows, summarize_result(mala_label, mala, beta_true); cols = :union)
        plot_sampler_result(mala_label, mala, beta_true, outdir)

        ula_label = "ula_step_$step"
        ula = run_one_sampler(
            :ula,
            X,
            time,
            event;
            seed = 300 + round(Int, -log10(step)),
            step = step,
        )
        push!(rows, summarize_result(ula_label, ula, beta_true); cols = :union)
        plot_sampler_result(ula_label, ula, beta_true, outdir)
    end

    CSV.write(joinpath(outdir, "summary.csv"), rows)
    plot_summary_metrics(rows, outdir)

    println()
    println("Wrote summary to ", abspath(joinpath(outdir, "summary.csv")))
    println("Wrote plots to   ", abspath(outdir))
    println(rows)
end

main()
