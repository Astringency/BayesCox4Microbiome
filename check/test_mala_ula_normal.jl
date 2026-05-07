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

function run_one_sampler(sampler::Symbol, X, time, event; seed::Int, step::Float64)
    common = (
        prior = BayesianLassoPrior(),
        constraint = :prox,
        n_samples = 1000,
        warmup = 1000,
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

    rows = DataFrame()

    nuts = run_one_sampler(:nuts_explicit, X, time, event; seed = 101, step = 1e-6)
    push!(rows, summarize_result("nuts_explicit", nuts, beta_true); cols = :union)

    for step in (1e-7, 1e-6, 1e-5)
        mala = run_one_sampler(:mala, X, time, event; seed = 200 + round(Int, -log10(step)), step = step)
        push!(rows, summarize_result("mala_step_$step", mala, beta_true); cols = :union)

        ula = run_one_sampler(:ula, X, time, event; seed = 300 + round(Int, -log10(step)), step = step)
        push!(rows, summarize_result("ula_step_$step", ula, beta_true); cols = :union)
    end

    outdir = joinpath("results", "current_mala_ula_test")
    mkpath(outdir)
    CSV.write(joinpath(outdir, "summary.csv"), rows)

    println()
    println("Wrote summary to ", abspath(joinpath(outdir, "summary.csv")))
    println(rows)
end

main()
