using BayesCox4Micro
using DataFrames
using MCMCChains
using Random
using Statistics

function simulation_data(n::Int, p::Int; seed::Int = 2027)
    rng = MersenneTwister(seed)
    X = rand(rng, n, p) .+ 0.1
    beta = zeros(p)
    beta[1:min(8, p)] .= range(0.5, -0.3; length = min(8, p))
    risk = X * beta
    time = exp.(-0.15 .* risk) .* (rand(rng, n) .* 10 .+ 1)
    event = rand(rng, n) .< 0.7
    event[1] = true
    return X, time, Int.(event)
end

function run_case(label, f)
    GC.gc()
    timed = @timed f()
    result = timed.value
    diag = sampler_diagnostics(result)
    accept =
        if result.samples isa MHSamples
            result.samples.acceptance_rate
        elseif result.samples isa MultiMHSamples
            mean(sample.acceptance_rate for sample in result.samples.chains)
        else
            NaN
        end
    ess_bulk_min, rhat_max = try
        stats = DataFrame(summarystats(result_to_chains(result)))
        ess_col =
            :ess_bulk in propertynames(stats) ? :ess_bulk :
            :ess in propertynames(stats) ? :ess : nothing
        rhat_col = :rhat in propertynames(stats) ? :rhat : nothing
        ess = ess_col === nothing ? NaN : minimum(skipmissing(stats[!, ess_col]))
        rhat = rhat_col === nothing ? NaN : maximum(skipmissing(stats[!, rhat_col]))
        Float64(ess), Float64(rhat)
    catch
        NaN, NaN
    end
    draws = size(posterior_beta_draws(result), 1)
    println("$label,$(diag.total_seconds),$(diag.allocated_bytes / 2.0^20),$(diag.gc_percent),$accept,$draws,$ess_bulk_min,$rhat_max,$(timed.time),$(timed.bytes / 2.0^20)")
    return result
end

function main()
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 400
    p = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 50
    n_samples = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 25
    warmup = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 25
    chains = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 2
    X, time, event = simulation_data(n, p)
    prior = BayesianLassoPrior()

    println("sampler,total_seconds,allocated_mb,gc_percent,acceptance_rate,draws,ess_bulk_min,rhat_max,outer_seconds,outer_allocated_mb")

    for (label, f) in (
        ("nuts_turing", () -> sample_bayes_cox(
            X,
            time,
            event;
            prior = prior,
            sampler = :nuts,
            nuts_backend = :turing,
            constraint = :prox,
            n_samples = n_samples,
            warmup = warmup,
            chains = chains,
            progress = false,
            seed = 11,
            transform = :none,
        )),
        ("nuts_explicit", () -> sample_bayes_cox(
            X,
            time,
            event;
            prior = prior,
            sampler = :nuts,
            nuts_backend = :explicit,
            constraint = :prox,
            n_samples = n_samples,
            warmup = warmup,
            chains = chains,
            parallel_chains = true,
            progress = false,
            seed = 12,
            transform = :none,
        )),
        ("mala", () -> sample_bayes_cox(
            X,
            time,
            event;
            prior = prior,
            sampler = :mala,
            constraint = :prox,
            n_samples = n_samples,
            warmup = warmup,
            chains = chains,
            multichains = true,
            mala_step_size = 1e-6,
            progress = false,
            seed = 13,
            transform = :none,
        )),
        ("ula", () -> sample_bayes_cox(
            X,
            time,
            event;
            prior = prior,
            sampler = :ula,
            constraint = :prox,
            n_samples = n_samples,
            warmup = warmup,
            chains = chains,
            multichains = true,
            mala_step_size = 1e-6,
            progress = false,
            seed = 14,
            transform = :none,
        )),
    )
        try
            run_case(label, f)
        catch err
            println("$label,NaN,NaN,NaN,0,NaN")
            @warn "benchmark case failed" label exception = (err, catch_backtrace())
        end
    end
end

main()
