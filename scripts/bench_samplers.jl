using BayesCox4Micro
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
    draws = size(posterior_beta_draws(result), 1)
    println("$label,$(timed.time),$(timed.bytes / 2.0^20),$accept,$draws,$(diag.gc_percent)")
    return result
end

function main()
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 400
    p = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 50
    n_samples = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 25
    warmup = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 25
    X, time, event = simulation_data(n, p)
    prior = BayesianLassoPrior()

    println("sampler,time_seconds,allocated_mb,acceptance_rate,draws,gc_percent")

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
            chains = 1,
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
            chains = 1,
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
