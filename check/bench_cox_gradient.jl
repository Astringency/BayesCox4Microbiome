using BayesCox4Micro
using ForwardDiff
using Random

function simulation_data(n::Int, p::Int; seed::Int = 2026)
    rng = MersenneTwister(seed)
    X = rand(rng, n, p) .+ 0.1
    time = sort(rand(rng, n) .* 10 .+ 1)
    event = rand(rng, n) .< 0.65
    event[1] = true
    data = prepare_cox_data(X, time, Int.(event); transform = :none)
    beta = 0.05 .* randn(rng, p)
    return data, beta
end

function best_timed(f; reps::Int = 5)
    best = nothing
    for _ in 1:reps
        GC.gc()
        timed = @timed f()
        if best === nothing || timed.time < best.time
            best = timed
        end
    end
    return best
end

function main()
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 400
    p = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 50
    reps = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 5
    data, beta = simulation_data(n, p)
    work = CoxGradientWorkspace(data)
    grad = zeros(Float64, p)

    cox_score_gradient(data, beta)
    cox_score_gradient!(grad, data, beta, work)

    alloc = best_timed(reps = reps) do
        cox_score_gradient(data, beta)
    end
    prealloc = best_timed(reps = reps) do
        cox_score_gradient!(grad, data, beta, work)
    end

    println("benchmark,n,p,reps,time_seconds,allocated_mb")
    println("cox_score_gradient,$n,$p,$reps,$(alloc.time),$(alloc.bytes / 2.0^20)")
    println("cox_score_gradient!,$n,$p,$reps,$(prealloc.time),$(prealloc.bytes / 2.0^20)")

    if p <= 80
        fd = best_timed(reps = max(1, min(reps, 2))) do
            ForwardDiff.gradient(b -> cox_logpartiallikelihood(data, b), beta)
        end
        maxerr = maximum(abs.(grad .- ForwardDiff.gradient(b -> cox_logpartiallikelihood(data, b), beta)))
        println("forwarddiff_gradient,$n,$p,$(max(1, min(reps, 2))),$(fd.time),$(fd.bytes / 2.0^20)")
        println("max_abs_error_fast_vs_forwarddiff,$n,$p,1,$maxerr,0.0")
    end
end

main()
