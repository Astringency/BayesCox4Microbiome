using Test, Random, LinearAlgebra, BayesCox4Micro

@testset "Parallel chain targets do not capture shared workspaces" begin
    B = BayesCox4Micro
    for name in names(B; all = true)
        s = string(name)
        if startswith(s, "#_sample_hmc_explicit#") || startswith(s, "#_sample_nuts_explicit#")
            for method in methods(getfield(B, name))
                lowered = Base.uncompressed_ast(method)
                boxed_targets = [stmt for stmt in lowered.code if
                    stmt isa Expr && stmt.head == :(=) && stmt.args[1] isa Core.SlotNumber &&
                    occursin("Core.Box", string(stmt)) && lowered.slotnames[stmt.args[1].id] == :target]
                @test isempty(boxed_targets)
            end
        end
    end
end

@testset "Serial/parallel posterior draws match for fixed seeds" begin
    if Threads.nthreads() < 2
        @test_skip Threads.nthreads() >= 2 # also run this file with julia -t 4
    else
        old_blas = BLAS.get_num_threads()
        BLAS.set_num_threads(1)
        try
            rng = MersenneTwister(925)
            X = exp.(randn(rng, 80, 12))
            time, event = collect(1.0:80.0), ones(Int, 80)
            for sampler in (:hmc, :nuts_explicit), constraint in (:prox, :exact)
                extra = sampler == :hmc ? (; hmc_n_leapfrog = 5) : (; max_depth = 4)
                kwargs = (prior = RegularizedHorseshoePrior(p0 = 4.0), sampler = sampler,
                    constraint = constraint, noncentered = true, use_car = true,
                    sample_car_strength = true, n_samples = 24, warmup = 24, chains = 4,
                    seed = 172, progress = false)
                serial = sample_bayes_cox(X, time, event; kwargs..., multichains = false, extra...)
                parallel = sample_bayes_cox(X, time, event; kwargs..., multichains = true, extra...)
                @test posterior_beta_draws(serial) == posterior_beta_draws(parallel)
                @test posterior_logposterior_draws(serial) == posterior_logposterior_draws(parallel)
            end
        finally
            BLAS.set_num_threads(old_blas)
        end
    end
end
