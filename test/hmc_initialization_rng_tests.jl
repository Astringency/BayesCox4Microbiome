using Test, Random, BayesCox4Micro

@testset "HMC random starts use the supplied seed" begin
    data_rng = MersenneTwister(551)
    X = exp.(randn(data_rng, 24, 5))
    times = collect(1.0:24.0)
    events = ones(Int, 24)
    prior = RegularizedHorseshoePrior(p0 = 2.0)
    for constraint in (:prox, :soft_center, :exact), noncentered in (false, true), chains in (1, 4)
        options = (; prior, sampler = :hmc, constraint, noncentered, chains,
            init = :rand, n_samples = 6, warmup = 6, hmc_n_leapfrog = 3, progress = false)
        Random.seed!(813)
        first = sample_bayes_cox(X, times, events; options..., seed = 94, multichains = true)
        Random.seed!(909)
        rand(17) # Ambient random draws must not alter an explicitly seeded fit.
        second = sample_bayes_cox(X, times, events; options..., seed = 94, multichains = true)
        @test posterior_beta_draws(first) == posterior_beta_draws(second)
        @test posterior_logposterior_draws(first) == posterior_logposterior_draws(second)
        different = sample_bayes_cox(X, times, events; options..., seed = 95, multichains = true)
        @test posterior_beta_draws(first) != posterior_beta_draws(different)
        if chains == 4
            serial = sample_bayes_cox(X, times, events; options..., seed = 94, multichains = false)
            @test posterior_beta_draws(first) == posterior_beta_draws(serial)
            @test posterior_logposterior_draws(first) == posterior_logposterior_draws(serial)
        end
    end
end
