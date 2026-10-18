using Test, Random, LinearAlgebra, BayesCox4Micro

@testset "Prox-SSLASSO feasible random initialization" begin
    X = [.2 .3 .5; .5 .1 .4; .1 .6 .3; .4 .4 .2; .3 .5 .2; .2 .6 .2]
    data = prepare_cox_data(X, collect(1.:6.), [1,1,0,1,1,0]; standardize=true)
    prior = SpikeSlabLassoPrior(spike_scale=.04, slab_scale=.5, proximal=true)
    for constraint in (:prox, :soft_center, :none, :exact), init in (:zero, :rand)
        gamma = BitVector([true, false, true])
        base = CoxProxSSLassoTarget(data, prior, gamma, Ref(.5);
            constraint=constraint == :exact ? :none : constraint,
            prox_sum0_lambda=1e-3, prox_l1_lambda=1e-6,
            alpha_prior_shape=1., alpha_prior_scale=3.,
            prox_root_tol=1e-10, prox_root_maxiter=100, init)
        target = constraint == :exact ? BayesCox4Micro.CoxExactZeroTarget(base) : base
        u = initial_state(target, MersenneTwister(41))
        @test u == initial_state(target, MersenneTwister(41))
        full = BayesCox4Micro.sample_state(target, u)
        @test isfinite(logdensity(target, u))
        @test all(isfinite, gradient!(zeros(length(u)), target, u))
        if init == :zero
            @test full == zeros(4)
        else
            @test u != initial_state(target, MersenneTwister(42))
            weights = BayesCox4Micro._ssl_weights(prior, gamma)
            @test dot(weights, abs.(full[1:3])) < exp(full[4])
            a = constraint == :exact ? 1. ./ data.col_scale : ones(3)
            constraint != :none && @test abs(dot(a, full[1:3])) < 1e-12
        end
    end
end

@testset "Matrix API preserves ordinary SSLASSO initialization" begin
    X = [.2 .3 .5; .5 .1 .4; .1 .6 .3; .4 .4 .2; .3 .5 .2; .2 .6 .2]
    time, event = collect(1.:6.), [1,1,0,1,1,0]
    data = prepare_cox_data(X, time, event)
    prior = SpikeSlabLassoPrior(spike_scale=.04, slab_scale=.5)
    opts = (; prior, sampler=:nuts_explicit, init=:rand, seed=41, chains=1,
        warmup=2, n_samples=4, max_depth=3, progress=false)
    from_matrix = sample_bayes_cox(X, time, event; opts...)
    from_data = sample_bayes_cox(data; opts...)
    @test from_matrix.samples.draws == from_data.samples.draws
end

@testset "Prox-SSLASSO forwards init through all HMC/NUTS chain paths" begin
    X = [.2 .3 .5; .5 .1 .4; .1 .6 .3; .4 .4 .2; .3 .5 .2; .2 .6 .2]
    prior = SpikeSlabLassoPrior(spike_scale=.04, slab_scale=.5, proximal=true)
    for sampler in (:hmc, :nuts_explicit), constraint in (:prox, :exact), chains in (1, 2)
        function fit(init; parallel=false)
            sample_bayes_cox(X, collect(1.:6.), [1,1,0,1,1,0];
                prior, sampler, constraint, init, chains, parallel_chains=parallel,
                seed=41, n_samples=4, warmup=2, hmc_n_leapfrog=3, max_depth=3,
                progress=false, prox_l1_lambda=1e-6, prox_sum0_lambda=1e-3,
                alpha_prior_shape=1., alpha_prior_scale=3.)
        end
        samples(result) = chains == 1 ? [result.samples] : result.samples.chains
        zero, random, repeat = samples(fit(:zero)), samples(fit(:rand)), samples(fit(:rand))
        @test all(z.draws != r.draws for (z,r) in zip(zero, random))
        @test all(r.draws == s.draws for (r,s) in zip(random, repeat))
        if chains == 2
            parallel = samples(fit(:rand; parallel=true))
            @test random[1].draws != random[2].draws
            @test all(r.draws == s.draws for (r,s) in zip(random, parallel))
        end
    end
end
