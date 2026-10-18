using Test, Random, LinearAlgebra, BayesCox4Micro

@testset "Sampling wall-clock partitions" begin
    B = BayesCox4Micro
    timed = (time = 12.0, bytes = 128, gctime = 0.5)
    # Serial: warmup [1,3] and [6,9], retained/overhead [3,6] and [9,13].
    serial = B._SamplingPhaseClock(UInt64(1e9), UInt64[3e9, 9e9], UInt64[6e9, 12e9])
    d = B._phase_diagnostics(timed, serial, 10, false)
    @test d.warmup_seconds == 5.0
    @test d.sample_seconds == 7.0
    # Parallel: use the last warmup boundary, not a sum of overlapping chains.
    parallel = B._SamplingPhaseClock(UInt64(1e9), UInt64[3e9, 4e9], UInt64[10e9, 12e9])
    d = B._phase_diagnostics(timed, parallel, 10, true)
    @test d.warmup_seconds == 3.0
    @test d.sample_seconds == 9.0
    @test d.warmup_seconds + d.sample_seconds == d.total_seconds
    @test d.allocated_bytes == 128
    @test d.gc_seconds == 0.5
    empty = B._SamplingPhaseClock(2)
    d = B._phase_diagnostics(timed, empty, 0, true)
    @test d.warmup_seconds == 0.0
    @test d.sample_seconds == d.total_seconds == 12.0
    d = B._phase_diagnostics(timed, empty, 10, true)
    @test ismissing(d.warmup_seconds)
    @test ismissing(d.sample_seconds)
end

@testset "HMC phase timing without progress output" begin
    B = BayesCox4Micro
    rng = MersenneTwister(927)
    X = exp.(randn(rng, 20, 4))
    time, event = collect(1.0:20.0), ones(Int, 20)
    old_blas = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        for (method, constraint) in (("cox-rhs", :prox), ("cox-rhs", :soft_center),
                                     ("prox-sslasso", :prox)), warmup in (0, 4)
            prior = cox_paper_prior(method; p0 = 2.0)
            options = (; prior, constraint, sampler = :hmc, n_samples = 6, warmup,
                thin = 2, hmc_n_leapfrog = 2, hmc_step_size = 1e-3,
                seed = 29, progress = false)
            results = Dict()
            for (chains, multichains) in ((1, false), (4, false), (4, true))
                result = sample_bayes_cox(X, time, event; options..., chains, multichains)
                results[(chains, multichains)] = result
                d = sampler_diagnostics(result)
                @test isfinite(d.warmup_seconds)
                @test d.sample_seconds > 0
                @test d.warmup_seconds + d.sample_seconds ≈ d.total_seconds
                @test (warmup == 0 ? d.warmup_seconds == 0 : d.warmup_seconds > 0)
                @test size(posterior_beta_draws(result)) == (6chains, 4)
                transitions = B._result_transitions(result)
                @test count(==("warmup"), transitions.phase) == warmup * chains
                @test count(==("sample"), transitions.phase) == 12chains
            end
            @test posterior_beta_draws(results[(4, false)]) == posterior_beta_draws(results[(4, true)])
            @test posterior_logposterior_draws(results[(4, false)]) == posterior_logposterior_draws(results[(4, true)])
        end

        # The timing adaptor must preserve the uninstrumented AdvancedHMC path.
        data = prepare_cox_data(X, time, event)
        prior = cox_paper_prior("cox-rhs"; p0 = 2.0)
        samples = map((nothing, B._SamplingPhaseClock(1))) do clock
            target = B._make_explicit_nuts_target(data, prior;
                constraint = :prox, prox_sum0_lambda = 1e-3)
            chain_rng = MersenneTwister(86)
            initial = B._generic_explicit_initial(prior, data, nothing; rng = chain_rng)
            clock === nothing || B._start_sampling_clock!(clock)
            B._sample_explicit_hmc_one(target; n_samples = 6, warmup = 4, thin = 2,
                rng = chain_rng, initial, target_accept = 0.8, hmc_n_leapfrog = 2,
                hmc_step_size = 1e-3, progress = false, phase_clock = clock)
        end
        @test samples[1].draws == samples[2].draws
        @test samples[1].logposterior == samples[2].logposterior
        @test isequal(samples[1].transitions, samples[2].transitions)

        for method in ("cox-rhs", "prox-sslasso"), multichains in (false, true)
            result = sample_clinical_cox(X, time, event, reshape(repeat([0., 1.], 10), :, 1);
                prior = cox_paper_prior(method; p0 = 2.0), n_samples = 6, warmup = 4,
                chains = 2, multichains, hmc_n_leapfrog = 2, hmc_step_size = 1e-3,
                seed = 30, progress = false)
            d = sampler_diagnostics(result)
            @test d.warmup_seconds > 0
            @test d.sample_seconds > 0
            @test d.warmup_seconds + d.sample_seconds ≈ d.total_seconds
        end
    finally
        BLAS.set_num_threads(old_blas)
    end
end
