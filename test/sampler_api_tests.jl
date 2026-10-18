using Test
using BayesCox4Micro

@testset "Removed Langevin samplers and options are rejected" begin
    X = [0.2 0.8; 0.4 0.6; 0.7 0.3; 0.5 0.5]
    time, event = [1., 2., 2., 4.], [1, 1, 0, 1]
    data = prepare_cox_data(X, time, event)
    options = (chains = 1, n_samples = 1, warmup = 0, progress = false)

    for sampler in (:mala, :ula)
        @test_throws ArgumentError sample_bayes_cox(data; sampler, options...)
        @test_throws ArgumentError sample_bayes_cox(X, time, event; sampler, options...)
    end

    for sampler in (:mh, :gibbs, :hmc, :nuts, :nuts_explicit, :nuts_turing)
        for removed in ((mala_step_size = 1e-4,), (mala_precondition = true,),
                        (mala_stepsize_schedule = :adaptive_sgld,))
            @test_throws MethodError sample_bayes_cox(data; sampler, options..., removed...)
            @test_throws MethodError sample_bayes_cox(X, time, event; sampler, options..., removed...)
        end
    end
end
