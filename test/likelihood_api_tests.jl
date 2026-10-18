using Test
using BayesCox4Micro

@testset "Cox likelihood approximation API is removed" begin
    for name in (:CoxApproximation, :build_cox_approximation!,
                 :cox_logpartiallikelihood_approx, :cox_approx_score_gradient!,
                 :cox_approx_loglik_and_gradient!)
        @test !isdefined(BayesCox4Micro, name)
    end

    X = [0.2 0.8; 0.4 0.6; 0.7 0.3; 0.5 0.5]
    time, event = [1., 2., 2., 4.], [1, 1, 0, 1]
    data = prepare_cox_data(X, time, event)
    for sampler in (:mh, :mala, :ula, :gibbs, :hmc, :nuts, :nuts_explicit, :nuts_turing)
        for removed in ((approx = false,), (approx = true,), (approx_point = :zero,))
            @test_throws MethodError sample_bayes_cox(data;
                sampler, chains = 1, n_samples = 1, warmup = 0, progress = false, removed...)
            @test_throws MethodError sample_bayes_cox(X, time, event;
                sampler, chains = 1, n_samples = 1, warmup = 0, progress = false, removed...)
        end
    end
end
