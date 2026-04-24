using Test
using Random
using BayesCox4Micro

@testset "sum-to-zero Moreau helpers" begin
    beta = [1.0, -2.0, 4.0, 0.5]
    proj = project_sum0(beta)
    @test abs(sum(proj)) < 1e-12
    lambda = 1e-3
    @test sum0_moreau(beta, lambda) ≈ 0.5 * sum(beta)^2 / (lambda * length(beta))
    @test sum0_moreau_gradient(beta, lambda) ≈ fill(sum(beta) / (lambda * length(beta)), length(beta))
end

@testset "l1 epigraph projection" begin
    beta = [0.1, -0.2, 0.3]
    inside = project_l1_epigraph(beta, 1.0)
    @test inside.beta ≈ beta
    @test inside.alpha ≈ 1.0

    beta2 = [3.0, -1.0, 0.5]
    alpha = 1.0
    proj = project_l1_epigraph(beta2, alpha)
    @test sum(abs, proj.beta) <= proj.alpha + 1e-8
    residual = sum(max(abs(b) - proj.nu, 0.0) for b in beta2) - proj.nu - alpha
    @test abs(residual) < 1e-8
    @test isfinite(l1_moreau(beta2, alpha, 1e-3))
end

@testset "weighted l1 epigraph projection" begin
    beta = [2.0, -0.5, 1.0]
    weights = ones(3)
    plain = project_l1_epigraph(beta, 0.7)
    weighted = project_weighted_l1_epigraph(beta, 0.7, weights)
    @test weighted.beta ≈ plain.beta
    @test weighted.alpha ≈ plain.alpha

    weights2 = [2.0, 0.5, 1.5]
    proj = project_weighted_l1_epigraph(beta, 0.8, weights2)
    @test sum(weights2 .* abs.(proj.beta)) <= proj.alpha + 1e-8
    @test isfinite(weighted_l1_moreau(beta, 0.8, weights2, 1e-3))
end

@testset "Cox score gradient finite difference" begin
    X = [0.2 0.3 0.5;
         0.1 0.6 0.3;
         0.4 0.2 0.4;
         0.3 0.3 0.4;
         0.5 0.2 0.3]
    time = [1.0, 2.0, 3.0, 4.0, 5.0]
    event = [1, 0, 1, 1, 0]
    data = prepare_cox_data(X, time, event; transform = :none)
    beta = [0.2, -0.1, 0.05]
    grad = cox_score_gradient(data, beta)
    epsfd = 1e-6
    for j in eachindex(beta)
        e = zeros(length(beta))
        e[j] = epsfd
        fd = (cox_logpartiallikelihood(data, beta .+ e) - cox_logpartiallikelihood(data, beta .- e)) / (2epsfd)
        @test grad[j] ≈ fd atol=1e-5 rtol=1e-5
    end
end

@testset "MALA samplers short chains" begin
    Random.seed!(12)
    X = rand(12, 4) .+ 0.1
    time = collect(1.0:12.0)
    event = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1]

    hs = sample_bayes_cox(
        X,
        time,
        event;
        prior = HorseshoePrior(p0 = 1.0),
        sampler = :mala,
        constraint = :soft_center,
        n_samples = 4,
        warmup = 4,
        mala_step_size = 1e-5,
        progress = false,
        seed = 1,
    )
    @test size(posterior_beta_draws(hs)) == (4, 4)
    @test 0 <= hs.samples.acceptance_rate <= 1

    lasso = sample_bayes_cox(
        X,
        time,
        event;
        prior = BayesianLassoPrior(),
        sampler = :mala,
        constraint = :prox,
        n_samples = 4,
        warmup = 4,
        mala_step_size = 1e-6,
        progress = false,
        seed = 2,
    )
    @test size(posterior_beta_draws(lasso)) == (4, 4)
    @test length(posterior_logposterior_draws(lasso)) == 4

    sslasso = sample_bayes_cox(
        X,
        time,
        event;
        prior = SpikeSlabLassoPrior(),
        sampler = :mala,
        constraint = :prox,
        n_samples = 4,
        warmup = 4,
        mala_step_size = 1e-6,
        progress = false,
        seed = 3,
    )
    @test size(posterior_beta_draws(sslasso)) == (4, 4)
    @test length(sslasso_inclusion_probabilities(sslasso)) == 4

    mh = sample_bayes_cox(
        X,
        time,
        event;
        prior = BayesianLassoPrior(lambda = 1.0),
        sampler = :mh,
        constraint = :none,
        n_samples = 3,
        warmup = 3,
        progress = false,
        seed = 4,
    )
    @test size(posterior_beta_draws(mh)) == (3, 4)
end
