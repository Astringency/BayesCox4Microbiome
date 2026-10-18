using Test
using Random
using BayesCox4Micro
using CSV
using DataFrames
using ForwardDiff
using JLD2
using MCMCChains
using Statistics

const B = BayesCox4Micro

include("data_preprocessing_tests.jl")

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

@testset "Cox score gradient correctness" begin
    X = [0.2 0.3 0.5;
         0.1 0.6 0.3;
         0.4 0.2 0.4;
         0.3 0.3 0.4;
         0.5 0.2 0.3]
    time = [1.0, 2.0, 3.0, 3.0, 5.0]
    event = [1, 0, 1, 1, 0]
    data = prepare_cox_data(X, time, event; transform = :none)
    @test data.tie_method == :efron
    beta = [0.2, -0.1, 0.05]
    grad = cox_score_gradient(data, beta)
    work = CoxGradientWorkspace(data)
    grad_prealloc = similar(grad)
    cox_score_gradient!(grad_prealloc, data, beta, work)
    grad_ad = ForwardDiff.gradient(b -> cox_logpartiallikelihood(data, b), beta)
    @test grad ≈ grad_prealloc atol=1e-12 rtol=1e-12
    @test grad ≈ grad_ad atol=1e-8 rtol=1e-8
    epsfd = 1e-6
    for j in eachindex(beta)
        e = zeros(length(beta))
        e[j] = epsfd
        fd = (cox_logpartiallikelihood(data, beta .+ e) - cox_logpartiallikelihood(data, beta .- e)) / (2epsfd)
        @test grad[j] ≈ fd atol=1e-5 rtol=1e-5
    end

    bad_grad = zeros(Float64, length(beta))
    bad_lp = B.cox_logpartiallikelihood_and_score_gradient!(
        bad_grad,
        data,
        fill(Inf, length(beta)),
        CoxGradientWorkspace(data),
    )
    @test bad_lp == -Inf
    @test any(isnan, bad_grad)
end

@testset "Cox Efron and Breslow ties" begin
    X = [0.2 0.3;
         0.1 0.6;
         0.4 0.2;
         0.3 0.3]
    time = [1.0, 2.0, 2.0, 3.0]
    event = [1, 1, 1, 0]
    beta = [0.25, -0.2]
    data_efron = prepare_cox_data(X, time, event; transform = :none)
    data_breslow = prepare_cox_data(X, time, event; transform = :none, tie_method = :breslow)

    eta = X * beta
    risk_t2 = sum(exp.(eta[time .>= 2.0]))
    tied_t2 = sum(exp.(eta[[2, 3]]))
    risk_t1 = sum(exp.(eta[time .>= 1.0]))
    manual_efron = eta[2] + eta[3] - log(risk_t2) -
        log(risk_t2 - 0.5 * tied_t2) + eta[1] - log(risk_t1)
    manual_breslow = eta[2] + eta[3] - 2log(risk_t2) + eta[1] - log(risk_t1)

    @test cox_logpartiallikelihood(data_efron, beta) ≈ manual_efron atol=1e-12 rtol=1e-12
    @test cox_logpartiallikelihood(data_breslow, beta) ≈ manual_breslow atol=1e-12 rtol=1e-12
    @test cox_logpartiallikelihood(data_efron, beta) != cox_logpartiallikelihood(data_breslow, beta)
    @test sum(cox_loglik_events(data_efron, beta)) ≈ cox_logpartiallikelihood(data_efron, beta)

    grad = cox_score_gradient(data_efron, beta)
    grad_ad = ForwardDiff.gradient(b -> cox_logpartiallikelihood(data_efron, b), beta)
    @test grad ≈ grad_ad atol=1e-8 rtol=1e-8
end

@testset "adaptive CAR graph helpers" begin
    W = zeros(6, 6)
    W[1, 2] = W[2, 1] = 2.0
    W[2, 3] = W[3, 2] = 4.0
    W[4, 5] = W[5, 4] = 1.0
    weights = B._prepare_car_weights(W, 6)
    degrees = B._car_weighted_degrees(weights)
    @test degrees ≈ [2.0, 6.0, 4.0, 1.0, 1.0, 0.0]
    active = [2.0, 6.0, 4.0, 1.0, 1.0]
    @test B._car_degree_scale(weights; method = :mean_active) ≈ mean(active)
    @test B._car_degree_scale(weights; method = :median_active) ≈ median(active)
    @test B._car_degree_scale(weights; method = :q90_active) ≈ quantile(active, 0.9)
    @test B._car_degree_scale(B._prepare_car_weights(zeros(3, 3), 3)) == 1.0
    @test_throws ArgumentError B._car_degree_scale(weights; method = :unknown)

    W3 = zeros(4, 4)
    W3[1, 2] = W3[2, 1] = 1.0
    W3[2, 3] = W3[3, 2] = 1.0
    @test B._car_rank(B._prepare_car_weights(W3, 4)) == 2
    @test B._car_rank(weights) == 3
    @test B._car_rank(B._prepare_car_weights(zeros(4, 4), 4)) == 0
end

function _explicit_initial_for_test(prior, data)
    p = size(data.z, 2)
    n_events = count(!=(0), data.event)
    return prior isa RegularizedHorseshoePrior ?
        initial_state(prior, p, n_events; n_obs = size(data.z, 1)) :
        initial_state(prior, p, n_events)
end

@testset "explicit NUTS posterior gradients" begin
    Random.seed!(21)
    X = rand(8, 3) .+ 0.2
    time = collect(1.0:8.0)
    event = [1, 0, 1, 1, 0, 1, 0, 1]
    data = prepare_cox_data(X, time, event; transform = :none)
    cases = [
        ("hs", cox_paper_prior("hs"; p0 = 1.0)),
        ("rhs", cox_paper_prior("rhs"; p0 = 1.0)),
        ("new-rhs", cox_paper_prior("new-rhs"; p0 = 1.0)),
        ("est-rhs", cox_paper_prior("est-rhs"; p0 = 1.0, X = X, time = time, event = event, transform = :none)),
        ("cox-hs", cox_paper_prior("cox-hs"; p0 = 1.0)),
        ("cox-rhs", cox_paper_prior("cox-rhs"; p0 = 1.0)),
        ("lasso", cox_paper_prior("lasso"; p0 = 1.0)),
        ("lasso-tuned", cox_paper_prior("lasso"; p0 = 1.0, lasso_lambda2_shape = 0.75)),
        ("sslasso", cox_paper_prior("sslasso"; p0 = 1.0)),
    ]
    @test !cox_paper_prior("lasso").proximal
    @test cox_paper_prior("prox-lasso").proximal
    @test !cox_paper_prior("sslasso").proximal
    @test cox_paper_prior("prox-sslasso").proximal
    tuned_rhs = cox_paper_prior("cox-rhs"; p0 = 1.0, rhs_slab_df = 2.0, rhs_slab_s2 = 0.5)
    @test tuned_rhs.slab_df == 2.0
    @test tuned_rhs.slab_s2 == 0.5
    tuned_hs = cox_paper_prior("cox-hs"; p0 = 2.0, hs_local_scale = 0.5)
    @test tuned_hs.p0 == 2.0
    @test tuned_hs.local_scale == 0.5
    tuned_rhs_local = cox_paper_prior("cox-rhs"; p0 = 2.0, rhs_local_scale = 0.25)
    @test tuned_rhs_local.p0 == 2.0
    @test tuned_rhs_local.local_scale == 0.25
    tuned_lasso = cox_paper_prior("lasso"; lasso_lambda2_shape = 2.0, lasso_lambda2_rate = 0.5)
    @test tuned_lasso.lambda2_shape == 2.0
    @test tuned_lasso.lambda2_rate == 0.5
    tuned_ssl = cox_paper_prior("sslasso"; sslasso_spike_scale = 0.02, sslasso_slab_scale = 1.0, sslasso_theta_a = 2.0, sslasso_theta_b = 8.0)
    @test tuned_ssl.spike_scale == 0.02
    @test tuned_ssl.slab_scale == 1.0
    @test tuned_ssl.theta_a == 2.0
    @test tuned_ssl.theta_b == 8.0
    for (name, prior) in cases
        target = B._make_explicit_nuts_target(
            data,
            prior;
            constraint = :soft_center,
            prox_sum0_lambda = 1e-3,
            prox_l1_lambda = 1e-3,
            alpha_prior_shape = 1.0,
            alpha_prior_scale = 0.1,
            prox_root_tol = 1e-10,
            prox_root_maxiter = 100,
        )
        u = _explicit_initial_for_test(prior, data)
        u .+= 0.05 .* randn(length(u))
        grad = zeros(Float64, length(u))
        gradient!(grad, target, u)
        grad_ad = ForwardDiff.gradient(x -> logdensity(target, x), u)
        @test maximum(abs.(grad .- grad_ad)) < 1e-7
        @test logdensity_and_gradient!(similar(grad), target, u)[1] ≈ logdensity(target, u)
    end

    car_weights = B._prepare_car_weights([0.0 1.0 0.3; 1.0 0.0 0.2; 0.3 0.2 0.0], 3)
    prior = HorseshoePrior(p0 = 1.0)
    target = B._make_explicit_nuts_target(
        data,
        prior;
        constraint = :none,
        prox_sum0_lambda = 1e-3,
        prox_l1_lambda = 1e-3,
        alpha_prior_shape = 1.0,
        alpha_prior_scale = 0.1,
        prox_root_tol = 1e-10,
        prox_root_maxiter = 100,
        car_weights = car_weights,
        use_car = true,
        car_precision = 1.7,
    )
    u = _explicit_initial_for_test(prior, data) .+ 0.05 .* randn(length(state_names(target)))
    grad = zeros(Float64, length(u))
    gradient!(grad, target, u)
    grad_ad = ForwardDiff.gradient(x -> logdensity(target, x), u)
    @test maximum(abs.(grad .- grad_ad)) < 1e-7

    rhs_prior = cox_paper_prior("cox-rhs"; p0 = 1.0)
    rhs_u = _explicit_initial_for_test(rhs_prior, data)
    rhs_u .+= 0.05 .* randn(length(rhs_u))
    rhs_grad = zeros(Float64, length(rhs_u))
    rhs_work = CoxGradientWorkspace(data)
    rhs_lp, _ = B.logposterior_and_gradient!(
        rhs_grad,
        rhs_u,
        data,
        rhs_prior,
        rhs_work;
        constraint = :none,
        car_weights = car_weights,
        use_car = true,
        car_precision = 0.05,
    )
    @test isfinite(rhs_lp)
    @test all(isfinite, rhs_grad)

    adaptive_prior = HorseshoePrior(p0 = 1.0)
    adaptive_target = B._make_explicit_nuts_target(
        data,
        adaptive_prior;
        constraint = :none,
        prox_sum0_lambda = 1e-3,
        prox_l1_lambda = 1e-3,
        alpha_prior_shape = 1.0,
        alpha_prior_scale = 0.1,
        prox_root_tol = 1e-10,
        prox_root_maxiter = 100,
        car_weights = car_weights,
        use_car = true,
        sample_car_strength = true,
        car_strength_prior_a = 1.0,
        car_strength_prior_b = 1.0,
        car_degree_scale = :mean_active,
        car_kappa_init = 0.5,
    )
    adaptive_u = initial_state(
        adaptive_target.prior,
        size(data.z, 2),
        count(!=(0), data.event);
        car_weights = car_weights,
        sample_car_strength = true,
        car_kappa_init = 0.5,
    )
    adaptive_u .+= 0.02 .* randn(length(adaptive_u))
    adaptive_grad = zeros(Float64, length(adaptive_u))
    adaptive_lp, _ = logdensity_and_gradient!(adaptive_grad, adaptive_target, adaptive_u)
    adaptive_names = state_names(adaptive_target)
    eta_idx = findfirst(==("logit_car_kappa"), adaptive_names)
    @test eta_idx !== nothing
    @test isfinite(adaptive_lp)
    @test all(isfinite, adaptive_grad)
    fd_eps = 1e-6
    step = zeros(length(adaptive_u))
    step[eta_idx] = fd_eps
    fd_eta = (logdensity(adaptive_target, adaptive_u .+ step) -
        logdensity(adaptive_target, adaptive_u .- step)) / (2fd_eps)
    @test adaptive_grad[eta_idx] ≈ fd_eta atol=1e-5 rtol=1e-5

    rhs_adaptive = cox_paper_prior("cox-rhs"; p0 = 1.0)
    rhs_adaptive_u = initial_state(
        rhs_adaptive,
        size(data.z, 2),
        count(!=(0), data.event);
        car_weights = car_weights,
        sample_car_strength = true,
        car_kappa_init = 0.5,
    )
    rhs_adaptive_grad = zeros(Float64, length(rhs_adaptive_u))
    rhs_adaptive_lp, _ = B.logposterior_and_gradient!(
        rhs_adaptive_grad,
        rhs_adaptive_u,
        data,
        rhs_adaptive,
        CoxGradientWorkspace(data);
        constraint = :none,
        car_weights = car_weights,
        use_car = true,
        sample_car_strength = true,
        car_strength_prior_a = 1.0,
        car_strength_prior_b = 1.0,
        car_degree_scale = :mean_active,
        car_kappa_init = 0.5,
    )
    @test isfinite(rhs_adaptive_lp)
    @test all(isfinite, rhs_adaptive_grad)
    @test length(rhs_adaptive_grad) == length(B._state_names(rhs_adaptive, size(data.z, 2); sample_car_strength = true))

    rhs_adaptive_nc_target = B._make_explicit_nuts_target(
        data,
        rhs_adaptive;
        constraint = :none,
        prox_sum0_lambda = 1e-3,
        prox_l1_lambda = 1e-3,
        alpha_prior_shape = 1.0,
        alpha_prior_scale = 0.1,
        prox_root_tol = 1e-10,
        prox_root_maxiter = 100,
        car_weights = car_weights,
        use_car = true,
        sample_car_strength = true,
        car_strength_prior_a = 1.0,
        car_strength_prior_b = 1.0,
        car_degree_scale = :mean_active,
        car_kappa_init = 0.5,
        noncentered = true,
    )
    rhs_adaptive_nc_u = initial_state(rhs_adaptive_nc_target)
    rhs_adaptive_nc_u[end] = -1000.0
    rhs_adaptive_nc_grad = zeros(Float64, length(rhs_adaptive_nc_u))
    rhs_adaptive_nc_lp, _ = logdensity_and_gradient!(
        rhs_adaptive_nc_grad,
        rhs_adaptive_nc_target,
        rhs_adaptive_nc_u,
    )
    @test isfinite(rhs_adaptive_nc_lp)
    @test all(isfinite, rhs_adaptive_nc_grad)
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
    @test !("eta" in lasso.samples.names)

    lasso_multi = sample_bayes_cox(
        X,
        time,
        event;
        prior = BayesianLassoPrior(),
        sampler = :mala,
        constraint = :prox,
        n_samples = 2,
        warmup = 2,
        chains = 2,
        mala_step_size = 1e-6,
        progress = false,
        seed = 22,
    )
    @test size(result_to_chains(lasso_multi), 3) == 2
    @test lasso_multi.samples isa MultiMHSamples
    @test size(posterior_beta_draws(lasso_multi)) == (4, 4)
    @test length(posterior_logposterior_draws(lasso_multi)) == 4

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

    car_gaussian = sample_bayes_cox(
        X,
        time,
        event;
        prior = HorseshoePrior(p0 = 1.0),
        sampler = :mh,
        use_car = true,
        car_precision = 0.05,
        n_samples = 2,
        warmup = 1,
        chains = 1,
        progress = false,
        seed = 16,
    )
    @test size(posterior_beta_draws(car_gaussian)) == (2, 4)

    W = zeros(4, 4)
    W[1, 2] = W[2, 1] = 1.0
    W[2, 3] = W[3, 2] = 0.5
    car_adaptive = sample_bayes_cox(
        X,
        time,
        event;
        prior = cox_paper_prior("cox-rhs"; p0 = 1.0),
        sampler = :mh,
        use_car = true,
        car_weights = W,
        sample_car_strength = true,
        car_strength_prior_a = 1.0,
        car_strength_prior_b = 1.0,
        car_degree_scale = :mean_active,
        car_kappa_init = 0.5,
        n_samples = 2,
        warmup = 1,
        chains = 1,
        progress = false,
        seed = 18,
    )
    @test "logit_car_kappa" in car_adaptive.samples.names
    @test size(posterior_beta_draws(car_adaptive)) == (2, 4)

    ula = sample_bayes_cox(
        X,
        time,
        event;
        prior = BayesianLassoPrior(),
        sampler = :ula,
        constraint = :prox,
        n_samples = 2,
        warmup = 2,
        mala_step_size = 1e-6,
        progress = false,
        seed = 5,
    )
    @test size(posterior_beta_draws(ula)) == (2, 4)
    @test length(posterior_logposterior_draws(ula)) == 2

    ula_multi = sample_bayes_cox(
        X,
        time,
        event;
        prior = BayesianLassoPrior(),
        sampler = :ula,
        constraint = :prox,
        n_samples = 2,
        warmup = 1,
        chains = 2,
        mala_step_size = 1e-6,
        progress = false,
        seed = 6,
    )
    @test ula_multi.samples isa MultiMHSamples
    @test size(result_to_chains(ula_multi), 3) == 2
    @test size(posterior_beta_draws(ula_multi)) == (4, 4)

    mala_prox_soft_center = sample_bayes_cox(
        X,
        time,
        event;
        prior = cox_paper_prior("prox-lasso"),
        sampler = :mala,
        constraint = :soft_center,
        n_samples = 2,
        warmup = 1,
        mala_step_size = 1e-6,
        progress = false,
        seed = 14,
    )
    @test size(posterior_beta_draws(mala_prox_soft_center)) == (2, 4)
    @test mala_prox_soft_center.samples.names[end] == "eta"

    ula_prox_soft_center = sample_bayes_cox(
        X,
        time,
        event;
        prior = cox_paper_prior("prox-lasso"),
        sampler = :ula,
        constraint = :soft_center,
        n_samples = 2,
        warmup = 1,
        mala_step_size = 1e-6,
        progress = false,
        seed = 15,
    )
    @test size(posterior_beta_draws(ula_prox_soft_center)) == (2, 4)
    @test ula_prox_soft_center.samples.names[end] == "eta"
end

@testset "NUTS backend smoke tests" begin
    Random.seed!(120)
    X = rand(10, 3) .+ 0.1
    time = collect(1.0:10.0)
    event = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]

    nuts_turing = sample_bayes_cox(
        X,
        time,
        event;
        prior = BayesianLassoPrior(),
        sampler = :nuts,
        nuts_backend = :turing,
        constraint = :prox,
        n_samples = 1,
        warmup = 1,
        chains = 1,
        progress = false,
        seed = 7,
    )
    @test size(posterior_beta_draws(nuts_turing)) == (1, 3)

    nuts_explicit = sample_bayes_cox(
        X,
        time,
        event;
        prior = cox_paper_prior("prox-lasso"),
        sampler = :nuts,
        nuts_backend = :explicit,
        constraint = :prox,
        n_samples = 2,
        warmup = 1,
        chains = 2,
        progress = false,
        seed = 8,
        parallel_chains = true,
    )
    @test nuts_explicit.samples isa MultiMHSamples
    @test size(result_to_chains(nuts_explicit), 3) == 2
    @test size(posterior_beta_draws(nuts_explicit)) == (4, 3)
    @test length(posterior_logposterior_draws(nuts_explicit)) == 4

    try
        sample_bayes_cox(
            X[:, 1:2],
            time,
            event;
            prior = cox_paper_prior("prox-sslasso"),
            sampler = :nuts,
            nuts_backend = :turing,
            constraint = :prox,
            n_samples = 1,
            warmup = 0,
            chains = 1,
            progress = false,
            seed = 88,
        )
        @test false
    catch err
        @test err isa ArgumentError
        @test occursin("prox-sslasso with Turing NUTS-Gibbs is disabled", sprint(showerror, err))
    end

    hs_explicit = sample_bayes_cox(
        X,
        time,
        event;
        prior = HorseshoePrior(p0 = 1.0),
        sampler = :nuts_explicit,
        constraint = :soft_center,
        n_samples = 1,
        warmup = 1,
        chains = 1,
        max_depth = 2,
        progress = false,
        seed = 9,
    )
    @test size(posterior_beta_draws(hs_explicit)) == (1, 3)

    hs_explicit_nc = sample_bayes_cox(
        X,
        time,
        event;
        prior = HorseshoePrior(p0 = 1.0),
        sampler = :nuts_explicit,
        noncentered = true,
        constraint = :soft_center,
        n_samples = 1,
        warmup = 1,
        chains = 1,
        max_depth = 2,
        progress = false,
        seed = 91,
    )
    @test hs_explicit_nc.samples isa MHSamples
    @test hs_explicit_nc.samples.names[1:3] == ["beta[1]", "beta[2]", "beta[3]"]
    @test size(posterior_beta_draws(hs_explicit_nc)) == (1, 3)

    rhs_turing_nc = sample_bayes_cox(
        X,
        time,
        event;
        prior = cox_paper_prior("cox-rhs"; p0 = 1.0),
        sampler = :nuts_turing,
        noncentered = true,
        constraint = :soft_center,
        n_samples = 1,
        warmup = 1,
        chains = 1,
        progress = false,
        seed = 92,
    )
    @test rhs_turing_nc.samples isa MHSamples
    @test rhs_turing_nc.samples.names[1:3] == ["beta[1]", "beta[2]", "beta[3]"]
    @test size(posterior_beta_draws(rhs_turing_nc)) == (1, 3)

    hs_multi = sample_bayes_cox(
        X,
        time,
        event;
        prior = HorseshoePrior(p0 = 1.0),
        sampler = :nuts_explicit,
        constraint = :soft_center,
        n_samples = 1,
        warmup = 1,
        chains = 2,
        max_depth = 2,
        progress = false,
        seed = 10,
    )
    @test hs_multi.samples isa MultiMHSamples
    @test size(posterior_beta_draws(hs_multi)) == (2, 3)

    sslasso_prox = sample_bayes_cox(
        X[:, 1:2],
        time,
        event;
        prior = cox_paper_prior("prox-sslasso"),
        sampler = :nuts_explicit,
        constraint = :prox,
        n_samples = 1,
        warmup = 0,
        chains = 1,
        max_depth = 1,
        progress = false,
        seed = 13,
    )
    @test size(posterior_beta_draws(sslasso_prox)) == (1, 2)
    @test length(sslasso_inclusion_probabilities(sslasso_prox)) == 2
    @test sslasso_prox.samples.names == ["beta[1]", "beta[2]", "eta", "gamma[1]", "gamma[2]", "theta"]
    @test all(isfinite, sslasso_prox.samples.final_step)
    @test all(sslasso_prox.samples.final_step .> 0)

    sslasso_prox_gibbs = sample_bayes_cox(
        X[:, 1:2],
        time,
        event;
        prior = cox_paper_prior("prox-sslasso"),
        sampler = :gibbs,
        constraint = :prox,
        n_samples = 1,
        warmup = 0,
        chains = 1,
        max_depth = 1,
        progress = false,
        seed = 131,
    )
    @test sslasso_prox_gibbs.samples.names == sslasso_prox.samples.names
    @test size(posterior_beta_draws(sslasso_prox_gibbs)) == (1, 2)
end

@testset "explicit HMC smoke tests" begin
    Random.seed!(121)
    X = rand(10, 3) .+ 0.1
    time = collect(1.0:10.0)
    event = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]

    hmc_lasso = sample_bayes_cox(
        X,
        time,
        event;
        prior = cox_paper_prior("prox-lasso"),
        sampler = :hmc,
        constraint = :prox,
        n_samples = 2,
        warmup = 1,
        thin = 2,
        chains = 1,
        hmc_n_leapfrog = 2,
        hmc_step_size = 1e-3,
        progress = false,
        seed = 14,
    )
    @test hmc_lasso.samples isa MHSamples
    @test size(posterior_beta_draws(hmc_lasso)) == (2, 3)
    @test length(posterior_logposterior_draws(hmc_lasso)) == 2
    @test 0 <= hmc_lasso.samples.acceptance_rate <= 1

    hmc_multi = sample_bayes_cox(
        X,
        time,
        event;
        prior = HorseshoePrior(p0 = 1.0),
        sampler = :hmc,
        constraint = :soft_center,
        n_samples = 1,
        warmup = 1,
        chains = 2,
        hmc_n_leapfrog = 1,
        hmc_step_size = 1e-3,
        progress = false,
        seed = 15,
    )
    @test hmc_multi.samples isa MultiMHSamples
    @test size(posterior_beta_draws(hmc_multi)) == (2, 3)

    hmc_ssl_prox = sample_bayes_cox(
        X[:, 1:2],
        time,
        event;
        prior = cox_paper_prior("prox-sslasso"),
        sampler = :hmc,
        constraint = :prox,
        n_samples = 1,
        warmup = 0,
        chains = 1,
        hmc_n_leapfrog = 1,
        hmc_step_size = 1e-3,
        progress = false,
        seed = 16,
    )
    @test size(posterior_beta_draws(hmc_ssl_prox)) == (1, 2)
    @test length(sslasso_inclusion_probabilities(hmc_ssl_prox)) == 2
    @test hmc_ssl_prox.samples.names == ["beta[1]", "beta[2]", "eta", "gamma[1]", "gamma[2]", "theta"]
    @test all(isfinite, hmc_ssl_prox.samples.final_step)
    @test all(hmc_ssl_prox.samples.final_step .> 0)
end

@testset "prox-sslasso block kernel adaptation" begin
    high_accept = B.ProxSSLassoBlockHMCState(
        2;
        step_size = 0.1,
        target_accept = 0.5,
        adapt_every = 2,
    )
    B._prox_ssl_block_record_transition!(high_accept, 0.9, false)
    B._prox_ssl_block_record_transition!(high_accept, 0.9, false)
    @test high_accept.step_size > 0.1

    frozen_step = high_accept.step_size
    B._prox_ssl_block_freeze!(high_accept)
    B._prox_ssl_block_record_transition!(high_accept, 0.0, false)
    B._prox_ssl_block_record_transition!(high_accept, 0.0, false)
    @test high_accept.step_size == frozen_step

    low_accept = B.ProxSSLassoBlockNUTSState(
        2;
        step_size = 0.1,
        target_accept = 0.9,
        adapt_every = 2,
    )
    B._prox_ssl_block_record_transition!(low_accept, 0.1, false)
    B._prox_ssl_block_record_transition!(low_accept, 0.1, false)
    @test low_accept.step_size < 0.1
end

function toy_result(samples)
    X = [0.2 0.3 0.5;
         0.1 0.6 0.3;
         0.4 0.2 0.4;
         0.3 0.3 0.4;
         0.5 0.2 0.3]
    time = [1.0, 2.0, 3.0, 4.0, 5.0]
    event = [1, 0, 1, 1, 0]
    data = prepare_cox_data(X, time, event; transform = :none)
    return CoxSamplerResult(samples, data, BayesianLassoPrior(lambda = 1.0), :mh)
end

@testset "sampler result saving and diagnostics" begin
    draws = [0.1 0.2; 0.2 0.1; 0.3 0.0; 0.4 -0.1; 0.5 -0.2; 0.6 -0.3]
    lps = collect(-6.0:-1.0)
    mh_samples = MHSamples(draws, lps, ["beta[1]", "beta[2]"], 0.5, 0.4, [0.1, 0.1])
    result = toy_result(mh_samples)

    chn = result_to_chains(result)
    @test chn isa MCMCChains.Chains
    @test size(chn, 3) == 1
    @test "logposterior" in string.(names(chn))

    wide = chains_to_wide_dataframe(chn)
    @test all(wide.chain .== 1)
    @test :iter in propertynames(wide)
    @test :draw in propertynames(wide)
    @test Symbol("beta[1]") in propertynames(wide)
    @test Symbol("beta[2]") in propertynames(wide)
    @test :logposterior in propertynames(wide)
    @test nrow(wide) == 6

    mktempdir() do dir
        save_run_outputs(
            result;
            output_dir = dir,
            run_id = "toy",
            metadata = Dict("prior_name" => "lasso", "sampler" => "mh"),
            settings = Dict("n_samples" => 6, "warmup" => 2, "thin" => 1, "seed" => 11),
        )
        @test isfile(joinpath(dir, "samples_wide.csv"))
        @test !isfile(joinpath(dir, "samples_long.csv"))
        @test isfile(joinpath(dir, "metadata.json"))
        @test isfile(joinpath(dir, "diagnostics.json"))
        @test isfile(joinpath(dir, "result_bundle.jld2"))

        bundle = JLD2.load(joinpath(dir, "result_bundle.jld2"), "result_bundle")
        @test haskey(bundle, "samples")
        @test haskey(bundle, "samples_chain")
        @test !haskey(bundle, "samples_long")
        @test nrow(CSV.read(joinpath(dir, "samples_wide.csv"), DataFrame)) == 6

        diag = diagnose_run(dir; formats = "png", top_k_beta = 2, max_lag = 3)
        @test isfile(joinpath(diag.output_dir, "diagnostics_summary.csv"))
        @test isfile(joinpath(diag.output_dir, "traceplot.png"))
        @test isfile(joinpath(diag.output_dir, "density.png"))
        @test isfile(joinpath(diag.output_dir, "acf.png"))
        @test !isfile(joinpath(diag.output_dir, "rhat_scatter.png"))
    end

    mktempdir() do dir
        save_run_outputs(
            result;
            output_dir = dir,
            run_id = "toy_compact",
            save_jld2 = false,
            sample_thin = 2,
        )
        @test isfile(joinpath(dir, "samples_wide.csv"))
        @test !isfile(joinpath(dir, "samples_long.csv"))
        @test !isfile(joinpath(dir, "result_bundle.jld2"))
        @test nrow(CSV.read(joinpath(dir, "samples_wide.csv"), DataFrame)) == 3

        diag = diagnose_run(dir; formats = "png", top_k_beta = 2, max_lag = 1, plots = false)
        @test isfile(joinpath(diag.output_dir, "diagnostics_summary.csv"))
        @test isempty(diag.plot_paths)
        @test !isfile(joinpath(diag.output_dir, "traceplot.png"))
    end
end

@testset "Turing-style Chains saving and multi-chain R-hat plots" begin
    arr = Array{Float64}(undef, 8, 3, 2)
    for c in 1:2, i in 1:8
        arr[i, 1, c] = 0.1i + 0.01c
        arr[i, 2, c] = -0.1i + 0.01c
        arr[i, 3, c] = -i
    end
    chn = Chains(arr, [Symbol("beta[1]"), Symbol("beta[2]"), :logposterior])
    result = toy_result(chn)

    mktempdir() do dir
        save_run_outputs(result; output_dir = dir, run_id = "toy_chains")
        @test isfile(joinpath(dir, "result_bundle.jld2"))
        @test size(result_to_chains(result), 3) == 2

        diag = diagnose_run(dir; formats = "png", top_k_beta = 2, max_lag = 3)
        @test isfile(joinpath(diag.output_dir, "rhat_scatter.png"))
        @test isfile(joinpath(diag.output_dir, "rhat_histogram.png"))
    end
end

@testset "run_simulation CLI CAR flags" begin
    include(joinpath(@__DIR__, "..", "scripts", "run_simulation.jl"))
    parsed = parse_cli_args([
        "--car-precision", "0.05",
        "--car-weight-quantile", "0.75",
        "--sample-car-strength", "true",
        "--car-strength-prior-a", "1.0",
        "--car-strength-prior-b", "1.0",
        "--car-degree-scale", "mean_active",
        "--car-kappa-init", "0.5",
        "--mala-precondition", "true",
        "--mala-precondition-mode", "auto",
        "--mala-precondition-diag", "false",
        "--mala-metric-ridge", "1e-6",
        "--mala-metric-max-cholesky-tries", "8",
        "--mala-stepsize-schedule", "adaptive_sgld",
        "--mala-sgld-c0", "1.0",
        "--mala-sgld-target-accept", "0.5",
        "--mala-sgld-accept-low", "0.4",
        "--mala-sgld-accept-high", "0.6",
        "--mala-sgld-adapt-c0", "true",
        "--mala-sgld-adapt-every", "10",
        "--mala-sgld-c0-min", "1e-12",
        "--mala-sgld-c0-max", "1e12",
    ])
    @test isnothing(require_known_args!(parsed, simulation_cli_valid_keys()))
    @test arg(parsed, "car-precision", 1.0) == 0.05
    @test arg(parsed, "car-weight-quantile", 0.75) == 0.75
    @test arg(parsed, "sample-car-strength", false)
    @test arg(parsed, "car-strength-prior-a", 1.0) == 1.0
    @test arg(parsed, "car-strength-prior-b", 1.0) == 1.0
    @test Symbol(arg(parsed, "car-degree-scale", "mean_active")) == :mean_active
    @test arg(parsed, "car-kappa-init", 0.5) == 0.5
    @test arg(parsed, "mala-precondition", false)
    @test Symbol(arg(parsed, "mala-precondition-mode", "auto")) == :auto
    @test !arg(parsed, "mala-precondition-diag", true)
    @test arg(parsed, "mala-metric-ridge", 1.0) == 1e-6
    @test arg(parsed, "mala-metric-max-cholesky-tries", 1) == 8
    @test Symbol(arg(parsed, "mala-stepsize-schedule", "adaptive_mcmc")) == :adaptive_sgld
    @test arg(parsed, "mala-sgld-c0", 0.0) == 1.0
    @test arg(parsed, "mala-sgld-target-accept", 0.0) == 0.5
    @test arg(parsed, "mala-sgld-accept-low", 0.0) == 0.4
    @test arg(parsed, "mala-sgld-accept-high", 0.0) == 0.6
    @test arg(parsed, "mala-sgld-adapt-c0", false)
    @test arg(parsed, "mala-sgld-adapt-every", 1) == 10
    @test arg(parsed, "mala-sgld-c0-min", 1.0) == 1e-12
    @test arg(parsed, "mala-sgld-c0-max", 1.0) == 1e12

    removed = parse_cli_args(["--car-prior", "gaussian"])
    @test_throws ArgumentError require_known_args!(removed, simulation_cli_valid_keys())
end

include("mala_precondition_tests.jl")
include("simulation_diagnostics_tests.jl")
include("loo_tests.jl")
include("cindex_evaluation_tests.jl")
include("simulation_coverage_tests.jl")
include("exact_zero_tests.jl")
include("revision_diagnostics_tests.jl")
include("simulation_scenario_tests.jl")
include("parallel_sampling_tests.jl")
include("hmc_initialization_rng_tests.jl")
include("phylogenetic_car_tests.jl")
include("clinical_adjustment_tests.jl")
include("sslasso_gamma_reference_tests.jl")
include("sslasso_initialization_tests.jl")
include("simulation_initialization_resume_tests.jl")
include("t1d_partition_tests.jl")
include("realdata_alpha_scale_tests.jl")
include("tuning_transition_diagnostics_tests.jl")
