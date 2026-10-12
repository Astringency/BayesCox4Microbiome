using Test, Random, LinearAlgebra, Statistics, Distributions
using BayesCox4Micro

function exact_fd_gradient(f, x)
    g = zeros(length(x))
    for j in eachindex(x)
        h = 1e-5 * max(1, abs(x[j]))
        hi, lo = copy(x), copy(x)
        hi[j] += h
        lo[j] -= h
        g[j] = (f(hi) - f(lo)) / (2h)
    end
    return g
end

@testset "Exact zero-sum geometry and posterior gradients" begin
    B = BayesCox4Micro
    rng = MersenneTwister(491)
    X = exp.(randn(rng, 35, 5) .* [0.3 0.6 1.0 1.7 2.0])
    times = collect(1.0:35.0)
    events = ones(Int, 35)
    for standardize in (false, true)
        data = prepare_cox_data(X, times, events; standardize = standardize)
        a = 1 ./ data.col_scale
        H = B._zero_sum_basis(a)
        @test H' * H ≈ I atol=1e-14
        @test norm(a' * H) < 1e-14
        for prior in (HorseshoePrior(), RegularizedHorseshoePrior(p0 = 2.0)), noncentered in (false, true)
            target = B._make_explicit_nuts_target(data, prior; constraint = :exact,
                prox_sum0_lambda = 1e-3, noncentered = noncentered)
            u = initial_state(target, rng) .+ 0.15 .* randn(rng, dimension(target))
            full = B._exact_full_state(target, u)
            saved = B.sample_state(target, u)
            @test abs(dot(a, saved[1:5])) < 1e-12
            @test B._exact_internal_state(target, full) ≈ u atol=1e-12
            lp, grad = logdensity_and_gradient!(zeros(length(u)), target, u)
            @test lp ≈ logdensity(target, u)
            @test grad ≈ exact_fd_gradient(x -> logdensity(target, x), u) rtol=2e-5 atol=2e-5
        end
        prior = SpikeSlabLassoPrior(proximal = true)
        base = CoxProxSSLassoTarget(data, prior, [true, false, true, false, true], Ref(0.3);
            constraint = :none, prox_sum0_lambda = 1e-3, prox_l1_lambda = 0.1,
            alpha_prior_shape = 2.0, alpha_prior_scale = 1.0,
            prox_root_tol = 1e-12, prox_root_maxiter = 200)
        target = B.CoxExactZeroTarget(base)
        u = initial_state(target, rng) .+ 0.2 .* randn(rng, dimension(target))
        @test abs(dot(a, B.sample_state(target, u)[1:5])) < 1e-12
        lp, grad = logdensity_and_gradient!(zeros(length(u)), target, u)
        @test lp ≈ logdensity(target, u)
        @test grad ≈ exact_fd_gradient(x -> logdensity(target, x), u) rtol=1e-4 atol=1e-4
    end
end

@testset "Exact conditional Gaussian normalization and covariance" begin
    B = BayesCox4Micro
    rng = MersenneTwister(702)
    data = prepare_cox_data(exp.(randn(rng, 15, 4)), collect(1.0:15.0), ones(Int, 15); standardize = true)
    prior = RegularizedHorseshoePrior(p0 = 2.0)
    centered = B._make_explicit_nuts_target(data, prior; constraint = :exact,
        prox_sum0_lambda = 1e-3, noncentered = false)
    nc = B._make_explicit_nuts_target(data, prior; constraint = :exact,
        prox_sum0_lambda = 1e-3, noncentered = true)
    u = initial_state(nc, rng)
    u[5:8] .= [-0.9, 0.3, -0.2, 0.8] # unequal local scales
    u[1:3] .= [0.2, -0.3, 0.1]
    saved = B.sample_state(nc, u)
    c = B._exact_internal_state(centered, saved)
    variances = exp.(2 .* B._exact_logscales(nc.base, saved))
    D = Diagonal(variances)
    a, H = centered.a, centered.H
    V = inv(Symmetric(H' * inv(D) * H))
    @test logdensity(centered, c) ≈ logdensity(nc, u) - 0.5logdet(V) atol=1e-10
    @test B.sample_logdensity(nc, saved, u) ≈ logdensity(centered, c) atol=1e-10
    expected = Matrix(D) - (D*a)*(D*a)' / dot(a, D*a)
    @test H * V * H' ≈ expected atol=1e-12
    draws = zeros(12000, 4)
    for i in axes(draws, 1)
        u[1:3] .= randn(rng, 3)
        draws[i, :] .= B.sample_state(nc, u)[1:4]
    end
    @test cov(draws) ≈ expected rtol=0.06
    @test maximum(abs, draws*a) < 1e-12
end

@testset "Exact HMC/NUTS sampling and original coefficient scale" begin
    rng = MersenneTwister(113)
    X = exp.(randn(rng, 24, 4) .* [0.3 0.8 1.1 2.2])
    times, events = collect(1.0:24.0), ones(Int, 24)
    for sampler in (:hmc, :nuts_explicit), method in (:rhs_centered, :rhs_noncentered, :ssl)
        prior = method == :ssl ? SpikeSlabLassoPrior(proximal = true) : RegularizedHorseshoePrior(p0 = 2.0)
        extra = sampler == :hmc ? (; hmc_n_leapfrog = 3) : (; max_depth = 3)
        result = sample_bayes_cox(X, times, events; prior = prior, sampler = sampler,
            constraint = :exact, standardize = true, noncentered = method == :rhs_noncentered,
            n_samples = 8, warmup = 8, chains = 1, seed = 117, progress = false, extra...)
        draws = posterior_beta_draws(result)
        @test size(draws) == (8, 4)
        @test all(isfinite, draws)
        @test maximum(abs, draws * (1 ./ result.data.col_scale)) < 1e-11
        @test all(isfinite, posterior_logposterior_draws(result))
    end
    @test_throws ArgumentError sample_bayes_cox(X, times, events;
        sampler = :mala, constraint = :exact, n_samples = 2, warmup = 0)
end
