function _mala_prec_dataset(; tie_method::Symbol = :efron)
    rng = MersenneTwister(5501)
    n, p = 30, 5
    X = rand(rng, n, p) .+ 0.2
    time = Float64.(rand(rng, 1:8, n))
    event = Int.(rand(rng, n) .< 0.65)
    event[1] = 1
    data = prepare_cox_data(X, time, event; transform = :none, tie_method = tie_method)
    return data, X, time, event
end

function _mala_prec_prior_cases(X, time, event)
    return [
        ("hs", cox_paper_prior("hs"; p0 = 1.0)),
        ("rhs", cox_paper_prior("rhs"; p0 = 1.0)),
        ("new-rhs", cox_paper_prior("new-rhs"; p0 = 1.0)),
        ("est-rhs", cox_paper_prior("est-rhs"; p0 = 1.0, X = X, time = time, event = event, transform = :none)),
        ("cox-hs", cox_paper_prior("cox-hs"; p0 = 1.0)),
        ("cox-rhs", cox_paper_prior("cox-rhs"; p0 = 1.0)),
        ("lasso", cox_paper_prior("lasso")),
        ("sslasso", cox_paper_prior("sslasso")),
        ("prox-lasso", cox_paper_prior("prox-lasso")),
        ("prox-sslasso", cox_paper_prior("prox-sslasso")),
    ]
end

function _mala_prec_initial(prior, data; rng = MersenneTwister(99))
    p = size(data.z, 2)
    n_events = count(!=(0), data.event)
    if B._is_proximal_l1_prior(prior)
        base = [0.8, -0.6, 0.4, -0.1, 0.05]
        return vcat(base[1:p], [log(0.3)])
    elseif prior isa RegularizedHorseshoePrior
        u = initial_state(prior, p, n_events; n_obs = size(data.z, 1))
    else
        u = initial_state(prior, p, n_events)
    end
    u .+= 0.03 .* randn(rng, length(u))
    return u
end

function _fd_gradient(f, u; h = 1e-6)
    g = zeros(Float64, length(u))
    for j in eachindex(u)
        e = zeros(Float64, length(u))
        e[j] = h
        g[j] = (f(u .+ e) - f(u .- e)) / (2h)
    end
    return g
end

function _fd_hessian_from_gradient(gfun, u; h = 1e-5)
    d = length(u)
    H = zeros(Float64, d, d)
    for j in 1:d
        e = zeros(Float64, d)
        e[j] = h
        H[:, j] .= (gfun(u .+ e) .- gfun(u .- e)) ./ (2h)
    end
    return H
end

@testset "MALA gradient finite differences for paper priors" begin
    data, X, time, event = _mala_prec_dataset()
    rng = MersenneTwister(101)
    for (name, prior) in _mala_prec_prior_cases(X, time, event)
        u = _mala_prec_initial(prior, data; rng = rng)
        if prior isa BayesianLassoPrior && prior.proximal
            work = B.ProxGradientWorkspace(data)
            f = x -> B._lasso_prox_logposterior(
                x,
                data,
                prior;
                constraint = :prox,
                prox_sum0_lambda = 0.7,
                prox_l1_lambda = 0.9,
                alpha_prior_shape = 1.0,
                alpha_prior_rate = 0.4,
                prox_root_tol = 1e-12,
                prox_root_maxiter = 200,
            )
            grad = zeros(Float64, length(u))
            B._lasso_prox_gradient!(
                grad,
                u,
                data,
                work;
                constraint = :prox,
                prox_sum0_lambda = 0.7,
                prox_l1_lambda = 0.9,
                alpha_prior_shape = 1.0,
                alpha_prior_rate = 0.4,
                prox_root_tol = 1e-12,
                prox_root_maxiter = 200,
            )
        elseif prior isa SpikeSlabLassoPrior && prior.proximal
            gamma = [true, false, true, false, true]
            theta = 0.45
            work = B.ProxGradientWorkspace(data)
            f = x -> B._ssl_prox_logposterior(
                x,
                data,
                prior,
                gamma,
                theta;
                constraint = :prox,
                prox_sum0_lambda = 0.7,
                prox_l1_lambda = 0.9,
                alpha_prior_shape = 1.0,
                alpha_prior_rate = 0.4,
                prox_root_tol = 1e-12,
                prox_root_maxiter = 200,
            )
            grad = zeros(Float64, length(u))
            B._ssl_prox_gradient!(
                grad,
                u,
                data,
                prior,
                gamma,
                work;
                constraint = :prox,
                prox_sum0_lambda = 0.7,
                prox_l1_lambda = 0.9,
                alpha_prior_shape = 1.0,
                alpha_prior_rate = 0.4,
                prox_root_tol = 1e-12,
                prox_root_maxiter = 200,
            )
        else
            work = CoxGradientWorkspace(data)
            f = x -> logposterior(x, data, prior; constraint = :prox, prox_sum0_lambda = 0.7)
            grad = zeros(Float64, length(u))
            B.logposterior_and_gradient!(
                grad,
                u,
                data,
                prior,
                work;
                constraint = :prox,
                prox_sum0_lambda = 0.7,
            )
        end
        fd = _fd_gradient(f, u)
        @test maximum(abs.(grad .- fd)) < 2e-4
    end
end

@testset "Cox observed information finite differences" begin
    rng = MersenneTwister(102)
    beta = 0.1 .* randn(rng, 5)
    for tie in (:efron, :breslow)
        data, _, _, _ = _mala_prec_dataset(tie_method = tie)
        G = zeros(Float64, 5, 5)
        cox_observed_information_beta!(G, data, beta)
        H = _fd_hessian_from_gradient(b -> cox_score_gradient(data, b), beta)
        @test maximum(abs.(G .+ H)) < 1e-6
        @test isposdef(Symmetric(G + 1e-8I))
    end
end

@testset "MALA observed information Hessians" begin
    data, X, time, event = _mala_prec_dataset()
    rng = MersenneTwister(103)
    beta_cases = [
        cox_paper_prior("hs"; p0 = 1.0),
        cox_paper_prior("rhs"; p0 = 1.0),
        cox_paper_prior("cox-hs"; p0 = 1.0),
        cox_paper_prior("cox-rhs"; p0 = 1.0),
        cox_paper_prior("lasso"),
    ]
    p = size(data.z, 2)
    for prior in beta_cases
        u = _mala_prec_initial(prior, data; rng = rng)
        G = zeros(Float64, p, p)
        B.standard_observed_information_beta!(
            G,
            data,
            prior,
            u;
            constraint = :prox,
            prox_sum0_lambda = 0.7,
        )
        beta0 = collect(view(u, 1:p))
        gfun = b -> begin
            uu = copy(u)
            uu[1:p] .= b
            B._full_logposterior_gradient_beta(
                uu,
                data,
                prior;
                constraint = :prox,
                prox_sum0_lambda = 0.7,
            )
        end
        H = _fd_hessian_from_gradient(gfun, beta0)
        @test maximum(abs.(G .+ H)) < 1e-5
    end

    ssl_prior = cox_paper_prior("sslasso")
    u_ssl = _mala_prec_initial(ssl_prior, data; rng = rng)
    G_ssl = zeros(Float64, p, p)
    B.standard_observed_information_beta!(
        G_ssl,
        data,
        ssl_prior,
        u_ssl;
        constraint = :prox,
        prox_sum0_lambda = 0.7,
    )
    @test all(isfinite, G_ssl)
    @test G_ssl ≈ G_ssl' atol = 1e-12
    @test isposdef(Symmetric(G_ssl + 1e-6I))

    prox_prior = cox_paper_prior("prox-lasso")
    u_prox = [2.0, -1.8, 1.6, -0.2, 0.1, log(0.2)]
    G_prox = zeros(Float64, p + 1, p + 1)
    B.prox_lasso_observed_information!(
        G_prox,
        u_prox,
        data,
        prox_prior;
        constraint = :none,
        prox_sum0_lambda = 0.7,
        prox_l1_lambda = 0.9,
        alpha_prior_shape = 1.0,
        alpha_prior_rate = 0.4,
        prox_root_tol = 1e-12,
        prox_root_maxiter = 200,
    )
    prox_work = B.ProxGradientWorkspace(data)
    H_prox = _fd_hessian_from_gradient(u -> begin
        g = zeros(Float64, length(u))
        B._lasso_prox_gradient!(
            g,
            u,
            data,
            prox_work;
            constraint = :none,
            prox_sum0_lambda = 0.7,
            prox_l1_lambda = 0.9,
            alpha_prior_shape = 1.0,
            alpha_prior_rate = 0.4,
            prox_root_tol = 1e-12,
            prox_root_maxiter = 200,
        )
        g
    end, u_prox)
    @test maximum(abs.(G_prox .+ H_prox)) < 1e-5

    prox_ssl_prior = SpikeSlabLassoPrior(spike_scale = 0.5, slab_scale = 1.0, proximal = true)
    gamma = [true, false, true, false, true]
    u_wprox = [2.5, -2.0, 1.7, -0.1, 0.2, log(0.5)]
    G_wprox = zeros(Float64, p + 1, p + 1)
    B.prox_ssl_observed_information!(
        G_wprox,
        u_wprox,
        data,
        prox_ssl_prior,
        gamma;
        constraint = :none,
        prox_sum0_lambda = 0.7,
        prox_l1_lambda = 0.9,
        alpha_prior_shape = 1.0,
        alpha_prior_rate = 0.4,
        prox_root_tol = 1e-12,
        prox_root_maxiter = 200,
    )
    ssl_work = B.ProxGradientWorkspace(data)
    H_wprox = _fd_hessian_from_gradient(u -> begin
        g = zeros(Float64, length(u))
        B._ssl_prox_gradient!(
            g,
            u,
            data,
            prox_ssl_prior,
            gamma,
            ssl_work;
            constraint = :none,
            prox_sum0_lambda = 0.7,
            prox_l1_lambda = 0.9,
            alpha_prior_shape = 1.0,
            alpha_prior_rate = 0.4,
            prox_root_tol = 1e-12,
            prox_root_maxiter = 200,
        )
        g
    end, u_wprox)
    @test maximum(abs.(G_wprox .+ H_wprox)) < 1e-5
end

@testset "MALA observed information SPD checks" begin
    data, X, time, event = _mala_prec_dataset()
    p = size(data.z, 2)
    rng = MersenneTwister(104)
    for (name, prior) in _mala_prec_prior_cases(X, time, event)
        u = _mala_prec_initial(prior, data; rng = rng)
        if prior isa BayesianLassoPrior && prior.proximal
            G = zeros(Float64, p + 1, p + 1)
            B.prox_lasso_observed_information!(
                G,
                u,
                data,
                prior;
                constraint = :prox,
                prox_sum0_lambda = 0.7,
                prox_l1_lambda = 0.9,
                alpha_prior_shape = 1.0,
                alpha_prior_rate = 0.4,
                prox_root_tol = 1e-12,
                prox_root_maxiter = 200,
                ridge = 1e-6,
            )
        elseif prior isa SpikeSlabLassoPrior && prior.proximal
            G = zeros(Float64, p + 1, p + 1)
            B.prox_ssl_observed_information!(
                G,
                u,
                data,
                prior,
                [true, false, true, false, true];
                constraint = :prox,
                prox_sum0_lambda = 0.7,
                prox_l1_lambda = 0.9,
                alpha_prior_shape = 1.0,
                alpha_prior_rate = 0.4,
                prox_root_tol = 1e-12,
                prox_root_maxiter = 200,
                ridge = 1e-6,
            )
        else
            G = zeros(Float64, p, p)
            B.standard_observed_information_beta!(
                G,
                data,
                prior,
                u;
                constraint = :prox,
                prox_sum0_lambda = 0.7,
                ridge = 1e-6,
            )
        end
        @test all(isfinite, G)
        @test G ≈ G' atol = 1e-12
        @test isposdef(Symmetric(G))
    end

    target = B._make_explicit_nuts_target(
        data,
        cox_paper_prior("cox-rhs"; p0 = 1.0);
        constraint = :prox,
        prox_sum0_lambda = 0.7,
        prox_l1_lambda = 0.9,
        alpha_prior_shape = 1.0,
        alpha_prior_rate = 0.4,
        prox_root_tol = 1e-12,
        prox_root_maxiter = 200,
    )
    u_target = initial_state(target)
    G_target = zeros(Float64, p, p)
    observed_information!(G_target, target, u_target; ridge = 1e-6, mode = :auto)
    @test isposdef(Symmetric(G_target))

    prox_target = B._make_explicit_nuts_target(
        data,
        cox_paper_prior("prox-lasso");
        constraint = :prox,
        prox_sum0_lambda = 0.7,
        prox_l1_lambda = 0.9,
        alpha_prior_shape = 1.0,
        alpha_prior_rate = 0.4,
        prox_root_tol = 1e-12,
        prox_root_maxiter = 200,
    )
    u_prox_target = initial_state(prox_target)
    G_prox_target = zeros(Float64, p + 1, p + 1)
    observed_information!(G_prox_target, prox_target, u_prox_target; ridge = 1e-6, mode = :auto)
    @test all(isfinite, G_prox_target)
    @test G_prox_target ≈ G_prox_target' atol = 1e-12
end

@testset "Hessian-preconditioned MALA smoke tests" begin
    data, _, _, _ = _mala_prec_dataset()
    smoke_priors = [
        cox_paper_prior("hs"; p0 = 1.0),
        cox_paper_prior("cox-rhs"; p0 = 1.0),
        cox_paper_prior("lasso"),
        cox_paper_prior("sslasso"),
        cox_paper_prior("prox-lasso"),
        cox_paper_prior("prox-sslasso"),
    ]
    for schedule in (:adaptive_mcmc, :adaptive_sgld), prior in smoke_priors
        result = sample_bayes_cox(
            data;
            prior = prior,
            sampler = :mala,
            mala_precondition = true,
            mala_stepsize_schedule = schedule,
            mala_sgld_c0 = 1.0,
            constraint = :prox,
            prox_sum0_lambda = 0.7,
            prox_l1_lambda = 0.9,
            alpha_prior_shape = 1.0,
            alpha_prior_rate = 0.4,
            n_samples = 30,
            warmup = 20,
            thin = 1,
            mala_step_size = 1e-4,
            mala_min_step = 1e-8,
            mala_max_step = 1.0,
            progress = false,
            seed = 2400 + length(string(typeof(prior))) + (schedule == :adaptive_sgld ? 100 : 0),
        )
        samples = result.samples
        @test samples isa MHSamples
        @test size(posterior_beta_draws(result)) == (30, size(data.z, 2))
        @test all(isfinite, samples.draws)
        @test all(isfinite, samples.logposterior)
        @test isfinite(samples.acceptance_rate)
        @test all(isfinite, samples.final_step)
        @test all(samples.final_step .> 0)
    end
end

@testset "adaptive_sgld MALA step schedule" begin
    eps1 = B._mala_sgld_epsilon(2.0, 1, 30, 5, 1e-12, 1.0)
    eps10 = B._mala_sgld_epsilon(2.0, 10, 30, 5, 1e-12, 1.0)
    eps11 = B._mala_sgld_epsilon(2.0, 11, 30, 5, 1e-12, 1.0)
    @test eps1 > eps10 > eps11
    @test eps10 ≈ 2.0 * 10.0^(-0.55) / (30 * 5)
    @test B._adapt_mala_sgld_c0(1.0, 0.7, 0.4, 0.6, 1e-12, 1e12) ≈ 1.2
    @test B._adapt_mala_sgld_c0(1.0, 0.3, 0.4, 0.6, 1e-12, 1e12) ≈ 1 / 1.5
    @test B._adapt_mala_sgld_c0(1.0, 0.5, 0.4, 0.6, 1e-12, 1e12) ≈ 1.0

    data, _, _, _ = _mala_prec_dataset()
    c0 = 2.0
    result = sample_bayes_cox(
        data;
        prior = cox_paper_prior("hs"; p0 = 1.0),
        sampler = :mala,
        mala_precondition = true,
        mala_stepsize_schedule = :adaptive_sgld,
        mala_sgld_c0 = c0,
        mala_sgld_adapt_c0 = false,
        constraint = :none,
        n_samples = 3,
        warmup = 2,
        mala_step_size = 1e-4,
        progress = false,
        seed = 2501,
    )
    expected = B._mala_sgld_epsilon(c0, 5, size(data.z, 1), size(data.z, 2), 1e-8, 1.0)
    @test result.samples.final_step[1] ≈ expected

    post_only = sample_bayes_cox(
        data;
        prior = cox_paper_prior("hs"; p0 = 1.0),
        sampler = :mala,
        mala_precondition = true,
        mala_stepsize_schedule = :adaptive_sgld,
        mala_sgld_c0 = c0,
        mala_sgld_adapt_c0 = true,
        constraint = :none,
        n_samples = 3,
        warmup = 0,
        mala_step_size = 1e-4,
        progress = false,
        seed = 2502,
    )
    @test post_only.samples.final_step[1] ≈ B._mala_sgld_epsilon(c0, 3, size(data.z, 1), size(data.z, 2), 1e-8, 1.0)
end
