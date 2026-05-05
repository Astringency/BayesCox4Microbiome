using Test
using Random
using BayesCox4Micro
using CSV
using DataFrames
using ForwardDiff
using JLD2
using MCMCChains

const B = BayesCox4Micro

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
        ("sslasso", cox_paper_prior("sslasso"; p0 = 1.0)),
    ]
    for (name, prior) in cases
        target = B._make_explicit_nuts_target(
            data,
            prior;
            constraint = :soft_center,
            soft_center_scale = 0.5,
            prox_sum0_lambda = 1e-3,
            prox_l1_lambda = 1e-3,
            alpha_prior_shape = 1.0,
            alpha_prior_rate = 0.1,
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
        soft_center_scale = 0.5,
        prox_sum0_lambda = 1e-3,
        prox_l1_lambda = 1e-3,
        alpha_prior_shape = 1.0,
        alpha_prior_rate = 0.1,
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
        prior = BayesianLassoPrior(),
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

    hs_explicit = sample_bayes_cox(
        X,
        time,
        event;
        prior = HorseshoePrior(p0 = 1.0),
        sampler = :nuts_explict,
        constraint = :soft_center,
        n_samples = 1,
        warmup = 1,
        chains = 1,
        max_depth = 2,
        progress = false,
        seed = 9,
    )
    @test size(posterior_beta_draws(hs_explicit)) == (1, 3)

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
        prior = SpikeSlabLassoPrior(),
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
        @test isfile(joinpath(dir, "samples_long.csv"))
        @test isfile(joinpath(dir, "metadata.json"))
        @test isfile(joinpath(dir, "diagnostics.json"))
        @test isfile(joinpath(dir, "result_bundle.jld2"))

        bundle = JLD2.load(joinpath(dir, "result_bundle.jld2"), "result_bundle")
        @test haskey(bundle, "samples")
        @test haskey(bundle, "samples_chain")
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
            save_long_csv = false,
            save_jld2 = false,
            sample_thin = 2,
        )
        @test isfile(joinpath(dir, "samples_wide.csv"))
        @test !isfile(joinpath(dir, "samples_long.csv"))
        @test !isfile(joinpath(dir, "result_bundle.jld2"))
        @test nrow(CSV.read(joinpath(dir, "samples_wide.csv"), DataFrame)) == 3

        diag = diagnose_run(dir; formats = "png", top_k_beta = 2, max_lag = 1)
        @test isfile(joinpath(diag.output_dir, "diagnostics_summary.csv"))
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
