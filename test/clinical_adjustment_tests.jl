using BayesCox4Micro: _validation_cindices
using Test, Random, LinearAlgebra, BayesCox4Micro
const ClinicalBC = BayesCox4Micro

@testset "Joint clinical likelihood and gradients" begin
    rng = MersenneTwister(42)
    X = rand(rng, 18, 5); X ./= sum(X; dims = 2)
    time = Float64.(repeat(1:9, inner = 2)); event = repeat([1, 0, 1], 6)
    C = Float64.(rand(rng, 0:1, 18, 2))
    data = prepare_cox_data(X, time, event)
    for nc in (false, true), constraint in (:prox, :soft_center, :none)
        base = ClinicalBC._make_explicit_nuts_target(data, RegularizedHorseshoePrior();
            constraint, prox_sum0_lambda = 0.01, noncentered = nc)
        t = ClinicalBC.CoxClinicalTarget(base, C)
        u = initial_state(t, rng); u[1:5] .+= 0.02randn(rng, 5); u[end-1:end] .= [0.3, -0.2]
        analytic = gradient!(zeros(length(u)), t, u)
        numeric = zeros(length(u))
        for j in eachindex(u)
            a, b = copy(u), copy(u); a[j] += 1e-6; b[j] -= 1e-6
            numeric[j] = (logdensity(t, a) - logdensity(t, b))/2e-6
        end
        @test analytic ≈ numeric atol=2e-5 rtol=1e-5
        u[end-1:end] .= 0
        @test logdensity(t, u) - logdensity(base, u[1:end-2]) ≈
            2*(ClinicalBC.logpdf(ClinicalBC.TDist(3), 0)-log(2.5))
    end
    base = CoxProxSSLassoTarget(data, SpikeSlabLassoPrior(), trues(5), Ref(0.5);
        constraint = :prox, prox_sum0_lambda = 0.01, prox_l1_lambda = 0.001,
        alpha_prior_shape = 1.0, alpha_prior_scale = 5.0, prox_root_tol = 1e-10, prox_root_maxiter = 100)
    t = ClinicalBC.CoxClinicalTarget(base, C)
    u = initial_state(t, rng); u[1:5] .= [0.2,-0.1,0.05,0.04,-0.1]; u[end-1:end] .= [.4,-.2]
    numeric = zeros(length(u))
    for j in eachindex(u)
        a,b=copy(u),copy(u); a[j]+=1e-6; b[j]-=1e-6
        numeric[j]=(logdensity(t,a)-logdensity(t,b))/2e-6
    end
    @test gradient!(zeros(length(u)), t, u) ≈ numeric atol=2e-5 rtol=1e-5
    @test_throws DimensionMismatch ClinicalBC.CoxClinicalTarget(base, C[1:end-1, :])
    for prior in (RegularizedHorseshoePrior(), SpikeSlabLassoPrior())
        opts = (prior = prior, n_samples = 16, warmup = 10, chains = 2, seed = 90,
            hmc_n_leapfrog = 3, clinical_names = ["clinical[Sex_M]", "clinical[Antibiotics_Tylosin]"])
        serial = sample_clinical_cox(X, time, event, C; opts..., multichains = false)
        parallel = sample_clinical_cox(X, time, event, C; opts..., multichains = true)
        @test size(posterior_beta_draws(serial)) == (32, 5)
        @test size(posterior_clinical_draws(serial)) == (32, 2)
        @test all(isfinite, posterior_clinical_draws(serial))
        @test posterior_clinical_draws(serial) == posterior_clinical_draws(parallel)
        @test posterior_beta_draws(serial) == posterior_beta_draws(parallel)
    end
end

if !isdefined(Main, :fit_and_evaluate)
    include(joinpath(@__DIR__, "..", "scripts", "analysis_common.jl"))
end

@testset "Clinical nested CV slicing and prediction" begin
    X = hcat(collect(1.0:24.0), ones(24))
    C = hcat(100 .+ X[:, 1], 200 .+ X[:, 1])
    fitmock = function(x, t, e; clinical_X, kwargs...)
        @test clinical_X[:, 1] == 100 .+ x[:, 1]
        @test clinical_X[:, 2] == 200 .+ x[:, 1]
        return (ids = x[:, 1],)
    end
    scoremock = function(fit, xt, xv, tv, ev; clinical_X_val)
        @test isempty(intersect(fit.ids, xv[:, 1]))
        @test clinical_X_val[:, 1] == 100 .+ xv[:, 1]
        return Dict("mean" => .7, "median" => .7, "MAP" => .7)
    end
    mktempdir() do dir
        r = evaluate_nested_cindex(X, collect(1.:24.), ones(Int, 24);
            clinical_X = C, priors = ["cox-rhs"], base_sampler_kwargs = Dict(:sampler => :hmc),
            base_prior_kwargs = Dict(), base_car_kwargs = Dict(), base_model_kwargs = Dict(),
            dataset_id = "clinical_test", outdir = dir, outer_folds = 3, inner_folds = 2,
            prior_grids = Dict("cox-rhs" => [Dict(:rhs_local_scale => 1.)]),
            fit_function = fitmock, score_function = scoremock)
        @test r.summary.status[1] == "success"
        @test r.summary.CV_Cindex_mean[1] ≈ .7
    end
    rng = MersenneTwister(71); X = rand(rng, 24, 5); C = Float64.(rand(rng, 0:1, 24, 2))
    for method in ("cox-rhs", "prox-sslasso")
        fit = fit_and_evaluate(X[1:18, :], collect(1.:18.), ones(Int, 18);
            clinical_X = C[1:18, :], prior_name = method, sampler = :hmc,
            n_samples = 12, warmup = 8, chains = 1, p0 = 2, hmc_n_leapfrog = 3,
            compute_metrics = false, seed = 25)
        scores = _validation_cindices(fit, X[1:18, :], X[19:end, :], collect(19.:24.), ones(Int, 6);
            clinical_X_val = C[19:end, :])
        @test all(v -> 0 <= v <= 1, values(scores))
        @test_throws ArgumentError _validation_cindices(fit, X[1:18, :], X[19:end, :],
            collect(19.:24.), ones(Int, 6))
    end
end
