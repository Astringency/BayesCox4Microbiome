if !isdefined(@__MODULE__, :generate_simulation_data)
    include(joinpath(@__DIR__, "..", "scripts", "lib", "analysis_common.jl"))
end

@testset "simulation sanity checks" begin
    @testset "true beta layout nz=$nonzero" for nonzero in (6, 12)
        p = 50
        beta, idx = true_beta(p, nonzero)
        expected_values = nonzero == 6 ? DEFAULT_BETA6 : DEFAULT_BETA12
        @test idx == collect(18:2:(16 + 2nonzero))
        @test beta[idx] == expected_values
        @test count(!iszero, beta) == nonzero
        @test sum(beta) ≈ 0.0 atol = 1e-12
    end

    @testset "paper covariance only correlates true nonzeros" begin
        p = 50
        _, idx = true_beta(p, 6)
        sigma = Matrix(paper_covariance(p, idx))
        @test diag(sigma) == ones(p)
        @test sigma[18, 20] ≈ 0.75 - 0.015 * abs(18 - 20)
        @test sigma[18, 28] ≈ 0.75 - 0.015 * abs(18 - 28)
        @test sigma[1, 2] == 0.0
        @test sigma[18, 19] == 0.0
        @test sigma[17, 20] == 0.0
        @test isposdef(Symmetric(sigma))
    end

    @testset "generated event process $baseline" for baseline in (:weibull, :exponential, :gompertz)
        sim = generate_simulation_data(200, 50, 6; baseline = baseline, seed = 11)
        @test all(isapprox.(vec(sum(sim.X; dims = 2)), 1.0; atol = 1e-12, rtol = 1e-12))
        @test minimum(sim.X) > 0.0
        @test sim.eta ≈ log.(sim.X) * sim.beta atol = 1e-10 rtol = 1e-10

        event_rate = mean(sim.event .!= 0)
        censor_rate = mean(sim.event .== 0)
        @test 0.2 <= event_rate <= 0.9
        @test 0.1 <= censor_rate <= 0.8
        @test event_rate + censor_rate ≈ 1.0

        data = prepare_cox_data(sim.X, sim.time, sim.event; transform = :log)
        ll_true = cox_logpartiallikelihood(data, sim.beta)
        ll_zero = cox_logpartiallikelihood(data, zeros(length(sim.beta)))
        @test ll_true > ll_zero
    end
end

@testset "preprocessing consistency for simulated X" begin
    sim = generate_simulation_data(200, 150, 12; baseline = :weibull, seed = 7)
    data0 = prepare_cox_data(sim.X, sim.time, sim.event; transform = :log)
    data_default = prepare_cox_data(sim.X, sim.time, sim.event)

    @test data0.zero_replacement == 0.0
    @test data0.z ≈ log.(sim.X) atol = 1e-12 rtol = 1e-12
    @test data_default.z == data0.z
    @test data_default.z * sim.beta ≈ sim.eta atol = 1e-10 rtol = 1e-10
    @test_throws MethodError prepare_cox_data(sim.X, sim.time, sim.event; pseudocount = 1e-6)
end

@testset "posterior extraction and simulation summaries" begin
    X = [0.2 0.8; 0.4 0.6; 0.5 0.5]
    time = [1.0, 2.0, 3.0]
    event = [1, 0, 1]
    data = prepare_cox_data(X, time, event; transform = :none)

    draws = [
        20.0 99.0 10.0
        21.0 98.0 11.0
    ]
    samples = MHSamples(draws, [-1.0, -0.5], ["beta[2]", "theta", "beta[1]"], 0.5, 0.5, zeros(3))
    result = CoxSamplerResult(samples, data, BayesianLassoPrior(), :mh)
    @test posterior_beta_draws(result) == [10.0 20.0; 11.0 21.0]

    chain2 = MHSamples(draws .+ 1.0, [-2.0, -1.5], ["beta[2]", "theta", "beta[1]"], 0.5, 0.5, zeros(3))
    multi = CoxSamplerResult(MultiMHSamples([samples, chain2]), data, BayesianLassoPrior(), :mh)
    @test posterior_beta_draws(multi) == [10.0 20.0; 11.0 21.0; 11.0 21.0; 12.0 22.0]

    beta = [1.0, -2.0, 3.0]
    raw = DataFrame(
        variable = ["beta[3]", "beta[1]", "beta[2]"],
        mean = [3.5, 1.5, -1.5],
        median = [3.25, 1.25, -1.25],
        map = [3.0, 1.0, -1.0],
        lower = [2.0, 0.0, -3.0],
        upper = [4.0, 2.0, 0.0],
    )
    aligned = ensure_simulation_summary(raw, beta)
    @test aligned.variable == ["beta[1]", "beta[2]", "beta[3]"]
    @test aligned.true_beta == beta
    @test aligned.mean == [1.5, -1.5, 3.5]

    metrics = DataFrame(estimate = ["mean", "median", "MAP"])
    sim = (beta = beta,)
    annotated = annotate_simulation_metrics(
        metrics,
        sim,
        aligned;
        rep = 1,
        n = 3,
        p = 3,
        nonzero = 3,
        p0 = 3,
        baseline = :weibull,
        prior = "test",
    )
    @test annotated.MSE[annotated.estimate .== "mean"][1] ≈ mean((aligned.mean .- beta) .^ 2)
    @test annotated.MSE[annotated.estimate .== "mean"][1] != sum((aligned.mean .- beta) .^ 2)
end

@testset "global tau Cauchy scale" begin
    @test BayesCox4Micro._global_tau_cauchy_scale(0.2) ≈ 0.2
    @test_throws ArgumentError BayesCox4Micro._global_tau_cauchy_scale(0.0)
end
