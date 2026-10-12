using Test
using CSV
using DataFrames
using Statistics
using BayesCox4Micro

if !isdefined(@__MODULE__, :annotate_simulation_metrics)
    include(joinpath(@__DIR__, "..", "scripts", "analysis_common.jl"))
end

module CoverageSummaryReaders
include(joinpath(@__DIR__, "..", "scripts", "summarize_simulation_results.jl"))
end

@testset "95% marginal credible interval coverage" begin
    beta = [0., 0., 2., -1.]
    lower = [-0.1, 0.1, 1., -1.]
    upper = [0.1, 0.3, 1.5, -0.5]
    coverage = simulation_coverage(beta, lower, upper)
    @test coverage == (coverage_95_all = 0.5, coverage_95_nonzero = 0.5, coverage_95_zero = 0.5)
    @test simulation_coverage([2., -1.], [2., -2.], [3., -1.]).coverage_95_all == 1.0
    @test ismissing(simulation_coverage(zeros(2), [-1., -1.], [1., 1.]).coverage_95_nonzero)
    @test ismissing(simulation_coverage([1., -1.], [0., -2.], [2., 0.]).coverage_95_zero)
    @test all(ismissing, simulation_coverage(Float64[], Float64[], Float64[]))
    @test_throws DimensionMismatch simulation_coverage(beta, lower[1:3], upper)

    # A broken interval must not silently shrink the denominator or count as uncovered.
    for bad_lower in (missing, NaN, Inf, 0.5)
        result = simulation_coverage(beta, [bad_lower, lower[2:end]...], upper)
        @test ismissing(result.coverage_95_all)
        @test ismissing(result.coverage_95_zero)
        @test result.coverage_95_nonzero == 0.5
    end
    @test all(ismissing, simulation_coverage([NaN, beta[2:end]...], lower, upper))

    summary = DataFrame(true_beta = beta, q025 = lower, q975 = upper)
    metrics = DataFrame(estimate = ["mean", "median", "MAP"])
    add_simulation_coverage!(metrics, summary)
    for col in SIMULATION_COVERAGE_COLUMNS
        @test metrics[!, col] == fill(0.5, 3)
    end
    add_simulation_coverage!(metrics, select(summary, Not(:true_beta)))
    @test all(ismissing, metrics.coverage_95_all)
    add_simulation_coverage!(metrics, select(summary, Not(:q025)))
    @test all(ismissing, metrics.coverage_95_all)
end

@testset "coverage uses unstandardized posterior intervals" begin
    X = [1. 10.; 2. 30.; 4. 60.; 8. 100.]
    data = prepare_cox_data(X, [1., 2., 3., 4.], [1, 1, 0, 1]; transform = :none, standardize = true)
    original_draws = [0.8 -1.2; 0.9 -1.1; 1.1 -0.9; 1.2 -0.8]
    scaled_draws = original_draws .* reshape(data.col_scale, 1, :)
    summary = posterior_summary(beta_draws_unstandardized(data, scaled_draws))
    sim = (beta = [1., -1.],)
    metrics = annotate_simulation_metrics(DataFrame(estimate = ["mean", "median", "MAP"]), sim, summary;
        rep = 1, n = 4, p = 2, nonzero = 2, p0 = 2, baseline = :weibull, prior = "cox-rhs")
    @test metrics.coverage_95_all == ones(3)
    @test metrics.coverage_95_nonzero == ones(3)
    @test all(ismissing, metrics.coverage_95_zero)
end

@testset "resume persists coverage without refitting or test data" begin
    beta = [0., 0., 2., -1.]
    summary = DataFrame(variable = ["beta[$j]" for j in 1:4], mean = beta, median = beta, map = beta,
        lower = [-0.1, 0.1, 1., -1.], upper = [0.1, 0.3, 1.5, -0.5])
    mktempdir() do dir
        CSV.write(joinpath(dir, "posterior_summary.csv"), summary[[4, 2, 1, 3], :])
        CSV.write(joinpath(dir, "metrics.csv"), DataFrame(estimate = ["mean", "median", "MAP"], Cindex = [0.8, 0.7, 0.6]))
        write(joinpath(dir, "complete.txt"), "complete")
        recovered = load_or_recover_simulation_run(dir, (beta = beta,); rep = 1, n = 20, p = 4,
            nonzero = 2, baseline = :weibull, prior = "cox-rhs", p0 = 2)
        saved = CSV.read(joinpath(dir, "metrics.csv"), DataFrame)
        @test saved.Cindex == [0.8, 0.7, 0.6]
        for col in SIMULATION_COVERAGE_COLUMNS
            @test saved[!, col] == recovered.metrics[!, col] == fill(0.5, 3)
        end
        @test !isfile(joinpath(dir, "samples_wide.csv"))
    end
end

@testset "historical summaries backfill coverage and aggregate repetitions" begin
    mktempdir() do root
        beta = [0., 0., 2., -1.]
        run_dirs = String[]
        for rep in 1:3
            dir = joinpath(root, "runs", "rep$(rep)_cox-rhs_weibull_n20_p4_nz2")
            mkpath(dir)
            push!(run_dirs, dir)
            # Deliberately omit coverage from historical metrics; saved truth need not be ordered.
            CSV.write(joinpath(dir, "metrics.csv"), DataFrame(estimate = ["mean", "median", "MAP"], Cindex = fill(0.7, 3)))
            lower = rep == 1 ? [-0.1, 0.1, 1., -1.] : [-0.1, -0.1, 2., -2.]
            upper = rep == 1 ? [0.1, 0.3, 1.5, -0.5] : [0.1, 0.1, 3., -1.]
            posterior = DataFrame(true_beta = beta, lower = lower, upper = upper)
            rep == 3 && (posterior.lower = [NaN, lower[2:end]...])
            CSV.write(joinpath(dir, "posterior_summary.csv"), posterior[[4, 2, 1, 3], :])
        end
        long = CoverageSummaryReaders.collect_model_metrics(root, run_dirs)
        @test nrow(long) == 9
        @test all(ismissing, long[long.rep .== 3, :coverage_95_all])
        agg = CoverageSummaryReaders.summarize_model_metrics(long)
        @test nrow(agg) == 3 # One row per estimate; repeated estimate rows do not inflate B.
        @test agg.coverage_95_all_mean == fill(0.75, 3)
        @test agg.coverage_95_all_sd ≈ fill(std([0.5, 1.0]), 3)
        @test agg.coverage_95_all_n == fill(2, 3)
        @test agg.coverage_95_zero_mean == fill(0.75, 3)
        @test agg.coverage_95_zero_n == fill(2, 3)
        @test agg.coverage_95_nonzero_mean ≈ fill(mean([0.5, 1., 1.]), 3)
        @test agg.coverage_95_nonzero_n == fill(3, 3)
        # The run_simulation summary uses the same aggregators as the standalone reader.
        direct = combine(groupby(long, :estimate), simulation_coverage_summary_specs(long)...)
        @test direct.coverage_95_all_mean == agg.coverage_95_all_mean

        outputs = CoverageSummaryReaders.summarize_root(root)
        for key in ("model_metrics_summary", "simulation_results_summary")
            saved = CSV.read(outputs[key], DataFrame)
            @test saved.coverage_95_all_mean == fill(0.75, 3)
            @test saved.coverage_95_all_n == fill(2, 3)
        end
        # Missing saved truth is not reconstructed from a default simulation layout.
        CSV.write(joinpath(run_dirs[1], "posterior_summary.csv"), DataFrame(lower = zeros(4), upper = ones(4)))
        missing_rows = CoverageSummaryReaders.collect_model_metrics(root, run_dirs[1:1])
        missing_agg = CoverageSummaryReaders.summarize_model_metrics(missing_rows)
        @test all(ismissing, missing_agg.coverage_95_all_mean)
        @test all(ismissing, missing_agg.coverage_95_all_sd)
        @test missing_agg.coverage_95_all_n == zeros(3)

        single = CoverageSummaryReaders.summarize_model_metrics(long[long.rep .== 2, :])
        @test single.coverage_95_all_mean == ones(3)
        @test all(ismissing, single.coverage_95_all_sd)
        @test single.coverage_95_all_n == ones(3)
    end
end
