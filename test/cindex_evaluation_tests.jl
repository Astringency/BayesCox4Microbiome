using Test
using BayesCox4Micro
using DataFrames
using Statistics
using Random

if !isdefined(@__MODULE__, :generate_simulation_test_data)
    include(joinpath(@__DIR__, "..", "scripts", "analysis_common.jl"))
end
if !isdefined(@__MODULE__, :evaluate_nested_cindex)
    include(joinpath(@__DIR__, "..", "scripts", "tune_hyperparams_cindex.jl"))
end

@testset "independent simulation test cohorts" begin
    for baseline in (:weibull, :exponential, :gompertz)
        train = generate_simulation_data(30, 8, 2; seed = 17, baseline, baseline_rate = 0.2)
        test = generate_simulation_test_data(30, 8, 2; seed = 17, baseline, baseline_rate = 0.2)
        again = generate_simulation_test_data(30, 8, 2; seed = 17, baseline, baseline_rate = 0.2)
        @test size(test.X) == (1000, 8)
        @test test.X == again.X
        @test test.time == again.time
        @test test.beta == train.beta
        @test test.baseline_rate == train.baseline_rate
        @test test.censoring_upper == train.censoring_upper
        @test test.X[1:30, :] != train.X
        @test test.X != generate_simulation_test_data(30, 8, 2; seed = 18, baseline).X
    end
end

@testset "test scoring uses training preprocessing and saved coefficient scale" begin
    X = [0.0 0.2; 0.4 0.6; 0.8 0.2; 0.3 0.7]
    train = (X = X, time = [1., 2., 3., 4.], event = [1, 1, 0, 1])
    test = (X = [0.0 0.001; 0.2 0.8; 0.9 0.1], time = [1., 2., 3.],
        event = [1, 1, 0], test_seed = "fixture")
    beta = [1., -1.]
    summary = DataFrame(mean = beta, median = -beta, map = zeros(2))
    metrics = DataFrame(estimate = ["mean", "median", "MAP"], Cindex = [0.8, 0.2, 0.5])
    add_simulation_test_metrics!(metrics, summary, train, test)
    z = log.([0.1 0.001; 0.2 0.8; 0.9 0.1] ./ [0.101, 1., 1.])
    @test metrics.Cindex == metrics.Cindex_train == [0.8, 0.2, 0.5]
    @test metrics.Cindex_test[1] == concordance_index(test.time, test.event, z * beta)
    @test metrics.Cindex_test[2] == concordance_index(test.time, test.event, z * -beta)
    @test metrics.Cindex_test[3] == 0.5
    data = prepare_cox_data(X, train.time, train.event; standardize = true, pseudocount = 0.0)
    zstd = _validation_design_matrix(X, test.X, data; requested_pseudocount = 0.0)
    @test metrics.Cindex_test[1] == concordance_index(test.time, test.event, zstd * (beta .* data.col_scale))
end

@testset "resume backfills test scores from existing coefficient summaries" begin
    sim = generate_simulation_data(20, 4, 2; seed = 8)
    test = generate_simulation_test_data(20, 4, 2; seed = 8)
    summary = DataFrame(variable = ["beta[$j]" for j in 1:4],
        mean = sim.beta, median = sim.beta, map = sim.beta,
        lower = sim.beta .- 0.1, upper = sim.beta .+ 0.1)
    mktempdir() do dir
        CSV.write(joinpath(dir, "posterior_summary.csv"), summary)
        CSV.write(joinpath(dir, "metrics.csv"), DataFrame(
            estimate = ["mean", "median", "MAP"], Cindex = [0.8, 0.8, 0.8]))
        write(joinpath(dir, "complete.txt"), "complete")
        recovered = load_or_recover_simulation_run(dir, sim; rep = 1, n = 20, p = 4,
            nonzero = 2, baseline = :weibull, prior = "cox-rhs", p0 = 2, test_sim = test)
        saved = CSV.read(joinpath(dir, "metrics.csv"), DataFrame)
        @test saved.Cindex == [0.8, 0.8, 0.8]
        @test all(==(1000), saved.n_test)
        @test saved.Cindex_test == recovered.metrics.Cindex_test
        @test saved.Cindex_test[1] == concordance_index(test.time, test.event, log.(test.X) * sim.beta)
    end
end

@testset "nested selection excludes outer tests and resumes completed folds" begin
    X = hcat(collect(1.:24.), fill(1., 24))
    time, event = collect(1.:24.), ones(Int, 24)
    calls = NamedTuple[]
    fit_mock = function(x, t, e; kwargs...)
        @test kwargs[:compute_metrics] == false
        @test kwargs[:p0] === nothing
        @test vec(x[:, 1]) == t
        result = (ids = Int.(x[:, 1]), scale = kwargs[:rhs_local_scale])
        push!(calls, result)
        return result
    end
    score_mock = function(fit, xt, xv, tv, ev; requested_pseudocount)
        @test fit.ids == Int.(xt[:, 1])
        @test isempty(intersect(fit.ids, Int.(xv[:, 1])))
        @test tv == xv[:, 1]
        score = length(fit.ids) == 16 ? 0.4 : fit.scale == 0.5 ? 0.9 : 0.7
        return Dict("mean" => score, "median" => score - 0.05, "MAP" => score + 0.05)
    end
    grid = Dict("cox-rhs" => [Dict(:rhs_local_scale => 0.5), Dict(:rhs_local_scale => 1.)])
    defaults = (priors = ["cox-rhs"], base_sampler_kwargs = Dict(:sampler => :mh),
        base_prior_kwargs = Dict(), base_car_kwargs = Dict(),
        base_model_kwargs = Dict(:p0 => nothing, :constraint => :soft_center),
        dataset_id = "fixture", outer_folds = 3, inner_folds = 2,
        prior_grids = grid, fit_function = fit_mock, score_function = score_mock)
    mktempdir() do dir
        result = evaluate_nested_cindex(X, time, event; defaults..., outdir = dir)
        @test all(==("success"), result.folds.status)
        @test all(==(1), result.folds.selected_config_index)
        @test result.summary.CV_Cindex_mean[1] ≈ 0.4
        @test result.summary.CV_Cindex_median_mean[1] ≈ 0.35
        @test all(==(0.9), result.folds.inner_Cindex_mean)
        @test length(calls) == 3 * (2 * 2 + 1)
        manifest = JSON.parsefile(joinpath(dir, "nested_cv", "cox-rhs", "manifest.json"); use_mmap = false)
        @test manifest["mode"] == "nested_cv"
        tests = Int[]
        for fold in 1:3
            split = JSON.parsefile(joinpath(dir, "nested_cv", "cox-rhs", "outer_0$fold", "split_indices.json"); use_mmap = false)
            append!(tests, split["outer_test"])
            @test isempty(intersect(split["outer_train"], split["outer_test"]))
            @test sort(vcat(split["inner_validation"]...)) == sort(split["outer_train"])
            fold_calls = calls[(5fold - 4):5fold]
            @test all(c -> isempty(intersect(c.ids, split["outer_test"])), fold_calls)
            @test fold_calls[end].ids == split["outer_train"]
        end
        @test sort(tests) == collect(1:24)
        empty!(calls)
        resumed = evaluate_nested_cindex(X, time, event; defaults..., outdir = dir)
        @test isempty(calls)
        @test resumed.summary.CV_Cindex_mean == result.summary.CV_Cindex_mean
        @test_throws ArgumentError evaluate_nested_cindex(X, time, event;
            defaults..., outdir = dir, seed = 2)
    end
    bad_outer = function(fit, xt, xv, tv, ev; requested_pseudocount)
        score = length(fit.ids) == 16 ? NaN : 0.7
        Dict("mean" => score, "median" => score, "MAP" => score)
    end
    mktempdir() do dir
        result = evaluate_nested_cindex(X, time, event;
            merge(defaults, (score_function = bad_outer,))..., outdir = dir)
        @test result.summary.status[1] == "incomplete"
        @test ismissing(result.summary.CV_Cindex_mean[1])
        @test result.summary.n_successful_folds[1] == 0
    end
    bad_candidate = function(fit, xt, xv, tv, ev; requested_pseudocount)
        score = fit.scale == 0.5 ? NaN : 0.6
        Dict("mean" => score, "median" => score, "MAP" => score)
    end
    mktempdir() do dir
        result = evaluate_nested_cindex(X, time, event;
            merge(defaults, (score_function = bad_candidate,))..., outdir = dir)
        @test all(==(2), result.folds.selected_config_index)
        @test result.summary.CV_Cindex_mean[1] ≈ 0.6
    end
end

@testset "nested C-index with real posterior fitting" begin
    rng = MersenneTwister(9)
    X = rand(rng, 18, 3) .+ 0.1
    time = collect(1.:18.)
    event = ones(Int, 18)
    mktempdir() do dir
        result = evaluate_nested_cindex(X, time, event;
            priors = ["cox-rhs"], base_sampler_kwargs = Dict(:sampler => :mh,
                :n_samples => 8, :warmup => 4, :chains => 1, :progress => false),
            base_prior_kwargs = Dict(), base_car_kwargs = Dict(:use_car => false),
            base_model_kwargs = Dict(:p0 => 1, :standardize => true),
            outdir = dir, dataset_id = "posterior_smoke", outer_folds = 3, inner_folds = 2,
            prior_grids = Dict("cox-rhs" => [Dict(:rhs_local_scale => 1.)]))
        @test result.summary.status[1] == "success"
        @test 0 <= result.summary.CV_Cindex_mean[1] <= 1
        for fold in 1:3
            saved = joinpath(dir, "nested_cv", "cox-rhs", "outer_0$fold", "fit")
            meta = JSON.parsefile(joinpath(saved, "metadata.json"); use_mmap = false)
            @test meta["n"] == 12
            @test meta["evaluation"] == "nested_cv_outer_training"
            @test isfile(joinpath(saved, "samples_wide.csv"))
            @test isfile(joinpath(saved, "rank_diagnostics.csv"))
        end
    end
end

module CindexSummaryReaders
include(joinpath(@__DIR__, "..", "scripts", "summarize_realdata_convergence.jl"))
end

module CindexMethodReaders
include(joinpath(@__DIR__, "..", "scripts", "summarize_realdata_methods.jl"))
end

module CindexSimulationReaders
include(joinpath(@__DIR__, "..", "scripts", "summarize_simulation_results.jl"))
end

@testset "summaries reject historical tuning scores as nested tests" begin
    old = DataFrame(dataset_id = ["T1D"], constraint = ["prox"], prior = ["cox-rhs"],
        CV_Cindex_mean = [0.99])
    @test CindexSummaryReaders.fallback_cv_row(old, "T1D", "prox", "cox-rhs") === nothing
    old.evaluation = ["nested_cv"]
    old.status = ["incomplete"]
    @test CindexSummaryReaders.fallback_cv_row(old, "T1D", "prox", "cox-rhs") === nothing
    old.status .= "success"
    @test CindexSummaryReaders.fallback_cv_row(old, "T1D", "prox", "cox-rhs").CV_Cindex_mean == 0.99
    mktempdir() do dir
        table = DataFrame(dataset_id = ["T1D"], prior = ["cox-rhs"], cv_cindex = [0.99])
        CSV.write(joinpath(dir, "bayes_all_methods.csv"), table)
        @test ismissing(only(CindexMethodReaders.bayes_rows(dir)).cindex)
        table.cv_evaluation = ["nested_cv"]
        CSV.write(joinpath(dir, "bayes_all_methods.csv"), table)
        row = only(CindexMethodReaders.bayes_rows(dir))
        @test row.cindex == 0.99
        @test row.cindex_metric == "Nested-CV-Cindex"
        path = joinpath(dir, "nested_cv_summary.csv")
        _write_nested_table(path, DataFrame(prior = ["rhs"], CV_Cindex_mean = [0.6]), ["rhs"])
        _write_nested_table(path, DataFrame(prior = ["cox-rhs"], CV_Cindex_mean = [0.7]), ["cox-rhs"])
        _write_nested_table(path, DataFrame(prior = ["cox-rhs"], CV_Cindex_mean = [0.8]), ["cox-rhs"])
        saved = CSV.read(path, DataFrame)
        @test saved.prior == ["rhs", "cox-rhs"]
        @test saved.CV_Cindex_mean == [0.6, 0.8]
    end
    sim = DataFrame(run_dir = ["a", "b"], prior = ["cox-rhs", "cox-rhs"],
        Cindex = [0.9, 0.8], Cindex_train = [0.9, 0.8], Cindex_test = [0.6, 0.5])
    agg = CindexSimulationReaders.summarize_model_metrics(sim)
    @test agg.Cindex_mean[1] ≈ 0.85
    @test agg.Cindex_train_mean[1] ≈ 0.85
    @test agg.Cindex_test_mean[1] ≈ 0.55
end
