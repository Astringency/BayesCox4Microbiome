using Test
using BayesCox4Micro
using Statistics
using DataFrames
using Logging

if !isdefined(@__MODULE__, :evaluate_cox_loo)
    include(joinpath(@__DIR__, "..", "scripts", "analysis_common.jl"))
end

@testset "subject deletion includes censored risk-set effects" begin
    X = reshape([0.0, 1.0, 2.0], :, 1)
    t, e = [1.0, 2.0, 3.0], [1, 1, 0]
    train = prepare_cox_data(X[1:2, :], t[1:2], e[1:2]; transform = :none)
    augmented = _loo_augment(train, X[1:2, :], X[3:3, :], t[3], e[3])
    draws = reshape([-0.5, 0.0, 0.5, 1.0], :, 1)
    ratios = subject_deletion_logratios(train, augmented, draws)
    for (s, b) in enumerate(vec(draws))
        w = exp.(vec(X) .* b)
        expected = log((w[1] + w[2]) / sum(w)) + log(w[2] / (w[2] + w[3]))
        @test ratios[s] ≈ expected
        @test cox_loglik_events(augmented, [b])[3] == 0.0
        @test ratios[s] < 0.0
    end
    @test logmeanexp(ratios) > mean(ratios)
    @test augmented.risk !== train.risk
    @test length(train.risk.order) == 2
    @test length(augmented.risk.order) == 3
end

@testset "Efron deletion score is invariant to tied-subject ordering" begin
    X = reshape([0.0, 1.0, 2.0, 3.0], :, 1)
    t, e = [1.0, 2.0, 2.0, 3.0], [1, 1, 1, 0]
    draws = reshape([-1.0, 0.0, 0.5, 1.5], :, 1)
    function score(order)
        train = prepare_cox_data(X[order, :], t[order], e[order]; transform = :none)
        aug = _loo_augment(train, X[order, :], X[4:4, :], t[4], e[4])
        subject_deletion_logratios(train, aug, draws)
    end
    @test score([1, 2, 3]) ≈ score([1, 3, 2])
    # Deleting a tied event must change both d_t and the tied risk sum.
    train = prepare_cox_data(X[[1, 3, 4], :], t[[1, 3, 4]], e[[1, 3, 4]]; transform = :none)
    aug = _loo_augment(train, X[[1, 3, 4], :], X[2:2, :], t[2], e[2])
    for b in (0.0, 0.7)
        w = exp.(vec(X) .* b)
        full = log(w[1] / sum(w)) + log(w[2]) + log(w[3]) -
               log(sum(w[2:4])) - log(sum(w[2:4]) - (w[2] + w[3]) / 2)
        reduced = log(w[1] / sum(w[[1, 3, 4]])) + log(w[3] / sum(w[3:4]))
        @test only(subject_deletion_logratios(train, aug, reshape([b], 1, 1))) ≈ full - reduced
    end
end

@testset "held-out preprocessing uses training zero replacement and scaling" begin
    X = [0.0 0.2; 0.4 0.6; 0.8 0.2]
    t, e = [1.0, 2.0, 3.0], [1, 1, 0]
    train = prepare_cox_data(X, t, e; standardize = true)
    aug = _loo_augment(train, X, [0.0 0.001], 4.0, 0)
    expected = (log.([0.1, 0.001] ./ 0.101) .- train.col_mean) ./ train.col_scale
    @test aug.z[1:3, :] == train.z
    @test aug.z[4, :] ≈ expected
    @test aug.col_mean == train.col_mean
    @test aug.col_scale == train.col_scale
end

@testset "LOO modes, training-only refits and resumable folds" begin
    X = [0.1 0.9; 0.2 0.8; 0.3 0.7; 0.4 0.6]
    t, e = [1.0, 2.0, 3.0, 4.0], [1, 1, 0, 0]
    calls = Int[]
    function fake_fit(x, time, event; seed = 1, compute_metrics = true, kwargs...)
        @test !compute_metrics
        @test size(x, 1) == 3
        omitted = only(setdiff(t, time))
        @test x == X[findall(!=(omitted), t), :]
        @test seed == 10 + Int(omitted)
        push!(calls, Int(omitted))
        data = prepare_cox_data(x, time, event; transform = :none)
        draws = hcat(sin.(1:40) .* 0.1, cos.(1:40) .* 0.1)
        samples = MHSamples(draws, zeros(40), ["beta[1]", "beta[2]"], 1.0, 1.0, zeros(2))
        result = CoxSamplerResult(samples, data, BayesianLassoPrior(), :mh)
        return (result = result, p0 = missing)
    end
    kwargs = (seed = 10,)
    mktempdir() do dir
        pilot = evaluate_cox_loo(X, t, e; mode = :subject_refit, fit_kwargs = kwargs,
            fit_function = fake_fit, diagnostics = false, folds = [1, 3], checkpoint_dir = dir)
        @test calls == [1, 3]
        @test !only(pilot.summary.complete)
        @test ismissing(only(pilot.summary.LOOIC))
        completed = evaluate_cox_loo(X, t, e; mode = :subject_refit, fit_kwargs = kwargs,
            fit_function = fake_fit, diagnostics = false, checkpoint_dir = dir)
        @test calls == [1, 3, 2, 4]
        @test only(completed.summary.complete)
        @test nrow(completed.pointwise) == 4
        @test all(==(3), completed.pointwise.n_train)
        @test all(<(0), completed.pointwise.log_score)
        @test only(completed.summary.LOOIC) ≈ 2 * only(completed.summary.NLPML)
        @test only(completed.summary.LOO_LOG_SCORE) ≈ sum(completed.pointwise.log_score)
        @test ismissing(only(completed.summary.WAIC_pw2))
        @test ismissing(only(completed.summary.ELPD_LOO))
        @test_throws ArgumentError evaluate_cox_loo(X, t, e; mode = :subject_refit,
            fit_kwargs = (seed = 11,), diagnostics = false, checkpoint_dir = dir)
        @test_throws ArgumentError evaluate_cox_loo(X, t, e; mode = :subject_refit,
            fit_kwargs = kwargs, diagnostics = false, checkpoint_dir = dir, resume = false)
    end
    @test_throws ArgumentError evaluate_cox_loo(X, t, [1, 0, 0, 0]; mode = :subject_refit)
    @test_throws ArgumentError evaluate_cox_loo(X, t, e; mode = :unknown)
    @test_throws ArgumentError evaluate_cox_loo(X, t, e; mode = :subject_refit, folds = [1, 1])
    @test_throws ArgumentError evaluate_cox_loo(X, t, e; mode = :subject_refit, fit_kwargs = (approx = true,))
    failed = with_logger(NullLogger()) do
        evaluate_cox_loo(X, t, e; mode = :subject_refit, folds = [1], diagnostics = false,
            fit_function = (args...; kwargs...) -> error("test failed refit"))
    end
    @test only(failed.pointwise.status) == "failed"
    @test !only(failed.summary.complete)
    @test ismissing(only(failed.summary.LOOIC))

    data = prepare_cox_data(X, t, e; transform = :none)
    draws = hcat(sin.(1:40) .* 0.1, cos.(1:40) .* 0.1)
    old = fit_metrics(data, draws, vec(mean(draws; dims = 1)))
    # The default must reuse full-data metrics without any subject refit.
    fixed = evaluate_cox_loo(X, t, e;
        full_fit = (metrics = DataFrame([old]),), fit_function = (args...; kwargs...) -> error("unexpected refit"))
    @test only(fixed.summary.mode) == "fixed_risk"
    @test only(fixed.summary.target) == "event_attribution_fixed_risk"
    @test isempty(fixed.pointwise)
    @test only(fixed.summary.LOOIC) == old.LOOIC
    @test only(fixed.summary.NLPML) == old.LPML
    @test only(fixed.summary.WAIC_pw2) == old.WAIC_pw2
    # Stabilized harmonic mean must not overflow when ell << -709.
    extreme = fit_metrics(data, draws, [0.0, 0.0]; ll_event = fill(-1000.0, 2, 40), loo = old)
    @test extreme.LPML ≈ 2000.0
end
