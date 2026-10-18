using Test, BayesCox4Micro, Random, Statistics, DataFrames, CSV, MCMCChains

@testset "Bayesian criteria use explicit scales and one implementation" begin
    ll = [-1.0 -2.0 -3.0; -0.5 -1.0 -1.5]
    m = bayesian_fit_metrics(ll; point_loglik = -2.5, n_parameters = 2, n_observations = 10,
        deviance_at_mean = 4.0)
    lppd = sum(log.(vec(mean(exp.(ll); dims = 2))))
    @test m.AIC == 9.0
    @test m.BIC ≈ 5 + 2log(10)
    @test m.WAIC_pw2 ≈ -2lppd + 2sum(var(ll; dims = 2))
    @test m.WAIC_pw1 ≈ -2lppd + 4sum(log.(vec(mean(exp.(ll); dims = 2))) .- vec(mean(ll; dims = 2)))
    @test m.NLPML ≈ sum(log.(vec(mean(exp.(-ll); dims = 2))))
    @test m.DIC == 8.0
    @test !hasproperty(m, :LPML)
    @test_throws DimensionMismatch bayesian_fit_metrics(ll; point_loglik = -1., n_parameters = 2,
        n_observations = 10, chains = 2, draws_per_chain = 2)
    @test coefficient_mse([1., -1.], [2., -3.]) == 2.5
    @test_throws DimensionMismatch coefficient_mse([1.], [1., 2.])
end

@testset "Unified diagnostics and scatter thresholds" begin
    B = BayesCox4Micro
    rng = MersenneTwister(884)
    for n_chains in (1, 2, 4)
        arr = randn(rng, 100, 3, n_chains)
        arr[:, 2, :] .= -arr[:, 1, :]
        arr[:, 3, :] .= 0
        wide = chains_to_wide_dataframe(Chains(arr, [Symbol("beta[1]"), Symbol("beta[2]"), Symbol("gamma[1]")]))
        transition = merge((chain = 1,), B._transition_record(
            (numerical_error = true, hamiltonian_energy_error = Inf, acceptance_rate = 0.0), 1, 0))
        table = diagnostic_table(wide; transitions = DataFrame([transition]), runtime = Dict("total_seconds" => 2.5))
        @test Set(table.category) == Set(["convergence", "zero_sum", "selection_mixing", "transition", "runtime"])
        conv = table[table.category .== "convergence", :]
        @test all(==(100n_chains), conv.ess_threshold)
        constrow = conv[conv.parameters .== "gamma[1]", :]
        @test ismissing(only(constrow.ess_bulk))
        @test ismissing(only(constrow.rhat))
        n_chains == 1 && @test all(ismissing, conv.rhat)
        zerosum = table[table.category .== "zero_sum", :]
        @test all(iszero, zerosum.max_abs)
        for (metric, thresholds) in ((:rhat, [1.01, 1.1]), (:ess_bulk, [100n_chains]), (:ess_tail, [100n_chains]))
            p = B._convergence_scatter(table, metric, n_chains)
            @test p.series_list[1][:seriestype] == :scatter
            for (series, threshold) in zip(p.series_list[2:end], thresholds)
                @test series[:linestyle] == :dash
                @test all(==(threshold), filter(isfinite, series[:y]))
                @test series[:seriescolor] == B._require_pkg(:Plots).RGBA(1.0, 0.0, 0.0, 1.0)
            end
            @test length(p.series_list) == 1 + length(thresholds)
        end
        mktempdir() do dir
            result = save_diagnostic_outputs(dataframe_to_chains(wide), dir;
                transitions = DataFrame([transition]), runtime = Dict("total_seconds" => 2.5), plots = false)
            @test isfile(result.diagnostics_summary)
            @test only(read_diagnostics(dir; category = "runtime").total_seconds) == 2.5
            @test sort(filter(p -> endswith(p, ".csv"), readdir(dir))) == ["diagnostics_summary.csv", "transition_diagnostics.csv"]
        end
    end
end
