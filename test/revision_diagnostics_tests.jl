using Test, Random, Statistics, DataFrames, CSV, JLD2
using BayesCox4Micro

@testset "Zero-sum distribution and indicator mixing summaries" begin
    B = BayesCox4Micro
    wide = DataFrame(chain = repeat(1:2; inner = 6), iter = repeat(1:6, 2))
    wide[!, "beta[1]"] = collect(1.0:12.0)
    wide[!, "beta[2]"] = -collect(1.0:12.0) .+ repeat([-0.1, 0, 0.1], 4)
    wide[!, "gamma[1]"] = repeat([0, 0, 1, 1, 0, 1], 2)
    wide[!, "gamma[2]"] = zeros(12)
    sums = B.zero_sum_diagnostics(wide)
    @test nrow(sums) == 3
    pooled = only(eachrow(sums[sums.chain .== "pooled", :]))
    @test pooled.mean ≈ 0 atol=1e-14
    @test pooled.rms ≈ sqrt(2/3) * 0.1
    @test pooled.max_abs ≈ 0.1
    @test pooled.prob_abs_lt_1e_8 ≈ 1/3
    mixing = B.selection_mixing_diagnostics(wide)
    active = mixing[mixing.parameter .== "gamma[1]", :]
    @test all(==(3), active.n_switches)
    @test all(==(2), active.n_zero_to_one)
    @test all(==(1), active.n_one_to_zero)
    @test all(==(2), active.longest_dwell_draws)
    @test all(==(0.6), active.switch_rate)
    ranks = B.rank_convergence_diagnostics(wide)
    constrow = ranks[string.(ranks.parameters) .== "gamma[2]", :]
    @test only(constrow.constant_all_draws)
    @test ismissing(only(constrow.rhat))
    @test :ess_tail in propertynames(ranks)
end

@testset "HMC diagnostics survive output saving" begin
    B = BayesCox4Micro
    rng = MersenneTwister(32)
    X = exp.(randn(rng, 20, 4))
    for sampler in (:hmc, :nuts_explicit), ssl in (false, true)
        prior = ssl ? SpikeSlabLassoPrior(proximal = true) : RegularizedHorseshoePrior(p0 = 2.0)
        extra = sampler == :hmc ? (; hmc_n_leapfrog = 3) : (; max_depth = 3)
        thin = sampler == :hmc ? 2 : 1
        n_transitions = 8 + 8thin
        result = sample_bayes_cox(X, collect(1.0:20.0), ones(Int, 20);
            prior = prior, sampler = sampler, constraint = :exact,
            n_samples = 8, warmup = 8, thin = thin, seed = 84,
            chains = 1, progress = false, extra...)
        transitions = B._result_transitions(result)
        @test nrow(transitions) == n_transitions
        @test count(==("warmup"), transitions.phase) == 8
        @test count(==("sample"), transitions.phase) == 8thin
        @test all(x -> 0 <= x <= 1, transitions.acceptance_probability)
        @test eltype(transitions.divergent) == Bool
        summary = B.transition_diagnostics_summary(transitions)
        @test sum(summary.n_transitions) == n_transitions
        mktempdir() do dir
            save_run_outputs(result; output_dir = dir,
                settings = Dict("prox_sum0_lambda" => 1e-3, "noncentered" => true))
            recorded = B._require_pkg(:JSON).parse(read(joinpath(dir, "metadata.json"), String))
            @test recorded["prox_sum0_lambda"] == 1e-3
            @test recorded["noncentered"] == true
            for file in ("zero_sum_summary.csv", "rank_diagnostics.csv", "transition_diagnostics.csv", "transition_summary.csv")
                @test isfile(joinpath(dir, file))
            end
            @test nrow(CSV.read(joinpath(dir, "transition_diagnostics.csv"), DataFrame)) == n_transitions
            @test isfile(joinpath(dir, "selection_mixing.csv")) == ssl
            bundle = JLD2.load(joinpath(dir, "result_bundle.jld2"), "result_bundle")
            @test bundle["diagnostics"]["transition_diagnostics_available"]
            @test nrow(bundle["transition_diagnostics"]) == n_transitions
        end
    end
    @test B._transition_record((acceptance_rate = 0.1, hamiltonian_energy_error = 1001.0, numerical_error = false), 1, 0).divergent
end
