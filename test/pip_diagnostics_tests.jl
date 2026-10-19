using Test, BayesCox4Micro, DataFrames, Random, Statistics

@testset "Indicator PIP consistency" begin
    B = BayesCox4Micro
    rng = MersenneTwister(18010)
    wide = DataFrame(chain = repeat(1:4; inner = 1000), iter = repeat(1:1000, 4))
    wide[!, "gamma[1]"] = vcat([Float64.(rand(rng, 1000) .< p) for p in (0.2, 0.4, 0.6, 0.8)]...)
    wide[!, "gamma[2]"] = zeros(4000)
    wide[!, "gamma[3]"] = repeat([0., 0., 1., 1.]; inner = 1000)
    wide[!, "gamma[4]"] = copy(wide[!, "gamma[1]"])
    wide[1:1000, "gamma[4]"] .= 0
    mixing = B.selection_mixing_diagnostics(wide)
    result = pip_consistency_diagnostics(wide; mixing)
    g = mixing[mixing.parameter .== "gamma[1]", :]
    r = result[result.parameter .== "gamma[1]", :]
    @test only(r.pip_range) == maximum(g.inclusion_probability) - minimum(g.inclusion_probability)
    @test only(r.pip_mcse_standardized_max) ≈ maximum(
        abs(g.inclusion_probability[c] - g.inclusion_probability[d]) / hypot(g.pip_mcse[c], g.pip_mcse[d])
        for c in 1:4 for d in c+1:4)
    @test only(r.n_estimable_chain_pairs) == 6
    @test result.pip_range[2:3] == [0., 1.]
    @test all(ismissing, result.pip_mcse_standardized_max[2:4])
    @test result.conflicting_constant_chains == [false, false, true, false]
    @test result.n_estimable_chain_pairs[4] == 3
    @test !ismissing(result.pip_mcse_standardized_available_max[4])
    @test all(ismissing, mixing.pip_mcse[mixing.constant_chain])
    @test isequal(result, pip_consistency_diagnostics(wide[randperm(rng, nrow(wide)), :])[indexin(result.parameter,
        pip_consistency_diagnostics(wide[randperm(rng, nrow(wide)), :]).parameter), :])
    one_chain = pip_consistency_diagnostics(wide[wide.chain .== 1, :])
    @test all(ismissing, one_chain.pip_range)
    @test all(ismissing, one_chain.pip_mcse_standardized_max)
    @test isempty(pip_consistency_diagnostics(select(wide, :chain, :iter)))
    bad = copy(wide); bad[1, "gamma[1]"] = 0.2
    @test_throws ArgumentError pip_consistency_diagnostics(bad)
    @test_throws ArgumentError pip_consistency_diagnostics(vcat(wide, wide[1:1, :]))

    # Equal PIP and draw count, but long dwell times must increase mean MCSE.
    independent = Float64.(rand(rng, 2000) .< 0.5)
    correlated = repeat(independent; inner = 20)
    shuffled = correlated[randperm(rng, length(correlated))]
    a = B._indicator_chain_mcse(reshape(correlated, :, 1), [Symbol("gamma[1]")])[1]
    b = B._indicator_chain_mcse(reshape(shuffled, :, 1), [Symbol("gamma[1]")])[1]
    @test a > 2b
end
