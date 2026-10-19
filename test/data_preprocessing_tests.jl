module ZeroAbundancePreprocessingTests
using BayesCox4Micro: _validation_design_matrix

using Test
using BayesCox4Micro
using Statistics


@testset "zero-only abundance preprocessing" begin
    X = [0.0 0.2; 0.4 0.6; 0.0 0.0]
    original = copy(X)
    time, event = [1.0, 2.0, 3.0], [1, 0, 1]
    replaced = [0.1 0.2; 0.4 0.6; 0.1 0.1]

    raw = prepare_cox_data(X, time, event; closure = false)
    @test raw.z ≈ log.(replaced)
    @test raw.z[X .> 0] == log.(X[X .> 0])
    @test raw.zero_replacement == 0.1
    @test X == original

    closed = prepare_cox_data(X, time, event)
    @test closed.z ≈ log.([1 / 3 2 / 3; 0.4 0.6; 0.5 0.5])
    @test prepare_cox_data(X, time, event; transform = :none).z == X

    positive = [0.1 0.9; 0.4 0.6; 0.2 0.8]
    data = prepare_cox_data(positive, time, event; closure = false)
    @test data.z == log.(positive)
    @test data.zero_replacement == 0.0
    @test_throws ArgumentError prepare_cox_data(zeros(3, 2), time, event)
    @test_throws ArgumentError prepare_cox_data([0.0 -0.2; 0.4 0.6; 0.1 0.9], time, event)

    for pc in (0.0, 1e-6)
        @test_throws MethodError prepare_cox_data(X, time, event; pseudocount = pc)
    end
    @test_throws MethodError sample_bayes_cox(X, time, event; sampler = :mh, n_samples = 1, warmup = 0, pseudocount = 0.0)
    @test_throws MethodError estimate_p0_univariate_cox(X, time, event; pseudocount = 1e-6)
    @test_throws MethodError cox_paper_prior("cox-rhs"; p0 = 1, pseudocount = 1e-6)

    samples = MHSamples(zeros(2, 2), zeros(2), ["beta[1]", "beta[2]"], 1.0, 1.0, zeros(2))
    result = CoxSamplerResult(samples, closed, BayesianLassoPrior(), :mh)
    metadata = BayesCox4Micro._base_metadata(result)
    @test !haskey(metadata, "pseudocount")
    @test metadata["zero_replacement"] == 0.1
    @test metadata["zero_handling"] == "half_minimum_positive"
end

@testset "held-out zeros use only the training minimum" begin
    time, event = [1.0, 2.0, 3.0], [1, 0, 1]
    validation = [0.0 0.001; 0.2 0.8]
    original = copy(validation)
    # Both training cohorts have minimum positive abundance 0.2, including
    # the case where no training zeros were replaced.
    for X in ([0.0 0.2; 0.4 0.6; 0.8 0.2], [0.8 0.2; 0.4 0.6; 0.6 0.4])
        for standardize in (false, true), closure in (false, true)
            train = prepare_cox_data(X, time, event; standardize, closure)
            expected = log.(closure ? [0.1 / 0.101 0.001 / 0.101; 0.2 0.8] : [0.1 0.001; 0.2 0.8])
            if standardize
                expected = (expected .- reshape(train.col_mean, 1, :)) ./ reshape(train.col_scale, 1, :)
            end
            @test _validation_design_matrix(X, validation, train) ≈ expected
            @test _validation_design_matrix(X, X, train) ≈ train.z
            @test_throws MethodError _validation_design_matrix(X, validation, train; requested_pseudocount = 1e-6)
        end
    end
    @test validation == original
end

end
