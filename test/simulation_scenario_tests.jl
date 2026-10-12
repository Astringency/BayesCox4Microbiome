using Test, Random, LinearAlgebra, Statistics, Distributions
if !isdefined(Main, :generate_simulation_data)
    include(joinpath(@__DIR__, "..", "scripts", "analysis_common.jl"))
end

@testset "Legacy DGP remains byte-for-byte reproducible" begin
    n, p, nz, seed = 40, 50, 6, 187
    rng = MersenneTwister(seed)
    beta, idx = true_beta(p, nz)
    mu = zeros(p)
    mu[idx] .= log(0.5p)
    X = softmax_rows(rand(rng, MvNormal(mu, paper_covariance(p, idx)), n)')
    eta = log.(X) * beta
    params = baseline_parameters(:weibull)
    event_time = [baseline_event_time(rng, eta[i], :weibull; rate = params.rate, shape = params.shape) for i in 1:n]
    observed = apply_censoring(event_time, random_censoring_times(rng, n; upper = baseline_censoring_upper(:weibull)))
    sim = generate_simulation_data(n, p, nz; seed)
    @test sim.X == X
    @test sim.eta == eta
    @test sim.event_time == event_time
    @test sim.time == observed.time
    @test sim.event == observed.event
    @test sim.scenario == :baseline
end

@testset "Weak signals, correlated nulls and sequencing counts" begin
    baseline = generate_simulation_data(200, 50, 6; seed = 17)
    weak = generate_simulation_data(200, 50, 6; seed = 17, scenario = :S1)
    @test weak.beta == 0.35 .* baseline.beta
    @test weak.X == baseline.X
    @test weak.eta ≈ 0.35 .* baseline.eta
    @test weak.nonzero_index == baseline.nonzero_index
    s2 = generate_simulation_data(200, 50, 6; seed = 17, scenario = :S2)
    @test length(s2.paired_nulls) == 6
    @test all(iszero, s2.beta[s2.paired_nulls])
    @test isposdef(s2.latent_covariance)
    @test diag(s2.latent_covariance) ≈ ones(50)
    for (signal, null) in zip(s2.nonzero_index, s2.paired_nulls)
        @test s2.latent_covariance[signal, null] ≈ 0.6
    end
    s3 = generate_simulation_data(200, 50, 6; seed = 17, scenario = :S3)
    @test vec(sum(s3.sequencing_counts; dims = 2)) == s3.sequencing_depths
    @test 4000 < median(s3.sequencing_depths) < 6000
    @test 0.05 < s3.observed_zero_fraction < 0.8
    @test s3.observed_zero_fraction == mean(s3.X .== 0)
    @test s3.X == s3.sequencing_counts ./ s3.sequencing_depths
    @test vec(sum(s3.X; dims = 2)) ≈ ones(200)
    @test s3.eta ≈ log.(s3.latent_X) * s3.beta
    @test all(>(0), s3.latent_X)
    @test all(isfinite, s3.event_time)
    data = prepare_cox_data(s3.X, s3.time, s3.event; pseudocount = 0.0)
    replaced = copy(s3.X)
    replacement = minimum(filter(>(0), vec(s3.X))) / 2
    replaced[replaced .== 0] .= replacement
    replaced ./= sum(replaced; dims = 2)
    @test exp.(data.z) ≈ replaced
    @test all(isfinite, data.z)
    test = generate_simulation_test_data(200, 50, 6; seed = 17, scenario = :counts)
    @test size(test.X) == (1000, 50)
    @test test.beta == s3.beta
    @test test.scenario == :counts
    @test test.X[1:200, :] != s3.X
    @test test.X == generate_simulation_test_data(200, 50, 6; seed = 17, scenario = :counts).X
    @test_throws ArgumentError generate_simulation_data(20, 8, 2; scenario = :S2)
    @test_throws ArgumentError generate_simulation_data(20, 50, 6; scenario = :S3, depth_median = 0)
end

@testset "DGP metadata prevents incompatible result reuse" begin
    mktempdir() do dir
        meta = Dict("prior" => "cox-rhs", "scenario" => "weak", "weak_multiplier" => 0.35)
        open(joinpath(dir, "metadata.json"), "w") do io
            JSON.print(io, meta)
        end
        @test simulation_metadata_matches_typed(dir, meta)
        @test !simulation_metadata_matches_typed(dir, merge(meta, Dict("weak_multiplier" => 0.5)))
        @test !simulation_metadata_matches_typed(dir, Dict("scenario" => "baseline"))
        open(joinpath(dir, "metadata.json"), "w") do io
            JSON.print(io, Dict("prior" => "cox-rhs"))
        end
        @test simulation_metadata_matches_typed(dir, Dict("scenario" => "baseline"))
        @test !simulation_metadata_matches_typed(dir, meta)
    end
end
