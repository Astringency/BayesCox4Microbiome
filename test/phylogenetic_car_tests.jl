using Test, Random, LinearAlgebra, BayesCox4Micro
if !isdefined(Main, :generate_simulation_data)
    include(joinpath(@__DIR__, "..", "scripts", "analysis_common.jl"))
end

@testset "S4 Brownian shared ancestry and reproducible data" begin
    # Four leaves: siblings share one of two unit branches, cousins share none.
    small = planted_phylogeny(4, [3, 1]; covariance_model = :brownian)
    order = small.leaf_order
    expected = [1.0 .5 0 0; .5 1 0 0; 0 0 1 .5; 0 0 .5 1]
    @test Matrix(small.covariance)[order, order] == expected
    @test small.newick == "((x3:1,x1:1):1,(x2:1,x4:1):1);"
    @test resolve_simulation_scenario("S4") == :phylo_bm
    beta, idx = true_beta(50, 6)
    phylo = scenario_phylogeny(50, idx, :S4)
    @test phylo.height == 6
    @test isposdef(phylo.covariance)
    @test diag(phylo.covariance) == ones(50)
    @test phylo.covariance ≈ 1 .- phylo.distance ./ (2phylo.height)
    @test phylo.covariance == phylo.incidence * Diagonal(phylo.branch_lengths) * phylo.incidence' / phylo.height
    @test all(>(0), phylo.branch_lengths)
    @test vec(sum(phylo.incidence; dims=2)) == fill(phylo.height, 50)
    @test phylo.covariance != planted_phylogeny(50, idx).covariance
    @test phylo.weights == planted_phylogeny(50, idx).weights
    @test phylo.weights != phylo.misspecified_weights
    @test eigvals(Symmetric(phylo.weights)) ≈ eigvals(Symmetric(phylo.misspecified_weights)) atol=1e-13
    # The covariance of log ratios is also implied by the BM latent model.
    rng = MersenneTwister(427)
    latent = rand(rng, MvNormal(zeros(4), small.covariance), 30000)'
    ratios = log.(softmax_rows(latent))[:, 1:3] .- log.(softmax_rows(latent))[:, 4]
    contrast = hcat(Matrix{Float64}(I, 3, 3), -ones(3))
    @test maximum(abs, cov(ratios) - contrast * small.covariance * contrast') < .05
    sim = generate_simulation_data(200, 50, 6; scenario=:S4, seed=17)
    repeat_sim = generate_simulation_data(200, 50, 6; scenario=:phylo_bm, seed=17)
    test_sim = generate_simulation_test_data(200, 50, 6; scenario=:S4, seed=17)
    @test sim.X == repeat_sim.X
    @test sim.X != generate_simulation_data(200, 50, 6; scenario=:S4, seed=18).X
    @test sim.latent_covariance == test_sim.latent_covariance == phylo.covariance
    @test sim.beta == test_sim.beta == beta
    @test count(!iszero, beta) == 6
    @test abs(sum(beta)) < 1e-12
    @test all(>(0), sim.X)
    @test vec(sum(sim.X; dims=2)) ≈ ones(200)
    @test size(test_sim.X) == (1000, 50)
    @test test_sim.X[1:200, :] != sim.X
    @test occursin("phylo_bm", test_sim.test_seed)
    @test test_sim.test_seed != generate_simulation_test_data(200, 50, 6; scenario=:phylo, seed=17).test_seed
    meta = scenario_metadata(sim)
    @test meta == scenario_metadata(test_sim)
    mktempdir() do dir
        save_planted_phylogeny(dir, phylo)
        @test strip(read(joinpath(dir, "tree.newick"), String)) == phylo.newick
        @test merge(JSON.parsefile(joinpath(dir, "phylogeny.json")), Dict("scenario" => "phylo_bm")) == meta
        # Resume must reject missing or changed scientific fingerprints.
        open(joinpath(dir, "metadata.json"), "w") do io
            JSON.print(io, meta)
        end
        @test simulation_metadata_matches_typed(dir, meta)
        for key in BROWNIAN_DGP_METADATA_KEYS
            saved = copy(meta)
            delete!(saved, key)
            write(joinpath(dir, "metadata.json"), JSON.json(saved))
            @test !simulation_metadata_matches_typed(dir, meta)
        end
        write(joinpath(dir, "metadata.json"), JSON.json(merge(meta, Dict("dgp_version" => "other"))))
        @test !simulation_metadata_matches_typed(dir, meta)
        @test_throws ErrorException save_planted_phylogeny(dir, scenario_phylogeny(50, reverse(idx), :S4))
    end
end

@testset "Planted phylogeny and misspecified graph" begin
    beta, idx = true_beta(50, 6)
    phylo = planted_phylogeny(50, idx)
    @test phylo.leaf_order[1:6] == idx
    @test isposdef(phylo.covariance)
    @test diag(phylo.covariance) == ones(50)
    @test diag(phylo.weights) == zeros(50)
    @test phylo.weights == phylo.weights'
    @test any(>(0), phylo.weights)
    @test phylo.weights != phylo.misspecified_weights
    @test sort(vec(sum(phylo.weights; dims = 2))) ≈ sort(vec(sum(phylo.misspecified_weights; dims = 2)))
    @test eigvals(Symmetric(phylo.weights)) ≈ eigvals(Symmetric(phylo.misspecified_weights)) atol=1e-13
    @test phylo.weights == planted_phylogeny(50, idx).weights
    sim = generate_simulation_data(200, 50, 6; scenario = :phylo, seed = 1)
    @test sim.beta == beta
    @test sim.latent_covariance == phylo.covariance
    @test sim.X == generate_simulation_data(200, 50, 6; scenario = :phylo, seed = 1).X
    @test size(generate_simulation_test_data(200, 50, 6; scenario = :phylo).X) == (1000, 50)
end

@testset "Custom CAR weights reach posterior fitting" begin
    sim = generate_simulation_data(24, 8, 2; scenario = :phylo, seed = 27)
    phylo = planted_phylogeny(8, sim.nonzero_index)
    fits = Any[]
    for weights in (nothing, phylo.weights, phylo.misspecified_weights)
        fit = fit_and_evaluate(sim.X, sim.time, sim.event; prior_name = "cox-rhs",
            sampler = :hmc, n_samples = 12, warmup = 8, chains = 1,
            hmc_n_leapfrog = 3, noncentered = true, p0 = 2,
            use_car = weights !== nothing, car_weights = weights,
            sample_car_strength = weights !== nothing, seed = 7, progress = false,
            compute_metrics = false)
        @test all(isfinite, fit.draws)
        push!(fits, fit)
    end
    @test fits[1].draws != fits[2].draws
    @test fits[2].draws != fits[3].draws
end
