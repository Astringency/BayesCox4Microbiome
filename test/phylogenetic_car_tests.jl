using Test, Random, LinearAlgebra, Statistics, BayesCox4Micro
if !isdefined(Main, :generate_simulation_data)
    include(joinpath(@__DIR__, "..", "scripts", "lib", "analysis_common.jl"))
end

@testset "S4 Brownian shared ancestry and reproducible data" begin
    # Four leaves: siblings share one of two unit branches, cousins share none.
    small = planted_phylogeny(4, [3, 1]; covariance_model = :brownian)
    order = small.leaf_order
    expected = [1.0 .5 0 0; .5 1 0 0; 0 0 1 .5; 0 0 .5 1]
    @test Matrix(small.covariance)[order, order] == expected
    @test small.newick == "((x3:1,x1:1):1,(x2:1,x4:1):1);"
    @test resolve_simulation_scenario("S4") == :phylo_bm
    @test resolve_simulation_scenario("phylo") == :phylo_bm
    beta, idx = true_beta(50, 6)
    phylo = scenario_phylogeny(50, idx, :S4)
    @test phylo.height == 6
    @test isposdef(phylo.covariance)
    @test diag(phylo.covariance) == ones(50)
    @test phylo.covariance ≈ 1 .- phylo.distance ./ (2phylo.height)
    @test phylo.covariance == phylo.incidence * Diagonal(phylo.branch_lengths) * phylo.incidence' / phylo.height
    @test all(>(0), phylo.branch_lengths)
    @test vec(sum(phylo.incidence; dims=2)) == fill(phylo.height, 50)
    @test phylo.covariance != planted_phylogeny(50, idx; covariance_model=:exponential).covariance
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
    alias_sim = generate_simulation_data(200, 50, 6; scenario=:phylo, seed=17)
    @test alias_sim.X == sim.X && alias_sim.time == sim.time && alias_sim.event == sim.event
    @test test_sim.test_seed == generate_simulation_test_data(200, 50, 6; scenario=:phylo, seed=17).test_seed
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
        # Renaming the CLI alias must not make old exponential results reusable.
        write(joinpath(dir,"metadata.json"),JSON.json(merge(meta,Dict("scenario"=>"phylo"))))
        @test !simulation_metadata_matches_typed(dir,meta)
    end
end

@testset "S4 Oracle graph connects only nonzero coefficients" begin
    beta, idx = true_beta(50,6)
    phylo = scenario_phylogeny(50,idx,:S4)
    w = phylo.raw_tree_weights
    before = copy(w)
    cut = signal_only_car_weights(w,beta)
    removed = [(i,j) for j in 2:50 for i in 1:(j-1) if w[i,j]>0 && cut[i,j]==0]
    @test !isempty(removed)
    @test w == before
    @test cut == phylo.weights
    @test count(>(0),triu(cut,1)) == 7
    @test all(iszero(cut[i,j]) for i in 1:50 for j in 1:50 if iszero(beta[i]) || iszero(beta[j]))
    @test any(phylo.misspecified_weights[i,j]>0 for i in 1:50 for j in 1:50 if iszero(beta[i])!=iszero(beta[j]))
    @test sort(vec(cut)) == sort(vec(phylo.misspecified_weights))
    @test BayesCox4Micro._car_rank(BayesCox4Micro._prepare_car_weights(phylo.misspecified_weights,50)) == 4
    @test signal_only_car_weights(cut,beta) == cut
    @test issymmetric(cut) && all(iszero,diag(cut))
    @test all(cut[i,j]==w[i,j] for i in 1:50 for j in 1:50 if !iszero(beta[i]) && !iszero(beta[j]))
    @test BayesCox4Micro._car_rank(BayesCox4Micro._prepare_car_weights(w,50)) == 37
    @test BayesCox4Micro._car_rank(BayesCox4Micro._prepare_car_weights(cut,50)) == 4
    @test_throws DimensionMismatch signal_only_car_weights(w,beta[1:4])
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

@testset "All S4 graph and constraint combinations reach posterior fitting" begin
    sim = generate_simulation_data(24, 8, 2; scenario = :phylo, seed = 27)
    phylo = planted_phylogeny(8, sim.nonzero_index)
    learned = car_weight_matrix(sim.X;distance=:bray,threshold_quantile=0.5)
    @test learned != phylo.weights
    for constraint in (:soft_center,:prox)
        fits = Any[]
        for weights in (nothing, phylo.weights, phylo.misspecified_weights, learned)
            fit = fit_and_evaluate(sim.X, sim.time, sim.event; prior_name = "cox-rhs",
                sampler = :hmc, n_samples = 12, warmup = 8, chains = 1,
                hmc_n_leapfrog = 3, noncentered = true, p0 = 2, constraint,
                use_car = weights !== nothing, car_weights = weights,
                sample_car_strength = weights !== nothing, seed = 7, progress = false,
                compute_metrics = false)
            @test all(isfinite, fit.draws)
            push!(fits, fit)
        end
        @test fits[1].draws != fits[2].draws
        @test fits[2].draws != fits[3].draws
        @test fits[3].draws != fits[4].draws
    end
end

@testset "CAR threshold uses off-diagonal similarities" begin
    # One pair of columns is closer than all others, so the top off-diagonal
    # similarity survives the 99% threshold. Diagonal self-similarities are
    # large enough that including them would remove every edge.
    X = [1.0 1 0 0 0;
         0 1 0 0 0;
         0 0 1 0 0;
         0 0 0 1 0;
         0 0 0 0 1]
    K = car_similarity_matrix(car_distance_matrix(X))
    p = size(K, 1)
    offdiag = [abs(K[i, j]) for j in 2:p for i in 1:(j - 1)]
    cutoff = quantile(offdiag, 0.99)
    @test cutoff < maximum(offdiag)
    @test quantile(abs.(vec(K)), 0.99) > maximum(offdiag)

    W = car_weight_matrix(X)
    @test W == W'
    @test all(iszero, diag(W))
    @test count(x -> x != 0, W) == 2
    @test maximum(W) == maximum(offdiag)
end
