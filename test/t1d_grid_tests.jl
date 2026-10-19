using Test, BayesCox4Micro
@testset "T1D relaxed-plus-two grid stays separate" begin
    B=BayesCox4Micro
    @test length(B.T1D_SSLASSO_PARAM_GRID)==21
    pairs=Set((c["sslasso_spike_scale"],c["sslasso_slab_scale"]) for c in B.T1D_SSLASSO_PARAM_GRID)
    @test pairs==Set([(2.,10.),(2.,20.),(5.,20.),(5.,50.),(10.,50.),(.04,.5),(1.,10.)])
    for pair in pairs
        @test Set(c["prox_l1_lambda"] for c in B.T1D_SSLASSO_PARAM_GRID if
            (c["sslasso_spike_scale"],c["sslasso_slab_scale"])==pair)==Set([1e-5,1e-4,1e-3])
    end
    @test length(B.PRIOR_PARAM_GRIDS["prox-sslasso"])==57
    copy=B.t1d_prior_param_grids();copy["prox-sslasso"][1]["sslasso_spike_scale"]=999.
    @test B.T1D_SSLASSO_PARAM_GRID[1]["sslasso_spike_scale"]==2.
    @test B.t1d_prior_param_grids()["cox-rhs"]==B.PRIOR_PARAM_GRIDS["cox-rhs"]
end

@testset "T1D extended grid preserves existing candidate seeds and ordering" begin
    B=BayesCox4Micro
    old=B.t1d_prior_param_grids()["prox-sslasso"]
    extended=B.t1d_prior_param_grids(;version=B.T1D_SSLASSO_EXTENDED_GRID_VERSION)["prox-sslasso"]
    @test length(extended)==28
    @test extended[1:21]==old
    @test [(c["sslasso_spike_scale"],c["sslasso_slab_scale"]) for c in extended[22:28]]==B.T1D_SSLASSO_SCALE_PAIRS
    @test all(c["prox_l1_lambda"]==1e-6 for c in extended[22:28])
    @test length(unique(extended))==28
    @test_throws ErrorException B.t1d_prior_param_grids(;version="unknown")
end
