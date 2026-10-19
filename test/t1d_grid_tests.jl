using Test, BayesCox4Micro
@testset "T1D six-pair grid stays separate" begin
    B=BayesCox4Micro
    @test length(B.T1D_SSLASSO_PARAM_GRID)==24
    pairs=Set((c["sslasso_spike_scale"],c["sslasso_slab_scale"]) for c in B.T1D_SSLASSO_PARAM_GRID)
    @test pairs==Set([(1.,10.),(2.,10.),(2.,20.),(5.,20.),(5.,50.),(10.,50.)])
    for pair in pairs
        @test Set(c["prox_l1_lambda"] for c in B.T1D_SSLASSO_PARAM_GRID if
            (c["sslasso_spike_scale"],c["sslasso_slab_scale"])==pair)==Set([1e-5,1e-4,1e-3,1e-6])
    end
    @test length(B.PRIOR_PARAM_GRIDS["prox-sslasso"])==57
    copy=B.t1d_prior_param_grids();copy["prox-sslasso"][1]["sslasso_spike_scale"]=999.
    @test B.T1D_SSLASSO_PARAM_GRID[1]["sslasso_spike_scale"]==1.
    @test B.t1d_prior_param_grids()["cox-rhs"]==B.PRIOR_PARAM_GRIDS["cox-rhs"]
    @test_throws ErrorException B.t1d_prior_param_grids(;version="unknown")
end
