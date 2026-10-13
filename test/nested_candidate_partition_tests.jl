using Test
include(joinpath(@__DIR__,"..","scripts","analysis_common.jl"))
include(joinpath(@__DIR__,"..","scripts","tune_hyperparams_cindex.jl"))
@testset "Disjoint candidate checkpoints reproduce serial selection" begin
    X=hcat(collect(1.:24.),ones(24));time=collect(1.:24.);event=ones(Int,24)
    calls=Ref(0)
    fit=(x,t,e;kwargs...)->begin
        calls[]+=1
        (ids=Int.(x[:,1]),value=kwargs[:rhs_local_scale])
    end
    score=(f,xt,xv,tv,ev;requested_pseudocount)->begin
        @test isempty(intersect(f.ids,Int.(xv[:,1])))
        Dict("mean"=>f.value/3,"median"=>f.value/3,"MAP"=>f.value/3)
    end
    configs=[Dict{Symbol,Any}(:rhs_local_scale=>v) for v in [.5,1.,2.]]
    kw=(prior="cox-rhs",configs=configs,base_sampler_kwargs=Dict(:sampler=>:mh),
        base_prior_kwargs=Dict(),base_car_kwargs=Dict(),base_model_kwargs=Dict(:p0=>nothing),
        cv_folds=3,seed=7,fit_function=fit,score_function=score)
    serial=_select_inner_cindex(X,time,event;kw...)
    mktempdir() do p
        calls[]=0
        _select_inner_cindex(X,time,event;kw...,checkpoint_dir=p,config_indices=Int[])
        @test calls[]==0
        _select_inner_cindex(X,time,event;kw...,checkpoint_dir=p,config_indices=[2])
        @test calls[]==3
        _select_inner_cindex(X,time,event;kw...,checkpoint_dir=p,config_indices=[1,3])
        @test calls[]==9
        assembled=_select_inner_cindex(X,time,event;kw...,checkpoint_dir=p)
        @test calls[]==9 # selection from all saved candidates does not refit
        @test assembled.config==serial.config
        @test assembled.config_index==serial.config_index==3
        @test assembled.score==serial.score
        @test isequal(assembled.metrics[:,sort(names(assembled.metrics))],serial.metrics[:,sort(names(serial.metrics))])
    end
end
