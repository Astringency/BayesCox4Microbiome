using BayesCox4Micro: _actual_fit_kwargs, _cindex_fit_kwargs
using Test
isdefined(Main, :fit_and_evaluate) || include(joinpath(@__DIR__, "..", "scripts", "lib", "analysis_common.jl"))

@testset "Real-data alpha scale survives every tuning candidate" begin
    for prior in ("prox-lasso", "prox-sslasso")
        for config in PRIOR_PARAM_GRIDS[prior]
            @test !haskey(config, "alpha_prior_scale")
            for scale in (30.0, 32.0)
                kw = _actual_fit_kwargs(prior, Dict(), Dict(:alpha_prior_scale => scale),
                    Dict(), Dict(), config; p=30, realdata=true)
                @test kw[:alpha_prior_scale] == scale
            end
            kw = _actual_fit_kwargs(prior, Dict(), Dict(:alpha_prior_scale => nothing),
                Dict(), Dict(), config; p=30, realdata=true)
            @test kw[:alpha_prior_scale] == 30.0
            joint_kw = _cindex_fit_kwargs(prior, Dict(), Dict(), Dict(), Dict(), config,
                30, 1; clinical_columns=2)
            @test joint_kw[:alpha_prior_scale] == 32.0
        end
    end
    @test length(PRIOR_PARAM_GRIDS["prox-sslasso"]) == 57
end

@testset "Final fits use the same alpha default as tuning" begin
    X = [.2 .3 .5; .5 .1 .4; .1 .6 .3; .4 .4 .2; .3 .5 .2; .2 .6 .2]
    time, event = collect(1.:6.), [1,1,0,1,1,0]
    for clinical in (nothing, reshape([0.,1.,0.,1.,0.,1.],:,1))
        kwargs = (prior_name="prox-sslasso",sampler=:hmc,realdata=true,
            clinical_X=clinical,n_samples=16,warmup=8,chains=1,multichains=false,
            hmc_n_leapfrog=3,seed=7,progress=false,compute_metrics=false,
            sslasso_spike_scale=.04,sslasso_slab_scale=.5)
        automatic = fit_and_evaluate(X,time,event;kwargs...)
        explicit = fit_and_evaluate(X,time,event;kwargs...,
            alpha_prior_scale=size(X,2)+(clinical===nothing ? 0 : size(clinical,2)))
        @test automatic.result.samples.draws == explicit.result.samples.draws
        @test automatic.result.samples.logposterior == explicit.result.samples.logposterior
    end
end
