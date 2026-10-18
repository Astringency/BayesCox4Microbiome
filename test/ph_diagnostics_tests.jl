module PHDiagnosticsTests
using Test, BayesCox4Micro, Random, LinearAlgebra, Statistics, Distributions, CSV, DataFrames, JSON
include(joinpath(@__DIR__,"..","scripts","lib","diagnostics","ph.jl"))

@testset "Time Cox likelihood evaluates each whole risk set at event time" begin
    X=[0.2 -0.1; 0.8 0.3; -0.5 0.4; 0.6 -0.7; 0.3 0.9; -0.1 -0.4]
    t=[1.,1.,1.,2.,3.,4.];e=[1,1,0,1,0,1];b=[.2,-.1];s=[-.3,.4]
    data=prepare_cox_data(exp.(X),t,e;closure=false)
    @test time_cox_loglikelihood(X,t,e,b,zeros(2))≈cox_logpartiallikelihood(data,b)
    manual=0.0
    center=mean(log.(t[e.==1]));scale=std(log.(t[e.==1]))
    for tt in (1.,2.,4.)
        R=findall(t.>=tt);D=findall((t.==tt).&(e.==1));d=length(D)
        eta=X*(b+s*((log(tt)-center)/scale))
        manual+=sum(eta[D])-sum(log(sum(exp.(eta[R]))-l/d*sum(exp.(eta[D]))) for l in 0:d-1)
    end
    @test time_cox_loglikelihood(X,t,e,b,s)≈manual
    for scope in ("compositional","noncompositional","both")
        T=ph_time_design(4,2,scope)
        @test T'*T≈Matrix{Float64}(I,size(T,2),size(T,2))
        @test maximum(abs.(sum(T[1:4,:];dims=1)))<1e-14
    end
    @test size(ph_time_design(4,0,"static"))==(4,0)
    @test_throws ErrorException ph_time_design(4,0,"noncompositional")
end

@testset "SMC evidence has prior normalization and discrete pattern mass" begin
    reference=MvNormal([0.],reshape([1.],1,1))
    lp(u,g)=logpdf(Normal(),u[1])+7.0
    ll(u,g)=logpdf(Normal(u[1],0.8),0.7)
    result=smc_evidence(lp,ll,reference;particles=3000,moves=5,seed=972)
    analytic=logpdf(Normal(0,sqrt(1+0.8^2)),0.7)
    @test result.prior_logZ≈7.0 atol=1e-12
    @test result.logZ≈analytic atol=0.07
    shifted=smc_evidence((u,g)->lp(u,g)+11,ll,reference;particles=3000,moves=5,seed=972)
    @test shifted.logZ≈result.logZ atol=1e-10
    @test last(result.trace).temperature==1
    discrete=smc_evidence((u,g)->logpdf(Normal(),u[1])+log(g[1] ? .25 : .75),
        (u,g)->log(g[1] ? .8 : .2),reference;nbits=1,particles=4000,moves=5,seed=25)
    @test exp(discrete.logZ)≈.25*.8+.75*.2 atol=.025
    @test mean(discrete.indicators)≈(.25*.8)/(.25*.8+.75*.2) atol=.035
end

@testset "PH uses the static priors and all applicable scopes" begin
    rng=MersenneTwister(42)
    X=rand(rng,24,4).+.1;X./=sum(X;dims=2)
    t=Float64.(1:24);e=Int.(mod.(1:24,3).!=0);C=Float64.(rand(rng,Bool,24,2))
    mktempdir() do root
        for method in ("cox-rhs","prox-sslasso"), constraint in (:soft_center,:prox)
            arm=joinpath(root,string(constraint),"adjusted",method);mkpath(arm)
            selected=method=="cox-rhs" ? Dict("rhs_local_scale"=>.5) :
                Dict("sslasso_spike_scale"=>.1,"sslasso_slab_scale"=>1.,"prox_l1_lambda"=>1e-3,"alpha_prior_shape"=>1.)
            prior=prior_from_name(method,X,t,e;p0=2.,realdata=true,
                (Symbol(k)=>v for (k,v) in selected if k in ("rhs_local_scale","sslasso_spike_scale","sslasso_slab_scale"))...)
            fit=sample_clinical_cox(X,t,e,C;prior,constraint,prox_sum0_lambda=1e-3,
                alpha_prior_scale=6.,prox_l1_lambda=1e-3,n_samples=30,warmup=15,chains=1,hmc_n_leapfrog=3,seed=91)
            spec=Dict("method_name"=>method,"component_cols"=>["x$j" for j in 1:4],
                "constraint"=>string(constraint),"adjustment"=>"adjusted","prox_sum0_lambda"=>1e-3,"alpha_prior_scale"=>6.)
            BayesCox4Micro._write_tuning_json(joinpath(arm,"analysis_specification.json"),spec)
            BayesCox4Micro._write_tuning_json(joinpath(arm,"full_data_tuning","selected_config.json"),selected)
            save_run_outputs(fit;output_dir=joinpath(arm,"runs","realdata_"*method),metadata=Dict("p0"=>2.),save_jld2=false)
            clinical_names=["clinical[1]","clinical[2]"]
            design=DataFrame(hcat(fit.data.z,C),vcat(spec["component_cols"],clinical_names))
            insertcols!(design,1,:subject_index=>1:24,:time=>t,:status=>e)
            CSV.write(joinpath(arm,"ph_design.csv"),design)
            for scope in ("static","compositional","noncompositional","both")
                model=ph_model(arm,scope)
                u=mean(model.reference);g=falses(model.nbits)
                @test isfinite(model.logprior(u,g))
                @test isfinite(model.loglikelihood(u,g))
                @test abs(sum(model.slopes(u)[1:4]))<1e-12
                if scope=="static"
                    expected=cox_logpartiallikelihood(BayesCox4Micro.joint_cox_data(fit.data,C),model.coefficients(u))
                    @test model.loglikelihood(u,g)≈expected
                end
            end
            # Exercise the complete evidence/posterior/aggregation pipeline at a
            # small integration budget; statistical accuracy is tested analytically above.
            if (method=="cox-rhs" && constraint==:soft_center) || (method=="prox-sslasso" && constraint==:prox)
                out=run_ph_models(arm;particles=96,repetitions=2,moves=3,max_stages=1000)
                tab=CSV.read(joinpath(out,"evidence_repetitions.csv"),DataFrame)
                @test nrow(tab)==8
                comparison=aggregate_ph_evidence(arm)
                bf=CSV.read(joinpath(comparison,"bayes_factors.csv"),DataFrame)
                @test nrow(bf)==12
                for row in eachrow(bf)
                    reverse=bf[(bf.numerator.==row.denominator).&(bf.denominator.==row.numerator),:]
                    @test row.log_bayes_factor≈-only(reverse.log_bayes_factor)
                end
                @test_throws ErrorException run_ph_models(arm;particles=97,repetitions=2,moves=3,max_stages=1000)
            end
        end
        cross=aggregate_ph_evidence(root)
        table=CSV.read(joinpath(cross,"bayes_factors.csv"),DataFrame)
        @test nrow(table)==8*7
        @test any((table.numerator.=="soft_center/adjusted/cox-rhs/static") .&
                  (table.denominator.=="prox/adjusted/prox-sslasso/both"))
        @test count(==("static_vs_static"),table.comparison)==2
    end
end
end
