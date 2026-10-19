using Test, BayesCox4Micro, Random, LinearAlgebra
@testset "Time-effect and prior targets preserve static priors and gradients" begin
    B=BayesCox4Micro;rng=MersenneTwister(810)
    X=rand(rng,18,4).+.2;X./=sum(X;dims=2);time=Float64.(1:18);event=Int.(mod.(1:18,3).!=0)
    C=Float64.(rand(rng,Bool,18,2));data=prepare_cox_data(X,time,event)
    design=B.TimeCoxDesign(hcat(data.z,C),time,event)
    for method in ("cox-rhs","prox-sslasso"), constraint in (:prox,:soft_center), weight in (0.,1.)
        prior=prior_from_name(method,X,time,event;p0=2.,realdata=true)
        if method=="cox-rhs"
            base=B._make_explicit_nuts_target(data,prior;constraint,prox_sum0_lambda=1e-3,noncentered=true)
        else
            base=B.CoxProxSSLassoTarget(data,prior,Bool[true,false,true,false],Ref(.5);
                constraint,prox_sum0_lambda=1e-3,prox_l1_lambda=1e-3,
                alpha_prior_shape=1.,alpha_prior_scale=6.,alpha_prior_kind=:invgamma,
                prox_root_tol=1e-10,prox_root_maxiter=100,init=:zero)
        end
        joint=B.CoxClinicalTarget(base,C)
        target=B.CoxTimeVaryingTarget(joint,design,ph_time_design(4,2,"both");likelihood_weight=weight)
        u=initial_state(target,rng);u .+= .01randn(rng,length(u))
        lp,g=B.logdensity_and_gradient!(zeros(length(u)),target,u)
        h=1e-6
        finite_difference=[begin
            plus=copy(u);minus=copy(u);plus[j]+=h;minus[j]-=h
            (logdensity(target,plus)-logdensity(target,minus))/(2h)
        end for j in eachindex(u)]
        @test g≈finite_difference rtol=3e-4 atol=3e-4
        d=dimension(joint);u[d+1:end].=0
        b=B.sample_state(joint,u[1:d])[target.coefficient_indices]
        likelihood=time_cox_loglikelihood(design.X,time,event,b,zeros(6))
        expected=logdensity(joint,u[1:d])-(1-weight)*likelihood-5log(.5sqrt(2pi))
        @test logdensity(target,u)≈expected
        if method=="prox-sslasso" && weight==0
            radius=B.CoxRadiusPriorTarget(target)
            internal=B._radius_internal(radius,u)
            @test B.sample_state(radius,internal)≈u atol=1e-12
            jac=B._radius_scales(radius,internal)[3]
            @test logdensity(radius,internal)≈logdensity(target,u)+jac atol=1e-8
            _,rg=B.logdensity_and_gradient!(zeros(length(u)),radius,internal)
            fd=[begin
                plus=copy(internal);minus=copy(internal);plus[j]+=h;minus[j]-=h
                (logdensity(radius,plus)-logdensity(radius,minus))/(2h)
            end for j in eachindex(internal)]
            @test rg≈fd rtol=3e-4 atol=3e-4
            J=hcat([begin
                plus=copy(internal);minus=copy(internal);plus[j]+=h;minus[j]-=h
                (B.sample_state(radius,plus).-B.sample_state(radius,minus))./(2h)
            end for j in eachindex(internal)]...)
            @test first(logabsdet(J))≈jac atol=1e-6
        end
    end
end

@testset "SSL prior coordinates across smoothing and radius regimes" begin
    B=BayesCox4Micro;rng=MersenneTwister(901);p=30
    X=rand(rng,18,p).+.2;X./=sum(X;dims=2)
    time=Float64.(1:18);event=Int.(mod.(1:18,3).!=0)
    data=prepare_cox_data(X,time,event);design=B.TimeCoxDesign(data.z,time,event)
    prior=prior_from_name("prox-sslasso",X,time,event;realdata=true,
        sslasso_spike_scale=.04,sslasso_slab_scale=.5)
    for constraint in (:prox,:soft_center), alpha in (.2,1.,100.), mixed in (false,true)
        gamma=mixed ? Bool[isodd(j) for j in 1:p] : falses(p)
        base=B.CoxProxSSLassoTarget(data,prior,gamma,Ref(.3);
            constraint,prox_sum0_lambda=1e-6,prox_l1_lambda=1e-3,
            alpha_prior_shape=1.,alpha_prior_scale=30.,alpha_prior_kind=:invgamma,
            prox_root_tol=1e-10,prox_root_maxiter=100,init=:zero)
        target=B.CoxTimeVaryingTarget(base,design,ph_time_design(p,0,"static");likelihood_weight=0.)
        radius=B.CoxRadiusPriorTarget(target)
        u=vcat(randn(rng,p),log(alpha));v=B.sample_state(radius,u)
        jac=B._radius_scales(radius,u)[3];h=1e-6
        @test B._radius_internal(radius,v)≈u atol=1e-10
        @test logdensity(radius,u)≈logdensity(target,v)+jac atol=1e-8
        _,g=B.logdensity_and_gradient!(zeros(length(u)),radius,u)
        fd=[begin
            plus=copy(u);minus=copy(u);plus[j]+=h;minus[j]-=h
            (logdensity(radius,plus)-logdensity(radius,minus))/(2h)
        end for j in eachindex(u)]
        @test g≈fd rtol=3e-4 atol=3e-4
        J=hcat([begin
            plus=copy(u);minus=copy(u);plus[j]+=h;minus[j]-=h
            (B.sample_state(radius,plus).-B.sample_state(radius,minus))./(2h)
        end for j in eachindex(u)]...)
        @test first(logabsdet(J))≈jac atol=1e-6
    end
end
