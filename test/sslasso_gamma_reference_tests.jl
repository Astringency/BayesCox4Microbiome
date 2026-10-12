using Test, BayesCox4Micro

@testset "SSLASSO Gibbs indicators remain shared with continuous target" begin
    data = prepare_cox_data([.2 .3 .5; .5 .1 .4; .1 .6 .3; .4 .4 .2], [1.,2.,3.,4.], [1,1,0,1])
    for gamma in (trues(3), fill(true, 3))
        t = CoxProxSSLassoTarget(data, SpikeSlabLassoPrior(spike_scale=.04, slab_scale=.5), gamma, Ref(.5);
            constraint=:prox, prox_sum0_lambda=1e-3, prox_l1_lambda=1e-4,
            alpha_prior_shape=1., alpha_prior_scale=3., prox_root_tol=1e-10, prox_root_maxiter=100)
        @test t.gamma === gamma
        u = [.4,-.2,-.2,log(1.)]
        before_lp = logdensity(t, u)
        before_gradient = gradient!(zeros(4), t, u)
        gamma[1] = false
        @test !t.gamma[1]
        @test logdensity(t, u) != before_lp
        @test gradient!(zeros(4), t, u) != before_gradient
        gamma[1] = true
        @test logdensity(t, u) == before_lp
        exact = BayesCox4Micro.CoxExactZeroTarget(CoxProxSSLassoTarget(data,
            SpikeSlabLassoPrior(spike_scale=.04, slab_scale=.5), gamma, Ref(.5);
            constraint=:none, prox_sum0_lambda=1e-3, prox_l1_lambda=1e-4,
            alpha_prior_shape=1., alpha_prior_scale=3., prox_root_tol=1e-10, prox_root_maxiter=100))
        @test exact.base.gamma === gamma
        joint = BayesCox4Micro.CoxClinicalTarget(t, zeros(4, 2))
        @test joint.base.gamma === gamma
    end
end

@testset "Saved SSLASSO density agrees with saved indicators" begin
    X=[.2 .3 .5; .5 .1 .4; .1 .6 .3; .4 .4 .2; .3 .5 .2; .2 .6 .2]
    time=collect(1.:6.); event=[1,1,0,1,1,0]
    prior=SpikeSlabLassoPrior(spike_scale=.04, slab_scale=.5, proximal=true)
    for sampler in (:hmc, :nuts_explicit), constraint in (:prox, :exact)
        fit=sample_bayes_cox(X,time,event; prior, sampler, constraint, seed=4,
            n_samples=16, warmup=8, chains=1, hmc_n_leapfrog=3, progress=false,
            prox_l1_lambda=1e-4, prox_sum0_lambda=1e-3, alpha_prior_shape=1.,alpha_prior_scale=3.)
        sample=fit.samples
        gamma=fill(true,3); theta=Ref(.5)
        t=CoxProxSSLassoTarget(fit.data,prior,gamma,theta;
            constraint=constraint==:exact ? :none : :prox, prox_sum0_lambda=1e-3,prox_l1_lambda=1e-4,
            alpha_prior_shape=1.,alpha_prior_scale=3.,prox_root_tol=1e-10,prox_root_maxiter=100)
        @test any(sample.draws[:,5:7] .== 0)
        for i in axes(sample.draws,1)
            gamma .= Bool.(sample.draws[i,5:7]); theta[]=sample.draws[i,8]
            @test logdensity(t,sample.draws[i,1:4]) ≈ sample.logposterior[i] atol=1e-9
        end
    end
end
