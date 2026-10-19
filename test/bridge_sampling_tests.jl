using Test, BayesCox4Micro, Random, Distributions, Statistics, LinearAlgebra
@testset "Bridge evidence with normalized priors and discrete states" begin
    rng=MersenneTwister(4201);n=16000; labels=repeat(1:4;inner=n÷4)
    xprior=reshape(randn(rng,n),:,1)
    # y=0.7, sigma=0.8, prior N(0,1) has conjugate normal posterior.
    variance=inv(1+inv(.8^2)); mu=variance*.7/.8^2
    xpost=reshape(mu .+ sqrt(variance).*randn(rng,n),:,1)
    prior=(draws=xprior,indicators=falses(n,0),chain_labels=labels)
    post=(draws=xpost,indicators=falses(n,0),chain_labels=labels)
    lp(u,g)=logpdf(Normal(),u[1])+7.
    ll(u,g)=logpdf(Normal(u[1],.8),.7)
    result=bridge_evidence(lp,ll,post,prior;seed=51)
    @test result.logZ≈logpdf(Normal(0,sqrt(1+.8^2)),.7) atol=.015
    @test result.prior_logZ≈7. atol=.015
    shifted=bridge_evidence((u,g)->lp(u,g)+11.,ll,post,prior;seed=51)
    @test shifted.logZ≈result.logZ atol=1e-8
    @test result.numerical_screen
    priorbits=rand(rng,n,1).<.25
    postbits=rand(rng,n,1).<(.25*.8/(.25*.8+.75*.2))
    # Continuous draws for this model must have their unchanged N(0,1) law.
    discrete=bridge_evidence((u,g)->logpdf(Normal(),u[1])+log(g[1] ? .25 : .75),
        (u,g)->log(g[1] ? .8 : .2),
        (draws=reshape(randn(rng,n),:,1),indicators=postbits,chain_labels=labels),
        (draws=xprior,indicators=priorbits,chain_labels=labels);seed=81)
    @test exp(discrete.logZ)≈.25*.8+.75*.2 atol=.01
    @test_throws ArgumentError bridge_sampling(lp,fill(NaN,100,1))
end

@testset "RHS constrained Gaussian conditional" begin
    B=BayesCox4Micro;rng=MersenneTwister(88410)
    s=[.5,1.2,.8];a=.02
    draws=reduce(hcat,[B._rhs_constrained_normal(rng,s,a) for _ in 1:20000])'
    expected=Diagonal(s.^2)-(s.^2)*(s.^2)'/(a+sum(abs2,s))
    @test maximum(abs,vec(mean(draws;dims=1)))<.02
    @test Matrix(cov(draws))≈Matrix(expected) rtol=.04
    @test var(vec(sum(draws;dims=2)))≈a*sum(abs2,s)/(a+sum(abs2,s)) rtol=.04
end
