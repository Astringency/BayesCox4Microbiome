using Test, BayesCox4Micro, Random, Distributions, Statistics, LinearAlgebra
@testset "Prior slice kernels recover nontrivial normalized targets" begin
    B=BayesCox4Micro;rng=MersenneTwister(8541)
    mu=[.4,-.6];C=[.5 .15;.15 1.2];target=MvNormal(mu,C)
    ratio=z->logpdf(target,z)+dot(z,z)/2
    z=zeros(2);draws=zeros(40000,2)
    for i in 1:41000
        z=B._elliptical_slice(rng,z,ratio)
        i>1000 && (draws[i-1000,:].=z)
    end
    @test vec(mean(draws;dims=1))≈mu atol=.04
    @test Matrix(cov(draws))≈C rtol=.05
    # Jacobian included: exp(eta) follows Gamma(shape=3, scale=1/2).
    eta=0.;values=zeros(40000)
    for i in 1:41000
        eta=B._stepping_slice(rng,eta,x->3x-2exp(x))
        i>1000 && (values[i-1000]=exp(eta))
    end
    @test mean(values)≈1.5 atol=.025
    @test var(values)≈.75 atol=.04
end
