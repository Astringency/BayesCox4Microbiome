# Annealed SMC normalizing-constant ratios (Del Moral, Doucet & Jasra, 2006).
# https://doi.org/10.1111/j.1467-9868.2006.00553.x
# Each stage resamples before reversible Metropolis mutation; estimates and
# population diagnostics are retained. Independent repetitions measure Monte Carlo variation.
using Random, Statistics, LinearAlgebra, Distributions

function smc_logmeanexp(x)
    m = maximum(x)
    isfinite(m) || error("SMC has no finite particle weights")
    return m + log(mean(exp.(x .- m)))
end

function smc_weights(logweights)
    w = exp.(logweights .- maximum(logweights))
    w ./= sum(w)
    return w
end

function smc_systematic(rng, w)
    n=length(w); positions=(rand(rng) .+ (0:n-1))./n
    cumulative=cumsum(w); cumulative[end]=1.0
    return [searchsortedfirst(cumulative,x) for x in positions]
end

function smc_covariance(x)
    d=size(x,2)
    C=Matrix(cov(x))
    # Small relative ridge for populations with duplicate ancestors.
    ridge=1e-6 .* max.(diag(C),1e-8)
    return cholesky(Symmetric(C+Diagonal(ridge))).L .* (2.38/sqrt(d))
end

"""Anneal equal-weight particles from logbase to logtarget.

The discrete state is a vector of Bernoulli indicators. Symmetric bit flips
preserve the exact intermediate target. All proposals are corrected by MH.
"""
function smc_anneal(rng, particles, indicators, logbase, logtarget;
    ess_fraction=0.8, moves=12, max_stages=2000, phase="posterior")
    n,d=size(particles)
    n>=4 && d>=1 || error("SMC needs at least four particles and one parameter")
    0<ess_fraction<1 && moves>0 || error("invalid SMC controls")
    x=copy(particles); bits=copy(indicators)
    a=[logbase(view(x,i,:),view(bits,i,:)) for i in 1:n]
    b=[logtarget(view(x,i,:),view(bits,i,:)) for i in 1:n]
    all(isfinite,a) && all(isfinite,b) || error("nonfinite initial SMC density")
    temperature=0.0; logZ=0.0; trace=NamedTuple[]
    for stage in 1:max_stages
        delta=b.-a
        effective(step)=inv(sum(abs2,smc_weights(step.*delta)))
        remaining=1-temperature
        step=remaining
        if effective(step)<ess_fraction*n
            lo,hi=0.0,remaining
            for _ in 1:50
                mid=(lo+hi)/2
                if effective(mid)<ess_fraction*n; hi=mid; else; lo=mid; end
            end
            step=lo
        end
        step>eps(Float64) || error("SMC temperature stalled; increase particles or improve reference")
        log_increment=smc_logmeanexp(step.*delta)
        logZ+=log_increment
        weights=smc_weights(step.*delta)
        ess=inv(sum(abs2,weights)); ancestors=smc_systematic(rng,weights)
        # Freeze the proposal within this stage's mutation sweep.
        L=smc_covariance(x)
        x=x[ancestors,:];bits=bits[ancestors,:];a=a[ancestors];b=b[ancestors]
        temperature=min(1.0,temperature+step)
        accepted=0;attempted=0; bit_accepted=0
        for _ in 1:moves, i in 1:n
            proposal=view(x,i,:) .+ L*randn(rng,d)
            pa=logbase(proposal,view(bits,i,:));pb=logtarget(proposal,view(bits,i,:))
            ratio=(1-temperature)*(pa-a[i])+temperature*(pb-b[i])
            attempted+=1
            if isfinite(pa) && isfinite(pb) && log(rand(rng))<ratio
                x[i,:]=proposal;a[i]=pa;b[i]=pb;accepted+=1
            end
            if size(bits,2)>0
                j=rand(rng,axes(bits,2)); bits[i,j]=!bits[i,j]
                pa=logbase(view(x,i,:),view(bits,i,:));pb=logtarget(view(x,i,:),view(bits,i,:))
                ratio=(1-temperature)*(pa-a[i])+temperature*(pb-b[i])
                if isfinite(pa) && isfinite(pb) && log(rand(rng))<ratio
                    a[i]=pa;b[i]=pb;bit_accepted+=1
                else
                    bits[i,j]=!bits[i,j]
                end
            end
        end
        push!(trace,(phase=phase,stage=stage,temperature=temperature,ess=ess,
            unique_ancestors=length(unique(ancestors)),acceptance=accepted/attempted,
            indicator_acceptance=size(bits,2)==0 ? 0.0 : bit_accepted/attempted,
            log_increment=log_increment,log_normalizer=logZ))
        temperature>=1.0 && return (;particles=x,indicators=bits,logZ,trace)
    end
    error("SMC did not reach temperature 1 in $max_stages stages")
end

"""Normalize the actual prior, then integrate the likelihood under that prior.

This separate prior phase is essential: soft-center/prox penalties and Moreau
priors carry method-dependent normalizing constants. Dropping them gives invalid
cross-model Bayes factors. The returned log evidence is the SECOND phase ratio.
"""
function smc_evidence(logprior, loglikelihood, reference; nbits=0, particles=2048,
    seed=1, ess_fraction=0.8, moves=12, max_stages=2000)
    rng=MersenneTwister(seed)
    x=permutedims(rand(rng,reference,particles))
    bits=rand(rng,Bool,particles,nbits)
    logq(u,g)=logpdf(reference,u)-nbits*log(2)
    prior=smc_anneal(rng,x,bits,logq,logprior;ess_fraction,moves,max_stages,phase="prior")
    posterior=smc_anneal(rng,prior.particles,prior.indicators,logprior,
        (u,g)->logprior(u,g)+loglikelihood(u,g);ess_fraction,moves,max_stages,phase="posterior")
    return (;posterior...,prior_logZ=prior.logZ,trace=vcat(prior.trace,posterior.trace))
end
