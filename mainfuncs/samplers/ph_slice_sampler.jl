# Exact-invariant prior-only kernels, avoiding the sharp Moreau boundary's HMC
# step-size restriction. Gaussian references are proposal devices, not priors.
# Murray et al. (2010): https://proceedings.mlr.press/v9/murray10a.html
# Neal (2003), stepping out and shrinkage: https://doi.org/10.1214/aos/1056562461
function _elliptical_slice(rng,current,logratio;max_shrinks=10000)
    initial=logratio(current);isfinite(initial) || error("nonfinite current slice density")
    cutoff=initial+log(rand(rng));aux=randn(rng,length(current))
    angle=2pi*rand(rng);lo,hi=angle-2pi,angle
    for _ in 1:max_shrinks
        candidate=current.*cos(angle).+aux.*sin(angle)
        logratio(candidate)>cutoff && return candidate
        angle<0 ? (lo=angle) : (hi=angle)
        angle=lo+(hi-lo)*rand(rng)
    end
    error("elliptical slice bracket did not resolve")
end
function _stepping_slice(rng,current,logtarget;width=1.,max_steps=100,max_shrinks=10000)
    width>0 && max_steps>0 || throw(ArgumentError("invalid slice controls"))
    initial=logtarget(current);isfinite(initial) || error("nonfinite current slice density")
    cutoff=initial+log(rand(rng));lo=current-width*rand(rng);hi=lo+width
    left=rand(rng,0:max_steps-1);right=max_steps-1-left
    while left>0 && logtarget(lo)>cutoff;lo-=width;left-=1;end
    while right>0 && logtarget(hi)>cutoff;hi+=width;right-=1;end
    for _ in 1:max_shrinks
        candidate=lo+(hi-lo)*rand(rng)
        logtarget(candidate)>cutoff && return candidate
        candidate<current ? (lo=candidate) : (hi=candidate)
    end
    error("scalar slice bracket did not resolve")
end

function _ph_indicator_gibbs!(rng,model,beta,alpha,gamma,theta,weights)
    spike,slab=inv(model.prior.spike_scale),inv(model.prior.slab_scale)
    for j in eachindex(gamma)
        weights[j]=spike
        h0=weighted_l1_moreau(beta,alpha,weights,get(model.kwargs,:prox_l1_lambda,1e-3);
            prox_root_tol=1e-10,prox_root_maxiter=100)
        weights[j]=slab
        h1=weighted_l1_moreau(beta,alpha,weights,get(model.kwargs,:prox_l1_lambda,1e-3);
            prox_root_tol=1e-10,prox_root_maxiter=100)
        probability=_logistic(log(theta[])-log1p(-theta[])+log(slab/spike)-(h1-h0))
        gamma[j]=rand(rng)<probability;weights[j]=gamma[j] ? slab : spike
    end
    theta[]=clamp(rand(rng,Beta(model.prior.theta_a+count(gamma),
        model.prior.theta_b+length(gamma)-count(gamma))),eps(Float64),1-eps(Float64))
end

# Here gamma changes both beta(u,gamma) and the Jacobian. Updating gamma at
# fixed radius coordinates must therefore evaluate this full joint density.
function _ph_radius_indicator_gibbs!(rng,target,u,gamma,theta)
    micro=target.base.base isa CoxClinicalTarget ? target.base.base.base : target.base.base
    for j in eachindex(gamma)
        gamma[j]=false;lp0=logdensity(target,u)
        gamma[j]=true;lp1=logdensity(target,u)
        gamma[j]=rand(rng)<_logistic(lp1-lp0)
    end
    theta[]=clamp(rand(rng,Beta(micro.prior.theta_a+count(gamma),
        micro.prior.theta_b+length(gamma)-count(gamma))),eps(Float64),1-eps(Float64))
end

"""Draw the actual SSLASSO prior using slice updates in radius coordinates.
An elliptical update targets the coefficient block after dividing its density
by the auxiliary N(0,I) reference; scalar stepping-out updates log(alpha).
Indicators use their full transformed conditional, and theta its Beta conditional.
Independent clinical
and time coefficients are drawn directly from their normalized priors.
"""
function sample_ph_sslasso_prior(model;n_samples=10000,warmup=4000,chains=4,seed=1)
    model.method=="prox-sslasso" || throw(ArgumentError("SSLASSO prior required"))
    results=Vector{Any}(undef,chains)
    function one_chain(chain)
        rng=MersenneTwister(seed+100_003chain);gamma=rand(rng,Bool,model.p);theta=Ref(.5)
        base=_ph_target(model,gamma,theta;likelihood_weight=0.)
        target=CoxRadiusPriorTarget(base)
        row=rand(rng,axes(model.values,1));gamma.=model.saved_indicators[row,:]
        initial=vcat(model.values[row,:],zeros(size(base.slope_design,2)))
        u=_radius_internal(target,initial);p=model.p
        draws=zeros(n_samples,length(u));bits=falses(n_samples,p)
        function coefficient_ratio(z)
            candidate=copy(u);candidate[1:p].=z
            return logdensity(target,candidate)+dot(z,z)/2
        end
        function radius_density(eta)
            candidate=copy(u);candidate[p+1]=eta
            return logdensity(target,candidate)
        end
        for iteration in 1:warmup+n_samples
            u[1:p].=_elliptical_slice(rng,view(u,1:p),coefficient_ratio)
            u[p+1]=_stepping_slice(rng,u[p+1],radius_density)
            for j in 1:model.c;u[p+1+j]=2.5rand(rng,TDist(3));end
            q=size(base.slope_design,2)
            q>0 && (u[end-q+1:end].=model.time_prior_scale.*randn(rng,q))
            _ph_radius_indicator_gibbs!(rng,target,u,gamma,theta)
            full=sample_state(target,u)
            if iteration>warmup
                draws[iteration-warmup,:].=full;bits[iteration-warmup,:].=gamma
            end
        end
        return (;draws,indicators=bits)
    end
    if chains>1 && Threads.nthreads()>1
        Threads.@threads for c in 1:chains;results[c]=one_chain(c);end
    else
        for c in 1:chains;results[c]=one_chain(c);end
    end
    return (draws=reduce(vcat,[r.draws for r in results]),
        indicators=reduce(vcat,[r.indicators for r in results]),
        chain_labels=repeat(1:chains;inner=n_samples),transitions=DataFrame(),
        sampling_method="elliptical coefficient slice, scalar log-radius slice and exact indicator Gibbs")
end
