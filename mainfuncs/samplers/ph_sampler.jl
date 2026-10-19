function _ph_target(model, gamma, theta; likelihood_weight=1.0)
    if model.method=="cox-rhs"
        base=_make_explicit_nuts_target(model.data,model.prior;constraint=model.constraint,
            prox_sum0_lambda=model.sumlambda,noncentered=true,use_car=false)
    else
        base=CoxProxSSLassoTarget(model.data,model.prior,gamma,theta;
            constraint=model.constraint,prox_sum0_lambda=model.sumlambda,
            prox_l1_lambda=get(model.kwargs,:prox_l1_lambda,1e-3),
            alpha_prior_shape=get(model.kwargs,:alpha_prior_shape,1.),
            alpha_prior_scale=model.alpha_scale,alpha_prior_kind=:invgamma,
            prox_root_tol=1e-10,prox_root_maxiter=100,init=:zero)
    end
    model.c>0 && (base=CoxClinicalTarget(base,model.clinical_X;
        names=model.clinical_names,prior_df=3.,prior_scale=2.5))
    return CoxTimeVaryingTarget(base,model.time_design,ph_time_design(model.p,model.c,model.scope);
        slope_scale=model.time_prior_scale,likelihood_weight)
end

"""Draw from a frozen PH model's posterior or prior-only target.
Posteriors use HMC; prior-only draws use exact-invariant slice/Gibbs updates
for SSLASSO or independent rejection sampling for Cox-RHS.
The SSLASSO sampler retains exact gamma/theta Gibbs updates; evidence later
integrates theta analytically. Each chain owns every mutable target/workspace.
"""
function sample_ph_model(model; n_samples=10000,warmup=4000,chains=4,
    target_accept=0.95,hmc_n_leapfrog=300,seed=1,likelihood_weight=1.0,prior_sampler=:slice)
    n_samples>=32 && warmup>=0 && chains>=1 || throw(ArgumentError("invalid PH sampling budget"))
    model.method=="cox-rhs" && likelihood_weight==0 &&
        return sample_ph_rhs_prior(model;n_samples,chains,seed)
    prior_sampler in (:slice,:hmc) || throw(ArgumentError("invalid prior sampler"))
    model.method=="prox-sslasso" && likelihood_weight==0 && prior_sampler==:slice &&
        return sample_ph_sslasso_prior(model;n_samples,warmup,chains,seed)
    results=Vector{Any}(undef,chains)
    function one_chain(c)
        rng=MersenneTwister(seed+100_003c)
        gamma=rand(rng,Bool,model.nbits);theta=Ref(0.5)
        target=_ph_target(model,gamma,theta;likelihood_weight)
        row=rand(rng,axes(model.values,1))
        initial=vcat(model.values[row,:],zeros(size(target.slope_design,2)))
        micro=target.base isa CoxClinicalTarget ? target.base.base : target.base
        if micro isa CoxNoncenteredPosteriorTarget
            logs=_exact_logscales(micro,view(initial,1:dimension(micro)))
            initial[1:model.p]./=exp.(logs)
        else
            gamma.=model.saved_indicators[row,:]
        end
        if model.method=="prox-sslasso" && likelihood_weight==0
            target=CoxRadiusPriorTarget(target)
            initial=_radius_internal(target,initial)
        end
        if model.nbits==0
            result=_sample_explicit_hmc_one(target;n_samples,warmup,thin=1,rng,target_accept,
                hmc_n_leapfrog,progress=false,initial)
            return (draws=result.draws,indicators=falses(n_samples,0),transitions=result.transitions)
        end
        u=initial
        state=ProxSSLassoBlockHMCState(length(u);step_size=1e-3,n_leapfrog=hmc_n_leapfrog,
            inv_metric=:unit_diag,target_accept,adapt_every=50,min_step_size=1e-8,
            max_step_size=1.0,divergence_threshold=1000.,frozen=warmup==0,
            n_adapts=warmup,auto_step_size=true)
        draws=zeros(n_samples,length(sample_state_names(target)))
        indicators=falses(n_samples,model.nbits);transitions=NamedTuple[]
        p=model.p;weights=_ssl_weights(model.prior,gamma)
        for iteration in 1:warmup+n_samples
            transition=prox_ssl_block_hmc_step!(u,target,state;rng)
            push!(transitions,_transition_record(transition,iteration,warmup))
            iteration==warmup && _prox_ssl_block_freeze!(state)
            full=sample_state(target,u);beta=view(full,1:p);alpha=exp(clamp(full[p+1],-700.,700.))
            if target isa CoxRadiusPriorTarget
                _ph_radius_indicator_gibbs!(rng,target,u,gamma,theta)
                full=sample_state(target,u)
            else
                _ph_indicator_gibbs!(rng,model,beta,alpha,gamma,theta,weights)
            end
            if iteration>warmup
                draws[iteration-warmup,:].=full
                indicators[iteration-warmup,:].=gamma
            end
        end
        return (;draws,indicators,transitions)
    end
    if chains>1 && Threads.nthreads()>1
        Threads.@threads for c in 1:chains;results[c]=one_chain(c);end
    else
        for c in 1:chains;results[c]=one_chain(c);end
    end
    return (draws=reduce(vcat,[r.draws for r in results]),
        indicators=reduce(vcat,[r.indicators for r in results]),
        chain_labels=repeat(1:chains;inner=n_samples),
        transitions=DataFrame([merge((chain=c,),row) for c in 1:chains for row in results[c].transitions]))
end
