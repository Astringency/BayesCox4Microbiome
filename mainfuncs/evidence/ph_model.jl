"""Construct PH model densities from frozen static-model inputs. File I/O belongs to the caller."""
function ph_model_components(spec,selected,meta,design,saved,scope;time_prior_scale=0.5)
    time_prior_scale>0 || error("time prior scale must be positive")
    p=length(spec["component_cols"]); X=Matrix{Float64}(design[:,4:end]);c=size(X,2)-p
    time=Float64.(design.time);event=Int.(design.status)
    data=prepare_cox_data(exp.(X[:,1:p]),time,event;closure=false,standardize=false)
    method=spec["method_name"];constraint=Symbol(spec["constraint"])
    kwargs=Dict{Symbol,Any}(Symbol(k)=>v for (k,v) in selected if v!==nothing)
    # Build the identical static prior selected by full-data tuning, without CV.
    raw=exp.(X[:,1:p])
    # p0 belongs to the horseshoe global scale; SSLASSO final-fit metadata
    # correctly stores null because that prior has no p0 parameter.
    frozen_p0=method=="cox-rhs" ? Float64(meta["p0"]) : nothing
    prior=prior_from_name(method,raw,time,event;p0=frozen_p0,realdata=true,
        (k=>v for (k,v) in kwargs if k in (:rhs_local_scale,:rhs_slab_df,:rhs_slab_s2,
          :rhs_tau_scale,:sslasso_spike_scale,:sslasso_slab_scale,:sslasso_theta_a,:sslasso_theta_b))...)
    sumlambda=get(spec,"prox_sum0_lambda",1e-6)
    if method=="cox-rhs"
        base_names=BayesCox4Micro._state_names(prior,p)
        base_prior=(u,g)->BayesCox4Micro._prior_logposterior(u,data,prior)+
            BayesCox4Micro._constraint_logpenalty(view(u,1:p);constraint,prox_sum0_lambda=sumlambda)
        nbits=0
    elseif method=="prox-sslasso"
        base_names=vcat(["beta[$j]" for j in 1:p],["eta"])
        # Integrate theta exactly. BetaBinomial is a count probability; divide
        # by choose(p,k) to obtain the probability of this particular pattern.
        pattern=[logpdf(BetaBinomial(p,prior.theta_a,prior.theta_b),k)-
            sum(log,Float64.(1:p);init=0.0)+sum(log,Float64.(1:k);init=0.0)+
            sum(log,Float64.(1:p-k);init=0.0) for k in 0:p]
        base_prior = function (u,g)
            lp=BayesCox4Micro._ssl_prox_prior_logdensity(u,prior,g,0.5;
                prox_l1_lambda=get(kwargs,:prox_l1_lambda,1e-3),
                alpha_prior_shape=get(kwargs,:alpha_prior_shape,1.0),
                alpha_prior_scale=get(spec,"alpha_prior_scale",p+c),alpha_prior_kind=:invgamma,
                prox_root_tol=1e-10,prox_root_maxiter=100)
            return lp+p*log(2)+pattern[count(g)+1]+
                BayesCox4Micro._constraint_logpenalty(view(u,1:p);constraint,prox_sum0_lambda=sumlambda)
        end
        nbits=p
    else
        error("PH models support Cox-RHS and Prox-SSLASSO")
    end
    d=length(base_names);T=ph_time_design(p,c,scope);q=size(T,2)
    clinical_names=c==0 ? String[] : names(design)[4+p:end]
    names_internal=vcat(base_names,clinical_names,["time_contrast[$j]" for j in 1:q])
    center=mean(log.(time[event.!=0]));scale=std(log.(time[event.!=0]))
    all(>(0),time) && scale>0 || error("invalid log-time basis")
    function logprior(u,g)
        all(isfinite,u) || return -Inf
        lp=base_prior(view(u,1:d),g)
        for j in 1:c
            lp+=logpdf(TDist(3),u[d+j]/2.5)-log(2.5)
        end
        for j in 1:q
            lp+=logpdf(Normal(0,time_prior_scale),u[d+c+j])
        end
        return isnan(lp) ? -Inf : lp
    end
    coefficients(u)=vcat(u[1:p],u[d+1:d+c])
    slopes(u)=T*u[d+c+1:end]
    time_design=TimeCoxDesign(X,time,event;center,scale)
    loglikelihood(u,g)=first(time_cox_loglikelihood_and_score(time_design,coefficients(u),slopes(u)))
    values=Matrix{Float64}(saved[:,vcat(base_names,clinical_names)])
    size(values,1)>1 || error("PH evidence needs at least two saved posterior draws")
    # A convenience reference describes starting geometry; the selected prior
    # remains the model prior. Bridge proposals are fitted to separate draws.
    reference_mean=vcat(vec(mean(values;dims=1)),zeros(q))
    C=zeros(d+c+q,d+c+q)
    observed=Matrix(cov(values))
    C[1:d+c,1:d+c]=4 .* observed+Diagonal(max.(diag(observed).*1e-4,1e-6))
    q>0 && (C[d+c+1:end,d+c+1:end]=time_prior_scale^2 .* Matrix{Float64}(I,q,q))
    reference=MvNormal(reference_mean,Symmetric(C))
    return (;logprior,loglikelihood,reference,nbits,coefficients,slopes,names_internal,
        features=names(design)[4:end],center,scale,p,c,method,constraint,
        data,prior,kwargs,sumlambda,scope,time_prior_scale,
        alpha_scale=get(spec,"alpha_prior_scale",p+c),clinical_X=X[:,p+1:end],
        clinical_names,time_design,values,saved_chains=Int.(saved.chain),
        saved_indicators=nbits==0 ? falses(nrow(saved),0) : Bool.(Matrix(saved[:,["gamma[$j]" for j in 1:p]])))
end
