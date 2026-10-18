#!/usr/bin/env julia
# Full-data PH models and cross-model partial-likelihood Bayes factors.
using BayesCox4Micro, CSV, DataFrames, JSON, SHA, LinearAlgebra, Statistics, Distributions
include(joinpath(@__DIR__,"smc.jl"))
isdefined(@__MODULE__,:parse_cli_args) || include(joinpath(@__DIR__,"..","analysis_common.jl"))

"""Cox partial likelihood for beta(t)=beta + delta*g(t).

All subjects in a risk set use the SAME event time g(t), never their own observed
follow-up time. The Efron denominator includes censored subjects tied at that time.
"""
function time_cox_loglikelihood(X, time, event, beta, delta; tie_method=:efron,
    center=mean(log.(time[event .!= 0])), scale=std(log.(time[event .!= 0])))
    scale>0 && all(>(0),time) || throw(ArgumentError("PH time basis needs positive, distinct event times"))
    tie_method in (:efron,:breslow) || throw(ArgumentError("unsupported ties"))
    lp=0.0
    static=X*beta; slope=X*delta
    for t in sort(unique(time[event .!= 0]))
        risk=findall(>=(t),time); deaths=findall((time.==t).&(event.!=0))
        eta=static .+ slope.*((log(t)-center)/scale)
        shift=maximum(eta[risk]); isfinite(shift) || return -Inf
        s0=sum(exp.(eta[risk].-shift));e0=sum(exp.(eta[deaths].-shift));d=length(deaths)
        lp+=sum(eta[deaths].-shift)
        for l in 0:d-1
            lp-=log(s0-(tie_method==:efron ? l/d*e0 : 0.0))
        end
    end
    return lp
end

function ph_time_design(p,c,scope)
    scope in ("static","compositional","noncompositional","both") || error("unknown time scope")
    scope in ("noncompositional","both") && c==0 && error("time scope requires clinical covariates")
    # An orthonormal contrast basis gives an explicitly normalized, exact zero-sum
    # prior for compositional time slopes. Clinical time slopes remain unrestricted.
    B=nullspace(ones(1,p))
    micro=scope in ("compositional","both") ? vcat(B,zeros(c,p-1)) : zeros(p+c,0)
    clinical=scope in ("noncompositional","both") ? vcat(zeros(p,c),Matrix{Float64}(I,c,c)) : zeros(p+c,0)
    return hcat(micro,clinical)
end

function ph_model(arm, scope; time_prior_scale=0.5)
    time_prior_scale>0 || error("time prior scale must be positive")
    spec=JSON.parsefile(joinpath(arm,"analysis_specification.json"))
    selected=JSON.parsefile(joinpath(arm,"full_data_tuning","selected_config.json"))
    meta=JSON.parsefile(joinpath(arm,"runs","realdata_"*spec["method_name"],"metadata.json"))
    design=CSV.read(joinpath(arm,"ph_design.csv"),DataFrame)
    p=length(spec["component_cols"]); X=Matrix{Float64}(design[:,4:end]);c=size(X,2)-p
    time=Float64.(design.time);event=Int.(design.status)
    data=prepare_cox_data(exp.(X[:,1:p]),time,event;closure=false,standardize=false)
    method=spec["method_name"];constraint=Symbol(spec["constraint"])
    kwargs=Dict{Symbol,Any}(Symbol(k)=>v for (k,v) in selected if v!==nothing)
    # Build the identical static prior selected by full-data tuning, without CV.
    raw=exp.(X[:,1:p])
    prior=prior_from_name(method,raw,time,event;p0=Float64(meta["p0"]),realdata=true,
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
    loglikelihood(u,g)=time_cox_loglikelihood(X,time,event,coefficients(u),slopes(u);center,scale)
    saved=CSV.read(joinpath(arm,"runs","realdata_"*method,"samples_wide.csv"),DataFrame)
    values=Matrix{Float64}(saved[:,vcat(base_names,clinical_names)])
    size(values,1)>1 || error("PH evidence needs at least two saved posterior draws")
    # The reference only initializes integration. Its normalized density is
    # divided out, so it is not substituted for the selected static prior.
    reference_mean=vcat(vec(mean(values;dims=1)),zeros(q))
    C=zeros(d+c+q,d+c+q)
    observed=Matrix(cov(values))
    C[1:d+c,1:d+c]=4 .* observed+Diagonal(max.(diag(observed).*1e-4,1e-6))
    q>0 && (C[d+c+1:end,d+c+1:end]=time_prior_scale^2 .* Matrix{Float64}(I,q,q))
    reference=MvNormal(reference_mean,Symmetric(C))
    return (;logprior,loglikelihood,reference,nbits,coefficients,slopes,names_internal,
        features=names(design)[4:end],center,scale,p,c,method,constraint)
end

function ph_fingerprint(arm,settings)
    files=[joinpath(arm,path) for path in ("analysis_specification.json","ph_design.csv",
        "full_data_tuning/selected_config.json")]
    spec=JSON.parsefile(first(files))
    append!(files,[joinpath(arm,"runs","realdata_"*spec["method_name"],f) for f in ("metadata.json","samples_wide.csv")])
    root=normpath(joinpath(@__DIR__,"..","..",".."))
    append!(files,[joinpath(d,f) for folder in ("mainfuncs","priors","scripts/lib/diagnostics")
        for (d,_,fs) in walkdir(joinpath(root,folder)) for f in fs if endswith(f,".jl")])
    return bytes2hex(sha256(join([bytes2hex(sha256(read(f))) for f in sort(files)],"\n")*JSON.json(settings)))
end

function run_ph_models(arm; particles=2048,repetitions=4,moves=12,ess_fraction=0.8,
    max_stages=2000,time_prior_scale=0.5,seed=1)
    repetitions>=2 || error("PH evidence needs independent repetitions to report Monte Carlo variation")
    settings=(;particles,repetitions,moves,ess_fraction,max_stages,time_prior_scale,seed)
    fingerprint=ph_fingerprint(arm,settings)
    out=ensure_dir(joinpath(arm,"ph_diagnostics"))
    spec=JSON.parsefile(joinpath(arm,"analysis_specification.json"))
    scopes=spec["adjustment"]=="adjusted" ? ["static","compositional","noncompositional","both"] : ["static","compositional"]
    evidence=NamedTuple[]
    for (si,scope) in enumerate(scopes)
        directory=ensure_dir(joinpath(out,scope))
        receipt=joinpath(directory,"complete.json")
        if isfile(receipt)
            record=JSON.parsefile(receipt)
            record["fingerprint"]==fingerprint || error("PH inputs/settings changed; use a new output root")
            for (file,digest) in record["files"]
                isfile(joinpath(directory,file)) && bytes2hex(sha256(read(joinpath(directory,file))))==digest ||
                    error("PH output missing or changed: $(joinpath(directory,file))")
            end
            append!(evidence,[NamedTuple(r) for r in eachrow(CSV.read(joinpath(directory,"evidence.csv"),DataFrame))])
            continue
        end
        model=ph_model(arm,scope;time_prior_scale)
        model_seed=parse(Int,bytes2hex(sha256(fingerprint*scope))[1:8];base=16) % 1_000_000_000
        rows=NamedTuple[];draws=DataFrame[];traces=DataFrame[]
        for rep in 1:repetitions
            println("PH $scope repetition $rep/$repetitions: $arm");flush(stdout)
            result=smc_evidence(model.logprior,model.loglikelihood,model.reference;
                nbits=model.nbits,particles,seed=seed+model_seed+rep,ess_fraction,moves,max_stages)
            b=reduce(vcat,[permutedims(model.coefficients(u)) for u in eachrow(result.particles)])
            slopes=reduce(vcat,[permutedims(model.slopes(u)) for u in eachrow(result.particles)])
            tab=DataFrame(hcat(b,slopes),vcat(model.features,"time[" .* model.features .* "]"))
            insertcols!(tab,1,:repetition=>fill(rep,particles),:particle=>collect(1:particles))
            push!(draws,tab)
            trace=DataFrame(result.trace);insertcols!(trace,1,:repetition=>fill(rep,nrow(trace)));push!(traces,trace)
            push!(rows,(scope=scope,repetition=rep,log_evidence=result.logZ,
                prior_log_normalizer=result.prior_logZ,stages=nrow(trace),
                minimum_ess=minimum(trace.ess),mean_acceptance=mean(trace.acceptance),
                time_center=model.center,time_scale=model.scale,time_prior_scale=time_prior_scale))
        end
        CSV.write(joinpath(directory,"posterior_particles.csv"),vcat(draws...))
        CSV.write(joinpath(directory,"smc_diagnostics.csv"),vcat(traces...))
        CSV.write(joinpath(directory,"evidence.csv"),DataFrame(rows))
        all_draws=Matrix{Float64}(vcat(draws...)[:,3:end])
        CSV.write(joinpath(directory,"coefficients.csv"),posterior_summary(all_draws;names=names(first(draws))[3:end]))
        BayesCox4Micro._write_tuning_json(receipt,Dict("fingerprint"=>fingerprint,"settings"=>settings,
            "scope"=>scope,"status"=>"complete","fit"=>"full-data annealed SMC; no time-model CV",
            "files"=>Dict(file=>bytes2hex(sha256(read(joinpath(directory,file)))) for file in
                ("posterior_particles.csv","smc_diagnostics.csv","evidence.csv","coefficients.csv"))))
        append!(evidence,rows)
    end
    CSV.write(joinpath(out,"evidence_repetitions.csv"),DataFrame(evidence))
    spec["adjustment"]=="adjusted" || BayesCox4Micro._write_tuning_json(joinpath(out,"not_applicable.json"),
        Dict("noncompositional"=>"No noncompositional covariates in this static model",
             "both"=>"Identical to compositional when clinical covariates are absent"))
    return out
end

function aggregate_ph_evidence(root)
    models=NamedTuple[]; outcome_digest=nothing
    for (dir,_,files) in walkdir(root)
        "evidence_repetitions.csv" in files || continue
        arm=dirname(dir);spec=JSON.parsefile(joinpath(arm,"analysis_specification.json"))
        design=CSV.read(joinpath(arm,"ph_design.csv"),DataFrame)
        digest=bytes2hex(sha256(JSON.json([collect(design.time),collect(design.status)])))
        outcome_digest===nothing && (outcome_digest=digest)
        digest==outcome_digest || error("Cannot compare evidence from different observations/risk sets")
        rows=CSV.read(joinpath(dir,"evidence_repetitions.csv"),DataFrame)
        for group in groupby(rows,:scope)
            values=group.log_evidence
            logZ=smc_logmeanexp(values)
            weights=exp.(values.-maximum(values))
            se=std(weights)/sqrt(length(weights))/mean(weights)
            id=join((spec["constraint"],spec["adjustment"],spec["method_name"],first(group.scope)),"/")
            push!(models,(model=id,arm=abspath(arm),scope=first(group.scope),log_evidence=logZ,
                log_evidence_mcse=se,log_evidence_rep_sd=std(values),repetitions=length(values)))
        end
    end
    isempty(models) && error("No completed PH evidence under $root")
    length(unique(m.model for m in models))==length(models) ||
        error("Duplicate model arms: aggregate one experiment root at a time")
    comparisons=NamedTuple[]
    for a in models,b in models
        a.model==b.model && continue
        logBF=a.log_evidence-b.log_evidence
        push!(comparisons,(numerator=a.model,denominator=b.model,log_bayes_factor=logBF,
            bayes_factor=exp(logBF),log_bf_mcse=hypot(a.log_evidence_mcse,b.log_evidence_mcse),
            comparison=a.scope=="static" && b.scope=="static" ? "static_vs_static" :
                a.scope!="static" && b.scope!="static" ? "time_vs_time" : "time_vs_static"))
    end
    out=ensure_dir(joinpath(root,"ph_comparison"))
    CSV.write(joinpath(out,"model_evidence.csv"),DataFrame(models))
    CSV.write(joinpath(out,"bayes_factors.csv"),DataFrame(comparisons))
    write(joinpath(out,"method.txt"),"""
    Bayesian evidence uses the Cox Efron partial likelihood and normalized method-specific priors.
    Hyperparameters are fixed at the static model's full-data CV selection (empirical Bayes).
    Time effects use centered/scaled log event time, applied to every subject in each risk set.
    Compositional time slopes use an orthonormal zero-sum contrast; clinical slopes are unrestricted.
    The slope prior scale is fixed, never selected with additional cross-validation.
    Separate prior normalization accounts for soft-center/prox/Moreau normalizing constants.
    SMC independent repetitions estimate evidence uncertainty; inspect smc_diagnostics.csv and
    increase particles/moves/repetitions when mixing or repeated estimates are unstable.
    These are partial-likelihood Bayes factors, not full survival-likelihood Bayes factors.
    All ordered static/time/model/constraint/adjustment comparisons use the same observed subjects.
    SMC reference: https://doi.org/10.1111/j.1467-9868.2006.00553.x
    """)
    return out
end

function ph_main(args=parse_cli_args())
    root=arg(args,"root","")
    if arg(args,"aggregate-only",false)
        isempty(root) && error("--root required")
        return aggregate_ph_evidence(root)
    end
    arm=abspath(arg(args,"source-arm",""))
    isfile(joinpath(arm,"final_complete.json")) || error("PH requires a completed static final fit")
    completed=JSON.parsefile(joinpath(arm,"final_complete.json"))
    for (file,digest) in completed["files"]
        isfile(joinpath(arm,file)) && bytes2hex(sha256(read(joinpath(arm,file))))==digest ||
            error("Static final output missing or changed: $file")
    end
    out=run_ph_models(arm;particles=arg(args,"particles",2048),repetitions=arg(args,"repetitions",4),
        moves=arg(args,"moves",12),ess_fraction=arg(args,"ess-fraction",0.8),max_stages=arg(args,"max-stages",2000),
        time_prior_scale=arg(args,"time-prior-scale",0.5),seed=arg(args,"seed",1))
    rscript=get(ENV,"RSCRIPT_BIN","Rscript")
    script=joinpath(@__DIR__,"schoenfeld.R")
    run(`$rscript $script $(joinpath(arm,"ph_design.csv")) $(joinpath(arm,"ph_coefficients.csv")) $out`)
    aggregate_ph_evidence(isempty(root) ? arm : root)
end

abspath(PROGRAM_FILE)==abspath(@__FILE__) && ph_main()
