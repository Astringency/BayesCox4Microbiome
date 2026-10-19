#!/usr/bin/env julia
# Full-data PH models and cross-model partial-likelihood Bayes factors.
using BayesCox4Micro, CSV, DataFrames, JSON, SHA, LinearAlgebra, Statistics, Distributions
isdefined(@__MODULE__,:parse_cli_args) || include(joinpath(@__DIR__,"..","analysis_common.jl"))

function ph_model(arm,scope;time_prior_scale=0.5)
    spec=JSON.parsefile(joinpath(arm,"analysis_specification.json"))
    selected=JSON.parsefile(joinpath(arm,"full_data_tuning","selected_config.json"))
    run=joinpath(arm,"runs","realdata_"*spec["method_name"])
    meta=JSON.parsefile(joinpath(run,"metadata.json"))
    design=CSV.read(joinpath(arm,"ph_design.csv"),DataFrame)
    saved=CSV.read(IOBuffer(read(joinpath(run,"samples_wide.csv"))),DataFrame)
    return BayesCox4Micro.ph_model_components(spec,selected,meta,design,saved,scope;time_prior_scale)
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

function ph_draw_table(model,sample)
    table=DataFrame(sample.draws,model.names_internal)
    for j in 1:model.nbits;table[!,"gamma[$j]"]=Int.(sample.indicators[:,j]);end
    insertcols!(table,1,:chain=>sample.chain_labels,
        :iter=>reduce(vcat,[collect(1:count(==(c),sample.chain_labels)) for c in unique(sample.chain_labels)]))
    return table
end

function run_ph_models(arm; samples=10000,prior_samples=samples,warmup=4000,chains=4,leapfrog=300,
    target_accept=0.95,repetitions=4,time_prior_scale=0.5,seed=1)
    repetitions>=2 || error("PH evidence needs repeated bridge estimates")
    settings=(;method="bridge_sampling_v1",samples,prior_samples,warmup,chains,leapfrog,target_accept,repetitions,time_prior_scale,seed)
    fingerprint=ph_fingerprint(arm,settings)
    out=ensure_dir(joinpath(arm,"ph_diagnostics"))
    spec=JSON.parsefile(joinpath(arm,"analysis_specification.json"))
    scopes=spec["adjustment"]=="adjusted" ? ["static","compositional","noncompositional","both"] : ["static","compositional"]
    prior_model=ph_model(arm,"static";time_prior_scale)
    # Independent slope priors integrate to one. The same static prior normalizer
    # therefore applies to every time scope within an arm, and is computed once.
    prior_dir=ensure_dir(joinpath(out,"prior_normalization"))
    sample_settings=(;warmup,chains,target_accept,hmc_n_leapfrog=leapfrog)
    function sample_or_read(model,directory;prior_only=false,model_seed=seed)
        receipt=joinpath(directory,"sampling_complete.json")
        path=joinpath(directory,"samples_wide.csv")
        if isfile(receipt)
            record=JSON.parsefile(receipt)
            record["fingerprint"]==fingerprint || error("PH settings changed; use a new output root")
            bytes2hex(sha256(read(path)))==record["samples_sha256"] || error("PH draws changed")
            table=CSV.read(path,DataFrame)
            return (draws=Matrix{Float64}(table[:,model.names_internal]),
                indicators=model.nbits==0 ? falses(nrow(table),0) : Bool.(Matrix(table[:,["gamma[$j]" for j in 1:model.nbits]])),
                chain_labels=Int.(table.chain))
        end
        println("PH $(prior_only ? "prior draws" : model.scope*" HMC"): $arm");flush(stdout)
        actual_samples=prior_only ? prior_samples : samples
        result=BayesCox4Micro.sample_ph_model(model;sample_settings...,n_samples=actual_samples,seed=model_seed,
            likelihood_weight=prior_only ? 0.0 : 1.0)
        if hasproperty(result,:prior_rejection_log_normalizer)
            BayesCox4Micro._write_tuning_json(joinpath(directory,"rejection_check.json"),Dict(
                "sampling_method"=>result.sampling_method,
                "log_normalizer"=>result.prior_rejection_log_normalizer,
                "log_normalizer_mcse"=>result.prior_rejection_log_mcse,
                "proposals"=>result.prior_proposals))
        end
        table=ph_draw_table(model,result)
        CSV.write(path,table)
        CSV.write(joinpath(directory,"transition_diagnostics.csv"),result.transitions)
        CSV.write(joinpath(directory,"diagnostics_summary.csv"),diagnostic_table(table;transitions=result.transitions))
        BayesCox4Micro._write_tuning_json(receipt,Dict("fingerprint"=>fingerprint,
            "samples_sha256"=>bytes2hex(sha256(read(path))),"settings"=>settings,"prior_only"=>prior_only,
            "samples_per_chain"=>actual_samples,
            "sampling_method"=>hasproperty(result,:sampling_method) ? result.sampling_method : "HMC with indicator Gibbs where applicable"))
        return result
    end
    prior=sample_or_read(prior_model,prior_dir;prior_only=true,model_seed=seed+100000)
    prior_quality=BayesCox4Micro.continuous_convergence_summary(convergence_diagnostics(ph_draw_table(prior_model,prior)))
    prior_bridges=[BayesCox4Micro.ph_bridge_sampling(prior_model,prior;
        posterior=false,seed=seed+1000+rep) for rep in 1:repetitions]
    CSV.write(joinpath(prior_dir,"bridge_diagnostics.csv"),DataFrame(prior_bridges))
    evidence=NamedTuple[]
    for (si,scope) in enumerate(scopes)
        directory=ensure_dir(joinpath(out,scope))
        model=scope=="static" ? prior_model : ph_model(arm,scope;time_prior_scale)
        posterior=scope=="static" ?
            (draws=model.values,indicators=model.saved_indicators,chain_labels=model.saved_chains) :
            sample_or_read(model,directory;model_seed=seed+10000si)
        posterior_quality=BayesCox4Micro.continuous_convergence_summary(convergence_diagnostics(ph_draw_table(model,posterior)))
        BayesCox4Micro._write_tuning_json(joinpath(directory,"sampling_quality.json"),
            Dict("prior"=>prior_quality,"posterior"=>posterior_quality))
        rows=NamedTuple[];bridge_rows=NamedTuple[]
        for rep in 1:repetitions
            post_bridge=BayesCox4Micro.ph_bridge_sampling(model,posterior;seed=seed+100si+rep)
            prior_bridge=prior_bridges[rep]
            push!(bridge_rows,merge((repetition=rep,),post_bridge))
            push!(rows,(scope=scope,repetition=rep,
                log_evidence=post_bridge.log_normalizer-prior_bridge.log_normalizer,
                log_evidence_mcse=hypot(post_bridge.log_normalizer_mcse,prior_bridge.log_normalizer_mcse),
                prior_log_normalizer=prior_bridge.log_normalizer,
                numerical_screen=post_bridge.numerical_screen && prior_bridge.numerical_screen &&
                    prior_quality.continuous_screen_pass && posterior_quality.continuous_screen_pass,
                time_center=model.center,time_scale=model.scale,time_prior_scale=time_prior_scale))
        end
        b=reduce(vcat,[permutedims(model.coefficients(u)) for u in eachrow(posterior.draws)])
        slopes=reduce(vcat,[permutedims(model.slopes(u)) for u in eachrow(posterior.draws)])
        labels=vcat(model.features,"time[" .* model.features .* "]")
        CSV.write(joinpath(directory,"coefficients.csv"),posterior_summary(hcat(b,slopes);names=labels))
        CSV.write(joinpath(directory,"bridge_diagnostics.csv"),DataFrame(bridge_rows))
        CSV.write(joinpath(directory,"evidence.csv"),DataFrame(rows))
        BayesCox4Micro._write_tuning_json(joinpath(directory,"complete.json"),Dict(
            "fingerprint"=>fingerprint,"settings"=>settings,"scope"=>scope,"status"=>"complete",
            "fit"=>scope=="static" ? "frozen static HMC draws; bridge sampling" : "full-data HMC; bridge sampling; no time-model CV",
            "files"=>Dict(file=>bytes2hex(sha256(read(joinpath(directory,file)))) for file in
                ("bridge_diagnostics.csv","evidence.csv","coefficients.csv","sampling_quality.json"))))
        append!(evidence,rows)
    end
    CSV.write(joinpath(out,"evidence_repetitions.csv"),DataFrame(evidence))
    spec["adjustment"]=="adjusted" || BayesCox4Micro._write_tuning_json(joinpath(out,"not_applicable.json"),
        Dict("noncompositional"=>"No clinical covariates in this static model",
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
            logZ=mean(values)
            # Repetitions reuse MCMC draws, so their errors are correlated.
            # Do not divide the batch-means MCSE by sqrt(repetitions).
            se=max(maximum(group.log_evidence_mcse),std(values))
            id=join((spec["constraint"],spec["adjustment"],spec["method_name"],first(group.scope)),"/")
            push!(models,(model=id,arm=abspath(arm),scope=first(group.scope),log_evidence=logZ,
                log_evidence_mcse=se,log_evidence_rep_sd=std(values),repetitions=length(values),
                numerical_screen=all(group.numerical_screen)))
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
            numerical_screen=a.numerical_screen && b.numerical_screen,
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
    Posterior draws use HMC. RHS priors use independent exact hyperprior rejection;
    SSLASSO priors use slice/Gibbs updates in radius coordinates with exact Jacobians.
    Bridge proposals use conditional Gaussian coordinates; exact target densities are unchanged.
    Bridge sampling uses disjoint proposal-training and evaluation halves of each chain.
    MCSE uses batch means for correlated MCMC bridge weights. Repeated proposals share
    the MCMC draws; their repeated SD is not an independent-chain Monte Carlo error.
    Inspect bridge_diagnostics.csv and HMC diagnostics before interpreting a Bayes factor.
    A false numerical_screen means the estimate is not ready for scientific reporting.
    Pairwise MCSE ignores shared prior-normalizer covariance and is conservative within an arm.
    These are partial-likelihood Bayes factors, not full survival-likelihood Bayes factors.
    All ordered static/time/model/constraint/adjustment comparisons use the same observed subjects.
    Bridge reference: https://doi.org/10.1016/j.jmp.2017.09.005
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
    out=run_ph_models(arm;samples=arg(args,"samples",10000),
        prior_samples=arg(args,"prior-samples",arg(args,"samples",10000)),warmup=arg(args,"warmup",4000),
        chains=arg(args,"chains",4),leapfrog=arg(args,"leapfrog",300),target_accept=arg(args,"target-accept",0.95),
        repetitions=arg(args,"repetitions",4),time_prior_scale=arg(args,"time-prior-scale",0.5),seed=arg(args,"seed",1))
    rscript=get(ENV,"RSCRIPT_BIN","Rscript")
    script=joinpath(@__DIR__,"schoenfeld.R")
    run(`$rscript $script $(joinpath(arm,"ph_design.csv")) $(joinpath(arm,"ph_coefficients.csv")) $out`)
    aggregate_ph_evidence(isempty(root) ? arm : root)
end

abspath(PROGRAM_FILE)==abspath(@__FILE__) && ph_main()
