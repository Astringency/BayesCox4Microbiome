using Pkg
const PARTITION_CODE_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
Base.active_project() == joinpath(PARTITION_CODE_ROOT, "Project.toml") ||
    Pkg.activate(PARTITION_CODE_ROOT)

using BayesCox4Micro: _cindex_candidate_configs, _cindex_fit_kwargs, _dict_rows_dataframe, _finite_number, _fit_diagnostics, _json_ready, _kfold_indices, _loo_atomic_json, _loo_checkpoint_manifest, _nested_cindex_summary, _select_inner_cindex, _tuning_dict, _validation_cindices, _write_nested_table, _write_tuning_json
# T1D: clinical design, nested CV, candidate partitions and final posterior.
isdefined(@__MODULE__, :parse_cli_args) || include(joinpath(PARTITION_CODE_ROOT,"scripts","lib","analysis_common.jl"))

function t1d_main(args=parse_cli_args())
    require_known_args!(args,("stage","source-arm","retry-from","input","outdir","prior","adjusted",
        "constraint","samples","warmup","chains","seed","target-accept","leapfrog","alpha-prior-scale",
        "prox-sum0-lambda","prior-grid-version",
        "outer-folds","inner-folds","smoke","config-ids","prepare-only","final-samples","final-warmup",
        "run-ph","ph-samples","ph-prior-samples","ph-warmup","ph-chains","ph-leapfrog","ph-target-accept","ph-repetitions","ph-time-prior-scale"))
    stage=arg(args,"stage","all")
    (stage in ("all","prepare","full_tuning","final") || tryparse(Int,stage)!==nothing) || error("Unknown T1D stage")
    source=arg(args,"source-arm","")
    retry_from=arg(args,"retry-from","")
    if !isempty(retry_from)
        stage=="final" || error("--retry-from is only for a final posterior retry")
        isempty(source) && error("--retry-from requires a separate --source-arm destination")
        abspath(source)==abspath(retry_from) && error("Retry must preserve the original fit in a different directory")
        for relative in ("analysis_specification.json","full_data_tuning/selected_config.json","full_data_tuning/inner_cv_metrics.csv")
            src=joinpath(retry_from,relative); dest=joinpath(source,relative)
            isfile(src) || error("Missing retry input: $src")
            if isfile(dest)
                read(src)==read(dest) || error("Retry input differs: $dest")
            else
                mkpath(dirname(dest)); cp(src,dest)
            end
        end
    end
    if isempty(source)
        preparation=copy(args); preparation["prepare-only"]="true"
        source=t1d_prepare_specification(preparation)
        args["source-arm"]=source
    end
    source=abspath(source)
    s=t1d_partition_setup(source; input=get(args,"input",nothing),create_manifest=true)
    stage=="prepare" && return
    if haskey(args,"config-ids")
        (stage=="full_tuning" || tryparse(Int,stage)!==nothing) || error("Candidate partitions need a tuning/outer-fold stage")
        return t1d_candidate_worker(args)
    end
    if stage=="all"
        for fold in 1:s.spec["outer_folds"]
            with_t1d_stage_lock(source,string(fold)) do
                run_t1d_partition(s,string(fold))
            end
        end
        # Assemble only from completed training/test-fold checkpoints.
        rows=[Dict(Symbol(k)=>v for (k,v) in JSON.parsefile(joinpath(s.pdir,"outer_$(lpad(i,2,'0')).json"))) for i in 1:s.spec["outer_folds"]]
        root=joinpath(source,"nested_cv")
        _write_nested_table(joinpath(root,"nested_cv_folds.csv"),_dict_rows_dataframe(rows),[s.method])
        summary=_nested_cindex_summary(rows,[s.method];dataset_id="T1D",constraint=string(s.model[:constraint]),outer_folds=s.spec["outer_folds"],inner_folds=s.spec["inner_folds"])
        _write_nested_table(joinpath(root,"nested_cv_summary.csv"),summary,[s.method])
        args["stage"]="final"
        t1d_final_fit(args)
        return t1d_ph_diagnostics(source,args)
    end
    if stage=="final"
        t1d_final_fit(args)
        return t1d_ph_diagnostics(source,args)
    end
    with_t1d_stage_lock(source,stage) do
        run_t1d_partition(s,stage)
    end
end
# Joint Bayesian adjustment, nested CV, full-data tuning and posterior-mean PH inputs.

function t1d_clinical_design(df)
    all(x -> x in ("F", "M"), df.Sex) || error("Sex must be F/M without missing values")
    all(x -> x in ("Control", "Tylosin"), df.Antibiotics) || error("Antibiotics must be Control/Tylosin without missing values")
    return Float64.(hcat(df.Sex .== "M", df.Antibiotics .== "Tylosin")),
        ["clinical[Sex_M]", "clinical[Antibiotics_Tylosin]"]
end

function t1d_prepare_specification(args = parse_cli_args())
    input = arg(args, "input", joinpath("data", "T1D.csv"))
    root = arg(args, "outdir", joinpath("results", "realdata", "T1D"))
    method = arg(args, "prior", "cox-rhs")
    method in ("cox-rhs", "prox-sslasso") || error("unsupported prior")
    adjusted = arg(args, "adjusted", true)
    label = adjusted ? "adjusted" : "unadjusted"
    constraint = Symbol(arg(args, "constraint", "prox"))
    constraint in (:prox, :soft_center) || error("unsupported constraint")
    outdir = ensure_dir(joinpath(root, string(constraint), label, method))
    # Real-data sampling retains the existing T1D scale; simulation scale is separate.
    samples, warmup, chains = arg(args, "samples", 5000), arg(args, "warmup", 4000), arg(args, "chains", 4)
    seed = arg(args, "seed", 1)
    df = CSV.read(input, DataFrame)
    C, cnames = t1d_clinical_design(df)
    cols = filter(c -> !(c in ("Sex", "Antibiotics", "time", "status")), names(df))
    X, time, event = Matrix{Float64}(df[:, cols]), Float64.(df.time), Int.(df.status)
    clinical = adjusted ? C : nothing
    sampler = Dict(:sampler => :hmc, :n_samples => samples, :warmup => warmup, :chains => chains,
        :multichains => true, :target_accept => arg(args, "target-accept", 0.95),
        :hmc_n_leapfrog => arg(args, "leapfrog", 300),
        :noncentered => true, :progress => false)
    0 < sampler[:target_accept] < 1 || error("target-accept must lie strictly between 0 and 1")
    sampler[:hmc_n_leapfrog] > 0 || error("leapfrog must be positive")
    sum0_lambda = arg(args, "prox-sum0-lambda", 1e-6)
    isfinite(sum0_lambda) && sum0_lambda > 0 || error("prox-sum0-lambda must be finite and positive")
    model = Dict{Symbol,Any}(:constraint => constraint, :prox_sum0_lambda => sum0_lambda,
        :standardize => false, :p0 => nothing)
    adjusted && merge!(model, Dict(:clinical_names => cnames,
        :clinical_prior_df => 3.0, :clinical_prior_scale => 2.5))
    alpha_scale = arg(args, "alpha-prior-scale", Float64(size(X, 2) + (adjusted ? size(C, 2) : 0)))
    alpha_scale > 0 || error("alpha-prior-scale must be positive")
    prior_kwargs = Dict{Symbol,Any}(:alpha_prior_kind => :invgamma,
        :alpha_prior_shape => 1.0, :alpha_prior_scale => alpha_scale)
    car = Dict(:use_car => false, :sample_car_strength => false)
    outer, inner = arg(args, "outer-folds", 5), arg(args, "inner-folds", 5)
    metadata = Dict("dataset_name" => "T1D", "method_name" => method, "prior_name" => method,
        "adjustment" => label, "clinical_coding" => Dict("Sex_M" => "M=1,F=0", "Antibiotics_Tylosin" => "Tylosin=1,Control=0"),
        "clinical_prior" => "independent Student-t(df=3, location=0, scale=2.5); Cox adaptation",
        "clinical_prior_reference" => "https://doi.org/10.1214/08-AOAS191",
        "n" => length(time), "p" => size(X, 2), "n_samples" => samples, "warmup" => warmup,
        "chains" => chains, "constraint" => string(constraint), "prox_sum0_lambda" => sum0_lambda,
        "zero_handling" => "half_minimum_positive", "standardize" => false, "component_cols" => cols,
        "alpha_prior_scale" => alpha_scale,
        "alpha_scale_definition" => "fixed total coefficient count by default; clinical coefficients retain independent priors",
        "outer_folds" => outer, "inner_folds" => inner, "seed" => seed,
        "input"=>abspath(input), "smoke"=>arg(args,"smoke",false),
        "sampler" => "hmc", "hmc_n_leapfrog" => sampler[:hmc_n_leapfrog],
        "target_accept" => sampler[:target_accept],
        "note" => "Full-data tuning scores are not test performance. Clinical covariates are outside the taxon constraint.")
    specification_path = joinpath(outdir, "analysis_specification.json")
    if method == "prox-sslasso"
        version = arg(args, "prior-grid-version", BayesCox4Micro.T1D_SSLASSO_GRID_VERSION)
        metadata["prior_grid_version"] = version
        metadata["prior_grid"] = BayesCox4Micro.t1d_prior_param_grids(;version)[method]
    end
    if isfile(specification_path)
        JSON.parsefile(specification_path) == metadata || error("Existing T1D specification differs; use a new output root")
    else
        _write_tuning_json(specification_path, metadata)
    end
    return outdir
end

# Independent stage worker for the frozen T1D nested-CV experiment. Stop the
# serial driver before running workers; each outer fold/full tuning has one owner.
# Existing inner and outer fingerprints must match before any fit is performed.
using Dates

function t1d_partition_setup(source; input=nothing, smoke=false, create_manifest=false)
    spec = JSON.parsefile(joinpath(source, "analysis_specification.json"); use_mmap=false)
    input = input===nothing ? get(spec,"input",joinpath(PARTITION_CODE_ROOT,"data","T1D.csv")) : input
    method, adjusted = spec["method_name"], spec["adjustment"] == "adjusted"
    df = CSV.read(input, DataFrame)
    C, cnames = t1d_clinical_design(df)
    X = Matrix{Float64}(df[:, spec["component_cols"]])
    time, event = Float64.(df.time), Int.(df.status)
    clinical = adjusted ? C : nothing
    sampler = Dict(:sampler => :hmc, :n_samples => spec["n_samples"], :warmup => spec["warmup"],
        # Legacy specifications used 0.9; new specifications persist the chosen target.
        :chains => spec["chains"], :multichains => true, :target_accept => get(spec, "target_accept", 0.9),
        :hmc_n_leapfrog => spec["hmc_n_leapfrog"], :noncentered => true, :progress => false)
    model = Dict{Symbol,Any}(:constraint => Symbol(get(spec, "constraint", "prox")),
        :prox_sum0_lambda => get(spec, "prox_sum0_lambda", 1e-6),
        :standardize => false, :p0 => nothing)
    adjusted && merge!(model, Dict(:clinical_names => cnames, :clinical_prior_df => 3.0, :clinical_prior_scale => 2.5))
    alpha_scale = get(spec, "alpha_prior_scale", size(X,2) + (adjusted ? size(C,2) : 0))
    prior = Dict{Symbol,Any}(:alpha_prior_kind => :invgamma, :alpha_prior_shape => 1.0, :alpha_prior_scale => alpha_scale)
    car = Dict(:use_car => false, :sample_car_strength => false)
    grids = deepcopy(PRIOR_PARAM_GRIDS)
    if method == "prox-sslasso"
        version = get(spec, "prior_grid_version", "legacy_default")
        grids = BayesCox4Micro.t1d_prior_param_grids(;version)
        get(spec, "prior_grid", nothing) == grids[method] || error("T1D candidate grid differs from the frozen specification")
    end
    (smoke || get(spec,"smoke",false)) && (grids[method] = [Dict{String,Any}((method == "cox-rhs" ? "rhs_local_scale" : "sslasso_spike_scale") => 0.5)])
    configs = _cindex_candidate_configs(method, sampler; tune_priorparams=true, tune_algparams=false, prior_grids=grids)
    settings = Dict(:sampler => _tuning_dict(sampler), :prior => _tuning_dict(prior),
        :car => _tuning_dict(car), :model => _tuning_dict(model), :configs => Dict(method => configs),
        :outer_folds => spec["outer_folds"], :inner_folds => spec["inner_folds"], :seed => spec["seed"],
        :dataset_id => "T1D", :evaluation => "nested_cv_v1")
    clinical !== nothing && (settings[:clinical_columns] = size(clinical,2))
    prior_settings = merge(settings, Dict(:configs => configs, :prior_name => method))
    fingerprint = _loo_checkpoint_manifest(clinical === nothing ? X : hcat(X,clinical),
        time,event,prior_settings,nothing,true; mode="nested_cv")
    pdir = joinpath(source,"nested_cv",method)
    if create_manifest
        _loo_checkpoint_manifest(clinical === nothing ? X : hcat(X,clinical),
            time,event,prior_settings,pdir,true;mode="nested_cv")
    end
    manifest = JSON.parsefile(joinpath(pdir,"manifest.json"); use_mmap=false)
    manifest["fingerprint"] == fingerprint || error("partition does not match existing frozen data/settings/code fingerprint")
    return (; spec,method,adjusted,X,time,event,clinical,sampler,model,prior,car,configs,fingerprint,pdir,source)
end

function run_t1d_partition(setup, stage)
    (;spec,method,X,time,event,clinical,sampler,model,prior,car,configs,fingerprint,pdir,source) = setup
    if stage == "full_tuning"
        directory = ensure_dir(joinpath(source,"full_data_tuning"))
        selected = _select_inner_cindex(X,time,event; prior=method, configs,
            base_sampler_kwargs=sampler,base_prior_kwargs=prior,base_car_kwargs=car,base_model_kwargs=model,
            cv_folds=spec["inner_folds"],seed=spec["seed"],fit_function=fit_and_evaluate,
            score_function=_validation_cindices,clinical_X=clinical,checkpoint_dir=joinpath(directory,"inner_checkpoints"))
        selected.config === nothing && error("full-data tuning incomplete")
        write_table(joinpath(directory,"inner_cv_metrics.csv"),selected.metrics)
        _write_tuning_json(joinpath(directory,"selected_config.json"),selected.config)
        return
    end
    fold = parse(Int,stage)
    1 <= fold <= spec["outer_folds"] || error("outer fold out of range")
    label = "outer_$(lpad(fold,2,'0'))"
    checkpoint = joinpath(pdir,label*".json")
    if isfile(checkpoint)
        cached = JSON.parsefile(checkpoint; use_mmap=false)
        if get(cached,"fingerprint","") == fingerprint && get(cached,"status","") == "success"
            println("Reuse completed $label")
            return
        end
    end
    outer = _kfold_indices(event;k=spec["outer_folds"],seed=spec["seed"])
    test_idx = outer[fold]
    train_idx = setdiff(eachindex(time),test_idx)
    directory = ensure_dir(joinpath(pdir,label))
    inner = _select_inner_cindex(X[train_idx,:],time[train_idx],event[train_idx]; prior=method,configs,
        base_sampler_kwargs=sampler,base_prior_kwargs=prior,base_car_kwargs=car,base_model_kwargs=model,
        cv_folds=spec["inner_folds"],seed=spec["seed"]+1_000_000*fold,fit_function=fit_and_evaluate,
        score_function=_validation_cindices,clinical_X=clinical===nothing ? nothing : clinical[train_idx,:],
        checkpoint_dir=joinpath(directory,"inner_checkpoints"))
    write_table(joinpath(directory,"inner_cv_metrics.csv"),inner.metrics)
    _write_tuning_json(joinpath(directory,"split_indices.json"),Dict("outer_train"=>train_idx,"outer_test"=>test_idx,
        "inner_validation"=>[train_idx[idx] for idx in inner.folds]))
    inner.config === nothing && error("outer $fold has no complete inner candidate")
    _write_tuning_json(joinpath(directory,"selected_config.json"),inner.config)
    kwargs = _cindex_fit_kwargs(method,sampler,prior,car,model,inner.config,size(X,2),
        spec["seed"]+1_000_000*fold+999_999)
    clinical !== nothing && (kwargs[:clinical_X]=clinical[train_idx,:])
    fit = fit_and_evaluate(X[train_idx,:],time[train_idx],event[train_idx];kwargs...)
    clinical_score = clinical===nothing ? NamedTuple() : (clinical_X_val=clinical[test_idx,:],)
    scores = _validation_cindices(fit,X[train_idx,:],X[test_idx,:],time[test_idx],event[test_idx];
        clinical_score...)
    all(e -> _finite_number(get(scores,e,missing)) !== nothing,("mean","median","MAP")) || error("nonfinite outer score")
    row = Dict{Symbol,Any}(:dataset_id=>"T1D",:constraint=>string(model[:constraint]),:prior=>method,:outer_fold=>fold,
        :n_train=>length(train_idx),:n_test=>length(test_idx),:evaluation=>"nested_cv",:status=>"success",
        :fingerprint=>fingerprint,:error_message=>"",:inner_Cindex_mean=>inner.score,
        :selected_config_index=>inner.config_index,:selected_config=>JSON.json(_json_ready(inner.config)))
    for (k,v) in pairs(_fit_diagnostics(fit.result));row[Symbol("fit_",k)]=v;end
    for e in ("mean","median","MAP");row[Symbol("test_Cindex_",e)]=scores[e];end
    save_run_outputs(fit.result;output_dir=joinpath(directory,"fit"),run_id=label*"_"*method,
        metadata=Dict("method_name"=>method,"dataset_name"=>"T1D","outer_fold"=>fold,
            "evaluation"=>"nested_cv_outer_training","n"=>length(train_idx),"p"=>size(X,2),
            "fingerprint"=>fingerprint,"selected_config"=>inner.config),
        posterior_summary=fit.summary,save_jld2=false,save_wide_csv=true)
    _loo_atomic_json(checkpoint,_json_ready(row))
    # Root summary CSVs are assembled by the original serial driver after all
    # stage workers finish. Concurrent workers only write their own stage files.
end

function with_t1d_stage_lock(f::Function, source, stage)
    (stage=="full_tuning" || (tryparse(Int,stage)!==nothing && parse(Int,stage)>0)) || error("invalid T1D stage")
    # Linux deployment lock: an early-started stage and a later queued worker
    # must not write its checkpoints concurrently. Closing releases the lock;
    # the small lock file remains as a record and no outputs are removed.
    Sys.islinux() || return f()
    directory = ensure_dir(joinpath(source, "stage_locks"))
    open(joinpath(directory, stage * ".lock"), "a+") do io
        ccall(:flock, Cint, (Cint, Cint), fd(io), 2) == 0 || error("cannot acquire T1D stage lock")
        try
            return f()
        finally
            ccall(:flock, Cint, (Cint, Cint), fd(io), 8)
        end
    end
end

# New full-data fits after training-only tuning. No auxiliary Cox fitting.
using SHA

function export_final_posterior(fit, source, spec, X, time, event, clinical)
    run = joinpath(source,"runs","realdata_"*spec["method_name"])
    beta = posterior_beta_draws(fit.result)
    cdraw = clinical === nothing ? zeros(size(beta,1),0) : posterior_clinical_draws(fit.result)
    labels = vcat(spec["component_cols"], clinical === nothing ? String[] :
        ["clinical[Sex_M]","clinical[Antibiotics_Tylosin]"])
    all_draws = hcat(beta,cdraw)
    tab = posterior_summary(all_draws;names=labels)
    tab.selected_95 = (tab.lower .> 0) .| (tab.upper .< 0)
    tab.feature_type = vcat(fill("taxon",size(beta,2)),fill("clinical",size(cdraw,2)))
    tab.feature_id = labels
    taxonomy = CSV.read(joinpath(PARTITION_CODE_ROOT,"data","T1D_metadata.csv"),DataFrame;types=String)
    lookup = Dict(r.feature => (r.otu_id,r.genus) for r in eachrow(taxonomy))
    all(haskey(lookup,k) for k in spec["component_cols"]) || error("missing taxon annotation")
    tab.otu_id = [get(lookup,k,("",""))[1] for k in labels]
    tab.genus = [get(lookup,k,("",""))[2] for k in labels]
    tab.posterior_inclusion_probability = Vector{Union{Missing,Float64}}(missing,length(labels))
    wide = CSV.read(joinpath(run,"samples_wide.csv"),DataFrame)
    if spec["method_name"] == "prox-sslasso"
        for j in axes(beta,2)
            tab.posterior_inclusion_probability[j] = mean(wide[!,"gamma[$j]"])
        end
    end
    tab.pip_status = [t == "clinical" ? "not_applicable_unpenalized_clinical" :
        spec["method_name"] == "prox-sslasso" ? "Pr(gamma=1|data)" : "not_applicable_continuous_shrinkage" for t in tab.feature_type]
    write_table(joinpath(source,"posterior_features.csv"),tab)
    design = DataFrame(hcat(fit.result.data.z,clinical===nothing ? zeros(length(time),0) : clinical),labels)
    insertcols!(design,1,:subject_index=>collect(eachindex(time)),:time=>time,:status=>event)
    write_table(joinpath(source,"ph_design.csv"),design)
    write_table(joinpath(source,"ph_coefficients.csv"),DataFrame(feature_id=labels,posterior_mean=vec(mean(all_draws;dims=1))))
    eval_data = clinical === nothing ? fit.result.data : BayesCox4Micro.joint_cox_data(fit.result.data,clinical)
    ll = posterior_event_loglik(eval_data,all_draws)
    # Saved pointwise contributions use the full observed risk sets for every draw.
    write_table(joinpath(source,"event_loglik.csv"),DataFrame(permutedims(ll[findall(!=(0),event),:]),:auto))
    write_table(joinpath(source,"event_index.csv"),DataFrame(subject_index=findall(!=(0),event),time=time[findall(!=(0),event)]))
end

function t1d_final_fit(args=parse_cli_args())
    source = abspath(arg(args,"source-arm",""))
    stage = arg(args,"stage","final")
    s = t1d_partition_setup(source;create_manifest=true)
    stage == "prepare" && return
    if stage != "final"
        with_t1d_stage_lock(source,stage) do
            run_t1d_partition(s,stage)
        end
        return
    end
    (;spec,method,X,time,event,clinical,sampler,model,prior,car,configs)=s
    choice_file=joinpath(source,"full_data_tuning","selected_config.json")
    if !isfile(choice_file)
        with_t1d_stage_lock(source,"full_tuning") do
            run_t1d_partition(s,"full_tuning")
        end
    end
    selected=Dict(Symbol(k)=>v for (k,v) in JSON.parsefile(choice_file))
    # Preserve the declared CV budget unless a final-stage override is explicit.
    sampler[:n_samples]=arg(args,"final-samples",sampler[:n_samples])
    sampler[:warmup]=arg(args,"final-warmup",sampler[:warmup])
    kwargs=_cindex_fit_kwargs(method,sampler,prior,car,model,selected,size(X,2),spec["seed"])
    kwargs[:compute_metrics]=true
    clinical === nothing || (kwargs[:clinical_X]=clinical)
    completed=joinpath(source,"final_complete.json")
    if isfile(completed)
        record=JSON.parsefile(completed)
        old=JSON.parsefile(joinpath(source,"runs","realdata_"*method,"metadata.json"))
        old["n_samples"]==sampler[:n_samples] && old["warmup"]==sampler[:warmup] ||
            error("Completed final fit has a different budget; use --retry-from OLD --source-arm NEW --stage final")
        for (relative,digest) in record["files"]
            bytes2hex(sha256(read(joinpath(source,relative))))==digest || error("changed final output: $relative")
        end
        println("Verified completed final fit: $source")
        return
    end
    fit=fit_and_evaluate(X,time,event;kwargs...)
    fit.summary.variable=spec["component_cols"]
    write_table(joinpath(source,"coefficients.csv"),fit.summary)
    metrics=fit.metrics

    metrics.scoring_target=fill("event identity conditional on fixed full-data risk sets",nrow(metrics))
    metrics.LOO_method=fill("PSIS; no subject deletion or risk-set reconstruction",nrow(metrics))
    write_table(joinpath(source,"training_metrics.csv"),metrics)
    run=joinpath(source,"runs","realdata_"*method)
    meta=merge(spec,Dict("selected_config"=>selected,"p0"=>fit.p0,
        "n_samples"=>sampler[:n_samples],"warmup"=>sampler[:warmup],
        "data_sha256"=>bytes2hex(sha256(read(get(spec,"input",joinpath(PARTITION_CODE_ROOT,"data","T1D.csv"))))),
        "parameter_source"=>"full-data inner C-index tuning followed by joint Bayesian full-data fit",
        "clinical_prior"=>"independent Student-t(3,0,2.5); outside taxon constraints",
        "tie_method"=>"efron","use_car"=>false))
    save_run_outputs(fit.result;output_dir=run,run_id="realdata_"*method,
        metadata=meta,posterior_summary=fit.summary,save_jld2=false,save_wide_csv=true)
    export_final_posterior(fit,source,spec,X,time,event,clinical)
    files=Dict(relpath(joinpath(d,f),source)=>bytes2hex(sha256(read(joinpath(d,f))))
        for (d,_,fs) in walkdir(source) for f in fs if f in
        ("posterior_features.csv","ph_design.csv","ph_coefficients.csv","training_metrics.csv",
         "event_loglik.csv","event_index.csv","samples_wide.csv","metadata.json","selected_config.json"))
    _write_tuning_json(completed,Dict("status"=>"complete","files"=>files,"source"=>source))
end


# Disjoint candidate workers share a full-grid fingerprint and distinct caches.

function t1d_candidate_worker(args=parse_cli_args())
    source=abspath(arg(args,"source-arm",""))
    s=t1d_partition_setup(source;create_manifest=true)
    (;spec,method,X,time,event,clinical,sampler,model,prior,car,configs,pdir)=s
    stage=arg(args,"stage","full_tuning")
    ids=parse.(Int,split(arg(args,"config-ids","1"),','))
    all(i->1<=i<=length(configs),ids) || error("invalid candidate index")
    if stage == "full_tuning"
        train=collect(eachindex(time));seed=spec["seed"]
        directory=joinpath(source,"full_data_tuning")
    else
        fold=parse(Int,stage)
        test=_kfold_indices(event;k=spec["outer_folds"],seed=spec["seed"])[fold]
        train=setdiff(eachindex(time),test);seed=spec["seed"]+1_000_000*fold
        directory=joinpath(pdir,"outer_$(lpad(fold,2,'0'))")
    end
    checkpoint=joinpath(directory,"inner_checkpoints")
    _select_inner_cindex(X[train,:],time[train],event[train];prior=method,configs,
        base_sampler_kwargs=sampler,base_prior_kwargs=prior,base_car_kwargs=car,
        base_model_kwargs=model,cv_folds=spec["inner_folds"],seed,
        fit_function=fit_and_evaluate,score_function=_validation_cindices,
        clinical_X=clinical===nothing ? nothing : clinical[train,:],checkpoint_dir=checkpoint,
        config_indices=arg(args,"prepare-only",false) ? Int[] : ids)
    println("Completed candidate subset $ids for $source stage=$stage")
end


# Full/final requests include PH by default, including --source-arm retries.
function t1d_ph_diagnostics(source,args)
    arg(args,"run-ph",get(ENV,"RUN_PH","true")=="true") || return
    script=joinpath(@__DIR__,"..","diagnostics","ph.jl")
    options=String["--source-arm",source]
    for key in ("samples","prior-samples","warmup","chains","leapfrog","target-accept","repetitions","time-prior-scale")
        haskey(args,"ph-"*key) && append!(options,["--"*key,args["ph-"*key]])
    end
    cmd=`$(Base.julia_cmd()) --startup-file=no --project=$PARTITION_CODE_ROOT $script $options`
    run(cmd)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    t1d_main()
end
