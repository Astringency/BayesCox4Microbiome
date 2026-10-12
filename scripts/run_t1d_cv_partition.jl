#!/usr/bin/env julia
# Independent stage worker for the frozen T1D nested-CV experiment. Stop the
# serial driver before running workers; each outer fold/full tuning has one owner.
# Existing inner and outer fingerprints must match before any fit is performed.
const PARTITION_CODE_ROOT = get(ENV, "BCX_FROZEN_CODE", dirname(@__DIR__))
using Dates
include(joinpath(PARTITION_CODE_ROOT, "scripts", "run_t1d_clinical_revision.jl"))

function t1d_partition_setup(source; input = joinpath(PARTITION_CODE_ROOT, "data", "T1D.csv"), smoke=false)
    spec = JSON.parsefile(joinpath(source, "analysis_specification.json"); use_mmap=false)
    method, adjusted = spec["method_name"], spec["adjustment"] == "adjusted"
    df = CSV.read(input, DataFrame)
    C, cnames = t1d_clinical_design(df)
    X = Matrix{Float64}(df[:, spec["component_cols"]])
    time, event = Float64.(df.time), Int.(df.status)
    clinical = adjusted ? C : nothing
    sampler = Dict(:sampler => :hmc, :n_samples => spec["n_samples"], :warmup => spec["warmup"],
        :chains => spec["chains"], :multichains => true, :target_accept => 0.9,
        :hmc_n_leapfrog => spec["hmc_n_leapfrog"], :noncentered => true, :progress => false)
    model = Dict{Symbol,Any}(:constraint => :prox, :prox_sum0_lambda => 1e-6,
        :pseudocount => 1e-6, :standardize => false, :p0 => nothing)
    adjusted && merge!(model, Dict(:clinical_names => cnames, :clinical_prior_df => 3.0, :clinical_prior_scale => 2.5))
    prior = Dict{Symbol,Any}(:alpha_prior_kind => :invgamma, :alpha_prior_shape => 1.0, :alpha_prior_scale => size(X,2))
    car = Dict(:use_car => false, :sample_car_strength => false)
    grids = deepcopy(PRIOR_PARAM_GRIDS)
    smoke && (grids[method] = [Dict{String,Any}((method == "cox-rhs" ? "rhs_local_scale" : "sslasso_spike_scale") => 0.5)])
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
        requested_pseudocount=get(kwargs,:pseudocount,nothing),clinical_score...)
    all(e -> _finite_number(get(scores,e,missing)) !== nothing,("mean","median","MAP")) || error("nonfinite outer score")
    row = Dict{Symbol,Any}(:dataset_id=>"T1D",:constraint=>"prox",:prior=>method,:outer_fold=>fold,
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
    stage in ("1", "2", "3", "4", "5", "full_tuning") || error("invalid T1D stage")
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

function t1d_partition_main()
    args=parse_cli_args()
    source=abspath(arg(args,"source-arm",""))
    setup=t1d_partition_setup(source;input=arg(args,"input",joinpath(PARTITION_CODE_ROOT,"data","T1D.csv")),smoke=arg(args,"smoke",false))
    arg(args,"validate-only",false) && (println("Frozen fingerprint verified: $(setup.fingerprint)");return)
    stage=arg(args,"stage","1")
    started=string(Dates.now())
    println("Starting $(setup.method) $(setup.spec["adjustment"]) stage=$stage at $started");flush(stdout)
    with_t1d_stage_lock(source, stage) do
        run_t1d_partition(setup,stage)
    end
    println("Completed stage=$stage");flush(stdout)
end

abspath(PROGRAM_FILE)==abspath(@__FILE__) && t1d_partition_main()
