#!/usr/bin/env julia
# Frozen, resumable T1D alpha-count experiment. Jobs never reuse historical draws.
include(joinpath(@__DIR__, "..", "analysis_common.jl"))
include(joinpath(@__DIR__, "..", "tune_hyperparams_cindex.jl"))
using SHA, Dates
BLAS.set_num_threads(1)

function decoded_kwargs(raw)
    kw = Dict{Symbol,Any}(Symbol(k) => v for (k,v) in raw)
    for k in (:sampler, :nuts_backend, :constraint, :tie_method, :alpha_prior_kind, :prox_block_metric, :init)
        haskey(kw,k) && (kw[k] = Symbol(kw[k]))
    end
    return kw
end

function run_alpha_job(job, protocol, root)
    out = joinpath(root, job["id"])
    done = joinpath(out,"complete.json")
    isfile(done) && return
    mkpath(out)
    repo = normpath(joinpath(@__DIR__,"..",".."))
    input = joinpath(repo,"data","T1D.csv")
    bytes2hex(sha256(read(input))) == protocol["data_sha256"] || error("data fingerprint changed")
    df = CSV.read(input,DataFrame)
    cols = protocol["component_cols"]
    X, time, event = Matrix{Float64}(df[:,cols]), Float64.(df.time), Int.(df.status)
    adjusted = job["adjusted"]
    C = adjusted ? Float64.(hcat(df.Sex .== "M", df.Antibiotics .== "Tylosin")) : nothing
    kw = decoded_kwargs(job["kwargs"])
    kw[:realdata] = true
    kw[:compute_metrics] = false
    kw[:progress] = false
    adjusted && (kw[:clinical_names] = ["clinical[Sex_M]","clinical[Antibiotics_Tylosin]"])
    _write_tuning_json(joinpath(out,"job.json"),job)
    started = time_ns()
    if job["kind"] == "tuning"
        configs = [Dict{Symbol,Any}(Symbol(k)=>v for (k,v) in cfg) for cfg in protocol["configs"]]
        index = job["config_index"]
        offset = job["fit_seed_offset"]
        fitfun = (x,t,e; kwargs...) -> begin
            actual = Dict{Symbol,Any}(kwargs)
            actual[:seed] += offset
            fit_and_evaluate(x,t,e;actual...)
        end
        selected = _select_inner_cindex(X,time,event;
            prior="prox-sslasso",configs,base_sampler_kwargs=kw,
            base_prior_kwargs=Dict(),base_car_kwargs=Dict(),base_model_kwargs=Dict(),
            cv_folds=5,seed=job["cv_seed"],fit_function=fitfun,score_function=_validation_cindices,
            clinical_X=C,checkpoint_dir=joinpath(out,"checkpoints"),config_indices=[index])
        metrics = selected.metrics
        metrics.actual_fit_seed = job["cv_seed"] .+ 10_000 .* metrics.config_index .+ metrics.fold .+ offset
        metrics.alpha_prior_scale = fill(kw[:alpha_prior_scale],nrow(metrics))
        write_table(joinpath(out,"inner_cv_metrics.csv"),metrics)
        selected.config === nothing && error("candidate did not complete all five validation folds")
        _write_tuning_json(done,Dict("status"=>"complete","config_index"=>index,
            "score"=>selected.score,"elapsed_seconds"=>(time_ns()-started)/1e9))
    else
        adjusted && (kw[:clinical_X] = C)
        fit = fit_and_evaluate(X,time,event;kw...)
        beta = posterior_beta_draws(fit.result)
        clinical = adjusted ? posterior_clinical_draws(fit.result) : zeros(size(beta,1),0)
        labels = vcat(cols,adjusted ? kw[:clinical_names] : String[])
        summary = posterior_summary(hcat(beta,clinical);names=labels)
        summary.selected_95 = (summary.lower .> 0) .| (summary.upper .< 0)
        summary.feature_type = vcat(fill("taxon",size(beta,2)),fill("clinical",size(clinical,2)))
        wide = samples_wide_dataframe(fit.result)
        summary.pip = vcat([mean(wide[!,"gamma[$j]"]) for j in axes(beta,2)],fill(NaN,size(clinical,2)))
        write_table(joinpath(out,"posterior_features.csv"),summary)
        save_run_outputs(fit.result;output_dir=joinpath(out,"run"),run_id=job["id"],
            metadata=merge(job,Dict("data_sha256"=>protocol["data_sha256"],
                "julia_version"=>string(VERSION),"source_sha256"=>protocol["source_sha256"])),
            posterior_summary=summary,save_jld2=false,save_wide_csv=true)
        diagnostics = BayesCox4Micro.rank_convergence_diagnostics(wide)
        write_table(joinpath(out,"rank_diagnostics.csv"),diagnostics)
        _write_tuning_json(done,Dict("status"=>"complete",
            "taxa_selected"=>count(summary.selected_95[summary.feature_type .== "taxon"]),
            "clinical_selected"=>count(summary.selected_95[summary.feature_type .== "clinical"]),
            "elapsed_seconds"=>(time_ns()-started)/1e9))
    end
end

function alpha_main()
    protocol_path, root, worker, workers = ARGS[1], ARGS[2], parse(Int,ARGS[3]), parse(Int,ARGS[4])
    protocol = JSON.parsefile(protocol_path)
    for (index,job) in enumerate(protocol["jobs"])
        mod(index-1,workers)+1 == worker || continue
        println(now()," START ",job["id"]); flush(stdout)
        try
            run_alpha_job(job,protocol,root)
            println(now()," DONE ",job["id"]); flush(stdout)
        catch err
            mkpath(joinpath(root,job["id"]))
            _write_tuning_json(joinpath(root,job["id"],"error.json"),
                Dict("error"=>sprint(showerror,err,catch_backtrace()),"job"=>job))
            println(stderr,now()," FAILED ",job["id"]," ",sprint(showerror,err)); flush(stderr)
        end
        GC.gc()
    end
end
alpha_main()
