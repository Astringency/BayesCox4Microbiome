#!/usr/bin/env julia
using BayesCox4Micro: _divergence_count, _json_ready, _kfold_indices, _select_inner_cindex, _validation_cindices, _validation_design_matrix, _write_tuning_json
# Reproducible nested follow-up: saved common splits, training-only selection.
include(joinpath(@__DIR__, "..", "analysis_common.jl"))

using SHA, Dates
BLAS.set_num_threads(1)

function nested_decode(raw)
    kw = Dict{Symbol,Any}(Symbol(k) => v for (k,v) in raw)
    for k in (:sampler, :nuts_backend, :constraint, :tie_method, :alpha_prior_kind,
              :prox_block_metric, :init)
        haskey(kw,k) && (kw[k] = Symbol(kw[k]))
    end
    kw[:realdata] = true
    kw[:compute_metrics] = false
    kw[:progress] = false
    return kw
end

function continuous_quality(result, out)
    d = BayesCox4Micro.convergence_diagnostics(samples_wide_dataframe(result))
    c = d[.!startswith.(String.(d.parameters), "gamma["), :]
    valid = all(x -> !ismissing(x) && isfinite(x), Matrix(c[:,[:rhat,:ess_bulk,:ess_tail]]))
    rh = valid ? maximum(c.rhat) : Inf
    bu = valid ? minimum(c.ess_bulk) : 0.0
    ta = valid ? minimum(c.ess_tail) : 0.0
    div = _divergence_count(result)
    return Dict("continuous_rhat_max"=>rh,"continuous_ess_bulk_min"=>bu,
        "continuous_ess_tail_min"=>ta,"divergences"=>div,
        "strict_pass"=>valid && rh<=1.01 && bu>=400 && ta>=400 && !ismissing(div) && div==0)
end

function export_prediction(fit, Xtrain, Xval, t, e, Cval, ids, out)
    z = _validation_design_matrix(Xtrain,Xval,fit.result.data)
    beta = vec(mean(posterior_beta_draws(fit.result);dims=1))
    clinical = posterior_clinical_draws(fit.result)
    if size(clinical,2)>0
        z = hcat(z,Cval)
        beta = vcat(beta,vec(mean(clinical;dims=1)))
    end
    risk = z*beta
    write_table(joinpath(out,"predictions.csv"),DataFrame(row_index=ids,time=t,event=e,risk=risk))
    write_table(joinpath(out,"posterior_mean.csv"),DataFrame(index=eachindex(beta),mean=beta))
    return concordance_index(t,e,risk)
end

function run_nested_job(job, protocol, root, X, time, event, clinical)
    out=joinpath(root,job["id"])
    done=joinpath(out,"complete.json")
    isfile(done) && return
    mkpath(out)
    _write_tuning_json(joinpath(out,"job.json"),job)
    arm=only(filter(a->a["id"]==job["arm"],protocol["arms"]))
    outer=job["outer_fold"]
    split=protocol["splits"][outer]
    tr=Int.(split["outer_train"]); te=Int.(split["outer_test"])
    C=arm["adjusted"] ? clinical : nothing
    kw=nested_decode(arm["kwargs"])
    arm["adjusted"] && (kw[:clinical_names] = ["clinical[Sex_M]","clinical[Antibiotics_Tylosin]"])
    seed=protocol["seed"]+1_000_000*outer
    configs=[Dict{Symbol,Any}(Symbol(k)=>v for (k,v) in c) for c in protocol["configs"]]
    ci=job["config_index"]
    started=time_ns()
    if job["kind"]=="inner"
        folds=_kfold_indices(event[tr];k=5,seed)
        [tr[idx] for idx in folds]==[Int.(idx) for idx in split["inner_validation"]] ||
            error("saved inner folds differ from regenerated folds")
        current_fold=Ref(0)
        fitfun=(x,t,e;kwargs...)->begin
            current_fold[]=kwargs[:seed]-seed-10_000*ci
            fit=fit_and_evaluate(x,t,e;kwargs...)
            localout=joinpath(out,"fold_$(current_fold[])");mkpath(localout)
            _write_tuning_json(joinpath(localout,"quality.json"),continuous_quality(fit.result,localout))
            return fit
        end
        scorefun=(fit,xtrain,xval,t,e;clinical_X_val=nothing)->begin
            c=export_prediction(fit,xtrain,xval,t,e,clinical_X_val,tr[folds[current_fold[]]],
                joinpath(out,"fold_$(current_fold[])"))
            vc=_validation_cindices(fit,xtrain,xval,t,e;clinical_X_val)
            c==vc["mean"] || error("exported prediction differs from validation score")
            return vc
        end
        selected=_select_inner_cindex(X[tr,:],time[tr],event[tr];prior="prox-sslasso",configs,
            base_sampler_kwargs=kw,base_prior_kwargs=Dict(),base_car_kwargs=Dict(),
            base_model_kwargs=Dict(),cv_folds=5,seed,fit_function=fitfun,score_function=scorefun,
            clinical_X=C===nothing ? nothing : C[tr,:],checkpoint_dir=joinpath(out,"checkpoints"),
            config_indices=[ci])
        write_table(joinpath(out,"inner_cv_metrics.csv"),selected.metrics)
        selected.config===nothing && error("candidate incomplete")
        _write_tuning_json(done,Dict("status"=>"complete","score"=>selected.score,
            "elapsed_seconds"=>(time_ns()-started)/1e9))
    else
        merge!(kw,configs[ci])
        kw[:seed]=seed+999_999
        C===nothing || (kw[:clinical_X]=C[tr,:])
        # The retry rule depends only on diagnostics, never the outer score.
        for (attempt,budget) in enumerate(protocol["outer_budgets"])
            dest=joinpath(out,"attempt_$attempt");mkpath(dest)
            kw[:warmup]=budget["warmup"];kw[:n_samples]=budget["n_samples"]
            fit=fit_and_evaluate(X[tr,:],time[tr],event[tr];kw...)
            quality=continuous_quality(fit.result,dest)
            save_run_outputs(fit.result;output_dir=joinpath(dest,"fit"),
                run_id=job["id"],metadata=merge(job,Dict("kwargs"=>_json_ready(kw),
                    "data_sha256"=>protocol["data_sha256"],"source_sha256"=>protocol["source_sha256"],
                    "julia_version"=>string(VERSION),"outer_train"=>tr,"outer_test"=>te)),
                posterior_summary=fit.summary,save_jld2=false,save_wide_csv=true)
            c=export_prediction(fit,X[tr,:],X[te,:],time[te],event[te],C===nothing ? nothing : C[te,:],
                te,dest)
            record=merge(quality,Dict("status"=>"complete","test_Cindex"=>c,"attempt"=>attempt,
                "warmup"=>kw[:warmup],"samples"=>kw[:n_samples],"config_index"=>ci,
                "elapsed_seconds"=>(time_ns()-started)/1e9))
            _write_tuning_json(joinpath(dest,"complete.json"),record)
            if quality["strict_pass"] || attempt==length(protocol["outer_budgets"])
                _write_tuning_json(done,record)
                break
            end
            GC.gc()
        end
    end
end

function nested_followup_main()
    protocol_path,root,worker,workers=ARGS[1],ARGS[2],parse(Int,ARGS[3]),parse(Int,ARGS[4])
    protocol=JSON.parsefile(protocol_path)
    input=joinpath(@__DIR__,"..","..","data","T1D.csv")
    bytes2hex(sha256(read(input)))==protocol["data_sha256"] || error("data hash mismatch")
    df=CSV.read(input,DataFrame)
    X=Matrix{Float64}(df[:,protocol["component_cols"]])
    time=Float64.(df.time);event=Int.(df.status)
    clinical=Float64.(hcat(df.Sex .== "M",df.Antibiotics .== "Tylosin"))
    expected=_kfold_indices(event;k=5,seed=protocol["seed"])
    expected==[Int.(s["outer_test"]) for s in protocol["splits"]] || error("outer split mismatch")
    for (i,job) in enumerate(protocol["jobs"])
        mod(i-1,workers)+1==worker || continue
        println(now()," START ",job["id"]);flush(stdout)
        try
            run_nested_job(job,protocol,root,X,time,event,clinical)
            println(now()," DONE ",job["id"]);flush(stdout)
        catch err
            err isa InterruptException && rethrow()
            mkpath(joinpath(root,job["id"]))
            _write_tuning_json(joinpath(root,job["id"],"error.json"),
                Dict("error"=>sprint(showerror,err,catch_backtrace())))
            rethrow()
        end
        GC.gc()
    end
end
nested_followup_main()
