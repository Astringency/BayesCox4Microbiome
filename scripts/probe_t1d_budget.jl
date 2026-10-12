# Diagnostic full-data fits with unchanged priors and a longer sampling budget.
# No outer-CV result is recomputed or replaced.
using BayesCox4Micro, CSV, DataFrames, JSON, Statistics

function main(code, root, out)
    df=CSV.read(joinpath(code,"data/T1D.csv"),DataFrame)
    cols=filter(c->!(c in ["Sex","Antibiotics","time","status"]),names(df))
    X=Matrix{Float64}(df[:,cols])
    C=hcat(Float64.(df.Sex .== "M"),Float64.(df.Antibiotics .== "Tylosin"))
    for method in ["cox-rhs","prox-sslasso"]
        source=joinpath(root,"results",method=="cox-rhs" ? "" : "corrected_ssl",
                        "T1D_clinical","adjusted",method,"runs","realdata_"*method)
        meta=JSON.parsefile(joinpath(source,"metadata.json"));cfg=meta["selected_config"]
        dest=joinpath(out,method);mkpath(dest)
        isfile(joinpath(dest,"summary.json")) && continue
        kwargs=Dict(Symbol(k)=>v for (k,v) in cfg)
        prior=cox_paper_prior(method; p0=get(meta,"p0",nothing), X, time=df.time,event=df.status,
            realdata=true, (k=>v for (k,v) in kwargs if startswith(string(k),"rhs_") || startswith(string(k),"sslasso_"))...)
        println("START ",method);flush(stdout)
        timed=@timed sample_clinical_cox(X,df.time,df.status,C;prior,
            clinical_names=["clinical[Sex_M]","clinical[Antibiotics_Tylosin]"],
            n_samples=4000,warmup=2000,chains=4,multichains=true,seed=1,target_accept=.9,
            hmc_n_leapfrog=100,noncentered=true,constraint=:prox,prox_sum0_lambda=1e-6,
            pseudocount=1e-6,standardize=false,use_car=false,
            prox_l1_lambda=get(cfg,"prox_l1_lambda",1e-4),alpha_prior_kind=:invgamma,
            alpha_prior_shape=get(cfg,"alpha_prior_shape",1.),alpha_prior_scale=get(cfg,"alpha_prior_scale",30.),
            progress=false)
        save_run_outputs(timed.value;output_dir=dest,run_id=method,save_jld2=false,save_wide_csv=true,
            metadata=Dict("source"=>source,"p"=>size(X,2),"selected_config"=>cfg,
                "p0"=>get(meta,"p0",nothing),"n_samples"=>4000,"warmup"=>2000,"chains"=>4,
                "purpose"=>"unchanged prior, full-data diagnostic budget probe; not outer-CV"))
        rank=CSV.read(joinpath(dest,"rank_diagnostics.csv"),DataFrame)
        b=rank[startswith.(string.(rank.parameters),"beta["),:]
        vals(x)=[Float64(v) for v in x if !ismissing(v) && isfinite(v)]
        rh=maximum(vals(b.rhat));bu=minimum(vals(b.ess_bulk));ta=minimum(vals(b.ess_tail))
        summary=Dict("method"=>method,"beta_rhat_max"=>rh,"beta_ess_bulk_min"=>bu,
            "beta_ess_tail_min"=>ta,"beta_screen_pass"=>rh<=1.01 && bu>=400 && ta>=400,
            "seconds"=>timed.time,"config"=>cfg,"n_samples"=>4000,"warmup"=>2000)
        open(joinpath(dest,"summary.json"),"w") do io;JSON.print(io,summary,2);end
        println(JSON.json(summary));flush(stdout)
    end
end
main(ARGS...)
