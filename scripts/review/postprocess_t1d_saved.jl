#!/usr/bin/env julia
# Full-data inference summaries are separate from untouched outer predictions.
include(joinpath(@__DIR__,"..","analysis_common.jl"))
using SHA
BLAS.set_num_threads(1)
root=ARGS[1]
jobs=JSON.parsefile(joinpath(root,"postprocess_jobs.json"))
protocol=JSON.parsefile(joinpath(root,"protocol.json"))
df=CSV.read(joinpath(@__DIR__,"..","..","data","T1D.csv"),DataFrame)
X=Matrix{Float64}(df[:,protocol["component_cols"]]);time=Float64.(df.time);event=Int.(df.status)
data=prepare_cox_data(X,time,event;transform=:log,standardize=false,tie_method=:efron)
clinical=Float64.(hcat(df.Sex .== "M",df.Antibiotics .== "Tylosin"))
for j in jobs
    out=joinpath(root,"full_data",j["arm"]);mkpath(out)
    isfile(joinpath(out,"complete.json")) && continue
    println("START ",j["arm"]);flush(stdout)
    file=joinpath(j["run"],"samples_wide.csv")
    wide=CSV.read(file,DataFrame)
    beta=Matrix{Float64}(wide[:,["beta[$k]" for k in 1:30]])
    labels=String.(protocol["component_cols"])
    if j["adjusted"]
        clabels=["clinical[Sex_M]","clinical[Antibiotics_Tylosin]"]
        beta=hcat(beta,Matrix{Float64}(wide[:,clabels]))
        labels=vcat(labels,clabels)
    end
    eval_data=j["adjusted"] ? BayesCox4Micro.joint_cox_data(data,clinical) : data
    ll=posterior_event_loglik(eval_data,beta)[findall(!=(0),event),:]
    metrics=fit_metrics(eval_data,beta,vec(mean(beta;dims=1));chains=4,draws_per_chain=div(nrow(wide),4),ll_event=ll)
    tab=DataFrame([metrics]);tab.NLPML=tab.LPML
    tab.scoring_target=fill("event identity conditional on fixed full-data risk sets",1)
    write_table(joinpath(out,"training_metrics.csv"),tab)
    features=posterior_summary(beta;names=labels)
    features.feature_id=labels
    features.selected_95=(features.lower .> 0).|(features.upper .< 0)
    features.pip=vcat([mean(wide[!,"gamma[$k]"]) for k in 1:30],fill(NaN,length(labels)-30))
    write_table(joinpath(out,"posterior_features.csv"),features)
    design=DataFrame(eval_data.z,labels)
    insertcols!(design,1,:subject_index=>collect(eachindex(time)),:time=>time,:status=>event)
    write_table(joinpath(out,"ph_design.csv"),design)
    write_table(joinpath(out,"ph_coefficients.csv"),DataFrame(feature_id=labels,posterior_mean=vec(mean(beta;dims=1))))
    for name in ("rank_diagnostics.csv","transition_summary.csv","selection_mixing.csv","zero_sum_summary.csv")
        cp(joinpath(j["run"],name),joinpath(out,name);force=true)
    end
    open(joinpath(out,"complete.json"),"w") do io
        JSON.print(io,Dict("source"=>file,"source_sha256"=>bytes2hex(sha256(read(file))),
            "samples"=>nrow(wide),"status"=>"complete"),2)
    end
    println("DONE ",j["arm"]);flush(stdout)
end
