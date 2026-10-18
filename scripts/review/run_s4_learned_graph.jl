# Fit the frozen S4 cohorts using the native training-abundance graph builder.
# julia -t 4 --project=FROZEN_PROJECT this_script.jl ROOT 1,2,3 [replay]
using BayesCox4Micro, CSV, DataFrames, JSON, Statistics, LinearAlgebra, SHA
LinearAlgebra.BLAS.set_num_threads(1)

function screen(df)
    valid(v)=all(x -> !ismissing(x) && isfinite(x),v)
    ok=all(valid(df[!,k]) for k in (:rhat,:ess_bulk,:ess_tail))
    vals(k,fn)=valid(df[!,k]) ? fn(df[!,k]) : nothing
    r=vals(:rhat,maximum);b=vals(:ess_bulk,minimum);t=vals(:ess_tail,minimum)
    return Dict("rhat_max"=>r,"ess_bulk_min"=>b,"ess_tail_min"=>t,
      "pass"=>ok && r<=1.01 && b>=400 && t>=400)
end

function fit_rep(root,rep;replay=false)
    arm=replay ? "independent_replay" : "learned_abundance"
    dest=joinpath(root,"runs",arm,"rep"*lpad(rep,2,'0'))
    isfile(joinpath(dest,"complete.json")) && return
    mkpath(dest)
    label="rep$(rep)_S4_weibull_n200_p50_nz6_seed$(rep).csv"
    trainpath=joinpath(root,"cohorts","datasets","S4",label)
    train=CSV.read(trainpath,DataFrame)
    X=Matrix{Float64}(train[:,["x$j" for j in 1:50]])
    # Only training abundances enter the native graph builder. Neither truth,
    # test data, the generating tree, nor survival outcomes are passed to it.
    W=replay ? zeros(50,50) : car_weight_matrix(X;distance=:bray,threshold_quantile=.99)
    CSV.write(joinpath(dest,"graph_weights.csv"),DataFrame(W,:auto))
    prior=cox_paper_prior("cox-rhs";p0=6,rhs_slab_df=4.,rhs_slab_s2=2.,
      rhs_tau_scale=1.,rhs_local_scale=1.)
    println("START ",arm," rep",rep," edges=",count(>(0),W)÷2);flush(stdout)
    timed=@timed sample_bayes_cox(X,train.time,train.event;
      prior=prior,sampler=:hmc,hmc_n_leapfrog=100,n_samples=5000,warmup=2000,
      thin=1,chains=4,multichains=true,noncentered=true,init=:zero,
      seed=rep,target_accept=.8,constraint=:prox,prox_sum0_lambda=.001,
      standardize=false,tie_method=:efron,use_car=!replay,
      # Passing no weights exercises the actual internal graph-estimation path.
      car_distance=:bray,car_weight_quantile=.99,car_precision=1.,
      sample_car_strength=!replay,car_strength_prior_a=1.,car_strength_prior_b=1.,
      car_degree_scale=:mean_active,car_kappa_init=.5,progress=false)
    result=timed.value
    metadata=Dict("scenario"=>"S4","arm"=>arm,"replicate"=>rep,"seed"=>rep,
      "n"=>200,"p"=>50,"nonzero_count"=>6,"p0"=>6,"baseline"=>"weibull",
      "sampler"=>"hmc","hmc_n_leapfrog"=>100,"chains"=>4,"warmup"=>2000,
      "n_samples"=>5000,"init"=>"zero","target_accept"=>.8,"constraint"=>"prox",
      "prox_sum0_lambda"=>.001,"use_car"=>!replay,"car_graph"=>arm,
      "car_distance"=>"bray","car_weight_quantile"=>.99,"sample_car_strength"=>!replay,
      "graph_uses_truth"=>false,"graph_uses_test"=>false,"graph_uses_survival"=>false,
      "graph_estimator"=>"native car_weight_matrix, no supplied car_weights",
      "training_path"=>trainpath,"training_sha256"=>bytes2hex(sha256(read(trainpath))),
      "graph_sha256"=>bytes2hex(sha256(read(joinpath(dest,"graph_weights.csv")))),
      "runtime"=>timed.time)
    save_run_outputs(result;output_dir=dest,run_id=arm*"_rep$(rep)",save_jld2=false,
      save_wide_csv=true,metadata=metadata)
    # Evaluation inputs are deliberately loaded only after graph building/fitting.
    truthpath=joinpath(root,"cohorts","beta","S4","beta_"*label)
    truth=CSV.read(truthpath,DataFrame).true_beta
    testpath=joinpath(root,"cohorts","testdata","S4",label)
    test=CSV.read(testpath,DataFrame);Xt=Matrix{Float64}(test[:,["x$j" for j in 1:50]])
    draws=posterior_beta_draws(result);beta=vec(mean(draws;dims=1))
    coefs=DataFrame(taxon=1:50,true_beta=truth,beta_mean=beta,
      beta_sd=vec(std(draws;dims=1)),lower=[quantile(draws[:,j],.025) for j in 1:50],
      upper=[quantile(draws[:,j],.975) for j in 1:50])
    CSV.write(joinpath(dest,"coefficient_audit.csv"),coefs)
    z=log.(X ./ sum(X;dims=2));zt=log.(Xt ./ sum(Xt;dims=2));risk=zt*beta
    CSV.write(joinpath(dest,"test_predictions.csv"),DataFrame(time=test.time,event=test.event,risk=risk))
    ranks=BayesCox4Micro.read_diagnostics(dest; category = "convergence")
    br=ranks[startswith.(string.(ranks.parameters),"beta["),:]
    @assert nrow(br)==50
    bs=screen(br);alls=screen(ranks)
    fd = false_discovery(truth, coefs)
    coverage = simulation_coverage(truth, coefs.lower, coefs.upper)
    fp, tp = fd.FP, fd.TP
    nz = truth .!= 0
    degrees=vec(sum(W;dims=2));L=Diagonal(degrees)-W
    transitions=BayesCox4Micro.read_diagnostics(dest; category = "transition")
    kept=transitions[string.(transitions.phase).=="sample",:]
    record=merge(metadata,Dict("test_path"=>testpath,"test_sha256"=>bytes2hex(sha256(read(testpath))),
      "truth_sha256"=>bytes2hex(sha256(read(truthpath))),"MSE"=>coefficient_mse(truth, beta),
      "Cindex_train"=>concordance_index(train.time,train.event,z*beta),
      "Cindex_test"=>concordance_index(test.time,test.event,risk),"FP"=>fp,"TP"=>tp,"FN"=>fd.FN,
      "FDP"=>fd.FDR,"coverage_95_nonzero"=>coverage.coverage_95_nonzero,
      "coverage_95_zero"=>coverage.coverage_95_zero,"coverage_95_all"=>coverage.coverage_95_all,
      "beta_screen_pass"=>bs["pass"],"all_screen_pass"=>alls["pass"],
      "graph_edges"=>count(>(0),W)÷2,"graph_rank"=>rank(L),"isolated_taxa"=>count(iszero,degrees),
      "signal_null_edges"=>count(W[i,j]>0 && nz[i]!=nz[j] for j in 2:50 for i in 1:(j-1)),
      "signal_signal_edges"=>count(W[i,j]>0 && nz[i] && nz[j] for j in 2:50 for i in 1:(j-1)),
      "null_null_edges"=>count(W[i,j]>0 && !nz[i] && !nz[j] for j in 2:50 for i in 1:(j-1)),
      "n_divergent_sample"=>sum(kept.n_divergent),"mean_acceptance_probability"=>mean(kept.mean_acceptance_probability)))
    for (prefix,sc) in [("beta",bs),("all",alls)],key in ["rhat_max","ess_bulk_min","ess_tail_min"]
        record[prefix*"_"*key]=sc[key]
    end
    if !replay
        wide=CSV.read(joinpath(dest,"samples_wide.csv"),DataFrame)
        kappas=1 ./ (1 .+ exp.(-wide.logit_car_kappa))
        record["kappa_mean"]=mean(kappas)
        record["kappa_q025"]=quantile(kappas,.025);record["kappa_q975"]=quantile(kappas,.975)
    end
    open(joinpath(dest,"complete.json"),"w") do io;JSON.print(io,record,2);end
    println("DONE ",arm," rep",rep," C=",record["Cindex_test"]," MSE=",record["MSE"]);flush(stdout)
end

root=abspath(ARGS[1]);reps=parse.(Int,split(ARGS[2],','))
for rep in reps
    fit_rep(root,rep;replay=length(ARGS)>2 && ARGS[3]=="replay")
end
