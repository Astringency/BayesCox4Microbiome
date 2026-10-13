# Adaptive CAR edge ablation on the 50 frozen Brownian S4 cohorts.
# Oracle diagnostic: only graph changes; original model and sampler code are frozen.
using BayesCox4Micro, CSV, DataFrames, JSON, Statistics, LinearAlgebra, SHA

function run_probe(plan_path, job_id)
    plan = JSON.parsefile(plan_path)
    job = plan["jobs"][parse(Int, job_id)]
    source = job["source"]
    meta = JSON.parsefile(joinpath(source, "metadata.json"))
    train_path = replace(meta["test_data"], "/test_data/" => "/training_data/")
    train = CSV.read(train_path, DataFrame)
    test = CSV.read(meta["test_data"], DataFrame)
    X = Matrix{Float64}(train[:, ["x$j" for j in 1:50]])
    Xt = Matrix{Float64}(test[:, ["x$j" for j in 1:50]])
    truth = CSV.read(joinpath(source,"posterior_summary.csv"),DataFrame).true_beta
    weights_path = get(job, "weights", plan["weights"])
    W = Matrix{Float64}(CSV.read(weights_path,DataFrame))
    degree = mean(vec(sum(W; dims=2)))
    prior = cox_paper_prior("cox-rhs"; p0=6, rhs_slab_df=4., rhs_slab_s2=2.,
                           rhs_tau_scale=1., rhs_local_scale=1.)
    for name in job["cases"]
        dest = joinpath(job["output"],name)
        isfile(joinpath(dest,"complete.json")) && continue
        mkpath(dest)
        name in ("adaptive_cut", "adaptive_replay", "adaptive_misspecified_v2", "adaptive_correct_v2_replay") || error("Unknown case: $name")
        weights = copy(W)
        if name == "adaptive_cut"
            # Oracle diagnostic: remove signal-null edges using simulation truth.
            # Adaptive CAR recomputes the degree denominator and graph rank.
            # This matches the native model for the modified graph.
            for i in 1:50, j in 1:50
                (truth[i] == 0) != (truth[j] == 0) && (weights[i,j] = 0.)
            end
        end
        println("START ",job["scenario"]," rep",job["replicate"]," ",name); flush(stdout)
        timed = @timed sample_bayes_cox(X, train.time, train.event;
            prior=prior, sampler=:hmc, hmc_n_leapfrog=100,
            n_samples=5000, warmup=2000, thin=1, chains=4, multichains=true,
            noncentered=true, seed=Int(meta["seed"]), target_accept=.8,
            constraint=:prox, prox_sum0_lambda=.001, standardize=false,
            pseudocount=0., tie_method=:efron, use_car=true,
            car_weights=weights, sample_car_strength=true,
            car_strength_prior_a=1., car_strength_prior_b=1., progress=false)
        result = timed.value
        save_run_outputs(result; output_dir=dest, run_id=name,
            save_jld2=false, save_wide_csv=true,
            metadata=Dict("source"=>source,"scenario"=>job["scenario"],
              "replicate"=>job["replicate"],"case"=>name,"sample_car_strength"=>true,
              "original_degree_scale"=>degree,"modified_degree_scale"=>mean(vec(sum(weights;dims=2))),
              "oracle_edge_removal"=>name=="adaptive_cut", "warmup"=>2000,
              "graph_version"=>get(plan,"graph_version","raw_tree_v1"),
              "weights_file_sha256"=>bytes2hex(sha256(read(weights_path))),
              "cross_signal_edges"=>count(weights[i,j]>0 && iszero(truth[i])!=iszero(truth[j]) for j in 2:50 for i in 1:(j-1)),
              "n_samples"=>5000,"chains"=>4,"seed"=>meta["seed"],
              "training_sha256"=>bytes2hex(sha256(read(train_path))),
              "test_sha256"=>bytes2hex(sha256(read(meta["test_data"])))))
        draws = posterior_beta_draws(result)
        estimates = vec(mean(draws; dims=1))
        summary = DataFrame(taxon=1:50,true_beta=truth,beta_mean=estimates,
                            beta_sd=vec(std(draws;dims=1)),
                            beta_median=vec(median(draws;dims=1)), beta_map=posterior_beta_map(result),
                            lower=[quantile(draws[:,j],.025) for j in 1:50],
                            upper=[quantile(draws[:,j],.975) for j in 1:50])
        CSV.write(joinpath(dest,"coefficient_audit.csv"),summary)
        # All proportions here are positive and generated closed. Reclose as in
        # the original evaluation to avoid relying on CSV rounding.
        z = log.(X ./ sum(X;dims=2)); zt = log.(Xt ./ sum(Xt;dims=2))
        ranks = CSV.read(joinpath(dest,"rank_diagnostics.csv"),DataFrame)
        br = ranks[startswith.(string.(ranks.parameters),"beta["),:]
        finitecol(v) = Float64[x for x in v if !ismissing(x) && isfinite(x)]
        rh=maximum(finitecol(br.rhat)); eb=minimum(finitecol(br.ess_bulk)); et=minimum(finitecol(br.ess_tail))
        record = Dict("scenario"=>job["scenario"],"replicate"=>job["replicate"],
          "case"=>name,"sample_car_strength"=>true,"MSE"=>mean((estimates-truth).^2),
          "Cindex_train"=>concordance_index(train.time,train.event,z*estimates),
          "Cindex_test"=>concordance_index(test.time,test.event,zt*estimates),
          "beta_rhat_max"=>rh,"beta_ess_bulk_min"=>eb,"beta_ess_tail_min"=>et,
          "beta_screen_pass"=>rh<=1.01 && eb>=400 && et>=400,
          "seconds"=>timed.time,"source"=>source,"output"=>dest)
        for (label, idx) in [("pure_signal",[18,20,22,24]),("boundary_signal",[26,28]),
                             ("boundary_null",[1,2]),("other_null",setdiff(1:50,[1,2,18,20,22,24,26,28]))]
            record[label*"_sse"] = sum((estimates[idx]-truth[idx]).^2)
        end
        wide = CSV.read(joinpath(dest,"samples_wide.csv"),DataFrame)
        kappa_draws = 1 ./ (1 .+ exp.(-wide.logit_car_kappa))
        record["kappa_mean"] = mean(kappa_draws)
        record["kappa_median"] = median(kappa_draws)
        record["kappa_q025"] = quantile(kappa_draws,.025)
        record["kappa_q975"] = quantile(kappa_draws,.975)
        record["beta_sum_mean"] = sum(estimates)
        record["TP"] = count((summary.lower .> 0 .|| summary.upper .< 0) .&& (truth .!= 0))
        record["FP"] = count((summary.lower .> 0 .|| summary.upper .< 0) .&& (truth .== 0))
        open(joinpath(dest,"complete.json"),"w") do io; JSON.print(io,record,2); end
        println(JSON.json(record)); flush(stdout)
    end
end

run_probe(ARGS...)
