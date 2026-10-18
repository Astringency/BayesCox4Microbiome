# Paired mechanism probes on frozen training cohorts. Model code is unchanged.
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
    W = Matrix{Float64}(CSV.read(plan["weights"],DataFrame))
    degree = mean(vec(sum(W; dims=2)))
    prior = cox_paper_prior("cox-rhs"; p0=6, rhs_slab_df=4., rhs_slab_s2=2.,
                           rhs_tau_scale=1., rhs_local_scale=1.)
    for name in job["cases"]
        dest = joinpath(job["output"],name)
        isfile(joinpath(dest,"complete.json")) && continue
        mkpath(dest)
        kappa = name == "fixed005" ? .05 : name == "fixed020" ? .2 :
                name == "fixed050" || name == "cut050" ? .5 : 1.
        weights = copy(W)
        if name == "cut050"
            # Oracle diagnostic: remove signal-null edges using simulation truth.
            # Keep the original degree denominator, so every retained edge has
            # exactly the same penalty as in fixed050.
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
            tie_method=:efron, use_car=true,
            car_weights=weights, sample_car_strength=false,
            car_precision=kappa/degree, progress=false)
        result = timed.value
        save_run_outputs(result; output_dir=dest, run_id=name,
            save_jld2=false, save_wide_csv=true,
            metadata=Dict("source"=>source,"scenario"=>job["scenario"],
              "replicate"=>job["replicate"],"case"=>name,"fixed_kappa"=>kappa,
              "car_precision"=>kappa/degree,"original_degree_scale"=>degree,
              "oracle_edge_removal"=>name=="cut050", "warmup"=>2000,
              "n_samples"=>5000,"chains"=>4,"seed"=>meta["seed"],
              "training_sha256"=>bytes2hex(sha256(read(train_path))),
              "test_sha256"=>bytes2hex(sha256(read(meta["test_data"])))))
        draws = posterior_beta_draws(result)
        estimates = vec(mean(draws; dims=1))
        summary = DataFrame(taxon=1:50,true_beta=truth,beta_mean=estimates,
                            beta_sd=vec(std(draws;dims=1)))
        CSV.write(joinpath(dest,"coefficient_audit.csv"),summary)
        # All proportions here are positive and generated closed. Reclose as in
        # the original evaluation to avoid relying on CSV rounding.
        z = log.(X ./ sum(X;dims=2)); zt = log.(Xt ./ sum(Xt;dims=2))
        ranks = BayesCox4Micro.read_diagnostics(dest; category = "convergence")
        br = ranks[startswith.(string.(ranks.parameters),"beta["),:]
        finitecol(v) = Float64[x for x in v if !ismissing(x) && isfinite(x)]
        rh=maximum(finitecol(br.rhat)); eb=minimum(finitecol(br.ess_bulk)); et=minimum(finitecol(br.ess_tail))
        record = Dict("scenario"=>job["scenario"],"replicate"=>job["replicate"],
          "case"=>name,"fixed_kappa"=>kappa,"MSE"=>coefficient_mse(truth, estimates),
          "Cindex_train"=>concordance_index(train.time,train.event,z*estimates),
          "Cindex_test"=>concordance_index(test.time,test.event,zt*estimates),
          "beta_rhat_max"=>rh,"beta_ess_bulk_min"=>eb,"beta_ess_tail_min"=>et,
          "beta_screen_pass"=>rh<=1.01 && eb>=400 && et>=400,
          "seconds"=>timed.time,"source"=>source,"output"=>dest)
        for (label, idx) in [("pure_signal",[18,20,22,24]),("boundary_signal",[26,28]),
                             ("boundary_null",[1,2]),("other_null",setdiff(1:50,[1,2,18,20,22,24,26,28]))]
            record[label*"_sse"] = sum((estimates[idx]-truth[idx]).^2)
        end
        open(joinpath(dest,"complete.json"),"w") do io; JSON.print(io,record,2); end
        println(JSON.json(record)); flush(stdout)
    end
end

run_probe(ARGS...)
