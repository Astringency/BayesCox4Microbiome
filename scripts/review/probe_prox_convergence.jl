#= Small diagnostic contrasts on an existing training cohort; not formal results.

Usage: julia --project=FROZEN_CODE -t 4 probe_prox_convergence.jl RUN_DIR OUT CASES
No test outcomes are read. Each comma-separated case changes one setting.
=#
using BayesCox4Micro, CSV, DataFrames, JSON, Statistics, LinearAlgebra

function run_probes(source, out, cases)
    meta = JSON.parsefile(joinpath(source, "metadata.json"))
    p = Int(meta["p"])
    train = replace(meta["test_data"], "/test_data/" => "/training_data/")
    df = CSV.read(train, DataFrame)
    X = Matrix{Float64}(df[:, ["x$j" for j in 1:p]])
    configs = Dict(
        "control" => (;),
        "sum0_1e3" => (; prox_sum0_lambda=1e-3),
        "l1_1e3" => (; prox_l1_lambda=1e-3),
        "l1_1e4" => (; prox_l1_lambda=1e-4),
        "l1_1e6" => (; prox_l1_lambda=1e-6),
        "alpha_250" => (; alpha_prior_scale=250.0),
        "spike_010" => (; spike_scale=0.1),
        "nuts" => (; sampler=:nuts_explicit),
        "leapfrog_400" => (; hmc_n_leapfrog=400),
    )
    for name in split(cases, ',')
        dest = joinpath(out, name)
        isfile(joinpath(dest,"summary.json")) && continue
        mkpath(dest)
        cfg = merge((; spike_scale=0.04, slab_scale=0.5, alpha_prior_scale=50.0,
            prox_sum0_lambda=1e-6, prox_l1_lambda=1e-6, sampler=:hmc,
            hmc_n_leapfrog=100), configs[name])
        println("START ",name); flush(stdout)
        timed = @timed sample_bayes_cox(X, df.time, df.event;
            prior=SpikeSlabLassoPrior(spike_scale=cfg.spike_scale, slab_scale=cfg.slab_scale,
                                     theta_a=1.,theta_b=1.,proximal=true),
            sampler=cfg.sampler, hmc_n_leapfrog=cfg.hmc_n_leapfrog, max_depth=10,
            n_samples=2000, warmup=1000, chains=4, multichains=true, seed=Int(meta["replicate_id"]),
            target_accept=0.8, constraint=:prox, prox_sum0_lambda=cfg.prox_sum0_lambda,
            prox_l1_lambda=cfg.prox_l1_lambda, alpha_prior_shape=1., alpha_prior_scale=cfg.alpha_prior_scale,
            alpha_prior_kind=:invgamma, progress=false, standardize=false, pseudocount=0.)
        result = timed.value
        save_run_outputs(result; output_dir=dest, run_id=name, save_jld2=false, save_wide_csv=true,
            metadata=Dict("source"=>source,"purpose"=>"diagnostic pilot; not a formal replication",
                          "config"=>Dict(string(k)=>string(v) for (k,v) in pairs(cfg)),
                          "warmup"=>1000,"n_samples"=>2000,"chains"=>4,"p"=>p))
        ranks = CSV.read(joinpath(dest,"rank_diagnostics.csv"), DataFrame)
        beta = ranks[startswith.(string.(ranks.parameters),"beta["),:]
        gamma = ranks[startswith.(string.(ranks.parameters),"gamma["),:]
        wide = samples_wide_dataframe(result)
        blocks = [Dict("chain"=>first(g.chain),"alpha_mean"=>mean(exp.(g.eta)),
                       "mean_selected"=>mean(sum(Matrix(g[:,["gamma[$j]" for j in 1:p]]);dims=2)),
                       "mean_beta_l1"=>mean(sum(abs,Matrix(g[:,["beta[$j]" for j in 1:p]]);dims=2)))
                  for g in groupby(wide,:chain)]
        finitevals(col) = [Float64(x) for x in col if !ismissing(x) && isfinite(x)]
        rhat=maximum(finitevals(beta.rhat)); bulk=minimum(finitevals(beta.ess_bulk)); tail=minimum(finitevals(beta.ess_tail))
        summary=Dict("case"=>name,"config"=>Dict(string(k)=>string(v) for (k,v) in pairs(cfg)),
            "beta_rhat_max"=>rhat,"beta_ess_bulk_min"=>bulk,"beta_ess_tail_min"=>tail,
            "beta_screen_pass"=>rhat<=1.01 && bulk>=400 && tail>=400,
            "gamma_rhat_max"=>maximum(finitevals(gamma.rhat)),"seconds"=>timed.time,"chains"=>blocks)
        open(joinpath(dest,"summary.json"),"w") do io; JSON.print(io,summary,2); end
        println(JSON.json(summary)); flush(stdout)
    end
end

run_probes(ARGS...)
