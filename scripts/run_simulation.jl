#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "analysis_common.jl"))

function main()
    args = parse_cli_args()
    outdir = arg(args, "outdir", joinpath("results", "simulation"))
    reps = arg(args, "reps", 1)
    ns = split_ints(arg(args, "n", "200"))
    ps = split_ints(arg(args, "p", "50"))
    nonzeros = split_ints(arg(args, "nonzero", "6"))
    baselines = split_symbols(arg(args, "baseline", "weibull"))
    priors = split_strings(arg(args, "priors", "hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs,lasso,sslasso"))
    sampler = Symbol(arg(args, "sampler", "nuts"))
    n_samples = arg(args, "samples", 500)
    warmup = arg(args, "warmup", 200)
    chains = arg(args, "chains", 4)
    target_accept = arg(args, "target-accept", 0.8)
    seed0 = arg(args, "seed", 1)
    use_car = arg(args, "use-car", true)
    car_distance = Symbol(arg(args, "car-distance", "bray"))
    p0_arg = parse_optional_p0(args; default = nothing)
    p0_alpha = arg(args, "p0-alpha", 0.05)

    tables_dir = ensure_dir(joinpath(outdir, "tables"))
    figs_dir = ensure_dir(joinpath(outdir, "figures"))
    all_metrics = DataFrame()
    all_fd = DataFrame()

    for rep in 1:reps, n in ns, p in ps, nonzero in nonzeros, baseline in baselines
        sim_seed = seed0 + rep - 1
        sim = generate_simulation_data(n, p, nonzero; baseline = baseline, seed = sim_seed)
        p0_value = p0_arg === :truth ? nonzero : p0_arg
        for prior in priors
            label = "rep$(rep)_$(prior)_$(baseline)_n$(n)_p$(p)_nz$(nonzero)"
            fit = fit_and_evaluate(
                sim.X,
                sim.time,
                sim.event;
                prior_name = prior,
                sampler = sampler,
                n_samples = n_samples,
                warmup = warmup,
                chains = chains,
                target_accept = target_accept,
                seed = sim_seed,
                use_car = use_car && occursin("hs", lowercase(prior)),
                car_distance = car_distance,
                p0 = p0_value,
                p0_alpha = p0_alpha,
            )
            summary = copy(fit.summary)
            summary.variable .= ["beta[$j]" for j in 1:p]
            summary.true_beta = sim.beta
            summary.nonzero = sim.beta .!= 0
            write_table(joinpath(tables_dir, "coef_$label.csv"), summary)
            coefficient_plot(
                summary,
                joinpath(figs_dir, "coef_$label.png");
                truth = sim.beta,
                title = "Simulation $prior $baseline n=$n p=$p nonzero=$nonzero",
            )

            fd = false_discovery(sim.beta, summary)
            mse_mean = mean((summary.mean .- sim.beta) .^ 2)
            mse_median = mean((summary.median .- sim.beta) .^ 2)
            metrics = copy(fit.metrics)
            metrics.prior .= prior
            metrics.rep .= rep
            metrics.n .= n
            metrics.p .= p
            metrics.nonzero_count .= nonzero
            metrics.p0 .= fit.p0
            metrics.baseline .= string(baseline)
            metrics.MSE = [mse_mean, mse_median]
            append!(all_metrics, metrics; cols = :union)
            push!(all_fd, (
                rep = rep,
                n = n,
                p = p,
                nonzero_count = nonzero,
                p0 = fit.p0,
                baseline = string(baseline),
                prior = prior,
                FP = fd.FP,
                FN = fd.FN,
                TP = fd.TP,
                TN = fd.TN,
                FDR = fd.FDR,
            ); cols = :union)
        end
    end

    write_table(joinpath(tables_dir, "simulation_metrics_long.csv"), all_metrics)
    write_table(joinpath(tables_dir, "simulation_false_discovery.csv"), all_fd)

    grouped = combine(
        groupby(all_metrics, [:prior, :n, :p, :nonzero_count, :p0, :baseline, :estimate]),
        :MSE => mean => :MSE_mean,
        :MSE => std => :MSE_sd,
        :LPML => mean => :LPML_mean,
        :LPML => std => :LPML_sd,
        :WAIC_pw1 => mean => :WAIC_pw1_mean,
        :WAIC_pw1 => std => :WAIC_pw1_sd,
        :Cindex => mean => :Cindex_mean,
        :Cindex => std => :Cindex_sd,
    )
    write_table(joinpath(tables_dir, "simulation_metrics_summary.csv"), grouped)

    if nrow(all_metrics) > 0
        metrics_plot(all_metrics, joinpath(figs_dir, "simulation_metrics.png"); title = "Simulation metrics")
    end
    println("Simulation outputs written to $(abspath(outdir))")
end

main()
