#!/usr/bin/env julia

using Pkg

function _activate_repo_project()
    project = abspath(joinpath(@__DIR__, "..", "Project.toml"))
    active = Base.active_project()
    active !== project && Pkg.activate(dirname(project))
end

_activate_repo_project()

function _include_script(path::AbstractString)
    if get(ENV, "USE_REVISE", "false") in ("1", "true", "yes", "y")
        try
            @eval import Revise
            return Base.invokelatest(Revise.includet, path)
        catch err
            @warn "USE_REVISE was requested but Revise.includet failed; falling back to include." exception=(err, catch_backtrace())
        end
    end
    return include(path)
end

_include_script(joinpath(@__DIR__, "analysis_common.jl"))

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
    multichains = arg(args, "multichains", false)
    target_accept = arg(args, "target-accept", 0.8)
    seed0 = arg(args, "seed", 1)
    use_car = arg(args, "use-car", true)
    progress = arg(args, "progress", true)
    car_distance = Symbol(arg(args, "car-distance", "bray"))
    p0_arg = parse_optional_p0(args; default = nothing)
    p0_alpha = arg(args, "p0-alpha", 0.05)
    constraint = Symbol(arg(args, "constraint", "soft_center"))
    soft_center_scale = haskey(args, "soft-center-scale") ? parse(Float64, args["soft-center-scale"]) : nothing
    prox_sum0_lambda = arg(args, "prox-sum0-lambda", 1e-3)
    prox_l1_lambda = arg(args, "prox-l1-lambda", 1e-3)
    prox_root_tol = arg(args, "prox-root-tol", 1e-10)
    prox_root_maxiter = arg(args, "prox-root-maxiter", 100)
    mala_step_size = arg(args, "mala-step-size", 1e-4)
    mala_target_accept = arg(args, "mala-target-accept", 0.574)
    mala_adapt_every = arg(args, "mala-adapt-every", 50)
    mala_min_step = arg(args, "mala-min-step", 1e-8)
    mala_max_step = arg(args, "mala-max-step", 1.0)

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
                multichains = multichains,
                target_accept = target_accept,
                seed = sim_seed,
                use_car = use_car && occursin("hs", lowercase(prior)),
                car_distance = car_distance,
                p0 = p0_value,
                p0_alpha = p0_alpha,
                constraint = constraint,
                soft_center_scale = soft_center_scale,
                prox_sum0_lambda = prox_sum0_lambda,
                prox_l1_lambda = prox_l1_lambda,
                prox_root_tol = prox_root_tol,
                prox_root_maxiter = prox_root_maxiter,
                mala_step_size = mala_step_size,
                mala_target_accept = mala_target_accept,
                mala_adapt_every = mala_adapt_every,
                mala_min_step = mala_min_step,
                mala_max_step = mala_max_step,
                progress = progress,
            )
            summary = copy(fit.summary)
            summary.variable .= ["beta[$j]" for j in 1:p]
            summary.true_beta = sim.beta
            summary.nonzero = sim.beta .!= 0
            write_table(joinpath(tables_dir, "coef_$label.csv"), summary)
            save_run_outputs(
                fit.result;
                output_dir = joinpath(outdir, "runs", label),
                run_id = label,
                metadata = Dict(
                    "simulation_id" => label,
                    "replicate_id" => rep,
                    "method_name" => prior,
                    "prior_name" => prior,
                    "sampler" => string(sampler),
                    "baseline" => string(baseline),
                    "n" => n,
                    "p" => p,
                    "nonzero_count" => nonzero,
                    "p0" => fit.p0,
                ),
                settings = Dict(
                    "n_samples" => n_samples,
                    "warmup" => warmup,
                    "chains" => chains,
                    "multichains" => multichains,
                    "target_accept" => target_accept,
                    "seed" => sim_seed,
                    "thin" => 1,
                    "constraint" => string(constraint),
                    "soft_center_scale" => soft_center_scale,
                    "prox_sum0_lambda" => prox_sum0_lambda,
                    "prox_l1_lambda" => prox_l1_lambda,
                    "prox_root_tol" => prox_root_tol,
                    "prox_root_maxiter" => prox_root_maxiter,
                    "mala_step_size" => mala_step_size,
                    "mala_target_accept" => mala_target_accept,
                    "mala_adapt_every" => mala_adapt_every,
                    "mala_min_step" => mala_min_step,
                    "mala_max_step" => mala_max_step,
                    "use_car" => use_car && occursin("hs", lowercase(prior)),
                    "car_distance" => string(car_distance),
                    "p0_alpha" => p0_alpha,
                ),
                posterior_summary = summary,
            )
            coefficient_plot(
                summary,
                joinpath(figs_dir, "coef_$label.pdf");
                truth = sim.beta,
                title = "Simulation $prior $baseline n=$n p=$p nonzero=$nonzero",
                estimate = :mean,
            )
            coefficient_plot(
                summary,
                joinpath(figs_dir, "coef_median_$label.pdf");
                truth = sim.beta,
                title = "Simulation $prior median $baseline n=$n p=$p nonzero=$nonzero",
                estimate = :median,
            )
            coefficient_plot(
                summary,
                joinpath(figs_dir, "coef_map_$label.pdf");
                truth = sim.beta,
                title = "Simulation $prior MAP $baseline n=$n p=$p nonzero=$nonzero",
                estimate = :map,
            )

            fd = false_discovery(sim.beta, summary)
            mse_mean = mean((summary.mean .- sim.beta) .^ 2)
            mse_median = mean((summary.median .- sim.beta) .^ 2)
            mse_map = mean((summary.map .- sim.beta) .^ 2)
            metrics = copy(fit.metrics)
            metrics.prior .= prior
            metrics.rep .= rep
            metrics.n .= n
            metrics.p .= p
            metrics.nonzero_count .= nonzero
            metrics.p0 .= fit.p0
            metrics.baseline .= string(baseline)
            metrics.MSE = [mse_mean, mse_median, mse_map]
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

    add_elpd_diffs!(
        all_metrics;
        group_cols = [:rep, :n, :p, :nonzero_count, :baseline, :estimate],
    )
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
        :ELPD_LOO => mean => :ELPD_LOO_mean,
        :ELPD_LOO => std => :ELPD_LOO_sd,
        :ELPD_Diff => mean => :ELPD_Diff_mean,
        :ELPD_Diff => std => :ELPD_Diff_sd,
        :PLOO => mean => :PLOO_mean,
        :PLOO => std => :PLOO_sd,
        :LOOIC => mean => :LOOIC_mean,
        :LOOIC => std => :LOOIC_sd,
        :ParetoK_max => maximum => :ParetoK_max,
        :Cindex => mean => :Cindex_mean,
        :Cindex => std => :Cindex_sd,
    )
    write_table(joinpath(tables_dir, "simulation_metrics_summary.csv"), grouped)

    if nrow(all_metrics) > 0
        metrics_plot(all_metrics, joinpath(figs_dir, "simulation_metrics.pdf"); title = "Simulation metrics")
    end
    println("Simulation outputs written to $(abspath(outdir))")
end

main()
