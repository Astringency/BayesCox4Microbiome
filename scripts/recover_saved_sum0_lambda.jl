#!/usr/bin/env julia
# Identify an omitted lambda from stored posterior densities, without sampling.
include(joinpath(@__DIR__, "analysis_common.jl"))

function recover_sum0_lambda(source; max_draws = 100, gamma_mode = :stored)
    B = BayesCox4Micro
    meta = JSON.parse(read(joinpath(source, "metadata.json"), String))
    get(meta, "scenario", "baseline") == "baseline" || error("historical density replay currently requires baseline DGP")
    get(meta, "constraint", "") == "prox" || error("only proximal sum-zero runs have lambda")
    p = Int(meta["p"])
    sim = generate_simulation_data(Int(meta["n"]), p, Int(meta["nonzero_count"]);
        baseline = Symbol(meta["baseline"]), seed = Int(meta["seed"]))
    sum(sim.event) == meta["number_of_events"] || error("replayed event count differs")
    get(meta, "standardize", false) && error("scale recovery required for standardized historical fits")
    data = prepare_cox_data(sim.X, sim.time, sim.event;
        pseudocount = Float64(get(meta, "pseudocount", 0)), tie_method = Symbol(get(meta, "tie_method", "efron")))
    method = get(meta, "method_name", get(meta, "prior_name", ""))
    method in ("cox-rhs", "prox-sslasso") || error("unsupported historical method")
    wide = CSV.read(joinpath(source, "samples_wide.csv"), DataFrame)
    rows = unique(round.(Int, range(1, nrow(wide); length = min(max_draws, nrow(wide)))))
    x, y = Float64[], Float64[]
    prior = prior_from_name(method, sim.X, sim.time, sim.event; p0 = meta["p0"],
        rhs_slab_df = Float64(get(meta, "rhs_slab_df", 4)), rhs_slab_s2 = Float64(get(meta, "rhs_slab_s2", 2)),
        rhs_tau_scale = Float64(get(meta, "rhs_tau_scale", 1)), rhs_local_scale = Float64(get(meta, "rhs_local_scale", 1)),
        sslasso_spike_scale = get(meta, "sslasso_spike_scale", nothing),
        sslasso_slab_scale = get(meta, "sslasso_slab_scale", nothing),
        sslasso_theta_a = Float64(get(meta, "sslasso_theta_a", 1)), sslasso_theta_b = Float64(get(meta, "sslasso_theta_b", 1)))
    use_car = get(meta, "use_car", false)
    weights = use_car ? B._car_weight_edges(car_weight_matrix(sim.X;
        distance = Symbol(get(meta, "car_distance", "bray")),
        threshold_quantile = Float64(get(meta, "car_weight_quantile", .99)))) : nothing
    gamma, theta = fill(true, p), Ref(.5)
    target = if method == "prox-sslasso"
        ashape, ascale = B._prox_alpha_hyperparameters(prior,
            get(meta, "alpha_prior_shape", nothing), get(meta, "alpha_prior_scale", nothing))
        CoxProxSSLassoTarget(data, prior, gamma, theta; constraint = :none,
            prox_sum0_lambda = .001, prox_l1_lambda = Float64(meta["prox_l1_lambda"]),
            alpha_prior_shape = ashape, alpha_prior_scale = ascale,
            alpha_prior_kind = Symbol(get(meta, "alpha_prior_kind", "invgamma")),
            prox_root_tol = Float64(get(meta, "prox_root_tol", 1e-10)), prox_root_maxiter = Int(get(meta, "prox_root_maxiter", 100)))
    else
        B.CoxPosteriorTarget(data, prior; constraint = :none, prox_sum0_lambda = .001,
            use_car, car_weights = weights, car_precision = Float64(get(meta, "car_precision", 1)),
            sample_car_strength = get(meta, "sample_car_strength", false),
            car_strength_prior_a = Float64(get(meta, "car_strength_prior_a", 1)),
            car_strength_prior_b = Float64(get(meta, "car_strength_prior_b", 1)))
    end
    statecols = state_names(target)
    all(c -> c in names(wide), statecols) || error("stored state columns do not match target")
    for i in rows
        u = Float64[wide[i, col] for col in statecols]
        if method == "prox-sslasso"
            gamma .= gamma_mode == :frozen_all_true ? true : Bool.([wide[i, "gamma[$j]"] for j in 1:p])
            theta[] = wide[i, "theta"]
        end
        push!(x, sum(u[1:p])^2/(2p))
        push!(y, logdensity(target, u) - wide[i, "logposterior"])
    end
    slope = cov(x, y) / var(x)
    intercept = mean(y) - slope*mean(x)
    residual = maximum(abs.(y .- (intercept .+ slope.*x)))
    lambda = inv(slope)
    nearest = lambda > 0 && isfinite(lambda) ? 10.0^round(log10(lambda)) : NaN
    verified = slope > 0 && residual < 1e-6 && isapprox(lambda, nearest; rtol = 1e-5)
    return Dict("source" => abspath(source), "verified" => verified,
        "prox_sum0_lambda" => verified ? nearest : nothing, "estimated_lambda" => lambda,
        "max_logdensity_residual" => residual, "intercept" => intercept, "n_checked_draws" => length(rows),
        "gamma_state_mode" => method == "prox-sslasso" ? string(gamma_mode) : "not_applicable",
        "provenance" => "stored logposterior: lp_without_sum_penalty - lp_saved = intercept + sum(beta)^2/(2p*lambda)")
end

function recover_lambda_main()
    args = parse_cli_args()
    result = recover_sum0_lambda(arg(args, "source", ""); max_draws = arg(args, "max-draws", 100),
        gamma_mode = Symbol(arg(args, "gamma-mode", "stored")))
    out = arg(args, "out", "recovered_settings.json")
    _loo_atomic_json(out, result)
    println(JSON.json(result))
    result["verified"] || error("lambda could not be reliably recovered")
end
abspath(PROGRAM_FILE) == abspath(@__FILE__) && recover_lambda_main()
