#!/usr/bin/env julia

if any(arg -> arg in ("--help", "-h"), ARGS)
    println("""
    Usage:
      julia --project=. scripts/diagnose_overshrinkage.jl --input results/simulation --output results/diagnosis

    Options:
      --input PATH        Root directory containing simulation run directories.
      --output PATH       Directory for diagnosis CSV/MD outputs.
      --no-samples true   Skip samples_wide.csv even when present.
      --help, -h          Show this help message.

    Outputs:
      overshrinkage_summary.csv
      hs_rhs_parameter_diagnostics.csv
      prox_top_beta_diagnostics.csv
      overshrinkage_report.md
      overshrinkage_final_report.md
      overshrinkage_experiment_commands.md
    """)
    exit(0)
end

using Pkg

function _activate_repo_project()
    project = abspath(joinpath(@__DIR__, "..", "Project.toml"))
    active = Base.active_project()
    active !== project && Pkg.activate(dirname(project))
end

_activate_repo_project()

include(joinpath(@__DIR__, "analysis_common.jl"))

using Dates

const HS_RHS_PRIORS = Set(["hs", "rhs", "new-rhs", "est-rhs", "cox-hs", "cox-rhs"])
const RHS_PRIORS = Set(["rhs", "new-rhs", "est-rhs", "cox-rhs"])
const PROX_PRIORS = Set(["prox-lasso", "prox-sslasso"])
const HS_RHS_PARAMETER_COLUMNS = [
    :run_id, :prior, :p0_actual, :n_events, :theoretical_tau0,
    :tau_median, :tau_q025, :tau_q975, :tau_median_over_theoretical_tau0,
    :c2_median, :c2_q025, :c2_q975, :lambda_median_nonzero,
    :lambda_median_zero, :lambda_ratio_nonzero_zero,
    :beta_shrinkage_ratio_nonzero, :nonzero_MSE, :zero_MSE,
    :sign_prob_nonzero_mean, :use_car, :car_edge_count, :car_avg_degree,
    :car_nonzero_nonzero_edges, :car_nonzero_zero_edges,
]
const PROX_TOP_COLUMNS = [
    :run_id, :prior, :beta_index, :true_beta, :posterior_mean,
    :lower, :upper, :abs_posterior_mean, :is_nonzero,
]

prior_key(x) = lowercase(replace(string(x), "_" => "-", " " => "-"))
is_hs_rhs_prior(x) = prior_key(x) in HS_RHS_PRIORS
is_rhs_prior(x) = prior_key(x) in RHS_PRIORS
is_prox_prior(x) = prior_key(x) in PROX_PRIORS

function safe_read_json(path::AbstractString)
    isfile(path) || return Dict{String,Any}()
    try
        return Dict{String,Any}(JSON.parsefile(path))
    catch err
        @warn "Could not read JSON file." path exception=(err, catch_backtrace())
        return Dict{String,Any}()
    end
end

function safe_read_csv(path::AbstractString)
    isfile(path) || return DataFrame()
    try
        return CSV.read(path, DataFrame)
    catch err
        @warn "Could not read CSV file." path exception=(err, catch_backtrace())
        return DataFrame()
    end
end

function get_any(meta::AbstractDict, keys...; default = missing)
    for key in keys
        skey = string(key)
        haskey(meta, skey) && return meta[skey]
        key isa Symbol && haskey(meta, key) && return meta[key]
    end
    return default
end

function as_float(x; default = NaN)
    x === missing && return default
    x === nothing && return default
    x isa Number && return Float64(x)
    x isa AbstractString || return default
    s = strip(x)
    isempty(s) && return default
    lowercase(s) in ("missing", "nothing", "null", "nan") && return default
    try
        return parse(Float64, s)
    catch
        return default
    end
end

function as_int(x; default = missing)
    xf = as_float(x)
    isfinite(xf) || return default
    return Int(round(xf))
end

function as_bool(x; default = missing)
    x === missing && return default
    x === nothing && return default
    x isa Bool && return x
    x isa Number && return x != 0
    x isa AbstractString || return default
    s = lowercase(strip(x))
    s in ("true", "1", "yes", "y") && return true
    s in ("false", "0", "no", "n") && return false
    return default
end

finite_or_missing(x) = isfinite(as_float(x)) ? as_float(x) : missing

function mean_finite(v)
    vals = Float64[]
    for x in v
        xf = as_float(x)
        isfinite(xf) && push!(vals, xf)
    end
    return isempty(vals) ? missing : mean(vals)
end

function minimum_finite(v)
    vals = Float64[]
    for x in v
        xf = as_float(x)
        isfinite(xf) && push!(vals, xf)
    end
    return isempty(vals) ? missing : minimum(vals)
end

function maximum_finite(v)
    vals = Float64[]
    for x in v
        xf = as_float(x)
        isfinite(xf) && push!(vals, xf)
    end
    return isempty(vals) ? missing : maximum(vals)
end

function qstats(v)
    vals = Float64[]
    for x in v
        xf = as_float(x)
        isfinite(xf) && push!(vals, xf)
    end
    isempty(vals) && return (median = missing, q025 = missing, q975 = missing)
    return (
        median = median(vals),
        q025 = quantile(vals, 0.025),
        q975 = quantile(vals, 0.975),
    )
end

function beta_index(name)
    m = match(r"^beta\[(\d+)\]$", string(name))
    m === nothing && return nothing
    return parse(Int, m.captures[1])
end

function find_run_dirs(root::AbstractString)
    dirs = String[]
    isdir(root) || return dirs
    for (dir, _, files) in walkdir(root)
        if ("posterior_summary.csv" in files || "metadata.json" in files) &&
           basename(normpath(dir)) != "diagnose"
            push!(dirs, dir)
        end
    end
    return sort(unique(dirs))
end

function col_symbol(df::DataFrame, candidates::Vector{Symbol})
    props = propertynames(df)
    for c in candidates
        c in props && return c
    end
    return nothing
end

function prepared_beta_summary(summary::DataFrame, p_hint, nonzero_hint)
    isempty(summary) && return nothing
    var_col = col_symbol(summary, [:variable, :parameters, :parameter])
    var_col === nothing && return nothing
    rows = copy(summary)
    idx = [beta_index(v) for v in rows[!, var_col]]
    keep = [x !== nothing for x in idx]
    any(keep) || return nothing
    rows = rows[keep, :]
    idx = Int.([x for x in idx[keep]])
    rows[!, :_beta_index] = idx
    sort!(rows, :_beta_index)
    p = p_hint === missing ? maximum(rows._beta_index) : Int(p_hint)

    mean_col = col_symbol(rows, [:mean, :Mean])
    median_col = col_symbol(rows, [:median, Symbol("50.0%"), Symbol("50%")])
    lower_col = col_symbol(rows, [:lower, :q025, Symbol("2.5%")])
    upper_col = col_symbol(rows, [:upper, :q975, Symbol("97.5%")])
    true_col = col_symbol(rows, [:true_beta, :truth, :beta_true])

    mean_est = mean_col === nothing ? fill(NaN, nrow(rows)) : as_float.(rows[!, mean_col])
    med_est = median_col === nothing ? fill(NaN, nrow(rows)) : as_float.(rows[!, median_col])
    lower = lower_col === nothing ? fill(NaN, nrow(rows)) : as_float.(rows[!, lower_col])
    upper = upper_col === nothing ? fill(NaN, nrow(rows)) : as_float.(rows[!, upper_col])

    truth =
        if true_col !== nothing
            as_float.(rows[!, true_col])
        elseif p !== missing && nonzero_hint !== missing
            first(true_beta(Int(p), Int(nonzero_hint)))
        else
            fill(NaN, nrow(rows))
        end

    if length(truth) != nrow(rows)
        truth = truth[rows._beta_index]
    end
    return (
        rows = rows,
        p = p,
        index = rows._beta_index,
        mean = Float64.(mean_est),
        median = Float64.(med_est),
        lower = Float64.(lower),
        upper = Float64.(upper),
        truth = Float64.(truth),
    )
end

function metric_row(metrics::DataFrame)
    isempty(metrics) && return nothing
    if :estimate in propertynames(metrics)
        hit = findfirst(string.(metrics.estimate) .== "mean")
        hit !== nothing && return metrics[hit, :]
    end
    return metrics[1, :]
end

function metric_value(row, cols...; default = missing)
    row === nothing && return default
    for c in cols
        c in propertynames(row) && return row[c]
    end
    return default
end

function reconstructed_sim(meta::AbstractDict)
    n = as_int(get_any(meta, "n"))
    p = as_int(get_any(meta, "p"))
    nonzero = as_int(get_any(meta, "nonzero_count", "nonzero"))
    seed = as_int(get_any(meta, "seed"))
    baseline_raw = get_any(meta, "baseline"; default = missing)
    any(x -> x === missing, (n, p, nonzero, seed, baseline_raw)) && return nothing
    baseline = Symbol(string(baseline_raw))
    rate = finite_or_missing(get_any(meta, "baseline_rate"))
    shape = finite_or_missing(get_any(meta, "baseline_shape"))
    try
        return generate_simulation_data(
            n,
            p,
            nonzero;
            baseline = baseline,
            seed = seed,
            baseline_rate = rate === missing ? nothing : rate,
            baseline_shape = shape === missing ? nothing : shape,
        )
    catch err
        @warn "Could not reconstruct simulated data for CAR/tau diagnostics." exception=(err, catch_backtrace())
        return nothing
    end
end

function tau0_for_run(prior::AbstractString, meta::AbstractDict, beta_info, sim)
    key = prior_key(prior)
    p = beta_info.p
    n = as_int(get_any(meta, "n"); default = length(beta_info.truth))
    n_events = as_int(get_any(meta, "number_of_events", "n_events"); default = missing)
    p0 = as_float(get_any(meta, "p0"; default = get_any(meta, "p0_actual"; default = missing)))
    isfinite(p0) || return (value = missing, note = "missing p0")
    p0 <= 0 || p0 >= p && return (value = missing, note = "invalid p0")

    hs_scale = as_float(get_any(meta, "hs_tau_scale"); default = 1.0)
    rhs_scale = as_float(get_any(meta, "rhs_tau_scale"); default = 1.0)
    if key == "hs"
        return (value = hs_scale, note = "hs fixed tau scale")
    elseif key == "cox-hs"
        n_events === missing && return (value = missing, note = "missing n_events")
        return (value = hs_scale * cox_tau0(p, p0, n_events), note = "hs_tau_scale * p0/(p-p0)/sqrt(n_events)")
    elseif key == "cox-rhs"
        n_events === missing && return (value = missing, note = "missing n_events")
        return (value = rhs_scale * cox_tau0(p, p0, n_events), note = "rhs_tau_scale * p0/(p-p0)/sqrt(n_events)")
    elseif key == "new-rhs"
        return (value = rhs_scale * linear_tau0(p, p0, n, 1.0), note = "rhs_tau_scale * p0/(p-p0)/sqrt(n)")
    elseif key == "est-rhs"
        if sim !== nothing
            try
                est_prior = cox_paper_prior(
                    "est-rhs";
                    p0 = p0,
                    X = sim.X,
                    time = sim.time,
                    event = sim.event,
                    pseudocount = as_float(get_any(meta, "pseudocount"); default = 0.0),
                )
                return (value = rhs_scale * linear_tau0(p, p0, n, est_prior.pseudo_sigma), note = "rhs_tau_scale * estimated pseudo_sigma * p0/(p-p0)/sqrt(n)")
            catch
            end
        end
        return (value = missing, note = "est-rhs pseudo_sigma unavailable")
    elseif key == "rhs"
        pseudo_scale = as_float(get_any(meta, "rhs_pseudo_sigma_scale"); default = 10.0)
        return (value = rhs_scale * linear_tau0(p, p0, n, sqrt(pseudo_scale)), note = "rhs uses sampled pseudo_sigma; value uses initial/default sqrt(rhs_pseudo_sigma_scale)")
    end
    return (value = missing, note = "not hs/rhs")
end

function sample_column(wide::DataFrame, name::AbstractString)
    sym = Symbol(name)
    sym in propertynames(wide) || return nothing
    return as_float.(wide[!, sym])
end

function transformed_sample_column(wide::DataFrame, raw::AbstractString, lograw::AbstractString)
    v = sample_column(wide, raw)
    v !== nothing && return v
    lv = sample_column(wide, lograw)
    lv !== nothing && return exp.(lv)
    return nothing
end

function pseudo_sigma_from_samples(wide::Union{DataFrame,Nothing})
    wide === nothing && return nothing
    v = transformed_sample_column(wide, "pseudo_sigma2", "log_pseudo_sigma2")
    v === nothing && return nothing
    return sqrt.(v)
end

function tau_samples(wide::Union{DataFrame,Nothing}, prior::AbstractString, meta::AbstractDict, beta_info, tau0)
    wide === nothing && return nothing
    v = transformed_sample_column(wide, "tau", "log_tau")
    v !== nothing && return v
    raw = sample_column(wide, "tau_raw")
    raw === nothing && return nothing
    key = prior_key(prior)
    p = beta_info.p
    n = as_int(get_any(meta, "n"); default = length(beta_info.truth))
    p0 = as_float(get_any(meta, "p0"; default = missing))
    rhs_scale = as_float(get_any(meta, "rhs_tau_scale"); default = 1.0)
    if key == "rhs"
        pseudo = pseudo_sigma_from_samples(wide)
        if pseudo !== nothing && isfinite(p0)
            return raw .* (rhs_scale * (p0 / (p - p0)) / sqrt(float(n))) .* pseudo
        end
    end
    tau0 === missing && return nothing
    return raw .* Float64(tau0)
end

function c2_samples(wide::Union{DataFrame,Nothing})
    wide === nothing && return nothing
    return transformed_sample_column(wide, "c2", "log_c2")
end

function lambda_medians(wide::Union{DataFrame,Nothing}, p::Integer)
    wide === nothing && return nothing
    out = fill(NaN, p)
    found = false
    for j in 1:p
        v = transformed_sample_column(wide, "lambda[$j]", "log_lambda[$j]")
        v === nothing && continue
        out[j] = median(v)
        found = true
    end
    return found ? out : nothing
end

function beta_draw_matrix(wide::Union{DataFrame,Nothing}, p::Integer)
    wide === nothing && return nothing
    cols = Symbol.("beta[$j]" for j in 1:p)
    all(c -> c in propertynames(wide), cols) || return nothing
    return Matrix{Float64}(wide[:, cols])
end

function sign_probability_stats(draws, truth)
    draws === nothing && return (
        mean = missing,
        min = missing,
        max = missing,
        fn90 = missing,
        fn95 = missing,
        sign_error_count = missing,
    )
    nz = findall(!iszero, truth)
    isempty(nz) && return (mean = missing, min = missing, max = missing, fn90 = missing, fn95 = missing, sign_error_count = missing)
    probs = Float64[]
    for j in nz
        tj = truth[j]
        pgood = tj > 0 ? mean(draws[:, j] .> 0) : mean(draws[:, j] .< 0)
        push!(probs, pgood)
    end
    return (
        mean = mean(probs),
        min = minimum(probs),
        max = maximum(probs),
        fn90 = count(<(0.90), probs),
        fn95 = count(<(0.95), probs),
        sign_error_count = count(<(0.50), probs),
    )
end

function project_root(beta::AbstractVector, alpha::Real; weights = nothing, tol::Real = 1e-10, maxiter::Integer = 100)
    alpha < 0 && return NaN
    if weights === nothing
        sum(abs, beta) <= alpha && return 0.0
        hi = maximum(abs, beta)
        rootfun = nu -> sum(max(abs(b) - nu, 0.0) for b in beta) - nu - alpha
    else
        sum(weights[j] * abs(beta[j]) for j in eachindex(beta)) <= alpha && return 0.0
        hi = maximum(abs(beta[j]) / weights[j] for j in eachindex(beta))
        rootfun = nu -> sum(weights[j] * max(abs(beta[j]) - nu * weights[j], 0.0) for j in eachindex(beta)) - nu - alpha
    end
    lo = 0.0
    for _ in 1:maxiter
        mid = 0.5 * (lo + hi)
        val = rootfun(mid)
        abs(val) <= tol && return mid
        val > 0 ? (lo = mid) : (hi = mid)
    end
    return 0.5 * (lo + hi)
end

function prox_stats(wide::Union{DataFrame,Nothing}, beta_info, prior::AbstractString, meta::AbstractDict)
    defaults = (
        alpha_median = missing, alpha_q025 = missing, alpha_q975 = missing,
        l1_norm_beta_mean = missing, weighted_l1_norm_beta_mean = missing,
        projection_active_rate = missing, prox_penalty_mean = missing,
    )
    wide === nothing && return defaults
    draws = beta_draw_matrix(wide, beta_info.p)
    draws === nothing && return defaults
    eta = sample_column(wide, "eta")
    eta === nothing && return defaults
    alpha = exp.(clamp.(eta, -700.0, 700.0))
    l1 = vec(sum(abs.(draws); dims = 2))
    key = prior_key(prior)
    prox_l1_lambda = as_float(get_any(meta, "prox_l1_lambda"); default = 1e-3)
    weighted_l1 = copy(l1)
    penalties = fill(NaN, length(alpha))
    active = falses(length(alpha))
    if key == "prox-sslasso"
        s0 = as_float(get_any(meta, "sslasso_spike_scale"); default = 0.04)
        s1 = as_float(get_any(meta, "sslasso_slab_scale"); default = 0.5)
        gamma_cols = Symbol.("gamma[$j]" for j in 1:beta_info.p)
        if all(c -> c in propertynames(wide), gamma_cols)
            gamma = Matrix{Float64}(wide[:, gamma_cols])
            for i in axes(draws, 1)
                weights = [gamma[i, j] >= 0.5 ? inv(s1) : inv(s0) for j in 1:beta_info.p]
                weighted_l1[i] = sum(weights[j] * abs(draws[i, j]) for j in 1:beta_info.p)
                active[i] = weighted_l1[i] > alpha[i]
                nu = project_root(view(draws, i, :), alpha[i]; weights = weights)
                prox_alpha = active[i] ? alpha[i] + nu : alpha[i]
                dist2 = (alpha[i] - prox_alpha)^2
                for j in 1:beta_info.p
                    prox_beta = active[i] ? sign(draws[i, j]) * max(abs(draws[i, j]) - nu * weights[j], 0.0) : draws[i, j]
                    dist2 += (draws[i, j] - prox_beta)^2
                end
                penalties[i] = 0.5 * dist2 / prox_l1_lambda
            end
        else
            active .= weighted_l1 .> alpha
        end
    else
        active .= l1 .> alpha
        for i in axes(draws, 1)
            nu = project_root(view(draws, i, :), alpha[i])
            prox_alpha = active[i] ? alpha[i] + nu : alpha[i]
            dist2 = (alpha[i] - prox_alpha)^2
            for j in 1:beta_info.p
                prox_beta = active[i] ? sign(draws[i, j]) * max(abs(draws[i, j]) - nu, 0.0) : draws[i, j]
                dist2 += (draws[i, j] - prox_beta)^2
            end
            penalties[i] = 0.5 * dist2 / prox_l1_lambda
        end
    end
    aq = qstats(alpha)
    return (
        alpha_median = aq.median,
        alpha_q025 = aq.q025,
        alpha_q975 = aq.q975,
        l1_norm_beta_mean = mean(l1),
        weighted_l1_norm_beta_mean = mean(weighted_l1),
        projection_active_rate = mean(active),
        prox_penalty_mean = mean_finite(penalties),
    )
end

function mcmc_stats(run_dir::AbstractString, diagnostics::AbstractDict)
    diag_df = safe_read_csv(joinpath(run_dir, "diagnose", "diagnostics_summary.csv"))
    max_rhat = missing
    min_ess = missing
    if !isempty(diag_df)
        :rhat in propertynames(diag_df) && (max_rhat = maximum_finite(diag_df.rhat))
        ess_cols = [c for c in (:ess_bulk, :ess_tail, :ess) if c in propertynames(diag_df)]
        if !isempty(ess_cols)
            ess_vals = Float64[]
            for c in ess_cols, x in diag_df[!, c]
                xf = as_float(x)
                isfinite(xf) && push!(ess_vals, xf)
            end
            min_ess = isempty(ess_vals) ? missing : minimum(ess_vals)
        end
    end
    divergences = finite_or_missing(get_any(diagnostics, "divergences", "n_divergent", "num_divergent"; default = missing))
    accept_rate = finite_or_missing(get_any(diagnostics, "acceptance_rate", "accept_rate"; default = missing))
    step_size = finite_or_missing(get_any(diagnostics, "step_size", "final_step_size"; default = missing))
    bad =
        (max_rhat !== missing && as_float(max_rhat) > 1.01) ||
        (min_ess !== missing && as_float(min_ess) < 100.0) ||
        (divergences !== missing && as_float(divergences) > 0)
    return (max_rhat = max_rhat, min_ess = min_ess, divergences = divergences, accept_rate = accept_rate, step_size = step_size, bad_mcmc_flag = bad)
end

function car_stats(meta::AbstractDict, beta_info, sim)
    use_car = as_bool(get_any(meta, "use_car"); default = false)
    use_car == true || return (edge_count = missing, avg_degree = missing, nz_nz = missing, nz_zero = missing)
    sim === nothing && return (edge_count = missing, avg_degree = missing, nz_nz = missing, nz_zero = missing)
    dist = Symbol(string(get_any(meta, "car_distance"; default = "bray")))
    q = as_float(get_any(meta, "car_weight_quantile"); default = 0.75)
    try
        W = car_weight_matrix(sim.X; distance = dist, threshold_quantile = q)
        p = size(W, 1)
        nz = beta_info.truth .!= 0
        edge_count = 0
        nz_nz = 0
        nz_zero = 0
        for j in 2:p, i in 1:(j - 1)
            W[i, j] == 0 && continue
            edge_count += 1
            if nz[i] && nz[j]
                nz_nz += 1
            elseif nz[i] || nz[j]
                nz_zero += 1
            end
        end
        return (edge_count = edge_count, avg_degree = 2edge_count / p, nz_nz = nz_nz, nz_zero = nz_zero)
    catch err
        @warn "Could not compute CAR stats." exception=(err, catch_backtrace())
        return (edge_count = missing, avg_degree = missing, nz_nz = missing, nz_zero = missing)
    end
end

function process_run(run_dir::AbstractString; read_samples::Bool)
    meta = safe_read_json(joinpath(run_dir, "metadata.json"))
    diagnostics = safe_read_json(joinpath(run_dir, "diagnostics.json"))
    summary = safe_read_csv(joinpath(run_dir, "posterior_summary.csv"))
    metrics = safe_read_csv(joinpath(run_dir, "metrics.csv"))
    fd = safe_read_csv(joinpath(run_dir, "false_discovery.csv"))
    p_hint = as_int(get_any(meta, "p"); default = missing)
    nonzero_hint = as_int(get_any(meta, "nonzero_count", "nonzero"); default = missing)
    beta_info = prepared_beta_summary(summary, p_hint, nonzero_hint)
    beta_info === nothing && return nothing

    prior = string(get_any(meta, "prior_name", "method_name", "prior"; default = basename(run_dir)))
    prior = prior_key(prior)
    run_id = string(get_any(meta, "run_id", "simulation_id"; default = basename(run_dir)))
    n = as_int(get_any(meta, "n"); default = missing)
    p = beta_info.p
    nonzero = nonzero_hint === missing ? count(!iszero, beta_info.truth) : nonzero_hint
    seed = as_int(get_any(meta, "seed"); default = missing)
    rep = as_int(get_any(meta, "replicate_id", "rep"); default = missing)
    baseline = string(get_any(meta, "baseline"; default = missing))
    n_events = as_int(get_any(meta, "number_of_events", "n_events"); default = missing)
    censoring_rate = finite_or_missing(get_any(meta, "observed_censoring_rate", "censoring_rate"; default = missing))
    p0_actual = finite_or_missing(get_any(meta, "p0"; default = missing))
    p0_setting =
        p0_actual === missing ? "missing" :
        nonzero !== missing && as_float(p0_actual) == Float64(nonzero) ? "truth_or_manual_equal_truth" :
        "not_truth_or_auto"

    metric = metric_row(metrics)
    fd_row = isempty(fd) ? nothing : fd[1, :]
    truth = beta_info.truth
    nz = truth .!= 0
    zc = .!nz
    mean_est = beta_info.mean
    med_est = beta_info.median
    lower = beta_info.lower
    upper = beta_info.upper

    true_abs = mean(abs.(truth[nz]))
    post_abs_nz = mean(abs.(mean_est[nz]))
    post_med_abs_nz = mean(abs.(med_est[nz]))
    post_abs_zero = any(zc) ? mean(abs.(mean_est[zc])) : missing
    nonzero_mse = mean((mean_est[nz] .- truth[nz]) .^ 2)
    zero_mse = any(zc) ? mean((mean_est[zc] .- truth[zc]) .^ 2) : 0.0
    total_mse = mean((mean_est .- truth) .^ 2)
    ranks = sortperm(abs.(mean_est), rev = true)
    k = count(nz)
    hit_topk = count(j -> nz[j], ranks[1:min(k, length(ranks))])
    hit_top2k = count(j -> nz[j], ranks[1:min(2k, length(ranks))])

    wide_path = joinpath(run_dir, "samples_wide.csv")
    wide = read_samples && isfile(wide_path) ? safe_read_csv(wide_path) : nothing
    draws = beta_draw_matrix(wide, p)
    sign_stats = sign_probability_stats(draws, truth)
    sim = reconstructed_sim(meta)
    tau0 = tau0_for_run(prior, meta, beta_info, sim)
    tau = tau_samples(wide, prior, meta, beta_info, tau0.value)
    tau_q = tau === nothing ? (median = missing, q025 = missing, q975 = missing) : qstats(tau)
    c2 = c2_samples(wide)
    c2_q = c2 === nothing ? (median = missing, q025 = missing, q975 = missing) : qstats(c2)
    lambdas = lambda_medians(wide, p)
    lambda_nz = lambdas === nothing ? missing : mean_finite(lambdas[nz])
    lambda_zero = lambdas === nothing || !any(zc) ? missing : mean_finite(lambdas[zc])
    lambda_ratio =
        lambda_nz === missing || lambda_zero === missing || as_float(lambda_zero) == 0.0 ?
        missing : as_float(lambda_nz) / as_float(lambda_zero)
    local_escape = lambda_ratio === missing ? missing : as_float(lambda_ratio) > 2.0
    tau_ratio =
        tau_q.median === missing || tau0.value === missing || as_float(tau0.value) == 0.0 ?
        missing : as_float(tau_q.median) / as_float(tau0.value)
    prox = prox_stats(wide, beta_info, prior, meta)
    mcmc = mcmc_stats(run_dir, diagnostics)
    car = car_stats(meta, beta_info, sim)

    mse_value = finite_or_missing(metric_value(metric, :MSE; default = total_mse))
    interval_nz = mean(upper[nz] .- lower[nz])
    interval_zero = any(zc) ? mean(upper[zc] .- lower[zc]) : missing
    shrinkage_ratio = true_abs == 0 ? missing : post_abs_nz / true_abs
    prox_instability =
        is_prox_prior(prior) &&
        ((mse_value !== missing && as_float(mse_value) > 1.0) ||
         (shrinkage_ratio !== missing && as_float(shrinkage_ratio) < 0.1 && interval_nz > 1.0) ||
         maximum(abs.(mean_est)) > 5.0)

    row = Dict{Symbol,Any}(
        :run_id => run_id,
        :run_dir => run_dir,
        :prior => prior,
        :n => n,
        :p => p,
        :nonzero => nonzero,
        :baseline => baseline,
        :rep => rep,
        :seed => seed,
        :p0_setting => p0_setting,
        :p0_actual => p0_actual,
        :n_events => n_events,
        :censoring_rate => censoring_rate,
        :use_car => as_bool(get_any(meta, "use_car"); default = missing),
        :car_distance => get_any(meta, "car_distance"; default = missing),
        :constraint => get_any(meta, "constraint"; default = missing),
        :sampler => get_any(meta, "sampler"; default = missing),
        :chains => get_any(meta, "chains"; default = missing),
        :warmup => get_any(meta, "warmup"; default = missing),
        :samples => get_any(meta, "samples", "n_samples"; default = missing),
        :hs_tau_scale => finite_or_missing(get_any(meta, "hs_tau_scale"; default = 1.0)),
        :rhs_tau_scale => finite_or_missing(get_any(meta, "rhs_tau_scale"; default = 1.0)),
        :hs_local_scale => finite_or_missing(get_any(meta, "hs_local_scale"; default = 1.0)),
        :rhs_local_scale => finite_or_missing(get_any(meta, "rhs_local_scale"; default = 1.0)),
        :rhs_slab_df => finite_or_missing(get_any(meta, "rhs_slab_df"; default = 4.0)),
        :rhs_slab_s2 => finite_or_missing(get_any(meta, "rhs_slab_s2"; default = 2.0)),
        :rhs_pseudo_sigma_df => finite_or_missing(get_any(meta, "rhs_pseudo_sigma_df"; default = 3.0)),
        :rhs_pseudo_sigma_scale => finite_or_missing(get_any(meta, "rhs_pseudo_sigma_scale"; default = 10.0)),
        :lasso_lambda => finite_or_missing(get_any(meta, "lasso_lambda"; default = missing)),
        :lasso_lambda2_shape => finite_or_missing(get_any(meta, "lasso_lambda2_shape"; default = 1.0)),
        :lasso_lambda2_rate => finite_or_missing(get_any(meta, "lasso_lambda2_rate"; default = 0.1)),
        :lasso_tau2_shape => finite_or_missing(get_any(meta, "lasso_tau2_shape"; default = 1.0)),
        :sslasso_spike_scale => finite_or_missing(get_any(meta, "sslasso_spike_scale"; default = 0.04)),
        :sslasso_slab_scale => finite_or_missing(get_any(meta, "sslasso_slab_scale"; default = 0.5)),
        :sslasso_theta_a => finite_or_missing(get_any(meta, "sslasso_theta_a"; default = 1.0)),
        :sslasso_theta_b => finite_or_missing(get_any(meta, "sslasso_theta_b"; default = 1.0)),
        :alpha_prior_shape_setting => finite_or_missing(get_any(meta, "alpha_prior_shape"; default = missing)),
        :alpha_prior_rate_setting => finite_or_missing(get_any(meta, "alpha_prior_rate"; default = missing)),
        :prox_l1_lambda_setting => finite_or_missing(get_any(meta, "prox_l1_lambda"; default = 1e-3)),
        :prox_sum0_lambda_setting => finite_or_missing(get_any(meta, "prox_sum0_lambda"; default = 1e-3)),
        :MSE => mse_value,
        :FP => fd_row === nothing ? missing : metric_value(fd_row, :FP),
        :FN => fd_row === nothing ? missing : metric_value(fd_row, :FN),
        Symbol("C-index") => finite_or_missing(metric_value(metric, :Cindex, Symbol("C-index"); default = missing)),
        :WAIC => finite_or_missing(metric_value(metric, :WAIC, :WAIC_pw1; default = missing)),
        :LPML => finite_or_missing(metric_value(metric, :LPML; default = missing)),
        :true_nonzero_beta_mean_abs => true_abs,
        :posterior_mean_abs_nonzero => post_abs_nz,
        :posterior_median_abs_nonzero => post_med_abs_nz,
        :posterior_mean_abs_zero => post_abs_zero,
        :shrinkage_ratio_nonzero => shrinkage_ratio,
        :shrinkage_ratio_median_nonzero => true_abs == 0 ? missing : post_med_abs_nz / true_abs,
        :bias_nonzero_mean => mean(mean_est[nz] .- truth[nz]),
        :abs_bias_nonzero_mean => mean(abs.(mean_est[nz] .- truth[nz])),
        :coverage_zero_nonzero => mean((lower[nz] .<= 0.0) .& (upper[nz] .>= 0.0)),
        :coverage_zero_zero => any(zc) ? mean((lower[zc] .<= 0.0) .& (upper[zc] .>= 0.0)) : missing,
        :interval_width_nonzero_mean => interval_nz,
        :interval_width_zero_mean => interval_zero,
        :beta_rank_recovery_topk => hit_topk,
        :beta_rank_recovery_top2k => hit_top2k,
        :sign_prob_nonzero_mean => sign_stats.mean,
        :min_sign_prob_nonzero => sign_stats.min,
        :max_sign_prob_nonzero => sign_stats.max,
        :sign_prob_FN_90 => sign_stats.fn90,
        :sign_prob_FN_95 => sign_stats.fn95,
        :sign_error_count_nonzero => sign_stats.sign_error_count,
        :nonzero_MSE => nonzero_mse,
        :zero_MSE => zero_mse,
        :total_MSE => total_mse,
        :nonzero_MSE_contribution => nonzero_mse * count(nz) / p,
        :zero_MSE_contribution => zero_mse * count(zc) / p,
        :theoretical_tau0 => tau0.value,
        :tau0_note => tau0.note,
        :tau_median => tau_q.median,
        :tau_q025 => tau_q.q025,
        :tau_q975 => tau_q.q975,
        :tau_median_over_theoretical_tau0 => tau_ratio,
        :c2_median => c2_q.median,
        :c2_q025 => c2_q.q025,
        :c2_q975 => c2_q.q975,
        :lambda_median_nonzero => lambda_nz,
        :lambda_median_zero => lambda_zero,
        :lambda_ratio_nonzero_zero => lambda_ratio,
        :local_escape_success => local_escape,
        :tau_too_small_flag => tau_q.median === missing ? missing : (as_float(tau_q.median) < 1e-3 || (tau_ratio !== missing && as_float(tau_ratio) < 0.25)),
        :slab_too_small_flag => c2_q.median === missing ? missing : as_float(c2_q.median) < 1.0,
        :alpha_median => prox.alpha_median,
        :alpha_q025 => prox.alpha_q025,
        :alpha_q975 => prox.alpha_q975,
        :l1_norm_beta_mean => prox.l1_norm_beta_mean,
        :weighted_l1_norm_beta_mean => prox.weighted_l1_norm_beta_mean,
        :projection_active_rate => prox.projection_active_rate,
        :prox_penalty_mean => prox.prox_penalty_mean,
        :prox_instability_flag => prox_instability,
        :max_rhat => mcmc.max_rhat,
        :min_ess => mcmc.min_ess,
        :divergences => mcmc.divergences,
        :accept_rate => mcmc.accept_rate,
        :step_size => mcmc.step_size,
        :bad_mcmc_flag => mcmc.bad_mcmc_flag,
        :car_edge_count => car.edge_count,
        :car_avg_degree => car.avg_degree,
        :car_nonzero_nonzero_edges => car.nz_nz,
        :car_nonzero_zero_edges => car.nz_zero,
    )

    prox_top = DataFrame()
    if is_prox_prior(prior)
        order = sortperm(abs.(mean_est), rev = true)
        keep = order[1:min(10, length(order))]
        prox_top = DataFrame(
            run_id = fill(run_id, length(keep)),
            prior = fill(prior, length(keep)),
            beta_index = keep,
            true_beta = truth[keep],
            posterior_mean = mean_est[keep],
            lower = lower[keep],
            upper = upper[keep],
            abs_posterior_mean = abs.(mean_est[keep]),
            is_nonzero = nz[keep],
        )
    end
    return (row = row, prox_top = prox_top)
end

function write_reports(summary::DataFrame, hs_rhs::DataFrame, prox_top::DataFrame, output::AbstractString)
    now_s = string(now())
    n_runs = nrow(summary)
    priors_seen = n_runs == 0 ? String[] : sort(unique(string.(summary.prior)))
    hs_rows = n_runs == 0 ? DataFrame() : summary[[p in HS_RHS_PRIORS for p in summary.prior], :]
    lasso_rows = n_runs == 0 ? DataFrame() : summary[[p in ("lasso", "sslasso") for p in summary.prior], :]
    prox_rows = n_runs == 0 ? DataFrame() : summary[[p in PROX_PRIORS for p in summary.prior], :]

    function prior_table(rows)
        isempty(rows) && return "No matching runs were found.\n"
        grouped = combine(
            groupby(rows, :prior),
            :MSE => mean_finite => :MSE_mean,
            :shrinkage_ratio_nonzero => mean_finite => :shrinkage_mean,
            :coverage_zero_nonzero => mean_finite => :nonzero_CI_covers_zero_mean,
            :sign_prob_nonzero_mean => mean_finite => :sign_prob_mean,
            :nonzero_MSE => mean_finite => :nonzero_MSE_mean,
            :zero_MSE => mean_finite => :zero_MSE_mean,
        )
        io = IOBuffer()
        println(io, "| prior | mean MSE | shrinkage | nonzero CI covers 0 | sign prob | nonzero MSE | zero MSE |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|")
        for r in eachrow(grouped)
            println(io, "| $(r.prior) | $(round(as_float(r.MSE_mean); digits=4)) | $(round(as_float(r.shrinkage_mean); digits=4)) | $(round(as_float(r.nonzero_CI_covers_zero_mean); digits=4)) | $(round(as_float(r.sign_prob_mean); digits=4)) | $(round(as_float(r.nonzero_MSE_mean); digits=4)) | $(round(as_float(r.zero_MSE_mean); digits=4)) |")
        end
        return String(take!(io))
    end

    p0_msg =
        n_runs == 0 ? "No runs were available for p0 verification." :
        all((summary.p0_actual .== summary.nonzero) .| ismissing.(summary.p0_actual)) ?
        "Available metadata are consistent with p0=truth where p0 is recorded." :
        "At least one run has p0_actual different from nonzero; inspect overshrinkage_summary.csv."

    sslasso_msg = "No SSLASSO run is available locally."
    if n_runs > 0 && :prior in propertynames(summary) && any(summary.prior .== "sslasso")
        ssl = summary[summary.prior .== "sslasso", :]
        sslasso_msg = "Available SSLASSO evidence: mean MSE=$(round(as_float(mean_finite(ssl.MSE)); digits=4)), shrinkage_ratio_nonzero=$(round(as_float(mean_finite(ssl.shrinkage_ratio_nonzero)); digits=4)), nonzero_MSE=$(round(as_float(mean_finite(ssl.nonzero_MSE)); digits=4)), zero_MSE=$(round(as_float(mean_finite(ssl.zero_MSE)); digits=4)), sign_prob_nonzero_mean=$(round(as_float(mean_finite(ssl.sign_prob_nonzero_mean)); digits=4))."
    end

    report = """
    # Overshrinkage Diagnostic Report

    Generated: $now_s

    ## Available Results

    Found $n_runs run(s). Priors seen: $(isempty(priors_seen) ? "none" : join(priors_seen, ", ")).

    ## A. p0=truth verification

    $p0_msg

    Code path: `scripts/run_simulation.jl` maps `--p0 truth` to `p0_requested = nonzero`, writes it to metadata as `p0`, and passes it into `fit_and_evaluate`; `fit_and_evaluate` passes it to `cox_paper_prior`, whose `_resolve_p0` returns the supplied value without univariate screening.

    ## B. HS/RHS systematic shrinkage

    $(prior_table(hs_rows))

    If `shrinkage_ratio_nonzero < 0.3`, the run shows coefficient compression. If shrinkage is not low but `coverage_zero_nonzero` is high, the main issue is interval width or likelihood uncertainty.

    ## C. Cox-specific tau0

    Compare `hs` vs `cox-hs` and RHS variants vs `cox-rhs` in `overshrinkage_summary.csv` and `hs_rhs_parameter_diagnostics.csv`. The script records `theoretical_tau0`, posterior `tau_median`, and `tau_median_over_theoretical_tau0`.

    ## D. RHS slab

    RHS slab diagnostics use `c2_median`, `c2_q025`, and `c2_q975`. A small `c2_median`, especially below 1 when true nonzero beta magnitudes are about 1-2.5, is flagged as `slab_too_small_flag`.

    ## E. Local lambda escape

    Use `lambda_ratio_nonzero_zero`; values near 1 imply true nonzero coefficients did not escape global shrinkage. `local_escape_success` uses a ratio > 2 heuristic.

    ## F. CAR

    CAR diagnostics are in `car_edge_count`, `car_avg_degree`, `car_nonzero_nonzero_edges`, and `car_nonzero_zero_edges`. When nonzero-zero edges dominate and HS/RHS shrinkage worsens with `use_car=true`, CAR smoothing is likely mixing signal and noise.

    ## G. LASSO/SSLASSO

    $(prior_table(lasso_rows))

    SSLASSO defaults are spike scale 0.04 and slab scale 0.5 for simulation. Since true beta magnitudes are about 1-2.5, low sign probability or low shrinkage ratio under SSLASSO is consistent with a slab scale that is too tight.

    ## H. Prox anomaly

    $(prior_table(prox_rows))

    Prox-specific columns include `alpha_median`, `projection_active_rate`, `prox_penalty_mean`, and `prox_instability_flag`. `prox_top_beta_diagnostics.csv` lists the largest absolute posterior means for diagnosing whether MSE > 1 comes from missed signals or zero-variable blow-up.

    ## I. MSE decomposition

    Each run includes `nonzero_MSE`, `zero_MSE`, `nonzero_MSE_contribution`, and `zero_MSE_contribution`. Nonzero dominance indicates signal compression; zero dominance indicates false positives or unstable large zero estimates.
    """
    write(joinpath(output, "overshrinkage_report.md"), replace(report, r"(?m)^    " => ""))

    final = """
    # Overshrinkage Final Report

    Generated: $now_s

    ## 1. Executive summary

    Current local evidence is limited to $n_runs existing run(s), so strong empirical ranking across all requested priors requires the follow-up runs in `overshrinkage_experiment_commands.md`. The code audit and available output support these likely mechanisms:

    1. HS/RHS tau calibration is the first mechanism to test. `p0=truth` is passed correctly when requested, so `p0=auto` should not be treated as the main explanation for those runs. The cox-specific formula `p0/(p-p0)/sqrt(n_events)` can be very small, especially at p=150 and with fewer events.
    2. RHS slab `c2 ~ InverseGamma(df/2, df*slab_s2/2)` defaults to df=4 and slab_s2=2. This may cap coefficients whose true magnitude is about 1-2.5 if posterior c2 concentrates too low.
    3. CAR smoothing acts on HS/RHS local `log(lambda)` only. If the CAR graph connects true nonzeros mostly to zeros, local escape can be suppressed.
    4. SSLASSO simulation defaults use `s0=0.04`, `s1=0.5`; this slab scale is plausibly tight for the simulated nonzero beta values.
    5. Prox-lasso/prox-sslasso are not the same target as ordinary lasso/sslasso: they introduce an L1 epigraph variable `alpha=exp(eta)`, a Moreau-Yosida penalty, optional weighted projection, and the zero-sum constraint. MSE > 1 must be interpreted through `nonzero_MSE` vs `zero_MSE` and prox projection diagnostics.

    p0 status: $p0_msg

    $sslasso_msg

    ## 2. HS/RHS global-local shrinkage diagnosis

    Inspect `hs_rhs_parameter_diagnostics.csv`. Key fields are `theoretical_tau0`, `tau_median`, `tau_median_over_theoretical_tau0`, `lambda_ratio_nonzero_zero`, and `beta_shrinkage_ratio_nonzero`. If tau is very small and lambda ratios stay near 1, the global-local mechanism is not separating signal from noise.

    $(prior_table(hs_rows))

    ## 3. RHS slab diagnosis

    Use `c2_median` and `slab_too_small_flag`. If `c2_median` remains below about 1 while true nonzero beta magnitudes are near 1-2.5, slab bottleneck is plausible. Run the slab sensitivity commands before changing defaults.

    ## 4. CAR diagnosis

    CAR is only applied when `use_car=true` and the prior name contains `hs`. It penalizes squared differences in `log(lambda)` across CAR edges. The current simulation does not obviously generate clustered phylogenetic signals, so CAR can be harmful if nonzero-zero edges dominate.

    ## 5. LASSO/SSLASSO diagnosis

    LASSO defaults use `lambda2 ~ Gamma(1, rate=0.1)`, implying a broad but often strong global L1 scale. SSLASSO defaults use `s1=0.5`, likely too small for beta magnitudes 1-2.5 unless the likelihood is very informative. Report sign probability and rank recovery in addition to CI selection.

    $(prior_table(lasso_rows))

    ## 6. Prox-LASSO/Prox-SSLASSO anomaly diagnosis

    Prox methods optimize/sample a smoothed L1 epigraph target, not the same hierarchical target as ordinary lasso/sslasso. Check whether `zero_MSE` or `nonzero_MSE` dominates. High `projection_active_rate`, small `alpha_median`, or large `prox_penalty_mean` indicates the L1 epigraph constraint is active and may be over-compressing. Large zero estimates in `prox_top_beta_diagnostics.csv` indicate instability rather than shrinkage.

    $(prior_table(prox_rows))

    ## 7. Cox likelihood information diagnosis

    Use `n_events`, `censoring_rate`, p, and rank recovery. If the low-difficulty sanity check recovers signal but p=150/high-censoring cases do not, the likelihood is weak. If even n=400,p=50,use_car=false fails, the prior/implementation is the dominant issue.

    ## 8. Recommendations

    - Keep `p0=truth` for diagnosis and report `p0_actual` in every run.
    - Run HS/RHS tau-scale sensitivity first; the code now applies `tau_scale` as a multiplier to formula-based tau0 with default behavior unchanged.
    - For cox-HS/cox-RHS, test multiplicative calibration of `p0/(p-p0)/sqrt(n_events)`.
    - For RHS, run `rhs_slab_s2 = 2,5,10,25` before changing model structure.
    - Default `use_car=false` unless the simulated truth is clustered according to the CAR/phylogenetic graph.
    - Expand SSLASSO `s1` to 1, 2, and 5 in simulation comparisons.
    - Treat prox methods as experimental until MSE > 1 is traced to either signal compression, zero blow-up, or projection/alpha settings.
    - Report sign probability, rank recovery, and MSE decomposition alongside the 95% CI rule.
    """
    write(joinpath(output, "overshrinkage_final_report.md"), replace(final, r"(?m)^    " => ""))
end

function write_experiment_commands(output::AbstractString)
    text = raw"""
    # Overshrinkage Follow-up Experiment Commands

    These commands are intentionally small or medium. Start with the smoke test.

    ## Smoke Test

    ```powershell
    julia --project=. scripts/run_simulation.jl --outdir results/diagnosis/smoke --reps 1 --n 200 --p 50 --nonzero 6 --baseline weibull --priors hs,rhs,cox-hs,cox-rhs --samples 30 --warmup 20 --chains 1 --p0 truth --use-car false --diagnose false --save-wide-csv true --progress false
    julia --project=. scripts/diagnose_overshrinkage.jl --input results/diagnosis/smoke --output results/diagnosis/smoke_diagnosis
    ```

    ## Group 1: HS/RHS tau scale sensitivity

    Run one tau multiplier at a time to keep failures isolated:

    ```powershell
    foreach ($tau in "0.5","1","2","5","10") {
      julia --project=. scripts/run_simulation.jl --outdir "results/diagnosis/tau_scale_$tau" --reps 3 --n 200 --p 50,150 --nonzero 6 --baseline weibull --priors hs,rhs,new-rhs,est-rhs,cox-hs,cox-rhs --samples 300 --warmup 200 --chains 2 --p0 truth --use-car false --hs-tau-scale $tau --rhs-tau-scale $tau --diagnose false --save-wide-csv true --progress false
      julia --project=. scripts/diagnose_overshrinkage.jl --input "results/diagnosis/tau_scale_$tau" --output "results/diagnosis/tau_scale_${tau}_summary"
    }
    ```

    ## Group 2: CAR sensitivity

    ```powershell
    foreach ($car in "false","true") {
      julia --project=. scripts/run_simulation.jl --outdir "results/diagnosis/car_$car" --reps 3 --n 200 --p 50,150 --nonzero 6 --baseline weibull --priors hs,rhs,cox-hs,cox-rhs --samples 300 --warmup 200 --chains 2 --p0 truth --use-car $car --car-distance bray --diagnose false --save-wide-csv true --progress false
    }
    ```

    ## Group 3: RHS slab sensitivity

    ```powershell
    foreach ($s2 in "2","5","10","25") {
      julia --project=. scripts/run_simulation.jl --outdir "results/diagnosis/rhs_slab_$s2" --reps 3 --n 200 --p 50,150 --nonzero 6 --baseline weibull --priors rhs,new-rhs,est-rhs,cox-rhs --samples 300 --warmup 200 --chains 2 --p0 truth --use-car false --rhs-slab-s2 $s2 --diagnose false --save-wide-csv true --progress false
    }
    ```

    ## Group 4: LASSO/SSLASSO scale sensitivity

    ```powershell
    foreach ($s1 in "0.5","1","2","5") {
      julia --project=. scripts/run_simulation.jl --outdir "results/diagnosis/sslasso_s1_$s1" --reps 3 --n 200 --p 50,150 --nonzero 6 --baseline weibull --priors sslasso --samples 300 --warmup 200 --chains 2 --p0 truth --use-car false --sslasso-s1 $s1 --diagnose false --save-wide-csv true --progress false
    }
    foreach ($lambda in "0.1","0.2","0.5","1.0") {
      julia --project=. scripts/run_simulation.jl --outdir "results/diagnosis/lasso_lambda_$lambda" --reps 3 --n 200 --p 50,150 --nonzero 6 --baseline weibull --priors lasso --samples 300 --warmup 200 --chains 2 --p0 truth --use-car false --lasso-lambda $lambda --diagnose false --save-wide-csv true --progress false
    }
    ```

    ## Group 5: Prox anomaly diagnosis

    ```powershell
    foreach ($shape in "0.5","1","2") {
      foreach ($rate in "0.05","0.1","0.2") {
        julia --project=. scripts/run_simulation.jl --outdir "results/diagnosis/prox_shape${shape}_rate${rate}" --reps 3 --n 200 --p 50 --nonzero 6 --baseline weibull --priors prox-lasso,prox-sslasso --samples 300 --warmup 200 --chains 2 --p0 truth --use-car false --constraint prox --prox-l1-lambda 0.001 --alpha-prior-shape $shape --alpha-prior-rate $rate --diagnose false --save-wide-csv true --progress false
      }
    }
    ```

    ## Group 6: Cox likelihood sanity check

    ```powershell
    julia --project=. scripts/run_simulation.jl --outdir results/diagnosis/sanity_n400_p50 --reps 3 --n 400 --p 50 --nonzero 6 --baseline weibull --priors cox-rhs,hs,sslasso --samples 300 --warmup 200 --chains 2 --p0 truth --use-car false --diagnose false --save-wide-csv true --progress false
    ```
    """
    write(joinpath(output, "overshrinkage_experiment_commands.md"), replace(text, r"(?m)^    " => ""))
end

function ensure_columns(df::DataFrame, cols::Vector{Symbol})
    out = copy(df)
    for col in cols
        col in propertynames(out) || (out[!, col] = Any[missing for _ in 1:nrow(out)])
    end
    return out[:, cols]
end

function main()
    args = parse_cli_args()
    input = abspath(arg(args, "input", joinpath("results", "simulation")))
    output = ensure_dir(abspath(arg(args, "output", joinpath("results", "diagnosis"))))
    read_samples = !arg(args, "no-samples", false)

    rows = DataFrame()
    hs_rows = DataFrame()
    prox_top = DataFrame()
    for run_dir in find_run_dirs(input)
        processed = process_run(run_dir; read_samples = read_samples)
        processed === nothing && continue
        push!(rows, processed.row; cols = :union)
        if get(processed.row, :prior, "") in HS_RHS_PRIORS
            hs_row = Dict(
                :run_id => processed.row[:run_id],
                :prior => processed.row[:prior],
                :p0_actual => processed.row[:p0_actual],
                :n_events => processed.row[:n_events],
                :theoretical_tau0 => processed.row[:theoretical_tau0],
                :tau_median => processed.row[:tau_median],
                :tau_q025 => processed.row[:tau_q025],
                :tau_q975 => processed.row[:tau_q975],
                :tau_median_over_theoretical_tau0 => processed.row[:tau_median_over_theoretical_tau0],
                :c2_median => processed.row[:c2_median],
                :c2_q025 => processed.row[:c2_q025],
                :c2_q975 => processed.row[:c2_q975],
                :lambda_median_nonzero => processed.row[:lambda_median_nonzero],
                :lambda_median_zero => processed.row[:lambda_median_zero],
                :lambda_ratio_nonzero_zero => processed.row[:lambda_ratio_nonzero_zero],
                :beta_shrinkage_ratio_nonzero => processed.row[:shrinkage_ratio_nonzero],
                :nonzero_MSE => processed.row[:nonzero_MSE],
                :zero_MSE => processed.row[:zero_MSE],
                :sign_prob_nonzero_mean => processed.row[:sign_prob_nonzero_mean],
                :use_car => processed.row[:use_car],
                :car_edge_count => processed.row[:car_edge_count],
                :car_avg_degree => processed.row[:car_avg_degree],
                :car_nonzero_nonzero_edges => processed.row[:car_nonzero_nonzero_edges],
                :car_nonzero_zero_edges => processed.row[:car_nonzero_zero_edges],
            )
            push!(hs_rows, hs_row; cols = :union)
        end
        append!(prox_top, processed.prox_top; cols = :union)
    end

    CSV.write(joinpath(output, "overshrinkage_summary.csv"), rows)
    CSV.write(joinpath(output, "hs_rhs_parameter_diagnostics.csv"), ensure_columns(hs_rows, HS_RHS_PARAMETER_COLUMNS))
    CSV.write(joinpath(output, "prox_top_beta_diagnostics.csv"), ensure_columns(prox_top, PROX_TOP_COLUMNS))
    write_reports(rows, hs_rows, prox_top, output)
    write_experiment_commands(output)
    println("Overshrinkage diagnosis written to $(abspath(output))")
end

main()
