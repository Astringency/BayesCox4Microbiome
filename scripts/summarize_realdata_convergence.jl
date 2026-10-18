#!/usr/bin/env julia

using CSV
using BayesCox4Micro: prepare_cox_data
using DataFrames
using Dates
using JSON
using Plots
using Printf
using Statistics

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_REALDATA_ROOT = joinpath(ROOT, "results", "realdata")
const DEFAULT_TUNING_ROOT = joinpath(ROOT, "results", "realdata_tuning")
const DEFAULT_SUMMARY_DIR = joinpath(ROOT, "results", "summary", "realdata", "bayes_all_methods")
const DEFAULT_CV_FALLBACK = joinpath(ROOT, "results", "summary", "realdata", "bayes_cv_fallback", "cv_fallback_summary.csv")
const PRIOR_ORDER = ["hs", "rhs", "new-rhs", "est-rhs", "cox-hs", "cox-rhs", "lasso", "sslasso", "prox-lasso", "prox-sslasso"]
const GROUP_ORDER = Dict("T1D" => 1, "HCT" => 2, "TCGA" => 3)
const CONSTRAINT_ORDER = Dict("soft_center" => 1, "prox" => 2)
const PLOT_FONT = get(ENV, "REALDATA_PLOT_FONT", "Times")

function parse_args(args)
    opts = Dict{String,String}()
    i = 1
    while i <= length(args)
        key = args[i]
        startswith(key, "--") || throw(ArgumentError("unexpected argument: $key"))
        name = key[3:end]
        if i == length(args) || startswith(args[i + 1], "--")
            opts[name] = "true"
            i += 1
        else
            opts[name] = args[i + 1]
            i += 2
        end
    end
    return opts
end

function opt(opts, name, default)
    return get(opts, name, default)
end

function ensure_dir(path::AbstractString)
    mkpath(path)
    return path
end

function read_table(path::AbstractString)
    isfile(path) || return DataFrame()
    return CSV.read(path, DataFrame; normalizenames = false)
end

function write_table(path::AbstractString, table::DataFrame)
    ensure_dir(dirname(path))
    CSV.write(path, table)
    return path
end

function read_json(path::AbstractString)
    isfile(path) || return Dict{String,Any}()
    return JSON.parsefile(path)
end

function as_string(x)
    x === missing && return ""
    x === nothing && return ""
    return string(x)
end

function maybe_float(x)
    x === missing && return missing
    x === nothing && return missing
    if x isa Number
        xf = Float64(x)
        return isfinite(xf) ? xf : missing
    end
    s = strip(string(x))
    isempty(s) && return missing
    lowercase(s) in ("missing", "nothing", "nan", "na", "inf", "-inf") && return missing
    try
        xf = parse(Float64, s)
        return isfinite(xf) ? xf : missing
    catch
        return missing
    end
end

function maybe_int(x)
    xf = maybe_float(x)
    xf === missing && return missing
    return Int(round(xf))
end

function getval(row, name::AbstractString, default = missing)
    sym = Symbol(name)
    return sym in propertynames(row) ? row[sym] : default
end

function getdict(d::Dict, key::AbstractString, default = missing)
    return haskey(d, key) ? d[key] : default
end

function canonical_dataset(name::AbstractString)
    x = replace(strip(name), ".csv" => "")
    for suffix in ("_CIBERSORT_OS", "_CIBERSORT_survival")
        endswith(x, suffix) && (x = x[1:end - length(suffix)])
    end
    return x
end

function dataset_group(dataset_id::AbstractString)
    startswith(dataset_id, "TCGA_") && return "TCGA"
    startswith(dataset_id, "HCT") && return "HCT"
    return dataset_id
end

function split_csv_string(x)
    s = strip(as_string(x))
    isempty(s) && return String[]
    return [String(strip(part)) for part in split(s, ',') if !isempty(strip(part))]
end

function survival_columns(metadata::Dict{String,Any}, input::AbstractString)
    time_col = strip(as_string(getdict(metadata, "time_col", "")))
    status_col = strip(as_string(getdict(metadata, "status_col", "")))
    if !isempty(time_col) && !isempty(status_col)
        return time_col, status_col
    end
    return occursin("/TCGA/", replace(input, "\\" => "/")) ? ("os_time", "os") : ("time", "status")
end

function selected_count(path::AbstractString)
    df = read_table(path)
    nrow(df) == 0 && return missing
    total = 0
    for row in eachrow(df)
        sel = lowercase(strip(as_string(getval(row, "selected", ""))))
        if sel == "true"
            total += 1
            continue
        end
        lo = maybe_float(getval(row, "lower", missing))
        hi = maybe_float(getval(row, "upper", missing))
        if lo !== missing && hi !== missing && lo * hi > 0
            total += 1
        end
    end
    return total
end

function beta_columns(df::DataFrame)
    cols = [String(c) for c in names(df) if startswith(String(c), "beta[")]
    sort!(cols; by = c -> parse(Int, replace(replace(c, "beta[" => ""), "]" => "")))
    return cols
end

function rhat_tables(samples_path::AbstractString, component_names::Vector{String})
    samples = read_table(samples_path)
    nrow(samples) == 0 && return DataFrame(), (missing, missing, missing, missing, missing, missing)
    "chain" in names(samples) || return DataFrame(), (missing, missing, missing, missing, missing, missing)
    bcols = beta_columns(samples)
    chains = sort(unique(Int.(samples[!, :chain])))
    length(chains) < 2 && return DataFrame(), (missing, missing, missing, missing, missing, missing)

    rows = DataFrame(
        parameter = String[],
        variable = String[],
        rhat = Float64[],
        n_chains = Int[],
        n_per_chain_min = Int[],
        n_per_chain_max = Int[],
    )
    min_unique = typemax(Int)
    for (j, col) in enumerate(bcols)
        vals_by_chain = Vector{Vector{Float64}}()
        for chain in chains
            vals = Float64.(samples[samples[!, :chain] .== chain, Symbol(col)])
            push!(vals_by_chain, vals)
        end
        nmin = minimum(length.(vals_by_chain))
        nmax = maximum(length.(vals_by_chain))
        nmin < 2 && continue
        vals_by_chain = [v[1:nmin] for v in vals_by_chain]
        uniqs = [length(unique(round.(v; digits = 12))) for v in vals_by_chain]
        min_unique = min(min_unique, minimum(uniqs))
        means = [mean(v) for v in vals_by_chain]
        vars = [var(v; corrected = true) for v in vals_by_chain]
        w = mean(vars)
        rhat = if w <= 0
            Inf
        else
            m = length(vals_by_chain)
            mean_all = mean(means)
            b = nmin * sum((mu - mean_all)^2 for mu in means) / (m - 1)
            var_hat = (nmin - 1) / nmin * w + b / nmin
            sqrt(var_hat / w)
        end
        variable = j <= length(component_names) ? component_names[j] : ""
        push!(rows, (col, variable, rhat, length(chains), nmin, nmax))
    end
    if nrow(rows) == 0
        return rows, (missing, missing, missing, missing, missing, missing)
    end
    finite_rhat = [x for x in rows.rhat if isfinite(x)]
    nfinite = length(finite_rhat)
    rhat_median = isempty(finite_rhat) ? missing : median(finite_rhat)
    rhat_p95 = isempty(finite_rhat) ? missing : quantile(finite_rhat, 0.95)
    rhat_max = isempty(finite_rhat) ? missing : maximum(finite_rhat)
    return rows, (
        nrow(rows),
        nfinite,
        rhat_median,
        rhat_p95,
        rhat_max,
        min_unique == typemax(Int) ? missing : min_unique,
    )
end

function beta_mean_from_samples(samples_path::AbstractString)
    samples = read_table(samples_path)
    nrow(samples) == 0 && return nothing
    bcols = beta_columns(samples)
    isempty(bcols) && return nothing
    return [mean(Float64.(samples[!, Symbol(col)])) for col in bcols]
end

function prepared_data(metadata::Dict{String,Any})
    input = as_string(getdict(metadata, "input", ""))
    isempty(input) && return nothing
    input_path = isabspath(input) ? input : joinpath(ROOT, input)
    isfile(input_path) || return nothing
    df = CSV.read(input_path, DataFrame; normalizenames = false)
    time_col, status_col = survival_columns(metadata, input_path)
    !(time_col in names(df)) && return nothing
    !(status_col in names(df)) && return nothing
    keep = .!ismissing.(df[!, time_col]) .& .!ismissing.(df[!, status_col])
    df = df[keep, :]
    component_cols = split_csv_string(getdict(metadata, "component_cols", ""))
    isempty(component_cols) && return nothing
    p = length(component_cols)
    x = Matrix{Float64}(undef, nrow(df), p)
    for (j, col) in enumerate(component_cols)
        !(col in names(df)) && return nothing
        vals = coalesce.(df[!, col], 0.0)
        x[:, j] .= Float64.(vals)
    end
    standardize = lowercase(as_string(getdict(metadata, "standardize", "false"))) == "true"
    time = Float64.(df[!, time_col])
    event = Int.(df[!, status_col])
    data = prepare_cox_data(x, time, event; standardize)
    return data.z, data.time, data.event
end

function concordance_index(time::Vector{Float64}, event::Vector{Int}, risk::Vector{Float64})
    concordant = 0.0
    comparable = 0.0
    n = length(time)
    for i in 1:n
        event[i] == 0 && continue
        for j in 1:n
            if time[i] < time[j]
                comparable += 1
                if risk[i] > risk[j]
                    concordant += 1
                elseif risk[i] == risk[j]
                    concordant += 0.5
                end
            end
        end
    end
    return comparable > 0 ? concordant / comparable : missing
end

function recover_apparent_cindex(run_dir::AbstractString, metadata::Dict{String,Any})
    beta = beta_mean_from_samples(joinpath(run_dir, "samples_wide.csv"))
    beta === nothing && return missing
    data = prepared_data(metadata)
    data === nothing && return missing
    z, time, event = data
    size(z, 2) == length(beta) || return missing
    risk = z * beta
    return concordance_index(time, event, vec(risk))
end

function tuning_candidates(
    tuning_root::AbstractString,
    fallback_root::AbstractString,
    constraint::AbstractString,
    dataset_name::AbstractString;
    prior_config_source::AbstractString = "",
)
    candidates = String[]
    if !isempty(prior_config_source) && prior_config_source != "none" && isfile(prior_config_source)
        push!(candidates, joinpath(dirname(prior_config_source), "prior_tuning_summary.csv"))
    end
    constraints = constraint == "soft_center" ? ["soft_center"] : [constraint, "soft_center"]
    for root in (tuning_root, fallback_root)
        isempty(root) && continue
        for con in constraints
            for ds in (dataset_name, canonical_dataset(dataset_name))
                push!(candidates, joinpath(root, con, ds, "tuning", "prior_tuning_summary.csv"))
                push!(candidates, joinpath(root, con, ds * "_CIBERSORT_OS", "tuning", "prior_tuning_summary.csv"))
            end
        end
    end
    for con in constraints
        for ds in (dataset_name, canonical_dataset(dataset_name))
            push!(candidates, joinpath(DEFAULT_TUNING_ROOT, con, ds, "tuning", "prior_tuning_summary.csv"))
            push!(candidates, joinpath(DEFAULT_TUNING_ROOT, con, ds * "_CIBERSORT_OS", "tuning", "prior_tuning_summary.csv"))
        end
    end
    return unique(candidates)
end

function best_cv_row(
    tuning_root::AbstractString,
    fallback_root::AbstractString,
    constraint::AbstractString,
    dataset_name::AbstractString,
    prior::AbstractString;
    prior_config_source::AbstractString = "",
)
    for path in tuning_candidates(
        tuning_root,
        fallback_root,
        constraint,
        dataset_name;
        prior_config_source = prior_config_source,
    )
        isfile(path) || continue
        df = read_table(path)
        (nrow(df) == 0 || !("prior" in names(df)) || !("CV_Cindex_mean" in names(df))) && continue
        sub = df[string.(df[!, :prior]) .== prior, :]
        nrow(sub) == 0 && continue
        best_idx = nothing
        best_key = (-Inf, Inf)
        for i in 1:nrow(sub)
            cv = maybe_float(sub[i, :CV_Cindex_mean])
            cv === missing && continue
            se = "CV_Cindex_se" in names(sub) ? maybe_float(sub[i, :CV_Cindex_se]) : missing
            key = (Float64(cv), -(se === missing ? Inf : Float64(se)))
            if key > best_key
                best_key = key
                best_idx = i
            end
        end
        best_idx === nothing && continue
        return sub[best_idx, :], path
    end
    return nothing, ""
end

function metric_row(metrics::DataFrame, prior::AbstractString)
    nrow(metrics) == 0 && return nothing
    !("prior" in names(metrics)) && return nothing
    !("estimate" in names(metrics)) && return nothing
    sub = metrics[(string.(metrics[!, :prior]) .== prior) .& (string.(metrics[!, :estimate]) .== "mean"), :]
    nrow(sub) == 0 && return nothing
    return sub[1, :]
end

function component_names_from_metadata(metadata::Dict{String,Any})
    return split_csv_string(getdict(metadata, "component_cols", ""))
end

function safe_column_name(x::AbstractString)
    s = replace(strip(x), r"[^A-Za-z0-9_]+" => "_")
    s = replace(s, r"_+" => "_")
    s = strip(s, '_')
    isempty(s) && return "value"
    occursin(r"^[0-9]", s) && (s = "x_" * s)
    return s
end

function prior_order(prior::AbstractString)
    idx = findfirst(==(string(prior)), PRIOR_ORDER)
    return idx === nothing ? 999 : idx
end

function constraint_order(constraint::AbstractString)
    return get(CONSTRAINT_ORDER, string(constraint), 99)
end

function parameter_index(parameter::AbstractString)
    m = match(r"\[(\d+)\]", parameter)
    m === nothing && return typemax(Int)
    return parse(Int, m.captures[1])
end

function write_parameter_card(path::AbstractString, row::NamedTuple, metadata::Dict{String,Any})
    ensure_dir(dirname(path))
    open(path, "w") do io
        println(io, "# Real-data run parameters")
        println(io)
        println(io, "- Dataset: $(row.dataset_id)")
        println(io, "- Dataset group: $(row.dataset_group)")
        println(io, "- Constraint: $(row.constraint)")
        println(io, "- Prior: $(row.prior)")
        println(io, "- Method: $(row.method)")
        println(io, "- Input: $(as_string(getdict(metadata, "input", "")))")
        println(io, "- n: $(row.n)")
        println(io, "- p: $(row.p)")
        println(io, "- Events: $(row.events)")
        println(io, "- p0: $(row.p0)")
        println(io, "- p0 mode: $(as_string(getdict(metadata, "p0_requested", getdict(metadata, "p0_mode", ""))))")
        println(io, "- Prior config source: $(row.prior_config_source)")
        println(io)
        println(io, "## Sampler")
        println(io)
        for key in ("sampler", "nuts_backend", "n_samples", "warmup", "thin", "chains", "multichains", "target_accept", "max_depth", "seed", "noncentered")
            println(io, "- $key: $(as_string(getdict(metadata, key, "")))")
        end
        println(io)
        println(io, "## Prior and model hyperparameters")
        println(io)
        for key in ("hs_tau_scale", "hs_local_scale", "rhs_tau_scale", "rhs_local_scale", "rhs_slab_df", "rhs_slab_s2", "rhs_pseudo_sigma_df", "rhs_pseudo_sigma_scale", "lasso_lambda", "lasso_lambda2_shape", "lasso_lambda2_rate", "sslasso_spike_scale", "sslasso_slab_scale", "sslasso_theta_a", "sslasso_theta_b", "prox_l1_lambda", "alpha_prior_shape", "alpha_prior_scale", "alpha_prior_kind", "prox_sum0_lambda", "prox_root_tol", "prox_root_maxiter", "standardize", "tie_method")
            val = as_string(getdict(metadata, key, ""))
            isempty(val) || println(io, "- $key: $val")
        end
        println(io)
        println(io, "## Performance")
        println(io)
        println(io, "- Apparent C-index: $(row.apparent_cindex)")
        println(io, "- CV C-index mean: $(row.cv_cindex)")
        println(io, "- CV C-index SE: $(row.cv_cindex_se)")
        println(io, "- Selected coefficients: $(row.n_selected)")
        println(io, "- Rhat max: $(row.rhat_max)")
        println(io, "- Converged: $(row.converged)")
        println(io, "- Pareto k max: $(row.pareto_k_max)")
        println(io, "- Pareto k bad count: $(row.pareto_k_bad)")
    end
    return path
end

function fallback_cv_row(cv_fallback::DataFrame, dataset_id::AbstractString, constraint::AbstractString, prior::AbstractString)
    nrow(cv_fallback) == 0 && return nothing
    required = ("dataset_id", "constraint", "prior", "CV_Cindex_mean", "evaluation", "status")
    all(name -> name in names(cv_fallback), required) || return nothing
    sub = cv_fallback[
        (string.(cv_fallback[!, :dataset_id]) .== dataset_id) .&
        (string.(cv_fallback[!, :constraint]) .== constraint) .&
        (string.(cv_fallback[!, :prior]) .== prior) .&
        (string.(cv_fallback[!, :evaluation]) .== "nested_cv") .&
        (string.(cv_fallback[!, :status]) .== "success"),
        :,
    ]
    nrow(sub) == 0 && return nothing
    return sub[1, :]
end

function nested_cv_row(dataset_dir, tuning_root, fallback_root, constraint, dataset_name, prior)
    for root in unique([dataset_dir, joinpath(tuning_root, constraint, dataset_name),
                       joinpath(fallback_root, constraint, dataset_name)])
        path = joinpath(root, "nested_cv", "nested_cv_summary.csv")
        table = read_table(path)
        row = fallback_cv_row(table, canonical_dataset(dataset_name), constraint, prior)
        # Some dataset IDs retain the full input stem (e.g. TCGA_*_CIBERSORT_OS).
        row === nothing && (row = fallback_cv_row(table, dataset_name, constraint, prior))
        row === nothing || return row, path
    end
    return nothing, ""
end

function run_summary(;
    realdata_root::String,
    tuning_root::String,
    fallback_tuning_root::String,
    outdir::String,
    write_cards::Bool,
    cv_fallback_path::String = "",
)
    ensure_dir(outdir)
    cv_fallback = (!isempty(cv_fallback_path) && isfile(cv_fallback_path)) ? read_table(cv_fallback_path) : DataFrame()
    rhat_by_param = DataFrame()
    rhat_summary = DataFrame()
    rows = NamedTuple[]

    for constraint_dir in sort(filter(isdir, readdir(realdata_root; join = true)))
        constraint = basename(constraint_dir)
        for dataset_dir in sort(filter(isdir, readdir(constraint_dir; join = true)))
            dataset_name = basename(dataset_dir)
            dataset_id = canonical_dataset(dataset_name)
            group = dataset_group(dataset_id)
            metrics = read_table(joinpath(dataset_dir, "tables", "model_metrics.csv"))
            runs_dir = joinpath(dataset_dir, "runs")
            isdir(runs_dir) || continue
            for run_dir in sort(filter(isdir, readdir(runs_dir; join = true)))
                startswith(basename(run_dir), "realdata_") || continue
                prior = replace(basename(run_dir), "realdata_" => "")
                metadata = read_json(joinpath(run_dir, "metadata.json"))
                component_names = component_names_from_metadata(metadata)
                mrow = metric_row(metrics, prior)
                cindex = mrow === nothing ? missing : maybe_float(getval(mrow, "Cindex", missing))
                if cindex === missing
                    cindex = recover_apparent_cindex(run_dir, metadata)
                end
                prior_config_source = as_string(getdict(metadata, "prior_config_source", ""))
                tuningrow, tuningsource = best_cv_row(
                    tuning_root,
                    fallback_tuning_root,
                    constraint,
                    dataset_name,
                    prior;
                    prior_config_source = prior_config_source,
                )
                cvrow, cvsource = nested_cv_row(dataset_dir, tuning_root, fallback_tuning_root,
                    constraint, dataset_name, prior)
                if cvrow === nothing
                    cvrow = fallback_cv_row(cv_fallback, dataset_id, constraint, prior)
                    cvrow === nothing || (cvsource = cv_fallback_path)
                end
                coef_path = joinpath(dataset_dir, "tables", "coef_$prior.csv")
                n_selected = selected_count(coef_path)
                rhats, rstats = rhat_tables(joinpath(run_dir, "samples_wide.csv"), component_names)
                n_parameters, n_rhat_finite, rhat_median, rhat_p95, rhat_max, min_unique = rstats
                converged = rhat_max !== missing && Float64(rhat_max) <= 1.10 && (min_unique === missing || Int(min_unique) >= 20)
                row = (
                    dataset_id = dataset_id,
                    dataset_name = dataset_name,
                    dataset_group = group,
                    constraint = constraint,
                    method_family = constraint == "prox" ? "Bayes-prox" : "Bayes-soft-center",
                    method = "Bayes_$(constraint)_$(prior)",
                    prior = prior,
                    row_type = "bayes_final",
                    apparent_cindex = cindex,
                    cv_evaluation = cvrow === nothing ? "unavailable" : "nested_cv",
                    tuning_cv_cindex = tuningrow === nothing ? missing : maybe_float(getval(tuningrow, "CV_Cindex_mean", missing)),
                    tuning_cv_source = tuningsource,
                    cv_cindex = cvrow === nothing ? missing : maybe_float(getval(cvrow, "CV_Cindex_mean", missing)),
                    cv_cindex_se = cvrow === nothing ? missing : maybe_float(getval(cvrow, "CV_Cindex_se", missing)),
                    cv_cindex_sd = cvrow === nothing ? missing : maybe_float(getval(cvrow, "CV_Cindex_sd", missing)),
                    cv_cindex_map = cvrow === nothing ? missing : maybe_float(getval(cvrow, "CV_Cindex_MAP_mean", missing)),
                    cv_cindex_map_se = cvrow === nothing ? missing : maybe_float(getval(cvrow, "CV_Cindex_MAP_se", missing)),
                    cv_cindex_median = cvrow === nothing ? missing : maybe_float(getval(cvrow, "CV_Cindex_median_mean", missing)),
                    cv_cindex_median_se = cvrow === nothing ? missing : maybe_float(getval(cvrow, "CV_Cindex_median_se", missing)),
                    cv_source = cvsource,
                    n_selected = n_selected,
                    selected_features = "",
                    fit_success = true,
                    converged = converged,
                    rhat_median = rhat_median,
                    rhat_p95 = rhat_p95,
                    rhat_max = rhat_max,
                    n_parameters = n_parameters,
                    n_rhat_finite = n_rhat_finite,
                    n_rhat_gt_1_01 = nrow(rhats) == 0 ? missing : count(>(1.01), rhats.rhat),
                    n_rhat_gt_1_05 = nrow(rhats) == 0 ? missing : count(>(1.05), rhats.rhat),
                    n_rhat_gt_1_10 = nrow(rhats) == 0 ? missing : count(>(1.10), rhats.rhat),
                    min_unique_per_chain = min_unique,
                    p0 = maybe_float(getdict(metadata, "p0", missing)),
                    p0_requested = as_string(getdict(metadata, "p0_requested", getdict(metadata, "p0_mode", ""))),
                    n = maybe_int(getdict(metadata, "n", missing)),
                    p = maybe_int(getdict(metadata, "p", missing)),
                    events = maybe_int(getdict(metadata, "number_of_events", missing)),
                    apparent_cindex_se = missing,
                    waic_pw1 = mrow === nothing ? missing : maybe_float(getval(mrow, "WAIC_pw1", missing)),
                    waic_pw2 = mrow === nothing ? missing : maybe_float(getval(mrow, "WAIC_pw2", missing)),
                    lpml = mrow === nothing ? missing : maybe_float(getval(mrow, "LPML", missing)),
                    elpd_loo = mrow === nothing ? missing : maybe_float(getval(mrow, "ELPD_LOO", missing)),
                    elpd_loo_se = mrow === nothing ? missing : maybe_float(getval(mrow, "ELPD_LOO_SE", missing)),
                    looic = mrow === nothing ? missing : maybe_float(getval(mrow, "LOOIC", missing)),
                    looic_se = mrow === nothing ? missing : maybe_float(getval(mrow, "LOOIC_SE", missing)),
                    pareto_k_max = mrow === nothing ? missing : maybe_float(getval(mrow, "ParetoK_max", missing)),
                    pareto_k_bad = mrow === nothing ? missing : maybe_int(getval(mrow, "ParetoK_bad", missing)),
                    prior_config_source = prior_config_source,
                    run_dir = run_dir,
                    coef_path = coef_path,
                    sample_file = joinpath(run_dir, "samples_wide.csv"),
                )
                push!(rows, row)

                if nrow(rhats) > 0
                    rhats.dataset_id .= dataset_id
                    rhats.dataset_group .= group
                    rhats.constraint .= constraint
                    rhats.prior .= prior
                    rhats.sample_file .= joinpath(run_dir, "samples_wide.csv")
                    append!(rhat_by_param, rhats; cols = :union)
                end
                push!(rhat_summary, (
                    dataset_id = dataset_id,
                    dataset_group = group,
                    constraint = constraint,
                    prior = prior,
                    n_parameters = n_parameters,
                    n_rhat_finite = n_rhat_finite,
                    rhat_median = rhat_median,
                    rhat_p95 = rhat_p95,
                    rhat_max = rhat_max,
                    n_rhat_gt_1_01 = row.n_rhat_gt_1_01,
                    n_rhat_gt_1_05 = row.n_rhat_gt_1_05,
                    n_rhat_gt_1_10 = row.n_rhat_gt_1_10,
                    sample_file = joinpath(run_dir, "samples_wide.csv"),
                ); cols = :union)
                if write_cards
                    card_path = joinpath(run_dir, "parameters.md")
                    write_parameter_card(card_path, row, metadata)
                    write_parameter_card(joinpath(dataset_dir, "tables", "parameters_$prior.md"), row, metadata)
                end
            end
        end
    end

    all_methods = DataFrame(rows)
    if nrow(all_methods) > 0
        all_methods._group_order = [get(GROUP_ORDER, string(g), 99) for g in all_methods.dataset_group]
        all_methods._prior_order = [findfirst(==(string(p)), PRIOR_ORDER) === nothing ? 999 : findfirst(==(string(p)), PRIOR_ORDER) for p in all_methods.prior]
        sort!(all_methods, [:_group_order, :dataset_id, :constraint, :_prior_order])
        select!(all_methods, Not([:_group_order, :_prior_order]))
    end
    write_table(joinpath(outdir, "bayes_all_methods.csv"), all_methods)
    write_table(joinpath(outdir, "all_methods_performance.csv"), all_methods)
    write_table(joinpath(outdir, "bayes_rhat_by_parameter.csv"), rhat_by_param)
    write_table(joinpath(outdir, "bayes_rhat_summary.csv"), rhat_summary)
    write_rhat_wide_tables(outdir, rhat_by_param)

    for group in unique(string.(all_methods.dataset_group))
        gdir = ensure_dir(joinpath(outdir, "by_group", group))
        sub = all_methods[string.(all_methods.dataset_group) .== group, :]
        write_table(joinpath(gdir, "bayes_all_methods.csv"), sub)
        write_table(joinpath(gdir, "all_methods_performance.csv"), sub)
    end
    return all_methods
end

function finite_col(df::DataFrame, col::Symbol)
    !(col in propertynames(df)) && return Float64[]
    vals = Float64[]
    for v in df[!, col]
        vf = maybe_float(v)
        vf !== missing && push!(vals, vf)
    end
    return vals
end

function write_rhat_wide_tables(outdir::AbstractString, rhat_by_param::DataFrame)
    nrow(rhat_by_param) == 0 && return
    required = ("dataset_id", "dataset_group", "constraint", "prior", "variable", "parameter", "rhat")
    all(name -> name in names(rhat_by_param), required) || return
    root = ensure_dir(joinpath(outdir, "rhat_by_dataset"))
    for dataset_id in sort(unique(string.(rhat_by_param.dataset_id)))
        sub = rhat_by_param[string.(rhat_by_param.dataset_id) .== dataset_id, :]
        nrow(sub) == 0 && continue
        var_meta = Dict{String,Tuple{Int,String}}()
        for row in eachrow(sub)
            variable = strip(as_string(row.variable))
            isempty(variable) && (variable = as_string(row.parameter))
            idx = parameter_index(as_string(row.parameter))
            if !haskey(var_meta, variable) || idx < var_meta[variable][1]
                var_meta[variable] = (idx, variable)
            end
        end
        variables = [v for (_, v) in sort(collect(values(var_meta)); by = x -> x[1])]
        table = DataFrame(variable = variables)

        configs = unique(sub[:, [:constraint, :prior]])
        configs._constraint_order = [constraint_order(as_string(c)) for c in configs.constraint]
        configs._prior_order = [prior_order(as_string(p)) for p in configs.prior]
        sort!(configs, [:_constraint_order, :_prior_order])
        select!(configs, Not([:_constraint_order, :_prior_order]))

        for cfg in eachrow(configs)
            constraint = as_string(cfg.constraint)
            prior = as_string(cfg.prior)
            col = Symbol(safe_column_name("$(constraint)_$(prior)_rhat"))
            vals = Dict{String,Union{Missing,Float64}}()
            rows = sub[(string.(sub.constraint) .== constraint) .& (string.(sub.prior) .== prior), :]
            for row in eachrow(rows)
                variable = strip(as_string(row.variable))
                isempty(variable) && (variable = as_string(row.parameter))
                vals[variable] = maybe_float(row.rhat)
            end
            table[!, col] = [get(vals, v, missing) for v in variables]
        end

        path = write_table(joinpath(root, "$(safe_column_name(dataset_id))_rhat.csv"), table)
        group = as_string(sub[1, :dataset_group])
        if !isempty(group)
            gdir = ensure_dir(joinpath(outdir, "by_group", group, "rhat_by_dataset"))
            write_table(joinpath(gdir, basename(path)), table)
        end
    end
end

function init_plot_backend()
    try
        gr()
    catch
    end
    default(
        fontfamily = PLOT_FONT,
        guidefontsize = 18,
        tickfontsize = 13,
        legendfontsize = 12,
        titlefontsize = 16,
        linewidth = 2,
        grid = false,
        framestyle = :box,
    )
end

function write_bayes_plots(outdir::AbstractString, df::DataFrame, rhat_summary::DataFrame)
    nrow(df) == 0 && return
    figdir = ensure_dir(joinpath(outdir, "figures"))
    init_plot_backend()
    for group in ("T1D", "HCT", "TCGA")
        gdf = df[string.(df.dataset_group) .== group, :]
        nrow(gdf) == 0 && continue
        gfig = ensure_dir(joinpath(outdir, "by_group", group, "figures"))
        plot_cindex(joinpath(gfig, "cv_cindex_comparison"), gdf, :cv_cindex)
        plot_apparent_vs_cv(joinpath(gfig, "apparent_vs_cv_cindex"), gdf)
        plot_selected_counts(joinpath(gfig, "selected_features_count"), gdf)
        grhat = rhat_summary[string.(rhat_summary.dataset_group) .== group, :]
        plot_rhat(joinpath(gfig, "rhat_max"), grhat)
    end
    plot_cindex(joinpath(figdir, "cv_cindex_comparison"), df, :cv_cindex)
    plot_apparent_vs_cv(joinpath(figdir, "apparent_vs_cv_cindex"), df)
    plot_selected_counts(joinpath(figdir, "selected_features_count"), df)
    plot_rhat(joinpath(figdir, "rhat_max"), rhat_summary)
end

function save_plot_bundle(p, pathroot::AbstractString)
    savefig(p, pathroot * ".pdf")
    ps_path = pathroot * ".ps"
    savefig(p, ps_path)
    cp(ps_path, pathroot * ".eps"; force = true)
    rm(ps_path; force = true)
    savefig(p, pathroot * ".png")
end

function method_label(row)
    return "$(as_string(row.constraint)) / $(as_string(row.prior))"
end

function method_positions(df::DataFrame)
    labels = DataFrame(label = String[], constraint = String[], prior = String[])
    seen = Set{String}()
    for row in eachrow(df)
        label = method_label(row)
        label in seen && continue
        push!(seen, label)
        push!(labels, (label, as_string(row.constraint), as_string(row.prior)))
    end
    labels._constraint_order = [constraint_order(c) for c in labels.constraint]
    labels._prior_order = [prior_order(p) for p in labels.prior]
    sort!(labels, [:_constraint_order, :_prior_order])
    return labels.label, Dict(label => i for (i, label) in enumerate(labels.label))
end

function dataset_offsets(datasets::Vector{String})
    n = length(datasets)
    n <= 1 && return Dict(datasets[1] => 0.0)
    span = min(0.56, 0.08 * (n - 1))
    vals = range(-span / 2, span / 2; length = n)
    return Dict(ds => Float64(v) for (ds, v) in zip(datasets, vals))
end

function plot_metric_by_method(pathroot::AbstractString, df::DataFrame, col::Symbol, xlabel::AbstractString; xlims = :auto, reference = Float64[])
    sub = df[[maybe_float(v) !== missing for v in df[!, col]], :]
    nrow(sub) == 0 && return
    labels, pos = method_positions(sub)
    datasets = sort(unique(string.(sub.dataset_id)))
    offsets = dataset_offsets(datasets)
    height = max(900, 42 * length(labels) + 240)
    p = plot(;
        xlabel = xlabel,
        ylabel = "",
        xlims = xlims,
        yticks = (collect(1:length(labels)), labels),
        legend = :outerright,
        size = (1500, height),
        left_margin = 26Plots.mm,
        right_margin = 16Plots.mm,
        bottom_margin = 12Plots.mm,
        top_margin = 6Plots.mm,
    )
    for dataset in datasets
        rows = sub[string.(sub.dataset_id) .== dataset, :]
        nrow(rows) == 0 && continue
        x = Float64.(rows[!, col])
        y = [pos[method_label(row)] + offsets[dataset] for row in eachrow(rows)]
        scatter!(p, x, y; label = dataset, markersize = 6)
    end
    for x0 in reference
        vline!(p, [x0]; label = "", color = :gray, linestyle = :dash, linewidth = 1.5)
    end
    save_plot_bundle(p, pathroot)
end

function plot_cindex(pathroot::AbstractString, df::DataFrame, col::Symbol)
    plot_metric_by_method(pathroot, df, col, "CV C-index"; xlims = (0, 1))
end

function plot_apparent_vs_cv(pathroot::AbstractString, df::DataFrame)
    keep = [maybe_float(a) !== missing && maybe_float(c) !== missing for (a, c) in zip(df.apparent_cindex, df.cv_cindex)]
    sub = df[keep, :]
    nrow(sub) == 0 && return
    x = Float64.(sub.apparent_cindex)
    y = Float64.(sub.cv_cindex)
    groups = string.(sub.dataset_id)
    p = scatter(x, y;
        group = groups,
        xlabel = "Apparent C-index",
        ylabel = "CV C-index",
        xlims = (0, 1),
        ylims = (0, 1),
        legend = :bottomright,
        size = (1000, 900),
        markersize = 6,
        left_margin = 12Plots.mm,
        bottom_margin = 12Plots.mm,
    )
    plot!(p, [0, 1], [0, 1]; label = "", color = :gray, linestyle = :dash)
    save_plot_bundle(p, pathroot)
end

function plot_selected_counts(pathroot::AbstractString, df::DataFrame)
    plot_metric_by_method(pathroot, df, :n_selected, "Selected coefficients")
end

function plot_rhat(pathroot::AbstractString, df::DataFrame)
    (nrow(df) == 0 || !(:rhat_max in propertynames(df))) && return
    plot_metric_by_method(pathroot, df, :rhat_max, "max Rhat"; reference = [1.10])
end

function write_readme(outdir::AbstractString, realdata_root::AbstractString, tuning_root::AbstractString, fallback_tuning_root::AbstractString, nrows::Integer)
    open(joinpath(outdir, "README.md"), "w") do io
        println(io, "# Bayesian Real-data Summary")
        println(io)
        println(io, "- Generated at: $(Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"))")
        println(io, "- Real-data root: $realdata_root")
        println(io, "- Tuning root: $tuning_root")
        println(io, "- Fallback tuning root: $fallback_tuning_root")
        println(io, "- Rows: $nrows")
        println(io)
        println(io, "The apparent C-index is computed on the full fitted dataset. CV-Cindex uses only completed nested outer-test evaluations. Historical tuning CV scores are retained separately as tuning_cv_cindex; missing nested results are never replaced by tuning scores.")
        println(io, "Rows are not filtered to only the best model; every dataset, constraint, and prior run is retained.")
    end
end

function main()
    opts = parse_args(ARGS)
    realdata_root = normpath(opt(opts, "root", DEFAULT_REALDATA_ROOT))
    tuning_root = normpath(opt(opts, "tuning-root", DEFAULT_TUNING_ROOT))
    fallback_tuning_root = normpath(opt(opts, "fallback-tuning-root", DEFAULT_REALDATA_ROOT))
    outdir = normpath(opt(opts, "outdir", DEFAULT_SUMMARY_DIR))
    cv_fallback_path = normpath(opt(opts, "cv-fallback", DEFAULT_CV_FALLBACK))
    write_cards = lowercase(opt(opts, "write-parameter-cards", "false")) == "true"
    run_summary(
        realdata_root = realdata_root,
        tuning_root = tuning_root,
        fallback_tuning_root = fallback_tuning_root,
        outdir = outdir,
        write_cards = write_cards,
        cv_fallback_path = cv_fallback_path,
    )
    println(outdir)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
