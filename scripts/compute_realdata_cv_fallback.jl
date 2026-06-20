#!/usr/bin/env julia

using Pkg

function _activate_repo_project()
    project = abspath(joinpath(@__DIR__, "..", "Project.toml"))
    Base.active_project() == project || Pkg.activate(dirname(project))
end

_activate_repo_project()

using CSV
using DataFrames
using JSON
using Statistics

include(joinpath(@__DIR__, "analysis_common.jl"))
include(joinpath(@__DIR__, "tune_hyperparams_cindex.jl"))

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_BAYES_SUMMARY = joinpath(ROOT, "results", "summary", "realdata", "bayes_all_methods", "bayes_all_methods.csv")
const DEFAULT_OUTDIR = joinpath(ROOT, "results", "summary", "realdata", "bayes_cv_fallback")

const PRIOR_KEYS = [
    "hs_tau_scale", "hs_local_scale", "rhs_tau_scale", "rhs_local_scale",
    "rhs_slab_df", "rhs_slab_s2", "rhs_pseudo_sigma_df", "rhs_pseudo_sigma_scale",
    "lasso_lambda", "lasso_lambda2_shape", "lasso_lambda2_rate",
    "sslasso_spike_scale", "sslasso_slab_scale", "sslasso_theta_a", "sslasso_theta_b",
    "prox_l1_lambda", "alpha_prior_shape", "alpha_prior_scale", "alpha_prior_kind",
    "prox_sum0_lambda", "prox_root_tol", "prox_root_maxiter",
]
const INTEGER_PRIOR_KEYS = Set(["prox_root_maxiter"])

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

opt(opts, name, default) = get(opts, name, default)

function maybe_float(x)
    x === missing && return missing
    x === nothing && return missing
    x isa Number && return Float64(x)
    s = strip(string(x))
    isempty(s) && return missing
    lowercase(s) in ("missing", "nothing", "nan", "na") && return missing
    try
        return parse(Float64, s)
    catch
        return missing
    end
end

function maybe_int(x)
    v = maybe_float(x)
    v === missing && return missing
    return Int(round(v))
end

function maybe_bool(x)
    x isa Bool && return x
    s = lowercase(strip(string(x)))
    return s in ("1", "true", "yes", "y")
end

function split_csv_string(x)
    s = strip(string(x))
    isempty(s) && return String[]
    return [strip(part) for part in split(s, ',') if !isempty(strip(part))]
end

function read_json(path::AbstractString)
    isfile(path) || throw(ArgumentError("metadata not found: $path"))
    return JSON.parsefile(path)
end

function input_path_from_metadata(meta)
    input = string(meta["input"])
    return isabspath(input) ? input : abspath(joinpath(ROOT, input))
end

function drop_missing_survival_rows_local(df::DataFrame, time_col::Symbol, status_col::Symbol)
    mask = .!ismissing.(df[!, time_col]) .& .!ismissing.(df[!, status_col])
    return df[mask, :], count(.!mask)
end

function realdata_arrays(meta)
    input = input_path_from_metadata(meta)
    df = CSV.read(input, DataFrame)
    time_col = Symbol(get(meta, "time_col", "time"))
    status_col = Symbol(get(meta, "status_col", "status"))
    df, _ = drop_missing_survival_rows_local(df, time_col, status_col)
    component_cols = split_csv_string(get(meta, "component_cols", ""))
    isempty(component_cols) && throw(ArgumentError("metadata has no component_cols for $input"))
    X = Matrix{Float64}(coalesce.(Matrix(df[:, component_cols]), 0.0))
    time = Float64.(df[!, time_col])
    event = Int.(df[!, status_col])
    return X, time, event
end

function metadata_kwargs(meta; samples::Int, warmup::Int, chains::Int, seed::Int)
    kwargs = Dict{Symbol,Any}(
        :prior_name => string(meta["prior_name"]),
        :sampler => Symbol(get(meta, "sampler", "nuts")),
        :nuts_backend => Symbol(get(meta, "nuts_backend", "explicit")),
        :n_samples => samples,
        :warmup => warmup,
        :thin => Int(get(meta, "thin", 1)),
        :chains => chains,
        :multichains => maybe_bool(get(meta, "multichains", false)),
        :target_accept => Float64(get(meta, "target_accept", 0.9)),
        :max_depth => Int(get(meta, "max_depth", 10)),
        :hmc_n_leapfrog => Int(get(meta, "hmc_n_leapfrog", 10)),
        :hmc_step_size => get(meta, "hmc_step_size", nothing),
        :noncentered => maybe_bool(get(meta, "noncentered", false)),
        :seed => seed,
        :use_car => maybe_bool(get(meta, "use_car", false)),
        :pseudocount => Float64(get(meta, "pseudocount", 1e-6)),
        :standardize => maybe_bool(get(meta, "standardize", false)),
        :tie_method => Symbol(get(meta, "tie_method", "efron")),
        :p0 => get(meta, "p0", nothing),
        :p0_alpha => Float64(get(meta, "p0_alpha", 0.05)),
        :constraint => Symbol(get(meta, "constraint", "soft_center")),
        :prox_sum0_lambda => Float64(get(meta, "prox_sum0_lambda", 1e-6)),
        :progress => false,
        :approx => false,
        :approx_point => :zero,
        :realdata => true,
    )
    for key in PRIOR_KEYS
        haskey(meta, key) || continue
        val = meta[key]
        val === nothing && continue
        if key in ("alpha_prior_kind",)
            kwargs[Symbol(key)] = Symbol(val)
        elseif key in INTEGER_PRIOR_KEYS
            parsed = maybe_int(val)
            parsed === missing || (kwargs[Symbol(key)] = parsed)
        else
            parsed = maybe_float(val)
            parsed === missing || (kwargs[Symbol(key)] = parsed)
        end
    end
    return kwargs
end

function finite_values(rows, key)
    vals = Float64[]
    for row in rows
        val = get(row, key, missing)
        val === missing && continue
        isfinite(Float64(val)) && push!(vals, Float64(val))
    end
    return vals
end

mean_or_missing(vals) = isempty(vals) ? missing : mean(vals)
sd_or_missing(vals) = length(vals) <= 1 ? missing : std(vals)
se_or_missing(vals) = length(vals) <= 1 ? missing : std(vals) / sqrt(length(vals))

function summarize_cv(rows; row, samples, warmup, chains, cv_folds)
    mean_vals = finite_values(rows, :validation_Cindex_mean)
    median_vals = finite_values(rows, :validation_Cindex_median)
    map_vals = finite_values(rows, :validation_Cindex_MAP)
    return Dict{Symbol,Any}(
        :dataset_id => row.dataset_id,
        :dataset_name => row.dataset_name,
        :dataset_group => row.dataset_group,
        :constraint => row.constraint,
        :prior => row.prior,
        :n_folds => cv_folds,
        :n_successful_folds => length(mean_vals),
        :CV_Cindex_mean => mean_or_missing(mean_vals),
        :CV_Cindex_sd => sd_or_missing(mean_vals),
        :CV_Cindex_se => se_or_missing(mean_vals),
        :CV_Cindex_median_mean => mean_or_missing(median_vals),
        :CV_Cindex_median_se => se_or_missing(median_vals),
        :CV_Cindex_MAP_mean => mean_or_missing(map_vals),
        :CV_Cindex_MAP_se => se_or_missing(map_vals),
        :cv_samples => samples,
        :cv_warmup => warmup,
        :cv_chains => chains,
        :source => "fixed_final_config_cv",
        :run_dir => row.run_dir,
    )
end

function write_table(path, rows)
    mkpath(dirname(path))
    CSV.write(path, DataFrame(rows))
end

function main()
    opts = parse_args(ARGS)
    bayes_summary = normpath(opt(opts, "bayes-summary", DEFAULT_BAYES_SUMMARY))
    outdir = normpath(opt(opts, "outdir", DEFAULT_OUTDIR))
    samples = parse(Int, opt(opts, "samples", "400"))
    warmup = parse(Int, opt(opts, "warmup", "400"))
    chains = parse(Int, opt(opts, "chains", "2"))
    cv_folds = parse(Int, opt(opts, "cv-folds", "5"))
    seed = parse(Int, opt(opts, "seed", "20260618"))
    limit = parse(Int, opt(opts, "limit", "0"))
    force = lowercase(opt(opts, "force", "false")) in ("1", "true", "yes", "y")

    mkpath(outdir)
    summary_path = joinpath(outdir, "cv_fallback_summary.csv")
    folds_path = joinpath(outdir, "cv_fallback_folds.csv")
    existing = isfile(summary_path) ? CSV.read(summary_path, DataFrame) : DataFrame()
    done = Set{Tuple{String,String,String}}()
    if !force && nrow(existing) > 0
        for r in eachrow(existing)
            push!(done, (string(r.dataset_id), string(r.constraint), string(r.prior)))
        end
    end

    bayes = CSV.read(bayes_summary, DataFrame)
    targets = bayes[ismissing.(bayes.cv_cindex), :]
    limit > 0 && nrow(targets) > limit && (targets = first(targets, limit))

    summary_rows = nrow(existing) > 0 && !force ? [Dict(Symbol(c) => r[c] for c in names(existing)) for r in eachrow(existing)] : Dict{Symbol,Any}[]
    fold_rows = isfile(folds_path) && !force ? [Dict(Symbol(c) => r[c] for c in names(CSV.read(folds_path, DataFrame))) for r in eachrow(CSV.read(folds_path, DataFrame))] : Dict{Symbol,Any}[]

    for row in eachrow(targets)
        key = (string(row.dataset_id), string(row.constraint), string(row.prior))
        key in done && continue
        @info "Computing fixed-config CV fallback." dataset=row.dataset_id constraint=row.constraint prior=row.prior
        meta = read_json(joinpath(string(row.run_dir), "metadata.json"))
        X, time, event = realdata_arrays(meta)
        k = min(cv_folds, length(time))
        folds = _kfold_indices(event; k = k, seed = seed)
        local_rows = Dict{Symbol,Any}[]
        for (fold, val_idx) in enumerate(folds)
            train_idx = setdiff(1:length(time), val_idx)
            kwargs = metadata_kwargs(meta; samples = samples, warmup = warmup, chains = chains, seed = seed + 10_000 * fold)
            try
                fit = fit_and_evaluate(X[train_idx, :], time[train_idx], event[train_idx]; kwargs...)
                vc = _validation_cindices(
                    fit,
                    X[train_idx, :],
                    X[val_idx, :],
                    time[val_idx],
                    event[val_idx];
                    requested_pseudocount = get(kwargs, :pseudocount, nothing),
                )
                fr = Dict{Symbol,Any}(
                    :dataset_id => row.dataset_id,
                    :dataset_name => row.dataset_name,
                    :dataset_group => row.dataset_group,
                    :constraint => row.constraint,
                    :prior => row.prior,
                    :fold => fold,
                    :status => "success",
                    :validation_Cindex_mean => get(vc, "mean", missing),
                    :validation_Cindex_median => get(vc, "median", missing),
                    :validation_Cindex_MAP => get(vc, "MAP", missing),
                    :error_message => "",
                )
                push!(fold_rows, fr)
                push!(local_rows, fr)
            catch err
                fr = Dict{Symbol,Any}(
                    :dataset_id => row.dataset_id,
                    :dataset_name => row.dataset_name,
                    :dataset_group => row.dataset_group,
                    :constraint => row.constraint,
                    :prior => row.prior,
                    :fold => fold,
                    :status => "failed",
                    :validation_Cindex_mean => missing,
                    :validation_Cindex_median => missing,
                    :validation_Cindex_MAP => missing,
                    :error_message => sprint(showerror, err),
                )
                push!(fold_rows, fr)
                push!(local_rows, fr)
            end
        end
        push!(summary_rows, summarize_cv(local_rows; row = row, samples = samples, warmup = warmup, chains = chains, cv_folds = k))
        write_table(summary_path, summary_rows)
        write_table(folds_path, fold_rows)
    end
    println(summary_path)
end

main()
