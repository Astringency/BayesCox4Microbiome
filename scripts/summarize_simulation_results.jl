#!/usr/bin/env julia

using CSV
using DataFrames
using Dates
using JSON
using Statistics

const DEFAULT_ROOT = joinpath("results", "simulation")

function parse_cli_args(args = ARGS)
    out = Dict{String,String}()
    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--")
            key = arg[3:end]
            if occursin("=", key)
                k, v = split(key, "="; limit = 2)
                out[k] = v
            elseif i < length(args) && !startswith(args[i + 1], "--")
                out[key] = args[i + 1]
                i += 1
            else
                out[key] = "true"
            end
        end
        i += 1
    end
    return out
end

function split_csv(value::AbstractString)
    return [strip(x) for x in split(value, ",") if !isempty(strip(x))]
end

function csv_value(x)
    x === nothing && return missing
    x === missing && return missing
    x isa AbstractVector && return join(string.(x), ";")
    x isa Tuple && return join(string.(collect(x)), ";")
    x isa AbstractDict && return JSON.json(x)
    return x
end

function write_table(path::AbstractString, table::DataFrame)
    mkpath(dirname(path))
    CSV.write(path, table; transform = (col, val) -> csv_value(val))
    return path
end

file_complete(path::AbstractString) = isfile(path) && filesize(path) > 0

function safe_read_csv(path::AbstractString)
    file_complete(path) || return DataFrame()
    try
        return CSV.read(path, DataFrame)
    catch err
        @warn "Could not read CSV file." path exception=(err, catch_backtrace())
        return DataFrame()
    end
end

function safe_read_json(path::AbstractString)
    file_complete(path) || return Dict{String,Any}()
    try
        return Dict{String,Any}(JSON.parsefile(path))
    catch err
        @warn "Could not read JSON file." path exception=(err, catch_backtrace())
        return Dict{String,Any}()
    end
end

function maybe_number(x)
    x === missing && return missing
    x === nothing && return missing
    x isa Number && return x
    if x isa AbstractString
        s = strip(x)
        isempty(s) && return missing
        lowercase(s) in ("missing", "null", "nothing", "nan") && return missing
        try
            return parse(Int, s)
        catch
        end
        try
            return parse(Float64, s)
        catch
        end
    end
    return x
end

function get_any(dict::AbstractDict, keys...; default = missing)
    for key in keys
        skey = String(key)
        if haskey(dict, skey)
            return maybe_number(dict[skey])
        end
    end
    return default
end

function setting_from_dirname(name::AbstractString)
    m = match(r"^n(\d+)_p(\d+)_nonzero(\d+)_(.+)$", name)
    m === nothing && return Dict{String,Any}()
    return Dict{String,Any}(
        "n" => parse(Int, m.captures[1]),
        "p" => parse(Int, m.captures[2]),
        "nonzero_count" => parse(Int, m.captures[3]),
        "baseline" => m.captures[4],
    )
end

function run_from_dirname(name::AbstractString)
    m = match(r"^rep(\d+)_(.+)_([^_]+)_n(\d+)_p(\d+)_nz(\d+)$", name)
    m === nothing && return Dict{String,Any}()
    return Dict{String,Any}(
        "rep" => parse(Int, m.captures[1]),
        "prior" => m.captures[2],
        "baseline" => m.captures[3],
        "n" => parse(Int, m.captures[4]),
        "p" => parse(Int, m.captures[5]),
        "nonzero_count" => parse(Int, m.captures[6]),
    )
end

function run_context(root::AbstractString, run_dir::AbstractString)
    meta = safe_read_json(joinpath(run_dir, "metadata.json"))
    parsed_run = run_from_dirname(basename(run_dir))
    setting_dir = basename(dirname(dirname(run_dir)))
    parsed_setting = setting_from_dirname(setting_dir)

    n_value = get_any(meta, "n"; default = get(parsed_run, "n", get(parsed_setting, "n", missing)))
    p_value = get_any(meta, "p"; default = get(parsed_run, "p", get(parsed_setting, "p", missing)))
    nonzero_value = get_any(meta, "nonzero_count"; default = get(parsed_run, "nonzero_count", get(parsed_setting, "nonzero_count", missing)))
    baseline_value = get_any(meta, "baseline"; default = get(parsed_run, "baseline", get(parsed_setting, "baseline", missing)))
    prior_value = get_any(meta, "prior_name", "method_name"; default = get(parsed_run, "prior", missing))
    rep_value = get_any(meta, "replicate_id"; default = get(parsed_run, "rep", missing))

    return Dict{String,Any}(
        "root" => abspath(root),
        "setting_dir" => setting_dir,
        "run_dir" => abspath(run_dir),
        "run_id" => get_any(meta, "run_id", "simulation_id"; default = basename(run_dir)),
        "rep" => rep_value,
        "prior" => prior_value,
        "sampler" => get_any(meta, "sampler"; default = missing),
        "nuts_backend" => get_any(meta, "nuts_backend"; default = missing),
        "n" => n_value,
        "p" => p_value,
        "nonzero_count" => nonzero_value,
        "p0" => get_any(meta, "p0"; default = missing),
        "baseline" => baseline_value,
        "baseline_rate" => get_any(meta, "baseline_rate"; default = missing),
        "baseline_scale" => get_any(meta, "baseline_scale"; default = missing),
        "baseline_shape" => get_any(meta, "baseline_shape"; default = missing),
        "censoring_distribution" => get_any(meta, "censoring_distribution"; default = missing),
        "censoring_upper" => get_any(meta, "censoring_upper"; default = missing),
        "observed_censoring_rate" => get_any(meta, "observed_censoring_rate"; default = missing),
    )
end

function find_run_dirs(root::AbstractString)
    isdir(root) || return String[]
    runs = String[]
    for (dir, dirs, files) in walkdir(root)
        basename(dir) == "runs" || continue
        for entry in sort(readdir(dir; join = true))
            isdir(entry) && push!(runs, entry)
        end
    end
    return sort(unique(runs))
end

function has_sampling_output(run_dir::AbstractString)
    return file_complete(joinpath(run_dir, "samples_wide.csv")) ||
           file_complete(joinpath(run_dir, "result_bundle.jld2"))
end

function has_metric_output(run_dir::AbstractString)
    return file_complete(joinpath(run_dir, "posterior_summary.csv")) &&
           file_complete(joinpath(run_dir, "metrics.csv")) &&
           file_complete(joinpath(run_dir, "false_discovery.csv"))
end

function add_context_columns!(df::DataFrame, ctx::AbstractDict)
    for key in ("root", "setting_dir", "run_dir", "run_id", "rep", "prior", "sampler",
        "nuts_backend", "n", "p", "nonzero_count", "p0", "baseline",
        "baseline_rate", "baseline_scale", "baseline_shape", "censoring_distribution", "censoring_upper",
        "observed_censoring_rate")
        sym = Symbol(key)
        if !(sym in propertynames(df))
            df[!, sym] = fill(get(ctx, key, missing), nrow(df))
        end
    end
    return df
end

function row_dataframe(row::AbstractDict)
    df = DataFrame()
    for key in sort(collect(keys(row)))
        df[!, Symbol(key)] = [row[key]]
    end
    return df
end

function collect_sampling_diagnostics(root::AbstractString, run_dirs::Vector{String})
    rows = DataFrame()
    for run_dir in run_dirs
        ctx = run_context(root, run_dir)
        diag = safe_read_json(joinpath(run_dir, "diagnostics.json"))
        row = merge(copy(ctx), diag)
        row["has_sampling_output"] = has_sampling_output(run_dir)
        row["has_metric_output"] = has_metric_output(run_dir)
        row["has_parameter_diagnostics"] = file_complete(joinpath(run_dir, "diagnose", "diagnostics_summary.csv"))
        row["is_complete"] = row["has_sampling_output"] && row["has_metric_output"]
        append!(rows, row_dataframe(row); cols = :union)
    end
    return rows
end

function collect_model_metrics(root::AbstractString, run_dirs::Vector{String})
    rows = DataFrame()
    for run_dir in run_dirs
        ctx = run_context(root, run_dir)
        metrics = safe_read_csv(joinpath(run_dir, "metrics.csv"))
        isempty(metrics) && continue
        add_context_columns!(metrics, ctx)

        fd = safe_read_csv(joinpath(run_dir, "false_discovery.csv"))
        if nrow(fd) == 1
            for col in (:FP, :FN, :TP, :TN, :FDR)
                col in propertynames(fd) && (metrics[!, col] = fill(fd[1, col], nrow(metrics)))
            end
        end
        append!(rows, metrics; cols = :union)
    end
    if nrow(rows) > 0 && :ELPD_LOO in propertynames(rows)
        add_elpd_diffs!(rows)
    end
    return rows
end

function parameter_group(name)
    s = string(name)
    startswith(s, "beta[") && return "beta"
    startswith(s, "gamma[") && return "gamma"
    startswith(s, "tau2[") && return "tau2"
    s in ("theta", "logit_theta") && return "theta"
    s == "logposterior" && return "logposterior"
    return "other"
end

function parameter_index(name)
    m = match(r"\[(\d+)\]", string(name))
    m === nothing && return missing
    return parse(Int, m.captures[1])
end

function collect_parameter_diagnostics(root::AbstractString, run_dirs::Vector{String})
    rows = DataFrame()
    for run_dir in run_dirs
        path = joinpath(run_dir, "diagnose", "diagnostics_summary.csv")
        diag = safe_read_csv(path)
        isempty(diag) && continue
        ctx = run_context(root, run_dir)
        add_context_columns!(diag, ctx)
        param_col =
            :parameters in propertynames(diag) ? :parameters :
            :parameter in propertynames(diag) ? :parameter :
            nothing
        if param_col !== nothing
            diag[!, :parameter_group] = parameter_group.(diag[!, param_col])
            diag[!, :parameter_index] = parameter_index.(diag[!, param_col])
        else
            diag[!, :parameter_group] = fill("unknown", nrow(diag))
            diag[!, :parameter_index] = fill(missing, nrow(diag))
        end
        append!(rows, diag; cols = :union)
    end
    return rows
end

function finite_values(v)
    vals = Float64[]
    for x in skipmissing(v)
        x isa Number || continue
        xf = Float64(x)
        isfinite(xf) && push!(vals, xf)
    end
    return vals
end

finite_mean(v) = (vals = finite_values(v); isempty(vals) ? NaN : mean(vals))
finite_median(v) = (vals = finite_values(v); isempty(vals) ? NaN : median(vals))
finite_std(v) = (vals = finite_values(v); length(vals) <= 1 ? NaN : std(vals))
finite_min(v) = (vals = finite_values(v); isempty(vals) ? NaN : minimum(vals))
finite_max(v) = (vals = finite_values(v); isempty(vals) ? NaN : maximum(vals))
finite_sum(v) = (vals = finite_values(v); isempty(vals) ? NaN : sum(vals))

function count_unique_nonmissing(v)
    return length(unique(collect(skipmissing(v))))
end

function count_over_threshold(v, threshold)
    count = 0
    for x in skipmissing(v)
        x isa Number || continue
        xf = Float64(x)
        isfinite(xf) && xf > threshold && (count += 1)
    end
    return count
end

function add_elpd_diffs!(metrics::DataFrame)
    group_cols = [
        :root,
        :rep,
        :n,
        :p,
        :nonzero_count,
        :baseline,
        :baseline_rate,
        :baseline_scale,
        :baseline_shape,
        :censoring_distribution,
        :censoring_upper,
        :estimate,
    ]
    valid_cols = [c for c in group_cols if c in propertynames(metrics)]
    :ELPD_Diff in propertynames(metrics) || (metrics[!, :ELPD_Diff] = fill(NaN, nrow(metrics)))
    for group in groupby(metrics, valid_cols)
        vals = finite_values(group.ELPD_LOO)
        isempty(vals) && continue
        best = maximum(vals)
        rows = parentindices(group)[1]
        metrics[rows, :ELPD_Diff] .= Float64.(group.ELPD_LOO) .- best
    end
    return metrics
end

function summarize_sampling(rows::DataFrame)
    isempty(rows) && return DataFrame()
    group_cols = [c for c in [:root, :n, :p, :nonzero_count, :p0, :baseline, :baseline_rate, :baseline_scale, :baseline_shape, :censoring_distribution, :censoring_upper, :prior, :sampler] if c in propertynames(rows)]
    specs = Any[:run_dir => count_unique_nonmissing => :run_count]
    :is_complete in propertynames(rows) && push!(specs, :is_complete => (x -> count(==(true), skipmissing(x))) => :complete_run_count)
    for col in (:warmup_seconds, :sample_seconds, :total_seconds, :allocated_mb, :gc_seconds, :gc_percent,
        :acceptance_rate, :warmup_acceptance_rate, :observed_censoring_rate)
        col in propertynames(rows) || continue
        push!(specs, col => finite_mean => Symbol(col, "_mean"))
        push!(specs, col => finite_std => Symbol(col, "_sd"))
        push!(specs, col => finite_min => Symbol(col, "_min"))
        push!(specs, col => finite_max => Symbol(col, "_max"))
    end
    return sort!(combine(groupby(rows, group_cols), specs...), group_cols)
end

function summarize_model_metrics(rows::DataFrame)
    isempty(rows) && return DataFrame()
    group_cols = [c for c in [:root, :n, :p, :nonzero_count, :p0, :baseline, :baseline_rate, :baseline_scale, :baseline_shape, :censoring_distribution, :censoring_upper, :prior, :sampler, :estimate] if c in propertynames(rows)]
    specs = Any[:run_dir => count_unique_nonmissing => :run_count]
    mean_sd_cols = (:MSE, :loglik, :LPML, :AIC, :BIC, :WAIC_pw1, :WAIC_pw2, :DIC,
        :ELPD_LOO, :ELPD_LOO_SE, :ELPD_Diff, :PLOO, :PLOO_SE, :LOOIC, :LOOIC_SE,
        :PSIS_LOO, :Cindex, :FDR, :FP, :FN, :TP, :TN, :warmup_seconds,
        :sample_seconds, :total_seconds, :allocated_mb, :gc_seconds, :gc_percent)
    for col in mean_sd_cols
        col in propertynames(rows) || continue
        push!(specs, col => finite_mean => Symbol(col, "_mean"))
        push!(specs, col => finite_std => Symbol(col, "_sd"))
    end
    for col in (:ParetoK_max, :ParetoK_bad)
        col in propertynames(rows) || continue
        push!(specs, col => finite_mean => Symbol(col, "_mean"))
        push!(specs, col => finite_max => Symbol(col, "_max"))
    end
    return sort!(combine(groupby(rows, group_cols), specs...), group_cols)
end

function summarize_parameter_diagnostics(rows::DataFrame)
    isempty(rows) && return DataFrame()
    group_cols = [c for c in [:root, :n, :p, :nonzero_count, :p0, :baseline, :baseline_rate, :baseline_scale, :baseline_shape, :censoring_distribution, :censoring_upper, :prior, :sampler, :parameter_group] if c in propertynames(rows)]
    # threshold =
    #     :rhat_threshold in propertynames(rows) && any(!ismissing, rows.rhat_threshold) ?
    #     finite_mean(rows.rhat_threshold) : 1.01
    threshold = 1.1
    specs = Any[
        :run_dir => count_unique_nonmissing => :run_count,
        nrow => :parameter_row_count,
    ]
    if :rhat in propertynames(rows)
        push!(specs, :rhat => finite_mean => :rhat_mean)
        push!(specs, :rhat => finite_median => :rhat_median)
        push!(specs, :rhat => finite_max => :rhat_max)
        push!(specs, :rhat => (x -> count_over_threshold(x, threshold)) => :rhat_over_threshold_count)
    end
    for col in (:ess_bulk, :ess_tail, :ess_per_sec)
        col in propertynames(rows) || continue
        push!(specs, col => finite_mean => Symbol(col, "_mean"))
        push!(specs, col => finite_median => Symbol(col, "_median"))
        push!(specs, col => finite_min => Symbol(col, "_min"))
    end
    for col in (:mcse, :sd)
        col in propertynames(rows) || continue
        push!(specs, col => finite_mean => Symbol(col, "_mean"))
        push!(specs, col => finite_median => Symbol(col, "_median"))
        push!(specs, col => finite_max => Symbol(col, "_max"))
    end
    out = combine(groupby(rows, group_cols), specs...)
    out[!, :rhat_threshold] = fill(threshold, nrow(out))
    return sort!(out, group_cols)
end

function prefixed_columns(df::DataFrame, prefix::AbstractString, keep_cols)
    out = copy(df)
    keep = Set(Symbol.(keep_cols))
    rename_map = Dict{Symbol,Symbol}()
    for col in propertynames(out)
        col in keep && continue
        rename_map[col] = Symbol(prefix, col)
    end
    rename!(out, rename_map)
    return out
end

function safe_leftjoin(left::DataFrame, right::DataFrame; on)
    return leftjoin(left, right; on = on, matchmissing = :equal)
end

function build_overview(model_summary::DataFrame, sampling_summary::DataFrame, parameter_summary::DataFrame)
    isempty(model_summary) && return DataFrame()
    join_cols = [c for c in [:root, :n, :p, :nonzero_count, :p0, :baseline, :baseline_rate, :baseline_scale, :baseline_shape, :censoring_distribution, :censoring_upper, :prior, :sampler] if c in propertynames(model_summary)]
    overview = copy(model_summary)

    if !isempty(sampling_summary)
        sampling = prefixed_columns(sampling_summary, "sampling_", join_cols)
        overview = safe_leftjoin(overview, sampling; on = join_cols)
    end

    if !isempty(parameter_summary) && :parameter_group in propertynames(parameter_summary)
        beta = parameter_summary[parameter_summary.parameter_group .== "beta", :]
        if !isempty(beta)
            select!(beta, Not(:parameter_group))
            beta = prefixed_columns(beta, "beta_mcmc_", join_cols)
            overview = safe_leftjoin(overview, beta; on = join_cols)
        end
    end

    return overview
end

function remove_long_outputs(root::AbstractString)
    for name in (
        "sampling_diagnostics_long.csv",
        "model_metrics_long.csv",
        "parameter_mcmc_diagnostics_long.csv",
    )
        path = joinpath(root, name)
        isfile(path) && rm(path)
    end
    return nothing
end

function summarize_root(root::AbstractString)
    run_dirs = find_run_dirs(root)
    sampling_long = collect_sampling_diagnostics(root, run_dirs)
    model_long = collect_model_metrics(root, run_dirs)
    parameter_long = collect_parameter_diagnostics(root, run_dirs)

    sampling_summary = summarize_sampling(sampling_long)
    model_summary = summarize_model_metrics(model_long)
    parameter_summary = summarize_parameter_diagnostics(parameter_long)
    overview = build_overview(model_summary, sampling_summary, parameter_summary)

    remove_long_outputs(root)

    outputs = Dict(
        "sampling_diagnostics_summary" => write_table(joinpath(root, "sampling_diagnostics_summary.csv"), sampling_summary),
        "model_metrics_summary" => write_table(joinpath(root, "model_metrics_summary.csv"), model_summary),
        "parameter_mcmc_diagnostics_summary" => write_table(joinpath(root, "parameter_mcmc_diagnostics_summary.csv"), parameter_summary),
        "simulation_results_summary" => write_table(joinpath(root, "simulation_results_summary.csv"), overview),
    )

    report = [
        "Simulation result summary",
        "Generated: $(Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))",
        "Root: $(abspath(root))",
        "Run directories found: $(length(run_dirs))",
        "Sampling diagnostic rows: $(nrow(sampling_long))",
        "Model metric rows: $(nrow(model_long))",
        "Parameter diagnostic rows: $(nrow(parameter_long))",
        "",
        "Output files:",
    ]
    append!(report, ["- $(key): $(path)" for (key, path) in sort(collect(outputs); by = first)])
    write(joinpath(root, "simulation_results_summary_report.txt"), join(report, "\n") * "\n")

    println(join(report, "\n"))
    return outputs
end

function main()
    args = parse_cli_args()
    roots_arg = get(args, "roots", get(args, "root", DEFAULT_ROOT))
    roots = split_csv(roots_arg)
    isempty(roots) && push!(roots, DEFAULT_ROOT)
    for root in roots
        summarize_root(root)
    end
end

if normpath(abspath(PROGRAM_FILE)) == normpath(abspath(@__FILE__))
    main()
end
