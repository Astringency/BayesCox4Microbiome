#!/usr/bin/env julia

using CSV
using DataFrames

const ROOT = normpath(joinpath(@__DIR__, ".."))
const SUMMARY_ROOT = joinpath(ROOT, "results", "summary", "realdata")
const DEFAULT_BAYES_DIR = joinpath(SUMMARY_ROOT, "bayes_all_methods")
const DEFAULT_OUTDIR = joinpath(SUMMARY_ROOT, "method_comparison")
const DEFAULT_PLOT_SCRIPT = joinpath(ROOT, "scripts", "plot_realdata_method_comparison.R")
const GROUP_ORDER = Dict("T1D" => 1, "HCT" => 2, "TCGA" => 3)
const PRIOR_ORDER = ["hs", "rhs", "new-rhs", "est-rhs", "cox-hs", "cox-rhs", "lasso", "sslasso", "prox-lasso", "prox-sslasso"]
const CONSTRAINT_ORDER = Dict("soft_center" => 1, "prox" => 2)

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
ensure_dir(path::AbstractString) = (mkpath(path); path)

function read_table(path::AbstractString)
    isfile(path) || return DataFrame()
    return CSV.read(path, DataFrame; normalizenames = false)
end

function write_table(path::AbstractString, table::DataFrame)
    ensure_dir(dirname(path))
    CSV.write(path, table)
    return path
end

function as_string(x)
    x === missing && return ""
    x === nothing && return ""
    return string(x)
end

function maybe_float(x)
    x === missing && return missing
    x === nothing && return missing
    x isa Number && return isfinite(Float64(x)) ? Float64(x) : missing
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

function bool_value(x)
    s = lowercase(strip(as_string(x)))
    s in ("true", "1", "yes", "y") && return true
    s in ("false", "0", "no", "n") && return false
    return missing
end

function getval(row, name::AbstractString, default = missing)
    sym = Symbol(name)
    return sym in propertynames(row) ? row[sym] : default
end

function latest_dir(prefix::AbstractString)
    isdir(SUMMARY_ROOT) || return ""
    matches = sort(filter(p -> isdir(p) && startswith(basename(p), prefix), readdir(SUMMARY_ROOT; join = true)))
    return isempty(matches) ? "" : matches[end]
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

function split_features(text)
    s = strip(as_string(text))
    isempty(s) && return String[]
    return [String(strip(x)) for x in split(s, ';') if !isempty(strip(x))]
end

function safe_column_name(x::AbstractString)
    s = replace(strip(x), r"[^A-Za-z0-9_]+" => "_")
    s = replace(s, r"_+" => "_")
    s = strip(s, '_')
    isempty(s) && return "value"
    occursin(r"^[0-9]", s) && (s = "x_" * s)
    return s
end

function method_order(family, method)
    family = string(family)
    family == "Bayes-soft-center" && return 10
    family == "Bayes-prox" && return 20
    family == "FLORAL" && occursin("min", string(method)) && return 30
    family == "FLORAL" && return 31
    family == "coda4microbiome" && occursin("min", string(method)) && return 40
    family == "coda4microbiome" && return 41
    return 99
end

function prior_order(prior::AbstractString)
    idx = findfirst(==(string(prior)), PRIOR_ORDER)
    return idx === nothing ? 999 : idx
end

constraint_order(constraint::AbstractString) = get(CONSTRAINT_ORDER, string(constraint), 99)

function sort_methods!(df::DataFrame)
    nrow(df) == 0 && return df
    df._group_order = [get(GROUP_ORDER, string(g), 99) for g in df.dataset_group]
    df._method_order = [method_order(row.method_family, row.method) for row in eachrow(df)]
    df._constraint_order = [constraint_order(as_string(row.constraint)) for row in eachrow(df)]
    df._prior_order = [prior_order(as_string(row.prior)) for row in eachrow(df)]
    sort!(df, [:_group_order, :dataset_id, :_method_order, :_constraint_order, :_prior_order, :method])
    select!(df, Not([:_group_order, :_method_order, :_constraint_order, :_prior_order]))
    return df
end

function standard_row(; dataset_id, dataset_group, method_family, method, prior = "", constraint = "",
    row_type = "", fit_success = missing, converged = missing, cv_cindex = missing,
    cv_cindex_se = missing, cv_cindex_sd = missing, n_selected = missing,
    cindex_metric = "CV-Cindex",
    selected_features = "", rhat_max = missing, rhat_median = missing, p0 = missing,
    n = missing, p = missing, events = missing, nlpml = missing, waic_pw1 = missing,
    waic_pw2 = missing, elpd_loo = missing, elpd_loo_se = missing, looic = missing,
    looic_se = missing, pareto_k_max = missing, pareto_k_bad = missing, source = "",
    run_dir = "", coef_path = "")
    return (
        dataset_id = dataset_id,
        dataset_group = dataset_group,
        method_family = method_family,
        method = method,
        prior = prior,
        constraint = constraint,
        row_type = row_type,
        fit_success = fit_success,
        converged = converged,
        cindex_metric = cindex_metric,
        cindex = cv_cindex,
        cindex_se = cv_cindex_se,
        cindex_sd = cv_cindex_sd,
        n_selected = n_selected,
        selected_features = selected_features,
        rhat_max = rhat_max,
        rhat_median = rhat_median,
        p0 = p0,
        n = n,
        p = p,
        events = events,
        nlpml = nlpml,
        waic_pw1 = waic_pw1,
        waic_pw2 = waic_pw2,
        elpd_loo = elpd_loo,
        elpd_loo_se = elpd_loo_se,
        looic = looic,
        looic_se = looic_se,
        pareto_k_max = pareto_k_max,
        pareto_k_bad = pareto_k_bad,
        source = source,
        run_dir = run_dir,
        coef_path = coef_path,
    )
end

function bayes_rows(bayes_dir::AbstractString)
    path = joinpath(bayes_dir, "bayes_all_methods.csv")
    df = read_table(path)
    rows = NamedTuple[]
    nrow(df) == 0 && return rows
    for row in eachrow(df)
        nested = as_string(getval(row, "cv_evaluation", "")) == "nested_cv"
        push!(rows, standard_row(
            dataset_id = as_string(getval(row, "dataset_id", "")),
            dataset_group = as_string(getval(row, "dataset_group", "")),
            method_family = as_string(getval(row, "method_family", "")),
            method = as_string(getval(row, "method", "")),
            prior = as_string(getval(row, "prior", "")),
            constraint = as_string(getval(row, "constraint", "")),
            row_type = "bayes_final",
            fit_success = bool_value(getval(row, "fit_success", true)),
            converged = bool_value(getval(row, "converged", missing)),
            cindex_metric = "Nested-CV-Cindex",
            cv_cindex = nested ? maybe_float(getval(row, "cv_cindex", missing)) : missing,
            cv_cindex_se = nested ? maybe_float(getval(row, "cv_cindex_se", missing)) : missing,
            cv_cindex_sd = nested ? maybe_float(getval(row, "cv_cindex_sd", missing)) : missing,
            n_selected = maybe_int(getval(row, "n_selected", missing)),
            rhat_max = maybe_float(getval(row, "rhat_max", missing)),
            rhat_median = maybe_float(getval(row, "rhat_median", missing)),
            p0 = maybe_float(getval(row, "p0", missing)),
            n = maybe_int(getval(row, "n", missing)),
            p = maybe_int(getval(row, "p", missing)),
            events = maybe_int(getval(row, "events", missing)),
            nlpml = maybe_float(getval(row, "nlpml", missing)),
            waic_pw1 = maybe_float(getval(row, "waic_pw1", missing)),
            waic_pw2 = maybe_float(getval(row, "waic_pw2", missing)),
            elpd_loo = maybe_float(getval(row, "elpd_loo", missing)),
            elpd_loo_se = maybe_float(getval(row, "elpd_loo_se", missing)),
            looic = maybe_float(getval(row, "looic", missing)),
            looic_se = maybe_float(getval(row, "looic_se", missing)),
            pareto_k_max = maybe_float(getval(row, "pareto_k_max", missing)),
            pareto_k_bad = maybe_int(getval(row, "pareto_k_bad", missing)),
            source = path,
            run_dir = as_string(getval(row, "run_dir", "")),
            coef_path = as_string(getval(row, "coef_path", "")),
        ))
    end
    return rows
end

function floral_rows(floral_dir::AbstractString)
    path = joinpath(floral_dir, "metrics.csv")
    coef_path = joinpath(floral_dir, "selected_features.csv")
    df = read_table(path)
    rows = NamedTuple[]
    nrow(df) == 0 && return rows
    for row in eachrow(df)
        dataset_id = canonical_dataset(as_string(getval(row, "dataset_id", "")))
        push!(rows, standard_row(
            dataset_id = dataset_id,
            dataset_group = as_string(getval(row, "dataset_group", dataset_group(dataset_id))),
            method_family = "FLORAL",
            method = as_string(getval(row, "method", "")),
            row_type = "baseline",
            fit_success = bool_value(getval(row, "fit_success", missing)),
            cv_cindex = maybe_float(getval(row, "floral_cv_cindex", missing)),
            cv_cindex_se = maybe_float(getval(row, "floral_cv_cindex_se", missing)),
            n_selected = maybe_int(getval(row, "n_selected", missing)),
            selected_features = as_string(getval(row, "selected_features", "")),
            n = maybe_int(getval(row, "n_used", missing)),
            p = maybe_int(getval(row, "p", missing)),
            events = maybe_int(getval(row, "n_events", missing)),
            source = path,
            coef_path = coef_path,
        ))
    end
    return rows
end

function coda_rows(coda_dir::AbstractString)
    path = joinpath(coda_dir, "metrics.csv")
    coef_path = joinpath(coda_dir, "selected_features.csv")
    df = read_table(path)
    rows = NamedTuple[]
    nrow(df) == 0 && return rows
    for row in eachrow(df)
        dataset_id = canonical_dataset(as_string(getval(row, "dataset_id", "")))
        push!(rows, standard_row(
            dataset_id = dataset_id,
            dataset_group = as_string(getval(row, "dataset_group", dataset_group(dataset_id))),
            method_family = "coda4microbiome",
            method = as_string(getval(row, "method", "")),
            row_type = "baseline",
            fit_success = bool_value(getval(row, "fit_success", missing)),
            cv_cindex = maybe_float(getval(row, "mean_cv_cindex_package", missing)),
            cv_cindex_sd = maybe_float(getval(row, "sd_cv_cindex_package", missing)),
            n_selected = maybe_int(getval(row, "n_selected", missing)),
            selected_features = as_string(getval(row, "selected_features", "")),
            n = maybe_int(getval(row, "n_used", missing)),
            p = maybe_int(getval(row, "p", missing)),
            events = maybe_int(getval(row, "n_events", missing)),
            source = path,
            coef_path = coef_path,
        ))
    end
    return rows
end

function coefficient_rows(methods::DataFrame)
    rows = DataFrame(
        dataset_id = String[],
        dataset_group = String[],
        variable_index = Int[],
        method_family = String[],
        method = String[],
        prior = String[],
        constraint = String[],
        variable = String[],
        variable_kind = String[],
        mean = Union{Missing,Float64}[],
        median = Union{Missing,Float64}[],
        map = Union{Missing,Float64}[],
        lower = Union{Missing,Float64}[],
        upper = Union{Missing,Float64}[],
        selected = Bool[],
        interval_available = Bool[],
        source = String[],
    )
    for row in eachrow(methods)
        family = as_string(row.method_family)
        if startswith(family, "Bayes")
            coef_path = as_string(row.coef_path)
            isfile(coef_path) || continue
            coef = read_table(coef_path)
            nrow(coef) == 0 && continue
            for (j, crow) in enumerate(eachrow(coef))
                lower = maybe_float(getval(crow, "lower", missing))
                upper = maybe_float(getval(crow, "upper", missing))
                selected = lowercase(as_string(getval(crow, "selected", ""))) == "true"
                push!(rows, (
                    as_string(row.dataset_id),
                    as_string(row.dataset_group),
                    j,
                    family,
                    as_string(row.method),
                    as_string(row.prior),
                    as_string(row.constraint),
                    as_string(getval(crow, "variable", "")),
                    startswith(as_string(row.dataset_id), "TCGA_") ? "immune_cell_fraction" : "microbiome_taxon",
                    maybe_float(getval(crow, "mean", missing)),
                    maybe_float(getval(crow, "median", missing)),
                    maybe_float(getval(crow, "map", missing)),
                    lower,
                    upper,
                    selected,
                    lower !== missing && upper !== missing,
                    coef_path,
                ))
            end
        else
            coef_path = as_string(row.coef_path)
            selected_map = Dict{String,Union{Missing,Float64}}()
            if isfile(coef_path)
                coef = read_table(coef_path)
                required = ("dataset_id", "method", "feature", "coefficient")
                if nrow(coef) > 0 && all(name -> name in names(coef), required)
                    dataset_id = as_string(row.dataset_id)
                    method = as_string(row.method)
                    for crow in eachrow(coef)
                        canonical_dataset(as_string(getval(crow, "dataset_id", ""))) == dataset_id || continue
                        as_string(getval(crow, "method", "")) == method || continue
                        coefval = maybe_float(getval(crow, "coefficient", missing))
                        selected_map[as_string(getval(crow, "feature", ""))] = coefval === missing ? missing : Float64(coefval)
                    end
                end
            end
            if isempty(selected_map)
                for feature in split_features(row.selected_features)
                    selected_map[feature] = missing
                end
            end
            j = 0
            for (feature, coef) in selected_map
                j += 1
                val = coef === missing ? missing : Float64(coef)
                push!(rows, (
                    as_string(row.dataset_id),
                    as_string(row.dataset_group),
                    j,
                    family,
                    as_string(row.method),
                    "",
                    "",
                    feature,
                    startswith(as_string(row.dataset_id), "TCGA_") ? "immune_cell_fraction" : "microbiome_taxon",
                    val,
                    missing,
                    missing,
                    missing,
                    missing,
                    true,
                    false,
                    isempty(coef_path) ? "baseline_selected_features" : coef_path,
                ))
            end
        end
    end
    return rows
end

function dataset_variables(raw::DataFrame, dataset_id::AbstractString)
    ds = raw[string.(raw.dataset_id) .== dataset_id, :]
    bayes = ds[startswith.(string.(ds.method_family), "Bayes"), :]
    variables = DataFrame(variable = String[], variable_index = Int[])
    seen = Set{String}()
    for row in eachrow(sort(bayes, :variable_index))
        variable = as_string(row.variable)
        isempty(variable) && continue
        variable in seen && continue
        push!(seen, variable)
        push!(variables, (variable, Int(row.variable_index)))
    end
    next_idx = nrow(variables) + 1
    for row in eachrow(ds)
        variable = as_string(row.variable)
        isempty(variable) && continue
        variable in seen && continue
        push!(seen, variable)
        push!(variables, (variable, next_idx))
        next_idx += 1
    end
    return variables
end

function complete_coefficients(methods::DataFrame, raw::DataFrame)
    complete = similar(raw, 0)
    for dataset_id in sort(unique(string.(methods.dataset_id)))
        ds_methods = methods[string.(methods.dataset_id) .== dataset_id, :]
        variables = dataset_variables(raw, dataset_id)
        nrow(variables) == 0 && continue
        ds_raw = raw[string.(raw.dataset_id) .== dataset_id, :]
        for method_row in eachrow(ds_methods)
            method = as_string(method_row.method)
            family = as_string(method_row.method_family)
            method_raw = ds_raw[string.(ds_raw.method) .== method, :]
            selected_lookup = Dict{String,Any}()
            for r in eachrow(method_raw)
                selected_lookup[as_string(r.variable)] = r
            end
            for vrow in eachrow(variables)
                variable = as_string(vrow.variable)
                if haskey(selected_lookup, variable)
                    r = selected_lookup[variable]
                    push!(complete, (
                        as_string(r.dataset_id),
                        as_string(r.dataset_group),
                        Int(vrow.variable_index),
                        as_string(r.method_family),
                        as_string(r.method),
                        as_string(r.prior),
                        as_string(r.constraint),
                        variable,
                        as_string(r.variable_kind),
                        maybe_float(r.mean),
                        maybe_float(r.median),
                        maybe_float(r.map),
                        maybe_float(r.lower),
                        maybe_float(r.upper),
                        Bool(r.selected),
                        Bool(r.interval_available),
                        as_string(r.source),
                    ))
                elseif startswith(family, "Bayes")
                    push!(complete, (
                        dataset_id,
                        as_string(method_row.dataset_group),
                        Int(vrow.variable_index),
                        family,
                        method,
                        as_string(method_row.prior),
                        as_string(method_row.constraint),
                        variable,
                        startswith(dataset_id, "TCGA_") ? "immune_cell_fraction" : "microbiome_taxon",
                        missing,
                        missing,
                        missing,
                        missing,
                        missing,
                        false,
                        false,
                        as_string(method_row.coef_path),
                    ))
                else
                    push!(complete, (
                        dataset_id,
                        as_string(method_row.dataset_group),
                        Int(vrow.variable_index),
                        family,
                        method,
                        "",
                        "",
                        variable,
                        startswith(dataset_id, "TCGA_") ? "immune_cell_fraction" : "microbiome_taxon",
                        0.0,
                        missing,
                        missing,
                        missing,
                        missing,
                        false,
                        false,
                        as_string(method_row.coef_path),
                    ))
                end
            end
        end
    end
    complete.point_estimate = [
        maybe_float(row.mean) === missing ? missing : Float64(maybe_float(row.mean))
        for row in eachrow(complete)
    ]
    return complete
end

function write_coefficient_wide_tables(outdir::AbstractString, methods::DataFrame, complete::DataFrame)
    root = ensure_dir(joinpath(outdir, "coefficient_estimates"))
    for dataset_id in sort(unique(string.(methods.dataset_id)))
        ds_methods = methods[string.(methods.dataset_id) .== dataset_id, :]
        ds_coefs = complete[string.(complete.dataset_id) .== dataset_id, :]
        (nrow(ds_methods) == 0 || nrow(ds_coefs) == 0) && continue
        variables = sort(unique(ds_coefs[:, [:variable_index, :variable]]), :variable_index)
        table = DataFrame(variable = string.(variables.variable))
        for row in eachrow(ds_methods)
            method = as_string(row.method)
            family = as_string(row.method_family)
            base = safe_column_name(method)
            rows = ds_coefs[string.(ds_coefs.method) .== method, :]
            lookup = Dict(as_string(r.variable) => r for r in eachrow(rows))
            if startswith(family, "Bayes")
                for field in ("mean", "median", "map", "lower", "upper")
                    table[!, Symbol(base * "_" * field)] = [
                        haskey(lookup, v) ? maybe_float(getval(lookup[v], field, missing)) : missing
                        for v in table.variable
                    ]
                end
            else
                table[!, Symbol(base * "_estimate")] = [
                    haskey(lookup, v) ? maybe_float(lookup[v].point_estimate) : 0.0
                    for v in table.variable
                ]
            end
        end
        write_table(joinpath(root, "$(safe_column_name(dataset_id))_coefficients.csv"), table)
    end
end

function parameter_index(parameter::AbstractString)
    m = match(r"\[(\d+)\]", parameter)
    m === nothing && return missing
    return parse(Int, m.captures[1])
end

function rhat_long(bayes_dir::AbstractString)
    path = joinpath(bayes_dir, "bayes_rhat_by_parameter.csv")
    df = read_table(path)
    nrow(df) == 0 && return df
    out = copy(df)
    out.variable_index = [something(parameter_index(as_string(p)), i) for (i, p) in enumerate(out.parameter)]
    out.method = ["Bayes_$(as_string(c))_$(as_string(p))" for (c, p) in zip(out.constraint, out.prior)]
    out.method_family = [as_string(c) == "prox" ? "Bayes-prox" : "Bayes-soft-center" for c in out.constraint]
    out.plot_label = ["$(as_string(c)) / $(as_string(p))" for (c, p) in zip(out.constraint, out.prior)]
    return out
end

function metrics_table(methods::DataFrame)
    cols = [
        :dataset_id, :dataset_group, :method_family, :method, :prior, :constraint,
        :row_type, :fit_success, :converged, :cindex_metric, :cindex, :cindex_se,
        :cindex_sd, :n_selected, :rhat_median, :rhat_max, :p0, :n, :p, :events,
        :nlpml, :waic_pw1, :waic_pw2, :elpd_loo, :elpd_loo_se, :looic, :looic_se,
        :pareto_k_max, :pareto_k_bad, :source,
    ]
    return methods[:, cols]
end

function clean_method_comparison_output(outdir::AbstractString)
    ensure_dir(outdir)
    for sub in ("by_group", "figures", "metrics", "coefficient_estimates", "plot_data")
        path = joinpath(outdir, sub)
        try
            rm(path; recursive = true, force = true)
        catch err
            @warn "Could not remove old generated directory; continuing." path err
        end
    end
    for file in (
        "README.md",
        "all_methods_performance.csv",
        "all_tuning_and_baselines.csv",
        "best_per_prior_and_baselines.csv",
        "all_coefficients_for_forest.csv",
    )
        path = joinpath(outdir, file)
        try
            rm(path; force = true)
        catch err
            @warn "Could not remove old generated file; continuing." path err
        end
    end
    ensure_dir(joinpath(outdir, "metrics"))
    ensure_dir(joinpath(outdir, "coefficient_estimates"))
    ensure_dir(joinpath(outdir, "plot_data"))
    ensure_dir(joinpath(outdir, "figures", "forest"))
    ensure_dir(joinpath(outdir, "figures", "rhat"))
    ensure_dir(joinpath(outdir, "figures", "select"))
end

function wslpath_to_windows(path::AbstractString)
    try
        return strip(read(`wslpath -w $path`, String))
    catch
        return path
    end
end

function windows_path_to_wsl(path::AbstractString)
    try
        return strip(read(`wslpath -u $path`, String))
    catch
        return path
    end
end

function find_rscript()
    for exe in ("Rscript", "Rscript.exe")
        p = Sys.which(exe)
        if p !== nothing
            return p, endswith(lowercase(p), ".exe")
        end
    end
    try
        output = strip(read(`cmd.exe /c where Rscript`, String))
        if isempty(output)
            return "", false
        end
        first = strip(split(output, '\n')[1])
        local_path = windows_path_to_wsl(first)
        if isfile(local_path)
            return local_path, true
        end
    catch err
    end
    return "", false
end

function run_r_plots(
    outdir::AbstractString,
    plot_script::AbstractString;
    plot_priors::AbstractString = "",
    plot_baseline_families::AbstractString = "",
    combined_layouts::AbstractString = "",
    combined_datasets::AbstractString = "",
    combined_only::AbstractString = "",
)
    isfile(plot_script) || (@warn "R plot script not found." plot_script; return)
    rscript, windows_r = find_rscript()
    isempty(rscript) && (@warn "Rscript not found; plot data were written but figures were not regenerated."; return)
    script_arg = windows_r ? wslpath_to_windows(plot_script) : plot_script
    out_arg = windows_r ? wslpath_to_windows(outdir) : outdir
    cmd = `$rscript $script_arg --summary-dir $out_arg`
    if !isempty(strip(plot_priors)) && lowercase(strip(plot_priors)) != "all"
        cmd = `$cmd --priors $plot_priors`
    end
    if !isempty(strip(plot_baseline_families)) && lowercase(strip(plot_baseline_families)) != "all"
        cmd = `$cmd --baseline-families $plot_baseline_families`
    end
    if !isempty(strip(combined_layouts)) && lowercase(strip(combined_layouts)) != "none"
        cmd = `$cmd --combined-layouts $combined_layouts`
    end
    if !isempty(strip(combined_datasets)) && lowercase(strip(combined_datasets)) != "all"
        cmd = `$cmd --combined-datasets $combined_datasets`
    end
    if lowercase(strip(combined_only)) in ("true", "1", "yes", "y")
        cmd = `$cmd --combined-only true`
    end
    run(cmd)
end

function write_outputs(
    outdir::AbstractString,
    methods::DataFrame,
    bayes_dir::AbstractString;
    plot_script::String,
    make_plots::Bool,
    plot_priors::String,
    plot_baseline_families::String,
    combined_layouts::String,
    combined_datasets::String,
    combined_only::String,
)
    clean_method_comparison_output(outdir)
    sort_methods!(methods)
    raw = coefficient_rows(methods)
    complete = complete_coefficients(methods, raw)
    rhats = rhat_long(bayes_dir)

    write_table(joinpath(outdir, "metrics", "all_methods_metrics.csv"), metrics_table(methods))
    write_table(joinpath(outdir, "plot_data", "coefficients_long.csv"), complete)
    write_table(joinpath(outdir, "plot_data", "selection_matrix.csv"), complete[:, [:dataset_id, :dataset_group, :variable_index, :variable, :method_family, :method, :prior, :constraint, :selected]])
    write_table(joinpath(outdir, "plot_data", "rhat_long.csv"), rhats)
    write_coefficient_wide_tables(outdir, methods, complete)

    make_plots && run_r_plots(
        outdir,
        plot_script;
        plot_priors = plot_priors,
        plot_baseline_families = plot_baseline_families,
        combined_layouts = combined_layouts,
        combined_datasets = combined_datasets,
        combined_only = combined_only,
    )
end

function main()
    opts = parse_args(ARGS)
    bayes_dir = normpath(opt(opts, "bayes-summary-dir", DEFAULT_BAYES_DIR))
    floral_dir = opt(opts, "floral-dir", latest_dir("realdata_FLORAL_"))
    coda_dir = opt(opts, "coda-dir", latest_dir("realdata_coda4microbiome_"))
    outdir = normpath(opt(opts, "outdir", DEFAULT_OUTDIR))
    plot_script = normpath(opt(opts, "plot-script", DEFAULT_PLOT_SCRIPT))
    make_plots = lowercase(opt(opts, "plot", "true")) != "false"
    plot_priors = opt(opts, "plot-priors", "")
    plot_baseline_families = opt(opts, "plot-baseline-families", "")
    combined_layouts = opt(opts, "combined-layouts", "")
    combined_datasets = opt(opts, "combined-datasets", "")
    combined_only = opt(opts, "combined-only", "")

    rows = NamedTuple[]
    append!(rows, bayes_rows(bayes_dir))
    !isempty(floral_dir) && append!(rows, floral_rows(floral_dir))
    !isempty(coda_dir) && append!(rows, coda_rows(coda_dir))
    methods = DataFrame(rows)
    write_outputs(
        outdir,
        methods,
        bayes_dir;
        plot_script = plot_script,
        make_plots = make_plots,
        plot_priors = plot_priors,
        plot_baseline_families = plot_baseline_families,
        combined_layouts = combined_layouts,
        combined_datasets = combined_datasets,
        combined_only = combined_only,
    )
    println(outdir)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
