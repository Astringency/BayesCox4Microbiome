# Concordance, training-only preprocessing, tuning and nested cross-validation.
function concordance_index(
    time::AbstractVector{<:Real},
    event::AbstractVector,
    risk_score::AbstractVector{<:Real},
)
    n = length(time)
    length(event) == n || throw(DimensionMismatch("length(event) must match length(time)."))
    length(risk_score) == n || throw(DimensionMismatch("length(risk_score) must match length(time)."))
    e = [Int(v) for v in event]
    concordant = 0.0
    comparable = 0.0
    @inbounds for i in 1:n
        if e[i] != 0
            for j in 1:n
                if time[i] < time[j]
                    comparable += 1
                    if risk_score[i] > risk_score[j]
                        concordant += 1
                    elseif risk_score[i] == risk_score[j]
                        concordant += 0.5
                    end
                end
            end
        end
    end
    comparable > 0 || return NaN
    return concordant / comparable
end

# Shared training-only preprocessing for validation and subject-level LOO.
function _training_zero_replacement(X_train::AbstractMatrix)
    positives = Float64[x for x in X_train if x > 0]
    isempty(positives) && throw(ArgumentError("training X contains no positive values."))
    return minimum(positives) / 2
end

function _validation_design_matrix(
    X_train::AbstractMatrix,
    X_val::AbstractMatrix,
    train_data::CoxPreparedData,
)
    x = Matrix{Float64}(X_val)
    all(isfinite, x) || throw(ArgumentError("validation X must be finite."))
    all(>=(0), x) || throw(ArgumentError("validation X must be nonnegative."))

    z =
        if train_data.transform == :log
            if any(iszero, x)
                x[x .== 0.0] .= _training_zero_replacement(X_train)
            end
            if train_data.closure
                rowsums = vec(sum(x; dims = 2))
                all(>(0), rowsums) ||
                    throw(ArgumentError("each validation row sum after zero handling must be positive."))
                x ./= rowsums
            end
            log.(x)
        elseif train_data.transform == :none
            x
        else
            throw(ArgumentError("unsupported training transform $(train_data.transform)."))
        end

    if train_data.standardize
        z = (z .- reshape(train_data.col_mean, 1, :)) ./ reshape(train_data.col_scale, 1, :)
    end
    return z
end


const TUNING_OUTPUT_FILES = (
    prior_metrics = "prior_tuning_metrics.csv",
    prior_summary = "prior_tuning_summary.csv",
    algorithm_metrics = "algorithm_tuning_metrics.csv",
    algorithm_summary = "algorithm_tuning_summary.csv",
    all_results = "all_tuning_results.csv",
    best_config = "best_tuning_config.json",
)

_tuning_key(k::Symbol) = k
_tuning_key(k::AbstractString) = Symbol(replace(k, "-" => "_"))

function _tuning_dict(x)
    out = Dict{Symbol,Any}()
    for (k, v) in pairs(x)
        out[_tuning_key(k)] = v
    end
    return out
end

function _merge_tuning_kwargs(parts...)
    out = Dict{Symbol,Any}()
    for part in parts
        for (k, v) in pairs(_tuning_dict(part))
            out[k] = v
        end
    end
    return out
end

_kwget(d::AbstractDict, key::Symbol, default = missing) = get(d, key, default)
_kwget(nt::NamedTuple, key::Symbol, default = missing) = haskey(nt, key) ? getfield(nt, key) : default

function _json_ready(x)
    x === missing && return nothing
    x === nothing && return nothing
    x isa Symbol && return String(x)
    x isa AbstractString && return String(x)
    x isa Bool && return x
    x isa Number && return isfinite(Float64(x)) ? x : string(x)
    x isa AbstractVector && return [_json_ready(v) for v in x]
    x isa Tuple && return [_json_ready(v) for v in x]
    x isa NamedTuple && return Dict(String(k) => _json_ready(v) for (k, v) in pairs(x))
    x isa AbstractDict && return Dict(String(k) => _json_ready(v) for (k, v) in pairs(x))
    return string(x)
end

function _write_tuning_json(path::AbstractString, obj)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, _json_ready(obj), 4)
        write(io, "\n")
    end
    return path
end

function _append_or_write_table(path::AbstractString, table::DataFrame)
    mkpath(dirname(path))
    if _metrics_file_complete(path)
        old = CSV.read(path, DataFrame)
        combined = copy(old)
        append!(combined, table; cols = :union, promote = true)
        CSV.write(path, combined)
    else
        CSV.write(path, table)
    end
    return path
end

function _metadata_columns(metadata)
    out = Dict{Symbol,Any}()
    for (k, v) in pairs(metadata)
        key = _tuning_key(k)
        key == :beta && continue
        (v isa AbstractVector || v isa AbstractMatrix || v isa AbstractDict) && continue
        out[key] = v === nothing ? missing : v isa Symbol ? String(v) : v
    end
    return out
end

function _metadata_beta(metadata)
    for key in (:beta, "beta")
        haskey(metadata, key) && return collect(Float64, metadata[key])
    end
    return nothing
end

function _error_message(err)
    msg = sprint(showerror, err)
    return length(msg) > 2000 ? first(msg, 2000) : msg
end

function _finite_number(x)
    x === missing && return nothing
    x === nothing && return nothing
    try
        xf = Float64(x)
        return isfinite(xf) ? xf : nothing
    catch
        return nothing
    end
end

function _finite_values(rows::Vector{Dict{Symbol,Any}}, key::Symbol; estimate = nothing)
    vals = Float64[]
    for row in rows
        if estimate !== nothing && string(get(row, :estimate, "")) != estimate
            continue
        end
        xf = _finite_number(get(row, key, missing))
        xf === nothing || push!(vals, xf)
    end
    return vals
end

_mean_or_missing(vals::AbstractVector{<:Real}) = isempty(vals) ? missing : mean(vals)
_sd_or_missing(vals::AbstractVector{<:Real}) = length(vals) <= 1 ? missing : std(vals)
_se_or_missing(vals::AbstractVector{<:Real}) = length(vals) <= 1 ? missing : std(vals) / sqrt(length(vals))

function _dict_rows_dataframe(rows::Vector{Dict{Symbol,Any}})
    table = DataFrame()
    for row in rows
        push!(table, Dict(k => v === nothing ? missing : v for (k, v) in row); cols = :union)
    end
    return table
end

function _add_common_row_fields!(row, metadata_cols, mode, dataset_id)
    row[:mode] = String(mode)
    row[:dataset_id] = String(dataset_id)
    for (k, v) in pairs(metadata_cols)
        row[k] = v
    end
    return row
end

function _add_config_fields!(row, cfg::AbstractDict; prefix::AbstractString = "")
    for (k, v) in pairs(_tuning_dict(cfg))
        col =
            isempty(prefix) ?
            (k == :p0 ? :p0_config : k) :
            Symbol(prefix, k == :p0 ? "p0" : String(k))
        row[col] = v === nothing ? missing : v isa Symbol ? String(v) : v
    end
    return row
end

function _extract_algorithm_params(cfg::AbstractDict)
    result = Dict{Symbol,Any}()
    for (k, v) in pairs(cfg)
        startswith(String(k), "alg_") || continue
        v === nothing && continue
        ismissing(v) && continue
        param_name = Symbol(String(k)[5:end])  # strip "alg_" prefix
        result[param_name] = v isa AbstractString ? v : v
    end
    return result
end

function _p0_for_config(cfg::AbstractDict)
    haskey(cfg, :p0) && return cfg[:p0]
    haskey(cfg, "p0") && return cfg["p0"]
    return missing
end

function _is_l1_prior(prior::AbstractString)
    return prior in ("lasso", "sslasso", "prox-lasso", "prox-sslasso")
end

function _actual_fit_kwargs(
    prior::AbstractString,
    base_sampler_kwargs,
    base_prior_kwargs,
    base_car_kwargs,
    base_model_kwargs,
    override_kwargs;
    p::Integer,
    realdata::Bool,
    clinical_columns::Integer = 0,
)
    kwargs = _merge_tuning_kwargs(
        base_sampler_kwargs,
        base_car_kwargs,
        base_model_kwargs,
        base_prior_kwargs,
        override_kwargs,
    )
    kwargs[:prior_name] = prior
    kwargs[:realdata] = realdata
    # Alpha's scale is a fixed model setting, not a tuning-grid override.
    # Preserve an explicit value; otherwise count taxon and clinical coefficients.
    if realdata && prior in ("prox-lasso", "prox-sslasso") &&
       get(kwargs, :alpha_prior_scale, nothing) === nothing
        kwargs[:alpha_prior_scale] = Float64(p + clinical_columns)
    end
    use_car = Bool(get(kwargs, :use_car, false))
    actual_use_car = use_car && occursin("hs", lowercase(prior))
    kwargs[:use_car] = actual_use_car
    kwargs[:sample_car_strength] = actual_use_car && Bool(get(kwargs, :sample_car_strength, false))
    if _is_l1_prior(prior)
        kwargs[:noncentered] = false
    end
    return kwargs
end

function _kfold_indices(event::AbstractVector; k::Integer, seed::Integer)
    n = length(event)
    2 <= k <= n || throw(ArgumentError("cv_folds must be between 2 and n."))
    rng = MersenneTwister(seed)
    event_idx = shuffle(rng, findall(!=(0), Int.(event)))
    cens_idx = shuffle(rng, findall(==(0), Int.(event)))
    folds = [Int[] for _ in 1:k]
    for (i, idx) in enumerate(event_idx)
        push!(folds[mod1(i, k)], idx)
    end
    for (i, idx) in enumerate(cens_idx)
        push!(folds[mod1(i, k)], idx)
    end
    return sort!.(folds)
end

function _validation_cindices(fit, X_train, X_val, time_val, event_val;
    clinical_X_val = nothing)
    z_val = _validation_design_matrix(
        X_train,
        X_val,
        fit.result.data,
    )
    model_draws = posterior_beta_draws(fit.result)
    beta_mean = vec(mean(model_draws; dims = 1))
    beta_median = [median(view(model_draws, :, j)) for j in 1:size(model_draws, 2)]
    beta_map = posterior_beta_map(fit.result)
    clinical = posterior_clinical_draws(fit.result)
    if size(clinical, 2) > 0
        clinical_X_val === nothing && throw(ArgumentError("clinical validation columns are required"))
        size(clinical_X_val) == (length(time_val), size(clinical, 2)) || throw(DimensionMismatch("clinical validation data"))
        z_val = hcat(z_val, clinical_X_val)
        beta_mean = vcat(beta_mean, vec(mean(clinical; dims = 1)))
        beta_median = vcat(beta_median, [median(view(clinical, :, j)) for j in axes(clinical, 2)])
        beta_map = vcat(beta_map, clinical[argmax(posterior_logposterior_draws(fit.result)), :])
    elseif clinical_X_val !== nothing
        throw(ArgumentError("clinical validation columns supplied to an unadjusted fit"))
    end
    return Dict(
        "mean" => concordance_index(time_val, event_val, z_val * beta_mean),
        "median" => concordance_index(time_val, event_val, z_val * beta_median),
        "MAP" => concordance_index(time_val, event_val, z_val * beta_map),
    )
end

function _metric_row(metric_row)
    row = Dict{Symbol,Any}()
    for col in propertynames(metric_row)
        row[col] = metric_row[col]
    end
    row[:estimate] = string(metric_row[:estimate])
    return row
end

function _prior_failed_row(; mode, dataset_id, metadata_cols, prior, config_index, prior_config_index, fold, cfg, err)
    row = Dict{Symbol,Any}(
        :stage => "prior",
        :status => "failed",
        :error_message => _error_message(err),
        :prior => prior,
        :config_index => config_index,
        :prior_config_index => prior_config_index,
        :fold => fold,
        :estimate => "mean",
        :validation_Cindex => missing,
        :CV_Cindex => missing,
    )
    _add_common_row_fields!(row, metadata_cols, mode, dataset_id)
    _add_config_fields!(row, cfg)
    return row
end

function _summarize_prior_config(rows, cfg; mode, dataset_id, metadata_cols, prior, config_index, prior_config_index, k)
    failed = [get(r, :error_message, "") for r in rows if get(r, :status, "") == "failed"]
    mean_vals = _finite_values(rows, :validation_Cindex; estimate = "mean")
    median_vals = _finite_values(rows, :validation_Cindex; estimate = "median")
    map_vals = _finite_values(rows, :validation_Cindex; estimate = "MAP")
    n_success = length(unique(get(r, :fold, missing) for r in rows if get(r, :status, "") == "success"))
    row = Dict{Symbol,Any}(
        :stage => "prior",
        :status => n_success == k ? "success" : n_success == 0 ? "failed" : "partial_success",
        :error_message => isempty(failed) ? "" : join(unique(failed), " | "),
        :prior => prior,
        :config_index => config_index,
        :prior_config_index => prior_config_index,
        :n_folds => k,
        :n_successful_folds => n_success,
        :CV_Cindex_mean => _mean_or_missing(mean_vals),
        :CV_Cindex_sd => _sd_or_missing(mean_vals),
        :CV_Cindex_se => _se_or_missing(mean_vals),
        :CV_Cindex_median_mean => _mean_or_missing(median_vals),
        :CV_Cindex_median_se => _se_or_missing(median_vals),
        :CV_Cindex_MAP_mean => _mean_or_missing(map_vals),
        :CV_Cindex_MAP_se => _se_or_missing(map_vals),
    )
    alpha_effective = _finite_values(rows, :alpha_prior_shape; estimate = "mean")
    row[:alpha_prior_shape] = isempty(alpha_effective) ? missing : first(alpha_effective)
    for metric in (:ELPD_LOO, :WAIC_pw1, :WAIC_pw2, :NLPML, :LOOIC, :ParetoK_max, :Cindex, :MSE, :FDR, :FP, :FN, :TP, :TN)
        vals = _finite_values(rows, metric; estimate = "mean")
        row[Symbol(String(metric), "_mean")] = _mean_or_missing(vals)
    end
    _add_common_row_fields!(row, metadata_cols, mode, dataset_id)
    _add_config_fields!(row, cfg)
    return row
end

function _run_prior_tuning(
    X,
    time,
    event;
    priors,
    base_sampler_kwargs,
    base_prior_kwargs,
    base_car_kwargs,
    base_model_kwargs,
    mode::Symbol,
    dataset_id::AbstractString,
    cv_folds::Integer,
    seed::Integer,
    realdata::Bool,
    simulation_metadata,
)
    k = min(Int(cv_folds), size(X, 1))
    folds = _kfold_indices(event; k = k, seed = seed)
    beta_true = _metadata_beta(simulation_metadata)
    metadata_cols = _metadata_columns(simulation_metadata)
    metric_rows = Vector{Dict{Symbol,Any}}()
    summary_rows = Vector{Dict{Symbol,Any}}()
    config_records = Dict{Int,Tuple{String,Dict{Symbol,Any}}}()
    config_index = 0
    for prior in priors
        grid = get(PRIOR_PARAM_GRIDS, String(prior), [Dict{String,Any}()])
        for (prior_config_index, raw_cfg) in enumerate(grid)
            config_index += 1
            cfg = _tuning_dict(raw_cfg)
            config_records[config_index] = (String(prior), cfg)
            local_rows = Vector{Dict{Symbol,Any}}()
            for (fold, val_idx) in enumerate(folds)
                train_idx = setdiff(1:length(time), val_idx)
                fit_kwargs = _actual_fit_kwargs(
                    String(prior),
                    base_sampler_kwargs,
                    base_prior_kwargs,
                    base_car_kwargs,
                    base_model_kwargs,
                    cfg;
                    p = size(X, 2),
                    realdata = realdata,
                )
                fit_kwargs[:seed] = seed + 10_000 * config_index + fold
                try
                    fit = fit_and_evaluate(
                        X[train_idx, :],
                        time[train_idx],
                        event[train_idx];
                        fit_kwargs...,
                    )
                    validation_c = _validation_cindices(
                        fit,
                        X[train_idx, :],
                        X[val_idx, :],
                        time[val_idx],
                        event[val_idx],
                    )
                    for r in eachrow(fit.metrics)
                        estimate = string(r[:estimate])
                        row = _metric_row(r)
                        row[:stage] = "prior"
                        row[:status] = "success"
                        row[:error_message] = ""
                        row[:prior] = String(prior)
                        row[:config_index] = config_index
                        row[:prior_config_index] = prior_config_index
                        row[:fold] = fold
                        row[:validation_Cindex] = get(validation_c, estimate, missing)
                        row[:CV_Cindex] = row[:validation_Cindex]
                        row[:training_Cindex] = row[:Cindex]
                        row[:p0] = fit.p0 === nothing ? missing : fit.p0
                        row[:alpha_prior_shape] = get(fit_kwargs, :alpha_prior_shape, missing)
                        merge!(row, _simulation_accuracy(fit.summary, beta_true, estimate))
                        _add_common_row_fields!(row, metadata_cols, mode, dataset_id)
                        _add_config_fields!(row, cfg)
                        push!(metric_rows, row)
                        push!(local_rows, row)
                    end
                catch err
                    row = _prior_failed_row(
                        mode = mode,
                        dataset_id = dataset_id,
                        metadata_cols = metadata_cols,
                        prior = String(prior),
                        config_index = config_index,
                        prior_config_index = prior_config_index,
                        fold = fold,
                        cfg = cfg,
                        err = err,
                    )
                    push!(metric_rows, row)
                    push!(local_rows, row)
                end
            end
            push!(
                summary_rows,
                _summarize_prior_config(
                    local_rows,
                    cfg;
                    mode = mode,
                    dataset_id = dataset_id,
                    metadata_cols = metadata_cols,
                    prior = String(prior),
                    config_index = config_index,
                    prior_config_index = prior_config_index,
                    k = k,
                ),
            )
        end
    end
    return (
        metrics = _dict_rows_dataframe(metric_rows),
        summary = _dict_rows_dataframe(summary_rows),
        config_records = config_records,
    )
end

function _algorithm_grid(base_sampler_kwargs)
    sampler = Symbol(_kwget(base_sampler_kwargs, :sampler, :nuts))
    return get(ALGORITHM_PARAM_GRIDS, sampler, [Dict(:sampler => sampler)])
end

function _algorithm_failed_row(; mode, dataset_id, metadata_cols, prior, config_index, algorithm_config_index, prior_cfg, alg_cfg, err)
    row = Dict{Symbol,Any}(
        :stage => "algorithm",
        :status => "failed",
        :error_message => _error_message(err),
        :prior => prior,
        :config_index => config_index,
        :algorithm_config_index => algorithm_config_index,
        :estimate => "mean",
        :converged => false,
    )
    _add_common_row_fields!(row, metadata_cols, mode, dataset_id)
    _add_config_fields!(row, prior_cfg)
    _add_config_fields!(row, alg_cfg; prefix = "alg_")
    return row
end

function _summarize_algorithm_config(rows, prior_cfg, alg_cfg; mode, dataset_id, metadata_cols, prior, config_index, algorithm_config_index)
    success_rows = [r for r in rows if get(r, :status, "") == "success" && get(r, :estimate, "") == "mean"]
    base = isempty(success_rows) ? first(rows) : first(success_rows)
    row = Dict{Symbol,Any}(
        :stage => "algorithm",
        :status => isempty(success_rows) ? "failed" : "success",
        :error_message => isempty(success_rows) ? get(base, :error_message, "") : "",
        :prior => prior,
        :config_index => config_index,
        :algorithm_config_index => algorithm_config_index,
        :converged => get(base, :converged, false),
    )
    for metric in (
        :ELPD_LOO, :WAIC_pw1, :WAIC_pw2, :NLPML, :LOOIC, :ParetoK_max, :Cindex,
        :rhat_max, :rhat_median, :ess_bulk_min, :ess_bulk_median, :ess_tail_min,
        :ess_tail_median, :divergences, :acceptance_rate, :warmup_acceptance_rate,
        :final_step_mean, :warmup_seconds, :sample_seconds, :total_seconds,
        :allocated_mb, :gc_seconds, :gc_percent, :alpha_prior_shape,
        :MSE, :FDR, :FP, :FN, :TP, :TN,
    )
        row[metric] = get(base, metric, missing)
    end
    _add_common_row_fields!(row, metadata_cols, mode, dataset_id)
    _add_config_fields!(row, prior_cfg)
    _add_config_fields!(row, alg_cfg; prefix = "alg_")
    return row
end

function _run_algorithm_tuning(
    X,
    time,
    event;
    priors,
    prior_configs,
    base_sampler_kwargs,
    base_prior_kwargs,
    base_car_kwargs,
    base_model_kwargs,
    mode::Symbol,
    dataset_id::AbstractString,
    seed::Integer,
    realdata::Bool,
    simulation_metadata,
)
    beta_true = _metadata_beta(simulation_metadata)
    metadata_cols = _metadata_columns(simulation_metadata)
    metric_rows = Vector{Dict{Symbol,Any}}()
    summary_rows = Vector{Dict{Symbol,Any}}()
    config_records = Dict{Int,Tuple{String,Dict{Symbol,Any},Dict{Symbol,Any}}}()
    config_index = 0
    alg_grid = _algorithm_grid(base_sampler_kwargs)
    prior_items =
        prior_configs === nothing ?
        [(String(prior), Dict{Symbol,Any}()) for prior in priors] :
        prior_configs
    for (prior, prior_cfg) in prior_items
        for (algorithm_config_index, raw_alg_cfg) in enumerate(alg_grid)
            config_index += 1
            alg_cfg = _tuning_dict(raw_alg_cfg)
            config_records[config_index] = (String(prior), prior_cfg, alg_cfg)
            local_rows = Vector{Dict{Symbol,Any}}()
            fit_kwargs = _actual_fit_kwargs(
                String(prior),
                base_sampler_kwargs,
                base_prior_kwargs,
                base_car_kwargs,
                base_model_kwargs,
                merge(copy(prior_cfg), alg_cfg);
                p = size(X, 2),
                realdata = realdata,
            )
            fit_kwargs[:seed] = seed + 100_000 + 10_000 * config_index
            try
                fit = fit_and_evaluate(X, time, event; fit_kwargs...)
                diagnostics = _fit_diagnostics(fit.result)
                for r in eachrow(fit.metrics)
                    estimate = string(r[:estimate])
                    row = _metric_row(r)
                    row[:stage] = "algorithm"
                    row[:status] = "success"
                    row[:error_message] = ""
                    row[:prior] = String(prior)
                    row[:config_index] = config_index
                    row[:algorithm_config_index] = algorithm_config_index
                    row[:p0] = fit.p0 === nothing ? missing : fit.p0
                    row[:alpha_prior_shape] = get(fit_kwargs, :alpha_prior_shape, missing)
                    merge!(row, diagnostics)
                    merge!(row, _simulation_accuracy(fit.summary, beta_true, estimate))
                    _add_common_row_fields!(row, metadata_cols, mode, dataset_id)
                    _add_config_fields!(row, prior_cfg)
                    _add_config_fields!(row, alg_cfg; prefix = "alg_")
                    push!(metric_rows, row)
                    push!(local_rows, row)
                end
            catch err
                row = _algorithm_failed_row(
                    mode = mode,
                    dataset_id = dataset_id,
                    metadata_cols = metadata_cols,
                    prior = String(prior),
                    config_index = config_index,
                    algorithm_config_index = algorithm_config_index,
                    prior_cfg = prior_cfg,
                    alg_cfg = alg_cfg,
                    err = err,
                )
                push!(metric_rows, row)
                push!(local_rows, row)
            end
            push!(
                summary_rows,
                _summarize_algorithm_config(
                    local_rows,
                    prior_cfg,
                    alg_cfg;
                    mode = mode,
                    dataset_id = dataset_id,
                    metadata_cols = metadata_cols,
                    prior = String(prior),
                    config_index = config_index,
                    algorithm_config_index = algorithm_config_index,
                ),
            )
        end
    end
    return (
        metrics = _dict_rows_dataframe(metric_rows),
        summary = _dict_rows_dataframe(summary_rows),
        config_records = config_records,
    )
end

function _row_dict(df::DataFrame, i::Integer)
    return Dict(String(c) => df[i, c] for c in propertynames(df))
end

function _best_prior_index(summary::DataFrame)
    nrow(summary) == 0 && return nothing
    candidates = Int[]
    for i in 1:nrow(summary)
        get(summary[i, :], :status, "failed") == "failed" && continue
        _finite_number(summary[i, :CV_Cindex_mean]) === nothing && continue
        push!(candidates, i)
    end
    isempty(candidates) && return nothing
    ordered = sort(candidates; by = i -> (
        -Float64(summary[i, :CV_Cindex_mean]),
        something(_finite_number(summary[i, :CV_Cindex_se]), Inf),
        Int(summary[i, :config_index]),
    ))
    return first(ordered)
end
function _best_prior_configs(summary::DataFrame, config_records::Dict{Int,Tuple{String,Dict{Symbol,Any}}})
    nrow(summary) == 0 && return Dict{String,Dict{String,Any}}()
    result = Dict{String,Dict{String,Any}}()
    prior_groups = groupby(summary, :prior)
    for group_df in prior_groups
        prior_name = String(first(group_df[!, :prior]))
        candidates = Int[]
        for i in 1:nrow(group_df)
            get(group_df[i, :], :status, "failed") == "failed" && continue
            _finite_number(group_df[i, :CV_Cindex_mean]) === nothing && continue
            push!(candidates, i)
        end
        isempty(candidates) && continue
        ordered = sort(candidates; by = i -> (
            -Float64(group_df[i, :CV_Cindex_mean]),
            something(_finite_number(group_df[i, :CV_Cindex_se]), Inf),
            Int(group_df[i, :config_index]),
        ))
        best_local_idx = first(ordered)
        config_idx = Int(group_df[best_local_idx, :config_index])
        _, cfg = config_records[config_idx]
        cfg_out = Dict{String,Any}()
        for (k, v) in pairs(cfg)
            k_str = String(k)
            k_str == "p0" && continue
            cfg_out[k_str] = v === nothing ? nothing : v isa Symbol ? String(v) : v
        end
        result[prior_name] = cfg_out
    end
    return result
end


function _best_algorithm_index(summary::DataFrame)
    nrow(summary) == 0 && return nothing
    candidates = [i for i in 1:nrow(summary) if summary[i, :status] != "failed"]
    isempty(candidates) && return nothing
    ordered = sort(candidates; by = i -> (
        Bool(summary[i, :converged]) ? 0 : 1,
        -something(_finite_number(summary[i, :ELPD_LOO]), -Inf),
        something(_finite_number(summary[i, :total_seconds]), Inf),
        Int(summary[i, :config_index]),
    ))
    return first(ordered)
end

function _append_all_results!(tuning_dir::AbstractString, tables...)
    all = DataFrame()
    for table in tables
        table === nothing && continue
        nrow(table) == 0 && continue
        append!(all, table; cols = :union)
    end
    nrow(all) == 0 && return nothing
    _append_or_write_table(joinpath(tuning_dir, TUNING_OUTPUT_FILES.all_results), all)
    return all
end

function tune_hyperparams_cindex(
    X,
    time,
    event;
    priors,
    base_sampler_kwargs,
    base_prior_kwargs,
    base_car_kwargs,
    base_model_kwargs,
    outdir,
    mode::Symbol,
    dataset_id::AbstractString,
    tune_priorparams::Bool,
    tune_algparams::Bool,
    cv_folds::Integer = 5,
    seed::Integer = 1,
    realdata::Bool = false,
    simulation_metadata = Dict{String,Any}(),
    nested_cv::Bool = true,
    outer_folds::Integer = 5,
)
    tune_priorparams || tune_algparams || (realdata && nested_cv) || return nothing
    nested = realdata && nested_cv ? evaluate_nested_cindex(X, time, event;
        priors, base_sampler_kwargs, base_prior_kwargs, base_car_kwargs, base_model_kwargs,
        outdir, dataset_id, outer_folds, inner_folds = cv_folds, seed,
        tune_priorparams, tune_algparams) : nothing
    if realdata
        return _tune_realdata_full_cindex(X, time, event;
            priors, base_sampler_kwargs, base_prior_kwargs, base_car_kwargs, base_model_kwargs,
            outdir, dataset_id, cv_folds, seed, tune_priorparams, tune_algparams,
            simulation_metadata, nested)
    end
    tuning_dir = mkpath(joinpath(outdir, "tuning"))
    prior_result = nothing
    algorithm_result = nothing
    best_prior_idx = nothing
    best_algorithm_idx = nothing
    algorithm_prior_configs = nothing
    best_prior_configs = nothing

    if tune_priorparams
        prior_result = _run_prior_tuning(
            X,
            time,
            event;
            priors = priors,
            base_sampler_kwargs = base_sampler_kwargs,
            base_prior_kwargs = base_prior_kwargs,
            base_car_kwargs = base_car_kwargs,
            base_model_kwargs = base_model_kwargs,
            mode = mode,
            dataset_id = dataset_id,
            cv_folds = cv_folds,
            seed = seed,
            realdata = realdata,
            simulation_metadata = simulation_metadata,
        )
        _append_or_write_table(joinpath(tuning_dir, TUNING_OUTPUT_FILES.prior_metrics), prior_result.metrics)
        _append_or_write_table(joinpath(tuning_dir, TUNING_OUTPUT_FILES.prior_summary), prior_result.summary)
        best_prior_idx = _best_prior_index(prior_result.summary)
        if best_prior_idx !== nothing
            cfg_index = Int(prior_result.summary[best_prior_idx, :config_index])
            prior, cfg = prior_result.config_records[cfg_index]
            algorithm_prior_configs = [(prior, cfg)]
        end
        best_prior_configs = _best_prior_configs(prior_result.summary, prior_result.config_records)
        _write_tuning_json(joinpath(tuning_dir, "best_prior_configs.json"), best_prior_configs)
    end

    if tune_algparams
        if tune_priorparams && algorithm_prior_configs === nothing
            row = Dict{Symbol,Any}(
                :stage => "algorithm",
                :status => "failed",
                :error_message => "prior tuning produced no successful config",
                :mode => String(mode),
                :dataset_id => String(dataset_id),
                :converged => false,
            )
            for (k, v) in pairs(_metadata_columns(simulation_metadata))
                row[k] = v
            end
            failed_table = _dict_rows_dataframe([row])
            algorithm_result = (
                metrics = failed_table,
                summary = failed_table,
                config_records = Dict{Int,Tuple{String,Dict{Symbol,Any},Dict{Symbol,Any}}}(),
            )
        else
            algorithm_result = _run_algorithm_tuning(
                X,
                time,
                event;
                priors = priors,
                prior_configs = tune_priorparams ? algorithm_prior_configs : nothing,
                base_sampler_kwargs = base_sampler_kwargs,
                base_prior_kwargs = base_prior_kwargs,
                base_car_kwargs = base_car_kwargs,
                base_model_kwargs = base_model_kwargs,
                mode = mode,
                dataset_id = dataset_id,
                seed = seed,
                realdata = realdata,
                simulation_metadata = simulation_metadata,
            )
        end
        _append_or_write_table(joinpath(tuning_dir, TUNING_OUTPUT_FILES.algorithm_metrics), algorithm_result.metrics)
        _append_or_write_table(joinpath(tuning_dir, TUNING_OUTPUT_FILES.algorithm_summary), algorithm_result.summary)
        best_algorithm_idx = _best_algorithm_index(algorithm_result.summary)
    end

    _append_all_results!(
        tuning_dir,
        prior_result === nothing ? nothing : prior_result.summary,
        algorithm_result === nothing ? nothing : algorithm_result.summary,
    )

    best = Dict{String,Any}(
        "dataset_id" => dataset_id,
        "mode" => String(mode),
        "tune_priorparams" => tune_priorparams,
        "tune_algparams" => tune_algparams,
        "cv_folds" => cv_folds,
        "seed" => seed,
        "best_prior_config" => best_prior_idx === nothing ? nothing : _row_dict(prior_result.summary, best_prior_idx),
        "best_algorithm_config" => best_algorithm_idx === nothing ? nothing : _row_dict(algorithm_result.summary, best_algorithm_idx),
    )
    _write_tuning_json(joinpath(tuning_dir, TUNING_OUTPUT_FILES.best_config), best)

    return (
        tuning_dir = tuning_dir,
        prior = prior_result,
        algorithm = algorithm_result,
        best = best,
        best_prior_configs = best_prior_configs,
        nested = nested,
    )
end



# All selection is confined to outer training data.

function _cindex_candidate_configs(prior, sampler_kwargs;
    tune_priorparams, tune_algparams, prior_grids)
    prior_grid = tune_priorparams ? get(prior_grids, String(prior), nothing) : [Dict{Symbol,Any}()]
    prior_grid === nothing && throw(ArgumentError("no tuning grid for $prior"))
    algorithms = tune_algparams ? _algorithm_grid(sampler_kwargs) : [Dict{Symbol,Any}()]
    configs = [_merge_tuning_kwargs(p, a) for p in prior_grid for a in algorithms]
    isempty(configs) && throw(ArgumentError("empty tuning grid for $prior"))
    return configs
end

function _cindex_fit_kwargs(prior, sampler, prior_kwargs, car, model, config, p, seed;
    clinical_columns = 0)
    kwargs = _actual_fit_kwargs(prior, sampler, prior_kwargs, car, model, config;
        p, realdata = true, clinical_columns)
    kwargs[:seed] = seed
    kwargs[:compute_metrics] = false
    # Explicit p0 remains fixed; auto p0 is recalculated by each training fit.
    fixed_p0 = _kwget(model, :p0, nothing)
    fixed_p0 !== nothing && get(config, :p0, nothing) === nothing && (kwargs[:p0] = fixed_p0)
    return kwargs
end

function _select_inner_cindex(X, time, event; prior, configs, base_sampler_kwargs,
    base_prior_kwargs, base_car_kwargs, base_model_kwargs, cv_folds, seed,
    fit_function, score_function, clinical_X = nothing, checkpoint_dir = nothing,
    config_indices = eachindex(configs))
    folds = _kfold_indices(event; k = cv_folds, seed)
    checkpoint_dir === nothing || _loo_checkpoint_manifest(
        clinical_X === nothing ? X : hcat(X, clinical_X), time, event,
        Dict(:prior => prior, :configs => configs, :sampler => base_sampler_kwargs,
            :prior_kwargs => base_prior_kwargs, :car => base_car_kwargs, :model => base_model_kwargs,
            :folds => cv_folds, :seed => seed), checkpoint_dir, true; mode = "inner_cv")
    rows = Dict{Symbol,Any}[]
    scores = fill(-Inf, length(configs))
    for (config_index, config) in enumerate(configs)
        config_index in config_indices || continue
        values = Float64[]
        for (fold, val_idx) in enumerate(folds)
            train_idx = setdiff(eachindex(time), val_idx)
            row = Dict{Symbol,Any}(:config_index => config_index, :fold => fold,
                :prior => prior, :n_train => length(train_idx), :n_validation => length(val_idx),
                :status => "failed", :validation_Cindex => missing, :error_message => "")
            _add_config_fields!(row, config)
            checkpoint = checkpoint_dir === nothing ? nothing :
                joinpath(checkpoint_dir, "config_$(lpad(config_index, 4, '0'))_fold_$(fold).json")
            if checkpoint !== nothing && isfile(checkpoint)
                saved = JSON.parse(read(checkpoint, String))
                if get(saved, "status", "") == "success"
                    saved_score = _finite_number(get(saved, "validation_Cindex", nothing))
                    if saved_score !== nothing
                        push!(rows, Dict{Symbol,Any}(Symbol(k) => v === nothing ? missing : v for (k,v) in saved))
                        push!(values, saved_score)
                        continue
                    end
                end
            end
            try
                kwargs = _cindex_fit_kwargs(prior, base_sampler_kwargs, base_prior_kwargs,
                    base_car_kwargs, base_model_kwargs, config, size(X, 2),
                    seed + 10_000 * config_index + fold;
                    clinical_columns = clinical_X === nothing ? 0 : size(clinical_X, 2))
                clinical_X !== nothing && (kwargs[:clinical_X] = clinical_X[train_idx, :])
                fit = fit_function(X[train_idx, :], time[train_idx], event[train_idx]; kwargs...)
                if hasproperty(fit, :result) && fit.result isa CoxSamplerResult
                    for (key, value) in pairs(_fit_diagnostics(fit.result))
                        row[Symbol("fit_", key)] = value
                    end
                end
                clinical_score_kwargs = clinical_X === nothing ? NamedTuple() : (clinical_X_val = clinical_X[val_idx, :],)
                vc = score_function(fit, X[train_idx, :], X[val_idx, :], time[val_idx], event[val_idx];
                    clinical_score_kwargs...)
                c = _finite_number(get(vc, "mean", missing))
                c === nothing && error("inner validation fold has no finite C-index")
                row[:status] = "success"
                row[:validation_Cindex] = c
                row[:validation_Cindex_median] = get(vc, "median", missing)
                row[:validation_Cindex_MAP] = get(vc, "MAP", missing)
                push!(values, c)
            catch err
                err isa InterruptException && rethrow()
                row[:error_message] = sprint(showerror, err)
            end
            push!(rows, row)
            checkpoint === nothing || _loo_atomic_json(checkpoint, _json_ready(row))
        end
        # Never select a configuration on a convenient subset of successful folds.
        length(values) == cv_folds && (scores[config_index] = mean(values))
    end
    best = argmax(scores) # deterministic grid order breaks ties
    return (config = isfinite(scores[best]) ? configs[best] : nothing,
        config_index = best, score = scores[best], metrics = _dict_rows_dataframe(rows), folds = folds)
end

# The final full-data model uses the same selection rule as an outer training fit.
# These scores are saved for tuning diagnostics only, never as test performance.
function _tune_realdata_full_cindex(X, time, event; priors, base_sampler_kwargs,
    base_prior_kwargs, base_car_kwargs, base_model_kwargs, outdir, dataset_id,
    cv_folds, seed, tune_priorparams, tune_algparams, simulation_metadata, nested)
    tuning_dir = mkpath(joinpath(outdir, "tuning"))
    best_priors = Dict{String,Any}()
    best_algorithms = Dict{String,Any}()
    summaries = Dict{Symbol,Any}[]
    metrics = DataFrame()
    algorithm_keys = tune_algparams ? Set(k for cfg in _algorithm_grid(base_sampler_kwargs) for k in keys(cfg)) : Set{Symbol}()
    if tune_priorparams || tune_algparams
        for prior in String.(priors)
            configs = _cindex_candidate_configs(prior, base_sampler_kwargs;
                tune_priorparams, tune_algparams, prior_grids = PRIOR_PARAM_GRIDS)
            selected = _select_inner_cindex(X, time, event; prior, configs,
                base_sampler_kwargs, base_prior_kwargs, base_car_kwargs, base_model_kwargs,
                cv_folds, seed, fit_function = fit_and_evaluate, score_function = _validation_cindices)
            append!(metrics, selected.metrics; cols = :union)
            for (index, cfg) in enumerate(configs)
                sub = selected.metrics[selected.metrics.config_index .== index, :]
                ok = nrow(sub) == cv_folds && all(==("success"), sub.status)
                row = Dict{Symbol,Any}(:prior => prior, :config_index => index,
                    :prior_config_index => index, :dataset_id => dataset_id,
                    :mode => "realdata", :stage => "prior", :evaluation => "inner_tuning",
                    :status => ok ? "success" : "failed", :n_folds => cv_folds,
                    :n_successful_folds => count(==("success"), sub.status))
                merge!(row, _metadata_columns(simulation_metadata))
                for (col, mean_key, se_key) in ((:validation_Cindex, :CV_Cindex_mean, :CV_Cindex_se),
                    (:validation_Cindex_median, :CV_Cindex_median_mean, :CV_Cindex_median_se),
                    (:validation_Cindex_MAP, :CV_Cindex_MAP_mean, :CV_Cindex_MAP_se))
                    vals = col in propertynames(sub) ? filter(isfinite, Float64.(collect(skipmissing(sub[!, col])))) : Float64[]
                    valid = ok && length(vals) == cv_folds
                    row[mean_key] = valid ? mean(vals) : missing
                    row[se_key] = valid ? std(vals) / sqrt(cv_folds) : missing
                    col == :validation_Cindex && (row[:CV_Cindex_sd] = valid ? std(vals) : missing)
                end
                _add_config_fields!(row, cfg)
                push!(summaries, row)
            end
            _metrics_write_table(joinpath(tuning_dir, TUNING_OUTPUT_FILES.prior_metrics), metrics)
            _metrics_write_table(joinpath(tuning_dir, TUNING_OUTPUT_FILES.prior_summary), _dict_rows_dataframe(summaries))
            selected.config === nothing && error("no complete full-data tuning configuration for $prior")
            best_priors[prior] = Dict(String(k) => v for (k, v) in selected.config if k != :p0 && !(k in algorithm_keys))
            best_algorithms[prior] = Dict("alg_" * String(k) => v for (k, v) in selected.config if k in algorithm_keys)
        end
    end
    best = Dict{String,Any}("dataset_id" => dataset_id, "mode" => "realdata",
        "cv_folds" => cv_folds, "seed" => seed, "evaluation" => "inner_tuning",
        "tune_priorparams" => tune_priorparams, "tune_algparams" => tune_algparams,
        "best_algorithm_configs" => best_algorithms, "best_algorithm_config" => nothing)
    _write_tuning_json(joinpath(tuning_dir, "best_prior_configs.json"), best_priors)
    _write_tuning_json(joinpath(tuning_dir, TUNING_OUTPUT_FILES.best_config), best)
    summary = _dict_rows_dataframe(summaries)
    nrow(summary) > 0 && _metrics_write_table(joinpath(tuning_dir, TUNING_OUTPUT_FILES.all_results), summary)
    return (tuning_dir = tuning_dir, prior = (metrics = metrics, summary = summary),
        algorithm = nothing, best = best, best_prior_configs = best_priors, nested = nested)
end

function _nested_cindex_summary(rows, priors; dataset_id, constraint, outer_folds, inner_folds)
    summaries = Dict{Symbol,Any}[]
    for prior in priors
        local_rows = [r for r in rows if r[:prior] == prior]
        n_success = count(r -> r[:status] == "success", local_rows)
        complete = n_success == outer_folds
        row = Dict{Symbol,Any}(:dataset_id => dataset_id, :constraint => string(constraint),
            :prior => prior, :evaluation => "nested_cv", :n_folds => outer_folds,
            :inner_folds => inner_folds, :n_successful_folds => n_success,
            :status => complete ? "success" : "incomplete")
        for (estimate, mean_key, sd_key, se_key) in (
            ("mean", :CV_Cindex_mean, :CV_Cindex_sd, :CV_Cindex_se),
            ("median", :CV_Cindex_median_mean, :CV_Cindex_median_sd, :CV_Cindex_median_se),
            ("MAP", :CV_Cindex_MAP_mean, :CV_Cindex_MAP_sd, :CV_Cindex_MAP_se))
            vals = _finite_values(local_rows, Symbol("test_Cindex_", estimate))
            valid = complete && length(vals) == outer_folds
            row[mean_key] = valid ? mean(vals) : missing
            row[sd_key] = valid ? std(vals) : missing
            row[se_key] = valid ? std(vals) / sqrt(outer_folds) : missing
        end
        push!(summaries, row)
    end
    return _dict_rows_dataframe(summaries)
end

function _write_nested_table(path, table, priors)
    # Match the full-fit tables: rerunning one method keeps other methods' rows.
    if isfile(path)
        old = CSV.read(path, DataFrame)
        if :prior in propertynames(old)
            keep = [!(string(p) in priors) for p in old.prior]
            table = vcat(old[keep, :], table; cols = :union)
        end
    end
    return _metrics_write_table(path, table)
end

"""
Nested C-index evaluation: select hyperparameters on inner folds, refit on the
entire outer training fold, then score its untouched outer test fold. Report an
unweighted mean of within-test-fold concordances, never pooled cross-model pairs.
Completed outer folds are resumable; incomplete evaluations have missing scores.
Input columns and candidate grids must be chosen without using outer outcomes.
"""
function evaluate_nested_cindex(X, time, event; priors, base_sampler_kwargs,
    base_prior_kwargs, base_car_kwargs, base_model_kwargs, outdir,
    dataset_id::AbstractString, outer_folds::Integer = 5, inner_folds::Integer = 5,
    seed::Integer = 1, tune_priorparams::Bool = true, tune_algparams::Bool = false,
    prior_grids = PRIOR_PARAM_GRIDS, fit_function = fit_and_evaluate,
    score_function = _validation_cindices, clinical_X = nothing)
    n = length(time)
    size(X, 1) == n == length(event) || throw(DimensionMismatch("inconsistent data lengths"))
    clinical_X !== nothing && size(clinical_X, 1) != n && throw(DimensionMismatch("clinical row count"))
    2 <= outer_folds <= n || throw(ArgumentError("outer folds must be between 2 and n"))
    outer = _kfold_indices(event; k = outer_folds, seed)
    all(idx -> length(idx) >= 2 && n - length(idx) >= inner_folds >= 2, outer) ||
        throw(ArgumentError("not enough subjects for the requested inner/outer folds"))
    priors = String.(priors)
    configs = Dict(prior => _cindex_candidate_configs(prior, base_sampler_kwargs;
        tune_priorparams, tune_algparams, prior_grids) for prior in priors)
    settings = Dict(:sampler => _tuning_dict(base_sampler_kwargs),
        :prior => _tuning_dict(base_prior_kwargs), :car => _tuning_dict(base_car_kwargs),
        :model => _tuning_dict(base_model_kwargs), :configs => configs,
        :outer_folds => outer_folds, :inner_folds => inner_folds, :seed => seed,
        :dataset_id => dataset_id, :evaluation => "nested_cv_v1")
    clinical_X !== nothing && (settings[:clinical_columns] = size(clinical_X, 2))
    # Reuse the data/settings/code fingerprint machinery, with separate checkpoints
    # for each prior so adding another method does not invalidate completed methods.
    root = mkpath(joinpath(outdir, "nested_cv"))
    rows = Dict{Symbol,Any}[]
    constraint = _kwget(base_model_kwargs, :constraint, :soft_center)
    for prior in priors
        prior_dir = joinpath(root, prior)
        prior_settings = merge(settings, Dict(:configs => configs[prior], :prior_name => prior))
        fingerprint_X = clinical_X === nothing ? X : hcat(X, clinical_X)
        fingerprint = _loo_checkpoint_manifest(fingerprint_X, time, event, prior_settings, prior_dir, true; mode = "nested_cv")
        for (fold, test_idx) in enumerate(outer)
            train_idx = setdiff(eachindex(time), test_idx)
            checkpoint = joinpath(prior_dir, "outer_$(lpad(fold, 2, '0')).json")
            if isfile(checkpoint)
                saved = JSON.parsefile(checkpoint; use_mmap = false)
                if get(saved, "fingerprint", "") == fingerprint && get(saved, "status", "") == "success"
                    push!(rows, Dict{Symbol,Any}(Symbol(k) => v === nothing ? missing : v for (k, v) in saved))
                    continue
                end
            end
            @info "Nested C-index outer fold" dataset_id prior fold outer_folds
            row = Dict{Symbol,Any}(:dataset_id => dataset_id, :constraint => string(constraint),
                :prior => prior, :outer_fold => fold, :n_train => length(train_idx),
                :n_test => length(test_idx), :evaluation => "nested_cv", :status => "failed",
                :fingerprint => fingerprint, :error_message => "",
                :test_Cindex_mean => missing, :test_Cindex_median => missing, :test_Cindex_MAP => missing)
            fold_dir = mkpath(joinpath(prior_dir, "outer_$(lpad(fold, 2, '0'))"))
            try
                inner = _select_inner_cindex(X[train_idx, :], time[train_idx], event[train_idx];
                    prior, configs = configs[prior], base_sampler_kwargs, base_prior_kwargs,
                    base_car_kwargs, base_model_kwargs, cv_folds = inner_folds,
                    seed = seed + 1_000_000 * fold, fit_function, score_function,
                    clinical_X = clinical_X === nothing ? nothing : clinical_X[train_idx, :],
                    checkpoint_dir = joinpath(fold_dir, "inner_checkpoints"))
                _metrics_write_table(joinpath(fold_dir, "inner_cv_metrics.csv"), inner.metrics)
                _write_tuning_json(joinpath(fold_dir, "split_indices.json"), Dict(
                    "outer_train" => train_idx, "outer_test" => test_idx,
                    "inner_validation" => [train_idx[idx] for idx in inner.folds]))
                inner.config === nothing && error("no configuration succeeded in every inner fold")
                row[:inner_Cindex_mean] = inner.score
                row[:selected_config_index] = inner.config_index
                row[:selected_config] = JSON.json(_json_ready(inner.config))
                _write_tuning_json(joinpath(fold_dir, "selected_config.json"), inner.config)
                kwargs = _cindex_fit_kwargs(prior, base_sampler_kwargs, base_prior_kwargs,
                    base_car_kwargs, base_model_kwargs, inner.config, size(X, 2),
                    seed + 1_000_000 * fold + 999_999;
                    clinical_columns = clinical_X === nothing ? 0 : size(clinical_X, 2))
                clinical_X !== nothing && (kwargs[:clinical_X] = clinical_X[train_idx, :])
                fit = fit_function(X[train_idx, :], time[train_idx], event[train_idx]; kwargs...)
                if hasproperty(fit, :result) && fit.result isa CoxSamplerResult
                    for (key, value) in pairs(_fit_diagnostics(fit.result))
                        row[Symbol("fit_", key)] = value
                    end
                    # Keep the outer posterior so later metrics/diagnostics do not
                    # require fitting it again. Outer outcomes never enter the fit.
                    save_run_outputs(fit.result; output_dir = joinpath(fold_dir, "fit"),
                        run_id = "outer_$(lpad(fold, 2, '0'))_$(prior)",
                        metadata = Dict("dataset_name" => dataset_id, "method_name" => prior,
                            "evaluation" => "nested_cv_outer_training", "outer_fold" => fold,
                            "n" => length(train_idx), "p" => size(X, 2), "fingerprint" => fingerprint,
                            "selected_config" => _json_ready(inner.config)),
                        posterior_summary = fit.summary, save_jld2 = false, save_wide_csv = true)
                end
                clinical_score_kwargs = clinical_X === nothing ? NamedTuple() : (clinical_X_val = clinical_X[test_idx, :],)
                vc = score_function(fit, X[train_idx, :], X[test_idx, :], time[test_idx], event[test_idx];
                    clinical_score_kwargs...)
                for estimate in ("mean", "median", "MAP")
                    c = _finite_number(get(vc, estimate, missing))
                    c === nothing && error("outer test fold has no finite $estimate C-index")
                    row[Symbol("test_Cindex_", estimate)] = c
                end
                row[:status] = "success"
            catch err
                err isa InterruptException && rethrow()
                row[:error_message] = sprint(showerror, err)
                @warn "Nested C-index fold failed; resume to retry" prior fold error=row[:error_message]
            end
            _loo_atomic_json(checkpoint, _json_ready(row))
            push!(rows, row)
            _write_nested_table(joinpath(root, "nested_cv_folds.csv"), _dict_rows_dataframe(rows), priors)
            _write_nested_table(joinpath(root, "nested_cv_summary.csv"), _nested_cindex_summary(rows, priors;
                dataset_id, constraint, outer_folds, inner_folds), priors)
        end
    end
    summary = _nested_cindex_summary(rows, priors; dataset_id, constraint, outer_folds, inner_folds)
    folds = _dict_rows_dataframe(rows)
    _write_nested_table(joinpath(root, "nested_cv_folds.csv"), folds, priors)
    _write_nested_table(joinpath(root, "nested_cv_summary.csv"), summary, priors)
    return (summary = summary, folds = folds)
end
