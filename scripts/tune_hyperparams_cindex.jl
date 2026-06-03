using DataFrames
using Random
using Statistics

# Edit these lists directly to change the tuning search space.  The CLI does
# not accept grid values by design.
const PRIOR_PARAM_GRIDS = Dict{String,Vector{Dict{String,Any}}}(
    "hs" => [
        Dict("p0" => nothing, "hs_local_scale" => 1.0, "hs_tau_scale" => 1.0),
        Dict("p0" => 3, "hs_local_scale" => 0.5, "hs_tau_scale" => 1.0),
    ],
    "rhs" => [
        Dict("p0" => nothing, "rhs_local_scale" => 1.0, "rhs_tau_scale" => 1.0, "rhs_slab_df" => 4.0, "rhs_slab_s2" => 2.0),
        Dict("p0" => 3, "rhs_local_scale" => 0.5, "rhs_tau_scale" => 1.0, "rhs_slab_df" => 4.0, "rhs_slab_s2" => 1.0),
    ],
    "new-rhs" => [
        Dict("p0" => nothing, "rhs_local_scale" => 1.0, "rhs_slab_df" => 4.0, "rhs_slab_s2" => 2.0),
        Dict("p0" => 3, "rhs_local_scale" => 0.5, "rhs_slab_df" => 4.0, "rhs_slab_s2" => 1.0),
    ],
    "est-rhs" => [
        Dict("p0" => nothing, "rhs_local_scale" => 1.0, "rhs_slab_df" => 4.0, "rhs_slab_s2" => 2.0),
        Dict("p0" => 3, "rhs_local_scale" => 0.5, "rhs_slab_df" => 4.0, "rhs_slab_s2" => 1.0),
    ],
    "cox-hs" => [
        Dict("p0" => nothing, "hs_local_scale" => 1.0),
        Dict("p0" => 3, "hs_local_scale" => 0.5),
    ],
    "cox-rhs" => [
        Dict("p0" => nothing, "rhs_local_scale" => 1.0, "rhs_slab_df" => 4.0, "rhs_slab_s2" => 2.0),
        Dict("p0" => 3, "rhs_local_scale" => 0.5, "rhs_slab_df" => 4.0, "rhs_slab_s2" => 1.0),
    ],
    "lasso" => [
        Dict("lasso_lambda" => nothing, "lasso_lambda2_shape" => 1.0, "lasso_lambda2_rate" => 0.1),
        Dict("lasso_lambda" => nothing, "lasso_lambda2_shape" => 1.0, "lasso_lambda2_rate" => 1.0),
    ],
    "sslasso" => [
        Dict("sslasso_spike_scale" => nothing, "sslasso_slab_scale" => nothing, "sslasso_theta_a" => 1.0, "sslasso_theta_b" => 1.0),
        Dict("sslasso_spike_scale" => 0.1, "sslasso_slab_scale" => 1.0, "sslasso_theta_a" => 1.0, "sslasso_theta_b" => 1.0),
    ],
    "prox-lasso" => [
        Dict("lasso_lambda" => nothing, "prox_l1_lambda" => 1e-3, "alpha_prior_shape" => nothing, "alpha_prior_rate" => nothing),
        Dict("lasso_lambda" => nothing, "prox_l1_lambda" => 1e-2, "alpha_prior_shape" => nothing, "alpha_prior_rate" => nothing),
    ],
    "prox-sslasso" => [
        Dict("sslasso_spike_scale" => nothing, "sslasso_slab_scale" => nothing, "prox_l1_lambda" => 1e-3, "alpha_prior_shape" => nothing, "alpha_prior_rate" => nothing),
        Dict("sslasso_spike_scale" => 0.1, "sslasso_slab_scale" => 1.0, "prox_l1_lambda" => 1e-2, "alpha_prior_shape" => nothing, "alpha_prior_rate" => nothing),
    ],
)

const ALGORITHM_PARAM_GRIDS = Dict{Symbol,Vector{Dict{Symbol,Any}}}(
    :nuts => [
        Dict(:sampler => :nuts, :nuts_backend => :turing, :target_accept => 0.8, :max_depth => 10),
        Dict(:sampler => :nuts, :nuts_backend => :explicit, :target_accept => 0.9, :max_depth => 10),
    ],
    :nuts_turing => [
        Dict(:sampler => :nuts_turing, :nuts_backend => :turing, :target_accept => 0.8, :max_depth => 10),
        Dict(:sampler => :nuts_turing, :nuts_backend => :turing, :target_accept => 0.9, :max_depth => 10),
    ],
    :nuts_explicit => [
        Dict(:sampler => :nuts_explicit, :nuts_backend => :explicit, :target_accept => 0.8, :max_depth => 10),
        Dict(:sampler => :nuts_explicit, :nuts_backend => :explicit, :target_accept => 0.9, :max_depth => 10),
    ],
    :hmc => [
        Dict(:sampler => :hmc, :hmc_n_leapfrog => 10, :hmc_step_size => nothing),
        Dict(:sampler => :hmc, :hmc_n_leapfrog => 20, :hmc_step_size => nothing),
    ],
    :mh => [
        Dict(:sampler => :mh),
    ],
    :mala => [
        Dict(:sampler => :mala, :mala_step_size => 1e-4, :mala_precondition => false),
        Dict(:sampler => :mala, :mala_step_size => 1e-4, :mala_precondition => true),
    ],
    :ula => [
        Dict(:sampler => :ula, :mala_step_size => 1e-4, :mala_precondition => false),
        Dict(:sampler => :ula, :mala_step_size => 1e-5, :mala_precondition => false),
    ],
    :gibbs => [
        Dict(:sampler => :gibbs),
    ],
)

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
    ensure_dir(dirname(path))
    open(path, "w") do io
        JSON.print(io, _json_ready(obj), 4)
        write(io, "\n")
    end
    return path
end

function _append_or_write_table(path::AbstractString, table::DataFrame)
    ensure_dir(dirname(path))
    if file_complete(path)
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
        push!(table, row; cols = :union)
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
    realdata::Bool,
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

function _training_zero_replacement(X_train::AbstractMatrix)
    positives = Float64[x for x in X_train if x > 0]
    isempty(positives) && throw(ArgumentError("training X contains no positive values."))
    return minimum(positives) / 2
end

function _validation_design_matrix(
    X_train::AbstractMatrix,
    X_val::AbstractMatrix,
    train_data::CoxPreparedData;
    requested_pseudocount,
)
    x = Matrix{Float64}(X_val)
    all(isfinite, x) || throw(ArgumentError("validation X must be finite."))
    all(>=(0), x) || throw(ArgumentError("validation X must be nonnegative."))

    z =
        if train_data.transform == :log
            pc = requested_pseudocount === nothing ? train_data.pseudocount : Float64(requested_pseudocount)
            if pc == 0.0
                if any(iszero, x)
                    replacement = train_data.pseudocount > 0 ? train_data.pseudocount : _training_zero_replacement(X_train)
                    x[x .== 0.0] .= replacement
                end
            else
                x .+= pc
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

function _validation_cindices(fit, X_train, X_val, time_val, event_val; requested_pseudocount)
    z_val = _validation_design_matrix(
        X_train,
        X_val,
        fit.result.data;
        requested_pseudocount = requested_pseudocount,
    )
    model_draws = posterior_beta_draws(fit.result)
    beta_mean = vec(mean(model_draws; dims = 1))
    beta_median = [median(view(model_draws, :, j)) for j in 1:size(model_draws, 2)]
    beta_map = posterior_beta_map(fit.result)
    return Dict(
        "mean" => concordance_index(time_val, event_val, z_val * beta_mean),
        "median" => concordance_index(time_val, event_val, z_val * beta_median),
        "MAP" => concordance_index(time_val, event_val, z_val * beta_map),
    )
end

function _fit_diagnostics(result)
    diag = sampler_diagnostics(result)
    accept = _sampler_acceptance(result)
    chain_diag = _chain_diagnostics(result)
    divergences = _divergence_count(result)
    rhat_max = get(chain_diag, :rhat_max, missing)
    ess_min = get(chain_diag, :ess_bulk_min, missing)
    acc = get(accept, :acceptance_rate, missing)
    rhat_ok = rhat_max === missing || Float64(rhat_max) <= 1.10
    ess_ok = ess_min === missing || Float64(ess_min) >= 10.0
    div_ok = divergences === missing || Int(divergences) == 0
    acc_ok = acc === missing || 0.05 <= Float64(acc) <= 1.0
    out = Dict{Symbol,Any}(
        :warmup_seconds => diag.warmup_seconds,
        :sample_seconds => diag.sample_seconds,
        :total_seconds => diag.total_seconds,
        :allocated_mb => diag.allocated_bytes / 2.0^20,
        :gc_seconds => diag.gc_seconds,
        :gc_percent => diag.gc_percent,
        :divergences => divergences,
        :converged => rhat_ok && ess_ok && div_ok && acc_ok,
    )
    merge!(out, accept)
    merge!(out, chain_diag)
    return out
end

function _sampler_acceptance(result)
    if result.samples isa MHSamples
        return Dict{Symbol,Any}(
            :acceptance_rate => result.samples.acceptance_rate,
            :warmup_acceptance_rate => result.samples.warmup_acceptance_rate,
            :final_step_mean => mean(result.samples.final_step),
        )
    elseif result.samples isa MultiMHSamples
        samples = result.samples.chains
        return Dict{Symbol,Any}(
            :acceptance_rate => mean([s.acceptance_rate for s in samples]),
            :warmup_acceptance_rate => mean([s.warmup_acceptance_rate for s in samples]),
            :final_step_mean => mean([mean(s.final_step) for s in samples]),
        )
    end
    return Dict{Symbol,Any}(
        :acceptance_rate => missing,
        :warmup_acceptance_rate => missing,
        :final_step_mean => missing,
    )
end

function _param_col(df::DataFrame)
    :parameters in propertynames(df) && return :parameters
    :parameter in propertynames(df) && return :parameter
    :name in propertynames(df) && return :name
    return nothing
end

function _first_col(df::DataFrame, cols)
    for col in cols
        col in propertynames(df) && return col
    end
    return nothing
end

function _numeric_col_values(df::DataFrame, col::Symbol)
    vals = Float64[]
    col in propertynames(df) || return vals
    for x in df[!, col]
        xf = _finite_number(x)
        xf === nothing || push!(vals, xf)
    end
    return vals
end

function _chain_diagnostics(result)
    chains = try
        result_to_chains(result)
    catch
        return Dict{Symbol,Any}(
            :rhat_max => missing,
            :rhat_median => missing,
            :ess_bulk_min => missing,
            :ess_bulk_median => missing,
            :ess_tail_min => missing,
            :ess_tail_median => missing,
        )
    end
    stats = try
        MCMCChains = Base.require(Base.PkgId(Base.UUID("c7f686f2-ff18-58e9-bc7b-31028e88f75d"), "MCMCChains"))
        DataFrame(Base.invokelatest(getproperty(MCMCChains, :summarystats), chains))
    catch
        DataFrame()
    end
    isempty(stats) && return Dict{Symbol,Any}(
        :rhat_max => missing,
        :rhat_median => missing,
        :ess_bulk_min => missing,
        :ess_bulk_median => missing,
        :ess_tail_min => missing,
        :ess_tail_median => missing,
    )
    pcol = _param_col(stats)
    pcol === nothing && return Dict{Symbol,Any}()
    params = string.(stats[!, pcol])
    beta_rows = stats[occursin.(r"^beta\[\d+\]$", params), :]
    ess_col = _first_col(stats, (:ess_bulk, :ess, :n_eff))
    tail_col = _first_col(stats, (:ess_tail, :tail_ess))
    rhat_col = _first_col(stats, (:rhat, :r_hat))
    ess = ess_col === nothing ? Float64[] : _numeric_col_values(beta_rows, ess_col)
    tail = tail_col === nothing ? Float64[] : _numeric_col_values(beta_rows, tail_col)
    rhat = rhat_col === nothing ? Float64[] : _numeric_col_values(beta_rows, rhat_col)
    return Dict{Symbol,Any}(
        :rhat_max => isempty(rhat) ? missing : maximum(rhat),
        :rhat_median => isempty(rhat) ? missing : median(rhat),
        :ess_bulk_min => isempty(ess) ? missing : minimum(ess),
        :ess_bulk_median => isempty(ess) ? missing : median(ess),
        :ess_tail_min => isempty(tail) ? missing : minimum(tail),
        :ess_tail_median => isempty(tail) ? missing : median(tail),
    )
end

function _divergence_count(result)
    wide = try
        samples_wide_dataframe(result)
    catch
        return missing
    end
    for col in (:divergent, :is_divergent, :numerical_error)
        if col in propertynames(wide)
            return count(!=(0), Int.(wide[!, col]))
        end
    end
    return missing
end

function _simulation_accuracy(summary::DataFrame, beta_true, estimate::AbstractString)
    if beta_true === nothing
        return Dict{Symbol,Any}(
            :MSE => missing,
            :FDR => missing,
            :FP => missing,
            :FN => missing,
            :TP => missing,
            :TN => missing,
        )
    end
    col = estimate == "MAP" ? :map : Symbol(estimate)
    beta_hat = Float64.(summary[!, col])
    fd = false_discovery(beta_true, summary)
    return Dict{Symbol,Any}(
        :MSE => mean((beta_hat .- beta_true) .^ 2),
        :FDR => fd.FDR,
        :FP => fd.FP,
        :FN => fd.FN,
        :TP => fd.TP,
        :TN => fd.TN,
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
    for metric in (:ELPD_LOO, :WAIC_pw1, :WAIC_pw2, :LPML, :LOOIC, :ParetoK_max, :Cindex, :MSE, :FDR, :FP, :FN, :TP, :TN)
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
                        event[val_idx];
                        requested_pseudocount = get(fit_kwargs, :pseudocount, nothing),
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
        :ELPD_LOO, :WAIC_pw1, :WAIC_pw2, :LPML, :LOOIC, :ParetoK_max, :Cindex,
        :rhat_max, :rhat_median, :ess_bulk_min, :ess_bulk_median, :ess_tail_min,
        :ess_tail_median, :divergences, :acceptance_rate, :warmup_acceptance_rate,
        :final_step_mean, :warmup_seconds, :sample_seconds, :total_seconds,
        :allocated_mb, :gc_seconds, :gc_percent, :MSE, :FDR, :FP, :FN, :TP, :TN,
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
    tune_run_best::Bool = false,
)
    tune_priorparams || tune_algparams || return nothing
    tuning_dir = ensure_dir(joinpath(outdir, "tuning"))
    prior_result = nothing
    algorithm_result = nothing
    best_prior_idx = nothing
    best_algorithm_idx = nothing
    algorithm_prior_configs = nothing

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
        "tune_run_best" => tune_run_best,
    )
    _write_tuning_json(joinpath(tuning_dir, TUNING_OUTPUT_FILES.best_config), best)

    return (
        tuning_dir = tuning_dir,
        prior = prior_result,
        algorithm = algorithm_result,
        best = best,
    )
end
