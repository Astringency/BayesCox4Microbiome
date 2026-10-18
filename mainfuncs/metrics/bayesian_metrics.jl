using PosteriorStats
using JSON

# Bayesian fit criteria on the Cox event partial-likelihood scale.
function logmeanexp(v::AbstractVector)
    isempty(v) && throw(ArgumentError("logmeanexp requires at least one draw"))
    m = maximum(v)
    isinf(m) && return m
    return m + log(mean(exp.(v .- m)))
end

function posterior_event_loglik(data::CoxPreparedData, draws::AbstractMatrix)
    n = length(data.event)
    s = size(draws, 1)
    out = Matrix{Float64}(undef, n, s)
    for r in 1:s
        out[:, r] .= cox_loglik_events(data, view(draws, r, :))
    end
    return out
end

function posterior_draw_layout(result, n_total::Integer)
    if result.samples isa MHSamples
        return (chains = 1, draws = n_total)
    elseif result.samples isa MultiMHSamples
        n_chain = length(result.samples.chains)
        n_iter = size(result.samples.chains[1].draws, 1)
        return (chains = n_chain, draws = n_iter)
    end
    try
        value = getfield(result.samples, :value)
        n_iter, _, n_chain = size(value)
        if n_iter * n_chain == n_total
            return (chains = n_chain, draws = n_iter)
        end
    catch
    end
    return (chains = 1, draws = n_total)
end

function _loo_loglik_array(ll_event::AbstractMatrix, n_chains::Integer, n_draws::Integer)
    n_events, n_total = size(ll_event)
    n_chains * n_draws == n_total ||
        throw(DimensionMismatch("LOO chain/draw layout does not match log-likelihood samples."))
    out = Array{Float64}(undef, n_draws, n_chains, n_events)
    @inbounds for c in 1:n_chains
        offset = (c - 1) * n_draws
        for d in 1:n_draws
            out[d, c, :] .= view(ll_event, :, offset + d)
        end
    end
    return out
end

function psis_loo_metrics(ll_event::AbstractMatrix, n_chains::Integer, n_draws::Integer)
    if n_chains * n_draws < 26 || isempty(ll_event)
        return (
            ELPD_LOO = NaN,
            ELPD_LOO_SE = NaN,
            PLOO = NaN,
            PLOO_SE = NaN,
            LOOIC = NaN,
            LOOIC_SE = NaN,
            ParetoK_max = NaN,
            ParetoK_bad = missing,
        )
    end
    try
        loo_result = PosteriorStats.loo(_loo_loglik_array(ll_event, n_chains, n_draws))
        estimates = loo_result.estimates
        pareto_k = vec(collect(loo_result.pointwise.pareto_shape))
        looic = -2 * estimates.elpd
        return (
            ELPD_LOO = estimates.elpd,
            ELPD_LOO_SE = estimates.se_elpd,
            PLOO = estimates.p,
            PLOO_SE = estimates.se_p,
            LOOIC = looic,
            LOOIC_SE = 2 * estimates.se_elpd,
            ParetoK_max = maximum(pareto_k),
            ParetoK_bad = count(>(0.7), pareto_k),
        )
    catch err
        @warn "PSIS-LOO calculation failed; LOO metrics will be NaN." exception=(err, catch_backtrace())
        return (
            ELPD_LOO = NaN,
            ELPD_LOO_SE = NaN,
            PLOO = NaN,
            PLOO_SE = NaN,
            LOOIC = NaN,
            LOOIC_SE = NaN,
            ParetoK_max = NaN,
            ParetoK_bad = missing,
        )
    end
end

"""Bayesian fit criteria for a units-by-draws log-likelihood matrix.
WAIC, AIC, BIC, DIC and NLPML use the smaller-is-better convention.
`deviance_at_mean` is -2 log likelihood at the posterior mean parameters.
For Cox models the units are events conditional on fixed full-data risk sets;
use `evaluate_cox_loo(mode=:subject_refit)` for subject-deletion refitting.
"""
function bayesian_fit_metrics(event_loglik::AbstractMatrix;
    point_loglik::Real, n_parameters::Integer, n_observations::Integer,
    deviance_at_mean::Real = -2 * point_loglik,
    chains::Integer = 1, draws_per_chain::Integer = div(size(event_loglik, 2), chains),
    loo = nothing)
    chains > 0 && draws_per_chain > 0 || throw(ArgumentError("positive chain/draw counts required"))
    chains * draws_per_chain == size(event_loglik, 2) || throw(DimensionMismatch("chain/draw layout mismatch"))
    n_parameters >= 0 && n_observations > 0 || throw(ArgumentError("invalid model dimensions"))
    loo_metrics = loo === nothing ? psis_loo_metrics(event_loglik, chains, draws_per_chain) : loo
    lppd = [logmeanexp(view(event_loglik, i, :)) for i in axes(event_loglik, 1)]
    mean_ll = vec(mean(event_loglik; dims = 2))
    pw1 = 2 * sum(lppd .- mean_ll)
    pw2 = size(event_loglik, 2) > 1 ? sum(var(view(event_loglik, i, :)) for i in axes(event_loglik, 1); init = 0.0) : NaN
    nlpml = sum(logmeanexp(-view(event_loglik, i, :)) for i in axes(event_loglik, 1); init = 0.0)
    mean_deviance = -2 * mean(vec(sum(event_loglik; dims = 1)))
    return merge((
        loglik = Float64(point_loglik), NLPML = nlpml,
        AIC = -2 * point_loglik + 2 * n_parameters,
        BIC = -2 * point_loglik + log(n_observations) * n_parameters,
        WAIC_pw1 = -2 * sum(lppd) + 2 * pw1,
        WAIC_pw2 = -2 * sum(lppd) + 2 * pw2,
        DIC = 2 * mean_deviance - deviance_at_mean,
    ), (; (key => getproperty(loo_metrics, key) for key in
        (:ELPD_LOO, :ELPD_LOO_SE, :PLOO, :PLOO_SE, :LOOIC, :LOOIC_SE, :ParetoK_max, :ParetoK_bad))...))
end

function fit_metrics(data::CoxPreparedData, draws::AbstractMatrix, beta_hat::AbstractVector;
    chains::Integer = 1, draws_per_chain::Integer = div(size(draws, 1), chains),
    ll_event = nothing, loo = nothing)
    event_loglik = ll_event === nothing ? posterior_event_loglik(data, draws)[findall(!=(0), data.event), :] : ll_event
    criteria = bayesian_fit_metrics(event_loglik;
        point_loglik = cox_logpartiallikelihood(data, beta_hat),
        n_parameters = length(beta_hat), n_observations = length(data.time),
        deviance_at_mean = -2 * cox_logpartiallikelihood(data, vec(mean(draws; dims = 1))),
        chains, draws_per_chain, loo)
    return merge(criteria, (Cindex = concordance_index(data.time, data.event, data.z * beta_hat),))
end

"""Evaluate named coefficient estimates, sharing posterior likelihood and PSIS work."""
function posterior_fit_metrics(data::CoxPreparedData, draws::AbstractMatrix, estimates;
    chains::Integer = 1, draws_per_chain::Integer = div(size(draws, 1), chains))
    ll_event = posterior_event_loglik(data, draws)[findall(!=(0), data.event), :]
    loo = psis_loo_metrics(ll_event, chains, draws_per_chain)
    return DataFrame([merge((estimate = string(name),), fit_metrics(data, draws, beta;
        chains, draws_per_chain, ll_event, loo)) for (name, beta) in pairs(estimates)])
end

"""Evaluate a sampler result on its model scale and summarize coefficients on their original scale."""
function evaluate_fit(result::CoxSamplerResult; clinical_X = nothing, compute_metrics::Bool = true)
    model_draws = posterior_beta_draws(result)
    draws = beta_draws_unstandardized(result.data, model_draws)
    compute_metrics || return (result = result, draws = draws, summary = nothing, metrics = nothing, p0 = prior_p0(result.prior))
    layout = posterior_draw_layout(result, size(model_draws, 1))
    eval_data = clinical_X === nothing ? result.data : joint_cox_data(result.data, clinical_X)
    eval_draws = clinical_X === nothing ? model_draws : hcat(model_draws, posterior_clinical_draws(result))
    beta_map = beta_vector_unstandardized(result.data, posterior_beta_map(result))
    summary = posterior_summary(draws; map = beta_map)
    estimates = (mean = vec(mean(eval_draws; dims = 1)),
        median = [median(view(eval_draws, :, j)) for j in axes(eval_draws, 2)],
        MAP = vec(eval_draws[argmax(posterior_logposterior_draws(result)), :]))
    metrics = posterior_fit_metrics(eval_data, eval_draws, estimates; chains = layout.chains, draws_per_chain = layout.draws)
    for (key, value) in sampling_diagnostics_dict(result.diagnostics)
        metrics[!, Symbol(key)] = fill(value, nrow(metrics))
    end
    return (result = result, draws = draws, summary = summary, metrics = metrics, p0 = prior_p0(result.prior))
end

function add_elpd_diffs!(metrics::DataFrame; group_cols = Symbol[])
    :ELPD_LOO in propertynames(metrics) || return metrics
    :ELPD_Diff in propertynames(metrics) || (metrics.ELPD_Diff = fill(NaN, nrow(metrics)))
    valid_group_cols = [c for c in group_cols if c in propertynames(metrics)]
    groups = isempty(valid_group_cols) ? [metrics] : groupby(metrics, valid_group_cols)
    for group in groups
        vals = collect(skipmissing(group.ELPD_LOO))
        vals = vals[isfinite.(vals)]
        isempty(vals) && continue
        best = maximum(vals)
        rows = group isa SubDataFrame ? parentindices(group)[1] : axes(group, 1)
        metrics[rows, :ELPD_Diff] .= group.ELPD_LOO .- best
    end
    return metrics
end


using SHA
using Serialization

function check_loo_mode(mode::Symbol)
    mode in (:fixed_risk, :subject_refit) ||
        throw(ArgumentError("loo mode must be :fixed_risk or :subject_refit"))
    return mode
end

"""
    subject_deletion_logratios(train_data, augmented_data, draws)

Evaluate log L_PL(D; beta) - log L_PL(D_-i; beta) at TRAINING posterior
draws. Both data objects must use the same training-fitted transformation.
This is a cross-validated partial-likelihood ratio score, not a normalized
survival-time predictive density. Censored subjects affect this score too.
Using the total Efron likelihood avoids an arbitrary allocation of tied events.
"""
function subject_deletion_logratios(train_data::CoxPreparedData,
                                  augmented_data::CoxPreparedData,
                                  draws::AbstractMatrix)
    length(augmented_data.time) == length(train_data.time) + 1 ||
        throw(DimensionMismatch("augmented data must add exactly one subject"))
    train_data.tie_method == augmented_data.tie_method ||
        throw(ArgumentError("training and augmented tie methods must match"))
    return [cox_logpartiallikelihood(augmented_data, view(draws, s, :)) -
            cox_logpartiallikelihood(train_data, view(draws, s, :))
            for s in axes(draws, 1)]
end

function _loo_augment(train_data, X_train, X_test, time_test, event_test)
    z_test = _validation_design_matrix(X_train, X_test, train_data)
    time = vcat(train_data.time, Float64(time_test))
    data = CoxPreparedData(
        vcat(train_data.z, z_test), time, vcat(train_data.event, Int(event_test)),
        BayesCox4Micro._build_riskset(time), train_data.tie_method,
        train_data.transform, train_data.zero_replacement, train_data.closure,
        train_data.standardize, train_data.col_mean, train_data.col_scale,
    )
    return data
end

function _loo_atomic_json(path, value)
    mkpath(dirname(path))
    tmp = tempname(dirname(path))
    try
        open(tmp, "w") do io
            JSON.print(io, value, 2)
            write(io, '\n')
        end
        mv(tmp, path; force = true)
    finally
        isfile(tmp) && rm(tmp)
    end
    return path
end

function _loo_code_digest()
    root = normpath(joinpath(@__DIR__, "..", ".."))
    files = [joinpath(root, "main.jl"), joinpath(root, "Project.toml")]
    isfile(joinpath(root, "Manifest.toml")) && push!(files, joinpath(root, "Manifest.toml"))
    for subdir in ("mainfuncs", "priors", "scripts")
        for (dir, _, names) in walkdir(joinpath(root, subdir)), name in names
            endswith(name, ".jl") && push!(files, joinpath(dir, name))
        end
    end
    io = IOBuffer()
    for file in sort(files)
        write(io, relpath(file, root), '\0', read(file), '\0')
    end
    return bytes2hex(sha256(take!(io)))
end

function _loo_checkpoint_manifest(X, time, event, fit_kwargs, checkpoint_dir, resume; mode = "subject_refit")
    io = IOBuffer()
    settings = sort!(collect(pairs(fit_kwargs)); by = x -> string(first(x)))
    serialize(io, (Matrix{Float64}(X), Float64.(time), Int.(event), settings,
                   string(VERSION), _loo_code_digest()))
    fingerprint = bytes2hex(sha256(take!(io)))
    checkpoint_dir === nothing && return fingerprint
    path = joinpath(checkpoint_dir, "manifest.json")
    if isfile(path)
        resume || throw(ArgumentError("checkpoint exists; resume it or choose a new directory"))
        get(JSON.parsefile(path; use_mmap = false), "fingerprint", "") == fingerprint ||
            throw(ArgumentError("$mode checkpoint data, settings, Julia version or code changed; use a new directory"))
    else
        if isdir(checkpoint_dir) && !isempty(readdir(checkpoint_dir))
            throw(ArgumentError("nonempty checkpoint directory has no manifest"))
        end
        _loo_atomic_json(path, Dict("fingerprint" => fingerprint, "n" => length(time),
                                   "mode" => mode, "schema_version" => 1))
    end
    return fingerprint
end

function _loo_draw_diagnostics(result, draws)
    layout = posterior_draw_layout(result, size(draws, 1))
    layout.draws >= 4 || return (nothing, nothing)
    stats = convergence_diagnostics(result_to_chains(result))
    stats = stats[occursin.(r"^beta\[\d+\]$", stats.parameters), :]
    valid(values) = !isempty(values) && all(x -> x isa Real && isfinite(x), values)
    return (valid(stats.rhat) ? maximum(stats.rhat) : nothing,
            valid(stats.ess_bulk) ? minimum(stats.ess_bulk) : nothing)
end

function _loo_refit_fold(X, time, event, i, fit_kwargs, fit_function, diagnostics)
    train = [j for j in eachindex(time) if j != i]
    kwargs = Dict{Symbol,Any}(pairs(fit_kwargs))
    kwargs[:seed] = Int(get(kwargs, :seed, 1)) + i
    kwargs[:compute_metrics] = false
    started = time_ns()
    fit = fit_function(X[train, :], time[train], event[train]; kwargs...)
    data = fit.result.data
    data.time == Float64.(time[train]) && data.event == Int.(event[train]) ||
        throw(ArgumentError("refit must use exactly the training subjects, in input order"))
    draws = posterior_beta_draws(fit.result)
    size(draws, 1) > 0 && all(isfinite, draws) || error("nonfinite or empty posterior draws")
    augmented = _loo_augment(data, X[train, :], X[i:i, :], time[i], event[i])
    logratios = subject_deletion_logratios(data, augmented, draws)
    all(isfinite, logratios) || error("nonfinite subject-deletion log partial-likelihood ratios")
    max_rhat, min_ess = diagnostics ? _loo_draw_diagnostics(fit.result, draws) : (nothing, nothing)
    return Dict{String,Any}(
        "subject" => i, "event" => Int(event[i]), "n_train" => length(train),
        "events_train" => sum(event[train]), "status" => "success",
        "log_score" => logmeanexp(logratios),
        "mean_log_ratio" => mean(logratios), "posterior_draws" => size(draws, 1),
        "seed" => kwargs[:seed], "p0" => (ismissing(fit.p0) ? nothing : fit.p0),
        "beta_rhat_max" => max_rhat, "beta_ess_bulk_min" => min_ess,
        "elapsed_seconds" => (time_ns() - started) / 1e9,
    )
end

"""
    evaluate_cox_loo(X, time, event; mode=:fixed_risk, fit_kwargs=(;), ...)

`:fixed_risk`: existing event-factor PSIS-LOO, harmonic-mean NLPML, and WAIC.
`:subject_refit`: independently fit every D_-i (including censored subjects),
rebuild training risk sets, and score log mean_s exp(log L(D;beta_s)-log L(D_-i;beta_s)).
The latter reports LOO_LOG_SCORE, LOOIC=-2*score and NLPML=-score on a
partial-likelihood-ratio scale. ELPD_LOO, WAIC and Pareto-k are inapplicable
to this target and are missing, not relabeled fixed-risk estimates.

`folds` may select a pilot subset. Aggregate criteria remain missing unless all
n subjects succeed. `checkpoint_dir` stores one atomic JSON file per fold;
resume checks a fingerprint of data, fit settings, source and Julia version.
Supplied hyperparameters are held fixed; training-dependent prior scales and
preprocessing are recomputed inside each fit. No nested hyperparameter tuning.
`fit_function` is an optional fitting adapter with the fit_and_evaluate contract.
"""
function evaluate_cox_loo(X::AbstractMatrix, time::AbstractVector, event::AbstractVector;
    mode::Symbol = :fixed_risk, fit_kwargs = (;), full_fit = nothing,
    folds = eachindex(time), checkpoint_dir = nothing, resume::Bool = true,
    parallel::Bool = false, diagnostics::Bool = true,
    fit_function = fit_and_evaluate,
)
    check_loo_mode(mode)
    n = length(time)
    n == length(event) == size(X, 1) || throw(DimensionMismatch("X, time, event sizes differ"))
    n >= 2 || throw(ArgumentError("LOO requires at least two subjects"))
    all(x -> x in (0, 1), event) || throw(ArgumentError("event must be binary"))
    all(isfinite, time) && all(>=(0), time) || throw(ArgumentError("invalid survival times"))
    all(isfinite, X) && all(>=(0), X) || throw(ArgumentError("invalid covariates"))
    kwargs = Dict{Symbol,Any}(pairs(fit_kwargs))
    if mode == :fixed_risk
        kwargs[:compute_metrics] = true
        fit = full_fit === nothing ? fit_function(X, time, event; kwargs...) : full_fit
        fit.metrics === nothing && throw(ArgumentError("full_fit needs computed metrics"))
        m = first(eachrow(fit.metrics))
        summary = DataFrame(mode = ["fixed_risk"], target = ["event_attribution_fixed_risk"],
            n_subjects = [n], n_units = [sum(event)], n_completed = [sum(event)], complete = [true],
            LOO_LOG_SCORE = [m.ELPD_LOO], ELPD_LOO = [m.ELPD_LOO], LOOIC = [m.LOOIC],
            NLPML = [m.NLPML], WAIC_pw1 = [m.WAIC_pw1], WAIC_pw2 = [m.WAIC_pw2],
            ParetoK_max = [m.ParetoK_max], ParetoK_bad = [m.ParetoK_bad],
            estimator = ["psis; NLPML uses unsmoothed harmonic mean"])
        return (summary = summary, pointwise = DataFrame(), full_fit = fit)
    end
    sum(event) >= 2 || throw(ArgumentError("subject_refit requires an event in every training fold"))
    selected = collect(Int, folds)
    !isempty(selected) && all(i -> 1 <= i <= n, selected) && allunique(selected) ||
        throw(ArgumentError("folds must be nonempty, unique subject indices in 1:n"))
    if parallel
        Threads.nthreads() > 1 || throw(ArgumentError("parallel LOO requires julia -t N with N>1"))
        kwargs[:multichains] = false # parallelize folds, not nested chains
    end
    # Diagnostics affect saved records, so include the setting in checkpoint identity.
    fingerprint = _loo_checkpoint_manifest(X, time, event,
        merge(kwargs, Dict(:loo_diagnostics => diagnostics)), checkpoint_dir, resume)
    records = Vector{Dict{String,Any}}(undef, length(selected))
    function run_fold(k)
        i = selected[k]
        path = checkpoint_dir === nothing ? nothing : joinpath(checkpoint_dir, "fold_$(lpad(i, 5, '0')).json")
        if resume && path !== nothing && isfile(path)
            saved = JSON.parsefile(path; use_mmap = false)
            get(saved, "fingerprint", "") == fingerprint && get(saved, "subject", 0) == i ||
                error("invalid checkpoint for subject $i")
            if get(saved, "status", "") == "success"
                isfinite(saved["log_score"]) || error("invalid saved score for subject $i")
                records[k] = saved
                return
            end
        end
        record = try
            _loo_refit_fold(X, time, event, i, kwargs, fit_function, diagnostics)
        catch err
            err isa InterruptException && rethrow()
            @error "LOO refit failed" subject = i exception = (err, catch_backtrace())
            Dict{String,Any}("subject" => i, "event" => Int(event[i]), "status" => "failed",
                             "log_score" => nothing, "error" => sprint(showerror, err))
        end
        record["fingerprint"] = fingerprint
        path === nothing || _loo_atomic_json(path, record)
        records[k] = record
        @info "LOO fold finished" subject = i n status = record["status"]
    end
    if parallel
        Threads.@threads for k in eachindex(selected)
            run_fold(k)
        end
    else
        foreach(run_fold, eachindex(selected))
    end
    good = filter(r -> r["status"] == "success", records)
    complete = length(good) == n
    total = complete ? sum(r["log_score"] for r in good) : missing
    summary = DataFrame(mode = ["subject_refit"], target = ["subject_deletion_partial_likelihood_ratio"],
        n_subjects = [n], n_units = [n], n_completed = [length(good)], complete = [complete],
        LOO_LOG_SCORE = [total], ELPD_LOO = [missing], LOOIC = [-2 * total], NLPML = [-total],
        WAIC_pw1 = [missing], WAIC_pw2 = [missing], ParetoK_max = [missing], ParetoK_bad = [missing],
        estimator = ["independent MCMC refits; log posterior mean partial-likelihood ratio"])
    # Keep missing fields (e.g. failed fits and unavailable R-hat) as CSV missing values.
    keys_all = sort!(collect(union((Set(keys(r)) for r in records)...)))
    pointwise = DataFrame([Symbol(key) => [something(get(r, key, nothing), missing) for r in records]
                           for key in keys_all])
    sort!(pointwise, :subject)
    return (summary = summary, pointwise = pointwise, full_fit = full_fit)
end

# Shared result-table writing for CV and tuning outputs.
function _metrics_write_table(path::AbstractString, table)
    mkpath(dirname(path))
    return _write_csv(path, table)
end
_metrics_file_complete(path::AbstractString) = isfile(path) && filesize(path) > 0
