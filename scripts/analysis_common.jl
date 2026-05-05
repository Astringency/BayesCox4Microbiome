using BayesCox4Micro
using CSV
using DataFrames
using Distributions
using JSON
using LinearAlgebra

if !haskey(ENV, "GKSwstype") && !haskey(ENV, "DISPLAY")
    ENV["GKSwstype"] = "100"
end

import Plots
using PosteriorStats
using Random
using Statistics

const DEFAULT_BETA6 = [2.08, 1.31, -1.39, -1.86, 2.12, -2.26]
const DEFAULT_BETA12 = [2.08, -1.41, -1.39, -1.15, 2.12, 2.51, 1.31, -1.95, -1.86, 1.93, -1.34, -0.85]

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

arg(args, key, default::String) = get(args, key, default)
arg(args, key, default::Int) = parse(Int, get(args, key, string(default)))
arg(args, key, default::Float64) = parse(Float64, get(args, key, string(default)))
arg(args, key, default::Bool) = lowercase(get(args, key, string(default))) in ("true", "1", "yes", "y")

function split_symbols(s::AbstractString)
    return Symbol.(strip.(split(s, ",")))
end

function split_ints(s::AbstractString)
    return parse.(Int, strip.(split(s, ",")))
end

function split_strings(s::AbstractString)
    return strip.(split(s, ","))
end

function parse_optional_p0(args; default = nothing)
    haskey(args, "p0") || return default
    value = lowercase(strip(args["p0"]))
    value in ("auto", "none", "missing") && return nothing
    value == "truth" && return :truth
    return parse(Float64, value)
end

ensure_dir(path::AbstractString) = (mkpath(path); path)

function true_beta(p::Integer, nonzero::Integer)
    values =
        nonzero == 6 ? DEFAULT_BETA6 :
        nonzero == 12 ? DEFAULT_BETA12 :
        throw(ArgumentError("nonzero must be 6 or 12."))
    idx = collect(18:2:(16 + 2nonzero))
    maximum(idx) <= p || throw(ArgumentError("p=$p is too small for the paper's nonzero indices."))
    beta = zeros(Float64, p)
    beta[idx] .= values
    return beta, idx
end

function paper_covariance(p::Integer)
    sigma = [0.75 - 0.015 * abs(i - j) for i in 1:p, j in 1:p]
    sigma = Symmetric(sigma)
    vals, vecs = eigen(sigma)
    floor = 1e-6
    return Symmetric(vecs * Diagonal(max.(vals, floor)) * vecs')
end

function softmax_rows(U::AbstractMatrix)
    X = similar(U, Float64)
    for i in axes(U, 1)
        row = view(U, i, :)
        m = maximum(row)
        denom = sum(exp.(row .- m))
        @views X[i, :] .= exp.(row .- m) ./ denom
    end
    return X
end

function baseline_event_time(rng::AbstractRNG, eta::Real, baseline::Symbol; tmax::Real = 100.0)
    target = -log(rand(rng)) / exp(eta)
    if baseline == :exponential
        rate = 0.01
        return target / rate
    elseif baseline == :weibull
        shape = 2.0
        scale = 50.0
        return scale * target^(1 / shape)
    elseif baseline == :gompertz
        rate = 0.01
        shape = 0.02
        return log1p(shape * target / rate) / shape
    else
        throw(ArgumentError("baseline must be :weibull, :exponential, or :gompertz."))
    end
end

function generate_simulation_data(
    n::Integer,
    p::Integer,
    nonzero::Integer;
    baseline::Symbol = :weibull,
    seed::Integer = 1,
    tmax::Real = 100.0,
)
    rng = MersenneTwister(seed)
    beta, idx = true_beta(p, nonzero)
    mu = zeros(Float64, p)
    mu[idx] .= log(0.5 * p)
    U = rand(rng, MvNormal(mu, paper_covariance(p)), n)'
    X = softmax_rows(U)
    eta = log.(X) * beta
    event_time = [baseline_event_time(rng, eta[i], baseline; tmax = tmax) for i in 1:n]
    time = min.(event_time, tmax)
    event = Int.(event_time .<= tmax)
    if count(!=(0), event) == 0
        event[argmin(time)] = 1
    end
    return (X = X, time = Float64.(time), event = event, beta = beta, nonzero_index = idx)
end

function prior_from_name(
    name::AbstractString,
    X::AbstractMatrix,
    time::AbstractVector,
    event::AbstractVector;
    p0 = nothing,
    p0_alpha::Real = 0.05,
    realdata::Bool = false,
)
    return cox_paper_prior(
        name;
        p0 = p0,
        X = X,
        time = time,
        event = event,
        p0_alpha = p0_alpha,
        realdata = realdata,
    )
end

prior_p0(prior) = hasproperty(prior, :p0) ? getproperty(prior, :p0) : missing

function univariate_screen_table(X, time, event; names = nothing, alpha::Real = 0.05)
    screen = estimate_p0_univariate_cox(X, time, event; alpha = alpha)
    p = length(screen.pvalue)
    vars = names === nothing ? ["x$j" for j in 1:p] : string.(names)
    return DataFrame(
        variable = vars,
        estimate = screen.estimate,
        stderr = screen.stderr,
        zscore = screen.zscore,
        pvalue = screen.pvalue,
        significant = screen.significant,
        converged = screen.converged,
    ), screen
end

function posterior_summary(
    draws::AbstractMatrix;
    names = ["beta[$j]" for j in 1:size(draws, 2)],
    alpha::Real = 0.05,
    map = nothing,
)
    p = size(draws, 2)
    lower_p = alpha / 2
    upper_p = 1 - alpha / 2
    beta_map = map === nothing ? fill(NaN, p) : collect(Float64, map)
    length(beta_map) == p || throw(DimensionMismatch("MAP estimate length must match beta dimension."))
    return DataFrame(
        variable = names,
        mean = vec(mean(draws; dims = 1)),
        median = [median(view(draws, :, j)) for j in 1:p],
        map = beta_map,
        lower = [quantile(view(draws, :, j), lower_p) for j in 1:p],
        upper = [quantile(view(draws, :, j), upper_p) for j in 1:p],
    )
end

function false_discovery(beta::AbstractVector, summary::DataFrame)
    selected = (summary.lower .* summary.upper) .> 0
    truth = beta .!= 0
    fp = sum((.!truth) .& selected)
    tp = sum(truth .& selected)
    fn = sum(truth .& (.!selected))
    tn = sum((.!truth) .& (.!selected))
    fdr = fp + tp == 0 ? 0.0 : fp / (fp + tp)
    return (FP = fp, FN = fn, TP = tp, TN = tn, FDR = fdr)
end

function logmeanexp(v::AbstractVector)
    m = maximum(v)
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
            PSIS_LOO = NaN,
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
            PSIS_LOO = looic,
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
            PSIS_LOO = NaN,
        )
    end
end

function fit_metrics(
    data::CoxPreparedData,
    draws::AbstractMatrix,
    beta_hat::AbstractVector;
    chains::Integer = 1,
    draws_per_chain::Integer = size(draws, 1),
    ll_event = nothing,
    loo = nothing,
)
    point_loglik = cox_logpartiallikelihood(data, beta_hat)
    point_risk = data.z * beta_hat
    event_loglik =
        if ll_event === nothing
            ll = posterior_event_loglik(data, draws)
            event_rows = findall(!=(0), data.event)
            ll[event_rows, :]
        else
            ll_event
        end
    loo_metrics = loo === nothing ? psis_loo_metrics(event_loglik, chains, draws_per_chain) : loo
    alog_plik = sum(logmeanexp(view(event_loglik, i, :)) for i in axes(event_loglik, 1))
    mean_ll = vec(mean(event_loglik; dims = 2))
    pw1 = 2 * sum([logmeanexp(view(event_loglik, i, :)) for i in axes(event_loglik, 1)] .- mean_ll)
    pw2 = size(event_loglik, 2) > 1 ? sum(var(view(event_loglik, i, :)) for i in axes(event_loglik, 1)) : NaN
    lpml = sum(log(mean(exp.(-view(event_loglik, i, :)))) for i in axes(event_loglik, 1))
    sample_total = vec(sum(event_loglik; dims = 1))
    pD = mean(-2 .* sample_total .+ 2) - (-2 * point_loglik + 2)
    return (
        loglik = point_loglik,
        LPML = lpml,
        AIC = -2 * point_loglik + 2 * length(beta_hat),
        BIC = -2 * point_loglik + log(length(data.time)) * length(beta_hat),
        WAIC_pw1 = -2 * alog_plik + 2 * pw1,
        WAIC_pw2 = -2 * alog_plik + 2 * pw2,
        DIC = point_loglik - pD,
        ELPD_LOO = loo_metrics.ELPD_LOO,
        ELPD_LOO_SE = loo_metrics.ELPD_LOO_SE,
        PLOO = loo_metrics.PLOO,
        PLOO_SE = loo_metrics.PLOO_SE,
        LOOIC = loo_metrics.LOOIC,
        LOOIC_SE = loo_metrics.LOOIC_SE,
        ParetoK_max = loo_metrics.ParetoK_max,
        ParetoK_bad = loo_metrics.ParetoK_bad,
        PSIS_LOO = loo_metrics.PSIS_LOO,
        Cindex = concordance_index(data.time, data.event, point_risk),
    )
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

function fit_and_evaluate(
    X::AbstractMatrix,
    time::AbstractVector,
    event::AbstractVector;
    prior_name::AbstractString = "cox-rhs",
    sampler::Symbol = :nuts,
    nuts_backend::Symbol = :turing,
    n_samples::Integer = 1000,
    warmup::Integer = 500,
    chains::Integer = 4,
    multichains::Bool = false,
    target_accept::Real = 0.8,
    seed::Integer = 1,
    use_car::Bool = false,
    car_distance = :bray,
    p0 = nothing,
    p0_alpha::Real = 0.05,
    constraint::Symbol = :soft_center,
    soft_center_scale = nothing,
    prox_sum0_lambda::Real = 1e-3,
    prox_l1_lambda::Real = 1e-3,
    prox_root_tol::Real = 1e-10,
    prox_root_maxiter::Integer = 100,
    mala_step_size::Real = 1e-4,
    mala_target_accept::Real = 0.574,
    mala_adapt_every::Integer = 50,
    mala_min_step::Real = 1e-8,
    mala_max_step::Real = 1.0,
    realdata::Bool = false,
    progress::Bool = true,
)
    prior = prior_from_name(prior_name, X, time, event; p0 = p0, p0_alpha = p0_alpha, realdata = realdata)
    result =
        if sampler in (:nuts, :nuts_turing, :nuts_explicit, :nuts_explict)
            sample_bayes_cox(
                X,
                time,
                event;
                prior = prior,
                sampler = sampler,
                nuts_backend = nuts_backend,
                n_samples = n_samples,
                warmup = warmup,
                chains = chains,
                multichains = multichains,
                target_accept = target_accept,
                seed = seed,
                use_car = use_car,
                car_distance = car_distance,
                constraint = constraint,
                soft_center_scale = soft_center_scale,
                prox_sum0_lambda = prox_sum0_lambda,
                prox_l1_lambda = prox_l1_lambda,
                prox_root_tol = prox_root_tol,
                prox_root_maxiter = prox_root_maxiter,
                progress = progress,
            )
        elseif sampler == :mh || sampler == :mala || sampler == :ula
            if sampler == :mala || sampler == :ula
                sample_bayes_cox(
                    X,
                    time,
                    event;
                    prior = prior,
                    sampler = sampler,
                    n_samples = n_samples,
                    warmup = warmup,
                    chains = chains,
                    multichains = multichains,
                    target_accept = target_accept,
                    seed = seed,
                    use_car = use_car,
                    car_distance = car_distance,
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
            else
                sample_bayes_cox(
                    X,
                    time,
                    event;
                    prior = prior,
                    sampler = sampler,
                    n_samples = n_samples,
                    warmup = warmup,
                    chains = chains,
                    multichains = multichains,
                    target_accept = target_accept,
                    seed = seed,
                    use_car = use_car,
                    car_distance = car_distance,
                    constraint = constraint,
                    soft_center_scale = soft_center_scale,
                    prox_sum0_lambda = prox_sum0_lambda,
                    progress = progress,
                )
            end
        elseif sampler == :gibbs
            sample_bayes_cox(
                X,
                time,
                event;
                prior = prior,
                sampler = sampler,
                n_samples = n_samples,
                warmup = warmup,
                chains = chains,
                multichains = multichains,
                target_accept = target_accept,
                seed = seed,
                use_car = use_car,
                car_distance = car_distance,
                constraint = constraint,
                soft_center_scale = soft_center_scale,
                prox_sum0_lambda = prox_sum0_lambda,
                progress = progress,
            )
        else
            throw(ArgumentError("sampler must be :nuts, :nuts_turing, :nuts_explicit, :nuts_explict, :mh, :mala, :ula, or :gibbs."))
    end
    draws = posterior_beta_draws(result)
    layout = posterior_draw_layout(result, size(draws, 1))
    loglik = posterior_event_loglik(result.data, draws)
    event_rows = findall(!=(0), result.data.event)
    ll_event = loglik[event_rows, :]
    loo = psis_loo_metrics(ll_event, layout.chains, layout.draws)
    names = ["beta[$j]" for j in 1:size(draws, 2)]
    beta_map = posterior_beta_map(result)
    summary = posterior_summary(draws; names = names, map = beta_map)
    beta_mean = summary.mean
    beta_median = summary.median
    metrics_mean = fit_metrics(result.data, draws, beta_mean; chains = layout.chains, draws_per_chain = layout.draws, ll_event = ll_event, loo = loo)
    metrics_median = fit_metrics(result.data, draws, beta_median; chains = layout.chains, draws_per_chain = layout.draws, ll_event = ll_event, loo = loo)
    metrics_map = fit_metrics(result.data, draws, beta_map; chains = layout.chains, draws_per_chain = layout.draws, ll_event = ll_event, loo = loo)
    diagnostics = sampler_diagnostics(result)
    metrics = DataFrame(
        estimate = ["mean", "median", "MAP"],
        loglik = [metrics_mean.loglik, metrics_median.loglik, metrics_map.loglik],
        LPML = [metrics_mean.LPML, metrics_median.LPML, metrics_map.LPML],
        AIC = [metrics_mean.AIC, metrics_median.AIC, metrics_map.AIC],
        BIC = [metrics_mean.BIC, metrics_median.BIC, metrics_map.BIC],
        WAIC_pw1 = [metrics_mean.WAIC_pw1, metrics_median.WAIC_pw1, metrics_map.WAIC_pw1],
        WAIC_pw2 = [metrics_mean.WAIC_pw2, metrics_median.WAIC_pw2, metrics_map.WAIC_pw2],
        DIC = [metrics_mean.DIC, metrics_median.DIC, metrics_map.DIC],
        ELPD_LOO = [metrics_mean.ELPD_LOO, metrics_median.ELPD_LOO, metrics_map.ELPD_LOO],
        ELPD_LOO_SE = [metrics_mean.ELPD_LOO_SE, metrics_median.ELPD_LOO_SE, metrics_map.ELPD_LOO_SE],
        PLOO = [metrics_mean.PLOO, metrics_median.PLOO, metrics_map.PLOO],
        PLOO_SE = [metrics_mean.PLOO_SE, metrics_median.PLOO_SE, metrics_map.PLOO_SE],
        LOOIC = [metrics_mean.LOOIC, metrics_median.LOOIC, metrics_map.LOOIC],
        LOOIC_SE = [metrics_mean.LOOIC_SE, metrics_median.LOOIC_SE, metrics_map.LOOIC_SE],
        ParetoK_max = [metrics_mean.ParetoK_max, metrics_median.ParetoK_max, metrics_map.ParetoK_max],
        ParetoK_bad = [metrics_mean.ParetoK_bad, metrics_median.ParetoK_bad, metrics_map.ParetoK_bad],
        PSIS_LOO = [metrics_mean.PSIS_LOO, metrics_median.PSIS_LOO, metrics_map.PSIS_LOO],
        Cindex = [metrics_mean.Cindex, metrics_median.Cindex, metrics_map.Cindex],
        warmup_seconds = fill(diagnostics.warmup_seconds, 3),
        sample_seconds = fill(diagnostics.sample_seconds, 3),
        total_seconds = fill(diagnostics.total_seconds, 3),
        allocated_mb = fill(diagnostics.allocated_bytes / 2.0^20, 3),
        gc_seconds = fill(diagnostics.gc_seconds, 3),
        gc_percent = fill(diagnostics.gc_percent, 3),
    )
    return (result = result, draws = draws, summary = summary, metrics = metrics, p0 = prior_p0(prior))
end

function _plot_path_with_ext(path::AbstractString, ext::AbstractString)
    root, _ = splitext(path)
    return root * ext
end

function _save_plots_vector(plot_obj, path::AbstractString; save_eps::Bool = true)
    pdf_path = _plot_path_with_ext(path, ".pdf")
    ensure_dir(dirname(pdf_path))
    Plots.savefig(plot_obj, pdf_path)
    if save_eps
        eps_path = _plot_path_with_ext(path, ".eps")
        open(eps_path, "w") do io
            Base.invokelatest(show, io, MIME("application/postscript"), plot_obj)
        end
    end
    return pdf_path
end

function coefficient_plot(
    summary::DataFrame,
    path::AbstractString;
    truth = nothing,
    title = "Coefficient estimates",
    estimate::Symbol = :mean,
)
    x = 1:nrow(summary)
    estimate in propertynames(summary) ||
        throw(ArgumentError("summary does not contain estimate column $estimate."))
    (:lower in propertynames(summary) && :upper in propertynames(summary)) ||
        throw(ArgumentError("summary must contain lower and upper columns."))
    y = Float64.(summary[!, estimate])
    lower = Float64.(summary.lower)
    upper = Float64.(summary.upper)
    estimate_label = estimate == :map ? "MAP" : "posterior $(String(estimate))"
    y_values = vcat(lower, upper, y, [0.0], truth === nothing ? Float64[] : Float64.(truth))
    ymin, ymax = extrema(y_values)
    pad = max(0.08 * (ymax - ymin), 0.1)
    ymin -= pad
    ymax += pad
    try
        Plots.gr()
    catch
    end
    p = Plots.scatter(
        x,
        y;
        yerror = (y .- lower, upper .- y),
        label = estimate_label,
        xlabel = "Taxon index",
        ylabel = "Coefficient",
        title = title,
        ylims = (ymin, ymax),
        color = :steelblue,
        markerstrokecolor = :steelblue,
        markersize = 3,
        linewidth = 1.2,
        legend = :topright,
        grid = true,
        size = (900, 420),
    )
    Plots.hline!(p, [0.0]; label = "", color = :gray, linestyle = :dash, linewidth = 1)
    if truth !== nothing
        truthv = Float64.(truth)
        length(truthv) == nrow(summary) ||
            throw(DimensionMismatch("truth length must match summary row count."))
        Plots.scatter!(
            p,
            x,
            truthv;
            label = "true",
            color = :firebrick,
            markerstrokecolor = :firebrick,
            markersize = 3,
        )
    end
    return _save_plots_vector(p, path)
end

function _finite_mean(v)
    vals = Float64[]
    for x in skipmissing(v)
        xf = Float64(x)
        isfinite(xf) && push!(vals, xf)
    end
    return isempty(vals) ? NaN : mean(vals)
end

function _metric_values(rows::DataFrame, metric::Symbol)
    agg = combine(groupby(rows, [:prior, :estimate]), metric => _finite_mean => :value)
    priors = unique(agg.prior)
    estimates = unique(agg.estimate)
    return agg, priors, estimates
end

function _metric_panel_plot(rows::DataFrame, metric::Symbol; title::AbstractString)
    agg, priors, estimates = _metric_values(rows, metric)
    p = Plots.plot(; title = title, xlabel = "", ylabel = "", legend = false, grid = true)
    isempty(priors) && return p
    vals = [Float64(v) for v in agg.value if isfinite(Float64(v))]
    isempty(vals) && return p
    ymin = metric == :Cindex ? 0.0 : min(0.0, minimum(vals))
    ymax = metric == :Cindex ? 1.0 : max(0.0, maximum(vals))
    ymax == ymin && (ymax += 1.0)
    xbase = collect(1:length(priors))
    n_est = max(length(estimates), 1)
    bar_w = min(0.7 / n_est, 0.22)
    offsets = collect(range(-bar_w * (n_est - 1) / 2, bar_w * (n_est - 1) / 2; length = n_est))
    palette = [:steelblue, :darkorange, :seagreen, :purple]
    for (k, est) in enumerate(estimates)
        xs = Float64[]
        ys = Float64[]
        for (i, prior) in enumerate(priors)
            idx = findfirst((agg.prior .== prior) .& (agg.estimate .== est))
            idx === nothing && continue
            value = agg.value[idx]
            isfinite(value) || continue
            push!(xs, xbase[i] + offsets[k])
            push!(ys, Float64(value))
        end
        Plots.bar!(
            p,
            xs,
            ys;
            bar_width = bar_w,
            label = string(est),
            color = palette[mod1(k, length(palette))],
        )
    end
    labels = [length(string(prior)) > 10 ? string(prior)[1:10] : string(prior) for prior in priors]
    Plots.plot!(
        p;
        xticks = (xbase, labels),
        xrotation = 35,
        ylims = (ymin, ymax),
        legend = :topright,
    )
    return p
end

function metrics_plot(metrics::DataFrame, path::AbstractString; title = "Model metrics")
    plot_rows = copy(metrics)
    plot_rows.prior = string.(plot_rows.prior)
    plot_rows.estimate = string.(plot_rows.estimate)
    for col in (:prior, :estimate, :WAIC_pw1, :LOOIC, :LPML, :Cindex)
        col in propertynames(plot_rows) ||
            throw(ArgumentError("metrics table does not contain required column $col."))
    end
    try
        Plots.gr()
    catch
    end
    panels = (
        _metric_panel_plot(plot_rows, :WAIC_pw1; title = "WAIC"),
        _metric_panel_plot(plot_rows, :LOOIC; title = "LOOIC"),
        _metric_panel_plot(plot_rows, :LPML; title = "LPML"),
        _metric_panel_plot(plot_rows, :Cindex; title = "C-index"),
    )
    p = Plots.plot(
        panels...;
        layout = (1, 4),
        plot_title = title,
        size = (1320, 360),
    )
    return _save_plots_vector(p, path)
end

function write_table(path::AbstractString, table)
    ensure_dir(dirname(path))
    CSV.write(path, table)
    return path
end

file_complete(path::AbstractString) = isfile(path) && filesize(path) > 0

function simulation_samples_complete(run_dir::AbstractString)
    return file_complete(joinpath(run_dir, "samples_wide.csv")) ||
           file_complete(joinpath(run_dir, "result_bundle.jld2"))
end

function simulation_metric_outputs_complete(run_dir::AbstractString)
    return file_complete(joinpath(run_dir, "posterior_summary.csv")) &&
           file_complete(joinpath(run_dir, "metrics.csv")) &&
           file_complete(joinpath(run_dir, "false_discovery.csv"))
end

simulation_run_complete(run_dir::AbstractString) =
    simulation_samples_complete(run_dir) && simulation_metric_outputs_complete(run_dir)

simulation_run_recoverable(run_dir::AbstractString) =
    simulation_samples_complete(run_dir) && file_complete(joinpath(run_dir, "posterior_summary.csv"))

function mark_simulation_run_complete(run_dir::AbstractString)
    ensure_dir(run_dir)
    open(joinpath(run_dir, "complete.txt"), "w") do io
        write(io, "complete\n")
    end
    return joinpath(run_dir, "complete.txt")
end

function read_run_metadata(run_dir::AbstractString)
    path = joinpath(run_dir, "metadata.json")
    file_complete(path) || return Dict{String,Any}()
    try
        return Dict{String,Any}(JSON.parsefile(path))
    catch err
        @warn "Could not read metadata.json from $run_dir." exception=(err, catch_backtrace())
        return Dict{String,Any}()
    end
end

function _metadata_number(run_dir::AbstractString, key::AbstractString, default)
    meta = read_run_metadata(run_dir)
    haskey(meta, key) || return default
    value = meta[key]
    value isa Number && return value
    value isa AbstractString || return default
    try
        return parse(Float64, value)
    catch
        return default
    end
end

function _beta_index(name)
    m = match(r"^beta\[(\d+)\]$", string(name))
    m === nothing && return nothing
    return parse(Int, m.captures[1])
end

function ensure_simulation_summary(summary::DataFrame, beta::AbstractVector)
    out = copy(summary)
    if !(:variable in propertynames(out))
        if :parameters in propertynames(out)
            rename!(out, :parameters => :variable)
        else
            out[!, :variable] = ["beta[$j]" for j in eachindex(beta)]
        end
    end
    idx = [_beta_index(v) for v in out.variable]
    if all(!isnothing, idx)
        out[!, :_beta_index] = Int.(idx)
        out = sort(out, :_beta_index)
        select!(out, Not(:_beta_index))
    end
    if nrow(out) != length(beta)
        keep = [_beta_index(v) !== nothing for v in out.variable]
        if any(keep)
            out = out[keep, :]
            idx = [_beta_index(v) for v in out.variable]
            out[!, :_beta_index] = Int.(idx)
            out = sort(out, :_beta_index)
            select!(out, Not(:_beta_index))
        end
    end
    nrow(out) == length(beta) ||
        throw(ArgumentError("posterior summary row count $(nrow(out)) does not match beta length $(length(beta))."))
    if !(:lower in propertynames(out)) && :q025 in propertynames(out)
        rename!(out, :q025 => :lower)
    end
    if !(:upper in propertynames(out)) && :q975 in propertynames(out)
        rename!(out, :q975 => :upper)
    end
    if !(:median in propertynames(out)) && Symbol("50.0%") in propertynames(out)
        rename!(out, Symbol("50.0%") => :median)
    end
    if !(:map in propertynames(out))
        out[!, :map] = :median in propertynames(out) ? Float64.(out[!, :median]) : fill(NaN, nrow(out))
    end
    out[!, :variable] = ["beta[$j]" for j in eachindex(beta)]
    out[!, :true_beta] = collect(Float64, beta)
    out[!, :nonzero] = beta .!= 0
    return out
end

function _saved_samples_wide(run_dir::AbstractString)
    wide_path = joinpath(run_dir, "samples_wide.csv")
    if file_complete(wide_path)
        return CSV.read(wide_path, DataFrame)
    end
    if file_complete(joinpath(run_dir, "result_bundle.jld2"))
        chains = BayesCox4Micro._load_chains_from_run(run_dir)
        return chains_to_wide_dataframe(chains)
    end
    throw(ArgumentError("no saved samples found in $run_dir"))
end

function saved_beta_draws_and_layout(run_dir::AbstractString, p::Integer)
    wide = _saved_samples_wide(run_dir)
    if all(c -> c in propertynames(wide), (:chain, :iter))
        wide = sort(wide, [:chain, :iter])
    end
    beta_cols = Symbol.("beta[$j]" for j in 1:p)
    missing_cols = [c for c in beta_cols if !(c in propertynames(wide))]
    isempty(missing_cols) ||
        throw(ArgumentError("saved samples in $run_dir are missing beta columns: $(join(string.(missing_cols), ", "))"))
    draws = Matrix{Float64}(wide[:, beta_cols])
    if :chain in propertynames(wide)
        chains = sort(unique(Int.(wide.chain)))
        counts = [count(==(c), Int.(wide.chain)) for c in chains]
        length(unique(counts)) == 1 ||
            throw(ArgumentError("saved sample chains in $run_dir have unequal lengths."))
        return draws, (chains = length(chains), draws = first(counts))
    end
    return draws, (chains = 1, draws = nrow(wide))
end

function _diagnostic_number(run_dir::AbstractString, key::AbstractString, default = NaN)
    path = joinpath(run_dir, "diagnostics.json")
    file_complete(path) || return default
    try
        diag = JSON.parsefile(path)
        haskey(diag, key) || return default
        value = diag[key]
        value isa Number && return Float64(value)
        value isa AbstractString && return parse(Float64, value)
    catch
    end
    return default
end

function metrics_from_saved_samples(run_dir::AbstractString, sim, summary::DataFrame)
    draws, layout = saved_beta_draws_and_layout(run_dir, length(sim.beta))
    data = prepare_cox_data(sim.X, sim.time, sim.event; transform = :log, pseudocount = 1e-6)
    loglik = posterior_event_loglik(data, draws)
    event_rows = findall(!=(0), data.event)
    ll_event = loglik[event_rows, :]
    loo = psis_loo_metrics(ll_event, layout.chains, layout.draws)
    beta_mean = Float64.(summary[!, :mean])
    beta_median = Float64.(summary[!, :median])
    beta_map = Float64.(summary[!, :map])
    metrics_mean = fit_metrics(data, draws, beta_mean; chains = layout.chains, draws_per_chain = layout.draws, ll_event = ll_event, loo = loo)
    metrics_median = fit_metrics(data, draws, beta_median; chains = layout.chains, draws_per_chain = layout.draws, ll_event = ll_event, loo = loo)
    metrics_map = fit_metrics(data, draws, beta_map; chains = layout.chains, draws_per_chain = layout.draws, ll_event = ll_event, loo = loo)
    return DataFrame(
        estimate = ["mean", "median", "MAP"],
        loglik = [metrics_mean.loglik, metrics_median.loglik, metrics_map.loglik],
        LPML = [metrics_mean.LPML, metrics_median.LPML, metrics_map.LPML],
        AIC = [metrics_mean.AIC, metrics_median.AIC, metrics_map.AIC],
        BIC = [metrics_mean.BIC, metrics_median.BIC, metrics_map.BIC],
        WAIC_pw1 = [metrics_mean.WAIC_pw1, metrics_median.WAIC_pw1, metrics_map.WAIC_pw1],
        WAIC_pw2 = [metrics_mean.WAIC_pw2, metrics_median.WAIC_pw2, metrics_map.WAIC_pw2],
        DIC = [metrics_mean.DIC, metrics_median.DIC, metrics_map.DIC],
        ELPD_LOO = [metrics_mean.ELPD_LOO, metrics_median.ELPD_LOO, metrics_map.ELPD_LOO],
        ELPD_LOO_SE = [metrics_mean.ELPD_LOO_SE, metrics_median.ELPD_LOO_SE, metrics_map.ELPD_LOO_SE],
        PLOO = [metrics_mean.PLOO, metrics_median.PLOO, metrics_map.PLOO],
        PLOO_SE = [metrics_mean.PLOO_SE, metrics_median.PLOO_SE, metrics_map.PLOO_SE],
        LOOIC = [metrics_mean.LOOIC, metrics_median.LOOIC, metrics_map.LOOIC],
        LOOIC_SE = [metrics_mean.LOOIC_SE, metrics_median.LOOIC_SE, metrics_map.LOOIC_SE],
        ParetoK_max = [metrics_mean.ParetoK_max, metrics_median.ParetoK_max, metrics_map.ParetoK_max],
        ParetoK_bad = [metrics_mean.ParetoK_bad, metrics_median.ParetoK_bad, metrics_map.ParetoK_bad],
        PSIS_LOO = [metrics_mean.PSIS_LOO, metrics_median.PSIS_LOO, metrics_map.PSIS_LOO],
        Cindex = [metrics_mean.Cindex, metrics_median.Cindex, metrics_map.Cindex],
        warmup_seconds = fill(_diagnostic_number(run_dir, "warmup_seconds", NaN), 3),
        sample_seconds = fill(_diagnostic_number(run_dir, "sample_seconds", NaN), 3),
        total_seconds = fill(_diagnostic_number(run_dir, "total_seconds", NaN), 3),
        allocated_mb = fill(_diagnostic_number(run_dir, "allocated_mb", NaN), 3),
        gc_seconds = fill(_diagnostic_number(run_dir, "gc_seconds", NaN), 3),
        gc_percent = fill(_diagnostic_number(run_dir, "gc_percent", NaN), 3),
    )
end

function annotate_simulation_metrics(metrics::DataFrame, sim, summary::DataFrame; rep, n, p, nonzero, p0, baseline, prior)
    out = copy(metrics)
    mse = Dict(
        "mean" => mean((Float64.(summary[!, :mean]) .- sim.beta) .^ 2),
        "median" => mean((Float64.(summary[!, :median]) .- sim.beta) .^ 2),
        "MAP" => mean((Float64.(summary[!, :map]) .- sim.beta) .^ 2),
    )
    out[!, :prior] = fill(prior, nrow(out))
    out[!, :rep] = fill(rep, nrow(out))
    out[!, :n] = fill(n, nrow(out))
    out[!, :p] = fill(p, nrow(out))
    out[!, :nonzero_count] = fill(nonzero, nrow(out))
    out[!, :p0] = fill(p0, nrow(out))
    out[!, :baseline] = fill(string(baseline), nrow(out))
    out[!, :MSE] = [get(mse, string(e), NaN) for e in out.estimate]
    return out
end

function simulation_false_discovery_table(sim, summary::DataFrame; rep, n, p, nonzero, p0, baseline, prior)
    fd = false_discovery(sim.beta, summary)
    return DataFrame(
        rep = [rep],
        n = [n],
        p = [p],
        nonzero_count = [nonzero],
        p0 = [p0],
        baseline = [string(baseline)],
        prior = [prior],
        FP = [fd.FP],
        FN = [fd.FN],
        TP = [fd.TP],
        TN = [fd.TN],
        FDR = [fd.FDR],
    )
end

function load_or_recover_simulation_run(
    run_dir::AbstractString,
    sim;
    rep,
    n,
    p,
    nonzero,
    baseline,
    prior,
    p0,
)
    summary = ensure_simulation_summary(CSV.read(joinpath(run_dir, "posterior_summary.csv"), DataFrame), sim.beta)
    saved_p0 = _metadata_number(run_dir, "p0", p0)
    metrics_path = joinpath(run_dir, "metrics.csv")
    if file_complete(metrics_path) && !file_complete(joinpath(run_dir, "complete.txt"))
        # Interrupted new runs may have metrics written before an incomplete
        # sample file. Validate saved samples before reusing such a run.
        saved_beta_draws_and_layout(run_dir, length(sim.beta))
    end
    metrics =
        if file_complete(metrics_path)
            loaded = CSV.read(metrics_path, DataFrame)
            p0_value = :p0 in propertynames(loaded) && nrow(loaded) > 0 ? loaded.p0[1] : saved_p0
            annotate_simulation_metrics(loaded, sim, summary; rep = rep, n = n, p = p, nonzero = nonzero, p0 = p0_value, baseline = baseline, prior = prior)
        else
            recovered = metrics_from_saved_samples(run_dir, sim, summary)
            annotated = annotate_simulation_metrics(recovered, sim, summary; rep = rep, n = n, p = p, nonzero = nonzero, p0 = saved_p0, baseline = baseline, prior = prior)
            write_table(metrics_path, annotated)
            annotated
        end

    fd_path = joinpath(run_dir, "false_discovery.csv")
    p0_value = :p0 in propertynames(metrics) && nrow(metrics) > 0 ? metrics.p0[1] : saved_p0
    fd = simulation_false_discovery_table(sim, summary; rep = rep, n = n, p = p, nonzero = nonzero, p0 = p0_value, baseline = baseline, prior = prior)
    if !file_complete(fd_path)
        write_table(fd_path, fd)
    end
    mark_simulation_run_complete(run_dir)
    return (summary = summary, metrics = metrics, false_discovery = fd)
end
