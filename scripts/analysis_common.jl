using BayesCox4Micro
using CSV
using DataFrames
using Distributions
using LinearAlgebra
using Plots
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

function prior_from_name(name::AbstractString; p0::Real = 1.0, realdata::Bool = false)
    key = lowercase(replace(name, "_" => "-", " " => "-"))
    if key in ("hs", "horseshoe")
        return HorseshoePrior(tau_scale = 1.0, use_cox_tau0 = false, p0 = Float64(p0))
    elseif key in ("cox-hs", "coxhs")
        return HorseshoePrior(use_cox_tau0 = true, p0 = Float64(p0))
    elseif key in ("rhs", "regularized-horseshoe", "cox-rhs", "coxrhs")
        return RegularizedHorseshoePrior(use_cox_tau0 = true, p0 = Float64(p0), slab_df = 4.0, slab_s2 = 2.0)
    elseif key in ("lasso", "bayesian-lasso")
        return BayesianLassoPrior(lambda = nothing, lambda2_shape = 1.0, lambda2_rate = 0.1)
    elseif key in ("sslasso", "spike-slab", "spike-and-slab")
        return realdata ? SpikeSlabLassoPrior(spike_scale = 1.0, slab_scale = 10.0) :
                          SpikeSlabLassoPrior(spike_scale = 0.04, slab_scale = 0.5)
    else
        throw(ArgumentError("unsupported prior name: $name"))
    end
end

function posterior_summary(draws::AbstractMatrix; names = ["beta[$j]" for j in 1:size(draws, 2)], alpha::Real = 0.05)
    p = size(draws, 2)
    lower_p = alpha / 2
    upper_p = 1 - alpha / 2
    return DataFrame(
        variable = names,
        mean = vec(mean(draws; dims = 1)),
        median = [median(view(draws, :, j)) for j in 1:p],
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

function fit_metrics(data::CoxPreparedData, draws::AbstractMatrix, beta_hat::AbstractVector)
    point_loglik = cox_logpartiallikelihood(data, beta_hat)
    point_risk = data.z * beta_hat
    ll = posterior_event_loglik(data, draws)
    event_rows = findall(!=(0), data.event)
    ll_event = ll[event_rows, :]
    alog_plik = sum(logmeanexp(view(ll_event, i, :)) for i in axes(ll_event, 1))
    mean_ll = vec(mean(ll_event; dims = 2))
    pw1 = 2 * sum([logmeanexp(view(ll_event, i, :)) for i in axes(ll_event, 1)] .- mean_ll)
    pw2 = size(ll_event, 2) > 1 ? sum(var(view(ll_event, i, :)) for i in axes(ll_event, 1)) : NaN
    lpml = sum(log(mean(exp.(-view(ll_event, i, :)))) for i in axes(ll_event, 1))
    sample_total = vec(sum(ll_event; dims = 1))
    pD = mean(-2 .* sample_total .+ 2) - (-2 * point_loglik + 2)
    return (
        loglik = point_loglik,
        LPML = lpml,
        AIC = -2 * point_loglik + 2 * length(beta_hat),
        BIC = -2 * point_loglik + log(length(data.time)) * length(beta_hat),
        WAIC_pw1 = -2 * alog_plik + 2 * pw1,
        WAIC_pw2 = -2 * alog_plik + 2 * pw2,
        DIC = point_loglik - pD,
        Cindex = concordance_index(data.time, data.event, point_risk),
    )
end

function fit_and_evaluate(
    X::AbstractMatrix,
    time::AbstractVector,
    event::AbstractVector;
    prior_name::AbstractString = "cox-rhs",
    sampler::Symbol = :nuts,
    n_samples::Integer = 1000,
    warmup::Integer = 500,
    chains::Integer = 4,
    target_accept::Real = 0.8,
    seed::Integer = 1,
    use_car::Bool = false,
    car_distance = :bray,
    p0::Real = 1.0,
    realdata::Bool = false,
)
    prior = prior_from_name(prior_name; p0 = p0, realdata = realdata)
    result =
        if sampler == :nuts
            sample_bayes_cox(
                X,
                time,
                event;
                prior = prior,
                sampler = sampler,
                n_samples = n_samples,
                warmup = warmup,
                chains = chains,
                target_accept = target_accept,
                seed = seed,
                use_car = use_car,
                car_distance = car_distance,
            )
        elseif sampler == :mh
            sample_bayes_cox(
                X,
                time,
                event;
                prior = prior,
                sampler = sampler,
                n_samples = n_samples,
                warmup = warmup,
                seed = seed,
                use_car = use_car,
                car_distance = car_distance,
            )
        else
            throw(ArgumentError("sampler must be :nuts or :mh."))
        end
    draws = posterior_beta_draws(result)
    names = ["beta[$j]" for j in 1:size(draws, 2)]
    summary = posterior_summary(draws; names = names)
    beta_mean = summary.mean
    beta_median = summary.median
    metrics_mean = fit_metrics(result.data, draws, beta_mean)
    metrics_median = fit_metrics(result.data, draws, beta_median)
    metrics = DataFrame(
        estimate = ["mean", "median"],
        loglik = [metrics_mean.loglik, metrics_median.loglik],
        LPML = [metrics_mean.LPML, metrics_median.LPML],
        AIC = [metrics_mean.AIC, metrics_median.AIC],
        BIC = [metrics_mean.BIC, metrics_median.BIC],
        WAIC_pw1 = [metrics_mean.WAIC_pw1, metrics_median.WAIC_pw1],
        WAIC_pw2 = [metrics_mean.WAIC_pw2, metrics_median.WAIC_pw2],
        DIC = [metrics_mean.DIC, metrics_median.DIC],
        Cindex = [metrics_mean.Cindex, metrics_median.Cindex],
    )
    return (result = result, draws = draws, summary = summary, metrics = metrics)
end

function coefficient_plot(summary::DataFrame, path::AbstractString; truth = nothing, title = "Coefficient estimates")
    x = 1:nrow(summary)
    y = summary.mean
    yerr = (y .- summary.lower, summary.upper .- y)
    plt = scatter(x, y; yerror = yerr, label = "posterior mean", xlabel = "Taxon index",
        ylabel = "Coefficient", title = title, legend = :topright, markersize = 3)
    hline!(plt, [0.0]; label = "", color = :gray, linestyle = :dash)
    if truth !== nothing
        scatter!(plt, x, truth; label = "true", color = :red, markersize = 3)
    end
    savefig(plt, path)
    return path
end

function metrics_plot(metrics::DataFrame, path::AbstractString; title = "Model metrics")
    mean_rows = metrics[metrics.estimate .== "mean", :]
    plt = plot(layout = (1, 3), size = (1050, 330), title = title)
    bar!(plt[1], mean_rows.prior, mean_rows.WAIC_pw1; title = "WAIC", ylabel = "smaller is better", legend = false, xrotation = 35)
    bar!(plt[2], mean_rows.prior, mean_rows.LPML; title = "LPML", legend = false, xrotation = 35)
    bar!(plt[3], mean_rows.prior, mean_rows.Cindex; title = "C-index", legend = false, xrotation = 35, ylim = (0, 1))
    savefig(plt, path)
    return path
end

function write_table(path::AbstractString, table)
    ensure_dir(dirname(path))
    CSV.write(path, table)
    return path
end
