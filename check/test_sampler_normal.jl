using Pkg

function _activate_repo_project()
    project = abspath(joinpath(@__DIR__, "..", "Project.toml"))
    active = Base.active_project()
    active !== project && Pkg.activate(dirname(project))
end

_activate_repo_project()

using BayesCox4Micro
using Random
using Statistics
using LinearAlgebra
using Printf
using Plots

# ============================================================
# Target: d-dimensional standard normal N(0, I)
#
# log π(x) = -0.5 * ||x||^2
# ∇logπ(x) = -x
# ============================================================

logpost_normal(x) = -0.5 * dot(x, x)

function grad_normal(x)
    return -x
end

function parse_cli_args(args = ARGS)
    out = Dict{String,String}()
    i = 1

    while i <= length(args)
        a = args[i]

        if startswith(a, "--")
            key = a[3:end]

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

arg(args, key, default::Int) = parse(Int, get(args, key, string(default)))
arg(args, key, default::Float64) = parse(Float64, get(args, key, string(default)))
arg(args, key, default::String) = get(args, key, default)

function get_draws(samples)
    return getfield(samples, :draws)
end

function get_acceptance(samples)
    return getfield(samples, :acceptance_rate)
end

function get_warmup_acceptance(samples)
    return getfield(samples, :warmup_acceptance_rate)
end

function get_final_step(samples)
    return getfield(samples, :final_step)
end

function run_mala_chain(; d, n_samples, warmup, thin, step, seed)
    rng = MersenneTwister(seed)
    init = zeros(Float64, d)
    names = ["x[$j]" for j in 1:d]

    return BayesCox4Micro._run_full_mala(
        logpost_normal,
        grad_normal,
        init,
        names;
        n_samples = n_samples,
        warmup = warmup,
        thin = thin,
        rng = rng,
        mala_step_size = step,
        mala_target_accept = 0.574,
        mala_adapt_every = 50,
        mala_min_step = step / 100,
        mala_max_step = step * 100,
        progress = false,
        progress_desc = "normal MALA",
    )
end

function run_ula_chain(; d, n_samples, warmup, thin, step, seed)
    rng = MersenneTwister(seed)
    init = zeros(Float64, d)
    names = ["x[$j]" for j in 1:d]

    return BayesCox4Micro._run_full_ula(
        logpost_normal,
        grad_normal,
        init,
        names;
        n_samples = n_samples,
        warmup = warmup,
        thin = thin,
        rng = rng,
        mala_step_size = step,
        progress = false,
        progress_desc = "normal ULA",
    )
end

function summarize_one(label, samples; ula_step = nothing)
    draws = get_draws(samples)
    means = vec(mean(draws; dims = 1))
    vars = vec(var(draws; dims = 1))

    println()
    println("========== $label ==========")
    println("draws size              = ", size(draws))
    println("acceptance rate         = ", get_acceptance(samples))
    println("warmup acceptance rate  = ", get_warmup_acceptance(samples))
    println("final step              = ", get_final_step(samples))

    @printf("average mean            = %.4f\n", mean(means))
    @printf("max |mean|              = %.4f\n", maximum(abs.(means)))
    @printf("average variance        = %.4f\n", mean(vars))
    @printf("min variance            = %.4f\n", minimum(vars))
    @printf("max variance            = %.4f\n", maximum(vars))
    @printf("mean logpost            = %.4f\n", mean(getfield(samples, :logposterior)))

    if ula_step !== nothing
        # For standard normal ULA:
        # x_{t+1} = (1 - gamma)x_t + sqrt(2gamma) noise
        # stationary variance = 2 / (2 - gamma)
        @printf("ULA theoretical var     = %.4f\n", 2 / (2 - ula_step))
    end

    return (
        mean_abs_mean = mean(abs.(means)),
        max_abs_mean = maximum(abs.(means)),
        avg_var = mean(vars),
        min_var = minimum(vars),
        max_var = maximum(vars),
        acceptance = get_acceptance(samples),
    )
end

# ============================================================
# Plotting helpers
# ============================================================

function safe_label(label::String)
    return replace(lowercase(label), r"[^a-zA-Z0-9]+" => "_")
end

function ensure_plot_dir(plot_dir::String)
    isdir(plot_dir) || mkpath(plot_dir)
    return plot_dir
end

function normal_pdf(x)
    return exp(-0.5 * x^2) / sqrt(2π)
end

function plot_one_chain(
    label,
    samples;
    theoretical_var = 1.0,
    plot_dir = "figures",
    max_dims = 4,
)
    ensure_plot_dir(plot_dir)

    draws = get_draws(samples)
    logp = getfield(samples, :logposterior)

    n, d = size(draws)
    dims_to_plot = 1:min(d, max_dims)
    slug = safe_label(label)

    # ------------------------------------------------------------
    # 1. Trace plot for the first few dimensions
    # ------------------------------------------------------------
    p_trace = plot(
        title = "$label trace plot",
        xlabel = "Iteration",
        ylabel = "Value",
        legend = :topright,
    )

    for j in dims_to_plot
        plot!(
            p_trace,
            1:n,
            draws[:, j],
            label = "x[$j]",
            linewidth = 1,
        )
    end

    savefig(p_trace, joinpath(plot_dir, "$(slug)_trace.png"))

    # ------------------------------------------------------------
    # 2. Histogram for the first few dimensions
    # ------------------------------------------------------------
    for j in dims_to_plot
        p_hist = histogram(
            draws[:, j],
            bins = 50,
            normalize = :pdf,
            title = "$label histogram: x[$j]",
            xlabel = "x[$j]",
            ylabel = "Density",
            label = "samples",
            alpha = 0.6,
        )

        xs = range(-4, 4; length = 400)

        plot!(
            p_hist,
            xs,
            normal_pdf.(xs),
            label = "N(0, 1)",
            linewidth = 2,
        )

        savefig(p_hist, joinpath(plot_dir, "$(slug)_hist_x$j.png"))
    end

    # ------------------------------------------------------------
    # 3. Log-posterior trace
    # ------------------------------------------------------------
    p_logp = plot(
        1:length(logp),
        logp,
        title = "$label log posterior",
        xlabel = "Iteration",
        ylabel = "log posterior",
        label = "logπ(x)",
        linewidth = 1,
    )

    savefig(p_logp, joinpath(plot_dir, "$(slug)_logposterior.png"))

    # ------------------------------------------------------------
    # 4. Per-dimension variance
    # ------------------------------------------------------------
    vars = vec(var(draws; dims = 1))

    p_var = scatter(
        1:d,
        vars,
        title = "$label variance by dimension",
        xlabel = "Dimension",
        ylabel = "Sample variance",
        label = "sample variance",
    )

    hline!(
        p_var,
        [theoretical_var],
        label = "theoretical variance = $(round(theoretical_var; digits = 4))",
        linewidth = 2,
    )

    savefig(p_var, joinpath(plot_dir, "$(slug)_variance_by_dim.png"))

    return nothing
end

function plot_summary(mala_metrics, ula_metrics; ula_step, plot_dir = "figures")
    ensure_plot_dir(plot_dir)

    chains = 1:length(mala_metrics)
    ula_theoretical_var = 2 / (2 - ula_step)

    mala_avg_vars = [m.avg_var for m in mala_metrics]
    ula_avg_vars = [m.avg_var for m in ula_metrics]

    mala_max_means = [m.max_abs_mean for m in mala_metrics]
    ula_max_means = [m.max_abs_mean for m in ula_metrics]

    mala_acceptance = [m.acceptance for m in mala_metrics]
    ula_acceptance = [m.acceptance for m in ula_metrics]

    # ------------------------------------------------------------
    # Average variance by chain
    # ------------------------------------------------------------
    p_var = plot(
        chains,
        mala_avg_vars,
        marker = :circle,
        label = "MALA avg variance",
        title = "Average variance by chain",
        xlabel = "Chain",
        ylabel = "Average variance",
        linewidth = 2,
    )

    plot!(
        p_var,
        chains,
        ula_avg_vars,
        marker = :diamond,
        label = "ULA avg variance",
        linewidth = 2,
    )

    hline!(
        p_var,
        [1.0],
        label = "MALA target var = 1",
        linewidth = 2,
    )

    hline!(
        p_var,
        [ula_theoretical_var],
        label = "ULA theoretical var = $(round(ula_theoretical_var; digits = 4))",
        linewidth = 2,
    )

    savefig(p_var, joinpath(plot_dir, "summary_avg_variance.png"))

    # ------------------------------------------------------------
    # Max absolute mean by chain
    # ------------------------------------------------------------
    p_mean = plot(
        chains,
        mala_max_means,
        marker = :circle,
        label = "MALA max |mean|",
        title = "Max absolute mean by chain",
        xlabel = "Chain",
        ylabel = "max |mean|",
        linewidth = 2,
    )

    plot!(
        p_mean,
        chains,
        ula_max_means,
        marker = :diamond,
        label = "ULA max |mean|",
        linewidth = 2,
    )

    hline!(
        p_mean,
        [0.10],
        label = "warning threshold = 0.10",
        linewidth = 2,
    )

    savefig(p_mean, joinpath(plot_dir, "summary_max_abs_mean.png"))

    # ------------------------------------------------------------
    # Acceptance rate by chain
    # ------------------------------------------------------------
    p_acc = plot(
        chains,
        mala_acceptance,
        marker = :circle,
        label = "MALA acceptance",
        title = "Acceptance rate by chain",
        xlabel = "Chain",
        ylabel = "Acceptance rate",
        linewidth = 2,
    )

    plot!(
        p_acc,
        chains,
        ula_acceptance,
        marker = :diamond,
        label = "ULA acceptance",
        linewidth = 2,
    )

    hline!(
        p_acc,
        [0.574],
        label = "MALA target accept = 0.574",
        linewidth = 2,
    )

    savefig(p_acc, joinpath(plot_dir, "summary_acceptance.png"))

    return nothing
end

# ============================================================
# Main
# ============================================================

function main()
    args = parse_cli_args()

    d = arg(args, "d", 10)
    n_samples = arg(args, "samples", 20_000)
    warmup = arg(args, "warmup", 2_000)
    thin = arg(args, "thin", 1)
    chains = arg(args, "chains", 4)

    mala_step = arg(args, "mala-step", 0.5)
    ula_step = arg(args, "ula-step", 0.01)
    seed = arg(args, "seed", 1234)

    plot_dir = arg(args, "plot-dir", "figures")
    max_plot_dims = arg(args, "max-plot-dims", 4)

    println("Testing project MALA/ULA kernels on N(0, I)")
    println("d = $d, samples = $n_samples, warmup = $warmup, chains = $chains")
    println("mala_step = $mala_step, ula_step = $ula_step")
    println("plot_dir = ", abspath(plot_dir))
    println("max_plot_dims = $max_plot_dims")

    mala_metrics = []
    ula_metrics = []

    for c in 1:chains
        chain_seed = seed + 10_000 * c

        mala = run_mala_chain(
            d = d,
            n_samples = n_samples,
            warmup = warmup,
            thin = thin,
            step = mala_step,
            seed = chain_seed,
        )

        ula = run_ula_chain(
            d = d,
            n_samples = n_samples,
            warmup = warmup,
            thin = thin,
            step = ula_step,
            seed = chain_seed + 1,
        )

        mala_summary = summarize_one("MALA chain $c", mala)
        ula_summary = summarize_one("ULA chain $c", ula; ula_step = ula_step)

        push!(mala_metrics, mala_summary)
        push!(ula_metrics, ula_summary)

        plot_one_chain(
            "MALA chain $c",
            mala;
            theoretical_var = 1.0,
            plot_dir = plot_dir,
            max_dims = max_plot_dims,
        )

        plot_one_chain(
            "ULA chain $c",
            ula;
            theoretical_var = 2 / (2 - ula_step),
            plot_dir = plot_dir,
            max_dims = max_plot_dims,
        )
    end

    println()
    println("========== PASS / FAIL heuristic ==========")
    println("For target N(0, I):")
    println("  means should be close to 0")
    println("  MALA variance should be close to 1")
    println("  ULA variance should be close to 2 / (2 - gamma)")
    println()

    avg_mala_var = mean(m.avg_var for m in mala_metrics)
    avg_ula_var = mean(m.avg_var for m in ula_metrics)
    max_mala_mean = maximum(m.max_abs_mean for m in mala_metrics)
    max_ula_mean = maximum(m.max_abs_mean for m in ula_metrics)

    println("MALA avg variance over chains = ", avg_mala_var)
    println("MALA max |mean| over chains   = ", max_mala_mean)
    println("ULA avg variance over chains  = ", avg_ula_var)
    println("ULA max |mean| over chains    = ", max_ula_mean)
    println("ULA theoretical variance      = ", 2 / (2 - ula_step))

    plot_summary(
        mala_metrics,
        ula_metrics;
        ula_step = ula_step,
        plot_dir = plot_dir,
    )

    println()
    println("Saved diagnostic plots to: ", abspath(plot_dir))

    println()

    if abs(avg_mala_var - 1.0) < 0.10 && max_mala_mean < 0.10
        println("MALA sanity check: PASS")
    else
        println("MALA sanity check: WARNING")
        println("  MALA did not look close to N(0, I). Check step size or implementation.")
    end

    if abs(avg_ula_var - 2 / (2 - ula_step)) < 0.10 && max_ula_mean < 0.10
        println("ULA sanity check: PASS")
    else
        println("ULA sanity check: WARNING")
        println("  ULA did not match its expected stationary variance.")
    end
end

main()
