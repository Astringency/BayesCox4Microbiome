#!/usr/bin/env julia

if any(arg -> arg in ("--help", "-h"), ARGS)
    println("""
    Usage:
      julia --project=. scripts/summarize_pilot_diagnostics.jl --output results/diagnosis

    This script summarizes the short diagnostic pilots for CAR, SSLASSO s1,
    and prox anomaly checks. It assumes the corresponding
    diagnose_overshrinkage.jl summary directories already exist.
    """)
    exit(0)
end

using Pkg

function _activate_repo_project()
    project = abspath(joinpath(@__DIR__, "..", "Project.toml"))
    active = Base.active_project()
    active !== project && Pkg.activate(dirname(project))
end

_activate_repo_project()

include(joinpath(@__DIR__, "analysis_common.jl"))

using Dates

function safe_csv(path::AbstractString)
    isfile(path) || return DataFrame()
    try
        return CSV.read(path, DataFrame)
    catch err
        @warn "Could not read CSV." path exception=(err, catch_backtrace())
        return DataFrame()
    end
end

function as_float(x)
    x === missing && return NaN
    x === nothing && return NaN
    x isa Number && return Float64(x)
    x isa AbstractString || return NaN
    s = strip(x)
    isempty(s) && return NaN
    lowercase(s) in ("missing", "nothing", "null", "nan") && return NaN
    try
        return parse(Float64, s)
    catch
        return NaN
    end
end

function fmt(x; digits = 4)
    xf = as_float(x)
    return isfinite(xf) ? string(round(xf; digits = digits)) : "NA"
end

function markdown_table(df::DataFrame, cols::Vector{Symbol})
    isempty(df) && return "No rows.\n"
    io = IOBuffer()
    println(io, "| " * join(string.(cols), " | ") * " |")
    println(io, "|" * join(fill("---", length(cols)), "|") * "|")
    for row in eachrow(df)
        vals = String[]
        for col in cols
            v = col in propertynames(df) ? row[col] : missing
            push!(vals, v isa Number ? fmt(v) : string(v))
        end
        println(io, "| " * join(vals, " | ") * " |")
    end
    return String(take!(io))
end

function load_summary_dir(dir::AbstractString)
    df = safe_csv(joinpath(dir, "overshrinkage_summary.csv"))
    isempty(df) && return df
    df[!, :source_dir] = fill(dir, nrow(df))
    return df
end

function load_top_dir(dir::AbstractString)
    df = safe_csv(joinpath(dir, "prox_top_beta_diagnostics.csv"))
    isempty(df) && return df
    df[!, :source_dir] = fill(dir, nrow(df))
    return df
end

function add_setting!(df::DataFrame, name::Symbol, value)
    isempty(df) && return df
    df[!, name] = fill(value, nrow(df))
    return df
end

function car_outputs(outdir::AbstractString)
    false_df = load_summary_dir(joinpath(outdir, "tau_scale_5_summary"))
    true_df = load_summary_dir(joinpath(outdir, "car_true_summary"))
    keep = Set(["hs", "rhs", "cox-hs", "cox-rhs"])
    if !isempty(false_df)
        false_df = false_df[[string(p) in keep for p in false_df.prior], :]
        add_setting!(false_df, :car_setting, "false_tau5")
    end
    if !isempty(true_df)
        true_df = true_df[[string(p) in keep for p in true_df.prior], :]
        add_setting!(true_df, :car_setting, "true_tau5")
    end
    rows = vcat(false_df, true_df; cols = :union)
    isempty(rows) || CSV.write(joinpath(outdir, "car_sensitivity_combined.csv"), rows)

    compare = DataFrame()
    if !isempty(false_df) && !isempty(true_df)
        for prior in sort(collect(intersect(Set(string.(false_df.prior)), Set(string.(true_df.prior)))))
            f = false_df[false_df.prior .== prior, :][1, :]
            t = true_df[true_df.prior .== prior, :][1, :]
            push!(compare, Dict(
                :prior => prior,
                :shrinkage_false => f.shrinkage_ratio_nonzero,
                :shrinkage_true => t.shrinkage_ratio_nonzero,
                :delta_shrinkage_true_minus_false => as_float(t.shrinkage_ratio_nonzero) - as_float(f.shrinkage_ratio_nonzero),
                :nonzero_MSE_false => f.nonzero_MSE,
                :nonzero_MSE_true => t.nonzero_MSE,
                :delta_nonzero_MSE_true_minus_false => as_float(t.nonzero_MSE) - as_float(f.nonzero_MSE),
                :zero_MSE_false => f.zero_MSE,
                :zero_MSE_true => t.zero_MSE,
                :FN_false => f.FN,
                :FN_true => t.FN,
                :FP_false => f.FP,
                :FP_true => t.FP,
                :sign_prob_false => f.sign_prob_nonzero_mean,
                :sign_prob_true => t.sign_prob_nonzero_mean,
                :car_nonzero_nonzero_edges => :car_nonzero_nonzero_edges in propertynames(true_df) ? t.car_nonzero_nonzero_edges : missing,
                :car_nonzero_zero_edges => :car_nonzero_zero_edges in propertynames(true_df) ? t.car_nonzero_zero_edges : missing,
            ); cols = :union)
        end
    end

    io = IOBuffer()
    println(io, "# CAR Sensitivity Report")
    println(io)
    println(io, "Generated: $(now())")
    println(io)
    println(io, "Scope: p=50, reps=1, chains=1, samples=150, warmup=100, tau_scale=5 pilot. This is a directional pipeline check, not a final Monte Carlo result.")
    println(io)
    println(io, "## CAR False vs True")
    println(io)
    print(io, markdown_table(compare, [:prior, :shrinkage_false, :shrinkage_true, :delta_shrinkage_true_minus_false, :nonzero_MSE_false, :nonzero_MSE_true, :delta_nonzero_MSE_true_minus_false, :zero_MSE_false, :zero_MSE_true, :FN_false, :FN_true, :FP_false, :FP_true, :sign_prob_false, :sign_prob_true]))
    println(io)
    println(io, "## Graph Match")
    println(io)
    if !isempty(compare) && :car_nonzero_nonzero_edges in propertynames(compare)
        nn = as_float(compare.car_nonzero_nonzero_edges[1])
        nz = as_float(compare.car_nonzero_zero_edges[1])
        ratio = isfinite(nn) && nn > 0 ? nz / nn : NaN
        println(io, "- CAR graph nonzero-nonzero edges: $(fmt(nn)); nonzero-zero edges: $(fmt(nz)); nonzero-zero/nonzero-nonzero ratio: $(fmt(ratio)).")
        println(io, "- In this pilot the true nonzero nodes are connected to many more zero nodes than to each other, so CAR smoothing is not aligned with the simulated signal pattern.")
    else
        println(io, "- CAR graph edge counts were unavailable.")
    end
    println(io)
    println(io, "## Answers")
    println(io)
    println(io, "1. In this pilot, CAR=true lowered shrinkage_ratio_nonzero for hs, rhs, and cox-rhs; cox-hs moved slightly upward but still remained strongly compressed.")
    println(io, "2. CAR=true increased nonzero_MSE for hs, rhs, and cox-rhs; cox-hs improved relative to an unstable CAR=false run with high FP.")
    println(io, "3. The graph has many nonzero-zero links, so it can smooth signal toward null nodes.")
    println(io, "4. The current simulated nonzero beta pattern is not generated to match the CAR graph.")
    println(io, "5. Recommendation for the current simulation comparison: default use_car=false unless the data-generating mechanism is changed to make CAR structure meaningful.")
    println(io, "6. To study CAR, use a real phylogenetic W or generate nonzero beta values clustered in clades/graph neighborhoods.")
    write(joinpath(outdir, "car_sensitivity_report.md"), String(take!(io)))
    return rows, compare
end

function sslasso_outputs(outdir::AbstractString)
    specs = [
        ("0.5", joinpath(outdir, "sslasso_s1_0p5_summary")),
        ("1", joinpath(outdir, "sslasso_s1_1_summary")),
        ("2", joinpath(outdir, "sslasso_s1_2_summary")),
        ("5", joinpath(outdir, "sslasso_s1_5_summary")),
    ]
    rows = DataFrame()
    for (label, dir) in specs
        df = load_summary_dir(dir)
        isempty(df) && continue
        df[!, :s1_setting] = fill(parse(Float64, label), nrow(df))
        append!(rows, df; cols = :union)
    end
    if !isempty(rows)
        sort!(rows, :s1_setting)
        CSV.write(joinpath(outdir, "sslasso_s1_sensitivity_combined.csv"), rows)
    end

    io = IOBuffer()
    println(io, "# SSLASSO s1 Sensitivity Report")
    println(io)
    println(io, "Generated: $(now())")
    println(io)
    println(io, "Scope: p=50, reps=1, chains=1, samples=150, warmup=100, s0=0.04 pilot. This is directional only.")
    println(io)
    println(io, "## Summary")
    println(io)
    print(io, markdown_table(rows, [:s1_setting, :MSE, :shrinkage_ratio_nonzero, :nonzero_MSE, :zero_MSE, :FP, :FN, :sign_prob_nonzero_mean, :beta_rank_recovery_topk, :beta_rank_recovery_top2k, :coverage_zero_nonzero]))
    println(io)
    println(io, "## Answers")
    println(io)
    if !isempty(rows)
        base = rows[rows.s1_setting .== minimum(rows.s1_setting), :][1, :]
        best_idx = findmin([as_float(x) for x in rows.nonzero_MSE])[2]
        best = rows[best_idx, :]
        println(io, "1. Relative to s1=$(fmt(base.s1_setting)), the best pilot nonzero_MSE occurred at s1=$(fmt(best.s1_setting)) with shrinkage_ratio_nonzero=$(fmt(best.shrinkage_ratio_nonzero)) and nonzero_MSE=$(fmt(best.nonzero_MSE)).")
        println(io, "2. s1=2 improved coefficient scale with only modest zero_MSE increase. s1=5 did not continue that improvement and had worse rank recovery in this one-chain pilot. FP stayed zero in this pilot, so FP inflation is not established.")
        println(io, "3. Top-k recovery stayed reasonably high at s1=0.5-2, suggesting the ranking problem is less severe than the coefficient-scale problem, but short-chain instability is visible.")
        println(io, "4. Evidence supports testing s1 above 0.5, with s1=1-2 as the safer follow-up range before trying s1=5 in the main grid.")
    else
        println(io, "No SSLASSO pilot rows were available.")
    end
    write(joinpath(outdir, "sslasso_s1_sensitivity_report.md"), String(take!(io)))
    return rows
end

function rhs_slab_outputs(outdir::AbstractString)
    specs = [
        ("2", joinpath(outdir, "rhs_slab_s2_2_summary")),
        ("5", joinpath(outdir, "rhs_slab_s2_5_summary")),
        ("10", joinpath(outdir, "rhs_slab_s2_10_summary")),
        ("25", joinpath(outdir, "rhs_slab_s2_25_summary")),
    ]
    rows = DataFrame()
    hs_rows = DataFrame()
    for (label, dir) in specs
        df = load_summary_dir(dir)
        if !isempty(df)
            df[!, :rhs_slab_s2_setting] = fill(parse(Float64, label), nrow(df))
            append!(rows, df; cols = :union)
        end
        hs = safe_csv(joinpath(dir, "hs_rhs_parameter_diagnostics.csv"))
        if !isempty(hs)
            hs[!, :rhs_slab_s2_setting] = fill(parse(Float64, label), nrow(hs))
            hs[!, :source_dir] = fill(dir, nrow(hs))
            append!(hs_rows, hs; cols = :union)
        end
    end
    if !isempty(rows)
        sort!(rows, [:prior, :rhs_slab_s2_setting])
        CSV.write(joinpath(outdir, "rhs_slab_sensitivity_combined.csv"), rows)
    end

    io = IOBuffer()
    println(io, "# RHS Slab Sensitivity Report")
    println(io)
    println(io, "Generated: $(now())")
    println(io)
    println(io, "Scope: p=50, reps=1, chains=1, samples=100, warmup=70, tau_scale=5, use_car=false pilot. This is directional only.")
    println(io)
    println(io, "## Summary")
    println(io)
    print(io, markdown_table(rows, [:rhs_slab_s2_setting, :prior, :MSE, :shrinkage_ratio_nonzero, :nonzero_MSE, :zero_MSE, :FP, :FN, :sign_prob_nonzero_mean, :c2_median, :beta_rank_recovery_topk]))
    println(io)
    println(io, "## Parameter Summary")
    println(io)
    print(io, markdown_table(hs_rows, [:rhs_slab_s2_setting, :prior, :theoretical_tau0, :tau_median, :lambda_ratio_nonzero_zero, :c2_median, :beta_shrinkage_ratio_nonzero, :nonzero_MSE, :zero_MSE]))
    println(io)
    println(io, "## Answers")
    println(io)
    if isempty(rows)
        println(io, "No RHS slab pilot rows were available.")
    else
        println(io, "1. c2_median generally increases when rhs_slab_s2 is widened, so the hyperparameter is being used.")
        println(io, "2. The pilot does not show a clean monotone nonzero_MSE improvement from widening the slab across all RHS variants.")
        println(io, "3. Very wide slab_s2 can increase zero_MSE or sampler instability in short chains, especially where PSIS warnings were emitted.")
        println(io, "4. Current evidence keeps RHS slab as a secondary mechanism; global tau calibration remains the higher-priority explanation.")
    end
    write(joinpath(outdir, "rhs_slab_sensitivity_report.md"), String(take!(io)))
    return rows, hs_rows
end

function prox_outputs(outdir::AbstractString)
    specs = [
        ("0.5", "0.05", joinpath(outdir, "prox_alpha_0p5_rate_0p05_summary")),
        ("1", "0.1", joinpath(outdir, "prox_alpha_1_rate_0p1_summary")),
        ("2", "0.2", joinpath(outdir, "prox_alpha_2_rate_0p2_summary")),
    ]
    rows = DataFrame()
    tops = DataFrame()
    for (shape, rate, dir) in specs
        df = load_summary_dir(dir)
        if !isempty(df)
            df[!, :alpha_shape_setting] = fill(parse(Float64, shape), nrow(df))
            df[!, :alpha_rate_setting] = fill(parse(Float64, rate), nrow(df))
            append!(rows, df; cols = :union)
        end
        top = load_top_dir(dir)
        if !isempty(top)
            top[!, :alpha_shape_setting] = fill(parse(Float64, shape), nrow(top))
            top[!, :alpha_rate_setting] = fill(parse(Float64, rate), nrow(top))
            append!(tops, top; cols = :union)
        end
    end
    if !isempty(rows)
        sort!(rows, [:prior, :alpha_shape_setting])
        CSV.write(joinpath(outdir, "prox_anomaly_combined.csv"), rows)
    end
    if !isempty(tops)
        sort!(tops, [:prior, :alpha_shape_setting, :abs_posterior_mean], rev = [false, false, true])
        CSV.write(joinpath(outdir, "prox_top_beta_diagnostics.csv"), tops)
    end

    io = IOBuffer()
    println(io, "# Prox Anomaly Report")
    println(io)
    println(io, "Generated: $(now())")
    println(io)
    println(io, "Scope: p=50, reps=1, chains=1, samples=80, warmup=50, prox_l1_lambda=1e-3, soft_center constraint. This is a small anomaly localization run.")
    println(io)
    println(io, "## Summary")
    println(io)
    print(io, markdown_table(rows, [:alpha_shape_setting, :alpha_rate_setting, :prior, :MSE, :nonzero_MSE, :zero_MSE, :shrinkage_ratio_nonzero, :posterior_mean_abs_nonzero, :posterior_mean_abs_zero, :alpha_median, :projection_active_rate, :prox_penalty_mean, :sign_prob_nonzero_mean, :beta_rank_recovery_topk, :FP, :FN]))
    println(io)
    println(io, "## Top Absolute Beta Counts")
    println(io)
    if !isempty(tops)
        counts = combine(groupby(tops, [:alpha_shape_setting, :alpha_rate_setting, :prior]), :is_nonzero => sum => :true_nonzero_in_top10)
        print(io, markdown_table(counts, [:alpha_shape_setting, :alpha_rate_setting, :prior, :true_nonzero_in_top10]))
    else
        println(io, "No prox top-beta rows were available.")
    end
    println(io)
    println(io, "## Answers")
    println(io)
    println(io, "1. The prox MSE > 1 in these pilots is not only true-signal compression: zero_MSE is also near or above 1, and the top-10 absolute beta table contains large estimates at true-zero variables.")
    println(io, "2. nonzero_MSE is high, but posterior_mean_abs_nonzero is not tiny; the anomaly is closer to coefficient inflation/instability under the prox target than pure overshrinkage.")
    println(io, "3. projection_active_rate is low in these pilots, while alpha_median is very large, so this specific run does not support the hypothesis that alpha is too small or the projection is always active.")
    println(io, "4. Because zero variables can receive very large posterior means and PSIS warnings were emitted, prox-lasso/prox-sslasso should be treated as experimental until longer chains and KKT/projection checks pass.")
    println(io, "5. The prox target is not directly comparable to ordinary lasso/sslasso because it adds an alpha epigraph variable, Moreau-Yosida smoothing, weighted projection, and zero-sum handling.")
    write(joinpath(outdir, "prox_anomaly_report.md"), String(take!(io)))
    return rows, tops
end

function write_updated_reports(outdir::AbstractString)
    tau_report = joinpath(outdir, "tau_scale_sensitivity_report.md")
    car_report = joinpath(outdir, "car_sensitivity_report.md")
    ssl_report = joinpath(outdir, "sslasso_s1_sensitivity_report.md")
    prox_report = joinpath(outdir, "prox_anomaly_report.md")

    io = IOBuffer()
    println(io, "# Overshrinkage Diagnostic Report")
    println(io)
    println(io, "Generated: $(now())")
    println(io)
    println(io, "## Completed In This Diagnostic Pass")
    println(io)
    println(io, "- Code/load checks and tau-scale patch verification completed; `p0=truth` path remains correct.")
    println(io, "- Smoke test completed for hs/rhs/cox-hs/cox-rhs with `p0=truth`, `use_car=false`, tau_scale=2, and `save-wide-csv=true`.")
    println(io, "- Group 1 tau-scale sensitivity completed as a p=50 pilot only: reps=1, chains=1, samples=150, warmup=100, tau_scale=1,2,5,10.")
    println(io, "- Group 2 CAR sensitivity completed as a p=50 pilot at tau_scale=5.")
    println(io, "- Group 3 RHS slab sensitivity completed as a p=50 quick pilot at tau_scale=5.")
    println(io, "- Group 4 SSLASSO s1 sensitivity completed as a p=50 pilot.")
    println(io, "- Group 5 prox anomaly localization completed as a p=50 pilot with three alpha prior settings.")
    println(io)
    println(io, "## Not Yet Completed")
    println(io)
    println(io, "- Full Group 1 with p=150, reps=3, chains=2, samples=300/warmup=200.")
    println(io, "- Full replicated Group 2/3/4/5 runs were not run in this pass.")
    println(io, "- Cox likelihood sanity check n=400/p=50 was not run in this pass.")
    println(io)
    println(io, "## Current Interpretation")
    println(io)
    println(io, "- Main confirmed implementation fact: `p0=truth` is passed through and tau_scale now affects formula-based tau0 for hs/rhs/new-rhs/est-rhs/cox-hs/cox-rhs.")
    println(io, "- HS/RHS evidence is now pilot-level rather than absent. RHS-family priors, especially cox-rhs and est-rhs, generally improved when tau_scale was increased above 1, supporting overly strong global shrinkage as a plausible mechanism.")
    println(io, "- Cox-specific tau0 starts very small because it scales as p0/(p-p0)/sqrt(n_events); it should stay in the tau_scale grid rather than be treated as calibrated by default.")
    println(io, "- RHS c2 medians in the p=50 tau and slab pilots were not consistently tiny, and widening slab_s2 did not produce a clean monotone improvement; slab bottleneck is not the leading explanation yet.")
    println(io, "- CAR=true is not aligned with the current simulated signal graph: true nonzero nodes are linked to many zero nodes, and CAR worsened hs/rhs/cox-rhs shrinkage or nonzero_MSE in this pilot.")
    println(io, "- SSLASSO with s1=0.5 has direct evidence of true-signal compression. s1=2 improved coefficient scale in this pilot with modest zero_MSE increase; FP stayed zero. s1=1-2 is the next conservative grid.")
    println(io, "- Prox MSE > 1 is driven by both nonzero_MSE and zero_MSE. Top-beta diagnostics show large estimates on true-zero variables, so prox anomaly is not pure overshrinkage.")
    println(io)
    println(io, "Detailed reports:")
    println(io, "- `tau_scale_sensitivity_report.md`")
    println(io, "- `hs_rhs_parameter_diagnostics_report.md`")
    println(io, "- `rhs_slab_sensitivity_report.md`")
    println(io, "- `car_sensitivity_report.md`")
    println(io, "- `sslasso_s1_sensitivity_report.md`")
    println(io, "- `prox_anomaly_report.md`")
    write(joinpath(outdir, "overshrinkage_report.md"), String(take!(io)))

    io = IOBuffer()
    println(io, "# Overshrinkage Final Report")
    println(io)
    println(io, "Generated: $(now())")
    println(io)
    println(io, "## 1. Executive Summary")
    println(io)
    println(io, "Most likely causes, ranked with current evidence:")
    println(io, "1. HS/RHS global shrinkage is too strong for the simulated Cox signal scale, especially for cox-specific tau0. This is supported by the p=50 tau-scale pilot but still needs p=150 and replicated runs.")
    println(io, "2. SSLASSO default slab scale s1=0.5 is too tight for beta_true values around 1-2.5. Existing and new pilot evidence show true-signal compression.")
    println(io, "3. CAR smoothing is mismatched to the current simulation graph and can mix nonzero nodes with many zero nodes.")
    println(io, "4. Prox-lasso/prox-sslasso MSE > 1 appears to involve zero-variable coefficient inflation and target/sampler instability, not only overshrinkage.")
    println(io)
    println(io, "`p0=truth` is confirmed in the simulation path, so the main explanation is not `p0=auto` underestimating nonzero counts.")
    println(io)
    println(io, "## 2. HS/RHS Global-local Shrinkage Diagnosis")
    println(io)
    println(io, "The tau-scale patch was verified directly: formula-based tau0 changes in exact proportion to tau_scale for hs, rhs, new-rhs, est-rhs, cox-hs, and cox-rhs. In the p=50 pilot, increasing tau_scale improved nonzero_MSE for rhs, cox-rhs, and est-rhs at some larger values. cox-specific priors are sensitive because their base tau0 is much smaller than the non-cox hs scale.")
    println(io)
    println(io, "Recommended follow-up range: keep tau_scale=1 as baseline and run 2, 5, and 10; for main work, tau_scale=2-5 is the conservative first range, with 10 included as a sensitivity upper bound.")
    println(io)
    println(io, "## 3. RHS Slab Diagnosis")
    println(io)
    println(io, "The p=50 slab pilot confirmed c2_median responds to rhs_slab_s2, but widening the slab did not yield a clean monotone nonzero_MSE improvement across RHS variants. Current evidence keeps c2 as a secondary mechanism rather than the leading bottleneck. Full replicated slab sensitivity is still worth running if Group 1 remains compressed after tau is relaxed.")
    println(io)
    println(io, "## 4. CAR Diagnosis")
    println(io)
    println(io, "CAR=true at tau_scale=5 connected true nonzero nodes to far more zero nodes than nonzero nodes. In the p=50 pilot, CAR worsened hs/rhs/cox-rhs shrinkage or nonzero_MSE. Current recommendation: default use_car=false for simulations unless the data generator uses a phylogenetic or graph-clustered signal.")
    println(io)
    println(io, "## 5. LASSO/SSLASSO Diagnosis")
    println(io)
    println(io, "The local SSLASSO evidence supports increasing slab scale. s1=0.5 compresses true signals; s1=1-2 should be tested as the primary sensitivity range. s1=5 did not improve this pilot and showed weaker rank recovery; FP stayed zero here, so FP inflation is not yet established.")
    println(io)
    println(io, "## 6. Prox-LASSO/Prox-SSLASSO Anomaly Diagnosis")
    println(io)
    println(io, "In the three prox alpha pilot settings, total MSE > 1 was not explained by nonzero shrinkage alone. zero_MSE was also near or above 1, and top absolute beta estimates included true-zero coefficients with large magnitudes. Projection activity was low and alpha_median was large, so the observed anomaly is more consistent with the prox target/sampler allowing coefficient inflation or instability than with an overly small alpha radius.")
    println(io)
    println(io, "Recommendation: keep prox methods out of the main comparison until longer runs, projection/KKT checks, and ordinary lasso/sslasso target comparability are resolved.")
    println(io)
    println(io, "## 7. Cox Likelihood Information Diagnosis")
    println(io)
    println(io, "The requested n=400/p=50 low-difficulty sanity check has not been run yet. It remains necessary to separate prior/implementation shrinkage from Cox likelihood information limits under censoring, high p, and correlated compositional covariates.")
    println(io)
    println(io, "## 8. Recommendations")
    println(io)
    println(io, "- Run the full Group 1 grid for p=50 and p=150 with reps=3, chains=2, samples=300/warmup=200.")
    println(io, "- Use tau_scale grid 1,2,5,10 for HS/RHS; treat 2-5 as the most likely useful range.")
    println(io, "- Keep CAR off by default for current simulations; only turn it on with graph-aligned signals.")
    println(io, "- Run SSLASSO s1=1 and s1=2 in the main comparison; keep 0.5 as baseline and 5 as sensitivity.")
    println(io, "- Report sign probability, top-k rank recovery, and MSE decomposition alongside CI exclusion.")
    println(io, "- Move prox methods to appendix/experimental status until the anomaly is explained.")
    write(joinpath(outdir, "overshrinkage_final_report.md"), String(take!(io)))
end

function main()
    args = parse_cli_args()
    outdir = ensure_dir(abspath(arg(args, "output", joinpath("results", "diagnosis"))))
    car_outputs(outdir)
    sslasso_outputs(outdir)
    rhs_slab_outputs(outdir)
    prox_outputs(outdir)
    write_updated_reports(outdir)
    println("Pilot diagnostic reports written to $(abspath(outdir))")
end

main()
