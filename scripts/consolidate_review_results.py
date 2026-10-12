#!/usr/bin/env python3
"""Consolidate local review summaries, preserving evidence before cleanup.

No sampler is called. Server posterior/data directories are never removed.
"""
import argparse
import collections
import datetime
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

from aggregate_revision_diagnostics import GROUPS, aggregate, number, read_csv, write_csv
from aggregate_revision_metrics import METRICS, matches, requested_cells, summarize, truth
from build_scenario_report import table, mean_sd

OLD_NAMES = ('revision_20261011', 'revision_initfix_20261012', 's4_20261012', 'simulation_S1_S4_20261012')
NUMERIC_GROUPS = ('n', 'p', 'nonzero', 'p0', 'prox_sum0_lambda', 'replicate')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def normalize(row):
    r = dict(row)
    for key in NUMERIC_GROUPS:
        r[key] = number(r.get(key))
        if key != 'prox_sum0_lambda' and r[key] is not None:
            assert r[key].is_integer(), (key, r[key])
            r[key] = int(r[key])
    for key in ('primary_eligible', 'convergence_screen_pass'):
        r[key] = truth(r[key])
    return r


def identity(row, estimate=False):
    r = dict(row)
    for key in NUMERIC_GROUPS:
        r[key] = number(r.get(key))
    return tuple(r.get(k) for k in GROUPS+['replicate']+(['estimate'] if estimate else []))


def merge_results(old_diag, old_metrics, new_diag, new_metrics, s4_diag, s4_metrics):
    """Replace by audited source identity, retaining even failed new screens."""
    originals = {r['source']: normalize(r) for r in old_diag if truth(r['primary_eligible'])}
    replacement = {r['original_source']: normalize(r) for r in new_diag}
    assert len(replacement) == len(new_diag), 'Duplicate replacement source'
    assert set(replacement) <= set(originals), 'Replacement has no eligible original'
    for source, new in replacement.items():
        assert not originals[source]['convergence_screen_pass'], 'Unexpected replacement of a prior pass'
        assert identity(originals[source]) == identity(new), 'Replacement changes the experimental cell'
        assert new['primary_eligible'], 'Replacement is not eligible'
    diag = []
    for source, old in originals.items():
        diag.append(replacement.get(source, dict(old, original_source=source,
            result_version='review_original', init='not_recorded')))
    diag += [dict(normalize(r), original_source=r['source'], result_version='s4_original', init='not_recorded')
             for r in s4_diag if truth(r['primary_eligible'])]
    new_metric_map = {(r['source'], r['estimate']): normalize(r) for r in new_metrics}
    assert len(new_metric_map) == len(new_metrics), 'Duplicate replacement metric'
    expected_new = {(r['source'], e) for r in new_diag for e in ('mean', 'median', 'MAP')}
    assert set(new_metric_map) == expected_new, 'Missing replacement estimate'
    old_metric_map = {(r['source'], r['estimate']): normalize(r) for r in old_metrics if truth(r['primary_eligible'])}
    assert len(old_metric_map) == 3*len(originals), 'Missing or duplicate original estimates'
    s4_metric_map = {(r['source'], r['estimate']): normalize(r) for r in s4_metrics if truth(r['primary_eligible'])}
    metric_rows = []
    for d in diag:
        for e in ('mean', 'median', 'MAP'):
            index = new_metric_map if d['result_version'] == 'initfix_random' else s4_metric_map if d['result_version'] == 's4_original' else old_metric_map
            m = index[(d['source'], e)]
            metric_rows.append(dict(m, **d))
    assert len({identity(r) for r in diag}) == len(diag), 'Duplicate experimental cell/replicate'
    assert len({identity(r, True) for r in metric_rows}) == len(metric_rows)
    assert len({r['source'] for r in diag}) == len(diag)
    return diag, metric_rows


def stage(results, root):
    assert root.parent == results and root.name not in OLD_NAMES
    assert not root.exists(), 'Stage into a new directory'
    root.mkdir()
    copied = {}
    def copy(source, target):
        assert source.is_file() and not source.is_symlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha(source) == sha(target)
        copied[str(source.relative_to(results))] = str(target.relative_to(root))
    revision = results/OLD_NAMES[0]
    final = revision/'reports_final_all_simulations'
    for p in final.rglob('*'):
        if not p.is_file():
            continue
        rel = p.relative_to(final)
        dest = root/'T1D'/Path(*rel.parts[1:]) if rel.parts[0] == 'T1D_summary' else root/'provenance/revision_original'/rel
        copy(p, dest)
    for p in revision.iterdir():
        if p.is_file() and p.suffix not in ('.zip', '.gz'):
            copy(p, root/'provenance/revision_original/delivery'/p.name)
    s4 = results/OLD_NAMES[2]
    for p in (s4/'summary_bundle').rglob('*'):
        if p.is_file():
            copy(p, root/'provenance/s4'/p.relative_to(s4/'summary_bundle'))
    for name in ('code_s4_v3.tar.gz', 'validation_local.json'):
        copy(s4/name, root/'provenance/s4'/name)
    init = results/OLD_NAMES[1]
    for name in ('plan.json', 'source_manifest.json', 'code.tar.gz', 'initialization_tests.log', 'verify_review_dgp.jl'):
        copy(init/name, root/'provenance/initfix'/name)
    for p in (init/'summary_export').rglob('*'):
        if p.is_file():
            copy(p, root/'provenance/initfix/export'/p.relative_to(init/'summary_export'))
    for name in ('dgp_comparison.json', 'analysis_common_from_review.diff', 'simulation_scenarios_from_review.diff', 'run_simulation_from_review.diff'):
        copy(init/'reports'/name, root/'provenance/initfix'/name)
    for name in ('paired_comparison.csv', 'group_comparison.csv', 'selected_runs.csv', 'summary.json'):
        copy(init/'reports'/name, root/'initialization'/name)
    for p in (init/'logs').rglob('*'):
        if p.is_file():
            copy(p, root/'provenance/initfix/logs'/p.relative_to(init/'logs'))
    inventory = []
    targets = [results/name for name in OLD_NAMES]+[results/'simulation_S1_S4_20261012.zip']
    for target in targets:
        for p in ([target] if target.is_file() else sorted(target.rglob('*'))):
            if p.is_file():
                assert not p.is_symlink(), p
                rel = str(p.relative_to(results))
                inventory.append(dict(path=rel, bytes=p.stat().st_size, sha256=sha(p),
                    retained_path=copied.get(rel), disposition='preserved_evidence' if rel in copied else 'superseded_snapshot_or_regenerated_summary'))
    dump(root/'provenance/cleanup_inventory.json', dict(results_root=str(results), targets=[str(t.relative_to(results)) for t in targets], files=inventory))
    print(json.dumps(dict(staged=str(root), old_files=len(inventory), preserved_evidence_files=len(copied))))


def detail(row):
    r = {k: row.get(k) for k in ['source', 'original_source', 'result_version']+GROUPS+['replicate']}
    for family in ('beta', 'gamma', 'all'):
        for key in ('rhat_max', 'ess_bulk_min', 'ess_tail_min', 'n_rhat_unavailable', 'n_ess_bulk_unavailable', 'n_ess_tail_unavailable'):
            r[family+'_'+key] = row.get(family+'_'+key)
    r['beta_screen_1_01'] = truth(row['convergence_screen_pass'])
    rhat, bulk, tail = [number(row.get('beta_'+k)) for k in ('rhat_max', 'ess_bulk_min', 'ess_tail_min')]
    available = number(row.get('beta_n_rhat_unavailable')) == 0 and number(row.get('n_beta_parameters')) == number(row['p'])
    r['beta_rhat_le_1_1'] = bool(available and rhat is not None and rhat <= 1.1)
    r['beta_ess_ge_400'] = bulk is not None and tail is not None and min(bulk, tail) >= 400
    r['beta_screen_1_1'] = r['beta_rhat_le_1_1'] and r['beta_ess_ge_400']
    gamma = [number(row.get('gamma_'+k)) for k in ('ess_bulk_min', 'ess_tail_min')]
    r['gamma_any_reported_ess_below_400'] = any(v is not None and v < 400 for v in gamma) if row['method'] == 'prox-sslasso' else None
    r['gamma_minimum_missing'] = any(v is None for v in gamma) if row['method'] == 'prox-sslasso' else None
    return r


def scenario_audits(root, diagnostics):
    old = read_csv(root/'provenance/s4/audits/S1_S3/numerical_audit_by_run.csv')
    new = read_csv(root/'provenance/initfix/export/numerical_audit_by_run.csv')
    lookup = {r['source']: r for r in old+new}
    sources = {r['source'] for r in diagnostics if r['scenario'] in ('weak', 'correlated_null', 'counts')}
    selected = [lookup[s] for s in sorted(sources)]
    assert len(selected) == 300 and all(int(r['n_score_checks']) == 6 for r in selected)
    out = root/'audits/scenarios/S1_S3'
    write_csv(out/'numerical_audit_by_run.csv', selected)
    dump(out/'numerical_audit.json', dict(all_metrics_match=True, verified_runs=300, verified_cindices=1800,
        max_abs_difference=max(float(r['max_abs_difference']) for r in selected),
        coverage_and_fdp_verified=True, posterior_resampled=False,
        reused_unchanged_audits=sum(r['source'] in {o['source'] for o in old} for r in selected),
        limitation='Independent recomputation from saved posterior coefficient summaries, not recomputation of quantiles from draws.'))
    for p in (root/'provenance/s4/audits/S4').iterdir():
        dest = root/'audits/scenarios/S4'/p.name
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)


def write_overview(root, diag, metrics, progress, requested):
    fail = [r for r in diag if not truth(r['convergence_screen_pass'])]
    init = [r for r in diag if r['result_version'] == 'initfix_random']
    methods = []
    for method in ('cox-rhs', 'prox-sslasso'):
        rows = [r for r in diag if r['method'] == method]
        methods.append([method, len(rows), sum(truth(r['convergence_screen_pass']) for r in rows),
                        sum(not truth(r['convergence_screen_pass']) for r in rows)])
    rows = [r for r in requested if r['selection'] == 'all_eligible' and r['estimate'] == 'mean']
    lines = ['# Review 实验统一结果汇总', '',
        '本目录合并原 revision、初始化修复重跑、S4 扩展和 S1–S4 报告。共有 **1,650 次不同模拟拟合**、4,950 行 mean/median/MAP 性能记录，另保留 T1D 的四组完整结果。34 个请求比较单元均有 50 次重复；其中 p0=6 与 λsum0=0.001 共用同一组 Cox-RHS 基线，所以是 33 个独立参数组合，不能将完成表各行直接相加作为拟合总数。', '',
        '## 新结果如何并入', '',
        '391 次之前未通过 β 筛查的 Prox-SSLASSO 拟合已按原实验单元和重复编号替换为初始化修复后的结果，所有重跑结果均保留，包括仍未通过的 22 次。其余 259 次 Prox-SSLASSO 结果沿用原拟合；因此本汇总并非全部重复都使用新初始化的实验。', '',
        '重跑使用独立可行随机初始化，不依赖普通 SSLASSO；训练/测试数据、种子和其余参数不变：λL1=1e-4、HMC 100 leapfrog、4 条链、每链预热 2000、保留 5000。λsum0 与 λL1 是不同参数，下表 λ 设置指零和约束参数。', '',
        '## 收敛和结果适用范围', '',
        f'β 筛查要求所有 β 的 rank-normalized R̂≤1.01、bulk ESS≥400、tail ESS≥400。合并后 {len(diag)-len(fail)}/{len(diag)} 次通过；未通过的结果仍进入主汇总。', '',
        table(['方法', '拟合数', 'β 筛查通过', '未通过'], methods), '',
        f'本轮初始化重跑 {sum(truth(r["convergence_screen_pass"]) for r in init)}/391 次通过。剩余 22 次均为弱信号场景，β 最大 R̂ 为 1.0832，均满足 1.1，但均有 bulk 或 tail ESS<400。旧的、未重跑的 Cox-RHS 仍有 55 次未通过，本次初始化修复不会改变它们。', '',
        '**β 通过不代表 γ 或全部参数通过。** 本轮重跑的非弱信号场景中，相关噪声 2 次、计数数据 2 次、旧 phylo 1 次的 γ bulk ESS<400，另有 γ tail ESS 不可计算。全部参数的诊断和缺失状态见 [逐次诊断](simulation/convergence_detail.csv)；没有将不可计算值填为 0 或算作通过。', '',
        '主表保留全部符合模型/预算要求的重复，报告跨重复均值 ± SD。CSV 同时提供 MCSE 和仅通过 β 筛查的敏感性汇总。MCSE 描述重复均值的 Monte Carlo 误差，不包含未混合导致的潜在偏差。初始实现被审计排除的 3,971 条历史/早期记录仍在 provenance 中，未进入当前主表。', '',
        '## 模拟性能：后验均值', '',
        '信号检出由边际 95% 可信区间不含 0 定义；empirical FDP=FP/max(TP+FP,1)，与 PIP>0.5 的变量选择口径不同。', '']
    labels = {'lambda': '零和 λ 敏感性', 'p0': 'Cox-RHS 先验稀疏度', 'scenarios': 'S1–S3 场景',
              'zero_sum': '零和约束比较', 'car': '旧 phylo 场景', 'car_s4': 'S4 Brownian 场景'}
    for suite in ('lambda', 'p0', 'scenarios', 'zero_sum', 'car', 'car_s4'):
        rr = [r for r in rows if r['suite'] == suite]
        lines += ['### '+labels[suite], '', table(['设置', '方法', 'β通过/50', '测试 C-index', 'MSE', '信号 coverage', 'FDP'],
            [[r['setting'], r['method'], r['n_convergence_screen_pass'], mean_sd(r, 'Cindex_test'),
              mean_sd(r, 'MSE'), mean_sd(r, 'coverage_95_nonzero'), mean_sd(r, 'empirical_FDP')] for r in rr]), '']
    clinical = read_csv(root/'T1D/nested_cv_summary.csv')
    lines += ['## T1D', '', '四组既有 T1D 结果及数值审计完整保留。本次 391 次重跑均为模拟实验，不表示 T1D 已使用修复后的初始化重新拟合。每组 5 个 outer folds，60 个保存的测试 C-index 已由原审计核验。', '',
        table(['方法', '临床调整', 'outer folds', '平均 CV C-index', '折间 SD'],
              [[r['method'], r['adjustment'], r['n_folds'], f"{float(r['CV_Cindex_mean']):.4f}", f"{float(r['CV_Cindex_sd']):.4f}"] for r in clinical]), '',
        '采样和 PH 适用性限制继续保留于 [sampling_quality.csv](T1D/sampling_quality.csv) 和 [ph_tests.csv](T1D/ph_tests.csv)。', '',
        '## 文件入口与复现', '',
        '- [完整逐次性能](simulation/performance_by_run.csv)、[性能汇总](simulation/performance_summary.csv)、[请求单元汇总](simulation/requested_performance.csv)、[完成表](simulation/completion.csv)。',
        '- [逐次诊断](simulation/diagnostics_by_run.csv)、[诊断汇总](simulation/diagnostics_summary.csv)、[阈值与 ESS 明细](simulation/convergence_detail.csv)。',
        '- [S1–S4 完整报告与图](scenarios/RESULTS.md)、[配对消融比较](simulation/paired_contrasts.csv)。',
        '- [初始化新旧对照](initialization/paired_comparison.csv)、[替换映射](initialization/replacements.csv)、[重跑阈值核对](initialization/threshold_comparison.csv)。',
        '- [数据与替换校验](audits/consolidation_checks.json)、[来源与旧文件清理记录](provenance/cleanup_inventory.json)、[保存证据说明](provenance/README.md)。',
        '- [T1D 临床调整配对比较](T1D/paired_cv_comparison.csv)、[临床系数](T1D/clinical_coefficients.csv)、[菌群比较](T1D/taxon_comparison.csv)。',
        '- 旧 phylo 使用 exp(−d/4) 协方差，S4 使用 Brownian 共享路径协方差；两者分开汇总。',
        '- 大型原始抽样和数据仍保留在 server197 的原实验目录，source 列为其真实路径。此次清理仅针对本地重复交付目录和压缩包。',
        '- 重建表格：`python3 scripts/consolidate_review_results.py build --root results/review_20261012`；重绘场景图：`Rscript scripts/plot_scenario_report.R results/review_20261012/scenarios`。',
        '- 校验：`python3 scripts/consolidate_review_results.py validate --root results/review_20261012`。这些步骤不重新抽样。', '']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    (root/'provenance/README.md').write_text('# 合并来源与历史证据\n\n'
        'revision_original 与 s4 保存合并前的最终证据；这些文件用于追溯，不是当前结果入口。'
        '其中旧诊断、旧性能表和旧审计只适用于它们记录的 source，不能套用到初始化重跑。'
        '当前结果从上级 RESULTS.md、simulation/ 和 scenarios/ 读取。\n\n'
        'initfix 保存冻结代码、391 次预先指定的重跑清单、新结果完整指标和独立数值校验；'
        'export 的每条记录保留 original_source。重建汇总只需本目录证据，不依赖已清理的四个本地目录。\n\n'
        '服务器上的原始抽样、数据和代码目录保留：\n\n'
        '- /large_storage/zhangxf/BayesCox/revision_20261011\n'
        '- /large_storage/zhangxf/BayesCox/revision_initfix_20261012\n'
        '- /large_storage/zhangxf/BayesCox/s4_20261012\n')


def build(root):
    old = root/'provenance/revision_original'; new = root/'provenance/initfix/export'; s4 = root/'provenance/s4/reports'
    inputs = [old/'diagnostic_aggregates_v2/diagnostics_by_run.csv', old/'performance_aggregates/metrics_by_run.csv',
              new/'diagnostics_by_run.csv', new/'metrics_by_run.csv', s4/'diagnostics/diagnostics_by_run.csv', s4/'performance/metrics_by_run.csv']
    raw = [read_csv(p) for p in inputs]
    assert len(raw[2]) == 391 and len(raw[3]) == 1173
    diag, metrics = merge_results(*raw)
    assert len(diag) == 1650 and len(metrics) == 4950
    sim = root/'simulation'
    write_csv(sim/'diagnostics_by_run.csv', diag)
    write_csv(sim/'performance_by_run.csv', metrics)
    write_csv(sim/'performance_summary.csv', summarize(metrics, GROUPS+['estimate']))
    diagnostics_summary = aggregate(diag)
    for row in diagnostics_summary:
        subset = [d for d in diag if all(d.get(k) == row.get(k) for k in GROUPS)
                  and (row['selection'] == 'all_eligible' or truth(d['convergence_screen_pass']))]
        for field in ('beta_rhat_max', 'beta_ess_bulk_min', 'beta_ess_tail_min', 'gamma_rhat_max', 'gamma_ess_bulk_min', 'gamma_ess_tail_min'):
            vals = [number(d.get(field)) for d in subset]; vals = [v for v in vals if v is not None]
            row[field+'_worst'] = (max(vals) if 'rhat' in field else min(vals)) if vals else None
    write_csv(sim/'diagnostics_summary.csv', diagnostics_summary)
    details = [detail(r) for r in diag]
    write_csv(sim/'convergence_detail.csv', details)
    progress, requested = [], []
    for suite, label, settings in requested_cells()+requested_cells('s4'):
        selected = [dict(r, suite=suite, setting=label) for r in metrics if matches(r, settings)]
        means = [r for r in selected if r['estimate'] == 'mean']
        assert len(means) == 50 and {int(r['replicate']) for r in means} == set(range(1, 51))
        progress.append(dict(suite=suite, setting=label, **settings, n_complete=50, expected_replicates=50,
            n_convergence_screen_pass=sum(truth(r['convergence_screen_pass']) for r in means),
            n_initialization_replacements=sum(r['result_version'] == 'initfix_random' for r in means), status='complete'))
        requested.extend(selected)
    requested_summary = summarize(requested, ['suite', 'setting']+GROUPS+['estimate'])
    write_csv(sim/'completion.csv', progress)
    write_csv(sim/'requested_performance.csv', requested_summary)
    scripts = Path(__file__).resolve().parent
    for profile in ('revision', 's4'):
        subprocess.run([sys.executable, str(scripts/'summarize_revision_contrasts.py'), '--metrics-by-run',
                        str(sim/'performance_by_run.csv'), '--outdir', str(sim/profile), '--profile', profile], check=True)
    for name in ('paired_contrasts.csv', 'paired_differences.csv'):
        write_csv(sim/name, read_csv(sim/'revision'/name)+read_csv(sim/'s4'/name))
    shutil.rmtree(sim/'revision'); shutil.rmtree(sim/'s4')
    replacements = [dict(original_source=r['original_source'], source=r['source'], tag=r['replacement_tag'],
                         beta_screen_pass=r['convergence_screen_pass']) for r in diag if r['result_version'] == 'initfix_random']
    write_csv(root/'initialization/replacements.csv', replacements)
    write_csv(root/'initialization/threshold_comparison.csv', [r for r in details if r['result_version'] == 'initfix_random'])
    scenario_audits(root, diag)
    subprocess.run([sys.executable, str(scripts/'build_scenario_report.py'), '--unified-results', str(sim),
        '--s4-reports', str(s4), '--numerical-audits-root', str(root/'audits/scenarios'), '--outdir', str(root/'scenarios')], check=True)
    # Retain the S4 tree and source records alongside the scenario presentation.
    for name in ('phylogeny', 'provenance'):
        source = root/'provenance/s4'/name
        if name == 'phylogeny':
            shutil.copytree(source, root/'scenarios/S4/phylogeny', dirs_exist_ok=True)
        else:
            for filename in ('experiment.json', 'formal.command.json', 'source_manifest.json'):
                shutil.copy2(source/filename, root/'scenarios/S4'/filename)
    write_overview(root, diag, metrics, progress, requested_summary)
    dump(root/'sources.json', dict(inputs=[dict(path=str(p.relative_to(root)), sha256=sha(p)) for p in inputs],
        current_simulation_fits=1650, replacements=391, replacement_policy='Use all predeclared reruns regardless of new convergence',
        historical_excluded_rows=len(raw[0])-1500, server_raw_files_preserved=True))
    validate(root, check_manifest=False)
    print('Built unified review summaries:', root)


def validate(root, check_manifest=True):
    diag = read_csv(root/'simulation/diagnostics_by_run.csv'); metrics = read_csv(root/'simulation/performance_by_run.csv')
    assert len(diag) == 1650 and len(metrics) == 4950
    assert len({identity(r) for r in diag}) == 1650 and len({identity(r, True) for r in metrics}) == 4950
    assert {(r['source'], e) for r in diag for e in ('mean', 'median', 'MAP')} == {(r['source'], r['estimate']) for r in metrics}
    assert sum(truth(r['convergence_screen_pass']) for r in diag) == 1573
    replacement = [r for r in diag if r['result_version'] == 'initfix_random']
    assert len(replacement) == 391 and sum(truth(r['convergence_screen_pass']) for r in replacement) == 369
    inventory = json.loads((root/'provenance/cleanup_inventory.json').read_text())
    for row in inventory['files']:
        if row['retained_path']:
            assert sha(root/row['retained_path']) == row['sha256'], row['retained_path']
    progress = read_csv(root/'simulation/completion.csv')
    assert len(progress) == 34 and all(r['n_complete'] == '50' for r in progress)
    assert len(read_csv(root/'T1D/nested_cv_folds.csv')) == 20
    assert len(read_csv(root/'T1D/nested_cv_summary.csv')) == 4
    # Independent arithmetic check of every aggregate metric, including SD/MCSE.
    checked = 0
    for row in read_csv(root/'simulation/performance_summary.csv'):
        selected = [r for r in metrics if identity(r)[:-1] == identity(row)[:-1] and r['estimate'] == row['estimate']
                    and (row['selection'] == 'all_eligible' or truth(r['convergence_screen_pass']))]
        assert len(selected) == int(row['n_replicates'])
        for key in METRICS:
            vals = [float(r[key]) for r in selected if number(r.get(key)) is not None]
            assert len(vals) == int(row[key+'_n'])
            if not vals:
                assert number(row[key+'_mean']) is None
                continue
            mean = math.fsum(vals)/len(vals)
            assert math.isclose(mean, float(row[key+'_mean']), rel_tol=1e-12, abs_tol=1e-12)
            if len(vals) > 1:
                sd = math.sqrt(math.fsum((v-mean)**2 for v in vals)/(len(vals)-1))
                assert math.isclose(sd, float(row[key+'_sd']), rel_tol=1e-12, abs_tol=1e-12)
                assert math.isclose(sd/math.sqrt(len(vals)), float(row[key+'_mcse']), rel_tol=1e-12, abs_tol=1e-12)
            checked += 1
    scenarios = read_csv(root/'scenarios/diagnostics_by_run.csv')
    assert len(scenarios) == 450 and len(read_csv(root/'scenarios/performance_by_run.csv')) == 1350
    for r in scenarios:
        active = next(d for d in diag if d['source'] == r['source'])
        assert r['convergence_screen_pass'] == active['convergence_screen_pass']
    # The unchanged S4 figures/tables must still use the original 150 fits.
    original_s4 = read_csv(root/'provenance/s4/reports/performance/metrics_by_run.csv')
    index = {(r['source'], r['estimate']): r for r in metrics}
    for row in original_s4:
        for metric in METRICS:
            assert number(row[metric]) == number(index[(row['source'], row['estimate'])][metric])
    if check_manifest:
        for report in (root/'RESULTS.md', root/'scenarios/RESULTS.md'):
            for target in re.findall(r'\]\(([^)]+)\)', report.read_text()):
                if '://' not in target and not target.startswith('#'):
                    assert (report.parent/target.split('#', 1)[0]).exists(), (report, target)
        for stem in ('scenarios_performance', 'S4_paired_effects'):
            assert (root/'scenarios/figures'/(stem+'.pdf')).read_bytes().startswith(b'%PDF')
            assert (root/'scenarios/figures'/(stem+'.png')).read_bytes().startswith(b'\x89PNG')
    if check_manifest and (root/'manifest.json').exists():
        for name, expected in json.loads((root/'manifest.json').read_text())['files'].items():
            assert sha(root/name) == expected['sha256'], name
    result = dict(all_checks_passed=True, simulation_fits=1650, metric_rows=4950, unique_cells=33,
        requested_views=34, replacements=391, retained_old_ssl=259, beta_pass=1573, beta_fail=77,
        rerun_beta_pass=369, rerun_beta_fail=22, t1d_arms=4, t1d_outer_folds=20,
        independently_checked_aggregate_metrics=checked, s4_metrics_unchanged=True,
        copied_evidence_hashes_match=True, no_sampling_performed=True)
    dump(root/'audits/consolidation_checks.json', result)
    return result


def manifest(root):
    files = {str(p.relative_to(root)): dict(sha256=sha(p), bytes=p.stat().st_size)
             for p in sorted(root.rglob('*')) if p.is_file() and p != root/'manifest.json'}
    dump(root/'manifest.json', dict(files=files, checks_passed=True, raw_posteriors_included=False))


def cleanup(root, results):
    assert (root/'manifest.json').is_file(), 'Finalize the bundle manifest before cleanup'
    validate(root)
    assert root.parent == results and root.name not in OLD_NAMES
    inv = json.loads((root/'provenance/cleanup_inventory.json').read_text())
    expected = set(OLD_NAMES+('simulation_S1_S4_20261012.zip',))
    assert set(inv['targets']) == expected
    current = set()
    for name in expected:
        p = results/name
        current.update(str(x.relative_to(results)) for x in ([p] if p.is_file() else p.rglob('*')) if x.is_file())
    already_removed = not any((results/name).exists() for name in expected)
    if not already_removed:
        assert current == {r['path'] for r in inv['files']}, 'Source inventory changed after staging'
        for row in inv['files']:
            assert sha(results/row['path']) == row['sha256'], 'Source file changed: '+row['path']
    # All data and provenance checks have passed before deleting any source.
    for name in sorted(expected):
        p = results/name
        assert p.parent == results and not p.is_symlink()
        if not p.exists():
            continue  # An optional archive may already be absent; finalization is resumable.
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
    assert not any((results/name).exists() for name in expected)
    dump(root/'audits/cleanup_completed.json', dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        deleted_local_paths=sorted(expected), finalized_after_removal=already_removed, old_files=len(inv['files']),
        old_bytes=sum(r['bytes'] for r in inv['files']), server_raw_directories_modified=False))
    manifest(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('stage', 'build', 'validate', 'manifest', 'cleanup'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--results-root', type=Path, default=Path('results'))
    args = parser.parse_args(); root = args.root.resolve(); results = args.results_root.resolve()
    if args.action == 'stage': stage(results, root)
    elif args.action == 'build': build(root)
    elif args.action == 'validate': print(json.dumps(validate(root)))
    elif args.action == 'manifest': manifest(root)
    else: cleanup(root, results)
