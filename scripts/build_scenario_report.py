#!/usr/bin/env python3
"""Build local S1--S4 summary files from audited saved reports, without sampling."""
import argparse
import hashlib
import json
from pathlib import Path

from aggregate_revision_diagnostics import GROUPS, aggregate, number, read_csv, write_csv
from aggregate_revision_metrics import METRICS, summarize, truth

LABELS = {'weak': 'S1', 'correlated_null': 'S2', 'counts': 'S3', 'phylo_bm': 'S4'}
GRAPH_LABELS = {'independent': 'Independent', 'correct': 'Correct CAR', 'misspecified': 'Misspecified CAR'}


def enrich(row):
    result = dict(row, setting=LABELS[row['scenario']])
    result['comparison'] = GRAPH_LABELS[row['car_graph']] if result['setting'] == 'S4' else row['method']
    return result


def fmt(value, digits=4):
    return f'{float(value):.{digits}f}' if number(value) is not None else 'NA'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def mean_sd(row, metric):
    return fmt(row[metric+'_mean']) + ' ± ' + fmt(row[metric+'_sd'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--legacy-reports', type=Path, required=True)
    parser.add_argument('--s4-reports', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    old, new, out = args.legacy_reports.resolve(), args.s4_reports.resolve(), args.outdir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    audit = json.loads((new/'saved_results_audit.json').read_text())
    assert audit['complete'] and audit['completed_fits'] == 150 and audit['all_available_checks_passed']
    numerical=[]
    for label,count in (('S1_S3',300),('S4',150)):
        path=new.parent/'audits'/label/'numerical_audit.json'
        checked=json.loads(path.read_text())
        assert checked['all_metrics_match'] and checked['verified_runs']==count
        numerical.append(checked)
        target=out/'audits'/label
        target.mkdir(parents=True,exist_ok=True)
        for name in ('numerical_audit.json','numerical_audit_by_run.csv'):
            (target/name).write_bytes((path.parent/name).read_bytes())
    metrics = [enrich(r) for r in read_csv(old/'performance_aggregates/metrics_by_run.csv')
               if r['scenario'] in ('weak','correlated_null','counts') and truth(r['primary_eligible'])]
    metrics += [enrich(r) for r in read_csv(new/'performance/metrics_by_run.csv') if truth(r['primary_eligible'])]
    diagnostics = [enrich(r) for r in read_csv(old/'diagnostic_aggregates_v2/diagnostics_by_run.csv')
                   if r['scenario'] in ('weak','correlated_null','counts') and truth(r['primary_eligible'])]
    diagnostics += [enrich(r) for r in read_csv(new/'diagnostics/diagnostics_by_run.csv') if truth(r['primary_eligible'])]
    assert len(diagnostics) == 450 and len(metrics) == 1350
    assert len({r['source'] for r in diagnostics}) == 450
    assert {(r['source'], r['estimate']) for r in metrics} == {
        (r['source'], e) for r in diagnostics for e in ('mean','median','MAP')}
    groups = ['setting', 'comparison'] + GROUPS
    summary = summarize(metrics, groups + ['estimate'])
    # aggregate() consumes typed booleans, while CSV readers return strings.
    typed = [{**r, 'primary_eligible': truth(r['primary_eligible']),
              'convergence_screen_pass': truth(r['convergence_screen_pass'])} for r in diagnostics]
    diagnostic_summary = [enrich(r) for r in aggregate(typed)]
    primary = [r for r in summary if r['selection']=='all_eligible' and r['estimate']=='mean']
    assert len(primary) == 9 and all(r['n_replicates']==50 for r in primary)
    order = {'cox-rhs':0,'prox-sslasso':1,'Independent':0,'Correct CAR':1,'Misspecified CAR':2}
    sortkey = lambda r:(r['setting'], order[r['comparison']])
    primary.sort(key=sortkey)
    progress = []
    for row in primary:
        subset = [r for r in diagnostics if (r['setting'],r['comparison']) == (row['setting'],row['comparison'])]
        assert {int(r['replicate']) for r in subset} == set(range(1,51))
        progress.append({k:row[k] for k in ('setting','comparison','n_replicates','n_convergence_screen_pass')})
    write_csv(out/'performance_by_run.csv', metrics)
    write_csv(out/'performance_summary.csv', summary)
    write_csv(out/'diagnostics_by_run.csv', diagnostics)
    write_csv(out/'diagnostics_summary.csv', diagnostic_summary)
    write_csv(out/'completion.csv', progress)
    contrasts = read_csv(new/'performance/paired_contrasts.csv')
    differences = read_csv(new/'performance/paired_differences.csv')
    write_csv(out/'S4_paired_contrasts.csv', contrasts)
    write_csv(out/'S4_paired_differences.csv', differences)
    for label in ('S1','S2','S3','S4'):
        write_csv(out/label/'performance_summary.csv', [r for r in summary if r['setting']==label])
        write_csv(out/label/'performance_by_run.csv', [r for r in metrics if r['setting']==label])
        write_csv(out/label/'diagnostics_by_run.csv', [r for r in diagnostics if r['setting']==label])
    (out/'S4'/'saved_results_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    main_contrasts = [r for r in contrasts if r['estimate']=='mean' and r['selection']=='all_eligible_pairs']
    assert len(main_contrasts)==2 and all(int(r['n_pairs'])==50 for r in main_contrasts)
    # These are Monte Carlo intervals for differences of simulation averages.
    contrast_rows = []
    for r in main_contrasts:
        row = [GRAPH_LABELS[r['setting']]+' − Independent', r['n_pairs']]
        for metric in ('Cindex_test','MSE','coverage_95_nonzero','empirical_FDP'):
            d=float(r[metric+'_mean_difference']); se=float(r[metric+'_paired_mcse'])
            row.append(f'{d:+.5f} ± {se:.5f}')
        contrast_rows.append(row)
    correct = next(r for r in main_contrasts if r['setting']=='correct')
    d=float(correct['Cindex_test_mean_difference']); se=float(correct['Cindex_test_paired_mcse'])
    lo,hi=d-1.96*se,d+1.96*se
    if lo>0:
        conclusion='正确图 CAR 在该 S4 设置下提高了平均测试 C-index，差异超过这里的 Monte Carlo 误差范围。'
    elif hi<0:
        conclusion='正确图 CAR 在该 S4 设置下的平均测试 C-index 更低，没有显示预测收益。'
    else:
        conclusion='正确图 CAR 与无 CAR 的平均测试 C-index 差异尚不能与 Monte Carlo 误差明确区分，未得到明确的预测改进证据。'
    fail_cells=[r for r in primary if r['n_convergence_screen_pass']<50]
    limits='；'.join(f"{r['setting']} {r['comparison']} 仅 {r['n_convergence_screen_pass']}/50 通过系数收敛筛查" for r in fail_cells)
    lines=[
        '# S1–S4 模拟实验完整汇总', '',
        '全部 9 个比较单元均完成 50 次重复，共 450 次拟合。S1–S3 复用已完成的正式实验（300 次）；S4 新增 Cox-RHS 的三组 CAR 比较（150 次）。本地仅保存汇总与诊断摘要，大规模后验抽样及中间文件保留在服务器。', '',
        '## 实验设置', '',
        '- 训练集 n=200，p=50，6 个真实非零系数；Weibull 基线。每次重复独立生成 n=1000 测试样本。',
        '- 每次拟合 4 条链，每链 2000 次 warm-up 和 5000 次保留抽样；共 20000 个后验抽样。',
        '- S1：真实系数为原系数的 0.35 倍；S2：6 个零系数变量分别与信号变量相关（潜在相关系数 0.6）。',
        '- S3：组成比例经多项分布产生测序计数，测序深度中位数 5000；零替换使用训练集最小正值的一半，再做闭合归一化。测试数据沿用训练得到的预处理。',
        '- S4：固定平衡系统发育树，50 个叶节点，正分支长度，树高 H=6。潜在高斯丰度的协方差为 Σᵢⱼ=根到最近共同祖先的共享路径长度/H，故对角线为 1。根状态固定为 0；softmax 转成组成数据。6 个信号预先置于相邻支系。',
        '- S4 中无 CAR、正确图 CAR、错误图 CAR 共享每次重复的训练集、测试集和系数真值。正确图来自生成树；错误图固定置换叶标签（seed=9017），仅改变拟合图，不改变数据。图边权 exp(−d/2)，只保留 d≤4 的非对角边。',
        '- S4 只运行 Cox-RHS：p0=6，prox 零和参数 0.001；CAR 强度使用现有自适应机制。S1–S3 沿用原已完成方法设置，不重新调参或采样。CAR 平滑局部对数收缩强度，并不是直接对丰度协方差或系数符号进行平滑。', '',
        '## 指标定义与汇总口径', '',
        '主表使用后验均值估计系数，显示 50 次重复的均值 ± 重复间标准差（SD）。CSV 同时包含 posterior mean、median、MAP，以及 SD、Monte Carlo 标准误（MCSE）和实际重复数。', '',
        '95% credible-interval coverage 使用各系数边际后验 2.5%–97.5% 分位区间，判断真实系数是否落入区间，再分别在全部、6 个信号和 44 个零系数中平均。区间不含 0 则选中该变量；每次重复 empirical FDP=FP/max(TP+FP,1)，汇报其跨重复平均。Coverage/FDP 来自同一组后验区间，不依赖 mean/median/MAP 的点估计选择。', '',
        '主结果包含所有符合实验设置的重复，不因收敛差而剔除。补充 CSV 的 convergence_screen_pass 表只保留所有 β 的 rank-normalized R-hat≤1.01、bulk ESS≥400、tail ESS≥400 的重复；这是敏感性汇总，不能替代主结果。', '',
        '## 预测与估计误差', '',
        table(['设置','方法/图','重复数','训练 C-index','测试 C-index','MSE'],
              [[r['setting'],r['comparison'],r['n_replicates'],mean_sd(r,'Cindex_train'),mean_sd(r,'Cindex_test'),mean_sd(r,'MSE')] for r in primary]), '',
        '## 区间覆盖率与经验 FDP', '',
        table(['设置','方法/图','Coverage：全部','Coverage：信号','Coverage：零系数','Empirical FDP'],
              [[r['setting'],r['comparison'],mean_sd(r,'coverage_95_all'),mean_sd(r,'coverage_95_nonzero'),mean_sd(r,'coverage_95_zero'),mean_sd(r,'empirical_FDP')] for r in primary]), '',
        '## S4：CAR 是否有效', '', '以后验均值作为系数估计：'+conclusion, '',
        f'正确图相对无 CAR 的平均测试 C-index 差为 {d:+.6f}，配对 MCSE={se:.6f}，近似 95% Monte Carlo 区间为 [{lo:+.6f}, {hi:+.6f}]。该区间反映 50 次模拟重复对平均差异的估计误差，不是参数的后验可信区间。', '',
        '下表为“当前图 − Independent”，显示平均差 ± 配对 MCSE。C-index 差越大越好，MSE/FDP 差越小越好；coverage 应结合名义 95% 判断，不能一概认为越大越好。', '',
        table(['比较','配对数','Δ 测试 C-index','Δ MSE','Δ 信号 coverage','Δ FDP'],contrast_rows), '',
        '该结论仅适用于本次固定树、固定稀疏信号布局与 CAR 强度先验。Brownian 丰度相关性本身不保证局部收缩先验一定带来预测收益。旧 phylo 使用 exp(−d/4) 的树距离协方差，已有树结构；S4 明确改用 Brownian 共享路径协方差，旧结果没有被重命名或覆盖。', '',
        '## 收敛与计算诊断', '',
        table(['设置','方法/图','通过 β 收敛筛查','平均最大 β R-hat','平均最小 bulk ESS','平均最小 tail ESS','采样发散总次数'],
            [[r['setting'],r['comparison'],str(r['n_convergence_screen_pass'])+'/50',
              fmt(next(x for x in diagnostic_summary if x['setting']==r['setting'] and x['comparison']==r['comparison'] and x['selection']=='all_eligible')['beta_rhat_max_mean']),
              fmt(next(x for x in diagnostic_summary if x['setting']==r['setting'] and x['comparison']==r['comparison'] and x['selection']=='all_eligible')['beta_ess_bulk_min_mean'],1),
              fmt(next(x for x in diagnostic_summary if x['setting']==r['setting'] and x['comparison']==r['comparison'] and x['selection']=='all_eligible')['beta_ess_tail_min_mean'],1),
              sum(int(float(x['n_divergent_sample'])) for x in diagnostics if x['setting']==r['setting'] and x['comparison']==r['comparison'])] for r in primary]), '',
        ('需结合采样质量解释：'+limits+'。这些结果保留在主表中，其 MCSE 不能概括未收敛造成的偏差。') if fail_cells else '全部 450 次拟合通过上述 β 收敛筛查；逐重复其他参数和转移诊断仍保留在摘要中。', '',
        '## 补充：后验中位数与 MAP', '',
        '下表保留其他点估计的预测与估计误差，显示均值 ± SD；区间 coverage 和 FDP 与主表共用同一后验区间。', '',
        table(['设置','方法/图','点估计','训练 C-index','测试 C-index','MSE'],
              [[r['setting'],r['comparison'],r['estimate'],mean_sd(r,'Cindex_train'),mean_sd(r,'Cindex_test'),mean_sd(r,'MSE')]
               for r in sorted(summary,key=lambda r:sortkey(r)+(r['estimate'],))
               if r['selection']=='all_eligible' and r['estimate'] in ('median','MAP')]), '',
        '## 独立数值核验', '',
        f"从服务器保存的训练/测试数据和后验系数摘要，独立重算全部 450 次拟合的 2700 个 C-index，并核对 MSE、coverage 和 FDP；全部匹配，最大绝对数值差 {max(r['max_abs_difference'] for r in numerical):.3g}。该核验读取已有系数摘要，不重新采样，也不重复读取大型抽样文件计算分位数。", '',
        '## 文件与复现', '',
        '- [完整性能汇总](performance_summary.csv)：全部设置、mean/median/MAP、主汇总和收敛筛查汇总。',
        '- [逐重复性能](performance_by_run.csv) 与 [逐重复诊断](diagnostics_by_run.csv)。',
        '- [S4 配对比较](S4_paired_contrasts.csv) 与 [逐重复配对差](S4_paired_differences.csv)。',
        '- [完成情况](completion.csv)、[S4 存档核验](S4/saved_results_audit.json)、[来源清单](sources.json)。',
        '- S1/、S2/、S3/、S4/ 子目录按设置保存完整汇总。',
        '- [S1–S4 性能图](figures/scenarios_performance.pdf) 与 [S4 配对效应图](figures/S4_paired_effects.pdf)；同时提供 PNG。',
        '- 服务器新实验：`/large_storage/zhangxf/BayesCox/s4_20261012/results/S4`；冻结代码：`/large_storage/zhangxf/BayesCox/s4_20261012/code_v3`。',
        '- S1–S3 服务器来源：`/large_storage/zhangxf/BayesCox/revision_20261011/results`，具体拟合路径见逐重复 CSV 的 source 列。',
        '- 本次运行：`python3 scripts/run_s4_pool.py --root S4_ROOT --julia JULIA_1_11_7 --partitions 6 --workers 18`；恢复时保持 partitions=6。只重算汇总：`bash scripts/refresh_s4_reports.sh S4_ROOT`。',
        '- 本地汇总：`python scripts/build_scenario_report.py --legacy-reports LEGACY_REPORTS --s4-reports S4_REPORTS --outdir OUTPUT`；作图：`Rscript scripts/plot_scenario_report.R OUTPUT`。',
        '- 汇总脚本不包含后验采样。已有兼容结果继续复用；大型后验 CSV 和中间数据不下载到本地。', '']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    sources=[]
    for path in (old/'performance_aggregates/metrics_by_run.csv',old/'diagnostic_aggregates_v2/diagnostics_by_run.csv',
                 new/'performance/metrics_by_run.csv',new/'diagnostics/diagnostics_by_run.csv',
                 new/'performance/paired_contrasts.csv',new/'saved_results_audit.json'):
        sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (out/'sources.json').write_text(json.dumps(dict(sources=sources,completed_fits=450,
        expected_cells=9,repetitions_per_cell=50,large_posterior_files_downloaded=False),indent=2)+'\n')
    print(json.dumps(dict(output=str(out),fits=len(diagnostics),metric_rows=len(metrics),summary_rows=len(summary))))


if __name__ == '__main__':
    main()
