#!/usr/bin/env python3
"""Queue the 2026-10-13 rerun through the repository's shell launchers.

Both hosts see the same output filesystem. Atomic mkdir claims are never
reclaimed on a timeout: an operator must verify the owning process first.
193 finishes the changed supplements before joining 197's main queue.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time


PROJECT = Path(__file__).resolve().parents[2]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def plan():
    tasks = []
    common = dict(REPS="50", CHAINS="4", THREADS="4", SAMPLES="5000",
                  SAMPLER="hmc", TARGET_ACCEPT="0.8", WORKERS="1",
                  SAMPLE_THIN="1", DIAGNOSE="false", PROGRESS_INTERVAL="10")

    def add(suite, tag, script, env, relative_out, expected, args=()):
        for start in range(1, 51, 5):
            reps = list(range(start, start + 5))
            batch = f"batch{start:02d}"
            tasks.append(dict(id=f"{suite}__{tag}__{batch}", suite=suite,
                              script=script, args=list(args), reps=reps,
                              env=common | env, output=f"{suite}/{tag}/{batch}",
                              relative_out=relative_out, expected=expected))

    # All 16 DGPs, three method/constraint combinations, 50 paired seeds.
    for baseline, n, p, nz in itertools.product(("weibull", "gompertz"), (200, 400), (50, 150), (6, 12)):
        for method, constraint in (("cox-rhs", "prox"), ("cox-rhs", "soft_center"), ("prox-sslasso", "prox")):
            rhs = method == "cox-rhs"
            tag = f"{baseline}_n{n}_p{p}_nz{nz}_{method}_{constraint}"
            env = dict(N_VALUES=str(n), P_VALUES=str(p), NONZERO_VALUES=str(nz),
                       BASELINES=baseline, PRIORS=method, CONSTRAINTS=constraint,
                       WARMUP="2000", HMC_N_LEAPFROG="300", INIT="zero" if rhs else "rand",
                       USE_CAR=str(rhs).lower(), SAMPLE_CAR_STRENGTH=str(rhs).lower(),
                       CAR_STRENGTH_PRIOR_B="1", PROX_SUM0_LAMBDA="1e-3", PROX_L1_LAMBDA="1e-6")
            expected = dict(method_name=method, baseline=baseline, n=n, p=p,
                            nonzero_count=nz, constraint=constraint, warmup=2000,
                            hmc_n_leapfrog=300, init=env["INIT"], use_car=rhs,
                            prox_sum0_lambda=0.001, prox_l1_lambda=1e-6)
            add("main", tag, "scripts/run_simulation_hspriors.sh" if rhs else "scripts/run_simulation_proxpriors.sh",
                env, f"{constraint}/n{n}_p{p}_nonzero{nz}_{baseline}", expected)

    # Only scientifically changed supplementary settings. The L1=1e-6 cell
    # is identical to lambda/SSL/1e-3 and is reused, not sampled twice.
    ssl = dict(method_name="prox-sslasso", baseline="weibull", n=400, p=50,
               nonzero_count=6, constraint="prox", warmup=2000,
               hmc_n_leapfrog=100, init="rand", prox_l1_lambda=1e-6, use_car=False)
    for value in ("1e-2", "1e-3", "1e-4", "1e-5"):
        add("supplementary", f"lambda_ssl_{value}", "scripts/review/run_sensitivity_revision.sh",
            dict(PRIORS="prox-sslasso", VALUES=value, WARMUP="2000", HMC_N_LEAPFROG="100", INIT="rand"),
            f"lambda/prox-sslasso/{value}", ssl | dict(prox_sum0_lambda=float(value)), ("lambda",))
    for value in ("3", "6", "12"):
        add("supplementary", f"p0_rhs_{value}", "scripts/review/run_sensitivity_revision.sh",
            dict(VALUES=value, WARMUP="2000", HMC_N_LEAPFROG="100", INIT="rand"),
            f"p0/soft_center/cox-rhs/{value}",
            ssl | dict(method_name="cox-rhs", p0=int(value), constraint="soft_center", use_car=True,
                       prox_sum0_lambda=0.001), ("p0",))
    for constraint, value in (("exact", "1e-3"), ("none", "1e-3"), ("soft_center", "1e-3"),
                              ("prox", "1e-3"), ("prox", "1e-6")):
        label = constraint if constraint != "prox" else "prox" + value
        add("supplementary", f"zero_ssl_{label}", "scripts/review/run_zero_sum_revision.sh",
            dict(PRIORS="prox-sslasso", CONSTRAINTS=constraint, SUM0_LAMBDA=value,
                 WARMUP="2000", HMC_N_LEAPFROG="100"), f"zero_sum/{label}/prox-sslasso",
            ssl | dict(baseline="gompertz", p=150, nonzero_count=12, constraint=constraint,
                       prox_sum0_lambda=float(value)), ("--init", "rand"))
    for scenario, method in itertools.product(("S1", "S2", "S3"), ("cox-rhs", "prox-sslasso")):
        rhs = method == "cox-rhs"
        add("supplementary", f"{scenario}_{method}", "scripts/review/run_scenarios_revision.sh",
            dict(PRIORS=method, SCENARIOS=scenario, WARMUP="5000", HMC_N_LEAPFROG="300"),
            f"scenarios/revised_20261013/{scenario}/{method}",
            ssl | dict(method_name=method, n=200, constraint="soft_center" if rhs else "prox",
                       scenario={"S1": "weak", "S2": "correlated_null", "S3": "counts"}[scenario],
                       warmup=5000, hmc_n_leapfrog=300, use_car=rhs, prox_sum0_lambda=0.001),
            ("--init", "rand"))
    counts = collections.Counter()
    for task in tasks:
        counts[task["suite"]] += len(task["reps"])
    assert counts == {"main": 2400, "supplementary": 900}
    assert len({t["id"] for t in tasks}) == len(tasks)
    return dict(schema=1, protocol="review_rerun_20261013_v1", expected_fits=dict(counts),
                policy="Retain every repetition, including diagnostic failures. Never mix historical and new draws.",
                reuse=dict(l1_1e6="supplementary/lambda_ssl_1e-3", unchanged=[
                    "coda4microbiome", "FLORAL", "T1D", "BCCox", "S4",
                    "Cox-RHS zero-sum comparisons", "Cox-RHS lambda sensitivity", "L1 1e-2/1e-3/1e-4/1e-5"],
                    followup="New S1 supersedes duplicate weak-confirm plans; historical tuning pilots remain provenance."),
                tasks=tasks)


def invocation(task, root, julia):
    taskroot = root / "fits" / task["output"]
    env = os.environ.copy()
    env.update(task["env"])
    env.update(JULIA_BIN=julia, OUT_ROOT=str(taskroot), REVISION_OUT=str(taskroot),
               LOG_ROOT=str(root / "logs" / task["id"]),
               REP_IDS=",".join(map(str, task["reps"])), OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", GKSwstype="100")
    args = ["bash", task["script"], *task["args"]]
    # sensitivity already forwards REP_IDS; the other launchers forward CLI.
    if "run_sensitivity_revision" not in task["script"]:
        args += ["--rep-ids", env["REP_IDS"]]
    return args, env, taskroot / task["relative_out"]


def verify(task, outdir):
    expected = dict(sampler="hmc", n_samples=5000, chains=4, target_accept=0.8,
                    saved_sample_thin=1, n_test=1000) | task["expected"]
    seen, sources = [], []
    for path in sorted((outdir / "runs").glob("*/metadata.json")):
        meta = json.loads(path.read_text())
        rep = meta["replicate_id"]
        if rep not in task["reps"]:
            raise ValueError(f"unexpected replicate {path}")
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(f"{path}: {key}: {meta.get(key)!r} != {value!r}")
        for name in ("metrics.csv", "posterior_summary.csv", "samples_wide.csv", "diagnostics.json",
                     "rank_diagnostics.csv", "transition_summary.csv", "zero_sum_summary.csv"):
            file = path.parent / name
            if not file.is_file() or file.stat().st_size == 0:
                raise ValueError(f"missing or empty {file}")
        metrics = list(csv.DictReader((path.parent / "metrics.csv").open()))
        if not metrics:
            raise ValueError(f"empty metrics: {path}")
        seen.append(rep)
        sources.append(str(path.parent))
    if sorted(seen) != task["reps"]:
        raise ValueError(f"replicates {seen} != {task['reps']}: {outdir}")
    return sources


def worker(args, protocol):
    root = Path(args.root).resolve()
    queue = root / "queue"
    for name in ("claims", "done", "failed", "workers"):
        (queue / name).mkdir(parents=True, exist_ok=True)
    all_tasks = protocol["tasks"]
    active = {}
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    host = socket.gethostname()
    phase = "supplementary" if args.host == "193" else "main"
    worker_file = queue / "workers" / f"{args.host}_{os.getpid()}.json"
    while True:
        for taskid, record in list(active.items()):
            task, process, log, outdir, owner = record
            rc = process.poll()
            if rc is None:
                continue
            log.close()
            result = owner | dict(exit_code=rc, finished_at=now())
            try:
                if rc != 0:
                    raise RuntimeError(f"launcher exit {rc}")
                result["sources"] = verify(task, outdir)
                dest = "done"
            except Exception as exc:
                result["error"] = str(exc)
                dest = "failed"
            write_json(queue / dest / f"{taskid}.json", result)
            print(f"{now()} {dest} {taskid}", flush=True)
            del active[taskid]
        if stopping:
            # Drain current jobs instead of abandoning live Julia children.
            if not active:
                break
        else:
            done = {p.stem for p in (queue / "done").glob("*.json")}
            failed = {p.stem for p in (queue / "failed").glob("*.json")}
            selected = [t for t in all_tasks if t["suite"] == phase]
            if phase == "supplementary" and all(t["id"] in done for t in selected):
                phase = "main"
                print(f"{now()} 193 supplements verified complete; joining main queue", flush=True)
                selected = [t for t in all_tasks if t["suite"] == phase]
            for task in selected:
                if len(active) >= args.workers:
                    break
                taskid = task["id"]
                if taskid in done or taskid in failed:
                    continue
                claim = queue / "claims" / taskid
                try:
                    claim.mkdir()
                except FileExistsError:
                    continue
                owner = dict(task=taskid, host=args.host, hostname=host,
                             worker_pid=os.getpid(), claimed_at=now(), phase=phase)
                write_json(claim / "owner.json", owner)
                cmd, env, outdir = invocation(task, root, args.julia)
                logpath = root / "logs" / f"{taskid}.launcher.log"
                logpath.parent.mkdir(parents=True, exist_ok=True)
                log = logpath.open("a")
                try:
                    process = subprocess.Popen(cmd, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
                except Exception as exc:
                    log.close()
                    write_json(queue / "failed" / f"{taskid}.json", owner | dict(error=str(exc), finished_at=now()))
                    continue
                owner.update(launcher_pid=process.pid, command=cmd, output=str(outdir),
                             environment={k: env[k] for k in task["env"]} | {k: env[k] for k in ("OUT_ROOT", "REVISION_OUT", "REP_IDS", "JULIA_BIN")})
                write_json(claim / "owner.json", owner)
                active[taskid] = task, process, log, outdir, owner
                print(f"{now()} started {taskid} pid={process.pid}", flush=True)
            if not active and all(t["id"] in done | failed for t in selected):
                write_json(worker_file, dict(host=args.host, pid=os.getpid(), phase=phase, state="failed" if failed else "complete", at=now()))
                return 1 if any(t["id"] in failed for t in selected) else 0
        write_json(worker_file, dict(host=args.host, pid=os.getpid(), phase=phase, state="draining" if stopping else "running",
                                    active={key: value[1].pid for key, value in active.items()}, at=now()))
        time.sleep(10)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "validate", "work"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--host", choices=("193", "197"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--julia", default="julia")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    protocol = plan()
    root = Path(args.root).resolve()
    path = root / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise RuntimeError("Existing protocol differs; use a new output root")
    else:
        write_json(path, protocol)
    if args.action == "plan":
        print(json.dumps(dict(tasks=len(protocol["tasks"]), expected_fits=protocol["expected_fits"])))
        return 0
    if args.action == "validate":
        import shlex
        commands = []
        for task in protocol["tasks"]:
            cmd, env, outdir = invocation(task, root, args.julia)
            env["DRY_RUN"] = "true"
            result = subprocess.run(cmd, cwd=PROJECT, env=env, capture_output=True, text=True, check=True)
            lines = [line for line in result.stdout.splitlines() if "scripts/run_simulation.jl " in line]
            if len(lines) != 1:
                raise RuntimeError(f"Expected one invocation for {task['id']}: {result.stdout}")
            argv = shlex.split(lines[0])
            def option(key):
                return argv[len(argv) - 1 - argv[::-1].index(key) + 1]
            assert option("--sampler") == "hmc"
            assert int(option("--hmc-n-leapfrog")) == task["expected"]["hmc_n_leapfrog"]
            assert option("--outdir") == str(outdir)
            assert option("--rep-ids") == env["REP_IDS"]
            assert option("--priors") == task["expected"]["method_name"]
            assert option("--constraint") == task["expected"]["constraint"]
            commands.append(dict(task=task["id"], command=argv))
        write_json(root / "dry_run_commands.json", commands)
        print(f"Validated {len(commands)} launcher invocations")
        return 0
    if not args.host:
        parser.error("--host required for work")
    return worker(args, protocol)


if __name__ == "__main__":
    raise SystemExit(main())
