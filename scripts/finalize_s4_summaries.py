#!/usr/bin/env python3
"""Wait for this S4 deployment, independently audit metrics, then package summaries."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve()
    status=root/'logs/formal.exit'
    while not status.is_file():
        time.sleep(5)
    assert status.read_text().strip()=='0', 'formal experiment did not finish successfully'
    env=dict(os.environ,PYTHONPATH=str(root/'code_v3/scripts'))
    command=[sys.executable,str(root/'analysis_tools/audit_scenario_metrics_v2.py'),
        '--diagnostics',str(root/'results/S4/reports/diagnostics/diagnostics_by_run.csv'),
        '--expected-runs','150','--outdir',str(root/'audits/S4')]
    subprocess.run(command,env=env,check=True)
    subprocess.run([sys.executable,str(root/'analysis_tools/export_s4_summaries.py'),
        '--root',str(root),'--output',str(root/'summary_bundle.zip')],check=True)
