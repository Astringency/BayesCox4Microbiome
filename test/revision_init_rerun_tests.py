"""Guard paired seeds, unchanged targets and fail-closed data comparisons."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rerun_revision_ssl_init import make_command, csv_data_match


class RerunChecks(unittest.TestCase):
    def test_seed_and_budget_are_preserved(self):
        meta = dict(replicate_id=21, seed=121, chains=4, n_samples=5000,
                    warmup=2000, n=200, p=50, nonzero_count=6,
                    sampler='hmc', hmc_n_leapfrog=100, prox_l1_lambda=1e-4,
                    prox_sum0_lambda=1e-6, use_car=False, noncentered=False)
        cmd = make_command(meta, Path('/new/output'), '/bin/julia')
        opts = dict(zip(cmd[cmd.index('--outdir')::2], cmd[cmd.index('--outdir')+1::2]))
        self.assertEqual(int(opts['--seed']) + int(opts['--rep-ids']) - 1, meta['seed'])
        self.assertEqual(opts['--init'], 'rand')
        self.assertEqual(opts['--warmup'], '2000')
        self.assertEqual(opts['--samples'], '5000')
        self.assertEqual(opts['--sampler'], 'hmc')
        self.assertEqual(opts['--hmc-n-leapfrog'], '100')
        self.assertEqual(float(opts['--prox-l1-lambda']), 1e-4)
        self.assertEqual(float(opts['--prox-sum0-lambda']), 1e-6)
        changed = make_command(meta, Path('/new/output'), '/bin/julia', 1e-6)
        diff = [(a,b) for a,b in zip(cmd,changed) if a!=b]
        self.assertEqual(diff, [('0.0001','1e-06')])

    def test_data_validation_rejects_changed_events_and_covariates(self):
        with tempfile.TemporaryDirectory() as scratch:
            a, b = Path(scratch)/'a.csv', Path(scratch)/'b.csv'
            a.write_text('x1,time,event\n0.2,1.0,1\n')
            b.write_text('x1,time,event\n0.20000000000001,1.0,1\n')
            self.assertLess(csv_data_match(a,b)['max_absolute_difference'], 1e-12)
            b.write_text('x1,time,event\n0.2,1.0,0\n')
            with self.assertRaises(AssertionError):
                csv_data_match(a,b)
            b.write_text('x1,time,event\n0.3,1.0,1\n')
            with self.assertRaises(AssertionError):
                csv_data_match(a,b)


if __name__ == '__main__':
    unittest.main()
