"""Check source replacement and prevent selective retention/double counting."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from consolidate_review_results import merge_results, detail


def fit(source, rep, passed=False, **extra):
    return dict(source=source, method='prox-sslasso', n='200', p='50', nonzero='6',
                baseline='weibull', scenario='weak', constraint='prox', prox_sum0_lambda='1e-6',
                p0='', car_graph='independent', replicate=str(rep), primary_eligible='True',
                convergence_screen_pass=str(passed), **extra)


def metrics(rows):
    return [dict(row, estimate=e, MSE=str(i+1)) for i,row in enumerate(rows) for e in ('mean','median','MAP')]


class ConsolidationTests(unittest.TestCase):
    def setUp(self):
        self.old = [fit('/old/1',1), fit('/old/2',2,True)]
        self.new = [fit('/new/1',1,False,original_source='/old/1',result_version='initfix_random')]

    def merge(self, new=None, new_metrics=None):
        new = self.new if new is None else new
        return merge_results(self.old,metrics(self.old),new,metrics(new) if new_metrics is None else new_metrics,[],[])

    def test_failed_rerun_replaces_original_for_all_estimates(self):
        diag, rows = self.merge()
        self.assertEqual({r['source'] for r in diag},{'/new/1','/old/2'})
        self.assertEqual(len(rows),6)
        self.assertIsInstance(diag[0]['replicate'], int)
        self.assertFalse(next(r for r in diag if r['source']=='/new/1')['convergence_screen_pass'])
        self.assertEqual({r['estimate'] for r in rows if r['source']=='/new/1'},{'mean','median','MAP'})

    def test_wrong_cell_and_missing_estimate_rejected(self):
        wrong=copy.deepcopy(self.new); wrong[0]['replicate']='3'
        with self.assertRaisesRegex(AssertionError,'changes the experimental cell'): self.merge(wrong)
        with self.assertRaisesRegex(AssertionError,'Missing replacement estimate'): self.merge(new_metrics=metrics(self.new)[:-1])

    def test_duplicate_or_unmatched_replacements_rejected(self):
        with self.assertRaisesRegex(AssertionError,'Duplicate replacement source'): self.merge(self.new*2)
        wrong=copy.deepcopy(self.new); wrong[0]['original_source']='/missing'
        with self.assertRaisesRegex(AssertionError,'no eligible original'): self.merge(wrong)

    def test_rhat_relaxation_does_not_override_low_or_missing_ess(self):
        row=dict(self.new[0],beta_rhat_max='1.08',beta_ess_bulk_min='100',beta_ess_tail_min='600',
                 beta_n_rhat_unavailable='0',n_beta_parameters='50',gamma_ess_bulk_min='300',gamma_ess_tail_min='')
        d=detail(row)
        self.assertTrue(d['beta_rhat_le_1_1'])
        self.assertFalse(d['beta_screen_1_1'])
        self.assertTrue(d['gamma_any_reported_ess_below_400'])
        self.assertTrue(d['gamma_minimum_missing'])


if __name__=='__main__': unittest.main()
