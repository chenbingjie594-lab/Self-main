from pathlib import Path
import copy
import hashlib
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from generate_plastic_bomo_stage17a_frozen_banks import verify_preprobe, retained_slots, reproducibility
from audit_plastic_bomo_stage17a import load


class FrozenBankTests(unittest.TestCase):
    def test_uniform_drop_without_replacement(self):
        slots = [{'slot_id':f'{c}{i}','class':c} for c in ('flash','black') for i in range(40)]
        kept = retained_slots(slots,[{'slot_id':'flash0'}])
        self.assertEqual(len(kept),79)
        self.assertNotIn('flash0',[r['slot_id'] for r in kept])
        with self.assertRaisesRegex(RuntimeError,'INSUFFICIENT'):
            retained_slots(slots,[{'slot_id':f'flash{i}'} for i in range(6)])

    def test_exact_three_phase_comparison(self):
        rows = [{'bank':b,'slot_id':str(i),'rgb_sha256':f'{b}{i}'} for b in 'ABC' for i in range(5)]
        self.assertEqual(reproducibility(rows,rows,rows)['status'],'PRE_FORMAL_POST_RGB_EXACT_PASS')
        changed = copy.deepcopy(rows); changed[0]['rgb_sha256']='different'
        self.assertEqual(reproducibility(rows,changed,rows)['status'],'PRE_FORMAL_POST_RGB_EXACT_FAILED')
        self.assertEqual(reproducibility(rows,rows,rows[:-1])['status'],'PRE_FORMAL_POST_RGB_EXACT_FAILED')
        self.assertEqual(reproducibility(rows,rows,rows[:-1]+[rows[0]])['status'],'PRE_FORMAL_POST_RGB_EXACT_FAILED')

    def test_actual_downloaded_preprobe(self):
        base=ROOT/'results_for_gpt/plastic_bomo_stage17a_vanilla_utility_stability'
        probe=base/'pregeneration_20261007_024210/probe'
        slotpath=base/'slots_frozen_20261007_020941/generation_slot_manifest.json'
        if not probe.is_dir(): self.skipTest('server probe assets not installed')
        slots=load(slotpath)['slots']; digest=hashlib.sha256(slotpath.read_bytes()).hexdigest()
        fixed, records=verify_preprobe(probe,slots,digest)
        self.assertEqual(len(records),30)
        self.assertEqual(len(fixed['slot_ids']),5)
        with self.assertRaises(RuntimeError): verify_preprobe(probe,slots,'wrong hash')


if __name__=='__main__': unittest.main()
