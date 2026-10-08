import ast
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from rebuild_plastic_bomo_stage17a_baseline import command


class RebuildTests(unittest.TestCase):
    def test_command_exact_budget_seed(self):
        c=command(Path('base'),Path('images'),Path('masks'),Path('output'),Path('audit.json'))
        for flag,value in {'--seed':'42','--max_train_steps':'2000','--learning_rate':'0.000001',
                           '--train_batch_size':'1','--gradient_accumulation_steps':'4','--text_noise_scale':'0.0'}.items():
            self.assertEqual(c[c.index(flag)+1],value)
        self.assertIn('--gradient_checkpointing',c)
        self.assertNotIn('--train_text_encoder',c)
        self.assertNotIn('--soft_mask',c)
        self.assertNotIn('--with_prior_preservation',c)
        self.assertNotIn('--resume_from_checkpoint',c)

    def test_new_identity_not_historical_rewrite(self):
        old=json.loads((ROOT/'experiments/baseline_split70_s42.json').read_text(encoding='utf-8'))
        new=json.loads((ROOT/'experiments/baseline_stage17a_rebuild_split70_s42.json').read_text(encoding='utf-8'))
        self.assertNotEqual(old['training']['checkpoint_root'],new['training']['checkpoint_root'])
        for k in ('learning_rate','max_train_steps','train_batch_size','gradient_accumulation_steps','text_noise_scale'):
            self.assertEqual(old['training'][k],new['training'][k])
        self.assertFalse(new['generation_authorized'])

    def test_training_entrypoint_syntax_and_optional_audit(self):
        s=(ROOT/'train_dreambooth_noise.py').read_text(encoding='utf-8')
        ast.parse(s)
        self.assertIn('accelerator.optimizer_step_was_skipped',s)
        self.assertIn('--training_audit_path',s)


if __name__=='__main__':unittest.main()
