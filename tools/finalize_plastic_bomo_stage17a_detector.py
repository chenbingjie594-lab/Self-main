"""Apply predeclared headroom gates; never train or evaluate a model."""
import argparse
from pathlib import Path
from plastic_bomo_stage17a_detector_common import ARMS, load, save, headroom

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    names=('realrepeat','vanilla_A','vanilla_B','vanilla_C')
    rows=[load(a.output/(n+'_metrics.json')) for n in names]
    budget=load(a.output/'training_budget_audit.json'); usage=load(a.output/'validation_usage_audit.json')
    if budget['status'] not in ('EQUAL_SUCCESSFUL_COMPUTE_PASS','EQUAL_SCHEDULE_WITH_UPDATE_VARIATION'):
        raise RuntimeError('SCHEDULE_AUDIT_NOT_PASSED')
    if any(usage['evaluations'][arm]['status']!='COMPLETE' or usage['evaluations'][arm]['uses']!=1 for arm in ARMS):
        raise RuntimeError('FINAL_VALIDATION_AUDIT_INCOMPLETE')
    for r in rows:
        if r['checkpoint_sha256']!=budget['arms'][r['arm']]['last_sha256'] or r['seed']!=42:
            raise RuntimeError('FINAL_METRICS_CHECKPOINT_MISMATCH')
    summary=headroom(rows)
    summary.update(full_experiment_complete=True,exact_successful_optimizer_compute_equal=budget['exact_successful_compute_equal'],
                   successful_step_range=budget['successful_step_range'],deep_pcb_training=0,deep_pcb_generation=0,
                   bootstrap_guard_modification=0,bootstrap_guard_selection=0)
    save(a.output/'synthetic_headroom_by_bank.json',summary)
    rr=rows[0]
    save(a.output/'classwise_headroom.json',{'mean_class_deltas_raw':summary['mean_class_deltas_raw'],
         'per_bank':{r['arm']:{c:{'ap50_95_delta_raw':r['per_class'][c]['ap50_95']-rr['per_class'][c]['ap50_95'],
             'recall_delta_raw':r['per_class'][c]['recall']-rr['per_class'][c]['recall']} for c in ('flash','black')} for r in rows[1:]}})
    save(a.output/'vanilla_nuisance_scale.json',{k:summary[k] for k in ('vanilla_std_pp','vanilla_range_pp','future_method_min_gain_pp','std_ddof','detector_seed')})
    save(a.output/'stage17a_status.json',summary)
    text=['# Stage17A Vanilla Synthetic Utility Stability Calibration','',summary['status'],'',
          'Primary: epoch150 last.pt mAP50-95. Detector seed=42; three generation banks, not three detector seeds.','',
          '## Final metrics','', '| Arm | mAP50-95 (%) | Delta vs RR (pp) |','|---|---:|---:|']
    for r in rows: text.append(f"| {r['arm']} | {100*r['map50_95']:.4f} | {100*(r['map50_95']-rr['map50_95']):+.4f} |")
    text += ['',f"Mean gain: {100*summary['mean_delta']:+.4f} pp; sample SD: {100*summary['std_delta']:.4f} pp; wins: {summary['wins_vs_RR']}/3.",
             f"Generation nuisance SD: {summary['vanilla_std_pp']:.4f} pp; range: {summary['vanilla_range_pp']:.4f} pp.",
             f"Future method minimum gain: {summary['future_method_min_gain_pp']:.4f} pp (does not change current gates).",'',
             '## Compute and exposure','',f"Successful optimizer steps exactly equal: {budget['exact_successful_compute_equal']}; range: {budget['successful_step_range']}.",
             'Equal scheduled draws, batches, epochs and optimizer attempts are audited separately from successful AMP updates. A/B/C primary and secondary augmentation fetch sequences are audited.',
             'RR repeats complete donor parent images and complete labels. Extra class-role counts match; exact bbox-instance exposure is not claimed.', '',
             '## Interpretation limits','',
             'Utility is measured only under the current frozen Plastic_Bomo annotation protocol. This is not proof of true physical-defect recognition.',
             '88 Flash + 30 Black boxes have XML lineage; two Black conversions have approximately 1px edge clipping differences. Supplemental Black has 25 images / 50 boxes with historical prelabel import and no instance-level human verification evidence. Suspected omissions and Small/Big Black semantic boundaries remain unchanged.',
             'The generator is the explicitly authorized train-only rebuild, not the unavailable historical checkpoint. Six known related-validation normal sources were excluded in a separately frozen background pool; unknown acquisition relationships are not ruled out.',
             'Stage16 stopped records and historical Stage14–15 conclusions remain unchanged. No new modules, selectors, BootstrapGuard or DeepPCB were used. Stage17B is not automatically started.','']
    (a.output/'STAGE17A_REPORT.md').write_text('\n'.join(text),encoding='utf-8')
    print(__import__('json').dumps(summary,indent=2))

if __name__=='__main__': main()
