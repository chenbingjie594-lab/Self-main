"""Train RR/VA/VB/VC with one seed; evaluate last.pt only after all four finish."""
import argparse
from collections import Counter
import csv
from pathlib import Path
import shutil
from plastic_bomo_stage17a_detector_common import (ARMS, load, save, sha, digest, verify_prepared,
    EpochSampler, TraceDatasetMixin, schedule, function_with_global)

def metric(result):
    b=result.box; ids={int(cid):i for i,cid in enumerate(b.ap_class_index)}
    if set(ids)!={0,1}: raise RuntimeError('FINAL_CLASS_METRICS_MISSING')
    return {'precision':float(b.mp),'recall':float(b.mr),'map50':float(b.map50),'map50_95':float(b.map),
            'per_class':{c:{'ap50':float(b.ap50[ids[i]]),'ap50_95':float(b.ap[ids[i]]),'recall':float(b.r[ids[i]])}
                         for i,c in enumerate(('flash','black'))},
            'raw_results_dict':{k:float(v) for k,v in result.results_dict.items()}}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('prepared','model','runs','output'): p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--device',default='0'); p.add_argument('--preflight_only',action='store_true')
    a=p.parse_args(); a.runs=a.runs.resolve(); a.output=a.output.resolve()
    cfg,mf=verify_prepared(a.prepared); det=cfg['detector']; n=mf['draws_per_epoch']
    if a.model.resolve()!=(Path(__file__).resolve().parents[1]/det['model']).resolve():
        raise RuntimeError('FROZEN_YOLO11S_PRETRAINED_PATH_REQUIRED')
    a.output.mkdir(parents=True,exist_ok=True)
    import torch,ultralytics,yaml
    from torch.utils.data import DataLoader
    from ultralytics import YOLO
    from ultralytics.data.build import seed_worker
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.data.dataset import YOLODataset
    # Defined at module scope for worker pickling; no numerical/model-path change.
    global TracedYOLODataset
    TracedYOLODataset=type('TracedYOLODataset',(TraceDatasetMixin,YOLODataset),{'__module__':__name__})
    # Call the installed native builder with a private constructor binding.
    # Do not mutate an existing object's __class__ or patch package globals.
    native_build=DetectionTrainer.build_dataset
    native_factory=native_build.__globals__.get('build_yolo_dataset')
    traced_factory=function_with_global(native_factory,'YOLODataset',YOLODataset,TracedYOLODataset)
    traced_build=function_with_global(native_build,'build_yolo_dataset',native_factory,traced_factory)
    class Loader(DataLoader):
        def reset(self): pass  # Fresh iterator/workers every epoch, no cross-epoch prefetch.
    initial={'protocol_content_sha256':digest(cfg),'prepared_manifest_sha256':sha(a.prepared/'detector_dataset_manifest.json'),
             'pretrained_sha256':sha(a.model),'torch':torch.__version__,'ultralytics':ultralytics.__version__,
             'training_script_sha256':sha(__file__),'common_script_sha256':sha(Path(__file__).with_name('plastic_bomo_stage17a_detector_common.py'))}
    initfile=a.output/'training_initialization_audit.json'
    if initfile.exists() and load(initfile)!=initial: raise RuntimeError('TRAINING_INITIALIZATION_CHANGED')
    save(initfile,initial); save(a.output/'stage17a_protocol.json',cfg)
    if a.preflight_only:
        # Exercise real transforms, collate and tracing on CPU before any optimizer is created.
        from ultralytics.cfg import get_cfg
        args=get_cfg(overrides={k:det[k] for k in ('imgsz','batch','mosaic','mixup','copy_paste','close_mosaic','rect')})
        summaries={}
        for arm in ARMS:
            ds=traced_factory(args,str(Path(mf['dataset_root'])/arm/'images/train'),1,
                  {'names':{0:'flash',1:'black'},'nc':2},mode='train',rect=False,stride=32)
            result=ds.collate_fn([ds[0]])
            trace=result.pop('stage17_fetches')
            if len(trace)!=1 or not trace[0]: raise RuntimeError('AUGMENTATION_TRACE_COLLATE_UNSUPPORTED')
            summaries[arm]={'dataset_size':len(ds),'probe_fetch_count':len(trace[0]),'trace':trace}
            if len(ds)!=n: raise RuntimeError('PREFLIGHT_DATASET_SIZE_CHANGED')
        save(a.output/'detector_runtime_preflight.json',{'status':'CPU_TRANSFORM_COLLATE_PASS','arms':summaries,
             'optimizer_steps':0,'official_validation_forwards':0})
        print('CPU_TRANSFORM_COLLATE_PASS'); return
    preflight=a.output/'detector_runtime_preflight.json'
    if not preflight.exists() or load(preflight)['status']!='CPU_TRANSFORM_COLLATE_PASS':
        raise RuntimeError('RUN_CPU_PREFLIGHT_FIRST')
    budgets={}; environments={}; order={}
    for arm in ARMS:
        if sha(a.model)!=initial['pretrained_sha256']: raise RuntimeError('PRETRAINED_WEIGHTS_CHANGED')
        run=a.runs/(arm+'_s42'); ledger=a.output/f'epoch_exposure_{arm}.json'
        last=run/'weights/last.pt'; argsfile=run/'args.yaml'; csvfile=run/'results.csv'
        state={'epochs':[],'optimizer_step_attempts':0,'successful_optimizer_steps':0,'suppressed_validation_calls':0}
        traces=a.output/'order_records'/arm; traces.mkdir(parents=True,exist_ok=True)
        names=[r['position'] for r in mf['positions'][arm]]
        class Trainer(DetectionTrainer):
            def build_dataset(self,*args,**kwargs):
                return traced_build(self,*args,**kwargs)
            def get_dataloader(self,path,batch_size=16,rank=0,mode='train'):
                if mode!='train': return super().get_dataloader(path,batch_size,rank,mode)
                if rank not in (-1,0): raise RuntimeError('SINGLE_GPU_ONLY')
                ds=self.build_dataset(path,mode,batch_size)
                if not isinstance(ds,TracedYOLODataset): raise RuntimeError('UNSUPPORTED_TRACED_DATASET_API')
                actual=[Path(x).stem for x in ds.im_files]
                if Counter(actual)!=Counter(names): raise RuntimeError('TRAINING_POSITIONS_CHANGED')
                state['sampler']=EpochSampler(actual)
                return Loader(ds,batch_size=1,sampler=state['sampler'],num_workers=self.args.workers,
                    collate_fn=ds.collate_fn,pin_memory=True,worker_init_fn=seed_worker,
                    generator=torch.Generator().manual_seed(6148914691236517205+42))
            def preprocess_batch(self,batch):
                fetches=batch.pop('stage17_fetches',None)
                if fetches is None: raise RuntimeError('AUGMENTATION_TRACE_MISSING')
                state['primary'].extend(Path(f).stem for f in batch['im_file'])
                state['augmentation'].extend([list(x) for x in fetches]); state['batches']+=1
                return super().preprocess_batch(batch)
            def validate(self):
                state['suppressed_validation_calls']+=1; return {},0.0
            def final_eval(self): return None
        def start(t):
            state['actual_optimizer']={'class':type(t.optimizer).__name__,
                'defaults':{k:v for k,v in t.optimizer.defaults.items() if isinstance(v,(str,int,float,bool,list,tuple,type(None)))}}
            original=t.optimizer.step; attempt=t.optimizer_step
            def step(*args,**kwargs):
                value=original(*args,**kwargs); state['successful_optimizer_steps']+=1; return value
            def attempted(*args,**kwargs):
                state['optimizer_step_attempts']+=1; return attempt(*args,**kwargs)
            t.optimizer.step=step; t.optimizer_step=attempted
        def es(t):
            state['sampler'].set_epoch(t.epoch)
            state.update(primary=[],augmentation=[],batches=0)
        def ee(t):
            expected=schedule(names,t.epoch)
            if state['primary']!=expected or state['batches']!=n:
                save(a.output/f'order_failure_{arm}.json',{'epoch':t.epoch+1,'expected':expected,'actual':state['primary']})
                raise RuntimeError('PRIMARY_EXPOSURE_SEQUENCE_MISMATCH')
            if any(not fetch or fetch[0]!=primary for fetch,primary in zip(state['augmentation'],state['primary'])):
                raise RuntimeError('PRIMARY_AUGMENTATION_BINDING_FAILED')
            save(traces/f'epoch_{t.epoch+1:03d}.json',{'primary_order':state['primary'],'all_image_fetches_per_primary':state['augmentation']})
            state['epochs'].append({'epoch':t.epoch+1,'draws':len(state['primary']),'base_draws':138,
                  'extra_draws':n-138,'batches':state['batches'],'primary_order_sha256':digest(state['primary']),
                  'augmentation_fetches_sha256':digest(state['augmentation']),
                  'augmentation_total_image_fetches':sum(map(len,state['augmentation'])),
                  'learning_rates':[float(g['lr']) for g in t.optimizer.param_groups],
                  'cumulative_attempted_steps':state['optimizer_step_attempts'],
                  'cumulative_successful_steps':state['successful_optimizer_steps']})
            save(ledger,{k:state[k] for k in ('epochs','optimizer_step_attempts','successful_optimizer_steps','suppressed_validation_calls','actual_optimizer')})
        if run.exists():
            if not last.exists() or not ledger.exists() or len(load(ledger)['epochs'])!=150:
                raise RuntimeError('INCOMPLETE_RUN_PRESERVED_USE_NEW_RUN_AND_OUTPUT_ROOT:'+str(run))
        else:
            model=YOLO(str(a.model.resolve()))
            model.add_callback('on_train_start',start); model.add_callback('on_train_epoch_start',es); model.add_callback('on_train_epoch_end',ee)
            model.train(data=str(Path(mf['dataset_root'])/arm/'data.yaml'),imgsz=1536,epochs=150,patience=151,batch=1,
                  seed=42,optimizer='auto',mosaic=1.,mixup=0.,copy_paste=0.,close_mosaic=10,deterministic=True,
                  rect=False,augment=False,val=False,device=a.device,project=str(a.runs),name=run.name,exist_ok=False,trainer=Trainer)
            del model; torch.cuda.empty_cache()
        state=load(ledger)
        if len(state['epochs'])!=150 or any(r['draws']!=n or r['batches']!=n for r in state['epochs']): raise RuntimeError('FIXED_BUDGET_FAILED')
        with csvfile.open(encoding='utf-8') as f: csvrows=list(csv.DictReader(f))
        if len(csvrows)!=150: raise RuntimeError('TRAINING_CSV_EPOCH_MISMATCH')
        args=yaml.safe_load(argsfile.read_text())
        expected={k:det[k] for k in ('imgsz','epochs','patience','batch','seed','optimizer','mosaic','mixup','copy_paste','close_mosaic','deterministic','rect','augment')}
        expected['val']=False
        if any(args.get(k)!=v for k,v in expected.items()): raise RuntimeError('TRAINING_ARGS_MISMATCH')
        shutil.copyfile(argsfile,a.output/f'{arm}_args.yaml'); shutil.copyfile(csvfile,a.output/f'{arm}_training_results.csv')
        budgets[arm]={'epochs':150,'primary_draws':sum(r['draws'] for r in state['epochs']),
              'base_draws':150*138,'extra_draws':150*(n-138),'batches':150*n,
              'optimizer_step_attempts':state['optimizer_step_attempts'],'successful_optimizer_steps':state['successful_optimizer_steps'],
              'amp_skipped_steps':state['optimizer_step_attempts']-state['successful_optimizer_steps'],'last_sha256':sha(last),
              'actual_optimizer':state['actual_optimizer'],
              'epoch_lr_schedule_sha256':digest([r['learning_rates'] for r in state['epochs']]),
              'suppressed_validation_calls':state['suppressed_validation_calls']}
        order[arm]={'primary':digest([r['primary_order_sha256'] for r in state['epochs']]),
                    'augmentation':digest([r['augmentation_fetches_sha256'] for r in state['epochs']])}
        environments[arm]=args
        save(a.output/'training_budget_audit.json',{'status':'TRAINING_IN_PROGRESS','arms':budgets})
        print(arm+' 150 epochs complete; official evaluation deferred',flush=True)
    verify_prepared(a.prepared)
    equal={k:len({b[k] for b in budgets.values()})==1 for k in ('epochs','primary_draws','base_draws','extra_draws','batches','optimizer_step_attempts','successful_optimizer_steps')}
    equal['primary_order']=len({r['primary'] for r in order.values()})==1
    equal['ABC_augmentation_fetches']=len({order[a]['augmentation'] for a in ('VA','VB','VC')})==1
    equal['actual_optimizer']=len({digest(b['actual_optimizer']) for b in budgets.values()})==1
    equal['epoch_lr_schedule']=len({b['epoch_lr_schedule_sha256'] for b in budgets.values()})==1
    ignored={'data','project','name','save_dir'}
    equal['hyperparameters']=len({digest({k:v for k,v in args.items() if k not in ignored}) for args in environments.values()})==1
    required=[k for k in equal if k!='successful_optimizer_steps']
    ok=all(equal[k] for k in required)
    save(a.output/'training_budget_audit.json',{'status':'EQUAL_SUCCESSFUL_COMPUTE_PASS' if ok and equal['successful_optimizer_steps'] else 'EQUAL_SCHEDULE_WITH_UPDATE_VARIATION' if ok else 'BUDGET_OR_ORDER_FAILED',
         'arms':budgets,'equal':equal,'exact_successful_compute_equal':equal['successful_optimizer_steps'],
         'required_schedule_fields':required,'successful_step_range':max(b['successful_optimizer_steps'] for b in budgets.values())-min(b['successful_optimizer_steps'] for b in budgets.values())})
    save(a.output/'training_order_audit.json',{'arms':order,'equal_primary_order':equal['primary_order'],
         'equal_ABC_augmentation_fetches':equal['ABC_augmentation_fetches'],'raw_records':'order_records/<arm>/epoch_*.json'})
    save(a.output/'training_environment_audit.json',{'initialization':initial,'actual_args':environments})
    if not ok: raise RuntimeError('SCHEDULE_OR_ORDER_GATE_FAILED_NO_OFFICIAL_EVALUATION')
    usagefile=a.output/'validation_usage_audit.json'
    usage=load(usagefile) if usagefile.exists() else {'training_forwards':0,'selection_uses':0,'checkpoint_selection_uses':0,'evaluations':{}}
    for arm in ARMS:
        target=a.output/({'RR':'realrepeat','VA':'vanilla_A','VB':'vanilla_B','VC':'vanilla_C'}[arm]+'_metrics.json')
        last=a.runs/(arm+'_s42')/'weights/last.pt'
        if target.exists():
            if load(target)['checkpoint_sha256']!=sha(last): raise RuntimeError('CACHED_METRICS_CHECKPOINT_CHANGED')
            continue
        if arm in usage['evaluations']: raise RuntimeError('FINAL_EVALUATION_ALREADY_ATTEMPTED_NO_AUTOMATIC_REPEAT:'+arm)
        usage['evaluations'][arm]={'status':'STARTED','checkpoint_sha256':sha(last),'uses':1}
        save(usagefile,usage)
        result=YOLO(str(last)).val(data=mf['official_data_yaml'],split='val',imgsz=1536,batch=1,device=a.device,
            verbose=False,plots=False,project=str(a.runs/'final_validation'),name=arm,exist_ok=False)
        save(target,{'arm':arm,'seed':42,'checkpoint':'epoch150 last.pt','checkpoint_sha256':sha(last),**metric(result)})
        usage['evaluations'][arm]['status']='COMPLETE'; save(usagefile,usage)
    verify_prepared(a.prepared)
    save(a.output/'detector_run_status.json',{'status':'FOUR_ARMS_FINAL_LAST_METRICS_COMPLETE','stage17b_auto_start':False})
    print('FOUR_ARMS_FINAL_LAST_METRICS_COMPLETE',flush=True)

if __name__=='__main__': main()
