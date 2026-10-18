"""D1 only: frozen-image diagnostics; no training, sampling, selection or MSDF changes."""
import argparse
from contextlib import contextmanager
import importlib
import importlib.metadata as md
import json
import math
import os
from pathlib import Path
import sqlite3
import statistics
import sys
import time

import chapter3_exp1_quality_common as c
import evaluate_plastic_bomo_chapter3_exp1 as e
import chapter3_exp1_d1_common as d
import chapter3_exp1_d1_lowrank as fast_fid


def csv_rows(path, rows):
    c.require(bool(rows), 'D1_EMPTY_OUTPUT', str(path))
    header = list(rows[0])
    c.require(all(set(r) == set(header) for r in rows), 'D1_CSV_SCHEMA_CHANGED', str(path))
    c.csv_write(path, header, [[r[k] for k in header] for r in rows])


def state(out, status, write=True, **extra):
    row = {'status': status, 'full_diagnosis_complete': False, **d.ZERO}
    for cls in ('Black', 'Flash'):
        for metric in ('KID', 'FID'):
            row[cls+'_'+metric+'_delta_bootstrap_mean'] = None
            row[cls+'_'+metric+'_delta_CI95'] = None
        for metric in ('nearest_real_LPIPS', 'PSNR', 'SSIM', 'MSE'):
            for method in ('raw', 'msdf'):
                row[cls+'_'+metric+'_'+method] = None
    row['appearance_shift_summary'] = None
    row.update(extra)
    if write: c.save(out / 'd1_status.json', row)
    return row


@contextmanager
def process_lock(out):
    stream = (out / '.diagnosis.lock').open('a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            if stream.tell() == 0:
                stream.write(b'0'); stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise c.Stop('D1_ALREADY_RUNNING', '同一输出目录已有诊断进程；不要同时启动。')
    try:
        yield
    finally:
        if os.name == 'nt':
            stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def prepare(out, cfg, normal_root):
    import numpy as np
    root, pairs, ref = d.source(cfg)
    normals = d.slots_and_normals(root, pairs, normal_root)
    protocol = {**cfg, 'implementation_sha256': d.implementation()}
    if (out / 'diagnostic_protocol.json').exists():
        fast_fid.verify_protocol(out, cfg)
    else:
        d.write_once(out / 'diagnostic_protocol.json', protocol)
    d.write_once(out / 'normal_asset_manifest.json', {'records': sorted(normals.values(), key=lambda x:x['slot_id'])})
    d.write_once(out / 'diagnostic_slot_manifest.json', d.selected_slots(pairs, cfg['seed']))
    d.write_once(out / 'resampling_index_plan.json', {'rounds_per_class': cfg['rounds'],
        'records': d.resampling_plan(np, ref['instance_class_counts'], cfg['rounds'], cfg['seed'])})
    names = ('diagnostic_protocol.json', 'normal_asset_manifest.json', 'diagnostic_slot_manifest.json', 'resampling_index_plan.json')
    d.write_once(out / 'prepared_binding.json', {'status': 'FROZEN_BEFORE_D1_RESULTS',
        'sha256': {n:c.sha(out/n) for n in names}, 'source_artifact_manifest_sha256':cfg['source_artifact_manifest_sha256']})
    fast_fid.bind(out, cfg)
    return root, pairs, ref, normals


def verify_prepared(out, cfg):
    b = c.load(out / 'prepared_binding.json')
    for n,h in b['sha256'].items(): c.checked_file(out/n,h)
    fast_fid.verify_protocol(out, cfg)
    fast_fid.bind(out, cfg)


def stage_done(out, name):
    path = out / ('stage_'+name+'_complete.json')
    if not path.exists(): return False
    for n,h in c.load(path)['sha256'].items(): c.checked_file(c.inside(out,n),h)
    print('[复用] '+name+' 已完成，哈希一致。', flush=True)
    return True


def seal(out, name, files):
    d.write_once(out/('stage_'+name+'_complete.json'), {'sha256':{Path(n).as_posix():c.sha(out/n) for n in files}})


def dependencies(root, phase):
    cfg = c.load(root/'metric_protocol.json')
    old = c.load(root/'metric_runtime.json')
    result = c.dependencies(cfg)
    needed = ('torch','torchvision','torch-fidelity','lpips','numpy','scipy','Pillow','matplotlib')
    # The diagnostic uses the same metric runtime, not a new environment/backbone.
    for n in needed:
        if result['package_versions'].get(n) != old['package_versions'].get(n):
            result['errors'].append({'package':n,'reason':'EXPERIMENT1_VERSION_MISMATCH',
                'expected':old['package_versions'].get(n),'actual':result['package_versions'].get(n)})
    try: result['scikit_image_version'] = md.version('scikit-image')
    except md.PackageNotFoundError:
        result['scikit_image_version'] = None
        if phase in ('all','background'):
            result['errors'].append({'package':'scikit-image','reason':'REQUIRED_FOR_STANDARD_MASKED_BACKGROUND_SSIM'})
    result['status'] = 'DEPENDENCY_REQUIRED' if result['errors'] else 'PASS'
    return result


def metric_runtime(root, out, dep, device):
    # Classical SSIM has its own before-first-use binding. Adding that missing
    # optional dependency must not invalidate already completed FID/KID rounds.
    core_dep={k:v for k,v in dep.items() if k!='scikit_image_version'}
    lib, runtime = e.runtime_modules(c.load(root/'metric_protocol.json'),core_dep,device)
    old = c.load(root/'metric_runtime.json')
    for name,row in old['metric_source_files'].items():
        c.require(runtime['metric_source_files'][name]['sha256']==row['sha256'], 'D1_METRIC_IMPLEMENTATION_CHANGED',name)
    for name,row in old['weights'].items():
        c.require(runtime['weights'][name]['sha256']==row['sha256'], 'D1_METRIC_WEIGHT_CHANGED',name)
    for name in ('normal_asset_manifest.json','diagnostic_protocol.json'):
        runtime[name+'_sha256'] = c.sha(out/name)
    runtime['normal_root_is_relocated_by_filename_and_exact_sha_only'] = True
    d.write_once(out/'diagnostic_runtime.json',runtime)
    return lib,runtime


def bootstrap(out, root, cfg, lib, ref, pairs):
    fast_fid.bootstrap(out, root, cfg, lib, ref, pairs, stage_done, seal, csv_rows)


def appearance(out, root, cfg, pairs, ref):
    if stage_done(out,'appearance'):return
    import numpy as np
    from PIL import Image
    rows=[]
    for r in ref['instances']:
        im=e.open_rgb({'path':r['image_path'],'canvas_wh':r['canvas_wh'],'crop':r['roi_xyxy']},Image)
        rows.append({'method':'Real','class':r['class'],'bank':'reference','slot_id':r['annotation_id'],**d.appearance_values(np,im)})
    for r in pairs['records']:
        for method,prefix in zip(c.METHODS,('raw','msdf')):
            im=e.open_rgb({'path':r[prefix+'_image_path'],'canvas_wh':r['canvas_wh'],'crop':r['roi_xyxy']},Image)
            rows.append({'method':method,'class':r['class'],'bank':r['bank'],'slot_id':r['slot_id'],**d.appearance_values(np,im)})
    csv_rows(out/'appearance_statistics.csv',rows)
    fields=list(d.appearance_values(np,np.zeros((3,3,3),dtype=np.uint8)))
    groups=[]
    for cls in c.CLASSES:
        for method in ('Real',*c.METHODS):
            for bank in (('reference',) if method=='Real' else (*c.BANKS,'pooled_descriptive')):
                rr=[r for r in rows if r['class']==cls and r['method']==method and (bank=='pooled_descriptive' or r['bank']==bank)]
                for f in fields:groups.append({'class':cls,'method':method,'bank':bank,'statistic':f,**d.summary(np,[r[f] for r in rr])})
    deviations={}
    for ci,cls in enumerate(c.CLASSES):
        deviations[cls]={}
        for f in ('gray_mean','RMS_contrast'):
            _,(raw,msdf)=d.by_slot_pair(np,[r for r in rows if r['method']!='Real'],f,cls,cfg)
            real=np.asarray([r[f] for r in rows if r['method']=='Real' and r['class']==cls])
            raw,msdf=np.asarray(raw),np.asarray(msdf)
            gi=np.random.default_rng(np.random.SeedSequence([cfg['seed'],10,ci+1])).integers(0,40,(cfg['rounds'],40))
            ri=np.random.default_rng(np.random.SeedSequence([cfg['seed'],11,ci])).integers(0,len(real),(cfg['rounds'],len(real)))
            diff=np.abs(msdf[gi].mean(1)-real[ri].mean(1))-np.abs(raw[gi].mean(1)-real[ri].mean(1))
            deviations[cls][f]=d.interval(np,diff)
    c.save(out/'appearance_summary.json',{'groups':groups,'paired_deviation_bootstrap':deviations,
        'not_a_realism_or_causal_metric':True,'RMS_contrast_equals_gray_std_div255_not_independent_evidence':True})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for ax,(cls,f) in zip(axes.flat,[(x,y) for x in c.CLASSES for y in ('gray_mean','RMS_contrast','mean_sobel_magnitude')]):
        vals=[[r[f] for r in rows if r['class']==cls and r['method']==m] for m in ('Real',*c.METHODS)]
        ax.boxplot(vals,positions=[1,2,3]);ax.set_xticks([1,2,3],[f'{m}\nn={len(v)}' for m,v in zip(('Real','Raw','MSDF'),vals)])
        ax.set_title(cls+' / '+f)
    fig.suptitle('Frozen ROI appearance statistics (diagnostic, not quality scores)')
    fig.savefig(out/'appearance_distributions.png',dpi=150);plt.close(fig)
    seal(out,'appearance',['appearance_statistics.csv','appearance_summary.json','appearance_distributions.png'])


def geometry(out,pairs,ref):
    if stage_done(out,'geometry'):return
    import numpy as np
    rows=[]
    for method,rr,idkey in (('Real',ref['instances'],'annotation_id'),('Synthetic',[r for r in pairs['records'] if r['bank']=='A'],'slot_id')):
        for r in rr:
            _,_,w,h=r['bbox'];cw,ch=r['canvas_wh']
            rows.append({'population':method,'class':r['class'],'id':r[idkey],
                'bbox_width_ratio':w,'bbox_height_ratio':h,'bbox_area_ratio':w*h,'aspect_ratio':w*cw/(h*ch)})
    result=[]
    for method in ('Real','Synthetic'):
        for cls in c.CLASSES:
            rr=[r for r in rows if r['population']==method and r['class']==cls]
            for f in ('bbox_width_ratio','bbox_height_ratio','bbox_area_ratio','aspect_ratio'):
                result.append({'population':method,'class':cls,'quantity':f,**d.summary(np,[r[f] for r in rr])})
    csv_rows(out/'bbox_geometry_records.csv',rows);csv_rows(out/'bbox_geometry_summary.csv',result)
    seal(out,'geometry',['bbox_geometry_records.csv','bbox_geometry_summary.csv'])


def nearest(out,root,cfg,pairs,ref,lib,runtime):
    if stage_done(out,'nearest'):return
    torch,np,Image=lib['torch'],lib['np'],lib['Image']
    old_hub=torch.hub.get_dir();torch.hub.set_dir(str(Path(runtime['weights']['alexnet']['path']).parent.parent))
    try:
        model=lib['lpips'].LPIPS(net='alex',version='0.1',lpips=True,pretrained=True,pnet_rand=False,
            model_path=runtime['weights']['lpips_linear']['path'],eval_mode=True,verbose=False).eval().requires_grad_(False).to(runtime['device'])
    finally:torch.hub.set_dir(old_hub)
    oldcfg=c.load(root/'metric_protocol.json')
    def tensor(path,canvas,crop):
        im=e.open_rgb({'path':path,'canvas_wh':canvas,'crop':crop},Image).resize(tuple(oldcfg['LPIPS']['resize_wh']),Image.Resampling.BILINEAR)
        return torch.from_numpy(np.asarray(im,dtype=np.float32).copy()/127.5-1).permute(2,0,1)
    real={cls:torch.stack([tensor(r['image_path'],r['canvas_wh'],r['roi_xyxy']) for r in ref['instances'] if r['class']==cls]) for cls in c.CLASSES}
    rows=[];forwards=0
    with torch.inference_mode(),e.heartbeat('nearest-real LPIPS；全部480生成ROI对同类全部真实ROI'):
        for i,r in enumerate(pairs['records']):
            for method,prefix in zip(c.METHODS,('raw','msdf')):
                a=tensor(r[prefix+'_image_path'],r['canvas_wh'],r['roi_xyxy'])[None].to(runtime['device']);dist=[]
                rr=real[r['class']]
                for start in range(0,len(rr),cfg['nearest_real_LPIPS']['batch_size']):
                    b=rr[start:start+cfg['nearest_real_LPIPS']['batch_size']].to(runtime['device'])
                    dist.extend(model(a.expand(len(b),-1,-1,-1),b,normalize=False).reshape(-1).cpu().tolist());forwards+=len(b)
                c.require(len(dist)==len(rr) and all(math.isfinite(x) for x in dist),'D1_NONFINITE_LPIPS',r['slot_id'])
                v=sorted(dist)
                rows.append({'method':method,'class':r['class'],'bank':r['bank'],'slot_id':r['slot_id'],
                    'real_reference_count':len(rr),'minimum_LPIPS':v[0],'top3_mean_LPIPS':statistics.mean(v[:3])})
            if (i+1)%10==0:print(f'[nearest LPIPS] {2*(i+1)}/480 ROIs',flush=True)
    del model
    if runtime['device'].startswith('cuda:'):torch.cuda.empty_cache()
    csv_rows(out/'nearest_real_lpips.csv',rows)
    result={cls:{k:d.by_slot_pair(np,rows,k,cls,cfg)[0] for k in ('minimum_LPIPS','top3_mean_LPIPS')} for cls in ('overall',*c.CLASSES)}
    result['pair_forward_count']=forwards;result['interpretation']='nearest-reference diagnostic,not formal realism metric'
    c.save(out/'nearest_real_lpips_summary.json',result)
    seal(out,'nearest',['nearest_real_lpips.csv','nearest_real_lpips_summary.json'])


def background(out,cfg,pairs,normals,normal_root):
    if stage_done(out,'background'):return
    import numpy as np
    from PIL import Image
    from skimage import metrics
    module=importlib.import_module('skimage.metrics._structural_similarity')
    d.write_once(out/'background_runtime.json',{'scikit_image':md.version('scikit-image'),
        'SSIM_source_sha256':c.sha(module.__file__),'protocol_sha256':c.sha(out/'diagnostic_protocol.json')})
    rows=[]
    for i,sid in enumerate(sorted(normals)):
        asset=normals[sid];g=asset['geometry'];box=g['paste_valid_crop_region_xyxy']
        with Image.open(Path(normal_root)/asset['normal_filename']) as im:n=im.convert('RGB').crop(box)
        for r in [x for x in pairs['records'] if x['slot_id']==sid]:
            for method,prefix in zip(c.METHODS,('raw','msdf')):
                patch=e.open_rgb({'path':r[prefix+'_image_path'],'canvas_wh':r['canvas_wh'],'crop':box},Image)
                rows.append({'method':method,'class':r['class'],'bank':r['bank'],'slot_id':sid,**d.background_values(np,n,patch,g,metrics)})
        if (i+1)%10==0:print(f'[background P-minus-B] {i+1}/80 slots',flush=True)
    for k in ('PSNR','SSIM','MSE'):
        csv_rows(out/('background_'+k.lower()+'.csv'),[{n:r[n] for n in ('method','class','bank','slot_id',k,'P_minus_B_pixels','SSIM_supported_centres')} for r in rows])
    result={cls:{k:d.by_slot_pair(np,rows,k,cls,cfg)[0] for k in ('PSNR','SSIM','MSE')} for cls in ('overall',*c.CLASSES)}
    c.save(out/'background_summary.json',result)
    seal(out,'background',['background_psnr.csv','background_ssim.csv','background_mse.csv','background_summary.json','background_runtime.json'])


def visualize(out,normal_root,normals):
    if stage_done(out,'visualize'):return
    import numpy as np
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files=[]
    for i,r in enumerate(c.load(out/'diagnostic_slot_manifest.json')['records']):
        a=normals[r['slot_id']];box=a['geometry']['paste_valid_crop_region_xyxy']
        with Image.open(Path(normal_root)/a['normal_filename']) as im:n=im.convert('RGB')
        raw=e.open_rgb({'path':r['raw_image_path'],'canvas_wh':r['canvas_wh'],'crop':None},Image)
        msdf=e.open_rgb({'path':r['msdf_image_path'],'canvas_wh':r['canvas_wh'],'crop':None},Image)
        fig,ax=plt.subplots(2,4,figsize=(12,6),constrained_layout=True)
        for col,(name,im) in enumerate((('Normal',n),('Real reference',None),('Raw SD2',raw),('MSDF',msdf))):
            for row,crop in enumerate((box,r['roi_xyxy'])):
                ax[row,col].axis('off');ax[row,col].set_title(name+(' / P' if row==0 else ' / frozen ROI'))
                if im is None:ax[row,col].text(.5,.5,'N/A: no direct held-out\nreference role',ha='center',va='center')
                else:ax[row,col].imshow(im.crop(crop))
        fig.suptitle(f'{r["class"]} / bank {r["bank"]} / random group {i+1}; no quality ranking')
        path=out/'diagnostic_grids'/f'{i+1:02d}_{r["class"]}_{r["bank"]}.png';path.parent.mkdir(exist_ok=True)
        fig.savefig(path,dpi=130);plt.close(fig);files.append(path.relative_to(out))
        n=np.asarray(n.crop(box),dtype=np.float64)
        fig,ax=plt.subplots(1,2,figsize=(8,4),constrained_layout=True)
        for axis,name,im in zip(ax,('Raw SD2','MSDF'),(raw,msdf)):
            diff=np.abs(np.asarray(im.crop(box),dtype=np.float64)-n).mean(2)
            heat=axis.imshow(diff,cmap='magma',vmin=0,vmax=255);axis.axis('off');axis.set_title('|'+name+' - Normal| in P')
        fig.colorbar(heat,ax=ax,label='mean absolute RGB difference, shared 0..255 scale')
        path=out/'residual_visualizations'/f'{i+1:02d}_{r["class"]}_{r["bank"]}.png';path.parent.mkdir(exist_ok=True)
        fig.savefig(path,dpi=130);plt.close(fig);files.append(path.relative_to(out))
    c.require(len(files)==48,'D1_VISUALIZATION_INCOMPLETE',len(files));seal(out,'visualize',files)


def finalize(out,cfg):
    for stage in ('bootstrap','appearance','geometry','nearest','background','visualize'):
        c.require(stage_done(out,stage),'D1_DIAGNOSTICS_INCOMPLETE',stage)
    boot=c.load(out/'bootstrap_summary.json');app=c.load(out/'appearance_summary.json')
    nr=c.load(out/'nearest_real_lpips_summary.json');bg=c.load(out/'background_summary.json')
    label,evidence=d.conclusion(boot,app,nr,bg)
    c.require(label in cfg['allowed_final_statuses'],'D1_CONCLUSION_INVALID',label)
    row=state(out,label,write=False,full_diagnosis_complete=True,appearance_shift_summary=app['paired_deviation_bootstrap'],evidence=evidence)
    for cls,title in (('flash','Flash'),('black','Black')):
        for metric in ('FID','KID'):
            row[title+'_'+metric+'_delta_bootstrap_mean']=boot[cls]['delta_'+metric]['mean']
            row[title+'_'+metric+'_delta_CI95']=boot[cls]['delta_'+metric]['CI95']
        for metric,summary in (('nearest_real_LPIPS',nr[cls]['minimum_LPIPS']),*[(k,bg[cls][k]) for k in ('PSNR','SSIM','MSE')]):
            for method,suffix in zip(c.METHODS,('raw','msdf')):row[title+'_'+metric+'_'+suffix]=summary[method]['mean']
    c.save(out/'flash_vs_black_summary.json',{'status':label,'evidence':evidence,'causal_mechanism_identified':False,
        'geometry_cannot_explain_paired_method_gap_by_itself':True,'LPIPS_diversity_is_not_realism':True,**d.ZERO})
    lines=['# Experiment1-D1 Black Failure Diagnosis','','Status: `'+label+'`.','','## Paired size27 resampling','',
        'Seed2026,1000 rounds per class, replacement sampling, fixed balanced A/B/C schedule, shared Raw/MSDF indices. '
        'Original cached2048-D features and unchanged torch-fidelity KID. FID uses the algebraically exact low-rank cross-Gram nuclear norm with unbiased covariance (ddof1), only after every comparison with the original standard2048-D implementation passes abs_error<=1e-5+1e-6*abs(standard_FID), explicitly authorized after the archived strict-only failure. '
        'All1000 rounds per class are recomputed in a separate lowrank-only cache; legacy slow rows are audit references only. Intervals are conditional empirical bootstrap intervals, not causal tests or uncertainty across retrained models.','']
    for cls in c.CLASSES:
        for metric in ('delta_KID','delta_FID'):
            x=boot[cls][metric];lines.append(f'- {cls} {metric}: mean={x["mean"]:.6g}, median={x["median"]:.6g}, 95% interval={x["CI95"]}, P(delta>0)={x["P_gt_0"]:.4f}.')
    lines+=['','## Observable shifts','','```json',json.dumps(evidence,indent=2),'```','',
        'All Flash diagnostics use the same protocol as Black. A shift shared by Flash is not Black-specific. '
        'Appearance statistics and nearest-reference LPIPS are descriptive diagnostics, not new realism scores. '
        'Greater cross-bank LPIPS variation must not be interpreted as better diversity or as causing degradation.','',
        'Background PSNR/MSE use all RGB pixels in P-minus-B. SSIM uses only11x11 windows fully contained in that region; '
        'canvas-minus-P copied pixels never enter the summaries. Infinity PSNR is retained and unsupported paired summaries are unavailable, not capped or filtered.','',
        'Raw and MSDF have identical condition bboxes; bbox geometry cannot explain the paired gap by itself. '
        'No direct held-out reference role exists for diagnostic grid slots; N/A panels are intentional, not missing selected images. '
        'All24 fixed-random matched groups and their residual plots are retained; residual scale is globally0..255.','',
        'No checkpoint, seed, crop, image selection, method, loss or attention module was changed. No training, generation, detector, BootstrapGuard or DWBG ran. '
        'A visual association is not proof of the MSDF causal mechanism. If evidence is not coherent, mechanism remains unresolved.','',
        '## Sources','',
        '- Experiment1 frozen input/metric protocol, runtime and feature cache (source commit '+cfg['source_commit']+').',
        '- https://github.com/toshas/torch-fidelity/blob/v0.3.0/torch_fidelity/metric_fid.py',
        '- https://github.com/toshas/torch-fidelity/blob/v0.3.0/torch_fidelity/metric_kid.py',
        '- https://github.com/richzhang/PerceptualSimilarity',
        '- https://scikit-image.org/docs/0.25.x/api/skimage.metrics.html','',
        'Diagnosis complete. No follow-up experiment is authorized automatically. Git publication is performed separately after review.','']
    (out/'D1_REPORT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    c.save(out/'d1_status.json',row)
    files={p.relative_to(out).as_posix():c.sha(p) for p in out.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.md','.png') and p.name!='d1_artifact_manifest.json'}
    c.save(out/'d1_artifact_manifest.json',{'status':'COMPLETE','sha256':files,'bootstrap_sqlite_is_resume_cache_not_scientific_output':True})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('prepare','bootstrap','appearance','geometry','nearest','background','visualize','finalize','all'),default='all')
    p.add_argument('--normal_root',type=Path,default=Path('/mnt/sda/cbj/dataset/Plastic_Bomo/empty_images_v2_p0'))
    p.add_argument('--output',type=Path,default=d.OUTPUT)
    p.add_argument('--device',default='0');a=p.parse_args();out=a.output.resolve()
    c.require(out==d.OUTPUT or out.is_relative_to(d.OUTPUT),'D1_OUTPUT_INVALID','only new D1 output tree is writable')
    out.mkdir(parents=True,exist_ok=True)
    try:
        with process_lock(out):
            cfg=c.load(d.CONFIG)
            print(f'[启动] D1 phase={a.phase} PID={os.getpid()}；只诊断、不训练或生成。',flush=True)
            root,pairs,ref,normals=prepare(out,cfg,a.normal_root);verify_prepared(out,cfg)
            if (out/'d1_artifact_manifest.json').exists() and (out/'d1_status.json').exists() and c.load(out/'d1_status.json').get('full_diagnosis_complete'):
                for n,h in c.load(out/'d1_artifact_manifest.json')['sha256'].items():c.checked_file(c.inside(out,n),h)
                print('[已退出] 现有D1完整，未覆盖、未重算。退出码=0。',flush=True);return
            if a.phase=='prepare':
                state(out,'D1_PREPARED_DIAGNOSTICS_NOT_RUN')
                (out/'D1_REPORT.md').write_text('# D1 preparation only\n\nNo D1 result or mechanism conclusion yet.\n',encoding='utf-8')
                print('[已退出] 配对、normal哈希、1000轮抽样和24组可视化清单已冻结；诊断未运行。退出码=0。',flush=True);return
            if a.phase in ('appearance','geometry','visualize'):
                # These frozen-image diagnostics need no torch or metric model.
                if a.phase=='appearance':appearance(out,root,cfg,pairs,ref)
                if a.phase=='geometry':geometry(out,pairs,ref)
                if a.phase=='visualize':visualize(out,a.normal_root,normals)
                state(out,'D1_PARTIAL_DIAGNOSTICS_COMPLETE',completed_phase=a.phase)
                print('[已退出] '+a.phase+' 已完成；D1仍不完整，没有最终机制结论。退出码=0。',flush=True);return
            dep=dependencies(root,a.phase);c.save(out/'dependency_report.json',dep)
            c.require(dep['status']=='PASS','DEPENDENCY_REQUIRED',dep['errors'])
            state(out,'D1_DIAGNOSTICS_RUNNING',phase=a.phase)
            with e.offline_metrics():
                lib,runtime=metric_runtime(root,out,dep,'cuda:'+a.device if a.device.isdecimal() else a.device)
                if a.phase in ('all','bootstrap'):bootstrap(out,root,cfg,lib,ref,pairs)
                if a.phase in ('all','appearance'):appearance(out,root,cfg,pairs,ref)
                if a.phase in ('all','geometry'):geometry(out,pairs,ref)
                if a.phase in ('all','nearest'):nearest(out,root,cfg,pairs,ref,lib,runtime)
                if a.phase in ('all','background'):background(out,cfg,pairs,normals,a.normal_root)
                if a.phase in ('all','visualize'):visualize(out,a.normal_root,normals)
                if a.phase in ('all','finalize'):
                    d.source(cfg);d.slots_and_normals(root,pairs,a.normal_root);verify_prepared(out,cfg);finalize(out,cfg)
            print('[已退出] D1阶段完成，退出码=0；未启动后续实验。',flush=True)
    except Exception as error:
        fallback='DEPENDENCY_REQUIRED' if isinstance(error,ImportError) else 'D1_DIAGNOSTIC_FAILED'
        label=getattr(error,'status',fallback);detail=getattr(error,'detail',str(error))
        if label!='D1_ALREADY_RUNNING' and not ((out/'d1_artifact_manifest.json').exists() and (out/'d1_status.json').exists() and c.load(out/'d1_status.json').get('full_diagnosis_complete')):
            state(out,label,detail=str(detail))
            (out/'D1_REPORT.md').write_text('# D1 incomplete\n\nStatus: `'+label+'`.\n\n'+str(detail)+'\n\nNo final diagnostic or causal conclusion. Completed stages/cached rounds are retained.\n',encoding='utf-8')
        print('[已退出] '+label+'：'+str(detail)+'；退出码='+('2' if label=='DEPENDENCY_REQUIRED' else '1'),flush=True)
        raise SystemExit(2 if label=='DEPENDENCY_REQUIRED' else 1)


if __name__=='__main__':main()
