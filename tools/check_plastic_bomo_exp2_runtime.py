"""Serial, offline baseline gradient probes; no optimizer steps or sampling."""
import argparse
from contextlib import contextmanager
import gc
import faulthandler
import importlib
import importlib.metadata
import os
from pathlib import Path
import signal
import sys
import threading
import time

import chapter3_exp1_quality_common as c
import chapter3_exp2_baseline_common as b
from exp2_v2_data import file_sha, load_training_records, within, save_json


if not hasattr(Path, 'is_relative_to'):
    # Preserve the exact containment check on the native Python3.8 interpreter;
    # do not edit the frozen Experiment1 common source or metric functions.
    def inside_compat(root, name):
        root = Path(root).resolve()
        path = (root/name).resolve()
        c.require(not Path(name).is_absolute() and within(path,root), 'INPUT_PATH_INVALID', str(name))
        return path
    c.inside = inside_compat


@contextmanager
def heartbeat(method):
    done = threading.Event()
    def report():
        while not done.wait(15):
            print(f'[运行中] {method} 模型加载/梯度验证仍在执行；没有optimizer更新或采样。', flush=True)
    worker = threading.Thread(target=report, daemon=True)
    worker.start()
    try:
        yield
    finally:
        done.set()
        worker.join(timeout=1)


@contextmanager
def stage(name):
    started = time.monotonic()
    print(f'[阶段开始] {name}', flush=True)
    try:
        yield
    except BaseException:
        elapsed = time.monotonic() - started
        print(f'[阶段异常] {name}；耗时={elapsed:.1f}s', flush=True)
        raise
    else:
        elapsed = time.monotonic() - started
        print(f'[阶段完成] {name}；耗时={elapsed:.1f}s', flush=True)


def verify_repaired_source(method, root):
    import hashlib
    manifest = c.load(c.REPO/'configs/plastic_bomo_exp2_source_repairs.json')
    records = []
    for name, expected in manifest['patched_source_sha256_LF'][method].items():
        path = c.inside(root, name)
        c.require(path.is_file(), 'REPAIRED_SOURCE_FILE_MISSING', str(path))
        actual = hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        c.require(actual == expected, 'REPAIRED_SOURCE_FILE_CHANGED', dict(file=str(path), expected=expected, actual=actual))
        records.append(dict(file=name, sha256_LF=actual))
    return records


@contextmanager
def single_gpu_lock():
    path = c.REPO / 'logs/exp2_baseline_runtime_gpu.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f'GPU_LOCK_EXISTS: {path}; check the recorded PID before removing a stale lock')
    with os.fdopen(handle, 'w') as stream:
        stream.write(str(os.getpid()))
    try:
        yield
    finally:
        if path.read_text() == str(os.getpid()):
            path.unlink()


def df_probe(args, torch, rows, device):
    import inspect
    import diffusers
    distribution = Path(importlib.metadata.distribution('diffusers').locate_file('')).resolve()
    pipeline_file = Path(inspect.getfile(diffusers.StableDiffusionInpaintPipeline)).resolve()
    c.require(within(pipeline_file, distribution) and not within(pipeline_file, c.REPO),
              'BASELINE_DIFFUSERS_SHADOWED', str(pipeline_file))
    from model import DefectFillModel
    from PIL import Image
    import numpy as np
    import torch.nn.functional as F
    model = DefectFillModel(device=str(device), base_model=str(args.base_model), local_files_only=True, enable_lfs=False)
    params = [p for component in (model.pipeline.unet, model.pipeline.text_encoder) for p in component.parameters() if p.requires_grad]
    model.pipeline.unet.train()
    reports = []
    for cls in ('flash', 'black'):
        row = next(r for r in rows if r['class'] == cls)
        image = torch.from_numpy(np.asarray(Image.open(row['image_path']).convert('RGB'), dtype=np.float32).copy())
        image = (image.permute(2,0,1)[None]/127.5-1).to(device, dtype=torch.float16)
        mask = torch.from_numpy((np.asarray(Image.open(row['mask_path']).convert('L')).copy() >= 128).astype(np.float32))[None,None].to(device, dtype=torch.float16)
        for p in params:
            p.grad = None
        text = model.get_text_embeddings('A photo of '+model.placeholder_token)
        with torch.no_grad():
            latent = model.pipeline.vae.encode(image).latent_dist.sample() * model.pipeline.vae.config.scaling_factor
            bg = model.pipeline.vae.encode(image*(1-mask)).latent_dist.sample() * model.pipeline.vae.config.scaling_factor
        noise = torch.randn_like(latent)
        t = torch.tensor([500], device=device, dtype=torch.long)
        native_mask = F.interpolate(mask, size=latent.shape[-2:], mode='nearest')
        noisy = model.scheduler.add_noise(latent, noise, t)
        result = model(noisy, bg, native_mask, t, text)
        attention = result['attention_loss']
        gradients = torch.autograd.grad(attention, params, allow_unused=True, retain_graph=True)
        nonzero = any(g is not None and bool(torch.isfinite(g).all()) and float(g.float().norm()) > 0 for g in gradients)
        c.require(nonzero, 'ATTENTION_GRADIENT_PROBE_FAILED', cls)
        loss = .5*model.compute_defect_loss(result['noise_pred'], noise, native_mask) + .05*attention
        loss.backward()
        c.require(bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params),
                  'NONFINITE_GRADIENT_PROBE', cls)
        reports.append(dict(class_name=cls, loss=float(loss.detach()), attention_loss=float(attention.detach()),
            attention_gradient_nonzero=nonzero, image_shape=list(image.shape), latent_shape=list(latent.shape), optimizer_steps=0))
    return reports


def ad_probe(args, torch, rows, device):
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from ldm.data.frozen_v2 import FrozenV2Dataset
    from torch.utils.data import default_collate
    from transformers import BertTokenizerFast

    tokenizer_path = Path(os.environ.get(
        'AD_BERT_TOKENIZER_PATH', '/mnt/sda/cbj/GLIP-main/bert-base-uncased'))
    c.require(all((tokenizer_path / name).is_file()
                  for name in ('tokenizer.json', 'tokenizer_config.json', 'vocab.txt')),
              'BERT_TOKENIZER_FILES_MISSING', str(tokenizer_path))
    original_from_pretrained = BertTokenizerFast.from_pretrained

    def load_local_bert_tokenizer(cls, model_name_or_path, *model_args, **kwargs):
        if model_name_or_path == 'bert-base-uncased':
            model_name_or_path = str(tokenizer_path.resolve())
            kwargs['local_files_only'] = True
        return original_from_pretrained(model_name_or_path, *model_args, **kwargs)

    BertTokenizerFast.from_pretrained = classmethod(load_local_bert_tokenizer)
    os.environ['AD_NAME_ANOMALY_FILE'] = str(args.manifest.resolve().parent/'name-anomaly.txt')
    with stage('读取 AnomalyDiffusion 配置'):
        cfg = OmegaConf.load(args.source/'configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml')
    cfg.model.params.generation_resolution = 512
    cfg.model.params.spatial_condition_resolution = 256
    cfg.model.params.image_size = 64
    cfg.model.params.ckpt_path = None
    with stage('从本地 checkpoint 读取 state_dict 到 CPU'):
        payload = torch.load(str(args.base_model), map_location='cpu')
    with stage('按配置构造 LatentDiffusion 模型'):
        model = instantiate_from_config(cfg.model)
    with stage('将 checkpoint 权重载入 CPU 模型'):
        missing, unexpected = model.load_state_dict(payload['state_dict'], strict=False)
    c.require(not [k for k in missing if k.startswith(('model.', 'first_stage_model.', 'cond_stage_model.'))],
              'AD_NATIVE_BACKBONE_KEYS_MISSING', missing)
    del payload
    with stage('将模型复制到 GPU'):
        model.to(device)
    with stage('准备 AnomalyDiffusion spatial encoder'):
        model.prepare_spatial_encoder(optimze_together=True, data_enhance=True)
    model.learning_rate = cfg.model.base_learning_rate
    reports = []
    for cls in ('flash', 'black'):
        batch = default_collate([FrozenV2Dataset(args.manifest, cls, repeats=1)[0]])
        batch['image'] = batch['image'].to(device)
        batch['mask'] = batch['mask'].to(device)
        model.zero_grad(set_to_none=True)
        z, conditioning, meta = model.get_input(batch, model.first_stage_key)
        c.require(list(z.shape[-2:]) == [64,64] and list(meta['mask_cond'].shape[-2:]) == [256,256],
                  'AD_V2_RESOLUTION_PROBE_FAILED', cls)
        loss, _ = model(z, conditioning, **meta)
        loss.backward()
        gradients = [p.grad for p in model.embedding_manager.parameters() if p.requires_grad and p.grad is not None]
        c.require(bool(torch.isfinite(loss)) and gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
                  and any(float(g.float().norm()) > 0 for g in gradients), 'AD_GRADIENT_PROBE_FAILED', cls)
        reports.append(dict(class_name=cls, loss=float(loss.detach()), latent_shape=list(z.shape),
            spatial_condition_shape=list(meta['mask_cond'].shape), optimizer_steps=0))
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['anomalydiffusion','defectfill'], required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--base_model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='0')
    args = parser.parse_args()
    started = time.monotonic()
    try:
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
        print('[诊断] 可向探针 PID 发送 SIGUSR1 输出 Python 全线程栈。', flush=True)
    except (AttributeError, OSError, RuntimeError, ValueError) as exc:
        print('[诊断] 无法注册 SIGUSR1 栈转储：' + str(exc), flush=True)
    print(f'[启动] {args.method} 修复后GPU梯度检查 PID={os.getpid()}；不训练、不采样。', flush=True)
    result = dict(status='RUNTIME_PROBE_FAILED', method=args.method, formal_generation_authorized=False,
        formal_training_authorized=False, optimizer_step_count=0, synthetic_generation_count=0,
        detector_training_count=0, final_eval_image_or_label_content_read=0)
    code = 1
    try:
        c.require(not args.output.exists(), 'PROBE_OUTPUT_EXISTS_USE_NEW_DIRECTORY', str(args.output))
        c.require(args.base_model.exists(), 'BASE_MODEL_MISSING', str(args.base_model))
        result['repaired_source_binding'] = verify_repaired_source(args.method,args.source)
        cfg = c.load(c.REPO/'configs/plastic_bomo_chapter3_exp2.json')
        exp1, _, reference = b.frozen_inputs(cfg)
        with single_gpu_lock(), b.forbid_content_reads(b.forbidden_paths(exp1, reference)), heartbeat(args.method):
            rows = load_training_records(args.manifest)
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            sys.path.insert(0, str(args.source.resolve()))
            import torch
            c.require(torch.cuda.is_available(), 'CUDA_REQUIRED_FOR_NATIVE_RUNTIME_PROBE', 'CPU unit tests do not qualify native runtime')
            device = torch.device('cuda:'+args.device)
            torch.cuda.set_device(device)
            torch.manual_seed(42)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.reset_peak_memory_stats(device)
            print('[运行中] 加载本地基础模型，检查Flash/Black的512前向与反向；不更新参数。', flush=True)
            reports = (ad_probe if args.method == 'anomalydiffusion' else df_probe)(args, torch, rows, device)
            result.update(status='REPAIRED_GRADIENT_PROBE_PASS_NOT_FORMAL_AUTHORIZATION', records=reports,
                gpu_name=torch.cuda.get_device_name(device), total_vram_bytes=torch.cuda.get_device_properties(device).total_memory,
                peak_vram_bytes=torch.cuda.max_memory_allocated(device), torch_version=torch.__version__)
            gc.collect()
            torch.cuda.empty_cache()
            code = 0
    except Exception as exc:
        result['reason'] = exc.status if isinstance(exc,c.Stop) else type(exc).__name__
        result['detail'] = f'{result["reason"]}: {exc}'
        print('[检查失败] '+result['detail'], flush=True)
    result['runtime_seconds'] = time.monotonic()-started
    if not args.output.exists():
        save_json(args.output, result)
    print(f'[已退出] {result["status"]}；退出码={code}。报告：{args.output}', flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
