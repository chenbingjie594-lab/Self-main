#!/usr/bin/env python3
"""CPU-only diagnosis of frozen Plastic_Bomo synthetic images.

Preparation freezes geometry, descriptive thresholds, and the existing Exp4
sample binding before any image pixels are opened. Analysis never trains,
generates, selects, replaces, or modifies source images.
"""
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/plastic_bomo_chapter3_exp5_validity_diagnosis.json"
METHOD_FIELDS = {
    "Raw SD2": "raw", "MSDF": "msdf", "AnomalyDiffusion": "anomalydiffusion"}
ZERO = {"training_count": 0, "generation_count": 0, "resampling_count": 0,
        "quality_selection_count": 0, "manual_replacement_count": 0,
        "historical_metric_change_count": 0, "gpu_use_count": 0}


class Stop(RuntimeError):
    def __init__(self, status, detail):
        self.status, self.detail = status, detail
        super().__init__(f"{status}: {detail}")


def require(ok, status, detail):
    if not ok:
        raise Stop(status, detail)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def portable(path):
    path = Path(path).resolve()
    return path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else str(path)


def checked(path, digest, status="FROZEN_INPUT_CHANGED"):
    path = resolve(path)
    require(path.is_file(), "DATA_REQUIRED", portable(path))
    actual = sha(path)
    require(actual == digest, status, {"path": portable(path), "expected": digest, "actual": actual})
    return path


def csv_save(path, header, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f); writer.writerow(header); writer.writerows(rows)
    tmp.replace(path)


def bbox_pixels(bbox, width, height):
    cx, cy, bw, bh = bbox
    require(len(bbox) == 4 and all(math.isfinite(v) for v in bbox)
            and 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1,
            "PROTOCOL_ISSUE_FOUND", bbox)
    box = [max(0, math.floor((cx - bw / 2) * width)),
           max(0, math.floor((cy - bh / 2) * height)),
           min(width, math.ceil((cx + bw / 2) * width)),
           min(height, math.ceil((cy + bh / 2) * height))]
    require(box[2] > box[0] and box[3] > box[1], "PROTOCOL_ISSUE_FOUND", box)
    return box


def roi_pixels(bbox, width, height):
    cx, cy, bw, bh = bbox
    return [max(0, math.floor((cx - .75 * bw) * width)),
            max(0, math.floor((cy - .75 * bh) * height)),
            min(width, math.ceil((cx + .75 * bw) * width)),
            min(height, math.ceil((cy + .75 * bh) * height))]


def prepare(config):
    config = Path(config).resolve(); cfg = load(config); output = resolve(cfg["output"])
    require(not output.exists() or not any(output.iterdir()), "OUTPUT_NOT_EMPTY", str(output))
    for name, digest in cfg["input_sha256"].items(): checked(name, digest)
    matched = load(resolve(cfg["exp4_root"]) / "three_method_matched_manifest.json")
    sampling = load(resolve(cfg["exp4_root"]) / "qualitative_sampling_manifest.json")
    slots = load(resolve(cfg["generation_slots"]))["slots"]
    require(matched["status"] == "FROZEN" and len(matched["records"]) == 240,
            "DATA_REQUIRED", "Exp4 matched population")
    require(sampling["status"] == "FROZEN_BEFORE_IMAGE_VIEW"
            and len(sampling["records"]) == 24, "DATA_REQUIRED", "Exp4 frozen sample")
    require(len(slots) == 80 and len({s["slot_id"] for s in slots}) == 80,
            "PROTOCOL_ISSUE_FOUND", "generation slots")
    sample_keys = [(r["bank"], r["slot_id"]) for r in sampling["records"]]
    require(len(sample_keys) == len(set(sample_keys)) == 24, "PROTOCOL_ISSUE_FOUND", "Exp4 sample keys")
    protocol = {
        "status": "FROZEN_BEFORE_PIXEL_ANALYSIS", "experiment": cfg["experiment"],
        "device": "CPU", "matched_groups": 240, "generated_images": 720,
        "diagnostic_subset_source": "Exp4 qualitative_sampling_manifest.json; exact 24 groups; no resampling",
        "regions": {
            "B": "original normalized xywh bbox rasterized by floor(x0,y0), ceil(x1,y1), clipped to final canvas",
            "P": "same bbox expanded by 25% of bbox width/height on every side; Experiment 1 crop",
            "background": "P minus B", "full": "4096x1024 final canvas"},
        "coordinate_audit": {
            "bbox_format": "normalized_xywh", "coordinate_system": "final 4096x1024 canvas",
            "generator_patch": "512x512 without resize_scale; inverse x_final=x_generator+crop_left, y_final=y_generator+crop_top",
            "method_geometry": "all three methods use the same frozen slot and paste_valid_crop_region_xyxy",
            "exp1_roi_rasterization": "floor lower bounds, ceil upper bounds, canvas clipping"},
        "change_pixel_thresholds_mean_abs_rgb": cfg["change_pixel_thresholds_mean_abs_rgb"],
        "threshold_freeze_statement": "all descriptive thresholds frozen before method pixels were opened",
        "low_information_candidate_rule": cfg["low_information_candidate_rule"],
        "low_information_interpretation": "human-review prompt only; never VALID/INVALID and never excluded",
        "residual_range": [0, 255], "per_image_normalization": False,
        "input_bindings": cfg["input_sha256"], "pixel_files_opened_before_freeze": 0,
        "historical_results_modified": False, **ZERO}
    output.mkdir(parents=True)
    save(output / "exp5_protocol.json", protocol)
    save(output / "exp5_sample_binding.json", {
        "status": "REUSED_EXACTLY_FROM_EXP4", "count": 24, "records": sampling["records"],
        "source_sha256": cfg["input_sha256"][portable(resolve(cfg["exp4_root"]) / "qualitative_sampling_manifest.json")]})
    save(output / "pre_analysis_binding.json", {
        "status": "FROZEN_BEFORE_PIXEL_ANALYSIS",
        "sha256": {n: sha(output / n) for n in ("exp5_protocol.json", "exp5_sample_binding.json")},
        "implementation_sha256": {portable(config): sha(config), portable(__file__): sha(__file__)},
        "pixel_files_opened": 0})
    save(output / "exp5_status.json", {"status": "PREPARED", "automatic_audit_complete": False,
        "human_review_status": "AWAITING_HUMAN", **ZERO})
    print("[Exp5] 协议与阈值已在读取图像像素前冻结；复用 Exp4 的原始24组。", flush=True)


def verify_prepared(cfg, config, output):
    b = load(output / "pre_analysis_binding.json")
    require(b["status"] == "FROZEN_BEFORE_PIXEL_ANALYSIS" and b["pixel_files_opened"] == 0,
            "PROTOCOL_ISSUE_FOUND", "pre-analysis binding")
    for n, d in b["sha256"].items(): checked(output / n, d, "PROTOCOL_ISSUE_FOUND")
    require(b["implementation_sha256"] == {portable(config): sha(config), portable(__file__): sha(__file__)},
            "PROTOCOL_ISSUE_FOUND", "implementation changed after threshold freeze")
    for name, digest in cfg["input_sha256"].items(): checked(name, digest)


def open_rgb(path, wh):
    from PIL import Image
    with Image.open(resolve(path)) as image:
        image.load(); require(list(image.size) == wh, "PROTOCOL_ISSUE_FOUND", portable(resolve(path)))
        return image.convert("RGB")


def region_stats(image, box):
    from PIL import ImageStat
    crop = image.crop(tuple(box)); gray = crop.convert("L"); stat = ImageStat.Stat(gray)
    hist = gray.histogram()
    return {"rgb_mean": list(ImageStat.Stat(crop).mean), "gray_mean": stat.mean[0],
            "gray_std": stat.stddev[0], "gray_hist": hist,
            "pixels": crop.width * crop.height}


def difference_stats(normal, generated, b, p, thresholds):
    from PIL import ImageChops, ImageStat
    n_p, g_p = normal.crop(tuple(p)), generated.crop(tuple(p))
    diff_p = ImageChops.difference(g_p, n_p)
    rb = [b[0]-p[0], b[1]-p[1], b[2]-p[0], b[3]-p[1]]
    diff_b = diff_p.crop(tuple(rb)); gen_b = g_p.crop(tuple(rb))
    npix, nb = diff_p.width * diff_p.height, diff_b.width * diff_b.height
    require(npix > nb > 0, "PROTOCOL_ISSUE_FOUND", {"B": b, "P": p})
    sp, sb = ImageStat.Stat(diff_p), ImageStat.Stat(diff_b)
    p_sum, b_sum = sp.sum, sb.sum; bg_n = npix - nb
    bg_mad = sum(p_sum[i] - b_sum[i] for i in range(3)) / (3 * bg_n)
    pixel_mean = [sum(px) / 3 for px in diff_b.getdata()]
    ratios = {f"bbox_changed_ratio_ge_{t}": sum(v >= t for v in pixel_mean) / nb for t in thresholds}
    gen_p_gray = g_p.convert("L"); gen_b_gray = gen_b.convert("L")
    gp, gb = ImageStat.Stat(gen_p_gray), ImageStat.Stat(gen_b_gray)
    gray_bg = (gp.sum[0] - gb.sum[0]) / bg_n
    rgb = ImageStat.Stat(gen_b).mean
    return {
        "bbox_mean_abs_rgb_diff": sum(sb.mean) / 3,
        "roi_mean_abs_rgb_diff": sum(sp.mean) / 3,
        "background_p_minus_b_mean_abs_rgb_diff": bg_mad,
        "bbox_max_abs_channel_diff": max(v[1] for v in sb.extrema),
        **ratios,
        "bbox_background_gray_contrast_abs": abs(gb.mean[0] - gray_bg),
        "bbox_rgb_mean_r": rgb[0], "bbox_rgb_mean_g": rgb[1], "bbox_rgb_mean_b": rgb[2],
        "bbox_gray_mean": gb.mean[0], "bbox_gray_std": gb.stddev[0],
        "bbox_width_px": b[2]-b[0], "bbox_height_px": b[3]-b[1],
        "roi_width_px": p[2]-p[0], "roi_height_px": p[3]-p[1]}


def scaled_box(box, sx, sy):
    return [round(box[0]*sx), round(box[1]*sy), round(box[2]*sx), round(box[3]*sy)]


def font(size, bold=False):
    from PIL import ImageFont
    n = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(Path("/usr/share/fonts/truetype/dejavu") / n), size)


def overlay_image(normal, b, p, label, wh):
    from PIL import Image, ImageDraw
    out = normal.resize(tuple(wh), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(out); sx, sy = wh[0]/normal.width, wh[1]/normal.height
    draw.rectangle(scaled_box(p, sx, sy), outline=(255, 215, 0), width=3)
    draw.rectangle(scaled_box(b, sx, sy), outline=(255, 0, 0), width=3)
    draw.rectangle((0, 0, min(1500, wh[0]), 38), fill=(255,255,255))
    draw.text((8, 6), label + " | B=red, P=yellow", fill=(0,0,0), font=font(20, True))
    return out


def diagnostic_grid(path, sample, row, images, b, p, cfg):
    from PIL import Image, ImageChops, ImageDraw
    labels = ("Conditioning Normal", "Raw SD2", "MSDF", "AnomalyDiffusion")
    fw, fh = cfg["display"]["full_wh"]; rw, rh = cfg["display"]["roi_wh"]
    cell, gap, left, top = fw + 20, 16, 95, 105
    y_full, y_roi, y_res = top, top + fh + 48, top + fh + 48 + rh + 48
    canvas = Image.new("RGB", (left + 4*cell + 3*gap + 25, y_res + rh + 35), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((left, 12), f"{sample['class'].title()} | Bank {sample['bank']} | Slot ID: {sample['slot_id']}",
              fill="black", font=font(29, True))
    for i, (label, image) in enumerate(zip(labels, images)):
        x = left + i*(cell+gap); draw.text((x+cell/2, 75), label, fill="black", font=font(22,True), anchor="mm")
        full = image.resize((fw,fh), Image.Resampling.BILINEAR)
        d = ImageDraw.Draw(full); d.rectangle(scaled_box(b, fw/image.width, fh/image.height), outline=(255,0,0), width=2)
        roi = image.crop(tuple(p)).resize((rw,rh), Image.Resampling.BILINEAR)
        residual = Image.new("RGB", image.size, (0,0,0)) if i == 0 else ImageChops.difference(image, images[0])
        residual = residual.crop(tuple(p)).resize((rw,rh), Image.Resampling.BILINEAR)
        for y, item in ((y_full,full),(y_roi,roi),(y_res,residual)):
            px=x+(cell-item.width)//2; canvas.paste(item,(px,y)); draw.rectangle((px-1,y-1,px+item.width,y+item.height),outline=(70,70,70),width=2)
    for y, name in ((y_full+fh/2,"Full+B"),(y_roi+rh/2,"ROI P"),(y_res+rh/2,"Residual")):
        draw.text((8,y),name,fill="black",font=font(21,True),anchor="lm")
    path.parent.mkdir(parents=True, exist_ok=True); canvas.save(path,"PNG",dpi=(300,300))


def histogram_figure(path, bbox_hist, roi_hist):
    from PIL import Image, ImageDraw
    w,h=1800,760; left,right,top,bottom=120,50,90,110
    canvas=Image.new("RGB",(w,h),"white"); draw=ImageDraw.Draw(canvas)
    draw.text((left,20),"Flash conditioning-normal grayscale distributions (40 unique slots)",fill="black",font=font(30,True))
    ymax=max(max(bbox_hist),max(roi_hist),1); pw=w-left-right; ph=h-top-bottom
    draw.line((left,top,left,h-bottom,w-right,h-bottom),fill="black",width=2)
    for values,color in ((bbox_hist,(220,30,30)),(roi_hist,(220,170,0))):
        pts=[]
        for i,v in enumerate(values): pts.append((left+i*pw/255,h-bottom-v/ymax*ph))
        draw.line(pts,fill=color,width=4)
    for x in (0,64,128,192,255):
        px=left+x*pw/255; draw.text((px,h-bottom+18),str(x),fill="black",font=font(18),anchor="ma")
    draw.text((w/2,h-42),"Grayscale intensity (fixed 0–255)",fill="black",font=font(22),anchor="mm")
    draw.line((w-520,42,w-450,42),fill=(220,30,30),width=4); draw.text((w-435,42),"B",fill="black",font=font(20),anchor="lm")
    draw.line((w-330,42,w-260,42),fill=(220,170,0),width=4); draw.text((w-245,42),"P",fill="black",font=font(20),anchor="lm")
    canvas.save(path,"PNG",dpi=(300,300))


def summary_rows(stats, dimensions):
    groups=defaultdict(list)
    for r in stats: groups[tuple(r[d] for d in dimensions)].append(r)
    metrics=["bbox_mean_abs_rgb_diff","roi_mean_abs_rgb_diff","background_p_minus_b_mean_abs_rgb_diff",
             "bbox_max_abs_channel_diff","bbox_changed_ratio_ge_5","bbox_changed_ratio_ge_10",
             "bbox_changed_ratio_ge_20","bbox_changed_ratio_ge_30","bbox_background_gray_contrast_abs",
             "bbox_rgb_mean_r","bbox_rgb_mean_g","bbox_rgb_mean_b","bbox_gray_mean","bbox_gray_std"]
    header=list(dimensions)+["n"]+[f"{m}_{s}" for m in metrics for s in ("mean","median","sd","min","max")]
    rows=[]
    for key,values in sorted(groups.items()):
        out=list(key)+[len(values)]
        for m in metrics:
            a=[float(r[m]) for r in values]
            out += [statistics.fmean(a),statistics.median(a),statistics.stdev(a) if len(a)>1 else 0,min(a),max(a)]
        rows.append(out)
    return header,rows


def analyze(config):
    config=Path(config).resolve(); cfg=load(config); output=resolve(cfg["output"])
    verify_prepared(cfg,config,output)
    matched=load(resolve(cfg["exp4_root"])/"three_method_matched_manifest.json")["records"]
    selected=load(output/"exp5_sample_binding.json")["records"]
    selection={(r["bank"],r["slot_id"]):r for r in selected}
    slots={s["slot_id"]:s for s in load(resolve(cfg["generation_slots"]))["slots"]}
    method_maps={}
    for method,path in cfg["method_manifests"].items():
        d=load(resolve(path)); require(d["status"]=="FROZEN" and len(d["records"])==240,"DATA_REQUIRED",method)
        method_maps[method]={(r["bank"],r["slot_id"]):r for r in d["records"]}
    keys={(r["bank"],r["slot_id"]) for r in matched}
    require(len(keys)==240 and all(set(m)==keys for m in method_maps.values()),"DATA_REQUIRED","three method key coverage")
    print("[Exp5] 首先校验720张生成图、三方法条件与80张去重 Normal 的 SHA256。",flush=True)
    normal_seen={}
    for i,row in enumerate(matched,1):
        key=(row["bank"],row["slot_id"]); slot=slots[row["slot_id"]]
        require(canonical_sha(slot)==row["frozen_slot_sha256"] and row["bbox"]==slot["bbox"]
                and row["class"]==slot["class"] and row["conditioning_normal_id"]==slot["conditioning_normal_id"],
                "PROTOCOL_ISSUE_FOUND", key)
        for method,prefix in METHOD_FIELDS.items():
            src=method_maps[method][key]
            require(src["frozen_slot"]==slot and src["image_sha256"]==row[prefix+"_image_sha256"],
                    "PROTOCOL_ISSUE_FOUND", {"method":method,"key":key})
            checked(row[prefix+"_image_path"],row[prefix+"_image_sha256"])
        normal_seen[row["conditioning_normal_path"]]=row["conditioning_normal_sha256"]
        if i%20==0: print(f"[Exp5] SHA/配对 {i}/240 组。",flush=True)
    for p,d in normal_seen.items(): checked(p,d)
    print(f"[Exp5] 输入完整性通过：720张生成图，{len(normal_seen)}张去重 Normal。开始CPU像素统计。",flush=True)
    from PIL import Image, ImageDraw
    geometry=[]; changes=[]; flash_b=[0]*256; flash_p=[0]*256; flash_seen=set()
    overlay_dir=output/"slot_geometry_overlays"; diag_dir=output/"diagnostic_grids"
    overlay_dir.mkdir(); diag_dir.mkdir()
    structural=[]
    for i,row in enumerate(matched,1):
        key=(row["bank"],row["slot_id"]); slot=slots[row["slot_id"]]; g=slot["geometry"]
        w,h=row["canvas_wh"]; b=bbox_pixels(row["bbox"],w,h); p=roi_pixels(row["bbox"],w,h)
        if row["roi_xyxy"]!=p: structural.append({"key":key,"issue":"EXP1_ROI_MISMATCH"})
        original=g["original_bbox_xyxy"]
        expected=[(row["bbox"][0]-row["bbox"][2]/2)*w,(row["bbox"][1]-row["bbox"][3]/2)*h,
                  (row["bbox"][0]+row["bbox"][2]/2)*w,(row["bbox"][1]+row["bbox"][3]/2)*h]
        if max(abs(a-z) for a,z in zip(original,expected))>1e-8: structural.append({"key":key,"issue":"NORMALIZED_TO_FINAL_MISMATCH"})
        x,y,_,_=g["fixed_crop_xyxy"]; target=g["generator_bbox_xyxy"]
        inverse=[target[0]+x,target[1]+y,target[2]+x,target[3]+y]
        if max(abs(a-z) for a,z in zip(original,inverse))>1e-8: structural.append({"key":key,"issue":"INVERSE_MAPPING_MISMATCH"})
        normal=open_rgb(row["conditioning_normal_path"],row["canvas_wh"])
        bs,ps=region_stats(normal,b),region_stats(normal,p); bg_n=ps["pixels"]-bs["pixels"]
        bg_mean=(ps["gray_mean"]*ps["pixels"]-bs["gray_mean"]*bs["pixels"])/bg_n
        low=ps["gray_mean"]<cfg["low_information_candidate_rule"]["gray_mean_lt"] and ps["gray_std"]<cfg["low_information_candidate_rule"]["gray_std_lt"]
        geometry.append({"bank":row["bank"],"slot_id":row["slot_id"],"class":row["class"],
            "canvas_width":w,"canvas_height":h,"bbox_x0":b[0],"bbox_y0":b[1],"bbox_x1":b[2],"bbox_y1":b[3],
            "roi_x0":p[0],"roi_y0":p[1],"roi_x1":p[2],"roi_y1":p[3],
            "bbox_width":b[2]-b[0],"bbox_height":b[3]-b[1],"roi_width":p[2]-p[0],"roi_height":p[3]-p[1],
            "roi_touches_canvas":p[0]==0 or p[1]==0 or p[2]==w or p[3]==h,
            "normal_bbox_mean_intensity":bs["gray_mean"],"normal_roi_mean_intensity":ps["gray_mean"],
            "normal_bbox_std_intensity":bs["gray_std"],"normal_roi_std_intensity":ps["gray_std"],
            "normal_background_p_minus_b_mean_intensity":bg_mean,
            "normal_bbox_black_fraction_lt20":sum(bs["gray_hist"][:20])/bs["pixels"],
            "normal_roi_black_fraction_lt20":sum(ps["gray_hist"][:20])/ps["pixels"],
            "low_information_roi_candidate":low,"structural_geometry_error":False})
        if row["class"]=="flash" and row["slot_id"] not in flash_seen:
            flash_seen.add(row["slot_id"]); flash_b=[a+z for a,z in zip(flash_b,bs["gray_hist"])]; flash_p=[a+z for a,z in zip(flash_p,ps["gray_hist"])]
        overlay=overlay_image(normal,b,p,f"{row['class'].title()} | Bank {row['bank']} | {row['slot_id']}",cfg["display"]["overlay_wh"])
        overlay.save(overlay_dir/f"{row['class']}_{row['bank']}_{row['slot_id']}.png","PNG",dpi=(300,300))
        images=[normal]
        for method,prefix in METHOD_FIELDS.items():
            generated=open_rgb(row[prefix+"_image_path"],row["canvas_wh"]); images.append(generated)
            changes.append({"method":method,"bank":row["bank"],"slot_id":row["slot_id"],"class":row["class"],
                            **difference_stats(normal,generated,b,p,cfg["change_pixel_thresholds_mean_abs_rgb"])})
        if key in selection:
            diagnostic_grid(diag_dir/selection[key]["output_name"],selection[key],row,images,b,p,cfg)
        if i%10==0: print(f"[Exp5] 像素/几何 {i}/240 组；统计 {len(changes)}/720 图。",flush=True)
    require(not structural,"PROTOCOL_ISSUE_FOUND",structural[:5])
    gh=list(geometry[0]); csv_save(output/"slot_geometry_audit.csv",gh,[[r[x] for x in gh] for r in geometry])
    ch=list(changes[0]); csv_save(output/"synthetic_change_statistics.csv",ch,[[r[x] for x in ch] for r in changes])
    for name,dims in (("classwise_change_summary.csv",("method","class")),("bankwise_change_summary.csv",("method","class","bank"))):
        header,rows=summary_rows(changes,dims); csv_save(output/name,header,rows)
    csv_save(output/"flash_normal_intensity_histograms.csv",["gray_value","bbox_B_count","roi_P_count"],[[i,flash_b[i],flash_p[i]] for i in range(256)])
    histogram_figure(output/"flash_normal_intensity_histograms.png",flash_b,flash_p)
    review_header=["class","bank","sample_index","slot_id","method","visibility","position","morphology","artifact","notes"]
    review=[]
    for s in selected:
        for method in METHOD_FIELDS: review.append([s["class"],s["bank"],s["sample_index"],s["slot_id"],method,"","","","",""])
    csv_save(output/"human_review_template.csv",review_header,review)
    (output/"human_review_protocol.md").write_text("""# Exp5 Human Review Protocol

Review the fixed 24-group diagnostic subset only. First consult `real_reference_examples.png` from Exp4; those images are representative real examples and are not paired ground truth.

For every generated image, fill all fields without deleting uncertain cases:

- visibility: `VISIBLE`, `NOT_VISIBLE`, or `UNCERTAIN`
- position: `INSIDE_BBOX`, `PARTIALLY_OUTSIDE`, `OUTSIDE_BBOX`, or `UNCERTAIN`
- morphology: `PLAUSIBLE`, `SUSPICIOUS`, or `UNCERTAIN`
- artifact: `YES`, `NO`, or `UNCERTAIN`
- notes: free text

The 72 rows describe only this frozen diagnostic subset. Do not report them as a 720-image validity rate. Low-information candidates are prompts for inspection, not invalidity labels.
""",encoding="utf-8")
    save(output/"human_review_status.json",{"human_review_status":"AWAITING_HUMAN","expected_rows":72,"completed_rows":0,"automatic_entries":0})
    low_counts=dict(Counter((r["class"],r["bank"]) for r in geometry if r["low_information_roi_candidate"]))
    save(output/"failure_category_summary.json",{
        "status":"AUTOMATIC_AUDIT_ONLY","A_slot_geometry":{"confirmed_coordinate_errors":0,"low_information_roi_candidates":sum(low_counts.values()),"candidate_counts_by_class_bank":{f"{k[0]}_{k[1]}":v for k,v in low_counts.items()},"candidate_interpretation":"AWAITING_HUMAN; not invalid"},
        "B_generation_method":{"automatic_failure_labels":0,"status":"AWAITING_HUMAN"},
        "C_cannot_determine":{"human_review_rows_pending":72},
        "causal_attribution_performed":False})
    write_report(output,geometry,changes,cfg)
    pngs=sorted(diag_dir.glob("*.png")); overlays=sorted(overlay_dir.glob("*.png"))
    require(len(pngs)==24 and len(overlays)==240,"PROTOCOL_ISSUE_FOUND",[len(pngs),len(overlays)])
    audit={"status":"PASS","matched_group_count":240,"generation_image_sha256_checks":720,
           "unique_normal_sha256_checks":len(normal_seen),"class_consistency_checks":720,
           "bbox_consistency_checks":720,"conditioning_normal_consistency_checks":720,"frozen_slot_consistency_checks":720,
           "coordinate_mapping_errors":0,"exp1_roi_mismatches":0,"diagnostic_grid_count":24,"geometry_overlay_count":240,
           "change_statistic_rows":720,"geometry_rows":240,"exp4_sample_reused_without_resampling":True,
           "thresholds_frozen_before_pixel_analysis":True,"human_review_status":"AWAITING_HUMAN",**ZERO}
    save(output/"integrity_audit.json",audit)
    public={"status":"AUDIT_COMPLETE_AWAITING_HUMAN","matched_groups":240,"generated_images_verified":720,
            "geometry_errors":0,"diagnostic_grids":24,"geometry_overlays":240,"automatic_statistic_rows":720,
            "human_review":"AWAITING_HUMAN","historical_metrics_modified":False,
            "diagnostic_images_uploaded":False,"reason":"public permission for Exp5 diagnostic images not requested",**ZERO}
    save(output/"EXP5_PUBLIC_SUMMARY.json",public)
    save(output/"EXP5_PUBLIC_PROTOCOL.json",{k:load(output/"exp5_protocol.json")[k] for k in
         ("status","device","matched_groups","generated_images","diagnostic_subset_source","regions","coordinate_audit","change_pixel_thresholds_mean_abs_rgb","threshold_freeze_statement","low_information_interpretation","residual_range","per_image_normalization","historical_results_modified")})
    artifacts={p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name!="exp5_status.json"}
    save(output/"exp5_status.json",{"status":"AUDIT_COMPLETE_AWAITING_HUMAN","automatic_audit_complete":True,
         "human_review_status":"AWAITING_HUMAN","matched_groups":240,"images_analyzed":720,
         "diagnostic_grid_count":24,"geometry_overlay_count":240,"artifact_sha256":artifacts,**ZERO})
    print("[Exp5] AUDIT_COMPLETE_AWAITING_HUMAN",flush=True)


def write_report(output,geometry,changes,cfg):
    def group(method,klass): return [r for r in changes if r["method"]==method and r["class"]==klass]
    def med(rows,key): return statistics.median(float(r[key]) for r in rows)
    low=[r for r in geometry if r["low_information_roi_candidate"]]
    flash=[r for r in geometry if r["class"]=="flash"]
    low_flash=[r for r in flash if r["low_information_roi_candidate"]]
    low_flash_unique=len({r["slot_id"] for r in low_flash})
    touch=sum(r["roi_touches_canvas"] for r in flash)
    lines=["# Chapter 3 Experiment 5 — Synthetic Defect Validity and Slot Diagnosis","",
    "## 1. 实验背景","","本实验诊断三个冻结方法在共同槽位上的可见变化与坐标关系，不训练、不生成、不筛选、不修改历史评价。","",
    "## 2. 数据与冻结协议","",f"240 个 `bank + slot_id` 匹配组、720 张生成图全部通过 SHA256 与 frozen-slot 一致性检查。固定复用 Exp4 的 24 组，不重新抽样。变化像素阈值 {cfg['change_pixel_thresholds_mean_abs_rgb']} 在打开方法图像前冻结。","",
    "## 3. bbox 与 ROI 几何审计","",f"bbox 确认为 final 4096×1024 画布上的 normalized xywh。B 使用 bbox 本体；P 为每侧扩展 25% 的 Exp1 ROI。240 组 normalized→final、generator→final 逆映射和 Exp1 ROI 均一致，确认坐标错误 0 个。Flash 的 P 接触画布边界 {touch}/{len(flash)} 组（分母含 A/B/C 重复 bank 条件）。","",
    "## 4. Flash 黑色 ROI 排查","",f"预冻结的低信息提示规则（P 灰度均值<20 且标准差<10）标记 Flash {len(low_flash)}/120 个 bank+slot 条件，对应 {low_flash_unique}/40 个唯一 Flash slot；全体两类共 {len(low)}/240。它只用于提示人工检查，不能判定无效或删除。逐槽位 B/P 均值、标准差、坐标、尺寸与固定 0–255 直方图已保存。亮度统计本身不能区分材料边界、正常黑背景与生成有效性。","",
    "## 5. AnomalyDiffusion 缺陷可见性分析","",]
    for klass in ("flash","black"):
        vals=[]
        for method in METHOD_FIELDS: vals.append(f"{method}={med(group(method,klass),'bbox_mean_abs_rgb_diff'):.4f}")
        lines += [f"- {klass.title()} 的 bbox 平均绝对 RGB 差中位数："+"，".join(vals)+"。"]
    edge_statement = (f"确认 {touch}/{len(flash)} 个 Flash 条件的 P 接触画布边界"
                      if touch else f"未发现 Flash 的 P 接触画布边界（0/{len(flash)}）")
    lines += ["","这些是连续像素变化描述量。变化小不等于没有缺陷，变化大也不等于真实或有效；人工可见性状态仍为 AWAITING_HUMAN。","",
    "## 6. MSDF Black 外观差异","",]
    ms=group("MSDF","black")
    lines += [f"MSDF Black 的 bbox 灰度均值中位数为 {med(ms,'bbox_gray_mean'):.4f}，bbox 灰度标准差中位数为 {med(ms,'bbox_gray_std'):.4f}，bbox 外 P−B 变化中位数为 {med(ms,'background_p_minus_b_mean_abs_rgb_diff'):.4f}。RGB、灰度、局部与背景变化的全分布已写入 CSV。颜色统计不能自动确认黄色/橙色光晕、边缘过硬或形态异常，也不支持模块或损失的因果归因。","",
    "## 7. 自动统计与人工审核边界","","自动统计覆盖全部 720 张图；人工模板只有固定子集的 72 个方法样本。当前没有人工填写，所有可见性、位置、形态和伪影判断均保持空白。未来填写后的比例必须保留 UNCERTAIN 并写明分母，不能外推为 720 张的总体有效率。","",
    "## 8. 已确认的问题","",f"未发现文件损坏、类别错配、conditioning-normal 错配、frozen-slot 错配、bbox 越界映射错误或 Exp1 ROI 计算错误；{edge_statement}。存在按预冻结亮度规则提示的低信息候选，但这不是无效生成判定。","",
    "## 9. 尚无法确认的问题","","Flash 黑色 ROI 是否代表不适合的生成槽位、AnomalyDiffusion 缺陷是否不可见、MSDF Black 是否存在重复光晕/拉长/硬边，均需要人工查看。现有自动统计不能区分槽位问题与方法生成问题，也不能定位网络模块或损失原因。","",
    "## 10. 下一步建议","","由研究者参照非配对真实示例，完成 72 行人工审核表；随后只汇总该固定诊断子集，并分别报告槽位问题、方法问题与无法判断。若人工确认严重协议问题，应另立纠正实验；本实验不修改 bbox、不删除 slot、不重算 FID/KID/LPIPS。",""]
    (output/"EXP5_REPORT.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("mode",choices=("prepare","analyze","all")); ap.add_argument("--config",default=str(CONFIG)); a=ap.parse_args()
    try:
        if a.mode in ("prepare","all"): prepare(a.config)
        if a.mode in ("analyze","all"): analyze(a.config)
    except Stop as e:
        print(f"[{e.status}] {e.detail}",file=sys.stderr,flush=True); return 2
    return 0


if __name__=="__main__": raise SystemExit(main())
