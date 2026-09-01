import argparse, os, random, re, json, types
import torch, numpy as np, cv2
from PIL import Image
from magic_ddim import DDIMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint_magic import \
     StableDiffusionInpaintPipeline_dynamic
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import \
     StableDiffusionInpaintPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler as StandardDDIMScheduler
from experiment_config import enabled_modules, load_config
from mdap import build_mdap_prior, find_reference_pair, infer_defect_mode
import glob
from diffusers.configuration_utils import FrozenDict 
from torchvision import transforms

RESOLUTION = 512

image_resize_center_crop = transforms.Compose(
    [
        transforms.Resize(RESOLUTION, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(RESOLUTION),
    ]
)
mask_resize_center_crop = transforms.Compose(
    [
        transforms.Resize(RESOLUTION, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(RESOLUTION),
    ]
)

# -------------------------------------------------------------------------
# 1) argument parser + miscellany
# -------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference-time mask-alignment augmentation.")
    # ───────── Keep existing arguments ─────────
    parser.add_argument("--defect_json", default=None,
                        help="Defect classification JSON; required only when CAMA is enabled.")
    parser.add_argument("--match_json", default=None,
                        help="Required only when the CAMA module is enabled.")
    parser.add_argument("--model_ckpt_root", required=True)
    parser.add_argument("--ddim_scheduler_root", default="./scheduler")
    parser.add_argument("--config", default=None,
                        help="JSON experiment config. See configs/baseline.json.")
    parser.add_argument("--prompt", default="a photo of a sks defect")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override the config seed. If omitted, use config seed (or 42 without a config).",
    )
    parser.add_argument("--categories", nargs='+', default=None)
    parser.add_argument("--blur_factor", type=int, default=0)
    parser.add_argument("--text_noise_scale", type=float, default=0.0)
    parser.add_argument("--output_name", default="./")
    parser.add_argument("--anomaly_strength_min", type=float, default=0.0)
    parser.add_argument("--anomaly_strength_max", type=float, default=0.0)
    parser.add_argument("--anomaly_stop_step", type=int, default=999999)
    parser.add_argument("--eta_mask_stop_step", type=int, default=999999)
    parser.add_argument("--normal_masks", default="./normal_masks")
    parser.add_argument(
        "--use_paired_normal",
        action="store_true",
        help=(
            "Use base_dir/<category>/paired_normal/<defect>/<mask stem>.* "
            "instead of sampling a random image from train/good."
        ),
    )
    parser.add_argument("--mask_dir", default=None,
                        help="Optional anomaly-mask root; defaults to dataset ground_truth masks.")
    parser.add_argument("--base_dir",    default="./mvtecad")
    parser.add_argument(
        "--reference_base_dir",
        default=None,
        help=(
            "Independent training-data root used by reference-conditioned "
            "modules such as RDA/MDAP. Defaults to --base_dir for backward "
            "compatibility; use a train split here during held-out evaluation."
        ),
    )
    parser.add_argument("--CAMA",        action="store_true")
    parser.add_argument("--use_random_mask", action="store_true")
    parser.add_argument("--dataset_type", choices=["mvtec_3d", "mvtec","visa","DAGM"],
                        default="mvtec_3d",
                        help="mvtec_3d: MVTec-3D Anomaly, mvtec: MVTec-AD 2-D")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="DDIM eta (0–1). Used only for DDIM scheduler; ignored by other schedulers.")
    parser.add_argument("--eta_mask_max", type=float, default=0.0,
                        help="eta_max inside the mask (upper bound when using a schedule).")
    parser.add_argument("--eta_mask_min", type=float, default=0.0,
                        help="eta_min inside the mask (lower bound when using a schedule).")

    # ───────── NEW: eta_mask scheduler / guard hyperparameters (passed into DDIM) ─────────
    parser.add_argument("--eta_mask_use_schedule", action="store_true",
                        help="If True, DDIM uses an internal step-wise schedule (eta_mask arguments are ignored).")
    parser.add_argument("--eta_mask_schedule",
                        choices=["constant","linear_down","linear_up",
                                 "cosine_down","cosine_up",
                                 "poly_down","poly_up",
                                 "exp_down","exp_up",
                                 "sigmoid_down","sigmoid_up"],
                        default="constant")
    parser.add_argument("--eta_mask_power", type=float, default=2.0,
                        help="Exponent p for polynomial schedule.")
    parser.add_argument("--eta_mask_exp_k", type=float, default=3.0,
                        help="Constant k for exponential schedule.")
    parser.add_argument("--eta_mask_sigmoid_k", type=float, default=8.0,
                        help="Slope parameter k for sigmoid schedule.")
    parser.add_argument("--eta_mask_segmented", action="store_true",
                        help="Remap the schedule to s∈[0,1] only over the positive (valid) interval "
                             "(interval start=max, end=0).")
    parser.add_argument("--eta_mask_guard",
                        choices=["none","clip_to_crit","zero_before_neg"],
                        default="none",
                        help="Critical-threshold guard mode.")
    parser.add_argument("--eta_mask_guard_margin", type=float, default=0.99,
                        help="Safety margin relative to the critical threshold (e.g., 0.99).")

    # ───────── Existing CFG scalar arguments (keep default behavior) ─────────
    parser.add_argument("--guidance_scale_inside", type=float, default=None,
                        help="Guidance scale inside the mask (if not set, use pipeline global guidance_scale).")
    parser.add_argument("--guidance_scale_outside", type=float, default=None,
                        help="Guidance scale outside the mask (if not set, use pipeline global guidance_scale).")

    # ───────── NEW: guidance_scale_inside scheduling / sampling control ─────────
    parser.add_argument("--gsi_use_schedule", action="store_true",
                        help="If True, use guidance_scale_inside as a step-wise schedule (max → min).")
    parser.add_argument("--gsi_schedule",
                        choices=["constant","linear","linear_down","cosine","cosine_down",
                                 "poly","poly_down","exp","exp_down","sigmoid","sigmoid_down"],
                        default="linear",
                        help="Selected schedule decreases from max to min.")
    parser.add_argument("--gsi_min", type=float, default=None,
                        help="Lower bound for guidance_scale_inside.")
    parser.add_argument("--gsi_max", type=float, default=None,
                        help="Upper bound for guidance_scale_inside.")
    parser.add_argument("--gsi_power", type=float, default=2.0,
                        help="Exponent for polynomial schedule.")
    parser.add_argument("--gsi_exp_k", type=float, default=3.0,
                        help="k parameter for exponential schedule.")
    parser.add_argument("--gsi_sigmoid_k", type=float, default=8.0,
                        help="k parameter for sigmoid schedule.")
    parser.add_argument("--gsi_sample_per_step", action="store_true",
                        help="If not using a schedule, sample per step from [gsi_min, gsi_max].")

    parsed = parser.parse_args()

    # A config file is the canonical experiment definition. Existing CLI flags
    # remain available for backwards compatibility when --config is omitted.
    if parsed.config:
        cfg = load_config(parsed.config)
        modules = cfg["modules"]
        # Keep older experiment JSON files usable after optional modules are
        # added. A missing optional module always means "disabled".
        prompt_noise = modules.get(
            "prompt_perturbation", {"enabled": False, "noise_scale": 0.0}
        )
        spatial = modules.get(
            "spatial_guidance",
            {
                "enabled": False, "inside": cfg["guidance_scale"],
                "outside": cfg["guidance_scale"], "use_schedule": False,
                "schedule": "cosine", "minimum": 4.0, "maximum": 7.5,
                "power": 2.0, "exp_k": 3.0, "sigmoid_k": 8.0,
                "sample_per_step": False,
            },
        )
        cama = modules.get("cama", {"enabled": False})
        ddim_noise = modules.get(
            "ddim_noise",
            {
                "enabled": False, "eta": 0.0,
                "anomaly_strength_min": 0.0,
                "anomaly_strength_max": 0.0,
                "anomaly_stop_step": 999999, "use_random_mask": False,
            },
        )
        mdap = modules.get("mdap", {"enabled": False})
        rda = modules.get("rda", {"enabled": False})
        carf = modules.get("carf", {"enabled": False})
        msdf = modules.get("msdf", {"enabled": False})

        parsed.prompt = cfg["prompt"]
        parsed.pipeline_mode = cfg["pipeline_mode"]
        parsed.num_inference_steps = cfg["num_inference_steps"]
        parsed.guidance_scale = cfg["guidance_scale"]
        parsed.blur_factor = cfg["blur_factor"]
        if parsed.seed is None:
            parsed.seed = cfg["seed"]
        parsed.device = cfg["device"]
        parsed.dtype = cfg["dtype"]
        parsed.normal_filter = cfg["normal_filter"]

        parsed.text_noise_scale = prompt_noise["noise_scale"] if prompt_noise["enabled"] else 0.0
        parsed.CAMA = bool(cama["enabled"])
        parsed.guidance_scale_inside = spatial["inside"] if spatial["enabled"] else cfg["guidance_scale"]
        parsed.guidance_scale_outside = spatial["outside"] if spatial["enabled"] else cfg["guidance_scale"]
        parsed.gsi_use_schedule = bool(spatial["enabled"] and spatial["use_schedule"])
        parsed.gsi_schedule = spatial["schedule"]
        parsed.gsi_min = spatial["minimum"]
        parsed.gsi_max = spatial["maximum"]
        parsed.gsi_power = spatial["power"]
        parsed.gsi_exp_k = spatial["exp_k"]
        parsed.gsi_sigmoid_k = spatial["sigmoid_k"]
        parsed.gsi_sample_per_step = bool(spatial["enabled"] and spatial["sample_per_step"])

        if ddim_noise["enabled"]:
            parsed.eta = ddim_noise["eta"]
            parsed.anomaly_strength_min = ddim_noise["anomaly_strength_min"]
            parsed.anomaly_strength_max = ddim_noise["anomaly_strength_max"]
            parsed.anomaly_stop_step = ddim_noise["anomaly_stop_step"]
            parsed.use_random_mask = ddim_noise["use_random_mask"]
        else:
            parsed.eta = 0.0
            parsed.eta_mask_min = 0.0
            parsed.eta_mask_max = 0.0
            parsed.anomaly_strength_min = 0.0
            parsed.anomaly_strength_max = 0.0
            parsed.use_random_mask = False

        parsed.experiment_name = cfg["experiment_name"]
        parsed.mdap = mdap
        parsed.rda = rda
        parsed.carf = carf
        parsed.msdf = msdf
        print(f"[CONFIG] experiment={parsed.experiment_name}, modules={enabled_modules(cfg) or ['baseline']}")
    else:
        if parsed.seed is None:
            parsed.seed = 42
        parsed.experiment_name = "legacy_cli"
        parsed.pipeline_mode = "custom"
        parsed.device = "cuda"
        parsed.dtype = "float16"
        parsed.mdap = {"enabled": False}
        parsed.rda = {"enabled": False}
        parsed.carf = {"enabled": False}
        parsed.msdf = {"enabled": False}
        parsed.normal_filter = {"enabled": False}
    return parsed


args = parse_args()


def extract_number_from_filename(fname):
    m = re.search(r"\d+", fname)
    return int(m.group()) if m else float("inf")


def monkey_patch_encode_prompt(pipe):
    old_encode = pipe.encode_prompt
    def new_encode_prompt(self, prompt, device, num_images_per_prompt,
                          do_classifier_free_guidance, negative_prompt=None,
                          prompt_embeds=None, negative_prompt_embeds=None,
                          lora_scale=None, clip_skip=None):
        prompt_embeds, neg_embeds = old_encode(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance,
            negative_prompt=negative_prompt, prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale, clip_skip=clip_skip)
        if getattr(self, "text_noise_scale", 0.0) > 0.0:
            s = self.text_noise_scale
            if do_classifier_free_guidance:
                half = prompt_embeds.shape[0] // 2
                uncond, cond = prompt_embeds[:half], prompt_embeds[half:]
                cond += torch.randn_like(cond) * s
                prompt_embeds = torch.cat([uncond, cond], 0)
            else:
                prompt_embeds += torch.randn_like(prompt_embeds) * s
        return prompt_embeds, neg_embeds
    pipe.encode_prompt = types.MethodType(new_encode_prompt, pipe)


def inpaint(pipe, image, prompt, mask=None, n_samples=4, device="cuda",
            blur_factor=0,
            anomaly_strength=0.0, anomaly_stop_step=999999, eta_mask_stop_step=999999,
            eta=0.0, eta_mask=0.0,
            guidance_scale_inside=None, guidance_scale_outside=None,
            # NEW: inside 
            gsi_use_schedule=False, gsi_schedule="linear",
            gsi_min=None, gsi_max=None,
            gsi_power=2.0, gsi_exp_k=3.0, gsi_sigmoid_k=8.0,
            gsi_sample_per_step=False, num_inference_steps=50,
            guidance_scale=7.5, mdap_prior_image=None,
            mdap_strength=0.0, mdap_schedule="cosine",
            mdap_end_fraction=0.7, rda=None,
            rda_reference_image=None, rda_reference_mask=None, carf=None,
            msdf=None, msdf_reference_image=None, msdf_reference_mask=None):
    from PIL import Image as PilImage
    if isinstance(image, str):
        image_pil = PilImage.open(image).convert("RGB")
    else:
        image_pil = image.convert("RGB") if image.mode != "RGB" else image
    if isinstance(mask, str):
        mask_pil = PilImage.open(mask).convert("RGB")
    else:
        mask_pil = mask.convert("RGB") if mask.mode != "RGB" else mask

    mask_pil = pipe.mask_processor.blur(mask_pil, blur_factor=blur_factor)

    if args.pipeline_mode == "standard":
        return pipe(
            prompt=[prompt] * n_samples,
            image=image_pil,
            mask_image=mask_pil,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            eta=eta,
        ).images

    return pipe(
        prompt=[prompt]*n_samples,
        image=image_pil,
        mask_image=mask_pil,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        mdap_prior_image=mdap_prior_image,
        mdap_strength=mdap_strength,
        mdap_schedule=mdap_schedule,
        mdap_end_fraction=mdap_end_fraction,
        rda_enabled=bool((rda or {}).get("enabled", False)),
        rda_path=(rda or {}).get("path"),
        rda_reference_image=rda_reference_image,
        rda_reference_mask=rda_reference_mask,
        carf_enabled=bool((carf or {}).get("enabled", False)),
        carf_path=(carf or {}).get("path"),
        msdf_enabled=bool((msdf or {}).get("enabled", False)),
        msdf_path=(msdf or {}).get("path"),
        msdf_reference_image=msdf_reference_image,
        msdf_reference_mask=msdf_reference_mask,
        anomaly_strength=anomaly_strength,
        anomaly_stop_step=anomaly_stop_step,
        eta_mask_stop_step=eta_mask_stop_step,
        use_random_mask=args.use_random_mask,
        eta=eta,
        eta_mask=eta_mask,

        guidance_scale_inside=guidance_scale_inside,
        guidance_scale_outside=guidance_scale_outside,

        gsi_use_schedule=gsi_use_schedule,
        gsi_schedule=gsi_schedule,
        guidance_scale_inside_min=gsi_min,
        guidance_scale_inside_max=gsi_max,
        gsi_power=gsi_power,
        gsi_exp_k=gsi_exp_k,
        gsi_sigmoid_k=gsi_sigmoid_k,
        gsi_sample_per_step=gsi_sample_per_step,
    ).images


def get_valid_normal_images(img_dir, filter_config):
    paths = [
        os.path.join(img_dir, name)
        for name in sorted(os.listdir(img_dir))
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not paths:
        raise ValueError(f"No images in {img_dir}")
    if not filter_config.get("enabled", False):
        return paths

    black_threshold = int(filter_config.get("black_threshold", 20))
    min_mean = float(filter_config.get("min_mean_luminance", 30.0))
    min_ratio = float(filter_config.get("min_nonblack_ratio", 0.60))
    valid = []
    rejected = []
    for path in paths:
        # A small thumbnail is sufficient for background screening and avoids
        # loading all full-resolution normal images into memory.
        luminance = np.asarray(
            Image.open(path).convert("L").resize((64, 64)),
            dtype=np.float32,
        )
        mean_luminance = float(luminance.mean())
        nonblack_ratio = float((luminance > black_threshold).mean())
        if mean_luminance >= min_mean and nonblack_ratio >= min_ratio:
            valid.append(path)
        else:
            rejected.append((os.path.basename(path), mean_luminance, nonblack_ratio))
    print(
        f"[NORMAL_FILTER] accepted={len(valid)}/{len(paths)}, "
        f"rejected={len(rejected)}, min_mean={min_mean}, min_ratio={min_ratio}"
    )
    if rejected:
        print(
            "[NORMAL_FILTER] rejected examples: "
            + ", ".join(
                f"{name}(mean={mean:.1f},ratio={ratio:.2f})"
                for name, mean, ratio in rejected[:5]
            )
        )
    if not valid:
        raise ValueError(
            "Normal-image filter rejected every image. "
            "Lower normal_filter thresholds in the config."
        )
    return valid


def get_random_image(valid_images):
    return random.choice(valid_images)


def load_object_mask(category, normal_img_path, normal_masks_dir):
    cat_dir = os.path.join(normal_masks_dir, category)
    base = os.path.splitext(os.path.basename(normal_img_path))[0]

    cand_dirs = [
        os.path.join(cat_dir, "train", "masks"),
        os.path.join(cat_dir, "masks"),
        cat_dir,
    ]

    candidates = []
    for d in cand_dirs:
        if not os.path.isdir(d):
            continue
        candidates.append(os.path.join(d, f"{base}_mask.png"))
        candidates.extend(sorted(glob.glob(os.path.join(d, f"{base}_mask_*.png"))))
        candidates.append(os.path.join(d, f"{base}.png"))
        candidates.append(os.path.join(d, "mask.png"))
    if not candidates:
        print(f"[load_object_mask] No candidate mask files found for: {normal_img_path}")

    seen = set()
    ordered_paths = []
    for p in candidates:
        if p not in seen and os.path.exists(p):
            seen.add(p)
            ordered_paths.append(p)

    for mask_path in ordered_paths:
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            return (m > 127).astype(np.uint8)

    return None

def debug_save_masks(original_mask_bin, min_x, min_y, max_x, max_y,
                     shifted_mask_bin, debug_save_path):
    H, W = original_mask_bin.shape
    left = np.zeros((H, W, 3), np.uint8)
    left[original_mask_bin > 0] = (255, 255, 255)
    n_lbl, lbl_map, stats, _ = cv2.connectedComponentsWithStats(original_mask_bin, 8)
    for lbl in range(1, n_lbl):
        x, y, bw, bh, _ = stats[lbl]
        if bw and bh:
            cv2.rectangle(left, (x, y), (x + bw - 1, y + bh - 1), (0, 0, 255), 2)

    right = np.zeros((H, W, 3), np.uint8)
    right[shifted_mask_bin > 0] = (255, 255, 255)
    cv2.imwrite(debug_save_path, np.concatenate([left, right], axis=1))

###############################################################################
# 3) CAMA: Context-Aware Mask Alignment
###############################################################################
def CAMA(
    class_val,
    code_mask_bin,
    obj_mask_np,
    normal_image_path,
    category,
    defect_class,
    defect_data,
    match_data,
    debug_save_dir=None,
    debug_name=None,
):
    """
    Return (final_mask, first_best_x, first_best_y, is_shifted)
    """
    H, W = code_mask_bin.shape
    base_normal = os.path.basename(normal_image_path)

    # ───────── ① Collect coordinates per defect_img ─────────
    by_defect = {}
    for it in match_data.get(category, {}).get(defect_class, []):
        if it["normal_img"] != base_normal:
            continue
        by_defect.setdefault(it["defect_img"], []).append((it["best_x"], it["best_y"]))

    if not by_defect:
        fallback = cv2.bitwise_and(code_mask_bin, obj_mask_np) if class_val == 0 else code_mask_bin
        if debug_save_dir and debug_name:
            debug_save_masks(
                code_mask_bin, 0, 0, 0, 0, fallback,
                os.path.join(debug_save_dir, f"{debug_name}_fallback.jpg")
            )
        return fallback, -1, -1, False

    # ───────── ② Randomly choose one defect_img ─────────
    chosen_defect, coords_all = random.choice(list(by_defect.items()))
    # coords_all: [(best_x, best_y), …]

    # ───────── ③ Extract components ─────────
    n_lbl, lbl_map, stats, _ = cv2.connectedComponentsWithStats(code_mask_bin, 8)
    comps = list(range(1, n_lbl))  # 0 is background
    if not comps:  # mask is empty
        return code_mask_bin, -1, -1, False

    n_comp = len(comps)
    n_coords = len(coords_all)

    # ───────── ④ Match coordinate list size to the number of components ─────────
    def rand_point_inside(mask):
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            # If obj_mask is empty, sample from the whole image
            return random.randint(0, W - 1), random.randint(0, H - 1)
        idx = random.randrange(ys.size)
        return int(xs[idx]), int(ys[idx])

    if n_coords >= n_comp:
        coords = random.sample(coords_all, n_comp)
    else:
        coords = list(coords_all)
        # Fill the shortage with random coordinates
        for _ in range(n_comp - n_coords):
            rx, ry = rand_point_inside(obj_mask_np if class_val == 0 else np.ones_like(obj_mask_np))
            coords.append((rx, ry))

    # Now comps and coords have the same length (n_comp)
    target_pairs = list(zip(comps, coords))

    shifted = np.zeros_like(code_mask_bin, np.uint8)
    for lbl, (best_x, best_y) in target_pairs:
        best_x = int(round(best_x))
        best_y = int(round(best_y))

        # If the component already contains the best point, keep it as is
        if 0 <= best_x < W and 0 <= best_y < H and code_mask_bin[best_y, best_x]:
            shifted |= (lbl_map == lbl).astype(np.uint8)
            continue

        x, y, bw, bh, _ = stats[lbl]
        if bw == 0 or bh == 0:
            continue

        crop = (lbl_map[y:y + bh, x:x + bw] == lbl).astype(np.uint8)

        # Simple translation without center correction
        tx = best_x - bw // 2
        ty = best_y - bh // 2
        for r in range(bh):
            for c in range(bw):
                if crop[r, c]:
                    yy = ty + r
                    xx = tx + c
                    if 0 <= xx < W and 0 <= yy < H:
                        shifted[yy, xx] = 1

    final = cv2.bitwise_and(shifted, obj_mask_np) if class_val == 0 else shifted

    if debug_save_dir and debug_name:
        ys, xs = np.where(code_mask_bin)
        debug_save_masks(
            code_mask_bin,
            xs.min() if xs.size else 0, ys.min() if ys.size else 0,
            xs.max() if xs.size else 0, ys.max() if ys.size else 0,
            final,
            os.path.join(debug_save_dir, f"{debug_name}.jpg")
        )

    first_best = coords[0]
    return final, first_best[0], first_best[1], True



def main():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    defect_data = {}
    if args.defect_json:
        with open(args.defect_json, "r", encoding="utf-8") as f:
            defect_data = json.load(f)
    match_data = {}
    if args.CAMA:
        if not args.defect_json:
            raise ValueError("--defect_json is required when CAMA is enabled")
        if not args.match_json:
            raise ValueError("--match_json is required when CAMA is enabled")
        with open(args.match_json, "r", encoding="utf-8") as f:
            match_data = json.load(f)

    if args.dataset_type == "mvtec_3d":
        default_cats = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel',
                        'foam', 'peach', 'potato', 'rope', 'tire']
    elif args.dataset_type == "visa":
        default_cats = ['candle','capsules','cashew','chewinggum','fryum','macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4','pipe_fryum']
    elif args.dataset_type == "DAGM":
        default_cats = ['Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10']
    else:        
        default_cats = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                        'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    categories = args.categories or default_cats

    for category in categories:
        device = args.device

        if args.dataset_type == "mvtec_3d":
            gt_path     = os.path.join(args.base_dir, category, "test")
            normal_root = os.path.join(args.base_dir, category, "train", "good", "rgb")
        else:
            gt_path     = os.path.join(args.base_dir, category, "ground_truth")
            normal_root = os.path.join(args.base_dir, category, "train", "good")

        if not os.path.exists(gt_path):
            print(f"[WARN] ground_truth path not found: {gt_path}")
            continue
        if not os.path.exists(normal_root):
            print(f"[WARN] normal image path not found: {normal_root}")
            continue
        valid_normal_images = get_valid_normal_images(
            normal_root, args.normal_filter
        )

        defect_classes = [d for d in os.listdir(gt_path)
                          if os.path.isdir(os.path.join(gt_path, d))
                          and d != "good"]
        for defect_class in defect_classes:
            if defect_data and defect_class not in defect_data.get(category, {}):
                print(f"[WARN] {defect_class} not in defect_json → skip")
                continue
            class_val = defect_data.get(category, {}).get(defect_class, 0)
            print(f"Category={category}, Defect={defect_class}, class_val={class_val}")

            mask_root = (os.path.join(args.mask_dir, category, defect_class)
                         if args.mask_dir else os.path.join(gt_path, defect_class))
            if not os.path.exists(mask_root):
                print(f"[WARN] {mask_root} not found → skip")
                continue

            ckpt_root = os.path.join(args.model_ckpt_root, category, defect_class)
            if not os.path.exists(ckpt_root):
                print(f"[WARN] checkpoint absent: {ckpt_root} → skip")
                continue

            dtype = getattr(torch, args.dtype)
            if args.pipeline_mode == "standard":
                pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    ckpt_root, torch_dtype=dtype
                )
                pipe.scheduler = StandardDDIMScheduler.from_pretrained(
                    args.ddim_scheduler_root
                )
            else:
                pipe = StableDiffusionInpaintPipeline_dynamic.from_pretrained(
                    ckpt_root, torch_dtype=dtype
                )
                pipe.scheduler = DDIMScheduler.from_pretrained(args.ddim_scheduler_root)

                sched = pipe.scheduler
                new_cfg = dict(sched.config)
                new_cfg.update({
                    "eta_mask_use_schedule": bool(args.eta_mask_use_schedule),
                    "eta_mask_schedule": args.eta_mask_schedule,
                    "eta_mask_min": float(args.eta_mask_min),
                    "eta_mask_max": float(args.eta_mask_max),
                    "eta_mask_power": float(args.eta_mask_power),
                    "eta_mask_exp_k": float(args.eta_mask_exp_k),
                    "eta_mask_sigmoid_k": float(args.eta_mask_sigmoid_k),
                    "eta_mask_guard": args.eta_mask_guard,
                    "eta_mask_guard_margin": float(args.eta_mask_guard_margin),
                    "eta_mask_segmented": bool(args.eta_mask_segmented),
                    "eta_mask_stop_step": int(args.eta_mask_stop_step),
                })
                sched._internal_dict = FrozenDict(new_cfg)

            if args.text_noise_scale > 0:
                pipe.text_noise_scale = args.text_noise_scale
                monkey_patch_encode_prompt(pipe)
            print(f"[PIPELINE] mode={args.pipeline_mode}, class={pipe.__class__.__name__}")
            pipe.to(device)

            mask_imgs = sorted(
                (os.path.join(mask_root, f) for f in os.listdir(mask_root)
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))),
                key=lambda x: extract_number_from_filename(os.path.basename(x)))

            if args.config:
                suffix = args.experiment_name
            else:
                suffix = (f"{args.experiment_name}_noise_{args.text_noise_scale}_"
                          f"guidance_scale_{args.gsi_min}_{args.gsi_max}_"
                          + ("align" if args.CAMA else "no_align"))
            output_base = os.path.normpath(args.output_name)
            # Accept both --output_name results and --output_name results/baseline
            # without creating results/baseline/baseline.
            if os.path.basename(output_base) == suffix:
                save_root = os.path.join(output_base, category, defect_class)
            else:
                save_root = os.path.join(output_base, suffix, category, defect_class)
            img_dir  = os.makedirs(os.path.join(save_root, "image"), exist_ok=True) or os.path.join(save_root, "image")
            norm_dir = os.makedirs(os.path.join(save_root, "normal"), exist_ok=True) or os.path.join(save_root, "normal")
            msk_dir  = os.makedirs(os.path.join(save_root, "masks"), exist_ok=True) or os.path.join(save_root, "masks")
            dbg_dir  = os.makedirs(os.path.join(save_root, "debug_mask"), exist_ok=True) or os.path.join(save_root, "debug_mask")
            mdap_dir = os.makedirs(os.path.join(save_root, "mdap"), exist_ok=True) or os.path.join(save_root, "mdap")

            for idx, mask_path in enumerate(mask_imgs):
                # -----------------------------
                # 1) normal image 512 resize + center crop
                # -----------------------------
                if args.use_paired_normal:
                    paired_dir = os.path.join(
                        args.base_dir, category, "paired_normal", defect_class
                    )
                    mask_stem = os.path.splitext(os.path.basename(mask_path))[0]
                    paired_candidates = [
                        os.path.join(paired_dir, mask_stem + extension)
                        for extension in (".jpg", ".jpeg", ".png", ".bmp")
                    ]
                    normal_img_path = next(
                        (path for path in paired_candidates if os.path.isfile(path)),
                        None,
                    )
                    if normal_img_path is None:
                        raise FileNotFoundError(
                            "Paired normal image not found for mask "
                            f"{mask_path}. Expected the same stem under {paired_dir}."
                        )
                else:
                    normal_img_path = get_random_image(valid_normal_images)
                normal_img = Image.open(normal_img_path).convert("RGB")
                normal_img = image_resize_center_crop(normal_img)

                # -----------------------------
                # 2) aug mask 512 resize + center crop
                # -----------------------------
                raw_mask_pil = Image.open(mask_path).convert("L")
                raw_mask_pil = mask_resize_center_crop(raw_mask_pil)
                mask_np = (np.array(raw_mask_pil) > 127).astype(np.uint8)

                # -----------------------------
                # 3) object mask 512 resize + center crop (CAMA용)
                # -----------------------------
                obj_mask = None
                if args.CAMA:
                    obj_mask = load_object_mask(category, normal_img_path, args.normal_masks)
                    if obj_mask is None:
                        obj_mask = np.ones_like(mask_np, np.uint8)
                    else:
                        obj_mask_pil = Image.fromarray((obj_mask * 255).astype(np.uint8))
                        obj_mask_pil = mask_resize_center_crop(obj_mask_pil)
                        obj_mask = (np.array(obj_mask_pil) > 127).astype(np.uint8)

                    if obj_mask.shape != mask_np.shape:
                        obj_mask = cv2.resize(obj_mask, mask_np.shape[::-1], interpolation=cv2.INTER_NEAREST)
                        obj_mask = (obj_mask > 0).astype(np.uint8)

                # -----------------------------
                # 4) CAMA
                # -----------------------------
                if args.CAMA:
                    final_mask, *_ = CAMA(
                        class_val, mask_np, obj_mask, normal_img_path,
                        category, defect_class, defect_data, match_data,
                        debug_save_dir=dbg_dir, debug_name=f"{idx}",
                    )
                else:
                    final_mask = mask_np

                final_mask_pil = Image.fromarray((final_mask * 255).astype(np.uint8))

                mdap_prior_image = None
                mdap_descriptor = None
                if args.mdap.get("enabled", False):
                    ref_image_path, ref_mask_path = find_reference_pair(
                        args.reference_base_dir or args.base_dir,
                        category,
                        defect_class,
                        idx,
                    )
                    ref_image = image_resize_center_crop(
                        Image.open(ref_image_path).convert("RGB")
                    )
                    ref_mask = mask_resize_center_crop(
                        Image.open(ref_mask_path).convert("L")
                    )
                    defect_mode = infer_defect_mode(
                        defect_class, args.mdap.get("modes")
                    )
                    mdap_prior_image, mdap_soft_mask, descriptor = build_mdap_prior(
                        ref_image,
                        ref_mask,
                        normal_img,
                        final_mask_pil,
                        defect_mode,
                        ring_width=int(args.mdap["ring_width"]),
                        threshold_sigma=float(args.mdap["threshold_sigma"]),
                        blur_sigma=float(args.mdap["blur_sigma"]),
                        contrast_min=float(args.mdap["contrast_min"]),
                        contrast_max=float(args.mdap["contrast_max"]),
                        amplitude=float(args.mdap["amplitude"]),
                    )
                    mdap_descriptor = descriptor.to_dict()
                    mdap_prior_image.save(os.path.join(mdap_dir, f"{idx}_prior.png"))
                    mdap_soft_mask.save(os.path.join(mdap_dir, f"{idx}_soft_mask.png"))
                    with open(os.path.join(mdap_dir, f"{idx}_descriptor.json"), "w", encoding="utf-8") as handle:
                        json.dump(mdap_descriptor, handle, ensure_ascii=False, indent=2)

                a_strength = random.uniform(args.anomaly_strength_min, args.anomaly_strength_max)
                eta_strength = random.uniform(args.eta_mask_min, args.eta_mask_max)

                rda_config = dict(args.rda)
                carf_config = dict(args.carf)
                msdf_config = dict(args.msdf)
                rda_reference_image = None
                rda_reference_mask = None
                if rda_config.get("enabled", False):
                    rda_config["path"] = os.path.join(
                        rda_config.get("root", args.model_ckpt_root),
                        category,
                        defect_class,
                        rda_config.get("filename", "rda.pt"),
                    )
                    reference_image_path, reference_mask_path = find_reference_pair(
                        args.reference_base_dir or args.base_dir,
                        category,
                        defect_class,
                        idx,
                    )
                    rda_reference_image = image_resize_center_crop(
                        Image.open(reference_image_path).convert("RGB")
                    )
                    rda_reference_mask = mask_resize_center_crop(
                        Image.open(reference_mask_path).convert("L")
                    )
                if carf_config.get("enabled", False):
                    carf_config["path"] = os.path.join(
                        carf_config.get("root", args.model_ckpt_root),
                        category,
                        defect_class,
                        carf_config.get("filename", "carf.pt"),
                    )
                msdf_reference_image = None
                msdf_reference_mask = None
                if msdf_config.get("enabled", False):
                    msdf_config["path"] = os.path.join(
                        msdf_config.get("root", args.model_ckpt_root),
                        category,
                        defect_class,
                        msdf_config.get("filename", "msdf.pt"),
                    )
                    reference_image_path, reference_mask_path = find_reference_pair(
                        args.reference_base_dir or args.base_dir,
                        category,
                        defect_class,
                        idx,
                    )
                    msdf_reference_image = image_resize_center_crop(
                        Image.open(reference_image_path).convert("RGB")
                    )
                    msdf_reference_mask = mask_resize_center_crop(
                        Image.open(reference_mask_path).convert("L")
                    )

                imgs = inpaint(
                    pipe, normal_img,
                    prompt=args.prompt,
                    mask=final_mask_pil, n_samples=1, device=device,
                    blur_factor=args.blur_factor,
                    anomaly_strength=a_strength,
                    anomaly_stop_step=args.anomaly_stop_step,
                    eta_mask_stop_step=args.eta_mask_stop_step,
                    eta=args.eta,
                    eta_mask=eta_strength,

                    guidance_scale_inside=(args.guidance_scale_inside if args.guidance_scale_inside is not None else 3.0),
                    guidance_scale_outside=(args.guidance_scale_outside if args.guidance_scale_outside is not None else 7.5),

                    # NEW: inside
                    gsi_use_schedule=args.gsi_use_schedule,
                    gsi_schedule=args.gsi_schedule,
                    gsi_min=args.gsi_min,
                    gsi_max=args.gsi_max,
                    gsi_power=args.gsi_power,
                    gsi_exp_k=args.gsi_exp_k,
                    gsi_sigmoid_k=args.gsi_sigmoid_k,
                    gsi_sample_per_step=args.gsi_sample_per_step,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    mdap_prior_image=mdap_prior_image,
                    mdap_strength=float(args.mdap.get("strength", 0.0)),
                    mdap_schedule=args.mdap.get("schedule", "cosine"),
                    mdap_end_fraction=float(args.mdap.get("end_fraction", 0.7)),
                    rda=rda_config,
                    rda_reference_image=rda_reference_image,
                    rda_reference_mask=rda_reference_mask,
                    carf=carf_config,
                    msdf=msdf_config,
                    msdf_reference_image=msdf_reference_image,
                    msdf_reference_mask=msdf_reference_mask,
                )

                out = f"{idx}.jpg"
                imgs[0].save(os.path.join(img_dir,  out))
                normal_img = normal_img.convert("RGB")
                normal_img.save(os.path.join(norm_dir, out), format="JPEG")
                final_mask_pil.convert("RGB").save(os.path.join(msk_dir, out))
                print(f"Saved {out}")


if __name__ == "__main__":
    main()
