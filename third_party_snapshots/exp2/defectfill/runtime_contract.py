"""Raw generation I/O/RNG adapter; no ranking, filtering, or metric computation."""
import hashlib
import json

BANKS = {"A": 42, "B": 3407, "C": 2026}


def slot_seed(slot_id, bank):
    if not isinstance(slot_id, str) or not slot_id or bank not in BANKS:
        raise ValueError("DEFECTFILL_SLOT_OR_BANK_INVALID")
    payload = json.dumps(dict(method="defectfill", slot_id=slot_id, bank_seed=BANKS[bank]),
                         ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def raw_generate(model, image, mask, prompt, slot_id, bank, steps=50, guidance_scale=7.5):
    import torch
    if tuple(image.shape) != (1, 3, 512, 512) or tuple(mask.shape) != (1, 1, 512, 512):
        raise ValueError("DEFECTFILL_V2_RAW_SHAPE_INVALID: exactly one512 crop and one mask required")
    generator = torch.Generator(device=image.device).manual_seed(slot_seed(slot_id, bank))
    with torch.no_grad():
        result = model.generate(image=image, mask=mask, prompt=prompt,
            num_inference_steps=steps, guidance_scale=guidance_scale, generator=generator)
    if tuple(result.shape) != (1, 3, 512, 512) or not bool(torch.isfinite(result).all()):
        raise RuntimeError("DEFECTFILL_RAW_OUTPUT_INVALID: stop this slot, no reroll")
    return result
