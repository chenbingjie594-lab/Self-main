"""Class-wise KID on fixed-size mask-centred crops."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from compare_kid import (
    IMAGE_SUFFIXES,
    InceptionFeatures,
    image_files,
    kid_distribution,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_path", required=True)
    parser.add_argument("--generated", nargs=2, action="append", required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--kid_seed", type=int, default=1234)
    parser.add_argument("--num_subsets", type=int, default=100)
    parser.add_argument("--max_subset_size", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def matching_file(folder, stem):
    for candidate_stem in (stem, f"{stem}_mask"):
        for suffix in IMAGE_SUFFIXES:
            candidate = Path(folder) / f"{candidate_stem}{suffix}"
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"No file for stem={stem} under {folder}")


def held_out_pairs(image_paths, mask_dir, train_ratio, seed):
    pairs = [(path, matching_file(mask_dir, path.stem)) for path in image_paths]
    if train_ratio is None:
        return pairs
    shuffled = pairs.copy()
    random.Random(seed).shuffle(shuffled)
    count = max(1, min(len(pairs) - 1, int(round(len(pairs) * train_ratio))))
    return sorted(shuffled[count:], key=lambda pair: pair[0].name)


def crop_around_mask(image, mask, crop_size):
    mask_array = np.asarray(mask.convert("L")) > 127
    ys, xs = np.where(mask_array)
    if not len(xs):
        raise ValueError("Encountered empty mask")
    cx = (int(xs.min()) + int(xs.max())) / 2.0
    cy = (int(ys.min()) + int(ys.max())) / 2.0
    side = min(crop_size, image.width, image.height)
    left = max(0, min(image.width - side, int(round(cx - side / 2))))
    top = max(0, min(image.height - side, int(round(cy - side / 2))))
    return image.crop((left, top, left + side, top + side))


def extract_features(pairs, model, device, batch_size, crop_size):
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])
    features = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = []
            for image_path, mask_path in pairs[start:start + batch_size]:
                image = Image.open(image_path).convert("RGB")
                mask = Image.open(mask_path)
                batch.append(preprocess(crop_around_mask(image, mask, crop_size)))
            features.append(model(torch.stack(batch).to(device)).cpu().numpy())
    return np.concatenate(features).astype(np.float64)


def generated_pairs(root, category, defect):
    image_dir = root / category / defect / "image"
    mask_dir = root / category / defect / "masks"
    return [(path, matching_file(mask_dir, path.stem)) for path in image_files(image_dir)]


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionFeatures().to(device)
    roots = [(label, Path(path)) for label, path in args.generated]
    overall = {label: [] for label, _ in roots}
    real_root = Path(args.real_path)

    for category_index, category in enumerate(args.categories):
        test_root = real_root / category / "test"
        defects = sorted(p.name for p in test_root.iterdir() if p.is_dir() and p.name != "good")
        for defect_index, defect in enumerate(defects):
            split_seed = args.split_seed + category_index * 100003 + defect_index * 1009
            real_pairs = held_out_pairs(
                image_files(test_root / defect),
                real_root / category / "ground_truth" / defect,
                args.train_ratio,
                split_seed,
            )
            real_features = extract_features(
                real_pairs, model, device, args.batch_size, args.crop_size
            )
            print(f"{category}/{defect}: local real={len(real_pairs)}")
            for label, root in roots:
                pairs = generated_pairs(root, category, defect)
                generated_features = extract_features(
                    pairs, model, device, args.batch_size, args.crop_size
                )
                subset = min(args.max_subset_size, len(real_features), len(generated_features))
                if subset < 2:
                    raise ValueError(f"Too few local crops for {category}/{defect}/{label}")
                rng = np.random.default_rng(
                    args.kid_seed + category_index * 100003 + defect_index * 1009
                )
                scores = kid_distribution(
                    real_features, generated_features, rng, subset, args.num_subsets
                )
                overall[label].append(scores.mean())
                print(
                    f"  {label}: Local-KIDx1000={scores.mean():.4f} +/- "
                    f"{scores.std(ddof=1):.4f} (subset={subset})"
                )
    print("Overall macro average:")
    for label, values in overall.items():
        values = np.asarray(values)
        print(f"  {label}: Local-KIDx1000={values.mean():.4f}")


if __name__ == "__main__":
    main()
