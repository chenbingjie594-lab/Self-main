"""Reproducible class-wise KID comparison for generated defect datasets."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_path", required=True)
    parser.add_argument(
        "--generated",
        nargs=2,
        action="append",
        metavar=("LABEL", "PATH"),
        required=True,
    )
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--kid_seed", type=int, default=1234)
    parser.add_argument("--num_subsets", type=int, default=100)
    parser.add_argument("--max_subset_size", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def image_files(folder):
    folder = Path(folder)
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def held_out_files(files, train_ratio, seed):
    if train_ratio is None:
        return files
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    train_count = max(1, min(len(files) - 1, int(round(len(files) * train_ratio))))
    return sorted(shuffled[train_count:])


class InceptionFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()
        model.dropout = torch.nn.Identity()
        self.model = model.eval()

    def forward(self, images):
        return self.model(images)


def extract_features(paths, model, device, batch_size):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    features = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch = [
                transform(Image.open(path).convert("RGB"))
                for path in paths[start : start + batch_size]
            ]
            features.append(model(torch.stack(batch).to(device)).cpu().numpy())
    return np.concatenate(features, axis=0).astype(np.float64)


def polynomial_kernel(first, second):
    feature_dim = first.shape[1]
    return (first @ second.T / feature_dim + 1.0) ** 3


def unbiased_mmd(first, second):
    first_kernel = polynomial_kernel(first, first)
    second_kernel = polynomial_kernel(second, second)
    cross_kernel = polynomial_kernel(first, second)
    first_count = len(first)
    second_count = len(second)
    first_term = (first_kernel.sum() - np.trace(first_kernel)) / (
        first_count * (first_count - 1)
    )
    second_term = (second_kernel.sum() - np.trace(second_kernel)) / (
        second_count * (second_count - 1)
    )
    return first_term + second_term - 2.0 * cross_kernel.mean()


def kid_distribution(real, generated, rng, subset_size, num_subsets):
    scores = []
    for _ in range(num_subsets):
        real_indices = rng.choice(len(real), subset_size, replace=False)
        generated_indices = rng.choice(len(generated), subset_size, replace=False)
        scores.append(unbiased_mmd(real[real_indices], generated[generated_indices]))
    return np.asarray(scores) * 1000.0


def main():
    args = parse_args()
    real_root = Path(args.real_path)
    generated_roots = [(label, Path(path)) for label, path in args.generated]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionFeatures().to(device)
    all_scores = {label: [] for label, _ in generated_roots}

    for category_index, category in enumerate(args.categories):
        real_test_root = real_root / category / "test"
        defect_names = sorted(
            path.name for path in real_test_root.iterdir()
            if path.is_dir() and path.name != "good"
        )
        for defect_index, defect in enumerate(defect_names):
            split_seed = args.split_seed + category_index * 100003 + defect_index * 1009
            real_paths = held_out_files(
                image_files(real_test_root / defect), args.train_ratio, split_seed
            )
            real_features = extract_features(
                real_paths, model, device, args.batch_size
            )
            print(f"{category}/{defect}: held-out real={len(real_paths)}")
            for label, generated_root in generated_roots:
                generated_paths = image_files(
                    generated_root / category / defect / "image"
                )
                generated_features = extract_features(
                    generated_paths, model, device, args.batch_size
                )
                subset_size = min(
                    args.max_subset_size,
                    len(real_features),
                    len(generated_features),
                )
                if subset_size < 2:
                    raise ValueError(f"Too few images for {category}/{defect}/{label}")
                rng = np.random.default_rng(
                    args.kid_seed + category_index * 100003 + defect_index * 1009
                )
                scores = kid_distribution(
                    real_features,
                    generated_features,
                    rng,
                    subset_size,
                    args.num_subsets,
                )
                all_scores[label].append(scores)
                print(
                    f"  {label}: KIDx1000={scores.mean():.4f} +/- "
                    f"{scores.std(ddof=1):.4f} (subset={subset_size})"
                )

    print("Overall macro average:")
    for label, distributions in all_scores.items():
        class_means = np.asarray([scores.mean() for scores in distributions])
        print(
            f"  {label}: KIDx1000={class_means.mean():.4f} "
            f"(class std={class_means.std(ddof=1) if len(class_means) > 1 else 0.0:.4f})"
        )


if __name__ == "__main__":
    main()
