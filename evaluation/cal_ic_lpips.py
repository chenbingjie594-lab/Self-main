import argparse
import random
import torch
import torch.nn as nn
from torchvision import utils
from tqdm import tqdm
import sys
import csv
import lpips
from torchvision import transforms, utils
from torch.utils import data
import os
from PIL import Image
import numpy as np

lpips_fn = lpips.LPIPS(net='vgg').cuda()
preprocess = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
device = 'cuda'

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')


def image_files(folder):
    return sorted(
        os.path.join(folder, name) for name in os.listdir(folder)
        if name.lower().endswith(IMAGE_EXTENSIONS)
        and os.path.isfile(os.path.join(folder, name))
    )

def ic_lpips(mvtec_path, gen_path, sample_name, anomaly_name):
    print(sample_name, anomaly_name)
    tar_path = '%s/%s/%s/image' % (gen_path, sample_name, anomaly_name)
    ori_path = '%s/%s/test/%s' % (mvtec_path, sample_name, anomaly_name)

    with torch.no_grad():
        original_files = image_files(ori_path)
        generated_files = image_files(tar_path)
        l = len(original_files)
        if l == 0 or not generated_files:
            raise ValueError(f"No evaluation images: real={ori_path}, generated={tar_path}")
        avg_dist = torch.zeros([l, ])
        input_tensors1 = []
        clusters = [[] for _ in range(l)]

        # Load original reference images
        for k, input1_path in enumerate(original_files):
            input_image1 = Image.open(input1_path).convert('RGB')
            input_tensor1 = preprocess(input_image1)
            input_tensor1 = input_tensor1.to(device)
            input_tensors1.append(input_tensor1)

        # Assign each generated image to the closest original image
        for input2_path in generated_files:
            min_dist = float('inf')
            input_image2 = Image.open(input2_path).convert('RGB')
            input_tensor2 = preprocess(input_image2).to(device)
            for k in range(l):
                dist = lpips_fn(input_tensors1[k], input_tensor2)
                if dist <= min_dist:
                    max_ind = k
                    min_dist = dist
            clusters[max_ind].append(input2_path)

        cluster_size = 50

        # Compute LPIPS distance within each cluster
        for k in range(l):
            print(k)
            files_list = clusters[k]
            random.shuffle(files_list)
            files_list = files_list[:cluster_size]
            dists = []
            for i in range(len(files_list)):
                for j in range(i + 1, len(files_list)):
                    input1_path = files_list[i]
                    input2_path = files_list[j]

                    input_image1 = Image.open(input1_path)
                    input_image2 = Image.open(input2_path)

                    input_tensor1 = preprocess(input_image1).to(device)
                    input_tensor2 = preprocess(input_image2).to(device)

                    dist = lpips_fn(input_tensor1, input_tensor2)
                    dists.append(dist)

            if dists:
                avg_dist[k] = torch.stack([d.reshape(()) for d in dists]).mean().cpu()
            else:
                avg_dist[k] = torch.nan

        return avg_dist[~torch.isnan(avg_dist)].mean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvtec_path", help="Path to MVTec dataset")
    parser.add_argument("--gen_path", help="Path to generated dataset")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Categories to evaluate; defaults to folders under gen_path")
    args = parser.parse_args()

    sample_names = args.categories or sorted(
        name for name in os.listdir(args.gen_path)
        if os.path.isdir(os.path.join(args.gen_path, name))
    )

    for sample_name in sample_names:
        dis = 0
        cnt = 0
        for anomaly_name in sorted(os.listdir('%s/%s' % (args.gen_path, sample_name))):
            if not os.path.isdir(os.path.join(args.gen_path, sample_name, anomaly_name)):
                continue
            dis += ic_lpips(args.mvtec_path, args.gen_path, sample_name, anomaly_name)
            cnt += 1
        if cnt == 0:
            print(f"[WARN] No defect classes found for {sample_name}")
            continue
        with open(f"{args.gen_path}_results.csv", "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([sample_name, str(float(dis / cnt))])
