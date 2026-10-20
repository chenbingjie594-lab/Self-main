"""Explicit TRAIN manifest adapter; never discover or split a directory."""
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from exp2_v2_data import load_training_records


class FrozenV2Dataset(Dataset):
    def __init__(self, manifest, class_name=None, repeats=20):
        self.records = load_training_records(manifest, class_name)
        self.repeats = repeats

    def __len__(self):
        return len(self.records) * self.repeats

    def __getitem__(self, index):
        from ldm.data.personalized import imagenet_templates_small
        row = self.records[index % len(self.records)]
        with Image.open(row["image_path"]) as image:
            image = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        with Image.open(row["mask_path"]) as mask:
            mask = (np.asarray(mask.convert("L"), dtype=np.float32) >= 128).astype(np.float32)
        return dict(image=image, mask=mask,
                    caption=random.choice(imagenet_templates_small).format("*"),
                    name="Plastic_Bomo+"+row["class"])
