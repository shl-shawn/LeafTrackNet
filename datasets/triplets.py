import os
import random
from typing import Dict, List, Tuple
from PIL import Image
from torch.utils.data import Dataset


class _BaseTripletDataset(Dataset):
    """Common scanner for MOT-style folders.
    Expects per-plant structure:
      {root}/{plant_id}/gt/gt.txt lines: frame,id,x,y,w,h,*,*
      {root}/{plant_id}/img/{frame:08d}.jpg
    """
    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        self.samples_by_leaf: Dict[Tuple[str, int], List[dict]] = {}

        plant_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        for plant_id in plant_dirs:
            gt_path = os.path.join(root, plant_id, "gt", "gt.txt")
            img_dir = os.path.join(root, plant_id, "img")
            if not os.path.exists(gt_path):
                continue
            with open(gt_path, "r") as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 6:
                        continue
                    frame   = int(parts[0])
                    leaf_id = int(parts[1])
                    x, y, w, h = map(float, parts[2:6])
                    bbox = [x, y, x + w, y + h]
                    img_path = os.path.join(img_dir, f"{frame:08d}.jpg")
                    if not os.path.exists(img_path):
                        continue
                    key = (plant_id, leaf_id)
                    self.samples_by_leaf.setdefault(key, []).append({
                        "plant_id": plant_id,
                        "leaf_id": leaf_id,
                        "frame": frame,
                        "bbox": bbox,
                        "image_path": img_path,
                    })
        self.plant_to_leaf_keys: Dict[str, List[Tuple[str, int]]] = {}
        for (plant_id, leaf_id) in self.samples_by_leaf:
            self.plant_to_leaf_keys.setdefault(plant_id, []).append((plant_id, leaf_id))

    def _load_crop(self, sample):
        img = Image.open(sample["image_path"]).convert("RGB")
        x1, y1, x2, y2 = sample["bbox"]
        patch = img.crop((x1, y1, x2, y2))
        return self.transform(patch) if self.transform else patch


class LeafTripletDataset(_BaseTripletDataset):
    """v1: anchor/positive from same leaf; negative from any other leaf (any plant)."""
    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        # choose an anchor leaf with ≥2 samples
        leaf_keys = list(self.samples_by_leaf.keys())
        anchor_key = random.choice(leaf_keys)
        samples = self.samples_by_leaf[anchor_key]
        while len(samples) < 2:
            anchor_key = random.choice(leaf_keys)
            samples = self.samples_by_leaf[anchor_key]
        anchor_sample, pos_sample = random.sample(samples, 2)
        # negative: different leaf (any plant)
        neg_key = random.choice(leaf_keys)
        while neg_key == anchor_key:
            neg_key = random.choice(leaf_keys)
        neg_sample = random.choice(self.samples_by_leaf[neg_key])

        a = self._load_crop(anchor_sample)
        p = self._load_crop(pos_sample)
        n = self._load_crop(neg_sample)
        return a, p, n


class LeafTripletDatasetV2(_BaseTripletDataset):
    """v2: negative sampled from a *different* leaf on the *same* plant."""
    def __init__(self, root: str, transform=None):
        super().__init__(root, transform)
        # keep only plants with ≥2 leaves and at least one leaf with ≥2 samples
        valid = []
        for plant_id, leaf_keys in self.plant_to_leaf_keys.items():
            if len(leaf_keys) < 2:
                continue
            if any(len(self.samples_by_leaf[k]) >= 2 for k in leaf_keys):
                valid.append(plant_id)
        self.valid_plants = valid

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        plant_id = random.choice(self.valid_plants)
        leaf_keys = self.plant_to_leaf_keys[plant_id]
        good_keys = [k for k in leaf_keys if len(self.samples_by_leaf[k]) >= 2]
        anchor_key = random.choice(good_keys)
        anchor_sample, pos_sample = random.sample(self.samples_by_leaf[anchor_key], 2)
        neg_key = random.choice([k for k in leaf_keys if k != anchor_key])
        neg_sample = random.choice(self.samples_by_leaf[neg_key])
        return self._load_crop(anchor_sample), self._load_crop(pos_sample), self._load_crop(neg_sample)


class LeafTripletDatasetV3(LeafTripletDatasetV2):
    """v3: same as v2 but enforce temporal window ±T for pos/neg relative to anchor frame."""
    def __init__(self, root: str, window_size: int, transform=None):
        super().__init__(root, transform)
        self.window_size = int(window_size)

    def __getitem__(self, idx):
        import random
        plant_id = random.choice(self.valid_plants)
        leaf_keys = self.plant_to_leaf_keys[plant_id]
        good_keys = [k for k in leaf_keys if len(self.samples_by_leaf[k]) >= 2]
        anchor_key = random.choice(good_keys)
        anchor_samples = self.samples_by_leaf[anchor_key]
        anchor_sample = random.choice(anchor_samples)
        a_frame = anchor_sample["frame"]
        # pos within window (fallback: any other sample of same leaf)
        pos_cands = [s for s in anchor_samples if s["frame"] != a_frame and abs(s["frame"] - a_frame) <= self.window_size]
        if not pos_cands:
            pos_cands = [s for s in anchor_samples if s["frame"] != a_frame]
        pos_sample = random.choice(pos_cands)
        # neg from other leaves; prefer within window
        neg_cands = []
        for nk in [k for k in leaf_keys if k != anchor_key]:
            for s in self.samples_by_leaf[nk]:
                if abs(s["frame"] - a_frame) <= self.window_size:
                    neg_cands.append(s)
        if not neg_cands:
            nk = random.choice([k for k in leaf_keys if k != anchor_key])
            neg_cands = self.samples_by_leaf[nk]
        neg_sample = random.choice(neg_cands)
        return self._load_crop(anchor_sample), self._load_crop(pos_sample), self._load_crop(neg_sample)