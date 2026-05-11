import os
import cv2
import random
import torch
import numpy as np
from torch.utils.data import Dataset

from Distortions_Continuous import CT_DIST, MRI_DIST


# Mean pixel value below which a loaded image is considered blank (0-1 scale).
# Acts as a runtime safety net in case any blank slices were not caught at
# extraction time. Raise to 0.04-0.06 if anatomically empty slices slip through.
BLANK_THRESHOLD = 5.0 / 255.0    # equivalent to 5/255 in uint8


def is_img(fname):
    return fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))


def _mean_pixel(path: str) -> float:
    """Fast mean pixel value check without full preprocessing."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(img.mean()) / 255.0


class TripletDataset(Dataset):
    def __init__(self, root, modality="ct", img_size=256,
                 blank_threshold=BLANK_THRESHOLD):
        all_paths = sorted(
            [os.path.join(root, f) for f in os.listdir(root) if is_img(f)]
        )

        # Filter blank / near-black images at init time so they never
        # appear as anchors, positives, or negatives during training/eval.
        self.paths = [p for p in all_paths
                      if _mean_pixel(p) >= blank_threshold]

        n_total   = len(all_paths)
        n_kept    = len(self.paths)
        n_removed = n_total - n_kept

        if n_removed > 0:
            print(f"[TripletDataset] {root}: "
                  f"removed {n_removed}/{n_total} blank images "
                  f"(threshold={blank_threshold:.4f})")

        if n_kept < 2:
            raise ValueError(
                f"Need at least 2 non-blank images in {root}. "
                f"Found {n_kept} after filtering {n_removed} blank images. "
                f"Try lowering blank_threshold."
            )

        self.N               = n_kept
        self.img_size        = img_size
        self.blank_threshold = blank_threshold
        self.dist            = CT_DIST if modality.lower() == "ct" else MRI_DIST

    def __len__(self):
        return self.N

    def load(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        img = cv2.resize(img, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        return np.clip(img, 0.0, 1.0)

    def to_tensor(self, x):
        x = np.asarray(x, dtype=np.float32)
        return torch.from_numpy(x).unsqueeze(0).repeat(3, 1, 1)

    def sample_far_index(self, idx, min_sep=50):
        min_sep    = min(min_sep, self.N - 1)
        left       = list(range(0, max(0, idx - min_sep + 1)))
        right      = list(range(min(self.N, idx + min_sep), self.N))
        candidates = left + right

        if not candidates:
            candidates = [j for j in range(self.N) if j != idx]

        return random.choice(candidates)

    def sample_fn(self):
        return random.choice(list(self.dist.values()))

    def __getitem__(self, idx):
        A = self.load(self.paths[idx])

        # One distortion family per sample, two alpha bands
        fn     = self.sample_fn()
        a_low  = random.uniform(0.0, 0.3)
        a_high = random.uniform(0.3, 1.0)

        if a_high < a_low:
            a_low, a_high = a_high, a_low

        P_low  = np.asarray(fn(A.astype(np.float32).copy(), float(a_low)),  dtype=np.float32)
        P_high = np.asarray(fn(A.astype(np.float32).copy(), float(a_high)), dtype=np.float32)
        P_low  = np.clip(P_low,  0.0, 1.0)
        P_high = np.clip(P_high, 0.0, 1.0)

        # Negatives: different patient, optionally distorted
        j1 = self.sample_far_index(idx, min_sep=50)
        j2 = self.sample_far_index(idx, min_sep=50)

        N1 = self.load(self.paths[j1])
        N2 = self.load(self.paths[j2])

        if random.random() < 0.5:
            N1, a_neg1 = self._maybe_distort(N1)
        else:
            a_neg1 = 0.0

        if random.random() < 0.5:
            N2, a_neg2 = self._maybe_distort(N2)
        else:
            a_neg2 = 0.0

        return (
            self.to_tensor(A),
            self.to_tensor(P_low),
            self.to_tensor(P_high),
            self.to_tensor(N1),
            self.to_tensor(N2),
            torch.tensor(a_low,  dtype=torch.float32),
            torch.tensor(a_high, dtype=torch.float32),
            torch.tensor(a_neg1, dtype=torch.float32),
            torch.tensor(a_neg2, dtype=torch.float32),
        )

    def _maybe_distort(self, img):
        a   = random.uniform(0.0, 1.0)
        fn  = self.sample_fn()
        out = fn(img.astype(np.float32).copy(), float(a))
        out = np.clip(np.asarray(out, dtype=np.float32), 0.0, 1.0)
        return out, a