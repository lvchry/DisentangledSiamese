# -*- coding: utf-8 -*-

import os
import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
from scipy.stats import spearmanr

from dataset_alt import TripletDataset
from model import DisentangledSiamese


MOD = "mr"  # "ct" or "mr"
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4
VAL_SPLIT = 0.2
SEED = 42

TAU = 0.3
K = 4.0

LAM_NOISE = 0.5
LAM_ORTHO = 0.1
LAM_ORDER = 0.5
LAM_RATIO = 0.5

RATIO_EPS = 1e-8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE = "/home1/zyan6527/GoldAtlas_Finale/pairedDatas"
train_path = os.path.join(BASE, "trainA" if MOD == "ct" else "trainB")
SUFFIX = f"{MOD}_alt1"


def safe_spearman(x, y):
    r = spearmanr(x, y).correlation
    if r is None or np.isnan(r):
        return 0.0
    return float(r)


def set_all_seeds(seed):
    seed = int(seed) % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def shape_a(a):
    """
    Thresholded exponential mapping:
    a <= TAU -> close to 0
    a > TAU  -> smooth increase
    """
    b = torch.relu(a - TAU)
    denom = math.expm1(K * (1.0 - TAU))
    if denom <= 0:
        return torch.zeros_like(a)
    return torch.expm1(K * b) / denom


def anat_distance(z1, z2):
    score = F.cosine_similarity(z1, z2, dim=1)
    return 0.5 * (1.0 - score)


def orth_loss(z_anat, z_noise_ortho):
    return ((z_anat * z_noise_ortho).sum(dim=1) ** 2).mean()


def sample_hard_negative_index(idx, n, rng, min_sep=50):
    min_sep = min(min_sep, n - 1)

    left = list(range(0, max(0, idx - min_sep + 1)))
    right = list(range(min(n, idx + min_sep), n))
    candidates = left + right

    if not candidates:
        candidates = [j for j in range(n) if j != idx]

    return int(rng.choice(candidates))


class FixedValDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def make_tensor(img_np):
    img_np = np.asarray(img_np, dtype=np.float32)
    return torch.from_numpy(img_np).unsqueeze(0).repeat(3, 1, 1)


def build_fixed_val_cache(full_ds, val_indices):
    """
    Deterministic validation cache.
    FIXES:
      1. Negatives sampled only from val_indices (was: full dataset -> train leakage).
      2. Negatives get independently sampled fn/alpha (was: anchor's fn/alpha -> artificially easy negatives).
    """
    cache = []
    val_indices = list(val_indices)
    n_val = len(val_indices)                  # FIX 1: scope to val only
    fn_list = list(full_ds.dist.values())

    for local_idx, idx in enumerate(val_indices):   # FIX 1: track local position
        A = full_ds.load(full_ds.paths[idx]).astype(np.float32)

        base_seed = SEED * 100000 + int(idx) * 100 + 17
        py_rng = random.Random(base_seed)
        np_rng = np.random.default_rng(base_seed)

        fn = fn_list[py_rng.randrange(len(fn_list))]

        a_low = py_rng.uniform(0.0, 0.3)
        a_high = py_rng.uniform(0.3, 1.0)
        if a_high < a_low:
            a_low, a_high = a_high, a_low

        P_low = fn(A.copy(), float(a_low))
        P_high = fn(A.copy(), float(a_high))
        P_low  = np.clip(np.asarray(P_low,  dtype=np.float32), 0.0, 1.0)
        P_high = np.clip(np.asarray(P_high, dtype=np.float32), 0.0, 1.0)

        # FIX 1: sample within val_indices only, using local position for min_sep
        j1_local = sample_hard_negative_index(local_idx, n_val, np_rng, min_sep=50)
        j2_local = sample_hard_negative_index(local_idx, n_val, np_rng, min_sep=50)
        N1 = full_ds.load(full_ds.paths[val_indices[j1_local]]).astype(np.float32)
        N2 = full_ds.load(full_ds.paths[val_indices[j2_local]]).astype(np.float32)

        # FIX 2: independent fn and alpha for each negative (mirrors dataset_alt._maybe_distort)
        fn_n1  = fn_list[py_rng.randrange(len(fn_list))]
        fn_n2  = fn_list[py_rng.randrange(len(fn_list))]
        a_neg1 = py_rng.uniform(0.0, 1.0)
        a_neg2 = py_rng.uniform(0.0, 1.0)

        N_low  = fn_n1(N1.copy(), float(a_neg1))
        N_high = fn_n2(N2.copy(), float(a_neg2))
        N_low  = np.clip(np.asarray(N_low,  dtype=np.float32), 0.0, 1.0)
        N_high = np.clip(np.asarray(N_high, dtype=np.float32), 0.0, 1.0)

        item = (
            make_tensor(A),
            make_tensor(P_low),
            make_tensor(P_high),
            make_tensor(N_low),
            make_tensor(N_high),
            torch.tensor(a_low,  dtype=torch.float32),
            torch.tensor(a_high, dtype=torch.float32),
            torch.tensor(a_neg1, dtype=torch.float32),  # actual neg alphas now
            torch.tensor(a_neg2, dtype=torch.float32),
        )
        cache.append(item)

    return FixedValDataset(cache)


def ratio_triplet(d_aa, d_pl, d_ph, d_nl, d_nh):
    """
    Average over the four combinations:
      |d(A,A)-d(A,P)| / |d(A,A)-d(A,N)|
    using P_low/P_high and N_low/N_high.
    """
    ratios = []
    for d_p in [d_pl, d_ph]:
        for d_n in [d_nl, d_nh]:
            ratios.append(torch.abs(d_aa - d_p) / (torch.abs(d_aa - d_n) + RATIO_EPS))
    return torch.stack(ratios, dim=0).mean()


def compute_losses(zAa, zAn, zAn_ortho, aAhat,
                   zPl, zPl_ortho, aPlhat,
                   zPh, zPh_ortho, aPhhat,
                   zNl, zNl_ortho, aNlhat,
                   zNh, zNh_ortho, aNhhat,
                   a_low, a_high):

    d_aa = anat_distance(zAa, zAa)
    d_pl = anat_distance(zAa, zPl)
    d_ph = anat_distance(zAa, zPh)
    d_nl = anat_distance(zAa, zNl)
    d_nh = anat_distance(zAa, zNh)

    target_pl = shape_a(a_low)
    target_ph = shape_a(a_high)
    target_neg = torch.ones_like(a_low)

    loss_aa = F.mse_loss(d_aa, torch.zeros_like(d_aa))
    loss_pl = F.mse_loss(d_pl, target_pl)
    loss_ph = F.mse_loss(d_ph, target_ph)
    loss_neg = 0.5 * (
        F.mse_loss(d_nl, target_neg) +
        F.mse_loss(d_nh, target_neg)
    )

    # Force high-alpha to be larger than low-alpha.
    loss_order = F.relu(0.05 - (d_ph - d_pl)).mean()

    # Auxiliary ratio objective: keep distortion movement small
    # relative to identity movement.
    ratio = ratio_triplet(d_aa, d_pl, d_ph, d_nl, d_nh)
    loss_ratio = ratio

    loss_noise = (
        F.mse_loss(aAhat, torch.zeros_like(aAhat)) +
        F.mse_loss(aPlhat, a_low) +
        F.mse_loss(aPhhat, a_high) +
        F.mse_loss(aNlhat, a_low) +
        F.mse_loss(aNhhat, a_high)
    ) / 5.0

    loss_ortho = (
        orth_loss(zAa, zAn_ortho) +
        orth_loss(zPl, zPl_ortho) +
        orth_loss(zPh, zPh_ortho) +
        orth_loss(zNl, zNl_ortho) +
        orth_loss(zNh, zNh_ortho)
    ) / 5.0

    total = (
        loss_aa +
        loss_pl +
        loss_ph +
        loss_neg +
        LAM_ORDER * loss_order +
        LAM_RATIO * loss_ratio +
        LAM_NOISE * loss_noise +
        LAM_ORTHO * loss_ortho
    )

    parts = {
        "total": total,
        "loss_aa": loss_aa.detach(),
        "loss_pl": loss_pl.detach(),
        "loss_ph": loss_ph.detach(),
        "loss_neg": loss_neg.detach(),
        "loss_order": loss_order.detach(),
        "loss_ratio": loss_ratio.detach(),
        "loss_noise": loss_noise.detach(),
        "loss_ortho": loss_ortho.detach(),
        "d_pl": d_pl.detach(),
        "d_ph": d_ph.detach(),
        "d_nl": d_nl.detach(),
        "d_nh": d_nh.detach(),
    }
    return parts


def main():
    print(f"=== TRAIN/VAL ({MOD}) ===")

    full_ds = TripletDataset(train_path, MOD)

    val_size = int(len(full_ds) * VAL_SPLIT)
    train_size = len(full_ds) - val_size

    train_ds, val_ds = random_split(
        full_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    val_cache_ds = build_fixed_val_cache(full_ds, val_ds.indices)
    val_loader = DataLoader(val_cache_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = DisentangledSiamese().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "spearman_label": [],
        "spearman_a": [],
        "lr": [],
        "train_ratio": [],
        "val_ratio": [],
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_ratio_vals = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for A, P_low, P_high, N_low, N_high, a_low, a_high, a_neg1, a_neg2 in pbar:
            A = A.to(device).float()
            P_low = P_low.to(device).float()
            P_high = P_high.to(device).float()
            N_low = N_low.to(device).float()
            N_high = N_high.to(device).float()

            a_low = a_low.to(device).float()
            a_high = a_high.to(device).float()
            a_neg1 = a_neg1.to(device).float()
            a_neg2 = a_neg2.to(device).float()

            opt.zero_grad()

            zAa, zAn, zAn_ortho, aAhat = model.forward_once(A)
            zPl, zPn, zPl_ortho, aPlhat = model.forward_once(P_low)
            zPh, zPn2, zPh_ortho, aPhhat = model.forward_once(P_high)
            zNl, zNn, zNl_ortho, aNlhat = model.forward_once(N_low)
            zNh, zNn2, zNh_ortho, aNhhat = model.forward_once(N_high)

            parts = compute_losses(
                zAa, zAn, zAn_ortho, aAhat,
                zPl, zPl_ortho, aPlhat,
                zPh, zPh_ortho, aPhhat,
                zNl, zNl_ortho, aNlhat,
                zNh, zNh_ortho, aNhhat,
                a_low, a_high,
            )

            loss = parts["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

            train_loss += float(loss.item())
            train_ratio_vals.append(float(parts["loss_ratio"].item()))
            pbar.set_postfix(loss=float(loss.item()), ratio=float(parts["loss_ratio"].item()))

        train_loss /= max(1, len(train_loader))
        train_ratio = float(np.mean(train_ratio_vals)) if train_ratio_vals else 0.0

        model.eval()
        val_loss = 0.0
        val_ratio_vals = []
        preds = []
        labs = []
        ext = []

        with torch.no_grad():
            for A, P_low, P_high, N_low, N_high, a_low, a_high, a_neg1, a_neg2 in val_loader:
                A = A.to(device).float()
                P_low = P_low.to(device).float()
                P_high = P_high.to(device).float()
                N_low = N_low.to(device).float()
                N_high = N_high.to(device).float()

                a_low = a_low.to(device).float()
                a_high = a_high.to(device).float()
                a_neg1 = a_neg1.to(device).float()
                a_neg2 = a_neg2.to(device).float()

                zAa, zAn, zAn_ortho, aAhat = model.forward_once(A)
                zPl, zPn, zPl_ortho, aPlhat = model.forward_once(P_low)
                zPh, zPn2, zPh_ortho, aPhhat = model.forward_once(P_high)
                zNl, zNn, zNl_ortho, aNlhat = model.forward_once(N_low)
                zNh, zNn2, zNh_ortho, aNhhat = model.forward_once(N_high)

                parts = compute_losses(
                    zAa, zAn, zAn_ortho, aAhat,
                    zPl, zPl_ortho, aPlhat,
                    zPh, zPh_ortho, aPhhat,
                    zNl, zNl_ortho, aNlhat,
                    zNh, zNh_ortho, aNhhat,
                    a_low, a_high,
                )

                loss = parts["total"]
                val_loss += float(loss.item())
                val_ratio_vals.append(float(parts["loss_ratio"].item()))

                preds.extend(parts["d_pl"].cpu().numpy().tolist())
                labs.extend([0] * len(parts["d_pl"]))
                ext.extend(a_low.detach().cpu().numpy().tolist())

                preds.extend(parts["d_ph"].cpu().numpy().tolist())
                labs.extend([0] * len(parts["d_ph"]))
                ext.extend(a_high.detach().cpu().numpy().tolist())

                preds.extend(parts["d_nl"].cpu().numpy().tolist())
                labs.extend([1] * len(parts["d_nl"]))
                ext.extend(a_neg1.detach().cpu().numpy().tolist())

                preds.extend(parts["d_nh"].cpu().numpy().tolist())
                labs.extend([1] * len(parts["d_nh"]))
                ext.extend(a_neg2.detach().cpu().numpy().tolist())

        val_loss /= max(1, len(val_loader))
        val_ratio = float(np.mean(val_ratio_vals)) if val_ratio_vals else 0.0

        scheduler.step(val_loss)

        rho_label = safe_spearman(preds, labs)
        rho_a = safe_spearman(preds, ext)
        current_lr = opt.param_groups[0]["lr"]

        print(f"\nEpoch {epoch}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR:         {current_lr:.6f}")
        print(f"  Spearman(label): {rho_label:.4f}")
        print(f"  Spearman(a):     {rho_a:.4f}")
        print(f"  Train ratio:     {train_ratio:.4f}")
        print(f"  Val ratio:       {val_ratio:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"best_{SUFFIX}.pth")
            print("  Saved best model")

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["spearman_label"].append(rho_label)
        history["spearman_a"].append(rho_a)
        history["lr"].append(current_lr)
        history["train_ratio"].append(train_ratio)
        history["val_ratio"].append(val_ratio)

    df = pd.DataFrame(history)
    df.to_csv(f"train_history_{SUFFIX}.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="train loss")
    plt.plot(df["epoch"], df["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Training loss ({MOD})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"train_curve_{SUFFIX}.png", dpi=200)
    plt.close()

    print("\nSaved:")
    print(f"  best_{SUFFIX}.pth")
    print(f"  train_history_{SUFFIX}.csv")
    print(f"  train_curve_{SUFFIX}.png")


if __name__ == "__main__":
    main()
