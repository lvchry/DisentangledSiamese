# -*- coding: utf-8 -*-

import os
import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from scipy.stats import spearmanr

from dataset import TripletDataset
from model import DisentangledSiamese


# ============================================================
# CONFIG
# ============================================================

MOD     = "mr"
DATASET = "cfbgbm"

# With 2 GPUs, effective batch size = BATCH_SIZE * 2
# Larger batch helps InfoNCE have enough negatives
BATCH_SIZE = 16
EPOCHS     = 50
LR         = 1e-4       # will be scaled by n_gpus in main
VAL_SPLIT  = 0.2
SEED       = 42

TAU = 0.3
K   = 4.0

LAM_NOISE = 0.5
LAM_ORTHO = 0.1
LAM_ORDER = 0.5
LAM_RATIO = 0.5
LAM_NCE   = 0.05        # reduced from 0.1 -- NCE is unstable at small batch
NCE_WARMUP = 5          # epochs before NCE kicks in
NCE_RAMP   = 10         # epochs over which NCE linearly ramps to full weight

NCE_TEMP  = 0.2         # raised from 0.07 -- less sharp, more stable

GRAD_CLIP = 1.0         # tightened from 2.0 to prevent NCE spikes

RATIO_EPS = 1e-8

BASES = {
    "goldatlas": "/home1/zyan6527/comprehensive_test/GoldAtlas-slices",
    "cfbgbm":    "/home1/zyan6527/comprehensive_test/CFB-GBM-slices",
}

BASE       = BASES[DATASET]
train_path = os.path.join(BASE, "trainA" if MOD == "ct" else "trainB")
SUFFIX     = f"{MOD}_{DATASET}"

# ============================================================


def safe_spearman(x, y):
    r = spearmanr(x, y).correlation
    if r is None or np.isnan(r):
        return 0.0
    return float(r)


def set_all_seeds(seed):
    seed = int(seed) % (2 ** 32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def shape_a(a):
    b     = torch.relu(a - TAU)
    denom = math.expm1(K * (1.0 - TAU))
    if denom <= 0:
        return torch.zeros_like(a)
    return torch.expm1(K * b) / denom


def anat_distance(z1, z2):
    return 0.5 * (1.0 - F.cosine_similarity(z1, z2, dim=1))


def orth_loss(z_anat, z_noise_ortho):
    return ((z_anat * z_noise_ortho).sum(dim=1) ** 2).mean()


def infonce_loss(z_anchor, z_positive, temperature=NCE_TEMP):
    B = z_anchor.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=z_anchor.device)
    z      = torch.cat([z_anchor, z_positive], dim=0)
    sim    = (z @ z.T) / temperature
    mask   = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, float("-inf"))
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B,     device=z.device),
    ])
    return F.cross_entropy(sim, labels)


def sample_hard_negative_index(idx, n, rng, min_sep=50):
    min_sep    = min(min_sep, n - 1)
    left       = list(range(0, max(0, idx - min_sep + 1)))
    right      = list(range(min(n, idx + min_sep), n))
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
    cache       = []
    val_indices = list(val_indices)
    n_val       = len(val_indices)
    fn_list     = list(full_ds.dist.values())

    for local_idx, idx in enumerate(val_indices):
        A         = full_ds.load(full_ds.paths[idx]).astype(np.float32)
        base_seed = SEED * 100000 + int(idx) * 100 + 17
        py_rng    = random.Random(base_seed)
        np_rng    = np.random.default_rng(base_seed)

        fn     = fn_list[py_rng.randrange(len(fn_list))]
        a_low  = py_rng.uniform(0.0, 0.3)
        a_high = py_rng.uniform(0.3, 1.0)
        if a_high < a_low:
            a_low, a_high = a_high, a_low

        P_low  = np.clip(np.asarray(fn(A.copy(), float(a_low)),  dtype=np.float32), 0.0, 1.0)
        P_high = np.clip(np.asarray(fn(A.copy(), float(a_high)), dtype=np.float32), 0.0, 1.0)

        j1_local = sample_hard_negative_index(local_idx, n_val, np_rng, min_sep=50)
        j2_local = sample_hard_negative_index(local_idx, n_val, np_rng, min_sep=50)
        N1 = full_ds.load(full_ds.paths[val_indices[j1_local]]).astype(np.float32)
        N2 = full_ds.load(full_ds.paths[val_indices[j2_local]]).astype(np.float32)

        fn_n1  = fn_list[py_rng.randrange(len(fn_list))]
        fn_n2  = fn_list[py_rng.randrange(len(fn_list))]
        a_neg1 = py_rng.uniform(0.0, 1.0)
        a_neg2 = py_rng.uniform(0.0, 1.0)

        N_low  = np.clip(np.asarray(fn_n1(N1.copy(), float(a_neg1)), dtype=np.float32), 0.0, 1.0)
        N_high = np.clip(np.asarray(fn_n2(N2.copy(), float(a_neg2)), dtype=np.float32), 0.0, 1.0)

        cache.append((
            make_tensor(A),
            make_tensor(P_low),
            make_tensor(P_high),
            make_tensor(N_low),
            make_tensor(N_high),
            torch.tensor(a_low,  dtype=torch.float32),
            torch.tensor(a_high, dtype=torch.float32),
            torch.tensor(a_neg1, dtype=torch.float32),
            torch.tensor(a_neg2, dtype=torch.float32),
        ))

    return FixedValDataset(cache)


def ratio_triplet(d_pl, d_ph, d_nl, d_nh):
    ratios = []
    for d_p in [d_pl, d_ph]:
        for d_n in [d_nl, d_nh]:
            ratios.append(d_p / (d_n + RATIO_EPS))
    return torch.stack(ratios, dim=0).mean()


def compute_losses(zAa, zAn, zAn_ortho, aAhat,
                   zPl, zPl_ortho, aPlhat,
                   zPh, zPh_ortho, aPhhat,
                   zNl, zNl_ortho, aNlhat,
                   zNh, zNh_ortho, aNhhat,
                   a_low, a_high, a_neg1, a_neg2,
                   lam_nce_current=LAM_NCE):

    d_pl = anat_distance(zAa, zPl)
    d_ph = anat_distance(zAa, zPh)
    d_nl = anat_distance(zAa, zNl)
    d_nh = anat_distance(zAa, zNh)

    target_pl  = shape_a(a_low)
    target_ph  = shape_a(a_high)
    target_neg = torch.ones_like(a_low)

    loss_pl    = F.mse_loss(d_pl, target_pl)
    loss_ph    = F.mse_loss(d_ph, target_ph)
    loss_neg   = 0.5 * (F.mse_loss(d_nl, target_neg) + F.mse_loss(d_nh, target_neg))
    loss_order = F.relu(0.05 - (d_ph - d_pl)).mean()
    loss_ratio = ratio_triplet(d_pl, d_ph, d_nl, d_nh)

    loss_noise = (
        F.mse_loss(aAhat,  torch.zeros_like(aAhat)) +
        F.mse_loss(aPlhat, a_low) +
        F.mse_loss(aPhhat, a_high) +
        F.mse_loss(aNlhat, a_neg1) +
        F.mse_loss(aNhhat, a_neg2)
    ) / 5.0

    loss_ortho = (
        orth_loss(zAa, zAn_ortho) +
        orth_loss(zPl, zPl_ortho) +
        orth_loss(zPh, zPh_ortho) +
        orth_loss(zNl, zNl_ortho) +
        orth_loss(zNh, zNh_ortho)
    ) / 5.0

    loss_nce = 0.5 * (
        infonce_loss(zAa, zPl) +
        infonce_loss(zAa, zPh)
    )

    total = (
        loss_pl +
        loss_ph +
        loss_neg +
        LAM_ORDER * loss_order +
        LAM_RATIO * loss_ratio +
        LAM_NOISE * loss_noise +
        LAM_ORTHO * loss_ortho +
        lam_nce_current * loss_nce
    )

    return {
        "total":      total,
        "loss_pl":    loss_pl.detach(),
        "loss_ph":    loss_ph.detach(),
        "loss_neg":   loss_neg.detach(),
        "loss_order": loss_order.detach(),
        "loss_ratio": loss_ratio.detach(),
        "loss_noise": loss_noise.detach(),
        "loss_ortho": loss_ortho.detach(),
        "loss_nce":   loss_nce.detach(),
        "d_pl":       d_pl.detach(),
        "d_ph":       d_ph.detach(),
        "d_nl":       d_nl.detach(),
        "d_nh":       d_nh.detach(),
    }


def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    dist.destroy_process_group()


def is_main(rank):
    return rank == 0


def train_worker(rank, world_size):
    """
    DDP worker function - one process per GPU.
    rank 0 is the main process that prints, saves, etc.
    """
    setup_ddp(rank, world_size)
    set_all_seeds(SEED + rank)

    device = torch.device(f"cuda:{rank}")

    if is_main(rank):
        print(f"=== TRAIN/VAL ({MOD.upper()}) - {DATASET} ===")
        print(f"  GPUs           : {world_size} x a40")
        print(f"  Per-GPU batch  : {BATCH_SIZE}  (effective: {BATCH_SIZE * world_size})")
        print(f"  LAM_NCE        : {LAM_NCE}, temp={NCE_TEMP}\n")

    full_ds  = TripletDataset(train_path, MOD)
    val_size = int(len(full_ds) * VAL_SPLIT)
    train_size = len(full_ds) - val_size

    train_ds, val_ds = random_split(
        full_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    # DistributedSampler ensures each GPU sees a non-overlapping subset
    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    # Val cache built only on rank 0, then all ranks load same cache
    val_cache_ds = build_fixed_val_cache(full_ds, val_ds.indices)
    val_sampler  = DistributedSampler(
        val_cache_ds, num_replicas=world_size, rank=rank, shuffle=False
    )
    val_loader = DataLoader(
        val_cache_ds,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    model = DisentangledSiamese().to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # Scale LR linearly with number of GPUs
    scaled_lr = LR * world_size
    opt = torch.optim.Adam(model.parameters(), lr=scaled_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "spearman_label": [], "spearman_a": [], "lr": [],
        "train_ratio": [], "val_ratio": [],
        "train_nce": [], "val_nce": [],
    }

    for epoch in range(1, EPOCHS + 1):
        # Must set epoch on sampler for proper shuffling across epochs
        train_sampler.set_epoch(epoch)

        # InfoNCE warmup: zero for first NCE_WARMUP epochs,
        # then linearly ramp to full LAM_NCE over NCE_RAMP epochs.
        # Prevents the spike seen when NCE kicks in too early.
        if epoch <= NCE_WARMUP:
            lam_nce_current = 0.0
        else:
            ramp = min(1.0, (epoch - NCE_WARMUP) / float(NCE_RAMP))
            lam_nce_current = LAM_NCE * ramp

        if is_main(rank):
            print(f"\nEpoch {epoch}  [LAM_NCE={lam_nce_current:.4f}]")

        # ---- Train ----
        model.train()
        train_loss_acc  = []
        train_ratio_acc = []
        train_nce_acc   = []

        if is_main(rank):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        else:
            pbar = train_loader

        for A, P_low, P_high, N_low, N_high, a_low, a_high, a_neg1, a_neg2 in pbar:
            A      = A.to(device).float()
            P_low  = P_low.to(device).float()
            P_high = P_high.to(device).float()
            N_low  = N_low.to(device).float()
            N_high = N_high.to(device).float()
            a_low  = a_low.to(device).float()
            a_high = a_high.to(device).float()
            a_neg1 = a_neg1.to(device).float()
            a_neg2 = a_neg2.to(device).float()

            opt.zero_grad()

            zAa, zAn, zAn_ortho, aAhat   = model.module.forward_once(A)
            zPl, zPn, zPl_ortho, aPlhat  = model.module.forward_once(P_low)
            zPh, zPn2, zPh_ortho, aPhhat = model.module.forward_once(P_high)
            zNl, zNn, zNl_ortho, aNlhat  = model.module.forward_once(N_low)
            zNh, zNn2, zNh_ortho, aNhhat = model.module.forward_once(N_high)

            parts = compute_losses(
                zAa, zAn, zAn_ortho, aAhat,
                zPl, zPl_ortho, aPlhat,
                zPh, zPh_ortho, aPhhat,
                zNl, zNl_ortho, aNlhat,
                zNh, zNh_ortho, aNhhat,
                a_low, a_high, a_neg1, a_neg2,
                lam_nce_current=lam_nce_current,
            )

            parts["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            train_loss_acc.append(float(parts["total"].item()))
            train_ratio_acc.append(float(parts["loss_ratio"].item()))
            train_nce_acc.append(float(parts["loss_nce"].item()))

            if is_main(rank):
                pbar.set_postfix(
                    loss=f"{parts['total'].item():.4f}",
                    nce=f"{parts['loss_nce'].item():.4f}",
                )

        train_loss  = float(np.mean(train_loss_acc))
        train_ratio = float(np.mean(train_ratio_acc))
        train_nce   = float(np.mean(train_nce_acc))

        # ---- Val ----
        model.eval()
        val_loss_acc  = []
        val_ratio_acc = []
        val_nce_acc   = []
        preds, labs, ext = [], [], []

        with torch.no_grad():
            for A, P_low, P_high, N_low, N_high, a_low, a_high, a_neg1, a_neg2 in val_loader:
                A      = A.to(device).float()
                P_low  = P_low.to(device).float()
                P_high = P_high.to(device).float()
                N_low  = N_low.to(device).float()
                N_high = N_high.to(device).float()
                a_low  = a_low.to(device).float()
                a_high = a_high.to(device).float()
                a_neg1 = a_neg1.to(device).float()
                a_neg2 = a_neg2.to(device).float()

                zAa, zAn, zAn_ortho, aAhat   = model.module.forward_once(A)
                zPl, zPn, zPl_ortho, aPlhat  = model.module.forward_once(P_low)
                zPh, zPn2, zPh_ortho, aPhhat = model.module.forward_once(P_high)
                zNl, zNn, zNl_ortho, aNlhat  = model.module.forward_once(N_low)
                zNh, zNn2, zNh_ortho, aNhhat = model.module.forward_once(N_high)

                parts = compute_losses(
                    zAa, zAn, zAn_ortho, aAhat,
                    zPl, zPl_ortho, aPlhat,
                    zPh, zPh_ortho, aPhhat,
                    zNl, zNl_ortho, aNlhat,
                    zNh, zNh_ortho, aNhhat,
                    a_low, a_high, a_neg1, a_neg2,
                    lam_nce_current=lam_nce_current,
                )

                val_loss_acc.append(float(parts["total"].item()))
                val_ratio_acc.append(float(parts["loss_ratio"].item()))
                val_nce_acc.append(float(parts["loss_nce"].item()))

                preds.extend(parts["d_pl"].cpu().numpy().tolist())
                labs.extend([0] * len(parts["d_pl"]))
                ext.extend(a_low.cpu().numpy().tolist())

                preds.extend(parts["d_ph"].cpu().numpy().tolist())
                labs.extend([0] * len(parts["d_ph"]))
                ext.extend(a_high.cpu().numpy().tolist())

                preds.extend(parts["d_nl"].cpu().numpy().tolist())
                labs.extend([1] * len(parts["d_nl"]))
                ext.extend(a_neg1.cpu().numpy().tolist())

                preds.extend(parts["d_nh"].cpu().numpy().tolist())
                labs.extend([1] * len(parts["d_nh"]))
                ext.extend(a_neg2.cpu().numpy().tolist())

        val_loss  = float(np.mean(val_loss_acc))
        val_ratio = float(np.mean(val_ratio_acc))
        val_nce   = float(np.mean(val_nce_acc))

        # Aggregate val loss across all GPUs for scheduler
        val_loss_tensor = torch.tensor(val_loss, device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
        val_loss_global = float(val_loss_tensor.item())

        scheduler.step(val_loss_global)

        rho_label  = safe_spearman(preds, labs)
        rho_a      = safe_spearman(preds, ext)
        current_lr = opt.param_groups[0]["lr"]

        if is_main(rank):
            print(f"  Train Loss      : {train_loss:.4f}")
            print(f"  Val Loss        : {val_loss_global:.4f}")
            print(f"  LR              : {current_lr:.6f}")
            print(f"  Spearman(label) : {rho_label:.4f}")
            print(f"  Spearman(a)     : {rho_a:.4f}")
            print(f"  Train NCE       : {train_nce:.4f}  (weight={lam_nce_current:.4f})")
            print(f"  Val NCE         : {val_nce:.4f}")
            print(f"  Train ratio     : {train_ratio:.4f}")
            print(f"  Val ratio       : {val_ratio:.4f}")

            if val_loss_global < best_val_loss:
                best_val_loss = val_loss_global
                # Save model.module.state_dict() not model.state_dict()
                # so checkpoint loads cleanly without DDP wrapper
                torch.save(model.module.state_dict(), f"best_{SUFFIX}.pth")
                print("  Saved best model")

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss_global)
            history["spearman_label"].append(rho_label)
            history["spearman_a"].append(rho_a)
            history["lr"].append(current_lr)
            history["train_ratio"].append(train_ratio)
            history["val_ratio"].append(val_ratio)
            history["train_nce"].append(train_nce)
            history["val_nce"].append(val_nce)

    if is_main(rank):
        df = pd.DataFrame(history)
        df.to_csv(f"train_history_{SUFFIX}.csv", index=False)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(df["epoch"], df["train_loss"], label="train")
        axes[0].plot(df["epoch"], df["val_loss"],   label="val")
        axes[0].set_title(f"Total loss ({MOD.upper()})")
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
        axes[0].legend()

        axes[1].plot(df["epoch"], df["train_nce"], label="train")
        axes[1].plot(df["epoch"], df["val_nce"],   label="val")
        axes[1].set_title("InfoNCE loss")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("nce loss")
        axes[1].legend()

        axes[2].plot(df["epoch"], df["spearman_label"], label="Spearman(id)")
        axes[2].plot(df["epoch"], df["spearman_a"],     label="Spearman(a)")
        axes[2].set_title("Spearman correlations")
        axes[2].set_xlabel("epoch"); axes[2].set_ylabel("rho")
        axes[2].legend()

        fig.tight_layout()
        fig.savefig(f"train_curve_{SUFFIX}.png", dpi=200)
        plt.close(fig)

        print("\nSaved:")
        print(f"  best_{SUFFIX}.pth")
        print(f"  train_history_{SUFFIX}.csv")
        print(f"  train_curve_{SUFFIX}.png")

    cleanup_ddp()


def main():
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No GPUs found. Check CUDA installation.")
    print(f"Launching DDP on {world_size} GPUs...")
    mp.spawn(train_worker, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()