# -*- coding: utf-8 -*-

import os
import random
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.stats import spearmanr
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from skimage.metrics import structural_similarity as ssim

from dataset import TripletDataset
from model import DisentangledSiamese
from Distortions_Continuous import CT_DIST, MRI_DIST


# ============================================================
# CONFIG
# ============================================================

MOD        = "mr"           # "ct" or "mr"
DATASET    = "cfbgbm"       # "goldatlas" or "cfbgbm"

BOOTSTRAPS   = 100
TAU          = 0.3          # threshold separating benign / malignant distortion
HARD_NEG_SEP = 50

# Retrieval -- used only for degradation and TAU-split plots, NOT reported as headline
RETRIEVAL_KS     = [1, 5, 10]
RETRIEVAL_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
RETRIEVAL_MAX_N  = 150   # subsample test set for O(n^2) pixel metrics
RETRIEVAL_SEED   = 0

# Distance-vs-alpha sampling
DIST_ALPHA_SAMPLES = 300    # pairs sampled per alpha level
DIST_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASES = {
    "goldatlas": "/home1/zyan6527/comprehensive_test/GoldAtlas-slices",
    "cfbgbm":    "/home1/zyan6527/comprehensive_test/CFB-GBM-slices",
}

BASE          = BASES[DATASET]
test_path     = os.path.join(BASE, "testA" if MOD == "ct" else "testB")
SUFFIX        = f"{MOD}_{DATASET}"
DIST          = CT_DIST if MOD == "ct" else MRI_DIST

# Main reported metrics -- retrieval NOT included in headline table
METRICS = ["ours", "mse", "ssim", "ncc", "mi", "dino", "grad"]
# cwssim removed: approximation unreliable (returns 0.0 silently) and 25x slower than mse

# ============================================================


# -- DINO -----------------------------------------------------

_dino_model = None

def _get_dino():
    global _dino_model
    if _dino_model is None:
        _dino_model = torch.hub.load(
            "facebookresearch/dino:main",
            "dino_vits8",
            pretrained=True,
        ).to(device).eval()
        for p in _dino_model.parameters():
            p.requires_grad_(False)
    return _dino_model

def _to_dino_input(x: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(x).unsqueeze(0).repeat(3, 1, 1)
    t = TF.normalize(t, mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    return t.unsqueeze(0).to(device)

def dino_distance(x: np.ndarray, y: np.ndarray) -> float:
    model = _get_dino()
    with torch.no_grad():
        fx = model(_to_dino_input(x))
        fy = model(_to_dino_input(y))
    cos = F.cosine_similarity(fx, fy, dim=1)
    return float(0.5 * (1.0 - cos).item())


# -- Gradient distance ----------------------------------------

def _sobel(x: np.ndarray) -> np.ndarray:
    gx  = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy  = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return mag / (np.linalg.norm(mag) + 1e-8)

def grad_distance(x: np.ndarray, y: np.ndarray) -> float:
    gx  = _sobel(x).ravel()
    gy  = _sobel(y).ravel()
    cos = float(np.dot(gx, gy) /
                (np.linalg.norm(gx) * np.linalg.norm(gy) + 1e-8))
    return 0.5 * (1.0 - cos)


# -- CW-SSIM --------------------------------------------------

def _cwssim(x: np.ndarray, y: np.ndarray, level: int = 5) -> float:
    scores = []
    xf = x.astype(np.float64)
    yf = y.astype(np.float64)
    for lv in range(1, level + 1):
        sigma  = 2.0 ** lv
        xr = gaussian_filter(xf, sigma)
        yr = gaussian_filter(yf, sigma)
        xi = gaussian_filter(xf, sigma * 0.7) - gaussian_filter(xf, sigma * 1.3)
        yi = gaussian_filter(yf, sigma * 0.7) - gaussian_filter(yf, sigma * 1.3)
        cx = xr + 1j * xi
        cy = yr + 1j * yi
        ws = max(1.0, sigma / 2.0)
        num   = gaussian_filter(np.real(cx * np.conj(cy)), ws)
        denom = gaussian_filter(np.abs(cx) ** 2 + np.abs(cy) ** 2, ws) + 1e-8
        scores.append(float(np.mean(np.abs(num) / denom)))
    return float(np.mean(scores))

def cwssim_distance(x: np.ndarray, y: np.ndarray) -> float:
    try:    return float(1.0 - _cwssim(x, y))
    except: return 0.0


# -- Classical metrics ----------------------------------------

def safe_spearman(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) != len(y) or len(x) == 0:
        return 0.0
    r = spearmanr(x, y).correlation
    return 0.0 if (r is None or np.isnan(r)) else float(r)

def safe_auc(y_true, y_score):
    try:    return float(roc_auc_score(y_true, y_score))
    except: return 0.0

def ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).mean() /
                 (np.sqrt((a**2).mean()) * np.sqrt((b**2).mean()) + 1e-8))

def mutual_information(a, b, bins=32):
    a = (a * (bins - 1)).astype(int)
    b = (b * (bins - 1)).astype(int)
    joint, _, _ = np.histogram2d(a.flatten(), b.flatten(), bins=bins)
    joint /= joint.sum() + 1e-8
    pa, pb = joint.sum(axis=1), joint.sum(axis=0)
    nz = joint > 0
    return float(np.sum(joint[nz] * np.log(
        joint[nz] / (pa[:, None] * pb[None, :])[nz])))

def ranking_accuracy(preds, a):
    preds, a = np.asarray(preds), np.asarray(a)
    correct = total = 0
    for i in range(len(preds)):
        for j in range(len(preds)):
            if a[i] > a[j]:
                total += 1
                if preds[i] > preds[j]:
                    correct += 1
    return correct / total if total else 0.0

def sample_hard_negative(idx, n, rng, min_sep=10):
    min_sep    = min(min_sep, n - 1)
    left       = list(range(0, max(0, idx - min_sep + 1)))
    right      = list(range(min(n, idx + min_sep), n))
    candidates = left + right or [j for j in range(n) if j != idx]
    return rng.choice(candidates)

def to_tensor(x):
    x = np.asarray(x, dtype=np.float32)
    return torch.from_numpy(x).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)

def anat_distance(model, x, y):
    m = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        z1, *_ = m.forward_once(to_tensor(x))
        z2, *_ = m.forward_once(to_tensor(y))
        return float(0.5 * (1.0 - F.cosine_similarity(z1, z2, dim=1)).item())

def compute_pair_metric(metric, model, x, y):
    if metric == "ours":    return anat_distance(model, x, y)
    if metric == "mse":     return float(np.mean((x - y) ** 2))
    if metric == "ssim":
        try:    return float(1.0 - ssim(x, y, data_range=1.0))
        except: return 0.0
    if metric == "cwssim":  return cwssim_distance(x, y)
    if metric == "ncc":
        try:    return float(1.0 - ncc(x, y))
        except: return 0.0
    if metric == "mi":
        try:    return float(-mutual_information(x, y))
        except: return 0.0
    if metric == "dino":
        try:    return dino_distance(x, y)
        except: return 0.0
    if metric == "grad":    return grad_distance(x, y)
    return 0.0

def triplet_ratio(d_aa, d_pl, d_ph, d_nl, d_nh, eps=1e-8):
    vals = [abs(d_aa - dp) / (abs(d_aa - dn) + eps)
            for dp in [d_pl, d_ph] for dn in [d_nl, d_nh]]
    return float(np.mean(vals))


# ============================================================
# STORY 1 + 2: Distortion awareness and benign/malignant eval
# ============================================================

def build_eval_tables(model, ds):
    """
    Builds three DataFrames:
    - dist_df  : distortion tracking (Story 1)
    - id_df    : identity discrimination / benign-malignant (Story 2)
    - ratio_df : geometric ratio property
    """
    dist_rows, id_rows, ratio_rows = [], [], []
    fn_list = list(DIST.values())
    rng     = random.Random(0)

    with torch.no_grad():
        for idx in tqdm(range(len(ds)), desc="  eval tables", ncols=80):
            A = ds.load(ds.paths[idx]).astype(np.float32)

            fn     = rng.choice(fn_list)
            a_low  = rng.uniform(0.0, 0.3)     # benign distortion
            a_high = rng.uniform(0.3, 1.0)     # malignant distortion
            if a_high < a_low:
                a_low, a_high = a_high, a_low

            P_low  = np.clip(np.asarray(fn(A.copy(), a_low),  dtype=np.float32), 0, 1)
            P_high = np.clip(np.asarray(fn(A.copy(), a_high), dtype=np.float32), 0, 1)

            j      = sample_hard_negative(idx, len(ds), rng, min_sep=HARD_NEG_SEP)
            N      = ds.load(ds.paths[j]).astype(np.float32)
            fn_nl, fn_nh = rng.choice(fn_list), rng.choice(fn_list)
            a_neg_l, a_neg_h = rng.uniform(0, 1), rng.uniform(0, 1)
            N_low  = np.clip(np.asarray(fn_nl(N.copy(), a_neg_l), dtype=np.float32), 0, 1)
            N_high = np.clip(np.asarray(fn_nh(N.copy(), a_neg_h), dtype=np.float32), 0, 1)

            dist_row_low  = {"a": float(a_low),  "lowa": int(a_low  > TAU), "band": "low"}
            dist_row_high = {"a": float(a_high), "lowa": int(a_high > TAU), "band": "high"}
            ratio_row     = {}

            for m in METRICS:
                d_aa = compute_pair_metric(m, model, A, A)
                d_pl = compute_pair_metric(m, model, A, P_low)
                d_ph = compute_pair_metric(m, model, A, P_high)
                d_nl = compute_pair_metric(m, model, A, N_low)
                d_nh = compute_pair_metric(m, model, A, N_high)

                dist_row_low[m]  = d_pl
                dist_row_high[m] = d_ph
                ratio_row[m]     = triplet_ratio(d_aa, d_pl, d_ph, d_nl, d_nh)

            dist_rows.append(dist_row_low)
            dist_rows.append(dist_row_high)
            ratio_rows.append(ratio_row)

            # Identity rows: same-patient positive pairs vs cross-patient negatives
            # label=0 -> same patient (positive), label=1 -> different patient (negative)
            P1_low  = np.clip(np.asarray(fn(A.copy(), a_low),  dtype=np.float32), 0, 1)
            P2_low  = np.clip(np.asarray(fn(A.copy(), a_low),  dtype=np.float32), 0, 1)
            P1_high = np.clip(np.asarray(fn(A.copy(), a_high), dtype=np.float32), 0, 1)
            P2_high = np.clip(np.asarray(fn(A.copy(), a_high), dtype=np.float32), 0, 1)

            id_rows_local = [
                {"label": 0, "band": "low",  "a": float(a_low),  "regime": "benign"},
                {"label": 1, "band": "low",  "a": float(a_low),  "regime": "benign"},
                {"label": 0, "band": "high", "a": float(a_high), "regime": "malignant"},
                {"label": 1, "band": "high", "a": float(a_high), "regime": "malignant"},
            ]
            pairs = [(P1_low, P2_low), (P1_low, N_low),
                     (P1_high, P2_high), (P1_high, N_high)]

            for row, (xa, xb) in zip(id_rows_local, pairs):
                for m in METRICS:
                    row[m] = compute_pair_metric(m, model, xa, xb)
            id_rows.extend(id_rows_local)

    return pd.DataFrame(dist_rows), pd.DataFrame(id_rows), pd.DataFrame(ratio_rows)


# ============================================================
# STORY 1+2: Distance vs alpha plot (THE main figure)
# ============================================================

def build_distance_vs_alpha(model, ds, n_samples=DIST_ALPHA_SAMPLES):
    """
    For each alpha level, sample n_samples anchor/positive pairs,
    compute distance for every metric, record mean +/- std.

    This plot directly shows:
    - Below TAU: ours stays flat (benign distortion ignored)
    - At TAU: elbow -- ours starts rising
    - Above TAU: ours rises sharply (malignant distortion detected)

    Classical metrics rise monotonically with no threshold behaviour.
    """
    fn_list = list(DIST.values())
    rng     = random.Random(0)
    results = {m: {"mean": [], "std": []} for m in METRICS}

    print("  Building distance-vs-alpha curves ...")
    for alpha in DIST_ALPHAS:
        dists = {m: [] for m in METRICS}
        for _ in range(n_samples):
            idx = rng.randint(0, len(ds) - 1)
            A   = ds.load(ds.paths[idx]).astype(np.float32)
            fn  = rng.choice(fn_list)
            P   = np.clip(
                np.asarray(fn(A.copy(), float(alpha)), dtype=np.float32),
                0.0, 1.0
            )
            for m in METRICS:
                dists[m].append(compute_pair_metric(m, model, A, P))

        for m in METRICS:
            results[m]["mean"].append(float(np.mean(dists[m])))
            results[m]["std"].append(float(np.std(dists[m])))
        print(f"    alpha={alpha:.1f} done")

    return results


def save_distance_vs_alpha_plot(dva_results):
    """
    Main paper figure: distance vs alpha with TAU marked.
    Shows the elbow at TAU for 'ours' -- the threshold story.
    Includes shaded benign/malignant regions.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = plt.cm.tab10.colors

    for i, m in enumerate(METRICS):
        means  = np.array(dva_results[m]["mean"])
        stds   = np.array(dva_results[m]["std"])
        lw     = 2.5 if m == "ours" else 1.2
        ls     = "-"  if m == "ours" else "--"
        a_plot = 0.95 if m == "ours" else 0.70
        color  = colors[i % len(colors)]

        ax.plot(DIST_ALPHAS, means, label=m, linewidth=lw,
                linestyle=ls, alpha=a_plot, color=color)
        if m == "ours":
            ax.fill_between(DIST_ALPHAS,
                            means - stds, means + stds,
                            alpha=0.15, color=color,
                            label="ours (+/- 1 std)")

    # Shade regimes
    ax.axvspan(0.0, TAU, alpha=0.06, color="green", zorder=0)
    ax.axvspan(TAU, 1.0, alpha=0.06, color="red",   zorder=0)
    ax.axvline(x=TAU, color="red", linestyle=":",
               linewidth=1.8, label=f"TAU={TAU} (threshold)")

    # Annotation
    ax.text(TAU / 2,       0.95, "benign",    ha="center",
            fontsize=10, color="green", transform=ax.get_xaxis_transform())
    ax.text((TAU + 1) / 2, 0.95, "malignant", ha="center",
            fontsize=10, color="red",   transform=ax.get_xaxis_transform())

    ax.set_xlabel("Distortion severity alpha", fontsize=12)
    ax.set_ylabel("Mean pairwise distance", fontsize=12)
    ax.set_title(f"Distance vs distortion severity -- {MOD.upper()} ({DATASET})",
                 fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = f"distance_vs_alpha_{SUFFIX}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Distance-vs-alpha plot -> {out}")


# ============================================================
# DEGRADATION PLOT -- split at TAU
# ============================================================

def patient_code_from_path(path: str) -> str:
    return Path(path).stem.split("_")[0]

def average_precision(relevance: np.ndarray) -> float:
    hits = ap = 0.0
    for rank, rel in enumerate(relevance, start=1):
        if rel:
            hits += 1
            ap   += hits / rank
    n_rel = relevance.sum()
    return ap / n_rel if n_rel > 0 else 0.0

def compute_retrieval_metrics(dist_matrix, patient_ids, ks=RETRIEVAL_KS):
    n       = len(patient_ids)
    recalls = {k: [] for k in ks}
    aps, mrrs = [], []

    for i in range(n):
        dists    = dist_matrix[i].copy()
        dists[i] = np.inf
        ranked   = np.argsort(dists)
        rel      = np.array([patient_ids[ranked[j]] == patient_ids[i]
                             for j in range(n - 1)])

        for k in ks:
            recalls[k].append(rel[:k].mean())
        aps.append(average_precision(rel))
        hits = np.where(rel)[0]
        mrrs.append(1.0 / (hits[0] + 1) if len(hits) > 0 else 0.0)

    result = {f"R@{k}": float(np.mean(recalls[k])) for k in ks}
    result["mAP"] = float(np.mean(aps))
    result["MRR"] = float(np.mean(mrrs))
    return result

def precompute_embeddings(model, ds):
    return precompute_embeddings_paths(model, ds, ds.paths)

def precompute_embeddings_paths(model, ds, paths):
    m = model.module if hasattr(model, "module") else model
    embeddings = []
    m.eval()
    with torch.no_grad():
        for path in paths:
            img = ds.load(path).astype(np.float32)
            z, *_ = m.forward_once(to_tensor(img))
            embeddings.append(F.normalize(z, dim=1).cpu())
    return torch.cat(embeddings, dim=0)

def precompute_dino_embeddings(ds):
    return precompute_dino_embeddings_paths(ds, ds.paths)

def precompute_dino_embeddings_paths(ds, paths):
    dino = _get_dino()
    embeddings = []
    with torch.no_grad():
        for path in paths:
            img = ds.load(path).astype(np.float32)
            z   = dino(_to_dino_input(img))
            embeddings.append(F.normalize(z, dim=1).cpu())
    return torch.cat(embeddings, dim=0)

def build_retrieval_table(model, ds, patient_ids):
    fn_list = list(DIST.values())
    rng     = random.Random(RETRIEVAL_SEED)
    n_full  = len(ds.paths)

    # Cap test set size for O(n^2) pixel metric computation
    if n_full > RETRIEVAL_MAX_N:
        idx_sub     = sorted(random.Random(RETRIEVAL_SEED + 1)
                             .sample(range(n_full), RETRIEVAL_MAX_N))
        paths_sub   = [ds.paths[i] for i in idx_sub]
        pids_sub    = [patient_ids[i] for i in idx_sub]
        print(f"  Subsampling {n_full} -> {RETRIEVAL_MAX_N} images for pixel metrics")
    else:
        idx_sub   = list(range(n_full))
        paths_sub = ds.paths
        pids_sub  = patient_ids

    n = len(paths_sub)
    print(f"  Loading {n} clean images ...")
    clean_imgs = [ds.load(p).astype(np.float32) for p in paths_sub]

    print("  Precomputing ours embeddings ...")
    Z_clean      = precompute_embeddings_paths(model, ds, paths_sub)
    print("  Precomputing dino embeddings ...")
    Z_dino_clean = precompute_dino_embeddings_paths(ds, paths_sub)

    rows = []
    for alpha in RETRIEVAL_ALPHAS:
        print(f"\n  Alpha = {alpha:.1f}")
        if alpha == 0.0:
            Z_query      = Z_clean
            Z_dino_query = Z_dino_clean
            query_imgs   = clean_imgs
        else:
            q_emb, q_dino_emb, query_imgs = [], [], []
            dino_model = _get_dino()
            m = model.module if hasattr(model, "module") else model
            with torch.no_grad():
                for i, path in enumerate(paths_sub):
                    img = clean_imgs[i].copy()
                    fn  = rng.choice(fn_list)
                    dist_img = np.clip(
                        np.asarray(fn(img, float(alpha)), dtype=np.float32), 0, 1)
                    query_imgs.append(dist_img)
                    z, *_ = m.forward_once(to_tensor(dist_img))
                    q_emb.append(F.normalize(z, dim=1).cpu())
                    zd = dino_model(_to_dino_input(dist_img))
                    q_dino_emb.append(F.normalize(zd, dim=1).cpu())
            Z_query      = torch.cat(q_emb,      dim=0)
            Z_dino_query = torch.cat(q_dino_emb, dim=0)

        D_ours = 0.5 * (1.0 - (Z_query @ Z_clean.T).numpy())
        D_dino = 0.5 * (1.0 - (Z_dino_query @ Z_dino_clean.T).numpy())

        pixel_metrics = ["mse", "ssim", "ncc", "mi", "grad"]
        D_pixel = {}
        for pm in pixel_metrics:
            print(f"    {pm} ...", end=" ", flush=True)
            D = np.zeros((n, n), dtype=np.float32)
            for i in tqdm(range(n), desc=f"      {pm}", ncols=60,
                          leave=False, disable=(n <= 50)):
                for j in range(n):
                    D[i, j] = 0.0 if i == j else \
                        compute_pair_metric(pm, None, query_imgs[i], clean_imgs[j])
            D_pixel[pm] = D
            print("done")

        all_D = {"ours": D_ours, "dino": D_dino}
        all_D.update(D_pixel)

        for m_name, D in all_D.items():
            ret = compute_retrieval_metrics(D, pids_sub)
            row = {"metric": m_name, "alpha": alpha}
            row.update(ret)
            rows.append(row)

    return pd.DataFrame(rows)


def save_degradation_plot(ret_df):
    """
    Relative degradation split at TAU:
    Panel 1: full range -- ours flattest overall
    Panel 2: below TAU (benign) -- ours most robust
    Panel 3: above TAU (malignant) -- ours drops (intended behaviour)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.tab10.colors

    for i, m in enumerate(METRICS):
        sub = ret_df[ret_df["metric"] == m].sort_values("alpha")
        if sub.empty:
            continue
        base = sub[sub["alpha"] == 0.0]["mAP"].values
        if len(base) == 0 or base[0] < 1e-6:
            continue

        rel    = sub["mAP"].values / base[0]
        alphas = sub["alpha"].values
        lw     = 2.5 if m == "ours" else 1.2
        ls     = "-"  if m == "ours" else "--"
        a_plot = 0.95 if m == "ours" else 0.70
        color  = colors[i % len(colors)]

        axes[0].plot(alphas, rel, label=m, linewidth=lw,
                     linestyle=ls, alpha=a_plot, color=color)

        # Below TAU
        mask_b = alphas <= TAU
        axes[1].plot(alphas[mask_b], rel[mask_b], label=m,
                     linewidth=lw, linestyle=ls, alpha=a_plot, color=color)

        # Above TAU
        mask_a = alphas >= TAU
        axes[2].plot(alphas[mask_a], rel[mask_a], label=m,
                     linewidth=lw, linestyle=ls, alpha=a_plot, color=color)

    # Compute dynamic ylim per panel from actual data
    all_rel    = {}   # metric -> full rel array
    all_rel_b  = {}   # metric -> benign rel array
    all_rel_m  = {}   # metric -> malignant rel array

    for m in METRICS:
        sub = ret_df[ret_df["metric"] == m].sort_values("alpha")
        if sub.empty: continue
        base = sub[sub["alpha"] == 0.0]["mAP"].values
        if len(base) == 0 or base[0] < 1e-6: continue
        rel    = sub["mAP"].values / base[0]
        alphas = sub["alpha"].values
        all_rel[m]   = rel
        all_rel_b[m] = rel[alphas <= TAU]
        all_rel_m[m] = rel[alphas >= TAU]

    def dynamic_ylim(data_dict, pad=0.02):
        """Return (ymin, ymax) with padding, snapped to sensible range."""
        all_vals = np.concatenate(list(data_dict.values()))
        lo = max(0.0, all_vals.min() - pad)
        hi = min(1.05, all_vals.max() + pad)
        # always show the 1.0 reference line
        hi = max(hi, 1.0 + pad)
        return lo, hi

    ylim_full    = dynamic_ylim(all_rel)
    ylim_benign  = dynamic_ylim(all_rel_b)
    ylim_malign  = dynamic_ylim(all_rel_m)

    titles = [
        f"Full range -- {MOD.upper()}",
        f"Benign regime (alpha <= {TAU}) -- {MOD.upper()}",
        f"Malignant regime (alpha >= {TAU}) -- {MOD.upper()}",
    ]
    ylims = [ylim_full, ylim_benign, ylim_malign]

    for ax, title, ylim in zip(axes, titles, ylims):
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Distortion severity alpha", fontsize=11)
        ax.set_ylabel("mAP relative to alpha=0", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=7, ncol=2)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)

    axes[1].axvspan(0.0, TAU, alpha=0.08, color="green", zorder=0)
    axes[2].axvspan(TAU, 1.0, alpha=0.08, color="red",   zorder=0)

    fig.tight_layout()
    out = f"degradation_{SUFFIX}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Degradation plot -> {out}")


def print_tau_split_summary(ret_df):
    """
    Print robustness below TAU and sensitivity above TAU.
    These are the numbers that support the threshold story.
    """
    print(f"\n  Robustness in benign regime (alpha <= {TAU}):")
    print(f"  {'metric':8s}  mAP retention  (higher = more robust)")
    print("  " + "-" * 42)
    rows = []
    for m in METRICS:
        sub  = ret_df[(ret_df["metric"] == m) & (ret_df["alpha"] <= TAU)]
        base = ret_df[(ret_df["metric"] == m) & (ret_df["alpha"] == 0.0)]["mAP"].values
        if len(base) == 0 or base[0] < 1e-6:
            continue
        rows.append((m, float((sub["mAP"] / base[0]).mean())))
    for m, r in sorted(rows, key=lambda x: -x[1]):
        marker = " <-- ours" if m == "ours" else ""
        print(f"  {m:8s}  {r:.3f}{marker}")

    print(f"\n  Sensitivity in malignant regime (alpha >= {TAU}):")
    print(f"  {'metric':8s}  mAP retention  (lower = more sensitive to distortion)")
    print("  " + "-" * 56)
    rows = []
    for m in METRICS:
        sub  = ret_df[(ret_df["metric"] == m) & (ret_df["alpha"] >= TAU)]
        base = ret_df[(ret_df["metric"] == m) & (ret_df["alpha"] == 0.0)]["mAP"].values
        if len(base) == 0 or base[0] < 1e-6:
            continue
        rows.append((m, float((sub["mAP"] / base[0]).mean())))
    for m, r in sorted(rows, key=lambda x: x[1]):
        marker = " <-- ours" if m == "ours" else ""
        print(f"  {m:8s}  {r:.3f}{marker}")


# ============================================================
# BOOTSTRAP SUMMARY (Stories 1 + 2 headline metrics)
# ============================================================

def bootstrap_summary(dist_df, id_df, ratio_df, n_boot=100):
    rng = np.random.default_rng(42)

    dist_spear_a    = {m: [] for m in METRICS}
    dist_spear_lowa = {m: [] for m in METRICS}
    dist_rank       = {m: [] for m in METRICS}
    id_spear        = {m: [] for m in METRICS}
    id_auc          = {m: [] for m in METRICS}
    id_auc_benign   = {m: [] for m in METRICS}
    id_auc_malignant= {m: [] for m in METRICS}
    ratio_mean      = {m: [] for m in METRICS}

    for _ in range(n_boot):
        dist_bs  = dist_df.sample(n=len(dist_df),   replace=True,
                                  random_state=int(rng.integers(0, 2**31-1)))
        id_bs    = id_df.sample(n=len(id_df),       replace=True,
                                random_state=int(rng.integers(0, 2**31-1)))
        ratio_bs = ratio_df.sample(n=len(ratio_df), replace=True,
                                   random_state=int(rng.integers(0, 2**31-1)))

        for m in METRICS:
            vals = dist_bs[m].values
            dist_spear_a[m].append(safe_spearman(vals, dist_bs["a"].values))
            dist_spear_lowa[m].append(safe_spearman(vals, dist_bs["lowa"].values))
            dist_rank[m].append(ranking_accuracy(vals, dist_bs["a"].values))

            vals_id = id_bs[m].values
            labels  = id_bs["label"].values
            id_spear[m].append(safe_spearman(vals_id, labels))
            id_auc[m].append(safe_auc(labels, vals_id))

            # AUC separately for benign and malignant regimes (Story 2)
            ben = id_bs[id_bs["regime"] == "benign"]
            mal = id_bs[id_bs["regime"] == "malignant"]
            id_auc_benign[m].append(
                safe_auc(ben["label"].values, ben[m].values) if len(ben) > 0 else 0.0)
            id_auc_malignant[m].append(
                safe_auc(mal["label"].values, mal[m].values) if len(mal) > 0 else 0.0)

            ratio_mean[m].append(float(np.mean(ratio_bs[m].values)))

    return {
        "dist_spear_a":     dist_spear_a,
        "dist_spear_lowa":  dist_spear_lowa,
        "dist_rank":        dist_rank,
        "id_spear":         id_spear,
        "id_auc":           id_auc,
        "id_auc_benign":    id_auc_benign,
        "id_auc_malignant": id_auc_malignant,
        "ratio_mean":       ratio_mean,
    }


def mean_std(x):
    x = np.asarray(x, dtype=np.float64)
    return (0.0, 0.0) if len(x) == 0 else (float(np.mean(x)), float(np.std(x)))


# ============================================================
# SUMMARY VISUAL -- headline table (no absolute mAP)
# ============================================================

def save_summary_visual(summary_df):
    """
    Headline table showing Story 1 and Story 2 metrics.
    Absolute retrieval mAP deliberately excluded.
    Columns: Spearman(a) | Rank Acc | Ratio | AUC | AUC_benign | AUC_malignant
    """
    cols = [
        "metric",
        "dist_spearman_a_mean",
        "dist_rank_mean",
        "ratio_mean",
        "id_auc_mean",
        "id_auc_benign_mean",
        "id_auc_malignant_mean",
    ]
    # Only include columns that exist
    cols = [c for c in cols if c == "metric" or c in summary_df.columns]
    vis_df = summary_df[cols].copy()

    # Higher is better for all except ratio (lower is better)
    highlight_max_cols = [c for c in cols[1:] if c != "ratio_mean"]
    highlight_min_cols = ["ratio_mean"] if "ratio_mean" in cols else []

    fmt = {c: "{:.4f}" for c in cols if c != "metric"}
    styler = vis_df.style.format(fmt)
    if highlight_max_cols:
        styler = styler.highlight_max(
            subset=highlight_max_cols, axis=0, color="#fff2a8")
    if highlight_min_cols:
        styler = styler.highlight_min(
            subset=highlight_min_cols, axis=0, color="#d6f5d6")
    styler.to_html(f"summary_{SUFFIX}.html")

    display_cols = cols[1:]
    n_cols = len(display_cols)
    fig, ax = plt.subplots(figsize=(2.2 * n_cols + 1, 0.7 + 0.45 * len(vis_df)))
    ax.axis("off")

    cell_text = [[f"{vis_df.iloc[i][c]:.4f}" for c in display_cols]
                 for i in range(len(vis_df))]
    table = ax.table(
        cellText=cell_text,
        rowLabels=vis_df["metric"].tolist(),
        colLabels=display_cols,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    max_vals = {c: vis_df[c].max() for c in highlight_max_cols if c in vis_df}
    min_ratio = vis_df["ratio_mean"].min() if "ratio_mean" in vis_df else None

    for i in range(len(vis_df)):
        for cidx, col_name in enumerate(display_cols):
            val = vis_df.iloc[i][col_name]
            if col_name == "ratio_mean" and min_ratio is not None:
                if np.isclose(val, min_ratio):
                    table[(i+1, cidx)].set_facecolor("#d6f5d6")
            elif col_name in max_vals and np.isclose(val, max_vals[col_name]):
                table[(i+1, cidx)].set_facecolor("#fff2a8")

    fig.tight_layout()
    fig.savefig(f"summary_{SUFFIX}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Summary table -> summary_{SUFFIX}.png / .html")


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"=== EVAL ({MOD.upper()}) -- {DATASET} ===\n")

    print("Loading DINO ViT-S/8 ...")
    _get_dino()
    print("DINO ready.\n")

    ds = TripletDataset(test_path, MOD)
    model = DisentangledSiamese()
    model.load_state_dict(torch.load(f"best_{SUFFIX}.pth", map_location="cpu"))

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for inference.")
        model = torch.nn.DataParallel(model)

    model = model.to(device)
    model.eval()

    # ---- Story 1 + 2: Spearman / AUC / Ratio eval ---------
    print("Building distortion and identity eval tables ...")
    try:
        dist_df, id_df, ratio_df = build_eval_tables(model, ds)
    except Exception as e:
        print(f"ERROR in build_eval_tables: {e}")
        traceback.print_exc()
        raise

    dist_df.to_csv(f"eval_distortion_{SUFFIX}.csv",  index=False)
    id_df.to_csv(f"eval_identity_{SUFFIX}.csv",      index=False)
    ratio_df.to_csv(f"eval_ratio_{SUFFIX}.csv",      index=False)
    print("Saved distortion / identity / ratio CSVs.")

    try:
        stats = bootstrap_summary(dist_df, id_df, ratio_df, n_boot=BOOTSTRAPS)
    except Exception as e:
        print(f"ERROR in bootstrap_summary: {e}")
        traceback.print_exc()
        raise

    # Print Story 1 results
    print("\n=== STORY 1: DISTORTION AWARENESS ===")
    print(f"  {'metric':8s}  Spearman(a)  RankAcc   Ratio")
    print("  " + "-" * 45)
    for m in METRICS:
        da_m, da_s = mean_std(stats["dist_spear_a"][m])
        dr_m, dr_s = mean_std(stats["dist_rank"][m])
        rr_m, rr_s = mean_std(stats["ratio_mean"][m])
        marker = " <--" if m == "ours" else ""
        print(f"  {m:8s}  {da_m:.4f}+-{da_s:.4f}  "
              f"{dr_m:.4f}+-{dr_s:.4f}  "
              f"{rr_m:.4f}+-{rr_s:.4f}{marker}")

    # Print Story 2 results
    print("\n=== STORY 2: BENIGN / MALIGNANT CLASSIFICATION ===")
    print(f"  {'metric':8s}  AUC(all)   AUC(benign)  AUC(malignant)")
    print("  " + "-" * 55)
    for m in METRICS:
        ia_m,  ia_s  = mean_std(stats["id_auc"][m])
        ib_m,  ib_s  = mean_std(stats["id_auc_benign"][m])
        im_m,  im_s  = mean_std(stats["id_auc_malignant"][m])
        marker = " <--" if m == "ours" else ""
        print(f"  {m:8s}  {ia_m:.4f}+-{ia_s:.4f}  "
              f"{ib_m:.4f}+-{ib_s:.4f}  "
              f"{im_m:.4f}+-{im_s:.4f}{marker}")

    # ---- Distance-vs-alpha plot (main figure) --------------
    print("\nBuilding distance-vs-alpha plot ...")
    try:
        dva_results = build_distance_vs_alpha(model, ds)
        save_distance_vs_alpha_plot(dva_results)
        # Save raw data
        dva_rows = []
        for m in METRICS:
            for alpha, mean_v, std_v in zip(
                DIST_ALPHAS,
                dva_results[m]["mean"],
                dva_results[m]["std"]
            ):
                dva_rows.append({"metric": m, "alpha": alpha,
                                 "mean": mean_v, "std": std_v})
        pd.DataFrame(dva_rows).to_csv(
            f"distance_vs_alpha_{SUFFIX}.csv", index=False)
    except Exception as e:
        print(f"  ERROR in distance-vs-alpha: {e}")
        traceback.print_exc()

    # ---- Retrieval / degradation (supporting figures) ------
    print("\n=== DEGRADATION ANALYSIS (supporting) ===")
    patient_ids = [patient_code_from_path(p) for p in ds.paths]
    n_patients  = len(set(patient_ids))
    print(f"  {len(ds.paths)} test images, {n_patients} unique patients")
    print(f"  Patient codes: {sorted(set(patient_ids))[:10]}")

    if n_patients < 2:
        print("  Skipping degradation -- fewer than 2 unique patients.")
        ret_df = None
    else:
        try:
            ret_df = build_retrieval_table(model, ds, patient_ids)
            ret_df.to_csv(f"retrieval_{SUFFIX}.csv", index=False)
            save_degradation_plot(ret_df)
            print_tau_split_summary(ret_df)
        except Exception as e:
            print(f"  ERROR in retrieval/degradation: {e}")
            traceback.print_exc()
            ret_df = None

    # ---- Summary table -------------------------------------
    summary_rows = []
    for m in METRICS:
        row = {
            "metric":                   m,
            "dist_spearman_a_mean":     mean_std(stats["dist_spear_a"][m])[0],
            "dist_spearman_a_std":      mean_std(stats["dist_spear_a"][m])[1],
            "dist_spearman_lowa_mean":  mean_std(stats["dist_spear_lowa"][m])[0],
            "dist_spearman_lowa_std":   mean_std(stats["dist_spear_lowa"][m])[1],
            "dist_rank_mean":           mean_std(stats["dist_rank"][m])[0],
            "dist_rank_std":            mean_std(stats["dist_rank"][m])[1],
            "id_auc_mean":              mean_std(stats["id_auc"][m])[0],
            "id_auc_std":              mean_std(stats["id_auc"][m])[1],
            "id_auc_benign_mean":       mean_std(stats["id_auc_benign"][m])[0],
            "id_auc_benign_std":        mean_std(stats["id_auc_benign"][m])[1],
            "id_auc_malignant_mean":    mean_std(stats["id_auc_malignant"][m])[0],
            "id_auc_malignant_std":     mean_std(stats["id_auc_malignant"][m])[1],
            "ratio_mean":               mean_std(stats["ratio_mean"][m])[0],
            "ratio_std":                mean_std(stats["ratio_mean"][m])[1],
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"summary_{SUFFIX}.csv", index=False)
    save_summary_visual(summary_df)

    print("\nSaved:")
    print(f"  eval_distortion_{SUFFIX}.csv")
    print(f"  eval_identity_{SUFFIX}.csv")
    print(f"  eval_ratio_{SUFFIX}.csv")
    print(f"  distance_vs_alpha_{SUFFIX}.csv / .png")
    print(f"  summary_{SUFFIX}.csv / .html / .png")
    if ret_df is not None:
        print(f"  retrieval_{SUFFIX}.csv")
        print(f"  degradation_{SUFFIX}.png")


if __name__ == "__main__":
    main()