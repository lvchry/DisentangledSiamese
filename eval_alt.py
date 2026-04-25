# -*- coding: utf-8 -*-

import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from skimage.metrics import structural_similarity as ssim

from dataset_alt import TripletDataset
from model import DisentangledSiamese
from Distortions_Continuous import CT_DIST, MRI_DIST


MOD = "mr"  # "ct" or "mr"
BOOTSTRAPS = 100
TAU = 0.3
HARD_NEG_SEP = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE = "/home1/zyan6527/GoldAtlas_Finale/pairedDatas"
test_path = os.path.join(BASE, "testA" if MOD == "ct" else "testB")

SUFFIX = f"{MOD}_alt1"

DIST = CT_DIST if MOD == "ct" else MRI_DIST
METRICS = ["ours", "mse", "ssim", "ncc", "mi"]


def safe_spearman(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) != len(y) or len(x) == 0:
        return 0.0
    r = spearmanr(x, y).correlation
    if r is None or np.isnan(r):
        return 0.0
    return float(r)


def safe_auc(y_true, y_score):
    try:
        return float(roc_auc_score(y_true, y_score))
    except:
        return 0.0


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.sqrt((a ** 2).mean()) * np.sqrt((b ** 2).mean())) + 1e-8
    return float((a * b).mean() / denom)


def mutual_information(a, b, bins=32):
    a = (a * (bins - 1)).astype(int)
    b = (b * (bins - 1)).astype(int)

    joint, _, _ = np.histogram2d(a.flatten(), b.flatten(), bins=bins)
    joint = joint / (np.sum(joint) + 1e-8)

    pa = joint.sum(axis=1)
    pb = joint.sum(axis=0)

    nz = joint > 0
    return float(np.sum(joint[nz] * np.log(joint[nz] / (pa[:, None] * pb[None, :])[nz])))


def ranking_accuracy(preds, a):
    preds = np.asarray(preds)
    a = np.asarray(a)

    correct = 0
    total = 0
    for i in range(len(preds)):
        for j in range(len(preds)):
            if a[i] > a[j]:
                total += 1
                if preds[i] > preds[j]:
                    correct += 1

    if total == 0:
        return 0.0
    return correct / total


def sample_hard_negative(idx, n, min_sep=10):
    min_sep = min(min_sep, n - 1)

    left = list(range(0, max(0, idx - min_sep + 1)))
    right = list(range(min(n, idx + min_sep), n))
    candidates = left + right

    if not candidates:
        candidates = [j for j in range(n) if j != idx]

    return random.choice(candidates)


def to_tensor(x):
    x = np.asarray(x, dtype=np.float32)
    return torch.from_numpy(x).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)


def anat_distance(model, x, y):
    with torch.no_grad():
        z1, *_ = model.forward_once(to_tensor(x))
        z2, *_ = model.forward_once(to_tensor(y))
        score = F.cosine_similarity(z1, z2, dim=1)
        dist = 0.5 * (1.0 - score)
        return float(dist.item())


def compute_pair_metric(metric, model, x, y):
    if metric == "ours":
        return anat_distance(model, x, y)
    if metric == "mse":
        return float(np.mean((x - y) ** 2))
    if metric == "ssim":
        try:
            return float(1.0 - ssim(x, y, data_range=1.0))
        except:
            return 0.0
    if metric == "ncc":
        try:
            return float(1.0 - ncc(x, y))
        except:
            return 0.0
    if metric == "mi":
        try:
            return float(-mutual_information(x, y))
        except:
            return 0.0
    return 0.0


def triplet_ratio(d_aa, d_pl, d_ph, d_nl, d_nh, eps=1e-8):
    vals = []
    for d_p in [d_pl, d_ph]:
        for d_n in [d_nl, d_nh]:
            denom = abs(d_aa - d_n)
            vals.append(abs(d_aa - d_p) / (denom + eps))
    return float(np.mean(vals))


def build_eval_tables(model, ds):
    dist_rows = []
    id_rows = []
    ratio_rows = []

    with torch.no_grad():
        for idx in range(len(ds)):
            A = ds.load(ds.paths[idx]).astype(np.float32)

            # One distortion family per anchor.
            fn = random.choice(list(DIST.values()))

            a_low = random.uniform(0.0, 0.3)
            a_high = random.uniform(0.3, 1.0)
            if a_high < a_low:
                a_low, a_high = a_high, a_low

            P_low = np.asarray(fn(A.copy(), a_low), dtype=np.float32)
            P_high = np.asarray(fn(A.copy(), a_high), dtype=np.float32)
            P_low = np.clip(P_low, 0.0, 1.0)
            P_high = np.clip(P_high, 0.0, 1.0)

            j = sample_hard_negative(idx, len(ds), min_sep=HARD_NEG_SEP)
            N = ds.load(ds.paths[j]).astype(np.float32)

            N_low = np.asarray(fn(N.copy(), a_low), dtype=np.float32)
            N_high = np.asarray(fn(N.copy(), a_high), dtype=np.float32)
            N_low = np.clip(N_low, 0.0, 1.0)
            N_high = np.clip(N_high, 0.0, 1.0)

            # Distortion-only rows.
            dist_row_low = {
                "a": float(a_low),
                "lowa": int(a_low > TAU),
                "band": "low",
            }
            dist_row_high = {
                "a": float(a_high),
                "lowa": int(a_high > TAU),
                "band": "high",
            }

            # Ratio row for this anchor.
            ratio_row = {}

            for m in METRICS:
                d_aa = compute_pair_metric(m, model, A, A)
                d_pl = compute_pair_metric(m, model, A, P_low)
                d_ph = compute_pair_metric(m, model, A, P_high)
                d_nl = compute_pair_metric(m, model, A, N_low)
                d_nh = compute_pair_metric(m, model, A, N_high)

                dist_row_low[m] = d_pl
                dist_row_high[m] = d_ph

                ratio_row[m] = triplet_ratio(d_aa, d_pl, d_ph, d_nl, d_nh)

            dist_rows.append(dist_row_low)
            dist_rows.append(dist_row_high)
            ratio_rows.append(ratio_row)

            # Identity-only rows.
            P1_low = np.asarray(fn(A.copy(), a_low), dtype=np.float32)
            P2_low = np.asarray(fn(A.copy(), a_low), dtype=np.float32)
            P1_high = np.asarray(fn(A.copy(), a_high), dtype=np.float32)
            P2_high = np.asarray(fn(A.copy(), a_high), dtype=np.float32)

            P1_low = np.clip(P1_low, 0.0, 1.0)
            P2_low = np.clip(P2_low, 0.0, 1.0)
            P1_high = np.clip(P1_high, 0.0, 1.0)
            P2_high = np.clip(P2_high, 0.0, 1.0)

            id_row_pos_low = {
                "label": 0,
                "band": "low",
                "a": float(a_low),
            }
            id_row_neg_low = {
                "label": 1,
                "band": "low",
                "a": float(a_low),
            }
            id_row_pos_high = {
                "label": 0,
                "band": "high",
                "a": float(a_high),
            }
            id_row_neg_high = {
                "label": 1,
                "band": "high",
                "a": float(a_high),
            }

            for m in METRICS:
                id_row_pos_low[m] = compute_pair_metric(m, model, P1_low, P2_low)
                id_row_neg_low[m] = compute_pair_metric(m, model, P1_low, N_low)
                id_row_pos_high[m] = compute_pair_metric(m, model, P1_high, P2_high)
                id_row_neg_high[m] = compute_pair_metric(m, model, P1_high, N_high)

            id_rows.append(id_row_pos_low)
            id_rows.append(id_row_neg_low)
            id_rows.append(id_row_pos_high)
            id_rows.append(id_row_neg_high)

    return pd.DataFrame(dist_rows), pd.DataFrame(id_rows), pd.DataFrame(ratio_rows)


def bootstrap_summary(dist_df, id_df, ratio_df, n_boot=100):
    rng = np.random.default_rng(42)

    dist_spear_a = {m: [] for m in METRICS}
    dist_spear_lowa = {m: [] for m in METRICS}
    dist_rank = {m: [] for m in METRICS}

    id_spear = {m: [] for m in METRICS}
    id_auc = {m: [] for m in METRICS}

    ratio_mean = {m: [] for m in METRICS}

    for _ in range(n_boot):
        dist_bs = dist_df.sample(
            n=len(dist_df),
            replace=True,
            random_state=int(rng.integers(0, 2**31 - 1))
        )
        id_bs = id_df.sample(
            n=len(id_df),
            replace=True,
            random_state=int(rng.integers(0, 2**31 - 1))
        )
        ratio_bs = ratio_df.sample(
            n=len(ratio_df),
            replace=True,
            random_state=int(rng.integers(0, 2**31 - 1))
        )

        for m in METRICS:
            vals = dist_bs[m].values
            dist_spear_a[m].append(safe_spearman(vals, dist_bs["a"].values))
            dist_spear_lowa[m].append(safe_spearman(vals, dist_bs["lowa"].values))
            dist_rank[m].append(ranking_accuracy(vals, dist_bs["a"].values))

            vals_id = id_bs[m].values
            labels = id_bs["label"].values
            id_spear[m].append(safe_spearman(vals_id, labels))
            id_auc[m].append(safe_auc(labels, vals_id))

            ratio_mean[m].append(float(np.mean(ratio_bs[m].values)))

    return {
        "dist_spear_a": dist_spear_a,
        "dist_spear_lowa": dist_spear_lowa,
        "dist_rank": dist_rank,
        "id_spear": id_spear,
        "id_auc": id_auc,
        "ratio_mean": ratio_mean,
    }


def mean_std(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return 0.0, 0.0
    return float(np.mean(x)), float(np.std(x))


def save_summary_visual(summary_df):
    cols = [
        "metric",
        "dist_spearman_a_mean",
        "dist_spearman_lowa_mean",
        "dist_rank_mean",
        "id_spearman_mean",
        "id_auc_mean",
        "ratio_mean",
    ]

    vis_df = summary_df[cols].copy()

    styler = vis_df.style.format({
        "dist_spearman_a_mean": "{:.4f}",
        "dist_spearman_lowa_mean": "{:.4f}",
        "dist_rank_mean": "{:.4f}",
        "id_spearman_mean": "{:.4f}",
        "id_auc_mean": "{:.4f}",
        "ratio_mean": "{:.4f}",
    }).highlight_max(
        subset=[
            "dist_spearman_a_mean",
            "dist_spearman_lowa_mean",
            "dist_rank_mean",
            "id_spearman_mean",
            "id_auc_mean",
        ],
        axis=0,
        color="#fff2a8",
    ).highlight_min(
        subset=["ratio_mean"],
        axis=0,
        color="#d6f5d6",
    )

    styler.to_html(f"summary_{SUFFIX}.html")

    fig, ax = plt.subplots(figsize=(15, 0.7 + 0.45 * len(vis_df)))
    ax.axis("off")

    display_cols = cols[1:]
    cell_text = []
    for _, row in vis_df.iterrows():
        cell_text.append([
            f"{row['dist_spearman_a_mean']:.4f}",
            f"{row['dist_spearman_lowa_mean']:.4f}",
            f"{row['dist_rank_mean']:.4f}",
            f"{row['id_spearman_mean']:.4f}",
            f"{row['id_auc_mean']:.4f}",
            f"{row['ratio_mean']:.4f}",
        ])

    table = ax.table(
        cellText=cell_text,
        rowLabels=vis_df["metric"].tolist(),
        colLabels=display_cols,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    max_cols = {
        "dist_spearman_a_mean": vis_df["dist_spearman_a_mean"].max(),
        "dist_spearman_lowa_mean": vis_df["dist_spearman_lowa_mean"].max(),
        "dist_rank_mean": vis_df["dist_rank_mean"].max(),
        "id_spearman_mean": vis_df["id_spearman_mean"].max(),
        "id_auc_mean": vis_df["id_auc_mean"].max(),
    }
    min_ratio = vis_df["ratio_mean"].min()

    col_idx = {
        "dist_spearman_a_mean": 0,
        "dist_spearman_lowa_mean": 1,
        "dist_rank_mean": 2,
        "id_spearman_mean": 3,
        "id_auc_mean": 4,
        "ratio_mean": 5,
    }

    for i in range(len(vis_df)):
        for col_name, idx in col_idx.items():
            val = vis_df.iloc[i][col_name]
            if col_name == "ratio_mean":
                if np.isclose(val, min_ratio):
                    table[(i + 1, idx)].set_facecolor("#d6f5d6")
            else:
                if np.isclose(val, max_cols[col_name]):
                    table[(i + 1, idx)].set_facecolor("#fff2a8")

    fig.tight_layout()
    fig.savefig(f"summary_{SUFFIX}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    print(f"=== EVAL ({MOD}) ===")

    ds = TripletDataset(test_path, MOD)
    model = DisentangledSiamese().to(device)
    model.load_state_dict(torch.load(f"best_{SUFFIX}.pth", map_location=device))
    model.eval()

    dist_df, id_df, ratio_df = build_eval_tables(model, ds)
    stats = bootstrap_summary(dist_df, id_df, ratio_df, n_boot=BOOTSTRAPS)

    print("\n=== FINAL RESULTS ===")

    for m in METRICS:
        print("\n" + m)

        da_mean, da_std = mean_std(stats["dist_spear_a"][m])
        dl_mean, dl_std = mean_std(stats["dist_spear_lowa"][m])
        dr_mean, dr_std = mean_std(stats["dist_rank"][m])
        is_mean, is_std = mean_std(stats["id_spear"][m])
        ia_mean, ia_std = mean_std(stats["id_auc"][m])
        rr_mean, rr_std = mean_std(stats["ratio_mean"][m])

        print("Distortion:")
        print("  Spearman(a):   ", f"{da_mean:.4f}", "+-", f"{da_std:.4f}")
        print("  Spearman(lowa): ", f"{dl_mean:.4f}", "+-", f"{dl_std:.4f}")
        print("  Rank Acc:      ", f"{dr_mean:.4f}", "+-", f"{dr_std:.4f}")

        print("Identity:")
        print("  Spearman(label):", f"{is_mean:.4f}", "+-", f"{is_std:.4f}")
        print("  AUC:            ", f"{ia_mean:.4f}", "+-", f"{ia_std:.4f}")

        print("Ratio:")
        print("  Triplet ratio:  ", f"{rr_mean:.4f}", "+-", f"{rr_std:.4f}")

    dist_df.to_csv(f"eval_distortion_{SUFFIX}.csv", index=False)
    id_df.to_csv(f"eval_identity_{SUFFIX}.csv", index=False)
    ratio_df.to_csv(f"eval_ratio_{SUFFIX}.csv", index=False)

    summary_rows = []
    for m in METRICS:
        summary_rows.append({
            "metric": m,
            "dist_spearman_a_mean": mean_std(stats["dist_spear_a"][m])[0],
            "dist_spearman_a_std": mean_std(stats["dist_spear_a"][m])[1],
            "dist_spearman_lowa_mean": mean_std(stats["dist_spear_lowa"][m])[0],
            "dist_spearman_lowa_std": mean_std(stats["dist_spear_lowa"][m])[1],
            "dist_rank_mean": mean_std(stats["dist_rank"][m])[0],
            "dist_rank_std": mean_std(stats["dist_rank"][m])[1],
            "id_spearman_mean": mean_std(stats["id_spear"][m])[0],
            "id_spearman_std": mean_std(stats["id_spear"][m])[1],
            "id_auc_mean": mean_std(stats["id_auc"][m])[0],
            "id_auc_std": mean_std(stats["id_auc"][m])[1],
            "ratio_mean": mean_std(stats["ratio_mean"][m])[0],
            "ratio_std": mean_std(stats["ratio_mean"][m])[1],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"summary_{SUFFIX}.csv", index=False)

    save_summary_visual(summary_df)

    print("\nSaved:")
    print(f"  eval_distortion_{SUFFIX}.csv")
    print(f"  eval_identity_{SUFFIX}.csv")
    print(f"  eval_ratio_{SUFFIX}.csv")
    print(f"  summary_{SUFFIX}.csv")
    print(f"  summary_{SUFFIX}.html")
    print(f"  summary_{SUFFIX}.png")


if __name__ == "__main__":
    main()
