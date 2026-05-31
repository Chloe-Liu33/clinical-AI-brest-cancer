"""Build 5-year and 10-year calibration plots from `ablation_results_cindex.pkl`.

Pre-req: main_C-index.py has been patched to also save per-seed arrays
  train_risks/train_times/train_events and test_risks/test_times/test_events.

For each chosen model:
  * Pool the 5 seeds (Breslow fitted per seed on its own train set;
    test predictions S_i(t) are concatenated across seeds for plotting).
  * Plot bin + smoothed calibration at t = 60 (5y) and t = 120 (10y).
  * Print Brier / ICI / E50 / E90, mean ± std across seeds.

Usage:
    python make_calibration.py
    python make_calibration.py --models "Ours (HSIC K=512)" "Clinical-only (MLP)"
"""
from __future__ import annotations
import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import calibration as cal


# ---------------------------------------------------------------------------
def _per_seed_calibration(per_seed, t_eval):
    """Return dict: pooled S_pred + test_times/events across seeds, and
    per-seed metric scalars."""
    pooled_S, pooled_t, pooled_e = [], [], []
    metrics = []
    for s in per_seed:
        H0 = cal.breslow_baseline(s["train_risks"], s["train_times"], s["train_events"])
        S = cal.predict_survival(s["test_risks"], t_eval, H0)
        pooled_S.append(S)
        pooled_t.append(s["test_times"])
        pooled_e.append(s["test_events"])
        bs  = cal.brier_score(s["test_times"], s["test_events"], S, t_eval,
                              train_times=s["train_times"], train_events=s["train_events"])
        ici = cal.calibration_ici(s["test_times"], s["test_events"], S, t_eval)
        metrics.append(dict(brier=bs, **ici))
    return (np.concatenate(pooled_S),
            np.concatenate(pooled_t),
            np.concatenate(pooled_e),
            metrics)


def _summarise(metrics):
    out = {}
    for k in ("brier", "ICI", "E50", "E90"):
        v = np.array([m[k] for m in metrics])
        out[k] = (float(v.mean()), float(v.std(ddof=1)))
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="processed_metabric/ablation_results_cindex.pkl")
    ap.add_argument("--out_dir", default="logs/calibration")
    ap.add_argument("--models", nargs="*",
                    default=["Clinical-only (MLP)",
                             "Late Fusion (Concat)",
                             "Ours (HSIC K=512)"])
    ap.add_argument("--horizons", nargs="+", type=float, default=[60.0, 120.0],
                    help="Times in months (60=5y, 120=10y).")
    args = ap.parse_args()

    pkl_path = os.path.abspath(args.pkl)
    out_dir  = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(pkl_path, "rb") as f:
        results = pickle.load(f)

    # Sanity check: do the chosen models have the per-seed arrays?
    for name in args.models:
        if name not in results:
            raise KeyError(f"Model '{name}' not found in {pkl_path}. "
                           f"Available: {list(results.keys())[:5]} ...")
        ps0 = results[name]["_per_seed"][0]
        for k in ("train_risks", "train_times", "train_events",
                  "test_risks",  "test_times",  "test_events"):
            if k not in ps0:
                raise KeyError(
                    f"Model '{name}' is missing '{k}' in per-seed dict.\n"
                    "Did you patch main_C-index.py to save these arrays?")

    # ----- 1. Per-model, per-horizon plot -----
    print(f"\n{'='*70}\n  CALIBRATION RESULTS  (mean ± std over 5 seeds)\n{'='*70}\n")
    header = f"  {'Model':<30s}  {'t':>5s}   {'Brier':>14s}   {'ICI':>14s}   {'E50':>14s}   {'E90':>14s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    summary_table = {}
    for name in args.models:
        per_seed = results[name]["_per_seed"]
        fig, axes = plt.subplots(1, len(args.horizons),
                                 figsize=(5.5 * len(args.horizons), 5.5), dpi=120)
        if len(args.horizons) == 1:
            axes = [axes]
        for ax, t in zip(axes, args.horizons):
            S_all, t_all, e_all, metrics = _per_seed_calibration(per_seed, t)
            cal.plot_calibration(t_all, e_all, S_all, t,
                                 model_name=f"{name}", ax=ax)
            summ = _summarise(metrics)
            summary_table[(name, t)] = summ
            print(f"  {name:<30s}  {t:>5.0f}   "
                  f"{summ['brier'][0]:.4f}±{summ['brier'][1]:.4f}   "
                  f"{summ['ICI'][0]:.4f}±{summ['ICI'][1]:.4f}   "
                  f"{summ['E50'][0]:.4f}±{summ['E50'][1]:.4f}   "
                  f"{summ['E90'][0]:.4f}±{summ['E90'][1]:.4f}")

        fig.suptitle(name, fontsize=14, fontweight="bold")
        fig.tight_layout()
        safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        path = os.path.join(out_dir, f"calib_{safe}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> saved {path}")

    # ----- 2. Combined comparison: one figure per horizon, all models overlaid -----
    print()
    for t in args.horizons:
        fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=120)
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Perfect")
        colours = plt.cm.tab10.colors
        for i, name in enumerate(args.models):
            per_seed = results[name]["_per_seed"]
            S_all, t_all, e_all, _ = _per_seed_calibration(per_seed, t)
            gp, ob = cal.calibration_smooth(t_all, e_all, S_all, t)
            ax.plot(gp, ob, "-", color=colours[i % 10], lw=2, label=name)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel(f"Predicted event probability at t = {t:.0f}")
        ax.set_ylabel(f"Observed event probability at t = {t:.0f}")
        yr = "5y" if abs(t - 60) < 1e-6 else ("10y" if abs(t - 120) < 1e-6 else f"{t:.0f}m")
        ax.set_title(f"Calibration comparison @ {yr}")
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", fontsize=10)
        out = os.path.join(out_dir, f"calib_compare_{int(t)}m.png")
        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> saved {out}")

    # ----- 3. Numeric summary saved as a tiny CSV -----
    csv_path = os.path.join(out_dir, "calibration_summary.csv")
    with open(csv_path, "w") as f:
        f.write("model,horizon_months,brier_mean,brier_std,ICI_mean,ICI_std,"
                "E50_mean,E50_std,E90_mean,E90_std\n")
        for (m, t), s in summary_table.items():
            f.write(f"\"{m}\",{t:.0f},"
                    f"{s['brier'][0]:.4f},{s['brier'][1]:.4f},"
                    f"{s['ICI'][0]:.4f},{s['ICI'][1]:.4f},"
                    f"{s['E50'][0]:.4f},{s['E50'][1]:.4f},"
                    f"{s['E90'][0]:.4f},{s['E90'][1]:.4f}\n")
    print(f"  -> saved {csv_path}")


if __name__ == "__main__":
    main()
