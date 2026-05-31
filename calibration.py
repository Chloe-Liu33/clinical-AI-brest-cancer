"""Survival calibration utilities for Cox PH models.

Inputs everywhere:
  * risks  : ndarray (N,)  -- model output (log-hazard offset, no baseline).
  * times  : ndarray (N,)  -- observed time (event time or censor time).
  * events : ndarray (N,)  -- 1 if event observed, 0 if right-censored.

Workflow:
  1.  H0_fn = breslow_baseline(train_risks, train_times, train_events)
  2.  S_pred = predict_survival(test_risks, t_eval, H0_fn)        # per-patient S_i(t)
  3.  fig    = plot_calibration(test_times, test_events, S_pred, t_eval)
  4.  bs     = brier_score(test_times, test_events, S_pred, t_eval,
                           train_times=train_times, train_events=train_events)  # IPCW

No external survival libraries required.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as _beta_dist


# ---------------------------------------------------------------------------
# 1.  Breslow baseline cumulative hazard
# ---------------------------------------------------------------------------
def breslow_baseline(risks: np.ndarray,
                     times: np.ndarray,
                     events: np.ndarray):
    """Return a callable H0(t) -- baseline cumulative hazard from training data.

    Implementation: at each unique event time t_k, dH0 = d_k / sum_{R(t_k)} exp(risk).
    H0 is a right-continuous step function in t.
    """
    risks  = np.asarray(risks,  dtype=np.float64)
    times  = np.asarray(times,  dtype=np.float64)
    events = np.asarray(events, dtype=np.int64)

    order = np.argsort(times)
    t_sorted = times[order]
    e_sorted = events[order]
    r_sorted = risks[order]
    exp_r    = np.exp(r_sorted)

    # Risk set at t_k = all i with time_i >= t_k.
    # If we walk in ascending time, sum_{i: t_i >= t_k} exp(risk_i)
    # = total_sum_at_start - cumulative_sum_strictly_below.
    total = exp_r.sum()
    cum_below = np.concatenate(([0.0], np.cumsum(exp_r)))[:-1]
    risk_set_sum = total - cum_below          # vector, same length as sorted

    # Take only event times; aggregate ties.
    event_mask = e_sorted == 1
    t_events    = t_sorted[event_mask]
    rss_events  = risk_set_sum[event_mask]

    # Aggregate ties: at each unique event time, sum of d_k events; risk-set sum
    # is the same for tied times so we use the first one.
    uniq_t, inv, counts = np.unique(t_events, return_inverse=True, return_counts=True)
    rss_uniq = np.zeros_like(uniq_t)
    rss_uniq[inv] = rss_events                 # last write wins, but they're equal
    dH = counts / rss_uniq                     # ties handled à la Breslow
    H_grid = np.cumsum(dH)

    def H0_fn(t):
        """Right-continuous step interpolation of H0(t)."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        idx = np.searchsorted(uniq_t, t, side="right") - 1
        out = np.where(idx < 0, 0.0, H_grid[np.clip(idx, 0, len(H_grid) - 1)])
        return out if out.size > 1 else float(out[0])

    H0_fn.t_grid = uniq_t
    H0_fn.H_grid = H_grid
    return H0_fn


def predict_survival(risks: np.ndarray, t_eval: float, H0_fn) -> np.ndarray:
    """Per-patient predicted survival probability at horizon t_eval."""
    H0_t = H0_fn(t_eval)
    return np.exp(-H0_t * np.exp(np.asarray(risks, dtype=np.float64)))


# ---------------------------------------------------------------------------
# 2.  Kaplan-Meier (used for both calibration bins and IPCW censoring)
# ---------------------------------------------------------------------------
def kaplan_meier(times: np.ndarray, events: np.ndarray):
    """Standard KM estimator. Returns (t_grid, S_grid) as right-continuous step."""
    times  = np.asarray(times,  dtype=np.float64)
    events = np.asarray(events, dtype=np.int64)
    order = np.argsort(times)
    t_sorted = times[order]
    e_sorted = events[order]

    n = len(t_sorted)
    # at each unique time, count d=events, c=censored, n_at_risk
    uniq_t, idx_first = np.unique(t_sorted, return_index=True)
    # number at risk at uniq_t[k] = n - idx_first[k]
    n_at_risk = n - idx_first
    # events at each unique time
    d_uniq = np.zeros_like(uniq_t)
    for k, t in enumerate(uniq_t):
        d_uniq[k] = (e_sorted[t_sorted == t] == 1).sum()

    # KM
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = 1.0 - d_uniq / n_at_risk
    S_grid = np.cumprod(factor)
    return uniq_t, S_grid


def km_at(t_eval: float, t_grid: np.ndarray, S_grid: np.ndarray) -> float:
    """Right-continuous step lookup of KM at t_eval."""
    if t_eval < t_grid[0]:
        return 1.0
    idx = np.searchsorted(t_grid, t_eval, side="right") - 1
    return float(S_grid[idx])


# ---------------------------------------------------------------------------
# 3.  Bin-and-KM calibration (the 'standard' deciles plot)
# ---------------------------------------------------------------------------
def calibration_bins(times: np.ndarray,
                     events: np.ndarray,
                     S_pred: np.ndarray,
                     t_eval: float,
                     n_bins: int = 10):
    """Quantile-bin patients by S_pred, KM-estimate observed S(t_eval) per bin.

    Returns dict with arrays:
      mean_pred  -- mean predicted S in each bin
      obs_S      -- KM-observed S(t_eval) in each bin
      ci_lo, ci_hi -- 95% CI for obs_S (Greenwood-style normal approx)
      n          -- patients per bin
    """
    times  = np.asarray(times)
    events = np.asarray(events)
    S_pred = np.asarray(S_pred)

    # quantile edges; use np.unique to absorb ties
    edges = np.quantile(S_pred, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        # degenerate: collapse to whatever bins survive
        edges = np.unique(np.concatenate([[S_pred.min()], edges, [S_pred.max()]]))

    bin_id = np.clip(np.searchsorted(edges[1:-1], S_pred, side="right"), 0, len(edges) - 2)
    K = len(edges) - 1

    mean_pred = np.full(K, np.nan)
    obs_S     = np.full(K, np.nan)
    ci_lo     = np.full(K, np.nan)
    ci_hi     = np.full(K, np.nan)
    n_arr     = np.zeros(K, dtype=int)

    for k in range(K):
        mask = bin_id == k
        n_arr[k] = mask.sum()
        if n_arr[k] < 2:
            continue
        mean_pred[k] = S_pred[mask].mean()
        t_grid, S_grid = kaplan_meier(times[mask], events[mask])
        S_t = km_at(t_eval, t_grid, S_grid)
        obs_S[k] = S_t
        # Greenwood log-log CI for KM at t -- simple, conservative
        # var(logS) ~= sum d/(n(n-d)) cumulative; we approximate by symmetric beta CI
        ci_lo[k], ci_hi[k] = _km_loglog_ci(times[mask], events[mask], t_eval)
    return dict(mean_pred=mean_pred, obs_S=obs_S, ci_lo=ci_lo, ci_hi=ci_hi, n=n_arr)


def _km_loglog_ci(times, events, t_eval, alpha=0.05):
    """KM log-log 95% CI at t_eval."""
    times  = np.asarray(times,  dtype=np.float64)
    events = np.asarray(events, dtype=np.int64)
    order = np.argsort(times)
    t_sorted = times[order]; e_sorted = events[order]
    uniq_t = np.unique(t_sorted)
    S = 1.0; var_log = 0.0; n = len(t_sorted); ci = (np.nan, np.nan)
    for t in uniq_t:
        if t > t_eval:
            break
        d = int(((t_sorted == t) & (e_sorted == 1)).sum())
        n_risk = int((t_sorted >= t).sum())
        if n_risk - d > 0 and n_risk > 0 and d > 0:
            S *= (1 - d / n_risk)
            var_log += d / (n_risk * (n_risk - d))
    if S <= 0 or S >= 1 or var_log == 0:
        return (max(S - 0.05, 0), min(S + 0.05, 1))
    # log-log transform
    z = 1.959963984540054
    se_loglog = np.sqrt(var_log) / np.log(S)
    log_neg_log_S = np.log(-np.log(S))
    lo = np.exp(-np.exp(log_neg_log_S + z * se_loglog))
    hi = np.exp(-np.exp(log_neg_log_S - z * se_loglog))
    return (float(lo), float(hi))


# ---------------------------------------------------------------------------
# 4.  Smooth Austin-Steyerberg calibration curve  (logistic-regression-on-S)
# ---------------------------------------------------------------------------
def calibration_smooth(times: np.ndarray,
                       events: np.ndarray,
                       S_pred: np.ndarray,
                       t_eval: float,
                       n_grid: int = 100):
    """Lightweight smoothed calibration via local logistic regression on cloglog(1-S).

    For each grid point of predicted S, we estimate observed S(t) using a
    KM-like restriction to neighbouring patients. This is the simplified
    Austin-Steyerberg style (Stata `pmcalplot`).
    """
    p_pred = 1.0 - np.asarray(S_pred)               # event probability
    p_pred = np.clip(p_pred, 1e-6, 1 - 1e-6)
    cll = np.log(-np.log(1 - p_pred))               # cloglog of event prob
    # Use sliding-window KM smoothing; window = 20% of data.
    order = np.argsort(p_pred)
    t_o = np.asarray(times)[order]
    e_o = np.asarray(events)[order]
    p_o = p_pred[order]
    n = len(p_o)
    win = max(20, int(0.2 * n))
    grid_p = np.linspace(p_o.min(), p_o.max(), n_grid)
    obs = np.empty_like(grid_p)
    for i, p in enumerate(grid_p):
        # find nearest 'win' points
        j = np.searchsorted(p_o, p)
        lo = max(0, j - win // 2)
        hi = min(n, lo + win)
        lo = max(0, hi - win)
        t_grid, S_grid = kaplan_meier(t_o[lo:hi], e_o[lo:hi])
        obs[i] = 1 - km_at(t_eval, t_grid, S_grid)
    return grid_p, obs   # both are *event probabilities*


# ---------------------------------------------------------------------------
# 5.  Brier score (IPCW-corrected, Graf 1999)
# ---------------------------------------------------------------------------
def brier_score(times: np.ndarray,
                events: np.ndarray,
                S_pred: np.ndarray,
                t_eval: float,
                train_times: np.ndarray | None = None,
                train_events: np.ndarray | None = None) -> float:
    """IPCW Brier score at t_eval.  Lower = better.  0.25 = trivial random.

    If train_times/events are supplied, censoring distribution G(t) is
    estimated from training set (preferred). Otherwise from test set itself.
    """
    times  = np.asarray(times,  dtype=np.float64)
    events = np.asarray(events, dtype=np.int64)
    S_pred = np.asarray(S_pred, dtype=np.float64)

    # KM of censoring distribution: events are flipped (1 - event).
    if train_times is None:
        ct, ce = times, 1 - events
    else:
        ct, ce = np.asarray(train_times), 1 - np.asarray(train_events)
    g_grid, G_grid = kaplan_meier(ct, ce)

    G_t  = max(km_at(t_eval, g_grid, G_grid), 1e-6)
    # weight per subject
    bs = 0.0
    for ti, ei, si in zip(times, events, S_pred):
        if ti <= t_eval and ei == 1:
            G_ti = max(km_at(ti, g_grid, G_grid), 1e-6)
            bs += (0.0 - si) ** 2 / G_ti              # subject died before t -> true S=0
        elif ti > t_eval:
            bs += (1.0 - si) ** 2 / G_t               # subject survived past t -> true S=1
        # else: censored before t -> contributes 0 (uninformative)
    return float(bs / len(times))


# ---------------------------------------------------------------------------
# 6.  Integrated Calibration Index (ICI), E50, E90
# ---------------------------------------------------------------------------
def calibration_ici(times: np.ndarray,
                    events: np.ndarray,
                    S_pred: np.ndarray,
                    t_eval: float):
    """Mean / median / 90th-percentile of |observed - predicted| event-prob,
    estimated from the smooth calibration curve.
    """
    grid_p, obs = calibration_smooth(times, events, S_pred, t_eval)
    # diff at each test patient by interpolating the smooth curve
    p_pred = 1.0 - np.asarray(S_pred)
    obs_at_pred = np.interp(p_pred, grid_p, obs)
    diffs = np.abs(obs_at_pred - p_pred)
    return dict(ICI=float(diffs.mean()),
                E50=float(np.quantile(diffs, 0.5)),
                E90=float(np.quantile(diffs, 0.9)))


# ---------------------------------------------------------------------------
# 7.  Plotting
# ---------------------------------------------------------------------------
def plot_calibration(times: np.ndarray,
                     events: np.ndarray,
                     S_pred: np.ndarray,
                     t_eval: float,
                     model_name: str = "Model",
                     n_bins: int = 10,
                     show_smooth: bool = True,
                     ax=None):
    """Calibration plot in event-probability space (1 - S).

    Diagonal = perfect calibration. Below diagonal = over-confident on death;
    above diagonal = under-confident.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=120)
    else:
        fig = ax.figure

    bins = calibration_bins(times, events, S_pred, t_eval, n_bins=n_bins)
    pred_event = 1.0 - bins["mean_pred"]
    obs_event  = 1.0 - bins["obs_S"]
    err_lo     = np.maximum(0.0, obs_event - (1.0 - bins["ci_hi"]))   # flip CI for event prob
    err_hi     = np.maximum(0.0, (1.0 - bins["ci_lo"]) - obs_event)

    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Perfect calibration")

    if show_smooth:
        gp, ob = calibration_smooth(times, events, S_pred, t_eval)
        ax.plot(gp, ob, "-", color="C0", alpha=0.6, lw=2, label="Smoothed (sliding KM)")

    ax.errorbar(pred_event, obs_event,
                yerr=[np.nan_to_num(err_lo, nan=0.0), np.nan_to_num(err_hi, nan=0.0)],
                fmt="o", color="C3", ms=6, capsize=3,
                label=f"Decile bins (n_bins={n_bins})")

    bs  = brier_score(times, events, S_pred, t_eval)
    ici = calibration_ici(times, events, S_pred, t_eval)
    txt = (f"Brier = {bs:.3f}\n"
           f"ICI  = {ici['ICI']:.3f}\n"
           f"E50  = {ici['E50']:.3f}\n"
           f"E90  = {ici['E90']:.3f}")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.8"))

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(f"Predicted event probability at t = {t_eval:g}")
    ax.set_ylabel(f"Observed event probability at t = {t_eval:g}")
    ax.set_title(f"{model_name} — calibration at t = {t_eval:g}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    return fig, ax


# ---------------------------------------------------------------------------
# 8.  End-to-end one-call helper
# ---------------------------------------------------------------------------
def calibrate_and_plot(train_risks, train_times, train_events,
                       test_risks,  test_times,  test_events,
                       eval_times=(60.0, 120.0),     # METABRIC times are months
                       model_name="Ours (HSIC K=512)",
                       savepath=None):
    """One-shot helper.  eval_times in the same units as `times` (months for METABRIC)."""
    H0  = breslow_baseline(train_risks, train_times, train_events)
    fig, axes = plt.subplots(1, len(eval_times),
                             figsize=(5.5 * len(eval_times), 5.5), dpi=120)
    if len(eval_times) == 1:
        axes = [axes]
    summary = {}
    for ax, t in zip(axes, eval_times):
        S = predict_survival(test_risks, t, H0)
        plot_calibration(test_times, test_events, S, t,
                         model_name=model_name, ax=ax)
        summary[t] = dict(
            brier=brier_score(test_times, test_events, S, t,
                              train_times=train_times, train_events=train_events),
            **calibration_ici(test_times, test_events, S, t))
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig, summary
