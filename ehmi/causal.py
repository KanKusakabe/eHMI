"""WP3 — causal / counterfactual analysis of the eHMI (treatment = ON vs OFF).

Because the eHMI condition was ASSIGNED by design (within-subject, counterbalanced),
`do(eHMI)` is well-founded here. Two levels of Pearl's ladder:

 (i)  CATE per behavioral type (rung 2): how the effect of the eHMI differs across
      the clusters found in WP2 -- both the raw observed ON-vs-OFF gap and a
      model-based flip.
 (ii) Individual counterfactual (rung 3): for a factual eHMI-ON trial, abduct the
      latent z (NF invertibility), then decode with eHMI turned OFF holding z +
      history + car kinematics fixed = "this same pedestrian, had there been no
      warning." Not point-validatable (assumption-dependent) -- flagged.

Cleanest treatment = ON vs OFF only (Left/Right direction is entangled with the
car's evasive maneuver).
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import config as C
from .model import FlowModel
from .train import _load, build_windows

TARGET_M = np.array([0.0, 0.0])  # filled from norm at runtime


def _load_full():
    ck = torch.load(C.RESULTS_DIR / "flow_full.pt", map_location="cpu", weights_only=False)
    net = FlowModel(ck["n_sessions"]); net.load_state_dict(ck["state"]); net.eval()
    return net, ck["sess_idx"]


@torch.no_grad()
def counterfactual_increments(net, batch, cf_ehmi=0):
    """Abduct z under the factual eHMI, decode with eHMI=cf_ehmi (held z)."""
    ctx = net._ctx(batch)
    z = net.flow(ctx).transform.inv(batch["x"])
    b2 = dict(batch); b2["ehmi"] = torch.full_like(batch["ehmi"], cf_ehmi)
    ctx2 = net._ctx(b2)
    x_cf = net.flow(ctx2).transform(z)
    return x_cf


def _trial_batch(feats, uid, norm, sess_idx):
    w = build_windows(feats[feats["trial_uid"] == uid], norm, sess_idx, stride=1)
    return w if len(w["x"]) else None


def main():
    feats, norm, vocab = _load()
    sessions = vocab["sessions"]
    sess_idx = {s: i for i, s in enumerate(sessions)}
    net, _ = _load_full()
    tm = np.array([norm[c][0] for c in C.TARGET_COLS])
    tf = np.array([norm[c][1] for c in C.TARGET_COLS])
    clus = json.loads((C.RESULTS_DIR / "cluster.json").read_text())
    user_cluster = {int(k): v for k, v in clus["user_cluster"].items()}
    best_k = clus["best_k"]

    meta = pd.read_parquet(C.PROC_DIR / "trial_meta.parquet").drop_duplicates("trial_uid").set_index("trial_uid")

    # per-trial outcomes (crossing frame): peak lateral evasion + net forward
    def outcomes(inc):  # inc: (T,2) increments
        path = np.cumsum(inc, axis=0)
        return abs(path[:, 1]).max(), path[-1, 0]

    on_trials = [u for u in feats["trial_uid"].unique()
                 if u in meta.index and meta.loc[u, "ehmi"] in ("Left", "Right")]
    off_trials = [u for u in feats["trial_uid"].unique()
                  if u in meta.index and meta.loc[u, "ehmi"] == "None"]

    # (i-a) raw observed ON vs OFF, per cluster
    def cluster_of(u):
        return user_cluster.get(int(feats[feats["trial_uid"] == u]["session"].iloc[0]), -1)

    def obs_inc(u):
        g = feats[feats["trial_uid"] == u].sort_values("k")
        return np.column_stack([g[c].values for c in C.TARGET_COLS])

    raw = {c: {"on_lat": [], "off_lat": [], "on_fwd": [], "off_fwd": []} for c in range(best_k)}
    for u in on_trials:
        c = cluster_of(u); pl, nf = outcomes(obs_inc(u)); raw[c]["on_lat"].append(pl); raw[c]["on_fwd"].append(nf)
    for u in off_trials:
        c = cluster_of(u); pl, nf = outcomes(obs_inc(u)); raw[c]["off_lat"].append(pl); raw[c]["off_fwd"].append(nf)

    # (i-b)+(ii) model counterfactual: flip ON->OFF, hold z + covariates
    cf_effect = {c: {"d_lat": [], "d_fwd": []} for c in range(best_k)}
    example = None
    for u in on_trials:
        b = _trial_batch(feats, u, norm, sess_idx)
        if b is None:
            continue
        x_cf = counterfactual_increments(net, b, cf_ehmi=0).numpy() * tf + tm
        x_fac = b["x"].numpy() * tf + tm
        pl_f, nf_f = outcomes(x_fac); pl_c, nf_c = outcomes(x_cf)
        c = cluster_of(u)
        cf_effect[c]["d_lat"].append(pl_f - pl_c)   # ON minus (counterfactual OFF)
        cf_effect[c]["d_fwd"].append(nf_f - nf_c)
        if example is None and len(x_fac) > 20:
            example = (u, np.cumsum(x_fac, 0), np.cumsum(x_cf, 0))

    # ---- figures ----
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    xc = np.arange(best_k)
    on_l = [np.nanmean(raw[c]["on_lat"]) if raw[c]["on_lat"] else np.nan for c in range(best_k)]
    off_l = [np.nanmean(raw[c]["off_lat"]) if raw[c]["off_lat"] else np.nan for c in range(best_k)]
    ax[0].bar(xc - 0.2, on_l, 0.4, label="eHMI ON", color="#2a6fb0")
    ax[0].bar(xc + 0.2, off_l, 0.4, label="eHMI OFF", color="#bbb")
    ax[0].set_title("Observed peak lateral evasion by type"); ax[0].set_xlabel("behavioral type")
    ax[0].set_xticks(xc); ax[0].legend(fontsize=8); ax[0].set_ylabel("peak |lateral| (m)")
    cfl = [np.nanmean(cf_effect[c]["d_lat"]) if cf_effect[c]["d_lat"] else np.nan for c in range(best_k)]
    ax[1].bar(xc, cfl, color="#d1651f")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_title("Counterfactual eHMI effect (ON − do(OFF))\nmodel, holding z + car fixed")
    ax[1].set_xlabel("behavioral type"); ax[1].set_xticks(xc); ax[1].set_ylabel("Δ peak |lateral| (m)")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "cate_by_cluster.png", dpi=110); plt.close(fig)

    if example is not None:
        uid, fac, cf = example
        fig, ax = plt.subplots(figsize=(5.4, 5))
        ax.plot(fac[:, 1], fac[:, 0], "-", color="#e05a7a", lw=2.4, label="factual (eHMI ON)")
        ax.plot(cf[:, 1], cf[:, 0], "--", color="#2a6fb0", lw=2.4, label="counterfactual: do(eHMI OFF)")
        ax.plot(0, 0, "o", color="#333", ms=7)
        ax.set_xlabel("lateral (m)  — axis stretched to show the eHMI effect")
        ax.set_ylabel("forward / crossing (m)")
        ax.set_title(f"Individual counterfactual (abduction)\n{uid}: eHMI ON vs do(OFF)")
        ax.legend(fontsize=9)  # not equal-aspect: lateral effect is ~cm vs metres forward
        fig.tight_layout(); fig.savefig(C.FIG_DIR / "counterfactual_example.png", dpi=110); plt.close(fig)

    def m(a):
        return float(np.nanmean(a)) if len(a) else float("nan")
    res = {"best_k": best_k, "n_on": len(on_trials), "n_off": len(off_trials),
           "raw_by_cluster": {c: {k: m(v) for k, v in raw[c].items()} for c in range(best_k)},
           "counterfactual_effect_by_cluster": {
               c: {"d_lat": m(cf_effect[c]["d_lat"]), "d_fwd": m(cf_effect[c]["d_fwd"]),
                   "n": len(cf_effect[c]["d_lat"])} for c in range(best_k)},
           "example_trial": example[0] if example else None,
           "note": "eHMI ON/OFF only (L/R entangled w/ car maneuver); counterfactual is "
                   "model-dependent (SCM assumption), not point-validatable."}
    (C.RESULTS_DIR / "causal.json").write_text(json.dumps(res, indent=2))
    print(f"causal: {len(on_trials)} ON / {len(off_trials)} OFF trials")
    print("  counterfactual eHMI effect (Δpeak|lat| m) by type:",
          {c: round(m(cf_effect[c]["d_lat"]), 4) for c in range(best_k)})


if __name__ == "__main__":
    main()
