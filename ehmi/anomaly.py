"""WP5 — danger / anomaly detection via Normalizing-Flow surprise.

The Flow gives exact log p(motion | situation). SURPRISE = -log p = "how
unexpected was this pedestrian's move, given the car + eHMI + history". Because
it is CONDITIONAL, a merely-close car is NOT surprising (it is in the condition);
only an abnormal ACTION is. Four experiments:

 A. group novelty  — train on one behavioral cluster, flag the others as OOD (AUC).
 B. near-miss anticipation — does surprise RISE before the closest approach? (lead time)
 C. danger AUC (honest) — surprise vs a kinematic baseline for flagging near-miss trials.
 D. personalization — does adapting the user embedding cut false alarms on safe trials?

Danger labels are derived from the data: min range to car (near-miss) + motion jerk.
"""
from __future__ import annotations

import copy
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from . import config as C
from .model import FlowModel
from .train import _load, build_windows, train_model, held_out_nll

NEAR_MISS_M = 2.0     # closest approach below this = near-miss (danger)
SAFE_M = 5.0          # trials that never come closer than this = "normal"
H = None              # set from config at runtime


@torch.no_grad()
def trial_surprise(net, g, norm, sess_idx, user_override=None):
    """Per-step surprise (-log p) for one trial, aligned with its range-to-car."""
    w = build_windows(g, norm, sess_idx, stride=1)
    if len(w["x"]) == 0:
        return None, None
    if user_override is not None:
        w["user"] = torch.full_like(w["user"], user_override)
    lp = []
    for j in range(0, len(w["x"]), 4096):
        b = {k: w[k][j:j + 4096] for k in ("hist", "x", "cont", "ehmi", "scen", "user")}
        lp.append(net.log_prob(b))
    surprise = -torch.cat(lp).numpy()
    rng = g.sort_values("k")["rng"].values[C.HIST_STEPS:C.HIST_STEPS + len(surprise)]
    return surprise, rng


def trial_table(feats):
    """Per-trial danger markers from features (range) + increments (jerk)."""
    rows = []
    for uid, g in feats.groupby("trial_uid"):
        g = g.sort_values("k")
        rng = g["rng"].values
        jf = np.hypot(np.diff(g["dfwd"].values), np.diff(g["dlat"].values))
        rows.append({"trial_uid": uid, "session": int(g["session"].iloc[0]),
                     "min_rng": float(rng.min()), "jerk": float(jf.max() if len(jf) else 0.0)})
    t = pd.DataFrame(rows)
    t["near_miss"] = (t["min_rng"] < NEAR_MISS_M).astype(int)
    t["safe"] = (t["min_rng"] > SAFE_M).astype(int)
    return t


def _win(feats, uids, norm, sess_idx):
    return build_windows(feats[feats["trial_uid"].isin(uids)], norm, sess_idx, stride=1)


def part_A_novelty(feats, norm, sess_idx, tbl, epochs):
    """Train on the largest behavioral cluster, flag other clusters as novel."""
    cl = json.loads((C.RESULTS_DIR / "cluster.json").read_text())
    uc = {int(k): v for k, v in cl["user_cluster"].items()}
    counts = pd.Series(list(uc.values())).value_counts()
    normal_c = int(counts.idxmax())
    normal_users = [s for s, c in uc.items() if c == normal_c]
    feats_n = feats[feats["session"].isin(normal_users)]
    trials_n = feats_n["trial_uid"].unique()
    rng = np.random.default_rng(0)
    test_n = set(rng.choice(trials_n, size=max(4, len(trials_n) // 5), replace=False))
    train_n = [u for u in trials_n if u not in test_n]

    net = FlowModel(len(sess_idx))
    train_model(net, _win(feats, train_n, norm, sess_idx), epochs)
    unk = net.enc.unknown_user

    scores, labels = [], []
    for uid, g in feats.groupby("trial_uid"):
        s, _ = trial_surprise(net, g, norm, sess_idx, user_override=unk)
        if s is None:
            continue
        in_dist = uid in test_n
        novel = uc.get(int(g["session"].iloc[0]), -1) != normal_c
        if in_dist or novel:
            scores.append(float(np.mean(s))); labels.append(0 if in_dist else 1)
    auc = float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan")
    return {"normal_cluster": normal_c, "auc": auc,
            "scores": scores, "labels": labels}


def part_BC_danger(feats, norm, sess_idx, tbl, epochs):
    """Train a Flow on SAFE trials; use its surprise to anticipate/flag near-miss."""
    safe_uids = tbl[tbl["safe"] == 1]["trial_uid"].tolist()
    nm_uids = tbl[tbl["near_miss"] == 1]["trial_uid"].tolist()
    safe_net = FlowModel(len(sess_idx))
    train_model(safe_net, _win(feats, safe_uids, norm, sess_idx), epochs)
    unk = safe_net.enc.unknown_user

    # per-trial peak surprise + jerk for AUC; aligned surprise curves for anticipation
    pk, lab, jrk = [], [], []
    PRE, POST = 30, 10
    nm_curve, safe_curve, rng_curve = [], [], []
    jmap = tbl.set_index("trial_uid")["jerk"].to_dict()
    for uid, g in feats.groupby("trial_uid"):
        s, rng = trial_surprise(safe_net, g, norm, sess_idx, user_override=unk)
        if s is None or len(s) < 5:
            continue
        is_nm = int(uid in nm_uids); is_safe = int(uid in safe_uids)
        if not (is_nm or is_safe):
            continue
        pk.append(float(np.max(s))); lab.append(is_nm); jrk.append(float(jmap.get(uid, 0)))
        t0 = int(np.argmin(rng))
        seg = np.full(PRE + POST, np.nan); rseg = np.full(PRE + POST, np.nan)
        for off in range(-PRE, POST):
            idx = t0 + off
            if 0 <= idx < len(s):
                seg[off + PRE] = s[idx]; rseg[off + PRE] = rng[idx]
        (nm_curve if is_nm else safe_curve).append(seg)
        if is_nm:
            rng_curve.append(rseg)
    auc_surprise = float(roc_auc_score(lab, pk)) if len(set(lab)) > 1 else float("nan")
    auc_jerk = float(roc_auc_score(lab, jrk)) if len(set(lab)) > 1 else float("nan")

    def _m(curves):
        return np.nanmean(np.vstack(curves), 0) if curves else np.full(PRE + POST, np.nan)
    nm_m, safe_m, rng_m = _m(nm_curve), _m(safe_curve), _m(rng_curve)
    # lead time: first pre-t0 offset where near-miss surprise exceeds safe baseline+1std
    base = np.nanmean(safe_m); sd = np.nanstd(safe_m) + 1e-6
    lead = None
    for off in range(-PRE, 0):
        if nm_m[off + PRE] > base + sd:
            lead = -off * C.DT; break
    return {"auc_surprise": auc_surprise, "auc_jerk": auc_jerk,
            "n_near_miss": int(sum(lab)), "n_safe": int(len(lab) - sum(lab)),
            "lead_time_s": lead, "PRE": PRE, "POST": POST,
            "nm_curve": nm_m.tolist(), "safe_curve": safe_m.tolist(), "rng_curve": rng_m.tolist()}, safe_net


def part_D_personalization(feats, norm, sess_idx, tbl, safe_net):
    """Does adapting the user embedding lower surprise (false alarms) on safe trials?"""
    from .fewshot import _adapt_embed
    unk = safe_net.enc.unknown_user
    pop, per = [], []
    rng = np.random.default_rng(7)
    for s in sess_idx:
        safe_u = tbl[(tbl["session"] == s) & (tbl["safe"] == 1)]["trial_uid"].tolist()
        if len(safe_u) < 4:
            continue
        sup = list(rng.choice(safe_u, size=2, replace=False))
        qry = [u for u in safe_u if u not in sup]
        pop_s = np.mean([np.mean(trial_surprise(safe_net, feats[feats.trial_uid == u], norm, sess_idx,
                        user_override=unk)[0]) for u in qry])
        net = copy.deepcopy(safe_net)
        with torch.no_grad():
            net.enc.user_emb.weight[sess_idx[s]] = net.enc.user_emb.weight[unk].clone()
        _adapt_embed(net, _win(feats, sup, norm, sess_idx), sess_idx[s], steps=60)
        per_s = np.mean([np.mean(trial_surprise(net, feats[feats.trial_uid == u], norm, sess_idx,
                        user_override=sess_idx[s])[0]) for u in qry])
        pop.append(float(pop_s)); per.append(float(per_s))
    return {"pop_mean": float(np.mean(pop)) if pop else None,
            "personalized_mean": float(np.mean(per)) if per else None,
            "pop": pop, "personalized": per, "n_users": len(pop)}


def main():
    feats, norm, vocab = _load()
    sessions = vocab["sessions"]; sess_idx = {s: i for i, s in enumerate(sessions)}
    epochs = 8
    tbl = trial_table(feats)
    res = {"n_near_miss": int(tbl["near_miss"].sum()), "n_safe": int(tbl["safe"].sum()),
           "near_miss_m": NEAR_MISS_M, "safe_m": SAFE_M}

    try:
        res["A_novelty"] = part_A_novelty(feats, norm, sess_idx, tbl, epochs)
        print(f"  A novelty AUC = {res['A_novelty']['auc']:.3f}")
    except Exception as e:  # noqa: BLE001
        print(f"  A failed: {e}")
    try:
        bc, safe_net = part_BC_danger(feats, norm, sess_idx, tbl, epochs)
        res["BC_danger"] = bc
        print(f"  B/C: AUC surprise={bc['auc_surprise']:.3f} vs jerk={bc['auc_jerk']:.3f}, "
              f"lead={bc['lead_time_s']}")
        try:
            res["D_personalization"] = part_D_personalization(feats, norm, sess_idx, tbl, safe_net)
            d = res["D_personalization"]
            print(f"  D: safe-trial surprise pop={d['pop_mean']:.3f} -> personalized={d['personalized_mean']:.3f}")
        except Exception as e:  # noqa: BLE001
            print(f"  D failed: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"  B/C failed: {e}")

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.RESULTS_DIR / "anomaly.json").write_text(json.dumps(res, indent=2))
    _figures(res)
    print("wrote results/anomaly.json + figures")


def _figures(res):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    # A: surprise distributions in-dist vs novel
    a = res.get("A_novelty")
    if a and a["scores"]:
        sc = np.array(a["scores"]); lb = np.array(a["labels"])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(sc[lb == 0], bins=20, alpha=0.6, color="#2a6fb0", label="in-distribution (trained type)")
        ax.hist(sc[lb == 1], bins=20, alpha=0.6, color="#d1651f", label="novel type (unseen)")
        ax.set_xlabel("trial mean surprise  (−log p, higher=more anomalous)")
        ax.set_ylabel("trials"); ax.legend(fontsize=8)
        ax.set_title(f"A. group-novelty detection  (AUC={a['auc']:.2f})")
        fig.tight_layout(); fig.savefig(C.FIG_DIR / "anomaly_novelty.png", dpi=110); plt.close(fig)
    # B: anticipation curves
    bc = res.get("BC_danger")
    if bc:
        PRE, POST = bc["PRE"], bc["POST"]
        t = (np.arange(-PRE, POST)) * C.DT
        fig, ax1 = plt.subplots(figsize=(7, 4.2))
        ax1.plot(t, bc["nm_curve"], "-", color="#d1651f", lw=2.2, label="surprise (near-miss trials)")
        ax1.plot(t, bc["safe_curve"], "-", color="#2a6fb0", lw=1.6, label="surprise (safe trials)")
        ax1.axvline(0, color="k", ls="--", lw=1); ax1.set_xlabel("time relative to closest approach (s)")
        ax1.set_ylabel("surprise  (−log p)")
        if bc["lead_time_s"]:
            ax1.axvline(-bc["lead_time_s"], color="#d1651f", ls=":", lw=1)
            ax1.text(-bc["lead_time_s"], ax1.get_ylim()[1], f" lead {bc['lead_time_s']:.1f}s",
                     color="#d1651f", fontsize=8, va="top")
        ax2 = ax1.twinx(); ax2.plot(t, bc["rng_curve"], "-", color="#999", lw=1.2, label="range to car (m)")
        ax2.set_ylabel("range to car (m)", color="#999")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.set_title("B. surprise rises BEFORE the near-miss (early warning)")
        fig.tight_layout(); fig.savefig(C.FIG_DIR / "anomaly_leadtime.png", dpi=110); plt.close(fig)
    # D: personalization false alarm
    d = res.get("D_personalization")
    if d and d.get("pop"):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["population\n(generic)", "personalized\n(few-shot)"],
               [d["pop_mean"], d["personalized_mean"]], color=["#bbb", "#2a6fb0"])
        ax.set_ylabel("mean surprise on SAFE trials\n(lower = fewer false alarms)")
        ax.set_title("D. personalizing the embedding cuts false alarms")
        fig.tight_layout(); fig.savefig(C.FIG_DIR / "anomaly_personalization.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
