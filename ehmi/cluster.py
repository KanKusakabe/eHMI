"""WP2 — behavioral fingerprints: cluster the 20 user embeddings, profile the
types, and test typing / individual identification by likelihood.

  * cluster the 20 learned embeddings (K=2..4, pick by silhouette; GMM/BIC cross-check)
  * profile each cluster with observable traits (crossing speed, eHMI compliance,
    safety rating, lateral evasion)
  * identification: does a user's OWN embedding explain their trials best? (fingerprint)
  * typing: assign each user to a behavioral cluster
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config as C
from .model import FlowModel
from .train import _load, build_windows, held_out_nll


def _user_traits(feats, meta):
    """Observable per-user behavioral traits, to interpret the clusters."""
    rows = {}
    fol = meta.set_index("trial_uid")["followed"].astype(str).str.lower()
    for s, g in feats.groupby("session"):
        per_trial = g.groupby("trial_uid")
        net_fwd = per_trial["dfwd"].sum()
        dur = per_trial.size() * C.DT
        speed = float((net_fwd / dur).mean())
        onmask = g["ehmi_idx"] > 0
        lat_on = float(g.loc[onmask, "dlat"].abs().mean()) if onmask.any() else float("nan")
        uids = g["trial_uid"].unique()
        comp = float(np.mean([1.0 if "ja" in fol.get(u, "") else 0.0 for u in uids]))
        saf = meta[meta["session"] == s]["safety"].astype(float)
        rows[s] = {"cross_speed": speed, "lat_evasion": lat_on,
                   "eHMI_compliance": comp, "safety": float(saf.mean())}
    return rows


def _win(feats, uids, norm, sess_idx):
    return build_windows(feats[feats["trial_uid"].isin(uids)], norm, sess_idx, stride=1)


def main():
    feats, norm, vocab = _load()
    sessions = vocab["sessions"]
    sess_idx = {s: i for i, s in enumerate(sessions)}
    meta = pd.read_parquet(C.PROC_DIR / "trial_meta.parquet").drop_duplicates("trial_uid")
    emb = np.load(C.RESULTS_DIR / "embeddings.npy")             # 20 x 8
    Z = StandardScaler().fit_transform(emb)

    # choose K by silhouette over 2..4
    sil = {}
    for k in (2, 3, 4):
        lab = KMeans(k, n_init=10, random_state=0).fit_predict(Z)
        sil[k] = float(silhouette_score(Z, lab))
    best_k = max(sil, key=sil.get)
    labels = KMeans(best_k, n_init=10, random_state=0).fit_predict(Z)
    user_cluster = {sessions[i]: int(labels[i]) for i in range(len(sessions))}

    # trait profiles per cluster
    traits = _user_traits(feats, meta)
    tnames = ["cross_speed", "lat_evasion", "eHMI_compliance", "safety"]
    profiles = {}
    for c in range(best_k):
        us = [s for s in sessions if user_cluster[s] == c]
        profiles[c] = {"n_users": len(us),
                       **{t: float(np.nanmean([traits[s][t] for s in us])) for t in tnames}}

    # ---- identification: which user's embedding best explains a user's trials ----
    ck = torch.load(C.RESULTS_DIR / "flow_full.pt", map_location="cpu", weights_only=False)
    full = FlowModel(ck["n_sessions"]); full.load_state_dict(ck["state"]); full.eval()
    conf = np.zeros((len(sessions), len(sessions)))
    for i, u in enumerate(sessions):
        w = _win(feats, list(feats[feats["session"] == u]["trial_uid"].unique()), norm, sess_idx)
        for j in range(len(sessions)):
            w2 = dict(w); w2["user"] = torch.full_like(w["user"], j)
            conf[i, j] = -held_out_nll(full, w2)              # higher logprob = better
    pred = conf.argmax(1)
    id_acc = float(np.mean([pred[i] == i for i in range(len(sessions))]))
    type_acc = float(np.mean([user_cluster[sessions[pred[i]]] == user_cluster[sessions[i]]
                              for i in range(len(sessions))]))

    # ---- figures ----
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    P = PCA(2).fit_transform(Z)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    sc = ax.scatter(P[:, 0], P[:, 1], c=labels, cmap="tab10", s=90)
    for i, s in enumerate(sessions):
        ax.annotate(str(s), (P[i, 0], P[i, 1]), fontsize=7, ha="center", va="center")
    ax.set_title(f"User embedding map (K={best_k} behavioral types)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "embedding_map.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(best_k)
    for i, t in enumerate(tnames):
        vals = [profiles[c][t] for c in range(best_k)]
        v = np.array(vals); v = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-9)
        ax.plot(x, v, "o-", label=t)
    ax.set_xticks(x); ax.set_xticklabels([f"type {c}\n(n={profiles[c]['n_users']})" for c in range(best_k)])
    ax.set_ylabel("normalized trait"); ax.set_title("Behavioral-type profiles"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "cluster_profiles.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.4))
    im = ax.imshow(conf, cmap="viridis")
    ax.set_title(f"Identification by likelihood\ntop-1 acc={id_acc:.0%}, typing acc={type_acc:.0%}")
    ax.set_xlabel("candidate user (embedding)"); ax.set_ylabel("true user")
    fig.colorbar(im, ax=ax, label="mean log-prob"); fig.tight_layout()
    fig.savefig(C.FIG_DIR / "identification.png", dpi=110); plt.close(fig)

    res = {"best_k": best_k, "silhouette": sil, "user_cluster": user_cluster,
           "profiles": profiles, "identification_top1": id_acc, "typing_acc": type_acc,
           "traits": traits}
    (C.RESULTS_DIR / "cluster.json").write_text(json.dumps(res, indent=2))
    print(f"cluster: K={best_k} (silhouette {sil}), identification top-1={id_acc:.0%}, "
          f"typing={type_acc:.0%}")
    print("  profiles:", json.dumps(profiles, indent=2))


if __name__ == "__main__":
    main()
