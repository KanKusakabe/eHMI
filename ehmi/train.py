"""Leave-One-User-Out training/eval of the eHMI cue->response Flow.

Metrics per held-out participant: held-out NLL (Flow, MDN, + Gaussian and
constant-velocity baselines), autoregressive rollout ADE/FDE (Flow vs
constant-velocity), and calibration (nominal vs empirical coverage).

    uv run python -m ehmi.train              # smoke: hold out 1 participant
    uv run python -m ehmi.train --loocv      # full 20-fold LOUO
    uv run python -m ehmi.train --fast       # fewer epochs / strided windows
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from . import config as C
from .model import FlowModel, MDN, gaussian_nll, constant_velocity_nll

DEVICE = "cpu"  # small model; CPU is reliable across macOS/zuko


def _load():
    feats = pd.read_parquet(C.PROC_DIR / "features.parquet")
    norm = json.loads((C.PROC_DIR / "norm_stats.json").read_text())
    vocab = json.loads((C.PROC_DIR / "vocab.json").read_text())
    return feats, norm, vocab


def _std(a, col, norm):
    m, s = norm[col]
    return (a - m) / s


def build_windows(feats, norm, sess_idx, stride=1):
    """Slide a HIST_STEPS window within each trial -> flat sample arrays."""
    H = C.HIST_STEPS
    hist, x, cont, ehmi, scen, user, sess = [], [], [], [], [], [], []
    for uid, g in feats.groupby("trial_uid", sort=False):
        g = g.sort_values("k")
        inc = np.column_stack([_std(g[c].values, c, norm) for c in C.TARGET_COLS])
        con = np.column_stack(
            [_std(g[c].values, c, norm) for c in C.CONT_COLS]
            + [g[c].values for c in C.BEARING_COLS])
        s = int(g["session"].iloc[0])
        e = int(g["ehmi_idx"].iloc[0])
        sc = 0 if g["scenario"].iloc[0] == "s1" else 1
        for i in range(H, len(g), stride):
            hist.append(inc[i - H:i])
            x.append(inc[i])
            cont.append(con[i])
            ehmi.append(e); scen.append(sc); sess.append(s)
            user.append(sess_idx.get(s, len(sess_idx)))
    return {
        "hist": torch.tensor(np.array(hist), dtype=torch.float32),
        "x": torch.tensor(np.array(x), dtype=torch.float32),
        "cont": torch.tensor(np.array(cont), dtype=torch.float32),
        "ehmi": torch.tensor(ehmi, dtype=torch.long),
        "scen": torch.tensor(scen, dtype=torch.long),
        "user": torch.tensor(user, dtype=torch.long),
        "session": np.array(sess),
    }


def _batch(d, idx):
    return {k: d[k][idx] for k in ("hist", "x", "cont", "ehmi", "scen", "user")}


def train_model(net, data, epochs, bs=1024, lr=1e-3):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(data["x"])
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for j in range(0, n, bs):
            idx = perm[j:j + bs]
            b = _batch(data, idx)
            loss = -net.log_prob(b).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
    return net


@torch.no_grad()
def held_out_nll(net, data):
    n = len(data["x"]); tot = 0.0
    for j in range(0, n, 4096):
        b = _batch(data, torch.arange(j, min(j + 4096, n)))
        tot += float(-net.log_prob(b).sum())
    return tot / n


@torch.no_grad()
def rollout_metrics(net, feats, norm, sess_idx, held_session, n_samp=20):
    """Autoregressive ADE/FDE in the crossing frame for held-out trials."""
    H, K = C.HIST_STEPS, C.HORIZON_STEPS
    tm, tf = np.array([norm[c][0] for c in C.TARGET_COLS]), np.array([norm[c][1] for c in C.TARGET_COLS])
    flow_ade, flow_fde, cv_ade, disp = [], [], [], []
    sub = feats[feats["session"] == held_session]
    for uid, g in sub.groupby("trial_uid", sort=False):
        g = g.sort_values("k")
        if len(g) < H + K + 1:
            continue
        inc_raw = np.column_stack([g[c].values for c in C.TARGET_COLS])
        con = np.column_stack(
            [_std(g[c].values, c, norm) for c in C.CONT_COLS]
            + [g[c].values for c in C.BEARING_COLS])
        e = int(g["ehmi_idx"].iloc[0]); sc = 0 if g["scenario"].iloc[0] == "s1" else 1
        u = sess_idx.get(int(g["session"].iloc[0]), len(sess_idx))
        # start the rollout at crossing onset (first sustained forward motion),
        # so ADE/FDE reflect the response, not the stationary waiting phase.
        speed = np.abs(inc_raw[:, 0])
        moving = np.where(speed[H:] > 0.02)[0]
        start = H + int(moving[0]) if len(moving) else H
        start = min(start, len(g) - K - 1)
        truth = np.cumsum(inc_raw[start:start + K], axis=0)   # real path from start
        # Flow rollout: K steps, n_samp particles
        hist = ((inc_raw[start - H:start] - tm) / tf)
        parts = np.repeat(hist[None], n_samp, axis=0)         # (P,H,2) standardized
        pos = np.zeros((n_samp, 2))
        paths = np.zeros((n_samp, K, 2))
        for t in range(K):
            b = {"hist": torch.tensor(parts, dtype=torch.float32),
                 "cont": torch.tensor(np.repeat(con[start + t][None], n_samp, 0), dtype=torch.float32),
                 "ehmi": torch.full((n_samp,), e), "scen": torch.full((n_samp,), sc),
                 "user": torch.full((n_samp,), u)}
            s = net.sample(b, 1)[0].numpy()                   # (P,2) standardized inc
            inc = s * tf + tm
            pos += inc
            paths[:, t] = pos
            parts = np.concatenate([parts[:, 1:], s[:, None]], axis=1)
        med = np.median(paths, axis=0)
        flow_ade.append(np.linalg.norm(med - truth, axis=1).mean())
        flow_fde.append(np.linalg.norm(med[-1] - truth[-1]))
        disp.append(np.linalg.norm(paths - med[None], axis=2).mean())
        # constant-velocity baseline
        v = inc_raw[start - 1]
        cvpath = np.cumsum(np.repeat(v[None], K, 0), axis=0)
        cv_ade.append(np.linalg.norm(cvpath - truth, axis=1).mean())
    f = lambda a: float(np.mean(a)) if a else float("nan")
    return f(flow_ade), f(flow_fde), f(cv_ade), f(disp)


@torch.no_grad()
def calibration(net, data, n_samp=64, levels=(0.1, 0.3, 0.5, 0.7, 0.9)):
    """Per-dim central-interval coverage: empirical vs nominal."""
    n = min(len(data["x"]), 4000)
    idx = torch.randperm(len(data["x"]))[:n]
    b = _batch(data, idx)
    samp = net.sample(b, n_samp).numpy()          # (S,n,2)
    true = b["x"].numpy()
    cov = {}
    for p in levels:
        lo, hi = np.quantile(samp, (1 - p) / 2, 0), np.quantile(samp, (1 + p) / 2, 0)
        inside = ((true >= lo) & (true <= hi)).mean()
        cov[str(p)] = float(inside)
    return cov


def run_fold(feats, norm, held, sess_idx, epochs, stride):
    tr = feats[feats["session"] != held]
    te = feats[feats["session"] == held]
    dtr = build_windows(tr, norm, sess_idx, stride=stride)
    dte = build_windows(te, norm, sess_idx, stride=1)

    net = FlowModel(len(sess_idx)).to(DEVICE)
    train_model(net, dtr, epochs)
    mdn = MDN(len(sess_idx)).to(DEVICE)
    train_model(mdn, dtr, epochs)

    x_te = dte["x"].numpy()
    hist_last = dte["hist"][:, -1].numpy()
    res = {
        "held_session": held, "n_test": int(len(x_te)),
        "flow_nll": held_out_nll(net, dte),
        "mdn_nll": held_out_nll(mdn, dte),
        "gauss_nll": gaussian_nll(x_te),
        "cv_nll": constant_velocity_nll(hist_last, x_te),
    }
    fa, ff, cva, disp = rollout_metrics(net, feats, norm, sess_idx, held)
    res.update(flow_ade=fa, flow_fde=ff, cv_ade=cva, rollout_disp=disp)
    res["calibration"] = calibration(net, dte)
    return res, net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loocv", action="store_true", help="all 20 folds")
    ap.add_argument("--fast", action="store_true", help="fewer epochs, strided windows")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    feats, norm, vocab = _load()
    sessions = vocab["sessions"]
    sess_idx = {s: i for i, s in enumerate(sessions)}
    epochs = args.epochs or (6 if args.fast else 12)
    stride = 2 if args.fast else 1
    folds = sessions if args.loocv else [sessions[-1]]

    all_res, last_net = [], None
    for held in folds:
        torch.manual_seed(0)
        res, net = run_fold(feats, norm, held, sess_idx, epochs, stride)
        last_net = net
        print(f"  held s{held:02d}: flowNLL={res['flow_nll']:.3f} mdnNLL={res['mdn_nll']:.3f} "
              f"cvNLL={res['cv_nll']:.3f} gaussNLL={res['gauss_nll']:.3f} "
              f"flowADE={res['flow_ade']:.2f} cvADE={res['cv_ade']:.2f}")
        all_res.append(res)

    def agg(k):
        v = [r[k] for r in all_res if r[k] == r[k]]
        return {"mean": float(np.mean(v)), "std": float(np.std(v))} if v else None
    summary = {k: agg(k) for k in
               ["flow_nll", "mdn_nll", "gauss_nll", "cv_nll", "flow_ade", "flow_fde", "cv_ade"]}
    out = {"mode": "loocv" if args.loocv else "smoke", "epochs": epochs,
           "n_participants": len(sessions), "folds": all_res, "summary": summary}
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.RESULTS_DIR / "metrics.json").write_text(json.dumps(out, indent=2))

    torch.save({"state": last_net.state_dict(), "n_sessions": len(sessions),
                "sess_idx": sess_idx}, C.RESULTS_DIR / "flow_model.pt")
    print("summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
