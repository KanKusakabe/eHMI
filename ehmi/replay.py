"""Replay real trials top-down with the model's short-horizon prediction fan.

One panel, time-synced: pedestrian trail, approaching car (colored by eHMI), and
P sampled Flow rollouts of the next ~1.5 s fanned from the pedestrian.

    uv run python -m ehmi.replay                 # one auto-picked eHMI trial
    uv run python -m ehmi.replay --batch 6 --gif # 6 diverse trials, mp4 + gif
    uv run python -m ehmi.replay --trial <uid>
"""
from __future__ import annotations

import argparse
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import numpy as np
import pandas as pd
import torch

from . import config as C
from .model import FlowModel

EHMI_COLOR = {"None": "#888888", "Left": "#d1651f", "Right": "#2a6fb0"}


def _trial_frame(px, py):
    cxy = np.column_stack([px - px.mean(), py - py.mean()])
    _, _, Vt = np.linalg.svd(cxy, full_matrices=False)
    fwd = Vt[0]
    if np.array([px[-1] - px[0], py[-1] - py[0]]) @ fwd < 0:
        fwd = -fwd
    return fwd, np.array([-fwd[1], fwd[0]])


def _load_model():
    ckpt = torch.load(C.RESULTS_DIR / "flow_full.pt", map_location="cpu", weights_only=False)
    if not (C.RESULTS_DIR / "flow_full.pt").exists():
        ckpt = torch.load(C.RESULTS_DIR / "flow_model.pt", map_location="cpu", weights_only=False)
    net = FlowModel(ckpt["n_sessions"]); net.load_state_dict(ckpt["state"]); net.eval()
    return net, ckpt["sess_idx"]


def _travel(uid):
    r = pd.read_parquet(C.RAW_DIR / f"{uid}.parquet")
    return float(np.hypot(r["ped_x"].iloc[-1] - r["ped_x"].iloc[0],
                          r["ped_y"].iloc[-1] - r["ped_y"].iloc[0]))


def render_trial(uid, feats, meta, norm, net, sess_idx, samples, horizon, stride, gif):
    raw = pd.read_parquet(C.RAW_DIR / f"{uid}.parquet")
    g = feats[feats["trial_uid"] == uid].sort_values("k")
    ehmi = meta.loc[uid, "ehmi"] if uid in meta.index else "None"
    px, py = raw["ped_x"].values, raw["ped_y"].values
    cx, cy = raw["drv_x"].values, raw["drv_y"].values
    fwd, lat = _trial_frame(px, py)
    tm = np.array([norm[c][0] for c in C.TARGET_COLS]); tf = np.array([norm[c][1] for c in C.TARGET_COLS])
    inc_std = np.column_stack([(g[c].values - norm[c][0]) / norm[c][1] for c in C.TARGET_COLS])
    con = np.column_stack([(g[c].values - norm[c][0]) / norm[c][1] for c in C.CONT_COLS]
                          + [g[c].values for c in C.BEARING_COLS])
    e_idx = C.EHMI_STATES.index(ehmi) if ehmi in C.EHMI_STATES else 0
    sc = 0 if raw["scenario"].iloc[0] == "s1" else 1
    u = sess_idx.get(int(raw["session"].iloc[0]), len(sess_idx))
    n = min(len(px), len(g) + C.HIST_STEPS)
    H, P, K = C.HIST_STEPS, samples, horizon

    @torch.no_grad()
    def fan(i):
        gi = i - H
        if gi < H:
            return None
        parts = np.repeat(inc_std[gi - H:gi][None], P, 0)
        pos = np.zeros((P, 2)); paths = np.zeros((P, K, 2))
        for t in range(K):
            ci = min(gi + t, len(con) - 1)
            b = {"hist": torch.tensor(parts, dtype=torch.float32),
                 "cont": torch.tensor(np.repeat(con[ci][None], P, 0), dtype=torch.float32),
                 "ehmi": torch.full((P,), e_idx), "scen": torch.full((P,), sc),
                 "user": torch.full((P,), u)}
            s = net.sample(b, 1)[0].numpy()
            world = (s * tf + tm)[:, :1] * fwd[None] + (s * tf + tm)[:, 1:2] * lat[None]
            pos = pos + world; paths[:, t] = pos
            parts = np.concatenate([parts[:, 1:], s[:, None]], 1)
        return px[i] + paths[:, :, 0], py[i] + paths[:, :, 1]

    cxm, cym = (px.min() + px.max()) / 2, (py.min() + py.max()) / 2
    span = max(px.max() - px.min(), py.max() - py.min(), 8.0) / 2 + 3
    xlim, ylim = (cxm - span, cxm + span), (cym - span, cym + span)
    ecol = EHMI_COLOR.get(ehmi, "#888")
    fig, ax = plt.subplots(figsize=(7, 6.5))
    frames = list(range(H + 1, n, stride))

    def draw(i):
        ax.clear(); ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
        ax.plot(px[:i + 1], py[:i + 1], "-", color="#e05a7a", lw=2, label="pedestrian (real)")
        ax.plot(px[i], py[i], "o", color="#e05a7a", ms=9)
        f = fan(i)
        if f is not None:
            for p in range(f[0].shape[0]):
                ax.plot(f[0][p], f[1][p], "-", color="#2a6fb0", alpha=0.2, lw=0.8)
            ax.plot([], [], "-", color="#2a6fb0", alpha=0.6, label="Flow prediction fan (~1.5s)")
        vx, vy = cx[i] - px[i], cy[i] - py[i]; d = math.hypot(vx, vy) + 1e-9
        ex = np.clip(cx[i], xlim[0] + 0.5, xlim[1] - 0.5); ey = np.clip(cy[i], ylim[0] + 0.5, ylim[1] - 0.5)
        ax.annotate("", xy=(px[i] + vx / d * span * 0.9, py[i] + vy / d * span * 0.9),
                    xytext=(px[i], py[i]), arrowprops=dict(arrowstyle="->", color=ecol, alpha=0.5, lw=1.5))
        ax.plot(ex, ey, "s", color=ecol, ms=16, label=f"car (eHMI={ehmi})")
        ax.set_title(f"{uid}\nt={i * C.DT:.1f}s  eHMI={ehmi}  range→car={d:.1f} m", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        return []

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fps = C.RATE_HZ / stride
    mp4 = C.FIG_DIR / f"replay_{uid}.mp4"
    try:
        anim.save(str(mp4), writer=FFMpegWriter(fps=fps, bitrate=1800))
    except Exception:  # noqa: BLE001
        anim.save(str(mp4).replace(".mp4", ".gif"), writer=PillowWriter(fps=fps))
    if gif:
        anim.save(str(C.FIG_DIR / f"replay_{uid}.gif"), writer=PillowWriter(fps=fps), dpi=70)
    plt.close(fig)
    print(f"  wrote replay_{uid}")


def _pick_batch(feats, meta, n):
    by = {"Left": [], "Right": [], "None": []}
    for u in feats["trial_uid"].unique():
        if u in meta.index:
            by[meta.loc[u, "ehmi"]].append(u)
    per = max(1, math.ceil(n / 3))
    picks = []
    for cond in ("Left", "Right", "None"):
        picks += sorted(by[cond], key=_travel, reverse=True)[:per]
    return picks[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", default=None)
    ap.add_argument("--batch", type=int, default=0, help="render N diverse trials")
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()

    feats = pd.read_parquet(C.PROC_DIR / "features.parquet")
    meta = pd.read_parquet(C.PROC_DIR / "trial_meta.parquet").drop_duplicates("trial_uid").set_index("trial_uid")
    norm = json.loads((C.PROC_DIR / "norm_stats.json").read_text())
    net, sess_idx = _load_model()

    if args.batch:
        uids = _pick_batch(feats, meta, args.batch)
    elif args.trial:
        uids = [args.trial]
    else:
        cand = [u for u in feats["trial_uid"].unique()
                if u in meta.index and meta.loc[u, "ehmi"] in ("Left", "Right")]
        uids = [max(cand, key=_travel)] if cand else [feats["trial_uid"].iloc[0]]

    for uid in uids:
        render_trial(uid, feats, meta, norm, net, sess_idx,
                     args.samples, args.horizon, args.stride, args.gif)


if __name__ == "__main__":
    main()
