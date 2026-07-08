"""Turn per-trial parquet into egocentric increments + cue conditioning.

Prediction target x = (dfwd, dlat, dyaw): the pedestrian's own-frame motion
increment over one 0.1 s step (frame-invariant, so scenario geometry is handled
automatically -- same trick as motionsim).

Conditioning c per step:
  * eHMI condition (None/Left/Right)            -- the designed cue (trial-level)
  * approaching-car kinematics: range, bearing (sin/cos), approach speed, TTC
                                                -- the temporally-varying cue
  * participant id (session) + scenario         -- context / LOUO fold
  * motion history (windowed at train time from the stored increments)

Outputs:
  data/processed/features.parquet   one row per usable step
  data/processed/norm_stats.json    mean/std for target + continuous cond
  data/processed/vocab.json         eHMI states, scenarios, sessions
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C

EPS = 1e-6


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def trial_features(df: pd.DataFrame, ehmi_idx: int) -> pd.DataFrame | None:
    if len(df) < C.HIST_STEPS + C.HORIZON_STEPS + 2:
        return None
    px, py = df["ped_x"].values, df["ped_y"].values
    dx_g, dy_g = df["drv_x"].values, df["drv_y"].values

    # global increment of the pedestrian
    ddx = np.diff(px, prepend=px[0])
    ddy = np.diff(py, prepend=py[0])

    # Per-trial "crossing frame": rotate so the principal axis of the trajectory
    # (the crossing direction) is +forward and the perpendicular is +lateral.
    # The motion-suit avatar yaw is logged constant, so this fixed rotation is a
    # stable, interpretable frame: dfwd = advance/retreat, dlat = veer L/R (which
    # is exactly what a directional eHMI is meant to influence). It is a rotation
    # normalization only (applied equally to increments and the driver-relative
    # vector), so it introduces no target leakage.
    cxy = np.column_stack([px - px.mean(), py - py.mean()])
    _, _, Vt = np.linalg.svd(cxy, full_matrices=False)
    fwd = Vt[0]
    net = np.array([px[-1] - px[0], py[-1] - py[0]])
    if net @ fwd < 0:
        fwd = -fwd
    lat = np.array([-fwd[1], fwd[0]])
    # Orient +lat to point AWAY from the car (car sits on the -lat side on
    # average) so lateral sign is consistent across trials: +dlat = evade away.
    rel_x, rel_y = dx_g - px, dy_g - py
    if np.nanmean(rel_x * lat[0] + rel_y * lat[1]) > 0:
        lat = -lat
    dfwd = ddx * fwd[0] + ddy * fwd[1]
    dlat = ddx * lat[0] + ddy * lat[1]

    # driver-relative cue, expressed in the same crossing frame
    rng = np.hypot(rel_x, rel_y)
    b = np.arctan2(rel_x * lat[0] + rel_y * lat[1], rel_x * fwd[0] + rel_y * fwd[1])
    appr = -np.gradient(rng) / C.DT                      # +ve = car closing in
    ttc = np.clip(rng / np.clip(appr, EPS, None), 0, 20)

    out = pd.DataFrame({
        "dfwd": dfwd, "dlat": dlat,
        "rng": rng, "bear_sin": np.sin(b), "bear_cos": np.cos(b),
        "appr": appr, "ttc": ttc,
    })
    out["k"] = np.arange(len(out))
    out["trial_uid"] = df["trial_uid"].iloc[0]
    out["session"] = int(df["session"].iloc[0])
    out["scenario"] = df["scenario"].iloc[0]
    out["ehmi_idx"] = ehmi_idx
    # drop the first row (increment undefined) and any non-finite rows
    out = out.iloc[1:].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return out if len(out) > C.HIST_STEPS + 2 else None


def main():
    meta = (pd.read_parquet(C.PROC_DIR / "trial_meta.parquet")
            .drop_duplicates("trial_uid", keep="first").set_index("trial_uid"))
    ehmi_to_idx = {s: i for i, s in enumerate(C.EHMI_STATES)}
    frames = []
    for path in sorted(C.RAW_DIR.glob("*.parquet")):
        df = pd.read_parquet(path)
        uid = df["trial_uid"].iloc[0]
        if uid in meta.index and int(meta.loc[uid, "missing"]) == 1:
            continue  # experimenter-flagged corrupt trial
        ehmi = meta.loc[uid, "ehmi"] if uid in meta.index else "None"
        f = trial_features(df, ehmi_to_idx.get(ehmi, 0))
        if f is not None:
            frames.append(f)
    feats = pd.concat(frames, ignore_index=True)

    tgt = list(C.TARGET_COLS)
    con = list(C.CONT_COLS)
    norm = {c: [float(feats[c].mean()), float(feats[c].std() + EPS)] for c in tgt + con}
    sessions = sorted(feats["session"].unique().tolist())

    C.PROC_DIR.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(C.PROC_DIR / "features.parquet", index=False)
    (C.PROC_DIR / "norm_stats.json").write_text(json.dumps(norm, indent=2))
    (C.PROC_DIR / "vocab.json").write_text(json.dumps(
        {"ehmi": C.EHMI_STATES, "scenarios": ["s1", "s2"], "sessions": sessions}, indent=2))

    print(f"features: {len(feats)} steps from {feats['trial_uid'].nunique()} trials, "
          f"{len(sessions)} participants")
    print("  target std:", {c: round(float(feats[c].std()), 4) for c in tgt})
    print("  eHMI step counts:", feats["ehmi_idx"].value_counts().to_dict())


if __name__ == "__main__":
    main()
