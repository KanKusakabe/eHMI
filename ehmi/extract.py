"""Parse raw Unity coupled-sim CSV logs -> per-trial parquet resampled to RATE_HZ.

Idempotent: existing outputs are skipped unless --overwrite. Output columns:
    t, ped_x, ped_y, ped_h, ped_yaw, drv_x, drv_y, drv_yaw, ehmi (str),
    session, scenario, trial_uid

One row = one 10 Hz sample. ``ehmi`` is the Driver0 blinker state held at each
sample (None/Left/Right); the response we later model is the pedestrian avatar
planar motion.
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from . import config as C


def _scenario_of(path) -> str | None:
    """Peek a data row to get the column count -> scenario key."""
    with open(path, "r", errors="ignore") as f:
        for i, line in enumerate(f):
            if i < 9:
                continue
            n = line.count(";") + 1
            return C.NCOLS_TO_SCENARIO.get(n)
    return None


def _uid(path, session: int, scen: str) -> str:
    ts = re.search(r"(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})", path.name)
    return f"s{session:02d}_{scen}_{ts.group(1) if ts else path.stem[-8:]}"


def _ehmi_label(txt: str) -> str:
    t = str(txt).lower()
    if "link" in t:
        return "Left"
    if "recht" in t:
        return "Right"
    return "None"


def build_meta():
    """Attach per-trial eHMI condition + ratings from the interim questionnaire.

    The questionnaire has one row per presented trial (session in col 0,
    presentation order preserved). CSV files sorted by timestamp are also in
    presentation order, so the j-th sorted CSV of a session pairs with that
    session's j-th questionnaire row. Written to data/processed/trial_meta.parquet.
    """
    import pandas as pd

    qpath = C.RAW_LOG_DIR / "interim_questionnaire.xlsx"
    q = pd.read_excel(qpath, header=None).iloc[1:].reset_index(drop=True)
    q = q.rename(columns={0: "session", 1: "order", 2: "ehmi_txt", 3: "safety",
                          4: "seen", 5: "followed", 9: "missing"})
    rows = []
    for session in C.PARTICIPANTS:
        sdir = C.RAW_LOG_DIR / f"Session{session}"
        if not sdir.exists():
            continue
        csvs = sorted(sdir.glob("*.csv"))
        qrows = q[q["session"] == session].reset_index(drop=True)
        for j, path in enumerate(csvs):
            scen = _scenario_of(path)
            if scen is None:
                continue
            uid = _uid(path, session, scen)
            qr = qrows.iloc[j] if j < len(qrows) else None
            rows.append({
                "trial_uid": uid,
                "session": session,
                "scenario": scen,
                "ehmi": _ehmi_label(qr["ehmi_txt"]) if qr is not None else "None",
                "safety": float(qr["safety"]) if qr is not None and pd.notna(qr["safety"]) else float("nan"),
                "followed": str(qr["followed"]) if qr is not None else "",
                "missing": int(qr["missing"]) if qr is not None and pd.notna(qr["missing"]) else 0,
            })
    meta = pd.DataFrame(rows)
    C.PROC_DIR.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(C.PROC_DIR / "trial_meta.parquet", index=False)
    print(f"meta: {len(meta)} trials, eHMI {meta['ehmi'].value_counts().to_dict()}, "
          f"{int((meta['missing']==1).sum())} flagged missing")


def _resample(t, arr, t_new, kind="linear"):
    """1-D linear interpolation of columns of ``arr`` onto ``t_new``."""
    out = np.empty((len(t_new), arr.shape[1]))
    for j in range(arr.shape[1]):
        out[:, j] = np.interp(t_new, t, arr[:, j])
    return out


def extract_file(path, session: int, overwrite=False):
    scen = _scenario_of(path)
    if scen is None:
        return None  # unknown column count -> skip
    cm = C.COLS[scen]

    ts = re.search(r"(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})", path.name)
    uid = f"s{session:02d}_{scen}_{ts.group(1) if ts else path.stem[-8:]}"
    out_path = C.RAW_DIR / f"{uid}.parquet"
    if out_path.exists() and not overwrite:
        return out_path

    # rows 1-9 are metadata + 3-level header; data starts at line 10.
    raw = pd.read_csv(path, sep=";", skiprows=9, header=None, dtype=str,
                      engine="c", on_bad_lines="skip")
    if len(raw) < 20:
        return None

    ehmi_raw = raw.iloc[:, cm["ehmi"]].fillna("None").str.strip().values
    ehmi_raw = np.where(np.isin(ehmi_raw, C.EHMI_STATES), ehmi_raw, "None")

    def num(col):
        return pd.to_numeric(raw.iloc[:, col], errors="coerce").values

    t = num(cm["time"]).astype(float)
    cols = {k: num(cm[k]) for k in
            ("ped_x", "ped_y", "ped_h", "ped_yaw", "drv_x", "drv_y", "drv_yaw")}

    good = np.isfinite(t) & np.isfinite(cols["ped_x"]) & np.isfinite(cols["ped_y"])
    if good.sum() < 20:
        return None
    t = t[good]
    t = t - t[0]
    ehmi_raw = ehmi_raw[good]
    stack = np.column_stack([cols[k][good] for k in cols])
    stack = pd.DataFrame(stack).interpolate(limit_direction="both").values

    # resample to a uniform 10 Hz grid
    dur = t[-1]
    t_new = np.arange(0.0, dur, C.DT)
    if len(t_new) < 10:
        return None
    res = _resample(t, stack, t_new)

    # eHMI: zero-order hold onto the new grid (nearest previous sample)
    idx = np.searchsorted(t, t_new, side="right") - 1
    idx = np.clip(idx, 0, len(ehmi_raw) - 1)
    ehmi_new = ehmi_raw[idx]

    df = pd.DataFrame({
        "t": t_new,
        "ped_x": res[:, 0], "ped_y": res[:, 1], "ped_h": res[:, 2], "ped_yaw": res[:, 3],
        "drv_x": res[:, 4], "drv_y": res[:, 5], "drv_yaw": res[:, 6],
        "ehmi": ehmi_new,
    })
    df["session"] = session
    df["scenario"] = scen
    df["trial_uid"] = uid
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    n_ok = 0
    for session in C.PARTICIPANTS:
        sdir = C.RAW_LOG_DIR / f"Session{session}"
        if not sdir.exists():
            continue
        for path in sorted(sdir.glob("*.csv")):
            try:
                out = extract_file(path, session, overwrite=args.overwrite)
                if out is not None:
                    n_ok += 1
            except Exception as e:  # noqa: BLE001
                print(f"  ! {path.name}: {e}")
    print(f"extracted {n_ok} trials -> {C.RAW_DIR}")
    build_meta()


if __name__ == "__main__":
    main()
