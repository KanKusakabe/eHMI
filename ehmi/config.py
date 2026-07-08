"""Paths, dataset column maps, and modeling scope for the eHMI project.

Data source: Bazilinskyy, Kooijman, ... De Winter (2022), "Get out of the way!
Examining eHMIs in critical driver-pedestrian encounters in a coupled simulator."
4TU.ResearchData, DOI 10.4121/20224281 (CC BY 4.0).

Raw logs are Unity "World Root" host-time logs, one CSV per trial, grouped into
Session1..Session20 (one folder per participant *duo* = one driver + one
pedestrian). Each CSV has 6 metadata rows, a 3-level header (rows 7-9:
agent / body-part / field), then numeric time-series rows at ~50-100 Hz.

Two scenarios are interleaved within each session, distinguished by column count:
  * Scenario 1 (S1): 190 columns  (1 crossing car "Driver0" + pedestrian)
  * Scenario 2 (S2): 209 columns  (2 cars + pedestrian)

The manually-driven car "Driver0" carries the directional eHMI (blue arrows);
its state lives in the ``blinkers`` column as one of {None, Left, Right} and is
our primary *cue*. The pedestrian wears a motion suit (full-body skeleton); we
model the ``avatar`` root planar motion as the *response*.
"""
from __future__ import annotations

from pathlib import Path

# --- paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_LOG_DIR = ROOT / "data" / "data"          # extracted Session*/ folders
RAW_DIR = ROOT / "data" / "raw"               # per-trial parquet (extract output)
PROC_DIR = ROOT / "data" / "processed"        # features.parquet, vocab, norm stats
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

# --- sampling ----------------------------------------------------------------
RATE_HZ = 10.0        # resample everything to this rate (matches motionsim)
DT = 1.0 / RATE_HZ

# --- modeling scope ----------------------------------------------------------
# 20 participant duos -> 20 pedestrians. Session number == participant id, which
# is also the LOUO fold id (the whole point: 20 users >> the 7 in motionsim).
PARTICIPANTS = list(range(1, 21))

# eHMI cue vocabulary (Driver0.blinkers). Index 0 is reserved for "no cue yet".
EHMI_STATES = ["None", "Left", "Right"]

# --- raw column indices (0-indexed) by scenario ------------------------------
# Verified from the 3-level CSV header. Unity ground plane = (pos_x, pos_z);
# pos_y is vertical (height). Heading = avatar rot_y (degrees).
COLS = {
    "s1": {  # 190 columns
        "time": 0,
        "drv_x": 2, "drv_h": 3, "drv_y": 4, "drv_yaw": 6,   # Driver0 pos_x, pos_y, pos_z, rot_y
        "ehmi": 8,                                          # Driver0 blinkers
        "ped_x": 46, "ped_h": 47, "ped_y": 48, "ped_yaw": 50,  # Pedestrian0.avatar
    },
    "s2": {  # 209 columns
        "time": 0,
        "drv_x": 2, "drv_h": 3, "drv_y": 4, "drv_yaw": 6,
        "ehmi": 8,
        "ped_x": 65, "ped_h": 66, "ped_y": 67, "ped_yaw": 69,
    },
}
NCOLS_TO_SCENARIO = {190: "s1", 209: "s2"}

# --- history / feature knobs -------------------------------------------------
HIST_STEPS = 10       # 1.0 s of motion history fed to the GRU encoder
HORIZON_STEPS = 20    # 2.0 s rollout horizon for ADE/FDE and replay fans

# --- feature/target schema (shared by features/model/train) ------------------
TARGET_COLS = ["dfwd", "dlat"]                    # per-step increment (crossing frame)
CONT_COLS = ["rng", "appr", "ttc"]                # continuous cue (standardized)
BEARING_COLS = ["bear_sin", "bear_cos"]           # driver bearing (already bounded)
TARGET_DIM = len(TARGET_COLS)
