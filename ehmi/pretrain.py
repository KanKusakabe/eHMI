"""Train ONE full-data Flow on all 20 users -> results/flow_full.pt.

Used by cluster.py (per-user embeddings = behavioral fingerprints) and causal.py
(conditional density for CATE + counterfactual abduction). Also serves as the
few-shot "oracle" (every user seen in training).
"""
from __future__ import annotations

import numpy as np
import torch

from . import config as C
from .model import FlowModel
from .train import _load, build_windows, train_model


def main(epochs: int = 15):
    feats, norm, vocab = _load()
    sessions = vocab["sessions"]
    sess_idx = {s: i for i, s in enumerate(sessions)}
    data = build_windows(feats, norm, sess_idx, stride=1)
    torch.manual_seed(0)
    net = FlowModel(len(sessions))
    train_model(net, data, epochs)
    emb = net.enc.user_emb.weight.detach().numpy()[:len(sessions)]  # 20 x 8
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "n_sessions": len(sessions),
                "sess_idx": sess_idx}, C.RESULTS_DIR / "flow_full.pt")
    np.save(C.RESULTS_DIR / "embeddings.npy", emb)
    print(f"pretrain: full model on {len(sessions)} users, {len(data['x'])} steps, "
          f"embeddings {emb.shape} -> results/flow_full.pt")


if __name__ == "__main__":
    main()
