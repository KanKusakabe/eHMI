"""WP1 — few-shot personalization of a new user from k of their own trials.

(a) EMBEDDING-ONLY adaptation from a standard population model: freeze everything,
    optimize only the held-out user's 8-dim embedding on k support trials.
(d) REPTILE meta-init + full fine-tune: meta-learn an initialization from which a
    few adaptation steps on k trials generalize (run on a bounded user subset).

Both evaluated LOUO: population/meta trained WITHOUT the held-out user; adapt on k
of their trials; score held-out NLL on their remaining trials. Curve vs k tells us
"how many trials to personalize a new person" — the individual-differences bottleneck.

    uv run python -m ehmi.fewshot           # (a) all users + (d) subset
    uv run python -m ehmi.fewshot --fast
"""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np
import torch

from . import config as C
from .model import FlowModel
from .train import _load, build_windows, train_model, held_out_nll

K_LIST = [0, 1, 2, 3, 5, 8]
N_DRAWS = 3           # random support draws to average per k
ADAPT_STEPS = 80
META_USERS = 6        # (d) is compute-heavy -> bounded subset
META_ITERS = 100


def _user_trials(feats, sess):
    return list(feats[feats["session"] == sess]["trial_uid"].unique())


def _win(feats, uids, norm, sess_idx):
    return build_windows(feats[feats["trial_uid"].isin(uids)], norm, sess_idx, stride=1)


def _adapt_embed(net, support, uidx, steps=ADAPT_STEPS, lr=0.05):
    """Optimize ONLY the held-out user's embedding row on support trials."""
    for p in net.parameters():
        p.requires_grad_(False)
    emb = net.enc.user_emb.weight
    emb.requires_grad_(True)
    opt = torch.optim.Adam([emb], lr=lr)
    n = len(support["x"])
    for _ in range(steps):
        idx = torch.randperm(n)[:1024]
        b = {k: support[k][idx] for k in ("hist", "x", "cont", "ehmi", "scen", "user")}
        loss = -net.log_prob(b).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    emb.requires_grad_(False)


def _eval_user_curve(feats, norm, sess_idx, U, epochs, stride, oracle_nll):
    """(a) embedding-only few-shot curve for one held-out user U."""
    uidx = sess_idx[U]
    unknown = FlowModel(len(sess_idx)).enc.unknown_user
    tr = build_windows(feats[feats["session"] != U], norm, sess_idx, stride=stride)
    torch.manual_seed(U)
    base = FlowModel(len(sess_idx))
    train_model(base, tr, epochs)

    trials = _user_trials(feats, U)
    rng = np.random.default_rng(U)
    out = {}
    for k in K_LIST:
        vals = []
        draws = 1 if k == 0 else N_DRAWS
        for _ in range(draws):
            net = copy.deepcopy(base)
            with torch.no_grad():
                net.enc.user_emb.weight[uidx] = net.enc.user_emb.weight[unknown].clone()
            if k == 0:
                query = trials
            else:
                sup = list(rng.choice(trials, size=min(k, len(trials) - 1), replace=False))
                query = [t for t in trials if t not in sup]
                _adapt_embed(net, _win(feats, sup, norm, sess_idx), uidx)
            vals.append(held_out_nll(net, _win(feats, query, norm, sess_idx)))
        out[k] = float(np.mean(vals))
    out["oracle"] = oracle_nll
    return out


def _reptile_meta(feats, norm, sess_idx, U, epochs, stride):
    """(d) Reptile meta-init trained WITHOUT user U (full-model inner adapt)."""
    meta = FlowModel(len(sess_idx))
    tr = build_windows(feats[feats["session"] != U], norm, sess_idx, stride=stride)
    train_model(meta, tr, max(2, epochs // 2))          # warm start
    train_users = [s for s in sess_idx if s != U]
    rng = np.random.default_rng(1000 + U)
    for _ in range(META_ITERS):
        u = int(rng.choice(train_users))
        sup = list(rng.choice(_user_trials(feats, u), size=3, replace=False))
        work = copy.deepcopy(meta)
        train_model(work, _win(feats, sup, norm, sess_idx), epochs=2)
        with torch.no_grad():
            for pm, pw in zip(meta.parameters(), work.parameters()):
                pm += 0.2 * (pw - pm)
    return meta


def _eval_meta_curve(feats, norm, sess_idx, U, epochs, stride):
    meta = _reptile_meta(feats, norm, sess_idx, U, epochs, stride)
    trials = _user_trials(feats, U)
    rng = np.random.default_rng(2000 + U)
    out = {}
    for k in K_LIST:
        if k == 0:
            out[k] = held_out_nll(meta, _win(feats, trials, norm, sess_idx))
            continue
        vals = []
        for _ in range(N_DRAWS):
            sup = list(rng.choice(trials, size=min(k, len(trials) - 1), replace=False))
            query = [t for t in trials if t not in sup]
            work = copy.deepcopy(meta)
            train_model(work, _win(feats, sup, norm, sess_idx), epochs=3)  # full finetune
            vals.append(held_out_nll(work, _win(feats, query, norm, sess_idx)))
        out[k] = float(np.mean(vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    feats, norm, vocab = _load()
    sessions = vocab["sessions"]
    sess_idx = {s: i for i, s in enumerate(sessions)}
    epochs = 6 if args.fast else 10
    stride = 2 if args.fast else 1

    # oracle = full model (user seen in training) per user
    oracle = {}
    try:
        ck = torch.load(C.RESULTS_DIR / "flow_full.pt", map_location="cpu", weights_only=False)
        full = FlowModel(ck["n_sessions"]); full.load_state_dict(ck["state"]); full.eval()
        for s in sessions:
            oracle[s] = held_out_nll(full, _win(feats, _user_trials(feats, s), norm, sess_idx))
    except Exception as e:  # noqa: BLE001
        print(f"  (oracle unavailable: {e})")

    embed = {}
    for U in sessions:
        embed[U] = _eval_user_curve(feats, norm, sess_idx, U, epochs, stride,
                                    oracle.get(U, float("nan")))
        print(f"  (a) user {U:2d}: " + " ".join(f"k{k}={embed[U][k]:+.2f}" for k in K_LIST))

    meta = {}
    for U in sessions[:META_USERS]:
        try:
            meta[U] = _eval_meta_curve(feats, norm, sess_idx, U, epochs, stride)
            print(f"  (d) user {U:2d}: " + " ".join(f"k{k}={meta[U][k]:+.2f}" for k in K_LIST))
        except Exception as e:  # noqa: BLE001
            print(f"  (d) user {U} failed: {e}")

    res = {"k_list": K_LIST, "embed": embed, "meta": meta,
           "oracle": oracle, "meta_users": sessions[:META_USERS]}
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.RESULTS_DIR / "fewshot.json").write_text(json.dumps(res, indent=2))

    def curve(d, users):
        return [float(np.mean([d[u][k] for u in users if u in d])) for k in K_LIST]
    print("  (a) embed mean curve:", [round(x, 3) for x in curve(embed, sessions)])
    if meta:
        print("  (d) meta  mean curve:", [round(x, 3) for x in curve(meta, sessions[:META_USERS])])
    print("wrote results/fewshot.json")


if __name__ == "__main__":
    main()
