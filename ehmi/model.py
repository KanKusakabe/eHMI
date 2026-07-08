"""Conditional density model: encoder + Normalizing Flow head (zuko NSF).

Predicts p(x | c) where x = (dfwd, dlat) is the next 0.1 s pedestrian increment
and c summarizes: motion history (GRU), eHMI cue (embedding), driver kinematics
(range/approach/TTC + bearing sin/cos, MLP), scenario + participant (embeddings).
Mirrors the motionsim design on a 20-user cue->response corpus.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import zuko

from . import config as C


class Encoder(nn.Module):
    def __init__(self, n_sessions, ctx_dim=64):
        super().__init__()
        self.gru = nn.GRU(C.TARGET_DIM, 32, batch_first=True)
        self.ehmi_emb = nn.Embedding(len(C.EHMI_STATES), 8)
        self.scen_emb = nn.Embedding(2, 4)
        # +1 row for "unknown" held-out participant (LOUO / new user)
        self.user_emb = nn.Embedding(n_sessions + 1, 8)
        self.unknown_user = n_sessions
        cont_in = len(C.CONT_COLS) + len(C.BEARING_COLS)
        self.cont = nn.Sequential(nn.Linear(cont_in, 32), nn.ReLU(), nn.Linear(32, 16))
        self.head = nn.Sequential(
            nn.Linear(32 + 8 + 4 + 8 + 16, ctx_dim), nn.ReLU(),
            nn.Linear(ctx_dim, ctx_dim), nn.ReLU())
        self.ctx_dim = ctx_dim

    def forward(self, hist, ehmi, scen, user, cont):
        _, h = self.gru(hist)
        h = h.squeeze(0)
        z = torch.cat([h, self.ehmi_emb(ehmi), self.scen_emb(scen),
                       self.user_emb(user), self.cont(cont)], dim=-1)
        return self.head(z)


class FlowModel(nn.Module):
    def __init__(self, n_sessions):
        super().__init__()
        self.enc = Encoder(n_sessions)
        self.flow = zuko.flows.NSF(features=C.TARGET_DIM, context=self.enc.ctx_dim,
                                   transforms=3, hidden_features=(64, 64))

    def _ctx(self, batch):
        return self.enc(batch["hist"], batch["ehmi"], batch["scen"],
                        batch["user"], batch["cont"])

    def log_prob(self, batch):
        return self.flow(self._ctx(batch)).log_prob(batch["x"])

    @torch.no_grad()
    def sample(self, batch, n=1):
        return self.flow(self._ctx(batch)).sample((n,))


class MDN(nn.Module):
    """Diagonal-Gaussian mixture head -- a strong, simple learned baseline."""
    def __init__(self, n_sessions, k=5):
        super().__init__()
        self.enc = Encoder(n_sessions)
        self.k = k
        d = C.TARGET_DIM
        self.pi = nn.Linear(self.enc.ctx_dim, k)
        self.mu = nn.Linear(self.enc.ctx_dim, k * d)
        self.log_sd = nn.Linear(self.enc.ctx_dim, k * d)

    def log_prob(self, batch):
        c = self.enc(batch["hist"], batch["ehmi"], batch["scen"], batch["user"], batch["cont"])
        d = C.TARGET_DIM
        logpi = torch.log_softmax(self.pi(c), dim=-1)
        mu = self.mu(c).view(-1, self.k, d)
        sd = torch.exp(self.log_sd(c).view(-1, self.k, d).clamp(-6, 3))
        x = batch["x"].unsqueeze(1)
        comp = (-0.5 * (((x - mu) / sd) ** 2) - torch.log(sd) - 0.9189385).sum(-1)
        return torch.logsumexp(logpi + comp, dim=-1)


# --- non-learned baselines (closed form NLL on standardized target) ----------
def gaussian_nll(x_std):
    """Single global Gaussian fit to the (already standardized) target."""
    mu = x_std.mean(0)
    var = x_std.var(0) + 1e-6
    nll = 0.5 * (((x_std - mu) ** 2) / var + np.log(2 * np.pi * var)).sum(1)
    return float(nll.mean())


def constant_velocity_nll(hist_last, x_std, sd=None):
    """Predict next increment = previous increment, with a fitted isotropic sd."""
    resid = x_std - hist_last
    if sd is None:
        sd = resid.std(0) + 1e-6
    nll = 0.5 * ((resid / sd) ** 2 + np.log(2 * np.pi * sd ** 2)).sum(1)
    return float(nll.mean())
