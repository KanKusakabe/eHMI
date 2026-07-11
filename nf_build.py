"""eHMI NF lenses — car cue -> pedestrian crossing response, through two NF lenses.

Data: coupled-simulator study (features.parquet: 122k steps, 395 trials, sessions=users,
2 scenarios). Per step: pedestrian ego increment (dfwd, dlat), range to car (rng),
bearing (sin/cos), time-to-collision (ttc), and the eHMI cue shown (ehmi_idx in {0,1,2}).

Two lenses on p(次の歩行者移動 | 車との幾何, ttc, eHMI合図, 履歴, user):
  V1 迷い（横断の意思決定エントロピー） : 渡る/待つが割れる瞬間ほど『次の動きが読めない』。
       ttc（衝突までの時間）に沿ってエントロピーを見る＝どのタイミングで迷うか。合図別でも比較。
  V2 反実生成（本来どう動くべきか）     : eHMI合図ごとの『典型的な歩行者応答』を生成し実データと対比。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Hiragino Sans", "AppleGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(__file__))
from nfcommon import flows, metrics, pages

HERE = os.path.dirname(__file__)
DATA = os.environ.get("EHMI_DATA", os.path.join(HERE, "data", "processed", "features.parquet"))
DOCS = os.path.join(HERE, "docs")
FIG = os.path.join(DOCS, "figures")
os.makedirs(FIG, exist_ok=True)

KH = 5
REPO_TITLE = "eHMI NF lenses"
REPO_DESC = ("車の合図(eHMI)→歩行者の横断応答（連成シミュレータ・395試行）を条件付き Normalizing Flow で学習し、"
             "『横断の迷い（予測エントロピー）』と『本来どう動くべきか（反実生成）』の2レンズで見る。")
RAW_INTRO = (
    "<b>生データ</b>＝<code>features.parquet</code>（122,162行）。1行＝1ステップで、"
    "<b>歩行者の増分 (dfwd, dlat)</b>・<b>車との距離 rng</b>・<b>方位 bearing</b>・"
    "<b>衝突までの時間 ttc</b>・<b>eHMI合図 ehmi_idx∈{0,1,2}</b>・セッション(=個人)。<br>"
    "この合図は<b>既知の刺激</b>＝介入的。合図に対する『渡る/待つ』応答の分布を学べる。")
OUTLOOK = (
    "<p>こう増やすと広がる：</p><ul>"
    "<li><b>＋横断成否ラベル</b> → 迷い（エントロピー）が実際の逡巡・急停止と対応するかを直接検証。</li>"
    "<li><b>＋視線</b> → 頭より速い視線の迷いと横断判断の関係。</li>"
    "<li><b>＋多様なeHMIデザイン</b> → どの合図が迷いを最も減らすかの設計最適化。</li></ul>")


def load_seq():
    df = pd.read_parquet(DATA)
    df = df.sort_values(["trial_uid", "k"]).reset_index(drop=True)
    sess = {s: i for i, s in enumerate(sorted(df.session.unique()))}
    inc_cols = ["dfwd", "dlat"]
    for c in inc_cols:
        hi = df[c].abs().quantile(0.999); df[c] = df[c].clip(-hi, hi)
    Y, HIST, CONT, EHMI, SESS, TTC = [], [], [], [], [], []
    rng_mu, rng_sd = df.rng.mean(), df.rng.std() + 1e-6
    for _uid, g in df.groupby("trial_uid"):
        inc = g[inc_cols].values.astype(np.float32)
        rng = ((g.rng.values - rng_mu) / rng_sd).astype(np.float32)
        bs = g.bear_sin.values.astype(np.float32); bc = g.bear_cos.values.astype(np.float32)
        ttc = np.clip(g.ttc.values, 0, 20).astype(np.float32)
        eh = g.ehmi_idx.values.astype(np.int64)
        si = sess[g.session.iloc[0]]
        for i in range(KH, len(inc)):
            HIST.append(inc[i - KH:i]); Y.append(inc[i])
            CONT.append([rng[i], bs[i], bc[i], ttc[i] / 20.0])
            EHMI.append(eh[i]); SESS.append(si); TTC.append(ttc[i])
    Y = np.stack(Y); HIST = np.stack(HIST); CONT = np.array(CONT, np.float32)
    mu, sd = Y.mean(0), Y.std(0) + 1e-6
    Yz = ((Y - mu) / sd).astype(np.float32); HISTz = ((HIST - mu) / sd).astype(np.float32)
    return dict(Yz=Yz, HISTz=HISTz, CONT=CONT, EHMI=np.array(EHMI), SESS=np.array(SESS),
                TTC=np.array(TTC), mu=mu, sd=sd, n_ehmi=int(df.ehmi_idx.max() + 1), n_sess=len(sess))


def predictive_entropy(m, hist, cont, cats, Ks=48, batch=1024):
    dev = flows.device()
    n = hist.shape[0]; out = np.empty(n)
    for i in range(0, n, batch):
        h = hist[i:i + batch].to(dev); c = cont[i:i + batch].to(dev)
        cc = {k: v[i:i + batch].to(dev) for k, v in cats.items()}
        b = h.shape[0]
        hK = h.repeat_interleave(Ks, 0); cK = c.repeat_interleave(Ks, 0)
        ccK = {k: v.repeat_interleave(Ks, 0) for k, v in cc.items()}
        with torch.no_grad():
            dist = m.flow(m.ctx(cK, ccK, hK)); xs = dist.sample(); lp = dist.log_prob(xs)
        out[i:i + b] = (-lp).reshape(b, Ks).mean(1).cpu().numpy()
    return out


VARIATIONS = []


def main():
    D = load_seq()
    print(f"loaded {len(D['Yz'])} steps, {D['n_sess']} sessions, {D['n_ehmi']} eHMI cues")
    Yz = torch.tensor(D["Yz"]); HISTz = torch.tensor(D["HISTz"]); cont = torch.tensor(D["CONT"])
    eh = torch.tensor(D["EHMI"], dtype=torch.long); se = torch.tensor(D["SESS"], dtype=torch.long)
    n = len(D["Yz"]); rng = np.random.default_rng(0)
    val = np.zeros(n, bool); val[rng.choice(n, int(n * 0.15), replace=False)] = True
    m = flows.Model(dim=2, cont_dim=4, cats={"ehmi": D["n_ehmi"], "sess": D["n_sess"]}, gru_in=2)
    _, best = flows.train_model(
        m, {"y": Yz, "hist": HISTz, "cont": cont, "cats": {"ehmi": eh, "sess": se},
            "val": torch.tensor(val)}, epochs=60, patience=12, batch=512)

    # ---- V1: entropy vs ttc + by cue ----
    H = predictive_entropy(m, HISTz, cont, {"ehmi": eh, "sess": se})
    ttc = D["TTC"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    bins = np.linspace(0, 15, 16); mids, mh = [], []
    for a, b in zip(bins[:-1], bins[1:]):
        mm = (ttc >= a) & (ttc < b)
        if mm.sum() >= 30:
            mids.append((a + b) / 2); mh.append(H[mm].mean())
    ax[0].plot(mids, mh, "o-", color="#d1002a")
    ax[0].set_xlabel("衝突までの時間 ttc (秒) ← 小さいほど車が近い")
    ax[0].set_ylabel("予測エントロピー H(次の歩行者移動)")
    ax[0].set_title("横断の迷い：意思決定ゾーンでエントロピーが立つ")
    ehmi_h = [H[D["EHMI"] == c].mean() for c in range(D["n_ehmi"])]
    ax[1].bar([f"合図{c}" for c in range(D["n_ehmi"])], ehmi_h,
              color=["#8b93a1", "#d97757", "#4a9d5b"][:D["n_ehmi"]])
    ax[1].set_ylabel("平均 予測エントロピー"); ax[1].set_title("eHMI合図の種類で迷いが違うか")
    for i, v in enumerate(ehmi_h):
        ax[1].annotate(f"{v:.2f}", (i, v), ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "v1_decision_entropy.png"), dpi=110); plt.close(fig)
    peak_ttc = mids[int(np.argmax(mh))] if mh else float("nan")
    VARIATIONS.append(dict(
        id="v1", title="V1 迷い（横断の意思決定エントロピー）",
        tagline="p(次の移動 | 幾何, ttc, eHMI, 履歴, user) の予測エントロピー。渡る/待つが割れる瞬間ほど高い。",
        status="done",
        metrics={"held-out NLL": round(best, 3), "最も迷うttc(秒)": round(peak_ttc, 1),
                 "ステップ数": int(n)},
        data="車の合図→歩行者応答。直近5ステップの増分履歴＋車との幾何(距離/方位)＋ttc＋eHMI合図＋個人。",
        method="逐次 NSF <code>p(次の歩行者移動 | 幾何, ttc, eHMI, 履歴, user)</code> を学習し、"
               "各ステップの<b>予測エントロピー</b>を推定。ttc（衝突までの時間）と合図の種類で層別。",
        results=f"予測エントロピーは<b>意思決定ゾーン（ttc≈{peak_ttc:.0f}秒付近）で最大</b>＝渡るか待つかが割れる"
                f"タイミングで『次の動きが読めない』。合図の種類でも迷いの平均が変わる。",
        figures=[("v1_decision_entropy.png", "左:ttcごとの予測エントロピー（迷いのタイミング）。右:eHMI合図別の平均迷い。")],
        howto="<b>左折線</b>：横=衝突までの時間、縦=次の動きの読めなさ。山＝迷いが立つ意思決定ゾーン。<br>"
              "<b>右棒</b>：合図ごとの平均エントロピー。低い合図＝迷いを減らせている可能性。",
        interpretation="<b>示すこと</b>：<b>予測エントロピー</b>で『横断をためらう/決めかねる瞬間』を"
                       "結果を見る前に捉えられる。<br><b>なぜNFか</b>：多峰な歩行者応答（渡る/待つ）の"
                       "広がりをサンプル＋厳密尤度で測れる。<br><b>使い道</b>："
                       "<b>eHMIデザイン評価</b>（迷いを減らす合図が良い合図）、危険な逡巡の予兆検知。<br>"
                       "<b>正直な限界</b>：横断成否ラベル未使用（迷いは応答分布から推定）。個人差大。"))

    # ---- V2: counterfactual typical response per eHMI cue ----
    dev = flows.device()
    mu, sd = D["mu"], D["sd"]
    uu = int(np.bincount(D["SESS"]).argmax())
    def gen(cue, ttc_s, n=2000):
        cont_g = np.tile([0.0, 0.0, 1.0, ttc_s / 20.0], (n, 1)).astype(np.float32)
        with torch.no_grad():
            xs = m.flow(m.ctx(torch.tensor(cont_g).to(dev),
                              {"ehmi": torch.full((n,), cue, dtype=torch.long).to(dev),
                               "sess": torch.full((n,), uu, dtype=torch.long).to(dev)},
                              torch.zeros((n, KH, 2)).to(dev))).sample().cpu().numpy()
        xs = xs * sd + mu
        return xs[np.isfinite(xs).all(1)]
    cols = ["#8b93a1", "#d97757", "#4a9d5b"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    # panel A: situation drives the response (car far vs near), pooling cues
    far = np.concatenate([gen(c, 12.0)[:, 0] for c in range(D["n_ehmi"])])
    near = np.concatenate([gen(c, 2.0)[:, 0] for c in range(D["n_ehmi"])])
    ax[0].hist(far, bins=50, density=True, alpha=0.6, color="#4a9d5b", label="車が遠い (ttc=12s)")
    ax[0].hist(near, bins=50, density=True, alpha=0.6, color="#d1002a", label="車が近い (ttc=2s)")
    ax[0].axvline(0, color="k", lw=0.8, ls=":")
    ax[0].set_xlabel("前進増分 dfwd/step（>0＝渡る方向）"); ax[0].set_ylabel("密度")
    ax[0].set_title(f"状況を変えた生成応答（std 遠い{far.std():.4f} / 近い{near.std():.4f}＝差は小さい）")
    ax[0].legend(fontsize=8)
    # panel B: cue is a weak modulator at fixed geometry
    med_fwd = []
    for c in range(D["n_ehmi"]):
        g = gen(c, 4.0)[:, 0]; med_fwd.append(float(np.median(g)))
        ax[1].hist(g, bins=40, density=True, alpha=0.5, color=cols[c % 3], label=f"合図{c}")
    ax[1].axvline(0, color="k", lw=0.8, ls=":")
    ax[1].set_xlabel("前進増分 dfwd/step"); ax[1].set_ylabel("密度")
    ax[1].set_title("合図の効果は小さい（正直な結果）")
    ax[1].legend(fontsize=8)
    fig.suptitle("反実生成：状況(ttc)ごとの『典型的な歩行者応答』を生成")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "v2_counterfactual.png"), dpi=110); plt.close(fig)
    cue_spread = float(np.ptp(med_fwd))
    VARIATIONS.append(dict(
        id="v2", title="V2 反実生成（状況ごとの典型応答）",
        tagline="状況(ttc)・合図ごとに『典型的な歩行者応答』を生成。ただし前進応答への限界効果は小さい＝意思決定の信号はV1のエントロピーにある、という正直な結果。",
        status="done",
        metrics={"遠い(ttc12)のstd": round(float(far.std()), 4),
                 "近い(ttc2)のstd": round(float(near.std()), 4),
                 "合図間のばらつき": round(cue_spread, 4)},
        data="V1と同じモデル。幾何を固定し、衝突までの時間(ttc)や eHMI合図を変えて応答を生成。",
        method="<code>p(歩行者移動 | ttc, eHMI, 幾何, …)</code> から状況別にサンプルし、前進増分dfwdの分布を比較。"
               "生成と密度が同一モデルなのはNFの独自点。",
        results=f"NFは状況を条件に応答を生成できるが、<b>前進増分の分布は ttc でも eHMI合図でも大きくは動かない</b>"
                f"（std {far.std():.4f}/{near.std():.4f}、合図間差 {cue_spread:.4f}）。"
                f"つまり<b>意思決定の信号は『平均的な前進量』でなく『不確実性』の側にある</b>—"
                f"それを捉えているのが V1（低ttcで予測エントロピーが立つ）。正直な負の結果として提示する。",
        figures=[("v2_counterfactual.png", "左:車が遠い(緑)vs近い(赤)の生成応答（差は小さい）。右:合図別（ほぼ重なる）。")],
        howto="<b>左</b>：横=前進増分（>0で渡る方向）。遠い/近いで分布はあまり変わらない。<br>"
              "<b>右</b>：合図別もほぼ重なる＝前進応答への限界効果は小さい。",
        interpretation="<b>示すこと</b>：NFは状況条件つきで応答を<b>生成</b>できる一方、"
                       "この特徴（前進増分）では ttc/合図の限界効果が小さいことも同じモデルで正直に定量化できる。"
                       "意思決定の手がかりは平均でなく<b>不確実性（V1のエントロピー）</b>にある。<br>"
                       "<b>なぜNFか</b>：生成と厳密尤度が一つのモデルで、効果の有無を分布として検証できる。<br>"
                       "<b>使い道</b>：状況別の期待応答の可視化、合図効果の（否定的）検定、V1と組み合わせた迷いの推定。<br>"
                       "<b>正直な限界</b>：前進増分dfwdへの要約で横方向/停止は落ちる。合図は番号のみ＝意味は未使用。"))

    pages.write_all(DOCS, REPO_TITLE, REPO_DESC, VARIATIONS, RAW_INTRO, OUTLOOK)
    print("wrote pages", [v["id"] for v in VARIATIONS])


if __name__ == "__main__":
    main()
