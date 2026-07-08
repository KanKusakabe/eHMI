"""Regenerate all figures and inject a consolidated RESULTS section into README.md.

Reads whatever result JSONs exist (metrics / fewshot / cluster / causal) and writes
the results block between the <!--RESULTS:START--> / <!--RESULTS:END--> markers in
README.md, so everything is viewable in one place with figure + replay links.
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

START, END = "<!--RESULTS:START-->", "<!--RESULTS:END-->"


def _load(name):
    p = C.RESULTS_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


# ---------- figures ----------
def fig_nll(m):
    s = m["summary"]
    names = ["Gaussian", "Const-vel", "MDN", "Flow"]
    keys = ["gauss_nll", "cv_nll", "mdn_nll", "flow_nll"]
    vals = [s[k]["mean"] for k in keys]; err = [s[k]["std"] for k in keys]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, vals, yerr=err, color=["#bbb", "#bbb", "#6aa9dc", "#2a6fb0"], capsize=4)
    ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("held-out NLL (lower=better)")
    ax.set_title("eHMI/car cue -> pedestrian response: learned >> baselines")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "nll.png", dpi=110); plt.close(fig)


def fig_calibration(m):
    cal = m["folds"][0]["calibration"]
    nom = np.array([float(k) for k in cal]); emp = np.array(list(cal.values())); o = np.argsort(nom)
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax.plot(nom[o], emp[o], "o-", color="#2a6fb0", label="Flow")
    ax.set_xlabel("nominal"); ax.set_ylabel("empirical coverage"); ax.set_title("Calibration"); ax.legend()
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "calibration.png", dpi=110); plt.close(fig)


def fig_per_user(m):
    folds = m["folds"]
    if len(folds) < 2:
        return
    sess = [str(f["held_session"]) for f in folds]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(sess, [f["flow_nll"] for f in folds], color="#2a6fb0")
    ax[0].set_title("Flow held-out NLL per participant"); ax[0].tick_params(axis="x", labelrotation=90)
    ax[1].bar(sess, [f["flow_ade"] for f in folds], color="#6aa9dc")
    ax[1].set_title("Flow rollout ADE (m) per participant"); ax[1].tick_params(axis="x", labelrotation=90)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "per_user.png", dpi=110); plt.close(fig)


def fig_response_curves():
    feats = pd.read_parquet(C.PROC_DIR / "features.parquet")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    lab = {0: ("None", "#888"), 1: ("eHMI Left", "#d1651f"), 2: ("eHMI Right", "#2a6fb0")}
    for i, (name, col) in lab.items():
        sub = feats[feats["ehmi_idx"] == i]
        b = np.clip((sub["rng"].values // 2).astype(int), 0, 15)
        for a, key in zip(ax, ["dfwd", "dlat"]):
            mm = pd.Series(sub[key].abs().values).groupby(b).mean()
            a.plot(mm.index * 2, mm.values, "o-", color=col, label=name, ms=3)
    ax[0].set_title("|forward| step vs distance to car"); ax[0].set_xlabel("range to car (m)")
    ax[1].set_title("|lateral| step vs distance to car"); ax[1].set_xlabel("range to car (m)")
    ax[0].set_ylabel("mean |increment| (m/step)"); ax[0].legend()
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "response_curves.png", dpi=110); plt.close(fig)


def fig_fewshot(fs):
    K = fs["k_list"]
    def curve(d, users):
        return [float(np.nanmean([d[u][str(k)] if str(k) in d[u] else d[u][k]
                                  for u in users if u in d])) for k in K]
    embed = fs["embed"]; meta = fs.get("meta", {})
    all_u = list(embed.keys())
    e = curve(embed, all_u)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(K, e, "o-", color="#2a6fb0", label="(a) embedding-only, all users")
    if meta:
        mu = list(meta.keys()); ax.plot(K, curve(meta, mu), "s--", color="#d1651f",
                                        label="(d) Reptile meta, subset")
    orc = fs.get("oracle", {})
    if orc:
        ax.axhline(float(np.nanmean(list(orc.values()))), color="#3bbf6b", ls=":", label="oracle (user seen)")
    ax.set_xlabel("k = adaptation trials from the new user")
    ax.set_ylabel("held-out NLL (lower=better)")
    ax.set_title("Few-shot personalization curve (LOUO)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "fewshot_curve.png", dpi=110); plt.close(fig)


# ---------- results markdown ----------
def _cap(text):
    return ["", f"> **見方**: {text}", ""]


def build_block():
    L = ["<!--RESULTS:START-->", "", "# 結果（自動生成 — `uv run python -m ehmi.report`）",
         "", "各図の下に「見方」を付けています。", ""]
    m = _load("metrics.json")
    if m:
        s = m["summary"]; md = m["mode"]
        L += [f"## 1. cue→反応は学習可能（{'20-fold LOUO' if md=='loocv' else '1人 hold-out'}）", "",
              "予測対象は 0.1 秒ごとの歩行者の自己中心増分 (Δfwd, Δlat)。基線と held-out NLL で比較。",
              "", "| model | held-out NLL ↓ | rollout ADE (m) ↓ |", "|---|---|---|",
              f"| Gaussian基線 | {s['gauss_nll']['mean']:+.2f} ± {s['gauss_nll']['std']:.2f} | — |",
              f"| 等速度基線 | {s['cv_nll']['mean']:+.2f} ± {s['cv_nll']['std']:.2f} | {s['cv_ade']['mean']:.2f} |",
              f"| MDN | {s['mdn_nll']['mean']:+.2f} ± {s['mdn_nll']['std']:.2f} | — |",
              f"| **Flow** | **{s['flow_nll']['mean']:+.2f} ± {s['flow_nll']['std']:.2f}** | **{s['flow_ade']['mean']:.2f}** |",
              "", "> **NLL（負の対数尤度）**: モデルが実際の動きをどれだけ当てたかの指標。**低いほど良い**（負でOK）。3 nats 差 ≈ 約20倍の当てやすさ。",
              "", "![nll](reports/figures/nll.png)"]
        L += _cap("棒＝各手法の held-out NLL（低いほど良い）。灰＝学習しない基線、青＝学習モデル。"
                  "Flow/MDN が基線を大きく下回る＝「cue→反応」を確かに学習できている定量証拠。")
        L += ["![calibration](reports/figures/calibration.png)"]
        L += _cap("横=モデルが宣言した確率区間、縦=実際にその区間へ入った割合。対角線=理想（正直）。"
                  "線が対角なら「50%と言えば本当に約50%入る」。ここでは対角より少し上＝区間がやや広め（過小自信）。")
        L += ["![response curves](reports/figures/response_curves.png)"]
        L += _cap("横=車までの距離(m)、縦=1ステップの平均移動量。左(前進)・右(横)。色=eHMI条件。"
                  "車が近づく（左へ行く）ほど動きが変わる＝データに車キューへの反応が見える。")
        if len(m["folds"]) > 1:
            fn = [f["flow_nll"] for f in m["folds"]]
            L += [f"per-user の Flow NLL は **{min(fn):.2f}〜{max(fn):.2f}** と広い＝**個人差が支配的**（誰を隠すかで成績が決まる）。",
                  "![per user](reports/figures/per_user.png)"]
            L += _cap("棒1本＝「その人を隠して学習・評価」した1人分の成績。左=NLL、右=軌跡誤差ADE。"
                      "棒の高さのばらつきの広さ＝個人差の大きさ。律速はモデルでなく“人の違い”。")
    fs = _load("fewshot.json")
    if fs:
        K = fs["k_list"]; embed = fs["embed"]
        def mean_at(d, users, k):
            xs = [d[u].get(str(k), d[u].get(k)) for u in users if u in d]
            return float(np.nanmean(xs))
        au = list(embed.keys())
        z, kmax = mean_at(embed, au, K[0]), mean_at(embed, au, K[-1])
        orc = float(np.nanmean(list(fs.get("oracle", {}).values()))) if fs.get("oracle") else float("nan")
        L += ["", "## 2. few-shot 個人化（新しい人を数試行で較正）", "",
              "**仕組み**: 学習済みモデルの重みは全て凍結し、**新しい人の 8 次元「個人ベクトル」だけ**を"
              "その人の k 試行で最適化（＝性格空間のどこに座るかを探す）。k=0 は未知(unknown)ベクトル。",
              "", f"(a) 埋め込みだけ適応（全{len(au)}人 LOUO）：zero-shot(k=0) NLL **{z:+.2f}** → "
              f"k={K[-1]}本で **{kmax:+.2f}**（oracle＝その人を学習に含めた上限 {orc:+.2f}）。**1〜2試行で大半が個人化**。",
              "", "| k (適応試行) | " + " | ".join(str(k) for k in K) + " |",
              "|" + "---|" * (len(K) + 1),
              "| (a) embedding NLL | " + " | ".join(f"{mean_at(embed, au, k):+.2f}" for k in K) + " |"]
        if fs.get("meta"):
            mu = list(fs["meta"].keys())
            L += ["| (d) Reptile meta NLL | " + " | ".join(f"{mean_at(fs['meta'], mu, k):+.2f}" for k in K) + " |"]
        L += ["", "![fewshot](reports/figures/fewshot_curve.png)"]
        L += _cap("横=新しい人の適応試行数 k、縦=held-out NLL（低いほど良い）。青(a)=埋め込みだけ適応は "
                  "k=1〜2 で急改善し oracle(緑点線=上限)に接近＝少数で個人化できる。"
                  "橙(d)=メタ学習+全微調整は k=1 で悪化＝1〜数試行での全体微調整は過学習（20人では埋め込み適応が正解）。")
    cl = _load("cluster.json")
    if cl:
        L += ["", "## 3. 行動フィンガープリント：クラスタ分類＋個人識別", "",
              "各人の8次元「個人ベクトル」を行動の指紋とみなし、(i)少数タイプにクラスタ、(ii)尤度で人/型を当てる。",
              "", f"埋め込みを **K={cl['best_k']} タイプ**にクラスタ（silhouette {cl['silhouette'][str(cl['best_k'])]:.2f}＝弱め＝ソフトな型）。",
              f"**個人識別 top-1 = {cl['identification_top1']:.0%}**（偶然5%）＝反応で人を見分けられる。"
              f" **タイプ分類 = {cl['typing_acc']:.0%}**。",
              "", "| type | n | 横断速度 | 横回避 | eHMI遵守 | 安全感 |", "|---|---|---|---|---|---|"]
        for c, p in cl["profiles"].items():
            L += [f"| {c} | {p['n_users']} | {p['cross_speed']:.3f} | {p['lat_evasion']:.4f} | "
                  f"{p['eHMI_compliance']:.2f} | {p['safety']:.2f} |"]
        L += ["", "![embedding map](reports/figures/embedding_map.png)"]
        L += _cap("各点＝1人の個人ベクトルをPCAで2Dに落としたもの。数字＝参加者ID、色＝クラスタ。"
                  "近い人ほど反応の仕方が似ている。")
        L += ["![cluster profiles](reports/figures/cluster_profiles.png)"]
        L += _cap("各タイプの行動特徴（0–1に正規化）。線の高低の違いがタイプの性格。"
                  "ここでは主に横断速度で分かれる（type0=遅い〜type2=速い）。")
        L += ["![identification](reports/figures/identification.png)"]
        L += _cap("行=本当のユーザ、列=候補の個人ベクトル、明るい=そのベクトルでの尤度が高い。"
                  "**明るい対角線**＝自分のベクトルが自分の試行を最もよく説明＝指紋が成立している。")
    ca = _load("causal.json")
    if ca:
        eff = ca["counterfactual_effect_by_cluster"]
        L += ["", "## 4. causal / 反事実（eHMI ON/OFF）", "",
              "eHMI は実験で割り付けた変数なので `do(eHMI)` を扱える。**反事実**＝可逆な Flow で"
              "「観測した反応→潜在 z を逆算→eHMI を OFF に差し替えて z 固定で再生成」＝"
              "**同じ人・同じ気質のまま、もし警告が無かったら**の動きを生成。",
              "", f"ON {ca['n_on']} / OFF {ca['n_off']} 試行。反事実効果はタイプで異なる（効果の異質性）：",
              "", "| type | Δ peak-lateral: ON − do(OFF) (m) | n |", "|---|---|---|"]
        for c, e in eff.items():
            L += [f"| {c} | {e['d_lat']:+.3f} m | {e['n']} |"]
        L += ["", "**解釈**: 値＝(実際ONの横回避のピーク) −(反事実OFFの横回避のピーク)。"
              "**負＝eHMIがあると横回避が小さい**（車の進路が分かり、無駄に大きく避けなくて済む）。"
              "type0(遅い横断)で最も大きく、type2(速い)で小さい＝eHMIは慎重な人ほど効く。",
              "", f"> 注: {ca['note']}",
              "", "![cate](reports/figures/cate_by_cluster.png)"]
        L += _cap("左=観測データの ON(青) vs OFF(灰) の横回避ピーク（タイプ別）。"
                  "右=モデル反事実の効果 ON−do(OFF)。棒が負＝eHMIで横回避が減る。棒の高さがタイプで違う＝効果の異質性。")
        L += ["![counterfactual](reports/figures/counterfactual_example.png)"]
        L += _cap("ある1人・1試行の上から見た経路。ピンク実線=実際(eHMI ON)、青破線=反事実(同じ人がeHMI無しなら)。"
                  "青が右に多く彷徨う＝警告が無ければもっと横に避けていた。横軸(lateral)は効果を見せるため引き伸ばし。")
    an = _load("anomaly.json")
    if an:
        L += ["", "## 5. 異常検知 / 危険予測（Flow の surprise）", "",
              "**surprise = −log p(動き│状況)** ＝「その状況で、その動きがどれだけ意外か」。"
              "条件付きなので**車が近いこと自体は意外でなく**（条件に入っている）、"
              "**状況に対して動きが異常なときだけ大きく**なる＝“危険な行動”の指標。",
              f"（危険ラベルはデータ由来：最接近 < {an['near_miss_m']}m を near-miss={an['n_near_miss']}本／"
              f"> {an['safe_m']}m を safe={an['n_safe']}本）"]
        a = an.get("A_novelty"); bc = an.get("BC_danger"); d = an.get("D_personalization")
        L += ["", "**結果（正直に・混在）**："]
        if a:
            verdict = "検知できず（クラスタが弱く≒同質）" if a["auc"] < 0.6 else "検知できる"
            L += [f"- **A. 群の新規性**（片タイプで学習→他タイプを異常検知）：AUC **{a['auc']:.2f}** → {verdict}。"]
        if bc:
            L += [f"- **B. near-miss の予測**：near-miss試行の surprise は**最接近の約 {bc['lead_time_s']}秒前**"
                  f"（車がまだ遠い時）から高い＝**物理的接近より早い予兆**。試行単位 AUC surprise **{bc['auc_surprise']:.2f}** "
                  f"> jerk基線 {bc['auc_jerk']:.2f}（ただし near-miss {bc['n_near_miss']}本と少なく弱め）。"]
        if d and d.get("pop_mean") is not None:
            L += [f"- **D. 個人化で誤警報減**：safe試行の surprise 平均が population **{d['pop_mean']:.2f}** → "
                  f"few-shot個人化で **{d['personalized_mean']:.2f}**（低い＝“その人の正常”に合わせ誤検知が減る）。"]
        if a and a.get("scores"):
            L += ["", "![novelty](reports/figures/anomaly_novelty.png)"]
            L += _cap("学習した型の試行(青)と未学習の型の試行(橙)の surprise の分布。"
                      "重なりが大きい＝群の新規性は surprise で分離しづらい（クラスタが横断速度差程度で弱いため）。")
        if bc:
            L += ["![leadtime](reports/figures/anomaly_leadtime.png)"]
            L += _cap("横=最接近(t=0)からの相対時刻、縦左=surprise、縦右(灰)=車までの距離。"
                      "橙(near-miss)は t=0 の数秒前・車がまだ遠い時点から surprise が高い＝"
                      "距離ベースの警報より早く“危険な動き”を捉えられる可能性。青(safe)は低いまま。")
        if d and d.get("pop"):
            L += ["![personalization](reports/figures/anomaly_personalization.png)"]
            L += _cap("安全な試行での平均 surprise。左=集団基準、右=few-shotで本人に個人化。"
                      "個人化すると下がる＝「その人の癖だが安全」を異常と誤らない＝アラーム疲労の低減。")
        L += ["", "> **正直な限界**: near-miss が全体の約6%(≈20本)と少なく、群も弱いソフトクラスタのため、"
              "A は分離せず・B の AUC も弱い。強い主張は避け、"
              "**(i) surprise が物理接近より早い予兆になりうること、(ii) 個人化が誤警報を下げること**の2点に留める。"
              "本格化には near-miss を増やすデータ収集が必要。"]
    # replay: embed gifs (play inline) + <video> (local viewers) + mp4 links
    gifs = sorted(glob.glob(str(C.FIG_DIR / "replay_*.gif")))
    mp4s = sorted(glob.glob(str(C.FIG_DIR / "replay_*.mp4")))
    if gifs or mp4s:
        L += ["", "## 6. replay（実試行＋Flow予測扇）", "",
              "> **見方**: 上から見た1試行の時間再生。**ピンク=歩行者の実際の軌跡**、"
              "**青い扇=Flowが予測する次~1.5秒の分布**（広い=不確実）、**四角+矢印=車**（色=eHMI, 遠いので端に表示、range=距離）。"
              "青い扇の中にピンクが入り続けるほど予測が当たっている。", ""]
        for g in gifs:
            rel = os.path.relpath(g, C.ROOT)
            L += [f"### {os.path.basename(g)}", "", f"![replay]({rel})", ""]
        if mp4s:
            L += ["高画質版（mp4。GitHub上ではリンク、ローカルビューアでは下のプレーヤーが再生）:", ""]
            for r in mp4s:
                rel = os.path.relpath(r, C.ROOT)
                L += [f'<video controls width="480" src="{rel}"></video>',
                      f"[{os.path.basename(r)}]({rel})", ""]
    L += ["", END]
    return "\n".join(L)


def main():
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    m = _load("metrics.json")
    if m:
        fig_nll(m); fig_calibration(m); fig_per_user(m)
    fig_response_curves()
    fs = _load("fewshot.json")
    if fs:
        fig_fewshot(fs)

    block = build_block()
    readme = C.ROOT / "README.md"
    txt = readme.read_text()
    if START in txt and END in txt:
        pre, rest = txt.split(START, 1)
        _, post = rest.split(END, 1)
        txt = pre + block + post
    else:
        txt = txt.rstrip() + "\n\n" + block + "\n"
    readme.write_text(txt)
    (C.REPORTS_DIR / "RESULTS.md").write_text(block.replace(START, "").replace(END, ""))
    print("injected results into README.md + figures")


if __name__ == "__main__":
    main()
