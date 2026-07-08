

# 結果（自動生成 — `uv run python -m ehmi.report`）

各図の下に「見方」を付けています。

## 1. cue→反応は学習可能（20-fold LOUO）

予測対象は 0.1 秒ごとの歩行者の自己中心増分 (Δfwd, Δlat)。基線と held-out NLL で比較。

| model | held-out NLL ↓ | rollout ADE (m) ↓ |
|---|---|---|
| Gaussian基線 | +2.70 ± 0.34 | — |
| 等速度基線 | +1.74 ± 0.52 | 0.49 |
| MDN | -1.17 ± 0.63 | — |
| **Flow** | **-1.25 ± 0.63** | **0.47** |

> **NLL（負の対数尤度）**: モデルが実際の動きをどれだけ当てたかの指標。**低いほど良い**（負でOK）。3 nats 差 ≈ 約20倍の当てやすさ。

![nll](reports/figures/nll.png)

> **見方**: 棒＝各手法の held-out NLL（低いほど良い）。灰＝学習しない基線、青＝学習モデル。Flow/MDN が基線を大きく下回る＝「cue→反応」を確かに学習できている定量証拠。

![calibration](reports/figures/calibration.png)

> **見方**: 横=モデルが宣言した確率区間、縦=実際にその区間へ入った割合。対角線=理想（正直）。線が対角なら「50%と言えば本当に約50%入る」。ここでは対角より少し上＝区間がやや広め（過小自信）。

![response curves](reports/figures/response_curves.png)

> **見方**: 横=車までの距離(m)、縦=1ステップの平均移動量。左(前進)・右(横)。色=eHMI条件。車が近づく（左へ行く）ほど動きが変わる＝データに車キューへの反応が見える。

per-user の Flow NLL は **-2.40〜-0.11** と広い＝**個人差が支配的**（誰を隠すかで成績が決まる）。
![per user](reports/figures/per_user.png)

> **見方**: 棒1本＝「その人を隠して学習・評価」した1人分の成績。左=NLL、右=軌跡誤差ADE。棒の高さのばらつきの広さ＝個人差の大きさ。律速はモデルでなく“人の違い”。


## 2. few-shot 個人化（新しい人を数試行で較正）

**仕組み**: 学習済みモデルの重みは全て凍結し、**新しい人の 8 次元「個人ベクトル」だけ**をその人の k 試行で最適化（＝性格空間のどこに座るかを探す）。k=0 は未知(unknown)ベクトル。

(a) 埋め込みだけ適応（全20人 LOUO）：zero-shot(k=0) NLL **-1.25** → k=8本で **-1.57**（oracle＝その人を学習に含めた上限 -1.97）。**1〜2試行で大半が個人化**。

| k (適応試行) | 0 | 1 | 2 | 3 | 5 | 8 |
|---|---|---|---|---|---|---|
| (a) embedding NLL | -1.25 | -1.48 | -1.52 | -1.53 | -1.55 | -1.57 |
| (d) Reptile meta NLL | -1.17 | -0.15 | -0.21 | -0.16 | -0.72 | -0.80 |

![fewshot](reports/figures/fewshot_curve.png)

> **見方**: 横=新しい人の適応試行数 k、縦=held-out NLL（低いほど良い）。青(a)=埋め込みだけ適応は k=1〜2 で急改善し oracle(緑点線=上限)に接近＝少数で個人化できる。橙(d)=メタ学習+全微調整は k=1 で悪化＝1〜数試行での全体微調整は過学習（20人では埋め込み適応が正解）。


## 3. 行動フィンガープリント：クラスタ分類＋個人識別

各人の8次元「個人ベクトル」を行動の指紋とみなし、(i)少数タイプにクラスタ、(ii)尤度で人/型を当てる。

埋め込みを **K=3 タイプ**にクラスタ（silhouette 0.15＝弱め＝ソフトな型）。
**個人識別 top-1 = 75%**（偶然5%）＝反応で人を見分けられる。 **タイプ分類 = 80%**。

| type | n | 横断速度 | 横回避 | eHMI遵守 | 安全感 |
|---|---|---|---|---|---|
| 0 | 7 | 0.107 | 0.0058 | 0.17 | 4.74 |
| 1 | 8 | 0.123 | 0.0059 | 0.15 | 4.83 |
| 2 | 5 | 0.151 | 0.0054 | 0.13 | 4.61 |

![embedding map](reports/figures/embedding_map.png)

> **見方**: 各点＝1人の個人ベクトルをPCAで2Dに落としたもの。数字＝参加者ID、色＝クラスタ。近い人ほど反応の仕方が似ている。

![cluster profiles](reports/figures/cluster_profiles.png)

> **見方**: 各タイプの行動特徴（0–1に正規化）。線の高低の違いがタイプの性格。ここでは主に横断速度で分かれる（type0=遅い〜type2=速い）。

![identification](reports/figures/identification.png)

> **見方**: 行=本当のユーザ、列=候補の個人ベクトル、明るい=そのベクトルでの尤度が高い。**明るい対角線**＝自分のベクトルが自分の試行を最もよく説明＝指紋が成立している。


## 4. causal / 反事実（eHMI ON/OFF）

eHMI は実験で割り付けた変数なので `do(eHMI)` を扱える。**反事実**＝可逆な Flow で「観測した反応→潜在 z を逆算→eHMI を OFF に差し替えて z 固定で再生成」＝**同じ人・同じ気質のまま、もし警告が無かったら**の動きを生成。

ON 110 / OFF 285 試行。反事実効果はタイプで異なる（効果の異質性）：

| type | Δ peak-lateral: ON − do(OFF) (m) | n |
|---|---|---|
| 0 | -0.134 m | 35 |
| 1 | -0.097 m | 48 |
| 2 | -0.070 m | 27 |

**解釈**: 値＝(実際ONの横回避のピーク) −(反事実OFFの横回避のピーク)。**負＝eHMIがあると横回避が小さい**（車の進路が分かり、無駄に大きく避けなくて済む）。type0(遅い横断)で最も大きく、type2(速い)で小さい＝eHMIは慎重な人ほど効く。

> 注: eHMI ON/OFF only (L/R entangled w/ car maneuver); counterfactual is model-dependent (SCM assumption), not point-validatable.

![cate](reports/figures/cate_by_cluster.png)

> **見方**: 左=観測データの ON(青) vs OFF(灰) の横回避ピーク（タイプ別）。右=モデル反事実の効果 ON−do(OFF)。棒が負＝eHMIで横回避が減る。棒の高さがタイプで違う＝効果の異質性。

![counterfactual](reports/figures/counterfactual_example.png)

> **見方**: ある1人・1試行の上から見た経路。ピンク実線=実際(eHMI ON)、青破線=反事実(同じ人がeHMI無しなら)。青が右に多く彷徨う＝警告が無ければもっと横に避けていた。横軸(lateral)は効果を見せるため引き伸ばし。


## 5. replay（実試行＋Flow予測扇）

> **見方**: 上から見た1試行の時間再生。**ピンク=歩行者の実際の軌跡**、**青い扇=Flowが予測する次~1.5秒の分布**（広い=不確実）、**四角+矢印=車**（色=eHMI, 遠いので端に表示、range=距離）。青い扇の中にピンクが入り続けるほど予測が当たっている。

### replay_s01_s1_2019_11_26_14_44_25.gif

![replay](reports/figures/replay_s01_s1_2019_11_26_14_44_25.gif)

### replay_s01_s2_2019_11_26_14_40_14.gif

![replay](reports/figures/replay_s01_s2_2019_11_26_14_40_14.gif)

### replay_s19_s2_2019_12_04_15_56_20.gif

![replay](reports/figures/replay_s19_s2_2019_12_04_15_56_20.gif)

### replay_s19_s2_2019_12_04_15_57_40.gif

![replay](reports/figures/replay_s19_s2_2019_12_04_15_57_40.gif)

### replay_s19_s2_2019_12_04_16_07_53.gif

![replay](reports/figures/replay_s19_s2_2019_12_04_16_07_53.gif)

### replay_s20_s2_2019_12_04_17_21_38.gif

![replay](reports/figures/replay_s20_s2_2019_12_04_17_21_38.gif)

高画質版（mp4。GitHub上ではリンク、ローカルビューアでは下のプレーヤーが再生）:

<video controls width="480" src="reports/figures/replay_s01_s1_2019_11_26_14_44_25.mp4"></video>
[replay_s01_s1_2019_11_26_14_44_25.mp4](reports/figures/replay_s01_s1_2019_11_26_14_44_25.mp4)

<video controls width="480" src="reports/figures/replay_s01_s2_2019_11_26_14_40_14.mp4"></video>
[replay_s01_s2_2019_11_26_14_40_14.mp4](reports/figures/replay_s01_s2_2019_11_26_14_40_14.mp4)

<video controls width="480" src="reports/figures/replay_s19_s2_2019_12_04_15_56_20.mp4"></video>
[replay_s19_s2_2019_12_04_15_56_20.mp4](reports/figures/replay_s19_s2_2019_12_04_15_56_20.mp4)

<video controls width="480" src="reports/figures/replay_s19_s2_2019_12_04_15_57_40.mp4"></video>
[replay_s19_s2_2019_12_04_15_57_40.mp4](reports/figures/replay_s19_s2_2019_12_04_15_57_40.mp4)

<video controls width="480" src="reports/figures/replay_s19_s2_2019_12_04_16_07_53.mp4"></video>
[replay_s19_s2_2019_12_04_16_07_53.mp4](reports/figures/replay_s19_s2_2019_12_04_16_07_53.mp4)

<video controls width="480" src="reports/figures/replay_s20_s2_2019_12_04_17_21_38.mp4"></video>
[replay_s20_s2_2019_12_04_17_21_38.mp4](reports/figures/replay_s20_s2_2019_12_04_17_21_38.mp4)


