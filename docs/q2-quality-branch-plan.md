# Q-2 → quality 枝分離プラン (分割実行・複数セッション/複数マシン)

> Phase 4 施策 F の quality head 有効化。当初の「ScorerNet 生ピクセル quality head 蒸留」は
> **構造的に失敗** (下記) → **quality を CLIP-embed 枝に分離** する設計へ pivot。
> 本 doc は**セッションをまたいで拾える分割タスク台帳**。各 Package は担当機・依存・入口/出口が
> 自己完結。次セッションはこの doc の「現在地」を見て次の未完 Package から着手する。

## 経緯 (確定した知見)

- **Q-1** (完了): waifu_scorer_v4 (apache-2.0・CLIP ViT-L image + MLP) で 180 枚実採点。
  `data/scorer/scorer.{train,val}.jsonl` 各行に `raw_waifu` / `quality_waifu` / `quality` 保持。
  生 `raw_waifu` は分散良好 (std0.81 / range[-1.71,1.78] / mean0.74)。
- **Q-2 Step①** (完了): `quality` を `raw/10 clamp` (→[0,0.18] 潰れ) から
  **z-sigmoid renorm** `q=1/(1+exp(-((raw-mean)/std)*k))` k=1.5 に置換。
  結果 std 0.0753→**0.298**・0クランプ消滅・二峰。実装 `scripts/dollma_renorm_quality.py`
  + test `scripts/tests/test_dollma_renorm_quality.py` (6 green)。退避 `*.q018.bak.jsonl`。
  provenance `data/scorer/scorer_quality_renorm.json`。
- **Q-2 Step②** (実施したが**不採用**): `train_scorer.py --freeze-b off` で ScorerNet の
  生ピクセル quality head (index0) を再訓練。ゲート3項 (quality_loss 非自明 / 予測非潰れ /
  anatomy 非退行 val_axis 0.004) は通過したが、**val corr(pred, teacher) が全設定で負**
  (6ep −0.25 / 12ep −0.28 / 20ep −0.30 / q_weight↑ で −0.33)。**追い込むほど悪化**。
  → **生ピクセル ResNet は 162 枚から CLIP 空間の美的品質を蒸留できない** (空間ミスマッチ)。
  負相関 head を reward に混ぜると LM を逆方向に押すため**出荷保留**。
  ※現行 `data/scorer/scorer_net.safetensors` はこの負相関 6ep 版のまま (Package A で anatomy 専用へ戻す)。

## 新設計 (pivot 後・確定)

**quality を教師 (waifu) と同じ CLIP 空間の独立枝に分離する。**

```
生成 PNG ──► NPU: CLIP ViT-L image encoder ──► embed[768] ──► 自作 quality MLP (waifu 蒸留) ──► raw
                                                                                    │
                                          ScorerNet (anatomy 8軸・生ピクセル conv・現状維持) │
                                                        │                                    ▼
                                                     axes[8] ───────────► reward_from_scorer(axes, quality)
                                                                          quality = z-sigmoid renorm(raw)
```

- **ScorerNet は anatomy 専用に戻す** (8軸・生ピクセルで正しく学べている・val_axis 0.004)。index0 quality head は出荷しない/凍結。
- **quality = 自作小型 MLP on CLIP image embed[768]**、waifu から蒸留。入力が既に CLIP 空間なので蒸留は素直に通る (生ピクセルの逆相関問題が消える)。「自作モデルで HW を叩く」方針とも合致 (waifu 直乗せでなく自作 MLP へ蒸留)。
- **HW: CLIP image encoder を NPU に追加** (現行 CLIP-L は text・image は別モデル)。生成後の遊休窓で採点 = dollama の芯「拡散の裏で遊休 HW を使い切る」。
- reward ループ (F) とデプロイ inline 採点の**両方**がこの枝で立つ。案2 (生ピクセル蒸留を数千枚で救う) は空間ミスマッチが根本原因の公算が高く**非採用**。

## 分割タスク台帳

> 各 Package は独立セッションで着手可。入口条件を満たしていれば前段の文脈再導出は不要。
> 完了したら本 doc の該当行 status と「現在地」を更新すること。

| Pkg | 内容 | 担当機 | 依存 | status |
|---|---|---|---|---|
| **A** | ScorerNet を anatomy 専用へ復帰 + golden 再生成 + meson 緑 + pivot 記録 | 本機 | なし | ✅ 完了 (2026-07-04) |
| **B** | CLIP image embed[768] harvest (180枚 + 将来 rollout)。`clip_image_embed` を jsonl に載せる | 本機=研究機相当 (open_clip) | なし (A と並列可) | ✅ 完了 (2026-07-04) |
| **C** | 自作 quality MLP を CLIP embed 上で waifu 蒸留・**val corr が正**を確認・safetensors 出力 | 本機 CPU | B | ✅ 完了 (2026-07-04) |
| **D** | NPU: CLIP image encoder OV 変換 + latency probe / 自作 quality MLP OV 変換 | 本機=研究機相当 (OV/NPU) | C | ✅ 完了 (2026-07-04) |
| **E** | reward 結線 (collect_rollouts で CLIP image→MLP→renorm→quality) + test / F-0a smoke 再走・std/分離を比較 | 本機 (結線) + smoke | C, D | ✅ 完了 (2026-07-04) |

> **A 完了メモ (2026-07-04)**: anatomy 専用 ScorerNet 出荷 (freeze_b=True・val_axis 0.00495)・
> golden 再生成 (PyTorch↔OV 1.335e-5 PASS・NPU compile 9.33ms)・meson Ok:46/Fail:0。
> **この環境で OV 2026.2 / NPU 可視 / meson exe 実走緑を確認** = 本機が研究機相当。
> B/D も同環境で実行可 (別マシン待ち不要)。追い込みスイープ知見: 生ピクセル蒸留は
> 不安定 (ep20 qw2.0 corr +0.40 / ep12 崩壊) で出荷基準外 → CLIP-embed 枝 pivot を数値支持。

### Package A — ScorerNet anatomy 専用復帰 (本機・小)
- 入口: なし (起点)。
- 作業: `train_scorer.py --freeze-b on` (or auto) で anatomy 専用 ScorerNet を再出力し
  `data/scorer/scorer_net{,_fp32}.safetensors` を差し替え (負相関 6ep 版を捨てる)。
  重み変更で `src/tests/data/scorer/golden_scorernet_meta.json` が破れるので
  `dollma_convert_scorer.py` 系で golden 再生成・corr 突合。`meson test -C build` 緑復帰
  (test_quality_scorer / test_scorer_runner / test_scoring_postprocess)。
  ※SAC で新規 exe 実走不可なら ビルド緑 + Python OV corr で代替。
- 出口: pipeline に anatomy 専用 ScorerNet のみが乗る (壊れた quality head を出荷しない)。meson 緑。

### Package B — CLIP image embed harvest (研究機・中)
- 入口: なし (A と並列可)。
- 作業: 研究機で open_clip `ViT-L-14-quickgelu` pretrained=openai で
  `data/scorer/img/*.png` (180) の image embed[768] を採り (L2 正規化・waifu 忠実)、
  `scorer.{train,val}.jsonl` の各行に `clip_image_embed:[768]` を載せる (schema 既存フィールド)。
  `dollma_score_quality_v4.py` の `score_image` が既に CLIP image encode をしているので流用。
- 出口: jsonl 各行が `clip_image_embed[768]` を持つ。将来 rollout でも同経路で採れる。

### Package C — 自作 quality MLP 蒸留 (本機 CPU・小)
- 入口: B 完了 (`clip_image_embed` 済み jsonl)。
- 作業: 自作小型 MLP (768→…→1) を定義し、入力 `clip_image_embed`・教師 `quality` (renorm 済み)
  で蒸留 (Huber or MSE)。**val corr(pred, teacher) が明確に正** (目標 +0.5 以上・同空間なので通るはず)
  を確認。safetensors 出力 (自作・再現的 seed 固定)。test 追加。
- 出口: 正相関の自作 quality MLP + safetensors + test 緑。**もし corr が正に立たなければ**
  蒸留 loss/容量/正規化を見直す (raw を教師にする等)。
- 補足: waifu MLP 構造 (768→2048→512→256→128→32→1) をそのまま自作蒸留してもよいし、
  より小型でも可 (CLIP 空間の線形性が高ければ小 MLP で足りる — 実験軸)。

### Package D — NPU 化 (研究機・中)
- 入口: C 完了 (自作 quality MLP safetensors)。
- 作業: CLIP ViT-L image encoder を OV 変換 (静的形状・NPU)・latency probe (CLIP text 7.85ms 相当か)。
  自作 quality MLP も OV 変換 (PyTorch↔OV err <1e-4 目安・B-3c 同様)。
- 出口: NPU image-encode 実測 + IR 一式。CLIP text/image の NPU 同居メモリ収支も確認。

### Package E — reward 結線 + smoke (本機結線 + 研究機 smoke)
- 入口: C (+D は smoke の NPU 経路に必要・torch で回すなら C だけでも smoke 可)。
- 作業: `collect_rollouts.py` で生成 PNG → CLIP image embed → 自作 quality MLP → z-sigmoid renorm
  (Step① 写像再利用) → `quality` → `reward_from_scorer(axes, quality)` (QUALITY_WEIGHT=0.3 経路)。
  `QUALITY_ENABLED` を quality 供給元付きで有効化。`test_dollma_reward_rollouts` を quality 経路まで拡張。
  研究機で F-0a smoke 再走 → reward std / best−worst / clean-clutter 分離を F-0a ベースライン
  (std0.038 / best−worst0.203 / |r| 0.007 vs 0.0285) と比較。目標 std>0.1 or 分離維持/改善。
- 出口: reward に生きた直交 quality 軸。smoke 数値。→ 合否で F-0b SFT へ。
- 副産物: rollout の (image, raw_waifu, axes) は蓄積すれば将来の追加蒸留コーパスにもなる。

## 現在地 (最終更新: 2026-07-04)

- ✅ Q-1 / Step① renorm / Step② (負相関で不採用と確定・pivot 決裁済み)
- ✅ **Package A 完了** (anatomy 専用 ScorerNet 出荷・golden 再生成・meson Ok:46/Fail:0)
- ✅ **Package B 完了** (180/180 行に clip_image_embed[768]・L2 norm 1.0・test 3 緑)
- ✅ **Package C 完了** (自作 QualityMLP 768→64→1・**OOF 5-fold corr +0.5253**・容量非依存で安定正・test 6 緑)
  - 出力 `data/scorer/quality_mlp{,_fp32}.safetensors` / `quality_mlp_stats.json`。天井=データ量 (162)・容量増無効・教師枚数増が最大レバー。
- ✅ **Package D 完了** (出荷経路が全て NPU で疎通)
  - D-1 QualityMLP OV: err 1.788e-7 / NPU 0.553ms。`models/quality-mlp/*`。
  - D-2 CLIP ViT-L image encoder OV: MHA fastpath 無効化で変換成功・**embed corr 1.000000** (L2 後 6.1e-7=蒸留 embed 空間と完全一致)・**NPU 85.55ms 全デバイス最速→採用**・304M image tower が NPU に載る (text 7.85ms の11倍で妥当)。`models/clip-image/*` (gitignore)。
  - 出荷推論: 生成 PNG → CLIP image(NPU 85.55ms) → QualityMLP(NPU 0.55ms) → sigmoid → Step① renorm。
- ✅ **Package E 完了** = Q-2 効果確定。quality 供給元を CLIP image(OV)→QualityMLP(OV)→sigmoid に結線
  (`dollma_collect_rollouts.py`・test 17 緑)。**同一80画像+保存 axes で quality だけ足す純分離実験** (`dollma_e2_quality_signal.py`):
  reward std **0.0377→0.0797 (2.1倍)** / best−worst **0.2031→0.3178 (>0.3)** / quality 分布 std 0.2575・全域非崩壊 /
  **corr(quality, anatomy worst) = 0.0755 ≈ 直交**。quality が reward 分散の支配成分に。
  - **QUALITY_WEIGHT 0.3→0.4 決着 (ユーザー決裁・2026-07-04)**: `dollma_reward.py` を 0.4 に変更 (test 定数参照化・17緑)。
    weight 0.4 で E-2 再計算 → **reward std 0.0797→0.1038 (>0.1 PL 一次ゲート達成)** / best−worst 0.3178→0.3575。
    根拠: anatomy 7/8死 (Limbs のみ) に対し quality は OOF +0.53 の信頼軸。**両 PL 閾値 (std>0.1 ∧ best−worst>0.3) をクリア**。

## 🎉 Q-2 quality 枝分離 — 全 Package 完了 (2026-07-04)

生ピクセル quality head 蒸留の構造的失敗 (val corr 負・不安定) を、**quality を教師と同じ CLIP 空間の
独立枝に分離** して解決。ScorerNet=anatomy 専用 / quality=自作 QualityMLP on CLIP image embed (waifu 蒸留・
OOF corr +0.53) / CLIP image encoder は NPU (85.55ms)。reward に anatomy 直交の quality 軸が入り信号 2.1倍。
**両 PL 閾値 (reward std 0.1038>0.1 ∧ best−worst 0.3575>0.3) をクリアし信号ゲート通過。**
次フェーズ = **F-0b (rejection-sampling SFT)**。伸ばすなら教師枚数増 (C で天井=データ量162 と確定・rollout 蓄積が最大レバー)。

## 参照
- reward: `scripts/dollma_reward.py` (QUALITY_WEIGHT=0.3 経路実装済み) / rollout: `scripts/dollma_collect_rollouts.py` (`QUALITY_ENABLED`)
- 採点: `scripts/dollma_score_quality_v4.py` (CLIP image + waifu MLP 実走・研究機) / ラベル: `scripts/dollma_make_scorer_labels.py`
- 訓練: `scripts/train_scorer.py` / 変換: `scripts/dollma_convert_scorer.py`
- ScorerNet: `src/infer/quality_scorer.hpp` / golden: `src/tests/data/scorer/golden_scorernet_meta.json`
- ライセンス: waifu=apache-2.0 (採用可・[[project_openrail_aesthetic_approved]])。deepghs(openrail)は保留。
- F 全体: CLAUDE.md 計測表 Phase4 F 行 / `docs/measurements-log.md` / [[project_phase4_F_status]]
