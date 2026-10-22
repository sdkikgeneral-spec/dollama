# Third-Party Notices / 第三者ライセンス表示

dollama 本体は **Apache License 2.0** (リポジトリ同梱 `LICENSE`) で配布される。
本ファイルは dollama が利用・同梱・配置する第三者モデルおよびライブラリの帰属
(attribution) とライセンスを記載する。**許諾的ライセンス (Apache-2.0 / MIT) および
RAIL 系 (OpenRAIL / CreativeML OpenRAIL) はいずれも、ライセンス全文と帰属を添付すれば
商用利用を許諾する**。RAIL 系には付随する使用行動制限条項 (Use Restrictions) があるため、
その範囲内で利用する。**例外として §2 には非商用ライセンスのモデル (Anima) を含む** (個人研究での研究利用のみ・
下記「非商用ライセンスについて」)。

> **注記**: 下表のうち、リポジトリ内ドキュメントで明示確認できたものは「確認済」、
> 一般に公開されている model card の記載に拠るものは「要照合」とした。**配布
> (リリース) 前に各 model card / リポジトリの LICENSE と最終照合すること。**

---

## 1. 同梱・配置される第三者モデル (shipped / deployed weights)

これらは生成・後処理パイプラインで実際にロードされ、出力や派生物 (蒸留した自作
ScorerNet 等) に寄与する。

| モデル | 著者 / 配布元 | ライセンス | 用途 (dollama 内) | 確認 |
|---|---|---|---|---|
| **Stable Diffusion XL base 1.0** | Stability AI | CreativeML Open RAIL++-M | SDXL UNet + VAE (拡散生成本体) | 要照合 |
| **CLIP ViT-L/14** | OpenAI | MIT | text encoder (NPU) / 美的スコアラの image embed | 要照合 |
| **WD14 SwinV2 Tagger** | SmilingWolf | Apache-2.0 | danbooru タグ抽出 (CPU)・解剖 8 軸ラベル教師 | 要照合 |
| **ISNet-anime (anime-segmentation)** | SkyTNT (skytnt) | Apache-2.0 | マッティング (α 抽出・透過 PNG) | 確認済 (docs) |
| **waifu-scorer-v4-beta** | Eugeoter | Apache-2.0 | 美的品質スコアラ教師 (Model B quality head・primary) | 確認済 (重み README) |
| **anime_aesthetic** | deepghs | OpenRAIL | 美的品質スコアラ教師 (評価/アンサンブル候補・使用制限条項あり) | 確認済 (docs) |
| **NoobAI-XL 1.1** (`Laxhar/noobai-XL-1.1` rev `814a274a`) | Laxhar Lab | Fair AI Public License 1.0-SD (model card `license_name`) | 2-6d preset `noobai-xl` = SDXL UNet+VAE+TE-L/G 差し替え (**導入済 2026-09-17**・推論専用・重み非再配布) | 確認済 (HF model card 2026-09-17 照合・`models/presets/noobai-xl/preset.json`) |
| **Animagine XL 4.0** (`cagliostrolab/animagine-xl-4.0` rev `2b7c1b39`) | Cagliostro Research Lab | **CreativeML Open RAIL++-M** (HF tag `license:openrail++`。旧記載「Fair AI」は誤りで訂正) | 2-6d preset `animagine-xl-4` = SDXL UNet+VAE+TE-L/G 差し替え (**導入済 2026-09-17**・推論専用・重み非再配布) | 確認済 (HF API/model card 2026-09-17 照合・`models/presets/animagine-xl-4/preset.json`) |
| **Illustrious XL v0.1 (early-release-v0)** (`OnomaAIResearch/Illustrious-xl-early-release-v0` rev `dca0dac3`) | OnomaAI Research | Fair AI Public License 1.0-SD (model card `license_name`。**ライセンスは版ごとに異なる**: v1.0/v1.1 = `sdxl-license` (SDXL LICENSE.md へのリンク) / v2.0 = CreativeML Open RAIL-M・2026-10-22 HF 照合。旧記載「1.0+ は独自 Illustrious License」は一次資料の裏付けが無く訂正) | 2-6d preset `illustrious-xl` = SDXL UNet+VAE+TE-L/G 差し替え (**導入済 2026-09-17・既定 preset**・推論専用・重み非再配布) | 確認済 (HF model card 2026-09-17 照合・`models/presets/illustrious-xl/preset.json`) |
| **Illustrious XL v2.0** (`OnomaAIResearch/Illustrious-XL-v2.0` rev `69459c1f`) | OnomaAI Research | **CreativeML Open RAIL-M** (HF tag `license:creativeml-openrail-m`・repo に LICENSE ファイルなし) | E-4 で preset `illustrious-xl-v2` = SDXL UNet+VAE+TE-L/G 差し替えに変換済 (`scripts/dollma_export_sdxl_preset.py`・研究機ローカル)。**ライセンス採用可 (2026-10-22 ユーザー決裁)**。**既定 preset ではない** (既定は `illustrious-xl` = v0.1 のまま)・推論専用・重み非再配布 | 確認済 (HF API/model card 2026-10-22 照合) |

**Illustrious XL v0.1 の利用規約**: v0.1 repo には model card のライセンスとは別に `TERM_OF_USE` ファイルがある
(2026-10-22 取得)。Onoma AI に対し知財侵害の訴訟等を起こした場合のライセンス終了 (Disputes)・
同規約違反の利用/配布に起因する第三者請求の補償 (indemnify)・商標ライセンスの不許諾 (Trademark License) を定める。
v2.0 repo には同種のファイルは無い。

**ScorerNet (自作 11.18M) について**: 上記 waifu-scorer-v4 / deepghs の採点を soft target
として蒸留した派生物。蒸留教師となった美的モデルのライセンス (Apache-2.0 / OpenRAIL) を
本ファイルで帰属表示する。OpenRAIL の Use Restrictions の範囲内で利用する。

## 2. オフライン教師 / probe のみ (配布物に同梱しない)

訓練データ収集・蒸留教師・性能比較に用いるのみで、dollama の配布バイナリ・重みには
同梱しない。

| モデル | 著者 / 配布元 | ライセンス | 用途 | 確認 |
|---|---|---|---|---|
| **TIPO-200M** | KBlueLeaf | Apache-2.0 | D6 外部教師 (蒸留・**不採用**) | 確認済 (docs) |
| **Qwen2-1.5B** | Alibaba (Qwen) | Qwen / Apache-2.0 系 | プロンプト生成 probe (Python のみ) | 要照合 |
| **DanTagGen (400M LLaMA)** | KBlueLeaf | (model card 参照) | 先行実装の品質基準・参照 | 要照合 |
| **Anima** (`circlestone-labs/Anima` rev `f973fc41`・aesthetic-v1.1 他) | CircleStone Labs / Comfy Org | **CircleStone Labs Non-Commercial License v1.2** (モデル本体と派生物は**非商用限定**・Outputs (生成画像) は商用可と明記) + **NVIDIA Open Model License** (Cosmos-Predict2-2B-Text2Image の Derivative Model) | **研究トラックとして採用 (2026-10-22 ユーザー決裁・個人研究)**: E-4 probe (`scripts/dollma_probe_anima.py`) で試走済 (判定なし)・自作 C++/CUDA 実装を予定。出荷既定ではない・重み非再配布。**A5 (backend 結線) 時に §1 へ移し、NVIDIA OML の "Built on NVIDIA Cosmos" 表示要否を判定する**。HTTP 結線後に第三者へ提供する構成は CircleStone §1.c の除外 (b) に当たり得る (下記「非商用ライセンスについて」) | 確認済 (HF LICENSE.md 2026-10-22 照合・研究機コピー `models/anima/LICENSE.md` と SHA256 一致 = 2026-10-22 研究機で `Get-FileHash` 再取得) |
| **Strata** v0.2.2 (`Niko1221/Strata`) | Niko1221 and the Strata contributors | MIT (同梱物の別ライセンス: `third_party/ggml` = MIT / web フォント = SIL OFL 1.1 / `data/experimental-speed-projection` のベクトル = **Qwen Community License 1.0**) | **ライセンス採用可 (2026-10-22 ユーザー決裁)**。用途は Phase 1 probe: 外部プロセスの LLM サーバとして `scripts/dollma_probe_strata.py` (127.0.0.1 の HTTP クライアント) からのみ使用・**dollama 本体に同梱しない**。**本線化は未決裁** | 確認済 (GitHub LICENSE / docs/HOW_IT_WORKS.md 2026-10-22 照合) |
| **Qwen3.6-35B-A3B** MTP 版 GGUF UD-Q4_K_XL (`unsloth/Qwen3.6-35B-A3B-MTP-GGUF` rev `5bc3e238` / 原 `Qwen/Qwen3.6-35B-A3B`) | Alibaba (Qwen) / 量子化 Unsloth | Apache-2.0 | **ライセンス採用可 (2026-10-22 ユーザー決裁)**。用途は Phase 1 probe (Strata 経由・研究機ローカル)・重み非再配布。**本線化は未決裁** | 確認済 (GGUF repo は HF API 2026-10-22 取得で rev `5bc3e238`・`license: apache-2.0`・license_link は原 repo。LICENSE 本文は GGUF repo に無く、原 repo `Qwen/Qwen3.6-35B-A3B` の `LICENSE` (Apache 2.0) で照合) |

## 3. ヘッダオンリーライブラリ (subprojects)

| ライブラリ | 著者 | ライセンス | 用途 | 確認 |
|---|---|---|---|---|
| **cpp-httplib** | yhirose | MIT | OpenAI 互換 HTTP サーバ | 要照合 (subprojects/) |
| **nlohmann/json** | Niels Lohmann | MIT | JSON 入出力 | 要照合 (subprojects/) |

---

## 使用制限条項 (RAIL 系) について

- **CreativeML Open RAIL++-M** (SDXL) / **OpenRAIL** (deepghs/anime_aesthetic) は、
  ライセンス全文と帰属の添付を条件に商用利用を許諾する。付随する Use Restrictions
  (有害・違法・差別的用途等の禁止) の範囲内で利用すること。dollama の用途
  (2D キャラクターイラスト生成・美的品質採点) はこれらの制限に該当しない。
- 各 RAIL ライセンス全文は配布時に同梱する (または該当 model card への参照を明示する)。
- **Fair AI Public License 1.0-SD** (NoobAI-XL 1.1 / Illustrious XL v0.1): 帰属を条件に商用可。
  コピーレフト条項 (改変/マージ版を**公開配布**する場合に同ライセンス+重み公開) は、dollama が
  checkpoint を**自ホスト参照するのみで重みを再配布しない**運用では非トリガー → 帰属表示で足りる。
  **2026-09-17 ユーザー決裁「採用可」** (推論専用・重み非再配布・本ファイルへの記載を条件)。
  **Illustrious XL はライセンスが版ごとに異なる** (v0.1 = Fair AI 1.0-SD / v1.0・v1.1 = sdxl-license /
  v2.0 = CreativeML Open RAIL-M・2026-10-22 HF model card 照合)。既定 preset として導入したのは v0.1 のみ。
  v2.0 はライセンス採用可 (2026-10-22 ユーザー決裁・§1 表・既定ではない)。旧記載「1.0+ は独自 Illustrious License」は
  2026-07-02 の暫定追記 (`8e77e91`・当時「要照合」) が出自で、`883152e` (2026-09-18) で v0.1 行が「確認済」に
  なった際に括弧書きだけ未照合のまま残った。一次資料の裏付けが無かったため訂正。
  Animagine XL 4.0 は Fair AI ではなく CreativeML Open RAIL++-M (上表・openrail 系は既決裁)。
- **CreativeML Open RAIL-M** (Illustrious XL v2.0): 本文に非商用限定の条項は無い。Attachment A の
  Use-based restrictions あり。本文 §6 で Licensor は Output への権利を主張しない (ただし Output の利用も
  License の条項に反してはならない)。参照した本文は CompVis 版で、v2.0 repo には本文のコピーが無い (下記「未確認事項」)。

## 非商用ライセンスについて (Anima)

- **CircleStone Labs Non-Commercial License v1.2**: モデル本体と派生物の利用・配布・派生物作成は
  Non-Commercial Purpose (§1.c) に限る (§2.b)。§1.c の Non-Commercial Purpose は、利用から直接・間接の
  支払いを受けない限りにおいて、(i) 公共の知識のための研究・実験・試験、個人の学習・娯楽・趣味など
  商業活動・事業運営・雇用上の職務に直接・間接に関係しない個人利用、(ii) 営利法人の内部での
  試験・評価・非商用研究開発 (非本番環境)、(iii) 慈善団体の利用。§1.c 末尾で
  (a) 収益を生む活動、(b) **第三者エンドユーザーと直接対話する、または影響を与える利用**、
  (c) 商用目的で他モデルを訓練・fine-tune・蒸留する利用は Non-Commercial Purpose に含まれないと明記。
  dollama は個人研究としてこの範囲で使う。**A5 で HTTP 結線した後に第三者へ提供する構成は除外 (b) に当たり得る**。**Outputs (生成画像) は §2.e で商用利用可**と
  明記 (model card も "You may use generated images commercially.")。再配布する場合は §3.b の Attribution
  Notice が必要 (dollama は重みを再配布しない)。原文 §4.a は Outputs の項を "2(d)" と参照しているが、
  取得版で Outputs を定めるのは §2.e (原文の参照ずれ・原文どおり記録)。
- **NVIDIA Open Model License** (Anima は Cosmos-Predict2-2B-Text2Image の Derivative Model): NVIDIA は出力の
  所有権を主張しない。Cosmos Model を使う product or service を distribute or make available する場合等は
  **"Built on NVIDIA Cosmos"** を関連 website / UI / about page / product documentation に表示する義務がある。
  再配布時は "Licensed by NVIDIA Corporation under the NVIDIA Open Model License" の Notice が必要。
  ガードレール回避で権利が自動終了する条項あり。照合したのは 2025-10-24 版。
- dollama での位置づけ: 研究トラックとして採用 (2026-10-22 ユーザー決裁・個人研究)。**出荷経路の既定には組み込まない**
  (既定切替は別決裁)。A5 (backend 結線) 時に本ファイル §1 へ移し、"Built on NVIDIA Cosmos" 表示の要否を判定する。

## 未確認事項 (2026-10-22 照合時点・未取得)

- illustrious-xl.ai 公式サイト側の利用規約 (未取得)。
- CreativeML Open RAIL-M 本文: v2.0 repo にコピーが無いため CompVis 版を参照。OnomaAI が意図する本文との同一性は未確認。
- Cosmos-Predict2-2B-Text2Image の README 本文 (gated・未取得)。同モデル公開時点の NVIDIA Open Model License の版と照合版 (2025-10-24) の異同は未確認。
- Anima 実行に使う Qwen3 0.6B text encoder / Qwen-Image VAE / `hdae/diffusers-anima` のライセンス (未取得)。
- Strata Windows exe への Qwen Community License 1.0 ベクトルの同梱有無 (未確認)。
- Apache-2.0 本文の帰属節 (§4) の原文照合 (冒頭のみ確認・未引用)。

## 更新方針

第三者モデル / ライブラリを追加・差し替えた際は本ファイルへ追記する。「要照合」項目は
リリース前に各 model card の LICENSE と最終照合し「確認済」へ更新する。
