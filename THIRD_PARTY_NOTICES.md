# Third-Party Notices / 第三者ライセンス表示

dollama 本体は **Apache License 2.0** (リポジトリ同梱 `LICENSE`) で配布される。
本ファイルは dollama が利用・同梱・配置する第三者モデルおよびライブラリの帰属
(attribution) とライセンスを記載する。**許諾的ライセンス (Apache-2.0 / MIT) および
RAIL 系 (OpenRAIL / CreativeML OpenRAIL) はいずれも、ライセンス全文と帰属を添付すれば
商用利用を許諾する**。RAIL 系には付随する使用行動制限条項 (Use Restrictions) があるため、
その範囲内で利用する。

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
| **Illustrious XL v0.1 (early-release-v0)** (`OnomaAIResearch/Illustrious-xl-early-release-v0` rev `dca0dac3`) | OnomaAI Research | Fair AI Public License 1.0-SD (model card `license_name`。1.0+ は独自 Illustrious License で**未導入**) | 2-6d preset `illustrious-xl` = SDXL UNet+VAE+TE-L/G 差し替え (**導入済 2026-09-17・既定 preset**・推論専用・重み非再配布) | 確認済 (HF model card 2026-09-17 照合・`models/presets/illustrious-xl/preset.json`) |

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
  **Illustrious XL は版依存** (1.0+ は独自 Illustrious License) ゆえ導入したのは v0.1 のみ。
  Animagine XL 4.0 は Fair AI ではなく CreativeML Open RAIL++-M (上表・openrail 系は既決裁)。

## 更新方針

第三者モデル / ライブラリを追加・差し替えた際は本ファイルへ追記する。「要照合」項目は
リリース前に各 model card の LICENSE と最終照合し「確認済」へ更新する。
