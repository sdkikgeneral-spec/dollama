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

## 更新方針

第三者モデル / ライブラリを追加・差し替えた際は本ファイルへ追記する。「要照合」項目は
リリース前に各 model card の LICENSE と最終照合し「確認済」へ更新する。
