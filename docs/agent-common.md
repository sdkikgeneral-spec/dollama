# サブエージェント共通ルール (非交渉)

`.claude/agents/*.md` の全エージェントが従う共通規約。**各定義はこのファイルを参照するだけで、
中身を複製しない。** 同じ規約を 10 箇所にコピーすると 10 箇所とも腐るため、保守点はここ 1 つに集約する。

各エージェント固有の知識 (担当ファイル・デバイス実測値・落とし穴) は各定義側に置く。

---

## 1. 走る機械の判定

このプロジェクトは **2 台**で回している。振られたタスクがどちらの機械を要求するかを最初に確かめる。

| | 開発機 | 研究機 |
|---|---|---|
| CPU | Intel Core i7-10700 | Intel Core Ultra 9 285 (NPU = AI Boost 搭載) |
| GPU | GTX 1080 Ti (sm_61・FP16 native 非対応) | RTX 5080 (Blackwell / sm_120 / VRAM 16GB) |
| NPU / iGPU | **無し** | NPU + Intel Xe iGPU (OpenVINO の `GPU.0`) |
| CUDA Toolkit | **nvcc 無し** | CUDA 12.8+ |
| meson 設定 | `with_cuda=false, with_openvino=false` | `with_cuda=true, with_openvino=true` |

判定手段 (作業前に確認する):

```powershell
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
nvcc --version
Get-Content build/meson-info/intro-buildoptions.json   # with_cuda / with_openvino
```

**研究機必須のタスクを開発機で振られたら、スクリプト著述・純ホストのビルド確認までで止めて
「実走は研究機」と報告する。** 勝手に別の HW で代替して回さない (数値の意味が変わるため)。
逆に開発機で完結する作業 (データ構築・小型 LM 訓練・純ホスト C++・UI) を研究機の空き待ちにしない。

## 2. コーディング規約

- **Allman スタイル**: 開き波括弧 `{` は必ず改行して次の行に置く (C++ / C# 共通)。
- `switch` の `case` ラベルは `switch` と同じインデント位置に揃える。
- **コメントは日本語**で書く (C++ / C# / Python 共通)。
- ファイル名プレフィックス `dollma_` は `scripts/` 配下のみ。`src/` 以下は自由。
- 例外を投げるのは CUDA / OpenVINO のエラー時のみ (`std::runtime_error`)。

## 3. テスト必須 (CLAUDE.md ルール4)

コンポーネントを実装したら**必ずテストも実装する**。緑を確認するまで完了と報告しない。

| 対象 | 置き場 | 実行 |
|---|---|---|
| C++ | `src/tests/test_<component>.cpp` | `meson test -C build` |
| C# | `ui.Tests/` (xUnit) | `dotnet test ui.Tests` |
| Python | `scripts/test_dollma_<対象>.py` | `python scripts/test_dollma_<対象>.py` (関数 `test_*` + `if __name__ == "__main__"` 自走・pytest でも動く形) |

テスト規約の詳細は `docs/testing.md`。**失敗を緑と報告しない**・スキップしたものは明示する。

## 4. 正典アーティファクトの保護

`.claude/hooks/dollama_protect_artifacts.py` (PreToolUse) が以下への Write/Edit を **deny** する。

- `data/bitnet/bitnet_dense.safetensors` / `data/bitnet/bitnet_dense_fp32.safetensors` (正典重み)
- `data/bitnet/golden/` 配下すべて (C++ 推論の golden)
- `data/bitnet/pairs.train.jsonl` / `data/bitnet/pairs.val.jsonl` (凍結 train/val)
- `data/bitnet/pairs.eval_diverse_a.jsonl` / `data/bitnet/pairs.eval_diverse_b.jsonl` (凍結 diverse-val)

**実験は必ず別名に出す** (`bitnet_dense_<suffix>.safetensors` 等)。正典の差し替えはユーザー決裁を経た
「まとめ焼き」のときだけ行い、その回で golden 再生成と C++ 側テストの緑確認を**同時に**済ませる。
凍結 val に手を入れたくなったら、上書きではなく**加算的に新ファイルを作る**。

## 5. 重み・golden の搬送

正典重み / golden / 大きなペアデータは gitignore されている。マシン間の移動は
`scripts/train_bitnet.py` の `--copy` / `--publish` で **NAS 経由**。

**cross-GPU の再生成は bit 非一致になるため、再訓練で代用せず exact コピーを運ぶ。**
「同じスクリプトを研究機で回せば同じ重みができる」は成立しない。

## 6. 研究機のビルドと実走 (SAC)

### ビルド手順 (研究機・毎回この手順)

```bash
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin:$PATH"
MESON="/c/Users/sdkik/AppData/Local/Python/pythoncore-3.14-64/Scripts/meson.exe"
"$MESON" setup build -Dwith_cuda=true   # CUDA 有効化に必須 (既定 false)
"$MESON" compile -C build
"$MESON" test -C build
```

- meson は PATH に無いのでフルパスで呼ぶ。**PowerShell からは通らないので Bash ツールを使う。**
- `-arch=sm_120` はトップレベルで `add_project_arguments` 済み。

### SAC (Smart App Control) — 実走の制約

**新規・変更した exe の実走は SAC にブロックされる。** ビルド・リンク・コミットは SAC ON のまま通る。

- **`meson test` の前に、必ずユーザーへ「SAC を OFF にしてください」と依頼する** (即時反映・
  再起動は不要)。黙って実行してブロックさせない。
- 症状は `WinError 4551` / Permission denied / 「アプリケーション制御ポリシーによってブロック」。
- **コードを疑う前に、既存 exe が走るかで切り分ける。** SAC のブロックを実装バグと誤診しない。
- 実走できないときの代替: OpenVINO の数値検証は Python (署名済みゆえ SAC に弾かれない) で
  IR を直接走らせて golden 突合し、ビルド緑をもって健全性の証左とする。

## 6.5 ファイル編集とツールの使い分け

- **部分修正は Edit を使う。大きいファイルを Write で全文置換しない** (事故の元)。
- 作業前に対象ファイルを必ず Read して現状を把握する。
- **ライセンス判断はサブエージェントに委譲しない。** 中継された承認を根拠にできないため、
  ライセンスが絡む決裁はユーザー本人に上げる。

## 6.6 生成スコープ (全エージェント共通・絶対)

**生成対象はキャラクターのみ。背景は生成しない。** 出力は切り抜き済みの**透過 PNG** で、
背景は Grok / Gemini / SD + CLIP STUDIO PAINT 側で合成される。

- **背景タグ・情景タグを入れない** (`city` / `forest` / `sunset sky` / `futuristic city` /
  `magical atmosphere` 等)。構図タグは**キャラの写り方**に限る (`upper body` / `full body` /
  `cowboy shot` / `looking at viewer`)。ライティングも**キャラに乗る光**に限る。
- **単独キャラが原則** (複数人は破綻しやすく、品質 FB でも worst 帯の主因だった)。
- `simple background` / `white background` / `transparent background` を使う。これは
  マッティング (ISNet) の精度に直接効く。
- 背景生成を要求するタスクは受けずに差し戻す。

## 7. 実装方針 (使う / 使わない)

自作は **HW を叩く研究コア**に限定し、配管は定番のヘッダオンリーライブラリを使う。

| 使う | 使わない |
|---|---|
| STL 全般 | PyTorch / LibTorch (C++ 側) |
| CUDA Runtime API | diffusers / stable-diffusion.cpp |
| 自作 Tensor / GEMM / Attention / CUDA カーネル | llama.cpp |
| OpenVINO C++ API (NPU / iGPU 推論グルー) | 重量級 HTTP フレームワーク (Drogon 等) |
| cpp-httplib (HTTP・単一ヘッダ) / nlohmann/json | 手書き HTTP・JSON パーサ |

- **cuBLAS / cuDNN は「到達困難になった重い GEMM / Conv のみ」フォールバック許容**。
  自作版に後で置換できる形で入れる。Attention・正規化・活性化は自作を維持する。CUTLASS 置換は不採用。
- Python 側 (訓練・変換・probe) では PyTorch を使う。禁止されているのは **C++ 実装での LibTorch 依存**。

## 8. docs 更新の分担

| ファイル | 書くもの |
|---|---|
| `CLAUDE.md` | 芯の確定事項と代表数値のみ (肥大させない) |
| `docs/measurements-log.md` | 計測の全文・条件・採否理由 |
| `docs/roadmap.md` | Phase ごとの段・状態・経緯 |
| `docs/training-spec.md` / `docs/dataset-spec.md` | 訓練・データの手順と再現コマンド |
| `docs/testing.md` | テスト規約 |
| `docs/character-bible-spec.md` | キャラ設定の構造・出力仕様 |

新しい数値が出たら該当 docs に追記し、**CLAUDE.md には芯だけ**足す。

## 9. 報告フォーマット

完了時は次を明示して返す。

1. **何を**やったか (触ったファイル)
2. **どの機械で**回したか (開発機 / 研究機)
3. **どの数値**が出たか (計測がある場合・条件込み)
4. **どのテストが緑**か (実行コマンドと結果)
5. **やらなかったこと / 未達** (スキップ・ブロッカーは隠さず書く)
6. どの docs に追記したか (または追記案)
