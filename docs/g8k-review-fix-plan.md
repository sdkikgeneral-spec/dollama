# G-8k S2〜S3c レビュー是正プラン (研究機で実施)

**使い方**: 研究機の Claude Code に「`docs/g8k-review-fix-plan.md` の通りに進めて」と渡す。
ステータス: **未着手**。着手時は CLAUDE.md ルール1 (プランモード承認) → ルール2 (`project-leader` 経由) に従う。

- 対象ブランチ: `feat/fast-mode-g0b-g3k` (G-8k S2〜S3c = `device_arena` 移行)
- 由来: 2026-08-20 に**開発機**で実施した静的コードレビュー (ビルド・実走なし)。
  レビュー時点の HEAD = `acca803`、merge-base = `0f8ffab`。
- **行番号は `acca803` 時点のもの**。§0 の merge 後にずれるので、着手時は必ず grep で再特定する。

---

## 0. 事前確認 (最初にやる)

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader   # RTX 5080 / 12.0 を確認
nvcc --version                                                   # 研究機であることの確認
git switch feat/fast-mode-g0b-g3k
git fetch origin
git merge origin/main      # レビュー時点でブランチは main から 11 コミット遅れ
```

`origin/main` の 11 コミットは UI (Blazor) 中心で CUDA 側との衝突はほぼ無い見込みだが、
`src/meson.build` は両方が触るので衝突したら手で解決する。**merge 後にまずビルドを通し、
既存 test が緑であることを確認してから是正に入る** (是正の効果と merge の副作用を混ぜない)。

ビルド手順 (`docs/agent-common.md` §6 と同一):

```bash
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin:$PATH"
MESON="/c/Users/sdkik/AppData/Local/Python/pythoncore-3.14-64/Scripts/meson.exe"
"$MESON" setup build -Dwith_cuda=true
"$MESON" compile -C build
"$MESON" test -C build
```

- meson は PATH に無いのでフルパス。**PowerShell からは通らないので Bash ツールを使う。**
- **`meson test` の前にユーザーへ「SAC を OFF にしてください」と依頼する** (新規 exe の実走が
  ブロックされる。症状 = `WinError 4551` / アプリケーション制御ポリシー)。黙って走らせない。

---

## 1. 是正項目

優先順。**F1〜F4 はコード修正、F5 はテスト追加、F6 は台帳、F7 は軽微。**
F6 は merge 前に必ず終わらせる (理由は当該節)。

### F1 [中] 生成を直列化する — 同時 2 リクエストでアリーナが壊れる

**症状**: `device_arena` の状態 (`owner` / `cur` / `offset` / `req_live`) は全て無ロック。
一方 HTTP 側に生成の排他が無く (`src/server/` に `mutex` の grep ヒット 0)、cpp-httplib は
既定でスレッドプールから並行ディスパッチする。`.cuh` のスレッド契約コメントは
「生成そのものは逐次ゆえ同時 2 本は元々成立しない」を前提に書かれているが、**コード上その
前提は成立していない**。

しかも埋め込み・latent・出力は per-call の raw `cudaMalloc` なので、**アリーナ化以前は
同時 2 リクエストがメモリ安全に成立していた**。今回の変更で以下が新規に生じている:

- 検出できた場合: 生成 T1 の step 間 (アリーナ静止の瞬間) に T2 が正規の所有権移譲で入り、
  T1 の次 forward の `check_thread` が throw → **T1 が途中 step で 500**、かつ
  `generate` 内の raw `cudaMalloc` (`d_latent` 等) は RAII が無いので**リーク**。
- すり抜けた場合: `check_thread` → カーソル bump が非アトミックなため、真の同時進入では
  両者が quiescent 判定を通過し**同一カーソルから重複払い出し = 画像のサイレント破壊**。

「落ちる契約」がレースフリーでないので保証になっていない、という指摘。

**修正**: `src/server/api.cpp:179` の `result = gen.generate(gr);` が **HTTP 側の唯一の
生成ファネル** (`/v1/images/generations` と `/v1/images/edits` は同じハンドラを通る)。
ここをファイルスコープの `std::mutex` + `std::lock_guard` で囲む。

- 生成器の実装 (`BackendImageGenerator` / `PipelineGenerator` / `Txt2ImgGenerator`) 側に
  入れると 3 箇所になるので**採らない**。CLI 経路 (`cli_generate.hpp`) は単一スレッドゆえ不要。
- alloc 回数の少ないプール経路では**性能影響は誤差**。並行受付は元々 VRAM 的に成立しない。
- ロック待ちの間 HTTP スレッドが張り付く点は許容 (現状の設計思想 = 生成は逐次)。
- `.cuh` のスレッド契約コメントを「api.cpp のファネル mutex で担保している」に**更新する**
  (前提を暗黙にしない)。

**受入条件**: 同時 2 リクエストを投げても 500 が出ず、両方が順に完走する。
最小確認は `curl` 2 本同時 (steps を小さく) で 200 × 2 + 画像ハッシュが単発時と一致。

### F2 [中低] ハンドル破棄が共有アリーナを巻き添えにする

**症状**: `src/infer/unet.cu:1380-1381` が `unet_weights_destroy` で
`device_arena_release(UNet)` / `(UNetPersist)` を呼ぶ。アリーナはグローバル共有なので、
パイプライン A と B が同居する構成で **A を破棄すると B の reserve (6080MiB) が消えて
`reserved_bytes = 0`** になる。以後 B の生成はチャンク成長へ静かに回帰 = **S4 で FAIL した
VRAM 事故モード (捨て分 +4.6GB → free 張り付き → WDDM ページング → 2 枚目以降 11s→25s) の
再来**。しかも shortage 警告は `reserved_bytes != 0` を条件にしているので**一切出ない**。

加えて `device_arena_release` は `check_thread` 経由で throw し得るが、呼び出し元
`~DiffusionPipeline` (`src/infer/diffusion.cu:336`) は暗黙 noexcept → 別スレッドの forward
生存中に破棄が走ると **`std::terminate`**。

これは CLAUDE.md に前科として記録されている L-2 → C0 の ODR 事故と**同型のクラス**
(「単体ハンドルの test は複数ハンドル同居の破棄経路を検査しない」) で、**今回も同居破棄を
通す test は追加されていない**。

**修正 (どちらか。判断は着手時に PL / ユーザーへ上げる)**:

- **(a) 参照カウント化 (推奨)**: `device_arena_reserve` / `_release` を acquire/release の
  対にし、カウントが 0 になるまで実解放しない。`reserve_arenas()` の呼び出し元
  (`diffusion.cu:333` ctor / `:420`) とハンドル破棄が対称になる。
- **(b) 所有をパイプラインへ移す**: `unet_weights_destroy` から release を外し、
  `~DiffusionPipeline` だけが release する。単純だが「UNet ハンドル単体を作って捨てる」
  test 経路 (`test_unet_fast` 等) でアリーナが残る挙動になるので、その明文化が要る。

いずれの場合も **`~DiffusionPipeline` 内の release を `try { } catch (...) { }` で囲む**
(dtor から例外を出さない)。飲んだ例外は `arena_profile_enabled()` に関わらず stderr へ 1 行。

**受入条件**: F5 のゲートが緑。かつ `test_unet_fast` / `test_vae_decode` /
`test_diffusion_batch2` が従来どおり緑 (単体ハンドル経路を壊していないこと)。

### F3 [低] reserve 不足が本番で見えない

**症状**: `src/kernels/device_arena.cu:346-352` の `reserve shortage` 警告が
`arena_profile_enabled()` (= `DOLLAMA_PROFILE=1`) 配下。将来 live peak が 6080MiB を
超えると**正しさは保たれたまま静かに 2 倍以上遅くなる** (S4 実測の 11s→25s)。

**修正**: `reserve_warned` の 1 回ガードは残したまま、`arena_profile_enabled()` の条件を外し
`std::fprintf(stderr, ...)` で**無条件に 1 回**出す。例外的イベントなので、本プロジェクトの
「無音の破壊より即死」思想 (`reshape_cur` の throw と同じ) に揃える。

**受入条件**: 意図的に reserve を小さくした一時ビルドで警告が 1 回だけ出ることを目視。
恒久 test は不要 (printf のため)。

### F4 [低] ctor の例外安全

**症状**: `src/infer/diffusion.cu:333` の ctor 末尾 `reserve_arenas()` で 6080MiB の
`cudaMalloc` が throw すると、直前に構築した unet / vae ハンドル (~7GB) が**リーク**する
(ctor throw で dtor は走らず、メンバは raw ハンドル)。16GB 板で他プロセス (ブラウザ / OV /
UI) が VRAM を握っていると、**起動時に現実的に起こり得る新規の失敗点** (従来は起動でき、
生成時に初めて詰まった)。

**修正**: fail-fast 自体は妥当なので挙動は変えず、ハンドルを RAII で包むか、ctor 内で
`try { reserve_arenas(); } catch (...) { /* 明示破棄 */ throw; }` にする。

**併せて (統計汚染のみ・任意)**: `device_arena_reserve` は旧チャンク全解放後の `cudaMalloc`
が throw すると `stats.reserved_bytes` が旧値のまま残る。dangling は無いので優先度は最低。

**受入条件**: ビルド緑 + 既存 test 緑 (異常系の再現は要らない)。

### F5 [テスト] 「ハンドル A 破棄 → 生存ハンドル B 継続」を捕まえるゲート

**制約**: **真の 2 パイプライン同居 test は書けない** — 重み常駐が 1 本 7067MB なので
2 本で 14.1GB + アリーナ 6.3GB > 16GB。よって**アリーナ層で等価な形に落とす**。

**追加先**: `src/tests/test_device_arena.cu` (既に `DOLLAMA_POOL=0` の自動枠も持つ)。

**ゲート内容**:
1. `reserve(UNet)` → 2 回 acquire 相当 → 1 回 release → **`reserved_bytes != 0` のまま**
   (F2(a) を採った場合)。F2(b) を採った場合は「`unet_weights_destroy` 相当の経路では
   release されない」ことを検査する形に読み替える。
2. release 後に再度 alloc しても払い出しが正しい (既存の release 再成長 test の延長)。
3. dtor 相当の経路で release が throw しても terminate しない (F2 の try/catch)。

**受入条件**: `meson test -C build` の `device_arena` / `device_arena_pool_off` 両枠で緑。

### F6 [台帳] `docs/measurements-log.md` の G-8k 行が FAIL のまま — **merge 前に必須**

**症状**: 台帳の G-8k 行は **S4 の「G4 FAIL / G5 FAIL / 方式の再設計が要る / PL 決裁待ち」**
で止まっている。その後の **S3b / S3c の PASS 実測 (13620MB / +310MB) はコミットメッセージに
しか存在しない**。この状態で merge すると**台帳の最終結論が「FAIL」で固定される**。

**修正**: S4 の FAIL 行は**消さずに残し** (加算的に書くのが本プロジェクトの型)、後続として
S3b / S3c の是正内容と §2 の再走実測を 1 行追記して決着させる。併せて:

- `docs/fast-mode-plan.md` の G-8k 節が backlog 表記のまま → S2〜S3c の進行を反映。
- `CLAUDE.md` の計測ベースライン表に G-8k 行が無い → 芯となる数値のみ追加。

### F7 [軽微] コメント・スタイル

- `src/kernels/device_arena.cuh`: 「C++14 前提のため inline 変数は使わない」は
  meson が `cpp_std=c++20` なので**根拠が事実と不一致**。挙動は無害だがコメントを直す。
- 同 `.cuh`: 「revert は `vae_decode.cu` の `kVaeArena` 1 行を戻すだけ」は不正確。
  `src/kernels/conv2d.cu:374 / 704 / 783` が `DeviceArenaId::UNet` をハードコードしているため、
  1 行 revert しても conv の f32 / im2col 中間は UNet アリーナ側に残る (正しさは各アリーナ
  独立 LIFO で保たれるので**コメント精度の問題**)。
- `src/tests/prof_arena_e2e.cu`: `if (...) { return ...; }` の単行 body と CAS ループの `{}` が
  厳格 Allman から逸脱 (既存コードにも同型あり・実害なし)。

---

## 2. 実走で決着させる検証 (静的には確定できない)

**SAC OFF を依頼してから**。F1〜F5 の修正後に回す。

| # | 検証 | 合否の見方 |
|---|---|---|
| V1 | `prof_arena_e2e` を **2 ラウンド (順序 1→2→3 / 3→2→1)** 再走 | PEAK_USED と rgb_hash。S3c 主張の **13620MB / POOL=0 比 +310MB** を再現するか |
| V2 | ゲート余白の頑健性 | 捨て分 205MiB / ゲート余白 **~200MB しかない**。UI (Blazor) + OV (NPU/iGPU) + ブラウザ同居状態で PEAK_USED を再測。WDDM 下の `cudaMemGetInfo` は同居プロセスで揺れる |
| V3 | ctor の 6080MiB **単発** `cudaMalloc` が断片化・同居 VRAM 下で通るか | 失敗時は F4 の起動時 throw 経路に落ちる |
| V4 | reserve 有効時に **1 枚目から `chunk_alloc=0`** (S3c 主張 chunks=1 / cudaMalloc=1)、step 2..20 の実 malloc/free = 0 | merge 後の tree で維持されているか |
| V5 | `DOLLAMA_POOL=0` と既定の **bit 一致**を外部 cmp | `DOLLAMA_G8K_DUMP` / `DOLLAMA_VAE_DUMP` + cmp。自動枠が無いので手動で 1 回 |
| V6 | `test_unet_fast` の poison 歩行 | cursor_peak を 16MiB 刻みで辿るため、チャンク境界の捨て分で cap が 256MiB 余分に立ち `over ≤ 512MiB` ゲートに **~440MiB まで接近し得る**。フレークが出たらここを疑う |

---

## 3. レビューで「問題なし」と確認済み (触らない)

無駄な再調査を避けるため記録する。以下は静的検証で穴が見つからなかった:

- **アリーナのコア論理**: 挿入位置 `a.cur` に対し生存 mark は常に `(chunk, offset) ≤ 挿入時
  カーソル` にしか存在し得ず (LIFO)、境界 (`mark=(cur,0)` で `offset==0` break → 挿入) でも
  rewind 先は「新旧チャンク双方の全域が mark 以後」となり**払い出し重複は起きない** (場合分け
  総当たり)。`align_up` のオーバーフローは SIZE_MAX 近傍のみ。`mark.chunk > size()` の検査は
  空アリーナ mark (== size) を正しく許容。
- **非同期カーネルとの整合**: ブランチ全体に非 default stream の生成が 0 → アリーナ再利用と
  カーネル実行の整合は default stream の逐次性で担保されている。
- **`DOLLAMA_POOL=0`**: 確保サイズ (align_up 差は cudaMalloc 粒度以下)・確保/解放位置・
  `reshape_cur` の free→malloc 連鎖まで**旧経路と等価**。
- **ODR (C0 の再発)**: `device_arena_*` の実体は 1 TU、Scratch 群は匿名 namespace 維持、
  関数内 static → **再発形跡なし**。
- **S3 の「和でなく max」**: 単一スレッド・default stream で UNet step 群と VAE decode は
  時間的に非重畳、decode 開始時に UNet アリーナは静止 (全 Scratch / persist scope が forward
  末尾で rewind 済み) → capacity = max(5914, UNet forward ~1095) = 5914 で成立。
- **live peak の設計値**: VAE decode 時の同時生存物を形状から独立に積算すると
  **5914.25 MiB** = コード定数 `kArenaLivePeakUnetMiB = 5914` と小数点以下まで一致
  (FP16 キャリー 1024 + FP32 キャリー 2048 + up2 Scratch32 2560 + conv f32 重み 2.25 +
  im2col col 252 + out_band 28)。Persist も 137.5 → `137` 一致。**live peak は形状決定的で
  分散を持たない**ため、S3c の「ヘッドルームを比例→固定」は算術的に正しい。
- **S3b が過大だった理由**: `6051` (= unet 5914 + persist 137 の**合算**) を UNet 単体の
  live peak と取り違えて 1.1 倍していた → S3c で分離。差 592MiB ≈ 実測差 630MB で整合。
- **テストのゲート設計**: 新ゲートは全て memcmp ビット一致 / 実 cudaMalloc=0 / cap−live≤512MiB の
  **ハードゲートで `ok=false` → `return 1` に配線済み**。poison (0xFF/0x00) → 再走 → memcmp は
  未初期化読みを能動的に炙る良い形。**CFG 増幅下 (g>1) の SSIM ゲートは 1 本も混ざっていない**
  (G-4k S3 の教訓「被験変数は g=1.0 で分離する」に適合)。

**既知かつ意図的なギャップ** (是正対象ではない): `prof_arena_e2e` は計測専用で meson test 未登録
(明記あり) → e2e VRAM の自動ゲートは無い。`test_unet_fast` / `test_vae_decode` の POOL=0 比較は
手動運用 (自動枠があるのは `test_device_arena` のみ)。

---

## 4. 完了報告に含めること

`docs/agent-common.md` §7 準拠:

1. **どの機械で**回したか (研究機 / SAC の状態)
2. F1〜F7 の実施可否 (**やらなかったものは明示**)
3. `meson test -C build` の結果 (件数と緑/赤。**失敗を緑と報告しない**)
4. §2 の V1〜V6 の実測値 (PEAK_USED / rgb_hash / 秒。**秒は characterization で、機体クロック
   ドリフトを併記**する)
5. F6 で台帳に書いた行の内容
