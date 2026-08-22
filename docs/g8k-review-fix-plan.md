# G-8k S2〜S3c レビュー是正プラン (研究機で実施)

**使い方**: 研究機の Claude Code に「`docs/g8k-review-fix-plan.md` の通りに進めて」と渡す。
ステータス: ✅ **クローズ (2026-08-22 = G-8k S6)**。決裁スコープ (**F1 / F3 / F4 / F2 は dtor try/catch のみ / F5 / F7**) を**見送りゼロで全件実施**し、研究機で **53/53 緑**。F6 は S5f〜S5h で解消済。
**本ファイルは「当時の計画」の原本として残す** — 実施結果・当初案との差・プランの記述誤りは
**§5 (末尾) に加算**した。実測値の正本は `docs/measurements-log.md` の「G-8k S6」計測行と
「G-8k S6 (T2)」節、コード側の実装記録は `docs/fast-mode-plan.md` の G-8k 実装記録「S6」項。
★**本文中の記述には後から誤りと判明したものが 2 点ある** (F1 の 2 箇所・★訂正を各所に入れた)。

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

(★**実績 (2026-08-22)**: 実際の merge は `5da3bfb` で **13 コミット** (起草時の見込み 11 から増えた)。`src/` と `meson.build` は **1 行も触られていない**ことを `git diff --name-only` で確認済 = 実体は UI/Blazor + docs のみ。時系列は **① 着手前に merge → ② その後に PL が「被験変数を増やさないため merge は後回し」と決裁 → ③ 既に済んでいたので、影響が無いことを確かめて続行** の順 (一次証拠で日付が取れるのは merge commit `5da3bfb` 自体だけで、②③ の順序は作業報告による)。merge 後のベースラインが `meson test` 53/53 緑であることを確認したうえで続行した。)

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

~~しかも埋め込み・latent・出力は per-call の raw `cudaMalloc` なので、**アリーナ化以前は
同時 2 リクエストがメモリ安全に成立していた**。今回の変更で以下が新規に生じている:~~
★**訂正 (S6・相互レビューで反証)**: この一文は**誤り**。生成経路が握るプロセス共有の無ロック
可変状態は **3 つ**あり、アリーナはそのうち最も新しい 1 つにすぎない —
`src/kernels/gemm.cu:363` の `g_cublas_handle` (**遅延生成が非アトミック**・導入 `fc97b76` /
2026-06-22 / 2-6 S3-B。`cublasSetStream` の呼び出しは repo に 0 箇所 = 全部デフォルトストリーム) と、
`src/kernels/groupnorm.cu:223-238` の `g_mb_buf` (**grow-only 再確保**・導入 `42ec1be` /
2026-07-11 / G-4k S1a。出荷 `--fast` は epilogue を含意するので実運用経路)。
→ **HTTP 並行生成は G-8k の退行ではなく、少なくとも 2-6 から一貫して不成立**であり、
**ファネル mutex を外せるのは 3 つすべてが個別にスレッド安全化されたとき**である。
以下の 2 点は「G-8k で新規に生じた」ではなく「G-8k が 3 つ目として足した」と読むこと:

- 検出できた場合: 生成 T1 の step 間 (アリーナ静止の瞬間) に T2 が正規の所有権移譲で入り、
  T1 の次 forward の `check_thread` が throw → **T1 が途中 step で 500**、かつ
  `generate` 内の raw `cudaMalloc` (`d_latent` 等) は RAII が無いので**リーク**。
- すり抜けた場合: `check_thread` → カーソル bump が非アトミックなため、真の同時進入では
  両者が quiescent 判定を通過し**同一カーソルから重複払い出し = 画像のサイレント破壊**。

「落ちる契約」がレースフリーでないので保証になっていない、という指摘。

**修正**: `src/server/api.cpp:179` の `result = gen.generate(gr);` が **HTTP 側の唯一の
生成ファネル** ~~(`/v1/images/generations` と `/v1/images/edits` は同じハンドラを通る)~~。
ここをファイルスコープの `std::mutex` + `std::lock_guard` で囲む。
★**訂正 (S6)**: 括弧内は**誤り**。`/v1/images/edits` は別ハンドラ (`src/server/api.cpp:406-411`) で
`write_error(res, 501, ...)` を返すだけで、**生成には一切到達しない**。したがって
「唯一の生成ファネル」という結論そのものは正しいが、その根拠は「edits も同じハンドラだから」ではなく
「**edits が 501 で生成へ行かないから**」である。

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

---

## 5. 実施結果 (2026-08-22 = G-8k S6・本ファイルはここでクローズ)

**どの機械で**: 実装・実走とも研究機 `KIK-WIN-RTX58` (SAC OFF)。負のコントロールのみ開発機
(MinGW g++ `-O3`・**スクラッチ配下でビルドし repo は未変更**)。
**プロセス**: F1 を `cpp-implementer`、F2〜F7 を `cuda-kernel-dev` が実装 → **相互 read-only レビュー**
(書き手と検査者を分ける)。★**実施順は「レビュー → 是正 → 実走」の直列ではない** (2026-08-23 の監査で実測時刻から是正):
`gpu-benchmarker` の **V1〜V6 (是正前ツリー) は 12 件の是正・53/53 緑より前**に走っており
(V1 = 23:46:58–23:57:21・採取全体の mtime 00:11 に対し、最終ツリーの testlog は 00:21:44)、**相互レビューと並行**していた。
時系列は **V1〜V6 (是正前ツリー) → 指摘 12 件の是正 → 研究機で `meson test` 53/53 緑 → 最終ツリーで V1f / V5f (00:26–00:31) を再確認** の順。
(★時刻の出所は研究機ローカルのログ mtime で、**本 repo からは検証できない**。)

### 5.1 F1〜F7 の実施可否

| # | 決裁 | 実施 | 当初案との差 |
|---|---|---|---|
| F1 | 実施 | ✅ | 差なし (ファイルスコープ mutex で `gen.generate(gr)` 1 文のみ)。ただし**症状記述に誤りが 2 点**あった (上記 ★訂正) |
| F2 | **dtor try/catch のみ**採用。(a) 参照カウント化 / (b) 所有移動は**不採用** | ✅ | 決裁どおり。`device_arena_release_noexcept` を**新設**して dtor 経路に使い、`~DiffusionPipeline` は `destroy_resources() noexcept` へ集約。**`maybe_release_arenas()` は noexcept 化していない** (generate 経路であって dtor ではない) |
| F3 | 実施 | ✅ | 差なし (`arena_profile_enabled()` 撤去・無条件 stderr 1 回)。ただし**警告文言が変わった** (§5.3) |
| F4 | 実施 | ✅ | ★**当初案 (`reserve_arenas()` だけを try で覆う) より範囲を広げた** — レビューで「1 行上の `vae_weights_create` が throw すると同じ 5.1GB リークが残る」と指摘され、**device 確保区間全体**を覆う形にした。F4b (統計汚染・「任意」扱いだった分) も実施 |
| F5 | 実施 | ✅ | ★**S6 で 4 ゲートを新設** (`git show HEAD:src/tests/test_device_arena.cu` に `release_noexcept` は 0 件 = **差分としては 4 本すべて新規**。作業過程で「3 本で着手 → レビュー指摘で不正 id (`arena_of` 経路) を追加」となったのは**未コミットの中間状態**で、S6 以前に 3 本あったわけではない)。**POOL 枠 / POOL OFF 枠の両方**に登録。**新規 meson test は 0 本** (`src/meson.build` 未接触・既存 exe 内の test 関数として増える) |
| F6 | 対象外 (S5f〜S5h で解消済) | — | 台帳の G-8k 行は S5 系の 7 ラウンドで決着済み |
| F7 | 実施 | ✅ | 差なし。`prof_arena_e2e.cu` の Allman 整形は 9 箇所。`device_arena.cuh` の単行 Allman 1 箇所は**据え置き** (プランのスコープ外) |

### 5.2 F2 の決裁根拠 ((a)/(b) を不採用にした理由)

production の `unet_weights_create` 呼び出し元は `src/infer/diffusion.cu` の 1 箇所だけで、
**パイプライン同居は現状発生しない**。ただし「同居しない」を担保しているのは値保持ではなく
`src/server/cli_generate.hpp` の**排他フォールバック梯子** (段1 が非 null なら段2 を作らない) という
不変条件である (`DiffusionPipeline` を値で持つラッパは `src/server/diffusion_runner.cu` と
`src/server/pipeline_generator.hpp` の 2 つ)。**梯子を「両方作って良い方を選ぶ」形に書き換えた瞬間に
同居が成立する**ので、そのときは所有権設計をやり直すこと。着手時期は **2-6d (SDXL 3 preset で
複数バックエンド常駐が現実になる)** と決裁済み。

なお `unet_weights_destroy` の呼び出し元は **9 箇所**で、dtor 経路は**そのうち 1 つだけ**
(直呼び 8 = `prof_unet_fast_warm` 1 / `test_unet` 1 / `test_unet_fast` 1 / `test_lora_runtime` 5)。
そのため noexcept 化の狙いは「terminate をここで初めて防ぐこと」ではなく、①dtor 経路の二重防御
②**UNet 側が拒否されても UNetPersist の解放へ進めること** (素の release だと 1 本目の throw で
2 本目に到達せず persist 176MiB が残る) の 2 点である。直呼び 8 箇所では release 失敗が throw から
「stderr 1 行 + 続行」に変わるが、throw が起きるのは「foreign thread + 生存確保あり」だけで
8 箇所はその状況を作らないため実質の契約緩和は無い。

### 5.3 F3 で変わった警告文言 (旧文言を引用している手順書があれば更新すること)

- **旧 (S3b 以来・`5da3bfb` 時点)**: stdout・`DOLLAMA_PROFILE=1` 配下・
  `... > reserve %zu MiB (falling back to chunk growth)`
- **新 (S6 確定)**: **stderr・無条件 1 回**・
  `... (falling back to chunk growth; reserve is undersized -- see reserve_arenas() in src/infer/diffusion.cu)`
- **接頭辞 `[ALLOC] reserve shortage: arena=…` は不変**。
- 文面を arena 非依存にしたのは相互レビュー指摘による: persist 側の予約量は
  `arena_reserve_persist_mb()` の**固定 176MiB** で、`DOLLAMA_ARENA_RESERVE_MB` は persist に対して
  「0 か否か」のキルスイッチとしてしか効かない → `arena=unet_persist` で「env を上げろ」と誘導すると嘘になる。

### 5.4 §2 の V1〜V6 の結果 (要旨・数値の正本は measurements-log)

H4 (step ループ内 malloc 0) / H5 (出力不変) / H6 (VRAM) は**是正前ツリー・最終ツリーとも全 PASS**。

- **V1**: 順序依存なし。★**S3c/S4b の絶対値 13620/13649MB は再現せず 13397MB。この 252MB 差は未説明**
  (★起草時の「同居量の差」という帰属は 2026-08-23 の監査で**撤回**した。`used_after_ctx=1317MB` は常駐
  1229 / 616 / 123 MiB のいずれでも不変で常駐差では説明できず、算術も合わない (常駐差 613MiB ≒ 643MB に対し
  実測差 252MB / POOL=0 側 212MB)。**判定は同一セッション `POOL=0` 比 delta なので合否には影響しない**。
  詳細は `docs/measurements-log.md` の「G-8k S6」計測行)。
  **delta は +380MB → +340MB でほぼ不変** = §2 の「S3c 主張の 13620MB / +310MB を再現するか」という
  問いの立て方が**絶対値寄りで不適切**だった。**判定は同一セッションの `POOL=0` 比 delta で行うこと**。
- **V2**: 同居 2048MiB の hog を上乗せしても **delta +340MB で清浄時と完全同値** → 絶対値は同居量ぶん
  平行移動するが delta は同居に不変。§2 が懸念した「WDDM 下で同居プロセスに揺れる」は delta では起きない。
- **V3**: ★**F4 の異常系は再現しなかった**。同居 hog を idle 2560〜10240MiB / active 8192・11264MiB まで
  振っても **9 条件**すべてで reserve 成功 (WDDM の eviction による。内訳 = idle 2560/3072/3584/4096 + 6144/8192/10240 + active 8192/11264 = 9 本。起草時の「8 条件」は誤りで 2026-08-23 の監査で是正)。**「F4 を実走で確認した」と書かないこと** —
  確定したのは拡大した try が正常系で無害であることまでで、異常系の正しさはコード上の推論に依拠する。
- **V4**: 1 枚目から `chunk_alloc=0`・step ループ内 実 malloc/free 0 を merge 後ツリーでも維持 (= H4)。
- **V5**: `DOLLAMA_POOL=0` と既定の外部 cmp が **6/6 BIT-EXACT**、最終ツリーでは **sha256 全 10 本一致**。
- **V6**: poison 歩行の `over = 245MiB` が **18 サンプル全部で固定** (ゲート 512MiB / 余白 267MiB)。
  §2 が懸念した「~440MiB まで接近し得る」は**出なかった**・フレークなし。
- 対照 `DOLLAMA_ARENA_RESERVE_MB=0` で **S4 の病態を再現** = reserve が何を防いでいるかの現地証拠。
  秒は **1 枚目 11.0s → 2 枚目以降が A ラウンド 27.1s / B ラウンド 24.3s** (★**1 走行内で 27.1 → 24.3 と回復したのではない**。
  `A3_reserve0` = 11.0411 / 27.1192 / 27.185s・`B3_reserve0` = 10.9766 / 24.3435 / 24.314s で、各ラウンド内では 2・3 枚目とも悪化したまま)。

### 5.5 §3「触らない」に足す一次証拠

§3 の「レビューで問題なしと確認済み」は S6 でも覆っていない。加えて S6 で新たに現物確認した点:

- **再入経路なし**: `IImageGenerator::generate` の production override は 4 実装
  (`backend_image_generator` / `pipeline_generator` / `txt2img_generator` / `stub_generator`) で、
  どれも他の `IImageGenerator` を保持しない。production の呼び出し元は `api.cpp` と `main.cpp` の 2 箇所のみで、
  `--http` と CLI は 1 プロセス内で排他 → **非再帰の `std::mutex` で足りる**。
- **`conv2d.cu:374 / 704 / 783` の `DeviceArenaId::UNet` ハードコードは現存** (F7 の revert 手順是正の根拠)。

### 5.6 本ファイルの行番号について

**§1〜§3 の行番号はすべて `acca803` (2026-08-19) 基準**で、S6 の変更で多くがずれている。
例: F1 が指す `src/server/api.cpp:179` は `5da3bfb` までは正しかったが、S6 で mutex の定義コメントを
足したため現在は `:225`。**参照するときは必ず grep で再特定すること** (§0 の指示と同じ)。
