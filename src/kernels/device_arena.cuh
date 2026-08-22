// device_arena — チャンク連結型 bump アリーナ (G-8k S1a 基盤)
// 対象: RTX5080 (Blackwell / sm_120) / CUDA Runtime API のみ
//
// ----------------------------------------------------------------
// 目的:
//   step ループ内の cudaMalloc / cudaFree を撲滅する。cudaFree は暗黙のデバイス
//   同期点であり、occupancy/latency 律速の阻害要因。さらに CUDA Graphs (G-1k) は
//   capture 中の malloc/free を許さないため、その前提条件でもある。
//
// 設計の急所 (これを外すと壊れる):
//   本アリーナの利用者は「同時生存が複数本」ある (unet.cu の Scratch は
//   transformer_block 1 回で 9-13 本が同時に生きている)。したがって
//   groupnorm.cu の g_mb_buf のような「grow したら古い領域を cudaFree して
//   再確保する」grow-only 単一バッファ方式は **絶対に一般化してはならない**
//   (生存中のポインタが dangling になり、memcmp が通ったり通らなかったりする)。
//   本実装は「現チャンクが足りなければ新チャンクを追加確保し、既存チャンクは
//   解放しない」チャンク連結型にすることで、**既に配ったポインタが無効化される
//   ことが原理的に無い** 構造になっている。
//
// 使い方 (S2 での Scratch 置換を想定):
//   {
//       DeviceArenaScope sc(DeviceArenaId::UNet);   // = mark()
//       __half* p = sc.alloc<__half>(n);            // 呼び出し側は従来と同形
//       ...
//   }                                              // デストラクタで rewind()
//
// キルスイッチ:
//   環境変数 DOLLAMA_POOL=0 (存在かつ "0") のとき、alloc は素の cudaMalloc、
//   rewind は対応する cudaFree にフォールバックし、旧経路が完全に復元される。
//
// VRAM 収支の急所 (G-8k S3b で確定・善意の再設計を止めるために残す):
//   **捨て分 (total_capacity - peak_request_bytes) の唯一の発生源はチャンク跨ぎ**である。
//   現チャンクの残りに入らない要求が来ると、その残りを丸ごと捨てて新チャンクへ移る。
//   GB 級の単発要求 (VAE の FP16 キャリー 512MiB x2 / FP32 キャリー 1024MiB x2 /
//   up2-up3 の Scratch32 512MiB-1024MiB 級) が並ぶと、これが積み上がる。
//   S4 の e2e 実測では capacity 10688MiB に対し真の live peak 6051MiB = 捨て分 4637MiB
//   となり、プロセス GPU peak が POOL=0 比 +2993MB で VRAM ゲートを割った
//   (物理 16302MB に張り付き WDDM ページングで 2 枚目以降が 11s -> 25s に劣化)。
//
//   **解は事前 reserve** (device_arena_reserve): live peak を包む 1 本を静止状態で
//   先に確保しておけば、跨ぎは原理的に起きない (capacity ≒ live peak)。既存の
//   「跨いだら新チャンク」経路はフォールバックとして残るので、reserve が足りなくても
//   壊れず遅くなるだけ。確保が初期化時に固まるため、1 枚目から chunk_alloc=0 になり
//   CUDA Graphs (G-1k) の capture 前提もウォーム 1 回で満たせる。
//
//   採らなかった案 (再提案を止めるための記録):
//     (1) VAE の GB 級を「固定寿命バッファ」に切り出す案は **不採用**。UNet アリーナの
//         capacity は ~1280MiB に戻るが、プロセス peak が立つのは VAE decode の瞬間で、
//         その瞬間この 1280MiB は丸ごと遊んでいる。試算 peak ≈ weights 7067 +
//         VAE 固定 ~5632 + 遊休 1280 + persist ~190 ≈ 14169MB で POOL=0 の 13309MB に
//         対し +860MB > 512MB = VRAM ゲート未達。「二相の live が同時常駐する」構造が
//         残るうえ vae_decode.cu の全面改修で S3 の BIT-EXACT 資産を危険に晒す。
//     (2) アリーナ基盤そのものの再設計 (フリーリスト化等) はコスト最大で、
//         「配ったポインタが無効化されない」という急所を触る危険が最も大きい。
//
// スレッド契約 (G-8k S2 で改訂):
//   同時使用は単一スレッドのみ。初回使用時に所有スレッド id を記録し、
//   **生存中の確保がある状態**で別スレッドが触ったら throw する (落ちる契約)。
//   ただしアリーナが静止状態 (カーソルが基点・生存確保ゼロ) のときは、所有権を
//   触ったスレッドへ移譲する (= 逐次的な引き渡しは許す)。
//   理由: HTTP サーバー (cpp-httplib) はリクエストごとにスレッドプールの別ワーカー
//   で生成を実行するため、リクエスト間で GPU 実行スレッドが変わる。生成そのものは
//   逐次 (同時 2 本は元々成立しない) なので、静止状態での移譲は安全であり、
//   同時進入だけを確実に検出する形に落とし込む。
//   注意: これは並行生成の直列化を代替しない。直列化は src/server/api.cpp の
//   生成ファネル mutex (g_generate_mutex) で担保している (handle_generations が
//   IImageGenerator::generate を lock_guard 下で呼ぶ 1 箇所のみ)。本アリーナの
//   スレッド検査は、その担保が外れたときに黙って壊れず落ちるための保険である。
//
// ODR 注意:
//   本ヘッダは TU 横断の外部リンケージを意図的に持つ (C0 の DeviceWeights /
//   Scratch 二重定義の再演を避けるため、シンボルは dollama::device_arena_* に
//   統一し、実体は device_arena.cu 1 本だけに置く)。ヘッダ内に __global__ は
//   置かない (置く場合は static 必須)。
// ----------------------------------------------------------------
#pragma once

#include <cstddef>
#include <cstdint>

namespace dollama
{

// ----------------------------------------------------------------
// アリーナ識別子。VAE の GB 級チャンクを UNet と混ぜないため分ける
// (S3 だけ独立に revert 可能・trim 粒度も分離できる)。
//
// UNet        : 入れ子 mark/rewind の**短命**スクラッチ (Scratch / conv2d im2col)。
// UNetPersist : UNet forward 1 回の**寿命いっぱい**生きる常駐 (temb / skip / d_cur)。
//               G-8k S2 で追加。短命側と同じアリーナに置くと、確保と解放が交差して
//               LIFO 契約が破れる (skip を積んだ後に Scratch が rewind すると skip の
//               領域が再利用されてしまう)。**アリーナを分けることが正しさの条件**で
//               あって、単なる整理ではない。
// VAE         : **S3 で不採用** (未使用のまま残す)。VAE decode のスクラッチは
//               UNet アリーナへ共有した (vae_decode.cu の kVaeArena)。理由:
//               VAE の同時生存ピークは ~5.5GiB (FP16 キャリー 2 本 1.0GiB + FP32
//               キャリー 2 本 2.0GiB + up2 の Scratch32 群 ~2.5GiB) で、専用アリーナに
//               置くと次画像の UNet step 中ずっと常駐して VRAM peak を押し上げる。
//               UNet アリーナへ共有すれば capacity は「和」ではなく「max」で済む。
//               VAE decode の時点で UNet アリーナは静止状態 (全 step 完了後) なので
//               LIFO 契約は破れない。conv2d の im2col は元から UNet アリーナ共有だった。
//               revert 手順 (正確に): vae_decode.cu の kVaeArena を DeviceArenaId::VAE
//               へ戻す **だけでは足りない**。conv2d.cu が DeviceArenaId::UNet を 3 箇所
//               (im2col / f32 中間) にハードコードしているため、VAE decode 中の conv も
//               UNet アリーナ側に残る。正しさは各アリーナ独立 LIFO で保たれるので
//               (壊れるのは VRAM 収支の読みだけ)、これはコメント精度の問題。
// ----------------------------------------------------------------
enum class DeviceArenaId : int
{
    UNet        = 0,
    VAE         = 1,
    UNetPersist = 2,
};

// 本数 (内部テーブルの寸法)。
// (プロジェクト全体は meson の cpp_std=c++20 だが、CUDA TU のホスト側は
//  **MSVC ホストのときのみ** src/meson.build で -Xcompiler /std:c++14 に落としてある
//  (条件は src/meson.build の if cpp.get_id() == 'msvc' ブロック)。理由は同ファイルの
//  一次記述どおり「CUDA 13.3 + MSVC で c++17/20 ヘッダ組合せの 0xC0000409 を回避」で、
//  どのコンポーネントが落ちるかは特定されていない (断定して書かない)。
//  本ヘッダは .cu からしか include されないので、C++17 の inline 変数には
//  頼らない。TU ローカル定数で足りる)
static constexpr int kDeviceArenaCount = 3;

// 表示名 ("unet" / "vae" / "unet_persist")。プロファイル出力・エラーメッセージ用。
const char* device_arena_name(DeviceArenaId id);

// 全アロケーションのアラインメント (バイト)。最低 256B (要件)。
static constexpr size_t kDeviceArenaAlign = 256;

// ----------------------------------------------------------------
// mark / rewind のトークン。LIFO でのみ使う (取得順の逆順に rewind する)。
//   chunk / offset : プール有効時のカーソル位置。
//   fallback_count : DOLLAMA_POOL=0 時の「その時点までに配ったポインタ本数」。
// ----------------------------------------------------------------
struct DeviceArenaMark
{
    DeviceArenaId id             = DeviceArenaId::UNet;
    size_t        chunk          = 0;
    size_t        offset         = 0;
    size_t        fallback_count = 0;
    size_t        req_bytes      = 0;  // その時点の「同時生存要求バイト」(S2b)
};

// ----------------------------------------------------------------
// 統計カウンタ (S4 で DOLLAMA_PROFILE=1 の [ALLOC] 行として出す)。
// ----------------------------------------------------------------
struct DeviceArenaStats
{
    uint64_t cuda_malloc_calls  = 0;  // 実 cudaMalloc 回数 (累積)
    uint64_t cuda_free_calls    = 0;  // 実 cudaFree 回数 (累積)
    uint64_t chunk_alloc_calls  = 0;  // 新規チャンク確保回数 (累積・プール経路のみ)
    uint64_t alloc_calls        = 0;  // alloc() 呼び出し回数 (累積)
    size_t   total_capacity     = 0;  // 現在の総容量 (バイト・全チャンク合計)
    size_t   live_chunks        = 0;  // 現在のチャンク本数
    size_t   bytes_in_use       = 0;  // 現在の使用中バイト (bump カーソル基準)
    size_t   peak_bytes_in_use  = 0;  // 使用中バイトのピーク
    // G-8k S2b: カーソル基準の bytes_in_use は「跨いだチャンクの捨て分」も数えるため、
    // 真の同時生存量 (= POOL=0 の live 合計と直接比較できる値) を別に持つ。
    // VRAM ゲート (capacity - 真の live peak <= 512MB) はこちらで判定する。
    size_t   live_request_bytes = 0;  // 現在の同時生存要求バイト (アライン後合計)
    size_t   peak_request_bytes = 0;  // 同上のピーク
    // G-8k S3b: device_arena_reserve() で先に確保した 1 本のサイズ (0 = 未 reserve)。
    // capacity - peak_request が捨て分なので、reserved_bytes >= peak_request なら
    // チャンク跨ぎは起きていない (= reserve が効いている) と読める。
    size_t   reserved_bytes     = 0;
};

// ----------------------------------------------------------------
// 低レベル API
// ----------------------------------------------------------------

// プールが有効か (DOLLAMA_POOL=0 なら false = 素の cudaMalloc/cudaFree)。
// getenv は初回のみ・以後キャッシュ。
bool device_arena_pool_enabled();

// 現在のカーソル位置を記録する。
DeviceArenaMark device_arena_mark(DeviceArenaId id);

// bytes 分をアリーナから切り出す (256B アライン)。bytes==0 は nullptr。
void* device_arena_alloc(DeviceArenaId id, size_t bytes);

// mark 時点までカーソルを巻き戻す。**チャンクは解放しない** (grow は単調)。
// DOLLAMA_POOL=0 のときのみ、対応する cudaFree を行う。
void device_arena_rewind(const DeviceArenaMark& mark);

// ----------------------------------------------------------------
// G-8k S3b: 事前 reserve。**静止状態でのみ呼べる** (生存確保があれば throw)。
//   既存チャンクを全解放し、bytes ちょうど 1 本を確保してカーソルを基点に戻す。
//   以後 bytes 以内の確保はこの 1 本から切られ、チャンク跨ぎ = 捨て分が消える。
//   bytes を超えたら従来どおり新チャンクを追加する (正しさは不変・遅くなるだけ)。
//   DOLLAMA_POOL=0 のときは **no-op** (旧経路を汚さない)。
//   bytes == 0 も no-op (reserve 無効化の指定は呼び出し側で「呼ばない」で表現する)。
// ----------------------------------------------------------------
void device_arena_reserve(DeviceArenaId id, size_t bytes);

// 指定アリーナの全チャンクを解放し、カーソルを初期化する。
// 定期 trim / step 間 trim は **実装しない** (目的を殺すため)。明示呼び出し専用。
void device_arena_release(DeviceArenaId id);

// ----------------------------------------------------------------
// G-8k T2 (F2): device_arena_release の noexcept ラッパ。
//   戻り値 true = 解放した / false = **1 バイトも解放していない**。
//   後者が契約として成立する根拠: device_arena_release が投げうるのは
//   arena_of の不正 id と check_thread の 2 つだけで、どちらも最初の cudaFree
//   より前に throw する。よって「途中まで解放して false」は原理的に起きない。
//   本ラッパは release 本体のロジックを **一切複製しない** (複製して部分解放を
//   作ると、生存ポインタが dangling になり本アリーナの芯が壊れる)。
//
//   使うのは unet_weights_destroy (src/infer/unet.cu) の 2 本だけ。狙いは
//     (1) dtor 経路 (~DiffusionPipeline -> destroy_resources -> unet_weights_destroy)
//         の二重防御。dtor から例外を出さないこと自体は destroy_resources() の
//         try/catch が担保しているので、ここは保険である
//         (「terminate をここで初めて防いでいる」わけではない = T2 相互レビュー 中2)。
//     (2) UNet 側が拒否されても UNetPersist の解放へ進めること。素の release だと
//         1 本目の throw で 2 本目に到達せず、persist 側が確実に残る。
//   なお unet_weights_destroy 自体は dtor 専用ではなく、test / prof から直呼びされる
//   経路が 8 箇所ある (そちらでは release 失敗が throw から stderr 1 行 + 続行に変わる)。
//   **generate 経路 (diffusion.cu の maybe_release_arenas) では使わない**:
//   あそこで失敗を握り潰すと、壊れた状態のまま reserve をやり直して生成が続く。
//
//   採らなかった案 (再提案を止めるための記録・ユーザー決裁済):
//     (a) アリーナの参照カウント化 / (b) アリーナ所有権の移動。
//     どちらも採らない。production の unet_weights_create 呼び出し元は
//     src/infer/diffusion.cu の 1 箇所だけで、**パイプライン同居は現状発生しない**。
//     ただし「同居しない」を担保しているのは値保持ではない (T2 相互レビュー 軽1):
//     DiffusionPipeline を値で持つラッパは 2 つある
//       - src/server/diffusion_runner.cu の DiffusionRunner::pipe_
//       - src/server/pipeline_generator.hpp の PipelineGenerator::pipe_
//     同居を防いでいるのは **src/server/cli_generate.hpp の排他フォールバック梯子**
//     (段1 が非 null なら if (!gen) で段2 を作らない) という不変条件である。
//     unique_ptr::operator= は新オブジェクトを構築し終えてから旧を破棄するので、
//     この梯子を「両方作って良い方を選ぶ」形に書き換えた瞬間に同居が成立し、
//     アリーナ (プロセス常駐・単一テーブル) を 2 本のパイプラインが取り合う。
//     **梯子を非排他にするなら、その前に所有権設計をやり直すこと。**
//     所有権設計は 2-6d (SDXL 3 preset で複数バックエンド常駐が現実になる) に
//     着手する時点で改めて設計する。
// ----------------------------------------------------------------
bool device_arena_release_noexcept(DeviceArenaId id) noexcept;

// 統計の読み取り。
DeviceArenaStats device_arena_stats(DeviceArenaId id);

// 累積カウンタのみゼロクリア (容量・チャンクは保持)。ベンチ区間の切り出し用。
void device_arena_reset_counters(DeviceArenaId id);

// 最初のチャンクの既定サイズ (バイト)。成長則の起点。テストが
// 「チャンク跨ぎ」を形状非依存に強制するために公開する。
size_t device_arena_first_chunk_bytes(DeviceArenaId id);

// ----------------------------------------------------------------
// RAII スコープ: 構築で mark()、破棄で rewind()。
//   S2 では unet.cu の Scratch をこれに置き換えるだけで、alloc の呼び出し側は
//   1 行も変わらない (要素数 n を受けて T* を返す形を維持)。
// ----------------------------------------------------------------
class DeviceArenaScope
{
public:
    explicit DeviceArenaScope(DeviceArenaId id)
        : mark_(device_arena_mark(id))
    {
    }

    ~DeviceArenaScope()
    {
        // デストラクタからは投げない (rewind は throw しうるため握り潰す)。
        // 所有スレッド違反はこの時点で既に alloc 側が検出している。
        try
        {
            device_arena_rewind(mark_);
        }
        catch (...)
        {
        }
    }

    DeviceArenaScope(const DeviceArenaScope&)            = delete;
    DeviceArenaScope& operator=(const DeviceArenaScope&) = delete;

    // 要素数 count 個ぶんの T 配列を切り出す (従来の Scratch::alloc と同形)。
    template <typename T>
    T* alloc(size_t count)
    {
        return static_cast<T*>(device_arena_alloc(mark_.id, count * sizeof(T)));
    }

    // バイト数指定版。
    void* alloc_bytes(size_t bytes)
    {
        return device_arena_alloc(mark_.id, bytes);
    }

    DeviceArenaId id() const { return mark_.id; }

private:
    DeviceArenaMark mark_;
};

} // namespace dollama
