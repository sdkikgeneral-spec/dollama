#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L-1 offline LoRA マージツール (dollama)。

SDXL base UNet の safetensors checkpoint へ LoRA を事前マージし、合成 checkpoint を
吐く offline 前処理。SDXL 推論は走らせず、host 側の safetensors 重み演算のみで完結する。

核は kohya 命名 LoRA を diffusers 命名 base へ写像し、低ランク積を加算すること:

    W' = W + scale * (B @ A),  scale = strength * (alpha / rank),  rank = A.shape[0]

  - LoRA は kohya 命名:
      lora_unet_<flattened>.lora_down.weight  = A,  shape [rank, in]
      lora_unet_<flattened>.lora_up.weight    = B,  shape [out, rank]
      lora_unet_<flattened>.alpha             = スカラ (無ければ alpha=rank)
  - base UNet は diffusers 命名 (UNet2DConditionModel.state_dict() の key)。

スコープ = UNet マージのみ。lora_te1_* / lora_te2_* (CLIP-L/bigG) は dollama では
OV IR 化済みで safetensors マージ経路に乗らないため検出して warn + skip する。

CLI:
  --base <base.safetensors> --lora <lora.safetensors> --out <merged.safetensors>
  --strength <float=1.0> --dtype <fp16|fp32, 既定は base に追従>

ベンチ (合成データ・本番非汚染):
  python scripts/dollma_merge_lora.py --bench           # smoke (小規模)
  DOLLAMA_BENCH=1 python scripts/dollma_merge_lora.py --bench   # フルベンチ
"""
import argparse
import os
import sys
import time

import numpy as np

try:
    from safetensors.numpy import load_file as _st_load
    from safetensors.numpy import save_file as _st_save
    HAVE_SAFETENSORS = True
except Exception:
    HAVE_SAFETENSORS = False


# safetensors の dtype 文字列 → numpy dtype。マージで扱う範囲のみ。
_DTYPE_TO_NP = {
    "fp16": np.float16,
    "fp32": np.float32,
}


# =====================================================================
# kohya → diffusers key 写像
# =====================================================================

# kohya 平坦名はモジュール階層を全て '_' で繋いでいる。'_' を素朴に '.' へ
# 置換すると `down_blocks` `transformer_blocks` `to_q` `time_embedding` 等の
# モジュール名内アンダースコアまで割れてしまう。
#
# 解は「ドットを除いた base key とマッチさせる」: kohya 平坦名 (lora_unet_ 接頭辞除去)
# は diffusers key からドットを除いた文字列と完全一致する。base の全 key を
# 「ドット除去名 → 元 key」で索けば曖昧さなく復元できる (base が真実源)。
#
# base 無しでも写像できるよう、代表モジュール辞書による構造復元もフォールバックで持つ。

# 末尾の代表サブモジュール (長い順に貪欲マッチ。複合 to_out_0 は to_out.0 へ)。
_LEAF_TOKENS = [
    ("to_out_0", "to_out.0"),
    ("ff_net_0_proj", "ff.net.0.proj"),
    ("ff_net_2", "ff.net.2"),
    ("to_q", "to_q"),
    ("to_k", "to_k"),
    ("to_v", "to_v"),
    ("proj_in", "proj_in"),
    ("proj_out", "proj_out"),
    ("proj", "proj"),
    ("conv1", "conv1"),
    ("conv2", "conv2"),
    ("conv_shortcut", "conv_shortcut"),
    ("conv", "conv"),
    ("time_emb_proj", "time_emb_proj"),
    ("linear_1", "linear_1"),
    ("linear_2", "linear_2"),
]

# 中間の既知複合モジュール名 (アンダースコアを含む語)。長い順。
_KNOWN_WORDS = [
    "transformer_blocks",
    "down_blocks",
    "up_blocks",
    "mid_block",
    "time_embedding",
    "add_embedding",
    "downsamplers",
    "upsamplers",
    "attentions",
    "resnets",
    "norm1",
    "norm2",
    "norm3",
    "attn1",
    "attn2",
]


def kohya_to_diffusers_key(kohya_name):
    """kohya LoRA モジュール名 (lora_unet_ 接頭辞・.lora_down 等の接尾辞は除去済み) を
    diffusers モジュール名 (ドット区切り・末尾 .weight は含まない) へ写像する。

    base 無しの純粋な構造復元。代表モジュールを確実に通す。
    返り値は diffusers モジュール名 (例 down_blocks.0.attentions.1...attn1.to_q)。
    """
    s = kohya_name
    out_tail = ""

    # 末尾の代表 leaf を切り出す
    for tok, rep in _LEAF_TOKENS:
        if s == tok:
            s = ""
            out_tail = rep
            break
        if s.endswith("_" + tok):
            s = s[: -(len(tok) + 1)]
            out_tail = rep
            break

    # 残りを左から既知語 + 数字インデックスで分解
    parts = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == "_":
            i += 1
            continue
        # 数字インデックス
        if s[i].isdigit():
            j = i
            while j < n and s[j].isdigit():
                j += 1
            parts.append(s[i:j])
            i = j
            continue
        # 既知複合語を貪欲に当てる
        matched = None
        for w in _KNOWN_WORDS:
            if s.startswith(w, i) and (i + len(w) == n or s[i + len(w)] == "_"):
                matched = w
                break
        if matched is not None:
            parts.append(matched)
            i += len(matched)
            continue
        # 未知の単純トークン: 次の '_' まで
        j = i
        while j < n and s[j] != "_":
            j += 1
        parts.append(s[i:j])
        i = j

    head = ".".join(parts)
    if head and out_tail:
        return head + "." + out_tail
    return head or out_tail


def build_base_dotless_index(base_keys):
    """base の全 key を「ドット除去名 → 元 key」で索く辞書を返す。

    kohya 平坦名は diffusers key のドット除去名と一致するため、これが最も確実な
    写像経路 (base が真実源)。.weight を落としたモジュール名でも索けるよう両方登録。
    """
    idx = {}
    for k in base_keys:
        dotless = k.replace(".", "_")
        idx[dotless] = k
        # 末尾 .weight を落としたモジュール名でも索けるように
        if k.endswith(".weight"):
            mod = k[: -len(".weight")]
            idx_mod = mod.replace(".", "_")
            idx.setdefault(idx_mod + "__weight__", k)
    return idx


def resolve_base_key(kohya_module, base_dotless_index):
    """kohya モジュール名 → base の diffusers .weight key を解決。

    優先順:
      1) base のドット除去索引で直接一致 (module + '_weight' の平坦名)
      2) 構造復元 kohya_to_diffusers_key + '.weight' が base に存在
    見つからなければ None。
    """
    # 1) ドット除去索引: kohya_module + "_weight" は base key の "<...>.weight" のドット除去
    flat_with_weight = kohya_module + "_weight"
    if flat_with_weight in base_dotless_index:
        return base_dotless_index[flat_with_weight]
    # モジュール名だけでも索ける登録
    if (kohya_module + "__weight__") in base_dotless_index:
        return base_dotless_index[kohya_module + "__weight__"]
    return None


# =====================================================================
# LoRA グルーピング (down/up/alpha を module 単位に束ねる)
# =====================================================================

def group_lora(lora_tensors):
    """LoRA テンソル辞書を module 単位に束ねる。

    返り値:
      unet_mods: {module(kohya, lora_unet_ 除去): {"down":A, "up":B, "alpha":float|None}}
      te_keys:   te1/te2 由来 key のリスト (スキップ対象)
      other_keys: 解釈できない key のリスト
    """
    unet_mods = {}
    te_keys = []
    other_keys = []

    for k, v in lora_tensors.items():
        # テキストエンコーダ LoRA はスコープ外
        if k.startswith("lora_te1_") or k.startswith("lora_te2_") or k.startswith("lora_te_"):
            te_keys.append(k)
            continue
        if not k.startswith("lora_unet_"):
            other_keys.append(k)
            continue

        body = k[len("lora_unet_"):]
        # 接尾辞で down/up/alpha を判定
        if body.endswith(".lora_down.weight"):
            mod = body[: -len(".lora_down.weight")]
            unet_mods.setdefault(mod, {})["down"] = v
        elif body.endswith(".lora_up.weight"):
            mod = body[: -len(".lora_up.weight")]
            unet_mods.setdefault(mod, {})["up"] = v
        elif body.endswith(".alpha"):
            mod = body[: -len(".alpha")]
            unet_mods.setdefault(mod, {})["alpha"] = float(np.asarray(v).reshape(-1)[0])
        elif body.endswith("_lora_down_weight"):
            # 完全平坦命名 (ドットなし) の variant
            mod = body[: -len("_lora_down_weight")]
            unet_mods.setdefault(mod, {})["down"] = v
        elif body.endswith("_lora_up_weight"):
            mod = body[: -len("_lora_up_weight")]
            unet_mods.setdefault(mod, {})["up"] = v
        elif body.endswith("_alpha"):
            mod = body[: -len("_alpha")]
            unet_mods.setdefault(mod, {})["alpha"] = float(np.asarray(v).reshape(-1)[0])
        else:
            other_keys.append(k)

    return unet_mods, te_keys, other_keys


# =====================================================================
# 低ランク積 + 加算 (fp32 計算)
# =====================================================================

def compute_delta(up_B, down_A, scale):
    """delta = scale * (B @ A) を fp32 で計算。2D を返す。"""
    B = np.asarray(up_B, dtype=np.float32)
    A = np.asarray(down_A, dtype=np.float32)
    # B: [out, rank], A: [rank, in]。Conv の場合 4D が来るので 2D へ畳む。
    if B.ndim > 2:
        B = B.reshape(B.shape[0], -1)
    if A.ndim > 2:
        A = A.reshape(A.shape[0], -1)
    delta = (B @ A) * np.float32(scale)
    return delta


def merge_one(base_W, up_B, down_A, scale):
    """base_W (任意 dtype/shape) へ delta を加算し、fp32 結果を返す。

    base_W が 2D (Linear) ならそのまま、4D (Conv) なら delta を reshape して加算。
    B@A の shape が base_W と非整合なら ValueError。
    """
    W = np.asarray(base_W, dtype=np.float32)
    delta = compute_delta(up_B, down_A, scale)

    if W.ndim == 2:
        if delta.shape != W.shape:
            raise ValueError(
                "shape mismatch (Linear): base W {} vs B@A {}".format(W.shape, delta.shape)
            )
        return W + delta
    else:
        # Conv 等: delta を base shape へ reshape (要素数一致が前提)
        if delta.size != W.size:
            raise ValueError(
                "shape mismatch (Conv): base W {} (numel {}) vs B@A numel {}".format(
                    W.shape, W.size, delta.size
                )
            )
        return W + delta.reshape(W.shape)


# =====================================================================
# マージ本体
# =====================================================================

class MergeReport:
    """マージ結果のレポート (テスト・CLI 双方が消費)。"""

    def __init__(self):
        self.merged_modules = []     # マージ成功 (diffusers key)
        self.te_skipped = []         # te1/te2 スキップ key
        self.unmapped = []           # 写像できなかった kohya module
        self.copied_untouched = 0    # LoRA が触れない base key (bitwise コピー)
        self.other_keys = []         # 解釈不能 LoRA key

    def summary(self):
        return (
            "merged={} te_skipped={} unmapped={} untouched_copied={} other={}".format(
                len(self.merged_modules), len(self.te_skipped),
                len(self.unmapped), self.copied_untouched, len(self.other_keys)
            )
        )


def merge_state_dicts(base_tensors, lora_tensors, strength=1.0, out_np_dtype=None):
    """base_tensors (dict name->ndarray) へ lora をマージし、(out_dict, report) を返す。

    out_np_dtype が None なら各 base テンソルの dtype を踏襲する。
    計算は fp32 → 出力 dtype へ round。
    """
    report = MergeReport()
    unet_mods, te_keys, other_keys = group_lora(lora_tensors)
    report.te_skipped = list(te_keys)
    report.other_keys = list(other_keys)

    base_index = build_base_dotless_index(base_tensors.keys())

    # module → 解決した base diffusers key + delta scale
    # まず触れる base key 集合を求める
    touched = {}  # base_key -> (up_B, down_A, scale, kohya_module)
    for mod, parts in unet_mods.items():
        if "down" not in parts or "up" not in parts:
            # down/up 片方欠落は不完全 LoRA → 写像漏れ扱いで報告
            report.unmapped.append("lora_unet_" + mod + " (incomplete: missing down/up)")
            continue

        A = parts["down"]
        B = parts["up"]
        rank = int(np.asarray(A).shape[0])
        alpha = parts.get("alpha", None)
        if alpha is None:
            alpha = float(rank)  # alpha 欠如時は alpha=rank → scale=strength
        scale = float(strength) * (float(alpha) / float(rank))

        base_key = resolve_base_key(mod, base_index)
        if base_key is None:
            # 構造復元フォールバック
            diff_mod = kohya_to_diffusers_key(mod)
            cand = diff_mod + ".weight"
            if cand in base_tensors:
                base_key = cand
            else:
                # base に存在しない diffusers key への写像 → 命名ズレの早期検出 (停止)
                raise KeyError(
                    "LoRA module 'lora_unet_{}' maps to '{}' which is absent in base. "
                    "(naming mismatch — refusing to silently drop)".format(mod, cand)
                )

        touched[base_key] = (B, A, scale, mod)

    # 全 base key を走査して出力を組む
    out = {}
    for name, W in base_tensors.items():
        target_dtype = out_np_dtype if out_np_dtype is not None else np.asarray(W).dtype
        if name in touched:
            B, A, scale, mod = touched[name]
            try:
                merged = merge_one(W, B, A, scale)
            except ValueError as e:
                raise ValueError("module 'lora_unet_{}' -> '{}': {}".format(mod, name, e))
            arr = merged.astype(target_dtype)
            report.merged_modules.append(name)
        else:
            # 触れない key はそのままコピー (dtype 指定があれば従う)
            arr = np.asarray(W)
            if out_np_dtype is not None and arr.dtype != out_np_dtype:
                arr = arr.astype(out_np_dtype)
            else:
                arr = arr.copy()
            report.copied_untouched += 1

        # NaN/Inf チェック (浮動小数のみ)
        if np.issubdtype(arr.dtype, np.floating):
            if not np.all(np.isfinite(arr.astype(np.float32))):
                raise FloatingPointError("non-finite value in merged tensor '{}'".format(name))
        out[name] = np.ascontiguousarray(arr)

    return out, report


# =====================================================================
# CLI
# =====================================================================

def _resolve_out_dtype(arg_dtype, base_first_dtype):
    """--dtype 引数 → numpy dtype。None なら base 追従 (= None を返してテンソル毎踏襲)。"""
    if arg_dtype is None:
        return None
    if arg_dtype not in _DTYPE_TO_NP:
        raise ValueError("--dtype must be fp16 or fp32, got {}".format(arg_dtype))
    return _DTYPE_TO_NP[arg_dtype]


def run_merge_files(base_path, lora_path, out_path, strength=1.0, dtype=None, verbose=True):
    """ファイル入出力のマージ。safetensors ライブラリ必須。(report) を返す。"""
    if not HAVE_SAFETENSORS:
        raise RuntimeError("safetensors ライブラリが必要です (py -3.12 で実行)")
    base = _st_load(base_path)
    lora = _st_load(lora_path)
    out_np_dtype = _resolve_out_dtype(dtype, None)
    out, report = merge_state_dicts(base, lora, strength=strength, out_np_dtype=out_np_dtype)
    _st_save(out, out_path)
    if verbose:
        print("[merge] " + report.summary())
        if report.te_skipped:
            print("[merge] te1/te2 LoRA を {} 件 skip (UNet マージのみ・スコープ外)".format(
                len(report.te_skipped)))
        if report.unmapped:
            print("[merge] 写像漏れ {} 件 (報告のみ・破棄):".format(len(report.unmapped)))
            for u in report.unmapped[:8]:
                print("          " + u)
        print("[merge] 出力 -> " + out_path)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description="dollama L-1 offline LoRA merge")
    ap.add_argument("--base", help="base UNet safetensors (diffusers 命名)")
    ap.add_argument("--lora", help="LoRA safetensors (kohya 命名)")
    ap.add_argument("--out", help="出力 merged safetensors")
    ap.add_argument("--strength", type=float, default=1.0, help="LoRA 適用強度 (既定 1.0)")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default=None,
                    help="出力 dtype (既定: base 追従)")
    ap.add_argument("--bench", action="store_true", help="合成ベンチを実行")
    args = ap.parse_args(argv)

    if args.bench:
        run_bench()
        return 0

    if not (args.base and args.lora and args.out):
        ap.error("--base --lora --out は必須 (--bench を除く)")

    run_merge_files(args.base, args.lora, args.out, strength=args.strength, dtype=args.dtype)
    return 0


# =====================================================================
# 合成ベンチ (本番非汚染・temp 上で実行)
# =====================================================================

def _make_synthetic_base(rng, scale_mb):
    """SDXL UNet 代表規模を模した合成 base。Linear/Conv テンソル群を fp16 で生成。

    scale_mb はおよその目標サイズ (合計バイト) を制御する係数 (テンソル本数を増減)。
    """
    base = {}
    # 代表 Linear: attention to_q/to_k/to_v/to_out + ff、Conv: resnet conv1/conv2。
    # SDXL の主要 Linear は 640/1280/2048 次元帯。本数で総量を稼ぐ。
    n_blocks = max(1, scale_mb)
    for b in range(n_blocks):
        pfx = "down_blocks.{}.attentions.0.transformer_blocks.0".format(b % 3)
        # attn1 self: 1280x1280
        for proj in ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0"]:
            base["{}.{}.weight".format(pfx, proj) + "#{}".format(b)] = \
                rng.standard_normal((1280, 1280)).astype(np.float16)
        # attn2 cross: q 1280x1280, k/v 1280x2048
        base["{}.attn2.to_q.weight#{}".format(pfx, b)] = \
            rng.standard_normal((1280, 1280)).astype(np.float16)
        base["{}.attn2.to_k.weight#{}".format(pfx, b)] = \
            rng.standard_normal((1280, 2048)).astype(np.float16)
        base["{}.attn2.to_v.weight#{}".format(pfx, b)] = \
            rng.standard_normal((1280, 2048)).astype(np.float16)
        # ff: 1280 -> 10240 (geglu proj) はサイズ大なので 1280x5120 に縮小
        base["{}.ff.net.0.proj.weight#{}".format(pfx, b)] = \
            rng.standard_normal((5120, 1280)).astype(np.float16)
        base["{}.ff.net.2.weight#{}".format(pfx, b)] = \
            rng.standard_normal((1280, 2560)).astype(np.float16)
        # Conv (触れない key の bitwise コピー経路も計測対象に含める)
        base["down_blocks.{}.resnets.0.conv1.weight#{}".format(b % 3, b)] = \
            rng.standard_normal((1280, 1280, 3, 3)).astype(np.float16)
    return base


def _make_synthetic_lora_for(base, rank, rng):
    """合成 base の Linear key 群に対応する kohya LoRA を生成 (Conv は触れない)。

    base のキーは合成用に '#n' を付けているため、ここでは別系統で
    「ドット除去 = kohya 平坦名」が一致するよう、base の Linear key から直接生成する。
    実マージ経路 (resolve_base_key) を通すため、'#n' を除いた純 diffusers key を使う。
    返り値: lora dict, および対応付け検証用に touched_base set。
    """
    lora = {}
    touched = set()
    for k in base.keys():
        # '#n' サフィックスを剥がした純 key
        pure = k.split("#")[0]
        if not pure.endswith(".weight"):
            continue
        W = base[k]
        if W.ndim != 2:
            continue  # Conv は LoRA 対象外 (触れない経路)
        out_dim, in_dim = W.shape
        mod = pure[: -len(".weight")].replace(".", "_")  # kohya 平坦名
        lora["lora_unet_{}.lora_down.weight".format(mod)] = \
            (rng.standard_normal((rank, in_dim)) * 0.01).astype(np.float16)
        lora["lora_unet_{}.lora_up.weight".format(mod)] = \
            (rng.standard_normal((out_dim, rank)) * 0.01).astype(np.float16)
        lora["lora_unet_{}.alpha".format(mod)] = np.array(float(rank), dtype=np.float32)
        touched.add(k)
    return lora, touched


def _build_base_index_with_suffix(base):
    """ベンチ用: '#n' サフィックス付き合成 base でも resolve できるよう、
    純 key (#n 除去) のドット除去索引を作る。実 base では不要 (このベンチ専用)。"""
    pure_map = {}
    for k in base.keys():
        pure = k.split("#")[0]
        pure_map[pure] = k
    return pure_map


def run_bench():
    """合成ベンチ: マージ throughput + 5.1GB 外挿 + 単一 Linear 純計算 throughput。"""
    import tempfile

    full = os.environ.get("DOLLAMA_BENCH") == "1"
    rng = np.random.default_rng(20260620)

    print("=== dollma_merge_lora 合成ベンチ ===")
    print("mode:", "FULL (DOLLAMA_BENCH=1)" if full else "SMOKE (set DOLLAMA_BENCH=1 for full)")

    # --- 副指標: 単一 Linear [1280,1280] + rank16 の B@A+加算 純計算 throughput ---
    W = rng.standard_normal((1280, 1280)).astype(np.float16)
    A = (rng.standard_normal((16, 1280)) * 0.01).astype(np.float16)
    B = (rng.standard_normal((1280, 16)) * 0.01).astype(np.float16)
    iters = 200 if full else 20
    # warmup
    for _ in range(3):
        _ = merge_one(W, B, A, 1.0)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = merge_one(W, B, A, 1.0)
    dt = time.perf_counter() - t0
    bytes_per = W.size * 4  # fp32 計算量基準 (W 読み + 書き相当)
    gbps = (bytes_per * iters) / dt / 1e9
    print("[single Linear 1280x1280 rank16] {:.1f} us/op  ~{:.2f} GB/s (fp32 W)".format(
        dt / iters * 1e6, gbps))

    # --- 主指標: 合成 base + LoRA フルマージ throughput ---
    scale_mb = 12 if full else 2  # ブロック数 (FULL でおよそ数百 MB)
    base = _make_synthetic_base(rng, scale_mb)
    total_base_bytes = sum(v.nbytes for v in base.values())

    for rank in ([4, 16, 32] if full else [16]):
        lora, touched = _make_synthetic_lora_for(base, rank, rng)
        pure_index = _build_base_index_with_suffix(base)

        # マージ計測 (in-memory: resolve_base_key を '#n' 索引で迂回する専用ループ)
        t0 = time.perf_counter()
        out = {}
        n_merged = 0
        for name, Wm in base.items():
            pure = name.split("#")[0]
            mod = pure[: -len(".weight")].replace(".", "_") if pure.endswith(".weight") else None
            dk = "lora_unet_{}.lora_down.weight".format(mod) if mod else None
            uk = "lora_unet_{}.lora_up.weight".format(mod) if mod else None
            if mod and dk in lora and uk in lora and np.asarray(Wm).ndim == 2:
                ak = "lora_unet_{}.alpha".format(mod)
                alpha = float(lora[ak]) if ak in lora else float(lora[dk].shape[0])
                sc = (alpha / float(lora[dk].shape[0]))
                out[name] = merge_one(Wm, lora[uk], lora[dk], sc).astype(np.float16)
                n_merged += 1
            else:
                out[name] = np.asarray(Wm).copy()
        dt = time.perf_counter() - t0

        gbps = total_base_bytes / dt / 1e9
        # 5.1GB 外挿
        extrap_s = 5.1e9 / (total_base_bytes / dt)
        print("[merge rank={:>2}] base {:.1f} MB / {} tensors ({} merged) "
              "-> {:.3f}s  {:.2f} GB/s  | 5.1GB 外挿 ~{:.1f}s".format(
                  rank, total_base_bytes / 1e6, len(base), n_merged,
                  dt, gbps, extrap_s))

    # --- 往復 I/O (safetensors save/load) もフルモードのみ計測 ---
    if full and HAVE_SAFETENSORS:
        with tempfile.TemporaryDirectory() as td:
            outp = os.path.join(td, "merged_bench.safetensors")
            t0 = time.perf_counter()
            _st_save({k: np.ascontiguousarray(v) for k, v in base.items()}, outp)
            t_save = time.perf_counter() - t0
            t0 = time.perf_counter()
            _ = _st_load(outp)
            t_load = time.perf_counter() - t0
            io_gbps_save = total_base_bytes / t_save / 1e9
            io_gbps_load = total_base_bytes / t_load / 1e9
            print("[I/O] save {:.3f}s ({:.2f} GB/s) / load {:.3f}s ({:.2f} GB/s) "
                  "| 5.1GB save 外挿 ~{:.1f}s".format(
                      t_save, io_gbps_save, t_load, io_gbps_load,
                      5.1e9 / (total_base_bytes / t_save)))

    print("=== ベンチ完了 (合成データ・本番非汚染) ===")


if __name__ == "__main__":
    sys.exit(main())
