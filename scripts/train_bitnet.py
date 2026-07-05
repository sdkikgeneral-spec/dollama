# -*- coding: utf-8 -*-
"""dollama Phase 4 dense 本線 #4 — タグ生成 LM 訓練スクリプト。

src/models/bitnet.hpp と数値的に等価な dense (FP) モデルを PyTorch で定義し、
data/bitnet/pairs.train.jsonl で hard CE 訓練、重みを safetensors 出力する。

設計の要点 (bitnet.hpp との整合):
  - VOCAB 4999 / D_MODEL 512 / N_LAYERS 8 / N_HEADS 8 / HEAD_DIM 64 / FFN 1792 / MAX_SEQ 64
  - RoPE base 10000 / RMSNorm eps 1e-5 / SwiGLU / causal attention / embed tied
  - RoPE は GPT-NeoX 系ペアリング (i, i+HEAD_DIM/2) → bitnet.hpp::apply_rope と一致
  - #4 は通常 FP Linear で学習 (BitLinear/QAT は入れない・ternary は #5 で後段)
  - 重みレイアウトは row-major [out, in] (nn.Linear.weight と一致)

本スクリプトは hard CE のみ (蒸留なし)。蒸留は次イテレーション。

語彙は data/bitnet/vocab.json を唯一のソースとして読む (二重定義しない)。
正規化・specials id は tokenizer.hpp と同一規則を Python 側で再現する。
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time


# ==============================================================
# 正典アセット搬送 (--copy / --publish) : torch 非依存・コピーのみ
# ==============================================================
# git 管理外の正典アセット (重み4本 / golden 6ファイル / a12k データ2本) を
# マシン間で exact バイトコピーする2方向モード。パスは実行時引数のみ (社内パスを
# repo に残さない)。cross-GPU で重みは bit 非一致ゆえ再生成不可 → バイトコピー必須。
#   --copy <src>    : src(リモート base) -> ローカル --data-dir へ取得。
#   --publish <dst> : ローカル --data-dir -> dst(リモート base) へ配置。
# torch より前 (下の import torch より上) で _maybe_transport() を呼び、指定時は
# ここで sys.exit(0) する。未指定時は即 return → 既存経路の torch import タイミング不変。
# vocab.json / pairs.val / pairs.train.diverse_b2000 / pairs.eval_diverse_{a,b} は
# git 追跡済ゆえ搬送対象外 (git pull で届く)。
_TRANSPORT_FIXED = [
    "bitnet_dense.safetensors",
    "bitnet_dense_fp32.safetensors",
    "bitnet_dense_identity.safetensors",
    "bitnet_dense_identity_fp32.safetensors",
    "pairs.identity.train.a12k.jsonl",
    "pairs.identity.val.a12k.jsonl",
]


def _transport_sha256(path):
    """ファイルの sha256 hex (torch 非依存・stdlib のみ)。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _transport_targets(source_base):
    """搬送対象の相対パス一覧。golden/ はハードコードせず source 側の実在ファイルを列挙
    (logits_golden/gen_golden/manifest の base/_identity 版・版差異/欠損に頑健)。"""
    rels = list(_TRANSPORT_FIXED)
    gdir = os.path.join(source_base, "golden")
    if os.path.isdir(gdir):
        for name in sorted(os.listdir(gdir)):
            if os.path.isfile(os.path.join(gdir, name)):
                rels.append(os.path.join("golden", name))
    return rels


def _transport_copy_one(src, dst):
    """src->dst を exact コピー。sha256+size 照合・冪等 (同一 skip)・差分/上書きログ。
    返り値 (status, ok_bool)。silent 上書きしない (旧値をログに出す)。"""
    if not os.path.exists(src):
        return ("MISSING_SRC", None)
    ssha = _transport_sha256(src)
    ssize = os.path.getsize(src)
    prefix = "NEW"
    old = ""
    if os.path.exists(dst):
        dsha = _transport_sha256(dst)
        dsize = os.path.getsize(dst)
        if dsha == ssha and dsize == ssize:
            return (f"SKIP_SAME sha={ssha[:12]} size={ssize}", True)
        prefix = "OVERWRITE"
        old = f" old(sha={dsha[:12]} size={dsize}) ->"
    d = os.path.dirname(dst)
    if d:
        os.makedirs(d, exist_ok=True)
    shutil.copyfile(src, dst)
    vsha = _transport_sha256(dst)
    vsize = os.path.getsize(dst)
    ok = (vsha == ssha and vsize == ssize)
    tag = "OK" if ok else "VERIFY_FAIL"
    return (f"{prefix} {tag}{old} new(sha={ssha[:12]} size={ssize})", ok)


def _run_transport(mode, remote_base, data_dir):
    """mode='copy' (remote_base->data_dir) / 'publish' (data_dir->remote_base)。torch 非依存。
    golden 列挙は常に source 側 (copy=remote / publish=local) の実在ファイルに基づく。"""
    if mode == "copy":
        source_base, dest_base = remote_base, data_dir
    else:
        source_base, dest_base = data_dir, remote_base
    print(f"[transport] mode={mode} source={os.path.abspath(source_base)} "
          f"dest={os.path.abspath(dest_base)}")
    if not os.path.isdir(source_base):
        raise SystemExit(f"[transport] source ディレクトリが無い: {source_base}")
    os.makedirs(dest_base, exist_ok=True)
    rels = _transport_targets(source_base)
    n_ok = n_skip = n_miss = n_fail = 0
    for rel in rels:
        status, ok = _transport_copy_one(os.path.join(source_base, rel),
                                         os.path.join(dest_base, rel))
        if status == "MISSING_SRC":
            n_miss += 1
            print(f"[transport] WARN skip (source 不在): {rel}")
            continue
        if status.startswith("SKIP_SAME"):
            n_skip += 1
        elif ok:
            n_ok += 1
        else:
            n_fail += 1
        print(f"[transport] {rel}: {status}")
    print(f"[transport] done: copied={n_ok} skipped_same={n_skip} "
          f"missing={n_miss} verify_fail={n_fail} (total_targets={len(rels)})")
    if n_fail:
        raise SystemExit(f"[transport] VERIFY_FAIL {n_fail} 件 (sha/size 不一致)")


def _maybe_transport():
    """torch import より前に呼ぶ。--copy/--publish 指定時のみコピーを実行し sys.exit(0)。
    未指定なら何もせず return (stdlib のみ・torch 非依存・既存経路の import タイミング不変)。"""
    argv = sys.argv[1:]

    def _get(name):
        for i, a in enumerate(argv):
            if a == name:
                return argv[i + 1] if i + 1 < len(argv) else ""
            if a.startswith(name + "="):
                return a[len(name) + 1:]
        return None

    copy_src = _get("--copy")
    publish_dst = _get("--publish")
    if copy_src is None and publish_dst is None:
        return  # 通常経路: 何もしない (この下の import torch へ進む)
    if copy_src is not None and publish_dst is not None:
        raise SystemExit("[transport] --copy と --publish は同時指定不可")
    data_dir = _get("--data-dir") or "data/bitnet"
    if copy_src is not None:
        if not copy_src:
            raise SystemExit("[transport] --copy <src> にパスが必要")
        _run_transport("copy", copy_src, data_dir)
    else:
        if not publish_dst:
            raise SystemExit("[transport] --publish <dst> にパスが必要")
        _run_transport("publish", publish_dst, data_dir)
    sys.exit(0)


# --copy/--publish 指定時はここで搬送して終了 (torch を import しない)。
# 未指定時は即 return し、以降の torch import は現行と同一タイミング。
_maybe_transport()

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================
# アーキ定数 (bitnet.hpp と厳密一致)
# ==============================================================
# 既定 (base = 現行 #1 本線) は src/models/bitnet.hpp::BitNet の constexpr と完全一致。
# 施策 D (容量増) のため n_layers / ffn のみ伸ばした d80m config を選べるが、
# **既定 (引数なし) は base のまま** = 現行と bitwise 非回帰。
# モジュールレベル定数は下の apply_arch() が config に応じて再束縛する
# (モデル定義・golden dump・export・sanity が全て call 時にこの globals を読むため、
#  apply_arch を model 構築前に呼ぶだけで全経路が config 連動になる)。
VOCAB_SIZE = 4999
D_MODEL = 512
N_LAYERS = 8
N_HEADS = 8
HEAD_DIM = D_MODEL // N_HEADS  # 64
FFN_DIM = 1792
MAX_SEQ_LEN = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5

# --------------------------------------------------------------
# アーキ config テーブル (権威スペック・C++ bitnet.hpp と完全一致させること)
# --------------------------------------------------------------
#   base : 現行 #1 本線。param 合計 = 32,976,896 (bitnet.hpp::param_count() 実績値)。
#   d80m : 施策 D 容量増。numel 合計 = 79,908,864 (同式・同値・実装後 assert で担保)。
# vocab / max_seq / d_model / n_heads は据え置き (tokenizer / 全 golden / カーネル前提の
# 連鎖崩壊を避ける)。伸ばすのは n_layers と ffn のみ。
ARCH_CONFIGS = {
    "base": {"d_model": 512, "n_layers": 8, "ffn": 1792},
    "d80m": {"d_model": 512, "n_layers": 16, "ffn": 2464},
}


def compute_param_count(vocab=None, d_model=None, n_layers=None,
                        n_heads=None, ffn=None):
    """C++ bitnet.hpp::param_count() と同式で numel 合計を求める (embed tied)。

      embed (tied)           : vocab * d_model
      per layer attn q/k/v/o : 4 * d_model * d_model
      per layer ffn g/u/d    : 3 * ffn * d_model
      per layer rmsnorm x2   : 2 * d_model
      final rmsnorm          : d_model
    embed tied のため lm_head は別カウントしない (embed を 1 回だけ数える)。
    引数省略時は現在のモジュール定数を使う。
    """
    vocab = VOCAB_SIZE if vocab is None else vocab
    d_model = D_MODEL if d_model is None else d_model
    n_layers = N_LAYERS if n_layers is None else n_layers
    ffn = FFN_DIM if ffn is None else ffn
    embed = vocab * d_model
    per_attn = 4 * d_model * d_model
    per_ffn = 3 * ffn * d_model
    per_norm = 2 * d_model
    per_layer = per_attn + per_ffn + per_norm
    final_norm = d_model
    return embed + per_layer * n_layers + final_norm


def apply_arch(arch=None, d_model=None, n_layers=None, ffn=None):
    """アーキ次元をモジュール定数へ再束縛する (model 構築前に 1 回呼ぶ)。

    優先順位: arch config を土台に、個別上書き (d_model/n_layers/ffn) があれば被せる。
    arch も個別上書きも None なら **既定 base を維持** (= 現行と完全一致・非回帰)。
    HEAD_DIM は d_model/n_heads から再計算する (現状 n_heads 据え置きなので 64 のまま)。
    vocab / max_seq / n_heads / rope_base / rms_eps は据え置き (config に含めない)。

    返り値: 適用後の dict (ログ用)。
    """
    global D_MODEL, N_LAYERS, FFN_DIM, HEAD_DIM
    # 1) base または指定 config を土台に取る。
    if arch is not None:
        if arch not in ARCH_CONFIGS:
            raise ValueError(
                f"unknown arch '{arch}' (choices: {sorted(ARCH_CONFIGS)})")
        cfg = dict(ARCH_CONFIGS[arch])
    else:
        # arch 未指定: 現行モジュール定数を土台 (= base 既定)。
        cfg = {"d_model": D_MODEL, "n_layers": N_LAYERS, "ffn": FFN_DIM}
    # 2) 個別上書き (指定があれば被せる)。
    if d_model is not None:
        cfg["d_model"] = d_model
    if n_layers is not None:
        cfg["n_layers"] = n_layers
    if ffn is not None:
        cfg["ffn"] = ffn
    # 3) 制約チェック (n_heads 据え置きなので d_model は n_heads で割り切れること)。
    if cfg["d_model"] % N_HEADS != 0:
        raise ValueError(
            f"d_model={cfg['d_model']} が n_heads={N_HEADS} で割り切れない")
    # 4) モジュール定数へ反映。
    D_MODEL = cfg["d_model"]
    N_LAYERS = cfg["n_layers"]
    FFN_DIM = cfg["ffn"]
    HEAD_DIM = D_MODEL // N_HEADS
    return {
        "vocab": VOCAB_SIZE, "d_model": D_MODEL, "n_layers": N_LAYERS,
        "n_heads": N_HEADS, "head_dim": HEAD_DIM, "ffn": FFN_DIM,
        "max_seq": MAX_SEQ_LEN, "param_count": compute_param_count(),
    }


# specials id (tokenizer.hpp TokenId と一致)
TOK_PAD = 0
TOK_BOS = 1
TOK_EOS = 2
TOK_SEP = 3
TOK_UNK = 4


# ==============================================================
# トークナイザ (vocab.json 駆動・tokenizer.hpp と同一規則)
# ==============================================================
class Tokenizer:
    """vocab.json を唯一のソースに読む完全一致タグトークナイザ。

    tokenizer.hpp の normalize / specials id / tags[i].id==5+i 規約を再現する。
    """

    def __init__(self, vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vj = json.load(f)
        specials = vj["specials"]
        if specials != ["<pad>", "<bos>", "<eos>", "<sep>", "<unk>"]:
            raise ValueError(f"specials mismatch: {specials}")
        self.specials = specials
        # tags[i].id == 5+i 連番検証 (tokenizer.hpp validate と同じ)
        self.tags = []
        for i, t in enumerate(vj["tags"]):
            expect_id = 5 + i
            if t["id"] != expect_id:
                raise ValueError(f"tags[{i}].id={t['id']} expected {expect_id}")
            self.tags.append(t["tag"])
        total = len(self.specials) + len(self.tags)
        if total != VOCAB_SIZE:
            raise ValueError(f"total vocab {total} != {VOCAB_SIZE}")
        # 正規化済み tag -> id 逆引き
        self.tag_to_id = {}
        for i, t in enumerate(self.tags):
            self.tag_to_id[self.normalize(t)] = 5 + i

    @staticmethod
    def normalize(s):
        """英数字に挟まれた '_' のみスペースへ。顔文字の '_' は保持。

        tokenizer.hpp::normalize と同一ロジック。
        """
        out = []
        n = len(s)
        for i, c in enumerate(s):
            if c == "_":
                prev_alnum = i > 0 and s[i - 1].isalnum()
                next_alnum = i + 1 < n and s[i + 1].isalnum()
                if prev_alnum and next_alnum:
                    out.append(" ")
                    continue
            out.append(c)
        return "".join(out)

    def tag_to_id_lookup(self, tag):
        """単一タグ -> id。未知なら <unk>。"""
        return self.tag_to_id.get(self.normalize(tag), TOK_UNK)

    def vocab_size(self):
        return len(self.specials) + len(self.tags)


# ==============================================================
# モデル定義 (bitnet.hpp dense 等価)
# ==============================================================
def build_rope_cache(seq_len, head_dim, base, device, dtype=torch.float32):
    """NeoX 系 RoPE の cos/sin キャッシュを作る。

    bitnet.hpp::apply_rope と同じ周波数:
      freq_i = base^(-(2i)/head_dim),  i in [0, head_dim/2)
      angle  = pos * freq_i
    返す cos/sin は [seq_len, head_dim/2]。
    """
    half = head_dim // 2
    i = torch.arange(half, device=device, dtype=dtype)
    freqs = base ** (-(2.0 * i) / head_dim)          # [half]
    pos = torch.arange(seq_len, device=device, dtype=dtype)  # [S]
    angles = torch.outer(pos, freqs)                 # [S, half]
    return torch.cos(angles), torch.sin(angles)      # 各 [S, half]


def apply_rope(x, cos, sin):
    """NeoX 系 RoPE を適用する。

    x:   [B, H, S, head_dim]
    cos/sin: [S, head_dim/2]
    bitnet.hpp::apply_rope の (i, i+half) ペア回転:
      out[i]      = a*cos - b*sin
      out[i+half] = a*sin + b*cos   (a=x[i], b=x[i+half])
    """
    half = x.shape[-1] // 2
    a = x[..., :half]   # [B,H,S,half]
    b = x[..., half:]
    cos = cos[None, None, :, :]  # [1,1,S,half]
    sin = sin[None, None, :, :]
    out_a = a * cos - b * sin
    out_b = a * sin + b * cos
    return torch.cat([out_a, out_b], dim=-1)


class RMSNorm(nn.Module):
    """bitnet.hpp::rms_norm 等価 (重み乗算あり・bias なし)。"""

    def __init__(self, dim, eps=RMS_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # FP32 で mean(x^2) を取る (参照実装が double 蓄積のため精度寄せ)
        dtype = x.dtype
        xf = x.float()
        ms = xf.pow(2).mean(dim=-1, keepdim=True)
        xn = xf * torch.rsqrt(ms + self.eps)
        return (xn.to(dtype)) * self.weight


class Block(nn.Module):
    """1 層: pre-RMSNorm attention + pre-RMSNorm SwiGLU FFN。bias なし。

    dropout (D5 / training-spec §11) は **訓練時正則化専用**。残差に足す attention 出力 /
    FFN 出力に置く (LLaMA 系 residual dropout)。eval 時 (model.eval()) は nn.Dropout が
    恒等になるため **forward は dropout=0 と完全一致** = #4/#6/A3 golden 非回帰。
    既定 dropout=0.0 なら訓練時も恒等 → 従来挙動と完全一致。
    """

    def __init__(self, dropout=0.0):
        super().__init__()
        self.attn_norm = RMSNorm(D_MODEL)
        self.wq = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.wk = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.wv = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.wo = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.ffn_norm = RMSNorm(D_MODEL)
        self.w_gate = nn.Linear(D_MODEL, FFN_DIM, bias=False)
        self.w_up = nn.Linear(D_MODEL, FFN_DIM, bias=False)
        self.w_down = nn.Linear(FFN_DIM, D_MODEL, bias=False)
        # dropout=0.0 のとき nn.Dropout は恒等 (train/eval とも) → 従来と完全一致。
        self.attn_drop = nn.Dropout(dropout)
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, h, cos, sin, attn_mask):
        B, S, _ = h.shape
        # --- attention ---
        x = self.attn_norm(h)
        q = self.wq(x).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)  # [B,H,S,Dh]
        k = self.wk(x).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.wv(x).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        scale = 1.0 / math.sqrt(HEAD_DIM)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B,H,S,S]
        scores = scores + attn_mask  # causal mask (上三角 -inf)
        probs = F.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, v)  # [B,H,S,Dh]
        ctx = ctx.transpose(1, 2).contiguous().view(B, S, D_MODEL)
        h = h + self.attn_drop(self.wo(ctx))
        # --- FFN (SwiGLU) ---
        x = self.ffn_norm(h)
        inter = F.silu(self.w_gate(x)) * self.w_up(x)
        h = h + self.ffn_drop(self.w_down(inter))
        return h


class BitNetDense(nn.Module):
    """bitnet.hpp の dense (FP) 等価モデル。embed tied lm_head。

    dropout は訓練時正則化専用 (D5 / training-spec §11・§1)。eval 時は恒等になり
    forward は dropout=0 と完全一致 → golden 非回帰。既定 0.0 で従来挙動。
    """

    def __init__(self, dropout=0.0):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL)
        # embed 直後の residual dropout (dropout=0.0 なら恒等)。
        self.embed_drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([Block(dropout) for _ in range(N_LAYERS)])
        self.final_norm = RMSNorm(D_MODEL)
        self._rope_cache = {}
        self._init_weights()

    def _init_weights(self):
        """bitnet.hpp::init_random と同趣旨の小さめ初期化。

        embed/Linear は N(0, 0.02)、norm は 1.0。embed tied で lm_head を兼ねるため
        標準の Embedding N(0,1) 初期化だとロジットが巨大になり初期 CE が log(V) を
        大きく超える (smoke で ~450)。0.02 std で初期 CE を ~log(4999)=8.5 近傍に置く。
        """
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, RMSNorm):
                nn.init.ones_(m.weight)

    def _rope(self, S, device):
        key = (S, device)
        if key not in self._rope_cache:
            self._rope_cache[key] = build_rope_cache(S, HEAD_DIM, ROPE_BASE, device)
        return self._rope_cache[key]

    def forward(self, tokens):
        # tokens: [B, S] long
        B, S = tokens.shape
        h = self.embed_drop(self.embed(tokens))  # [B,S,D] (dropout=0.0/eval で恒等)
        cos, sin = self._rope(S, tokens.device)
        # causal mask [1,1,S,S]: j>i を -inf
        mask = torch.full((S, S), float("-inf"), device=tokens.device)
        mask = torch.triu(mask, diagonal=1)[None, None, :, :]
        for layer in self.layers:
            h = layer(h, cos, sin, mask)
        h = self.final_norm(h)
        # tied lm_head: logits = h @ embed.weight^T
        logits = F.linear(h, self.embed.weight)  # [B,S,VOCAB]
        return logits


# ==============================================================
# データ整形
# ==============================================================
def load_pairs(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_sequence(tok, row, max_len):
    """1 ペアを自己回帰列に整形する (source で分岐・dataset-spec §13.1)。

    - source=="synthetic" (#1 既存形式・1-<sep>):
        <bos> text(タグ化) <sep> tags <eos>
        loss target 区間 = tags (1 個目 <sep> の次以降)。
    - source=="identity_cond" (A1 形式・2-<sep>):
        <bos> [identity tags] <sep> [scene text の greedy タグ] <sep> [target tags] <eos>
        loss target 区間 = target (2 個目 <sep> の次以降)。

    両形式とも「target 区間のみ CE」原則で揃う (tags_start = 最後の構造区切りの次)。

    トークン化:
      - text / scene text 側: tokenizer.hpp::encode_text と同じ greedy 最長一致
        (正規化 → 英数字連結語へ分割 → 最長連結タグを貪欲一致・非語彙語スキップ)。
        identity_cond の scene 区間も自然文 text の greedy タグ列とする。推論時 prompt は
        自然文しか来ないため、scene 条件も自然文由来トークンで揃える (§13: scene 区間 =
        scene text を greedy 一致したタグ列)。
      - identity / tags 側: 正準順序の id 列を tag_to_id で直接 id 化 (未知は <unk>)。

    返り値: (ids[list], tags_start_index)  tags_start_index は loss を取り始める
    ids-position (= 最後の <sep> の次)。
    """
    source = row.get("source", "synthetic")
    if source == "identity_cond":
        meta = row.get("meta", {})
        identity_tags = meta.get("identity_tags")
        if identity_tags is None or "scene_tags" not in meta:
            # meta が欠ける identity_cond は仕様違反 (A1 で必ず付く・§13.4)。
            raise ValueError(
                "source=identity_cond だが meta.identity_tags/scene_tags が無い "
                f"(post_id={meta.get('post_id')})")
        # --- identity 区間: identity_tags をそのまま id 化 (§13.3 identity ⊆ target) ---
        identity_ids = [tok.tag_to_id_lookup(t) for t in identity_tags]
        # --- scene 区間: 自然文 text を greedy 最長一致でタグ化 (§13 / A1 の意図) ---
        scene_ids = encode_text_greedy(tok, row["text"])
        # --- target 区間: 正準順序 target をそのまま id 化 ---
        target_ids = [tok.tag_to_id_lookup(t) for t in row["tags"]]

        ids = ([TOK_BOS] + identity_ids + [TOK_SEP]
               + scene_ids + [TOK_SEP] + target_ids + [TOK_EOS])
        # 2 個目 <sep> の位置 = 1 + len(identity) + 1 + len(scene)
        sep2_pos = 1 + len(identity_ids) + 1 + len(scene_ids)
        tags_start = sep2_pos + 1  # target 部の先頭位置
    else:
        # --- synthetic (#1): 従来 1-<sep> 形式 ---
        text_ids = encode_text_greedy(tok, row["text"])
        tag_ids = [tok.tag_to_id_lookup(t) for t in row["tags"]]
        ids = [TOK_BOS] + text_ids + [TOK_SEP] + tag_ids + [TOK_EOS]
        sep_pos = 1 + len(text_ids)  # <sep> の位置
        tags_start = sep_pos + 1     # tags 部の先頭位置

    if len(ids) > max_len:
        ids = ids[:max_len]
        ids[-1] = TOK_EOS
        # 切り詰めで tags_start が列外に出ると loss 区間が空になるが、
        # collate 側の start_t clamp で安全 (CE が全 -100 → 当該サンプル寄与 0)。
    return ids, tags_start


# 最長一致用: 語彙中の最長タグ単語数を求めるためのキャッシュ
_MAX_TAG_WORDS = None


def encode_text_greedy(tok, text):
    """tokenizer.hpp::encode_text と同じ greedy 最長一致 (bos/eos なし・本体 id のみ)。

    正規化 → 英数字連結の単語へ分割 → wi から最長連結タグを貪欲一致。
    非語彙単語はスキップ。
    """
    global _MAX_TAG_WORDS
    if _MAX_TAG_WORDS is None:
        m = 1
        for t in tok.tag_to_id.keys():
            wc = len(re.findall(r"[0-9A-Za-z]+", t))
            if wc > m:
                m = wc
        _MAX_TAG_WORDS = m
    norm = tok.normalize(text)
    # 英数字連結の単語 [begin,end)
    words = [(m.start(), m.end()) for m in re.finditer(r"[0-9A-Za-z]+", norm)]
    out = []
    wi = 0
    n = len(words)
    while wi < n:
        matched_id = None
        matched_words = 0
        max_j = min(n - wi, _MAX_TAG_WORDS)
        for j in range(max_j, 0, -1):
            b = words[wi][0]
            e = words[wi + j - 1][1]
            cand = norm[b:e]
            tid = tok.tag_to_id.get(cand)
            if tid is not None:
                matched_id = tid
                matched_words = j
                break
        if matched_words > 0:
            out.append(matched_id)
            wi += matched_words
        else:
            wi += 1
    return out


class PairDataset:
    """整形済みシーケンスを保持する単純なデータセット。

    samples[i] = (ids, tags_start, source, row)。
      - source: "synthetic" / "identity_cond" (混合訓練の source 別集計・retention 用)。
      - row: 元 JSON 行 (retention 計測で meta.identity_tags / text を参照する)。
    既存呼び出しは (ids, tags_start) の 2 要素だけを使うので、collate / eval は
    タプル先頭 2 要素のみ取り出す (後方互換)。
    """

    def __init__(self, tok, rows, max_len):
        self.samples = []
        for r in rows:
            ids, tags_start = build_sequence(tok, r, max_len)
            source = r.get("source", "synthetic")
            self.samples.append((ids, tags_start, source, r))

    def __len__(self):
        return len(self.samples)


def collate(batch, max_len, loss_mode):
    """バッチを [B, L] にパディングし、CE 用 input/target/loss_mask を作る。

    自己回帰: input = ids[:-1], target = ids[1:]。
    loss_mode:
      "tags"  : tags 部 (tags_start 以降の target) のみで loss。
      "all"   : pad を除く全位置で loss。
    pad 位置は ignore_index で除外する。
    """
    B = len(batch)
    L = max(len(s[0]) for s in batch)
    L = min(L, max_len)
    inp = torch.full((B, L - 1), TOK_PAD, dtype=torch.long)
    tgt = torch.full((B, L - 1), -100, dtype=torch.long)  # ignore_index=-100
    for bi, sample in enumerate(batch):
        ids, tags_start = sample[0], sample[1]
        ids = ids[:L]
        seq = torch.tensor(ids, dtype=torch.long)
        n = len(ids)
        inp[bi, : n - 1] = seq[:-1]
        # target は次トークン予測
        full_tgt = seq[1:]  # 長さ n-1、位置 t は ids[t+1] を予測
        if loss_mode == "tags":
            # ids 上の position p (>= tags_start) を予測する箇所のみ残す。
            # target 配列 index t は ids[t+1] を予測 → ids-position = t+1。
            # よって t+1 >= tags_start すなわち t >= tags_start-1 を残す。
            start_t = max(tags_start - 1, 0)
            full_tgt = full_tgt.clone()
            full_tgt[:start_t] = -100
        tgt[bi, : n - 1] = full_tgt
    return inp, tgt


def make_batches(dataset, batch_size, shuffle, rng):
    idx = list(range(len(dataset)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        chunk = idx[i : i + batch_size]
        yield [dataset.samples[j] for j in chunk]


# ==============================================================
# D5: 案A 共起 soft-label teacher (training-spec §11 / dataset-spec §12.2)
# ==============================================================
#
# 動機 (D2/D4 の負の結果を踏まえて):
#   D2/D4 の hard CE 混合蒸留は「入力側データ拡張」に過ぎず student がソフト知識を
#   一切見ていないのが欠陥だった (§10.4)。D5 は target を hard one-hot のままにせず、
#   **各 target 位置の正解分布そのものを軟化** する soft-label KL 蒸留。
#
# teacher (案A):
#   新しいペアは作らない。既存 cache/danbooru_posts.jsonl の **タグ共起経験分布** から
#   teacher を構築する。各 target 位置で gold 次タグ g を予測するとき:
#     - 主質量 main_mass を g に置く。
#     - 残余 (1-main_mass) を「現プレフィクス (= その系列で既に出た target タグ集合) と
#       corpus 上で共起するタグ」へ温度 T_cooc 付きで配る。
#   = 意味づけされた (共起で裏打ちされた) ラベルスムージング。プレフィクス条件付きなので
#     単なる一様スムージングより情報量が高い。
#
# 共起テーブル: post 内タグ集合の **無向ペア共起回数** を vocab id 空間で集計
#   (cooc[a][b] = a と b が同一 post に共起した回数)。on-the-fly 構築 (キャッシュは
#   .npz に保存して 2 回目以降は再利用)。新規ペアファイルは作らない。
class CoOccurrenceTeacher:
    """danbooru 共起から prefix 条件付き soft target を作る案A teacher。

    パラメータ (train_stats_kl.json に記録):
      main_mass  : gold 次タグに置く主質量 (残余 = 1-main_mass を共起候補へ)。
      t_cooc     : 共起スコア softmax の温度 (大きいほど平坦)。
      topn       : 残余を配る共起候補の上位件数 (プレフィクス和スコア上位)。
    """

    def __init__(self, tok, posts_path, main_mass=0.85, t_cooc=2.0, topn=32):
        self.tok = tok
        self.main_mass = float(main_mass)
        self.t_cooc = float(t_cooc)
        self.topn = int(topn)
        self.posts_path = posts_path
        # cooc[a] = {b: count} (a,b は vocab id・本体タグのみ 5..VOCAB-1)。
        self.cooc = {}
        self._build()

    def _build(self):
        import collections
        npz = self.posts_path + ".cooc.npz"
        # npz キャッシュ (再現的・軽量)。存在すれば読む。
        if os.path.exists(npz):
            try:
                import numpy as np
                z = np.load(npz)  # int32 配列のみ (pickle 不要)
                rows = z["rows"]; cols = z["cols"]; cnts = z["cnts"]
                for a, b, c in zip(rows.tolist(), cols.tolist(), cnts.tolist()):
                    self.cooc.setdefault(a, {})[b] = c
                print(f"[teacher] cooc cache 読込 {npz} (entries={len(rows)})")
                return
            except Exception as e:
                print(f"[teacher] cooc cache 読込失敗 ({e}) → 再構築")
                self.cooc = {}
        n_posts = 0
        with open(self.posts_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                raw = obj.get("tags_general", "")
                # post 内タグを vocab id 化 (空白区切り・<unk> は除外)。
                ids = set()
                for t in raw.split():
                    tid = self.tok.tag_to_id_lookup(t)
                    if tid >= 5:  # 本体タグのみ (specials/unk 除外)
                        ids.add(tid)
                ids = sorted(ids)
                for i in range(len(ids)):
                    a = ids[i]
                    da = self.cooc.setdefault(a, {})
                    for j in range(len(ids)):
                        if i == j:
                            continue
                        b = ids[j]
                        da[b] = da.get(b, 0) + 1
                n_posts += 1
        print(f"[teacher] cooc 構築: posts={n_posts} 中心タグ数={len(self.cooc)}")
        # npz 保存 (COO 形式)。
        try:
            import numpy as np
            rows = []; cols = []; cnts = []
            for a, d in self.cooc.items():
                for b, c in d.items():
                    rows.append(a); cols.append(b); cnts.append(c)
            np.savez_compressed(npz, rows=np.array(rows, dtype=np.int32),
                                cols=np.array(cols, dtype=np.int32),
                                cnts=np.array(cnts, dtype=np.int32))
            print(f"[teacher] cooc cache 保存 {npz} (entries={len(rows)})")
        except Exception as e:
            print(f"[teacher] cooc cache 保存スキップ ({e})")

    def soft_target(self, prefix_ids, gold_id):
        """1 target 位置の soft 分布 (dict {id: prob}) を返す。

        prefix_ids : その系列で gold より前に出た target タグ id 列 (本体タグのみ有効)。
        gold_id    : 正解次タグ id。
        分布: gold に main_mass、残余を prefix と共起するタグ上位 topn へ温度付きで配る。
          共起候補が無い (prefix 空 or 共起ゼロ) 場合は gold に全質量 (= hard と同じ)。
        """
        import math as _m
        if gold_id < 5:
            # specials (<eos>/<sep> 等) を予測する位置: 軟化しない (構造トークンは hard)。
            return {gold_id: 1.0}
        # prefix 本体タグの共起和スコアを集める。
        score = {}
        for p in prefix_ids:
            if p < 5:
                continue
            d = self.cooc.get(p)
            if not d:
                continue
            for b, c in d.items():
                if b == gold_id or b < 5:
                    continue
                score[b] = score.get(b, 0.0) + c
        if not score:
            return {gold_id: 1.0}
        # 上位 topn を温度付き softmax (log スケールで安定化)。
        items = sorted(score.items(), key=lambda kv: kv[1], reverse=True)[: self.topn]
        logs = [(_m.log(s + 1.0) / self.t_cooc) for _, s in items]
        mx = max(logs)
        exps = [_m.exp(l - mx) for l in logs]
        Z = sum(exps)
        resid = 1.0 - self.main_mass
        out = {gold_id: self.main_mass}
        for (b, _), e in zip(items, exps):
            out[b] = out.get(b, 0.0) + resid * (e / Z)
        return out


# ==============================================================
# D6: 外部教師 (TIPO-200M) soft target を npz から引く teacher (案c)
# ==============================================================
#
# 動機 (D5 を踏まえて):
#   D5 (案A 共起 teacher) は過学習を抑制したが top10 recall を底上げしなかった
#   (soft 追従と one-hot val recall のトレードオフ・training-spec 11)。D6 は
#   **真の外部知識転移** を狙い、TIPO-200M が学習した「条件付きタグ集合」を
#   自作 4999 vocab の疎 soft target として注入する (案b-tagset)。
#
# 設計 (KL plumbing は無改修流用):
#   teacher 本体 (transformers/CUDA) は訓練時にロードしない。事前に
#   scripts/dollma_d6_teacher_cache.py が出力した COO npz
#     (rows=sample_idx, poss=target_index_t, cols=vocab_id, probs)
#   を読むだけ。各 (sample_idx, t) の soft 分布は D5 と同様 gold に main_mass・
#   残余を TIPO 生成由来タグへ温度付きで配ったもの (和=1.0)。
#
#   インターフェースは CoOccurrenceTeacher と同一の soft_target(prefix_ids, gold_id)。
#   precompute_soft_targets / build_soft_target_tensor を一切変えずに使えるよう、
#   bind(dataset) で各 sample の soft 位置数を先に把握し、compute_sample_soft の
#   決定的反復順序 (sample を 0,1,2,... 順・各 sample 内は t 昇順に 1 回ずつ呼ぶ) に
#   同期して npz の (sample_idx, t) 分布を順送りで返す。soft 位置 0 の sample は
#   呼び出されないので自動でスキップする (順送りポインタが破綻しない)。
#   gold_id<5 (specials) は D5 同様 hard ({gold:1.0})・npz には soft 位置を持たない。
class ExternalTagTeacher:
    """D6: dollma_d6_teacher_cache.py の COO npz を読み、CoOccurrenceTeacher と同一
    I/F で疎 soft 分布を返す teacher。訓練時に TIPO 本体はロードしない (npz を読むだけ)。

    使い方 (CoOccurrenceTeacher と完全に同じ + bind を 1 行足すだけ):
        teacher = ExternalTagTeacher(tok, soft_npz_path)
        teacher.bind(dataset, max_len)            # 順送りの基準 (各 sample の呼数) を確定
        soft_cache = precompute_soft_targets(teacher, dataset, max_len)
        soft = build_soft_target_tensor(teacher, batch, tgt, max_len, soft_cache)
    """

    def __init__(self, tok, soft_npz_path):
        import numpy as np
        self.tok = tok
        self.soft_npz_path = soft_npz_path
        if not os.path.exists(soft_npz_path):
            raise FileNotFoundError(
                f"{soft_npz_path} が無い (D6 外部教師 soft npz)。"
                f"scripts/dollma_d6_teacher_cache.py で生成する。")
        z = np.load(soft_npz_path)
        rows = z["rows"]; poss = z["poss"]; cols = z["cols"]; probs = z["probs"]
        # by_sample[sample_idx] = list of (t, {vocab_id: prob})  (t 昇順)。
        tmp = {}
        for r, p, c, pr in zip(rows.tolist(), poss.tolist(),
                               cols.tolist(), probs.tolist()):
            tmp.setdefault(int(r), {}).setdefault(int(p), {})[int(c)] = float(pr)
        self.by_sample = {}
        for si, posmap in tmp.items():
            self.by_sample[si] = [(t, posmap[t]) for t in sorted(posmap.keys())]
        self.n_samples = int(z["n_samples"][0]) if "n_samples" in z.files \
            else (max(self.by_sample.keys()) + 1 if self.by_sample else 0)
        self.max_len = int(z["max_len"][0]) if "max_len" in z.files else None
        # bind 後に確定する順送り状態。
        self._order = None        # bind した dataset での sample_idx の走査順
        self._calls_per = None    # 各 sample が soft_target を呼ぶ回数 (= soft 位置数)
        self._oi = 0              # _order 内の現在 sample インデックス
        self._served = 0          # 現 sample で配信済みの呼数
        self._cur_list = []       # 現 sample の (t, dist) 列

    def bind(self, dataset, max_len):
        """dataset の走査順 (= precompute_soft_targets の順) で各 sample の soft 位置数を
        先に計算し、順送りの基準を確定する。compute_sample_soft の位置ロジックを teacher
        非依存に再現 (specials/tags_start 規則は同一) するだけで teacher.soft_target は呼ばない。
        """
        order = []
        calls = []
        for s in dataset.samples:
            ids, tags_start = s[0], s[1]
            order.append(id(ids))
            calls.append(self._count_soft_positions(ids, tags_start, max_len))
        self._order = order
        self._calls_per = calls
        # id(ids) -> その sample の soft 位置数 (precompute は id(ids) 順だが順送りで足りる)。
        self.reset()
        return self

    @staticmethod
    def _count_soft_positions(ids, tags_start, max_len):
        """compute_sample_soft が soft_target を呼ぶ回数 (= soft 位置数)。teacher 非依存。"""
        ids = ids[:max_len]
        n = len(ids)
        start_t = max(tags_start - 1, 0)
        cnt = 0
        for t in range(n - 1):
            pos = t + 1
            gold = ids[pos]
            if pos >= tags_start and gold >= 5:
                if t >= start_t:
                    cnt += 1
        return cnt

    def _advance_to_next_active(self):
        """次に soft_target を呼ぶ sample (soft 位置 > 0) までポインタを進める。"""
        while self._oi < len(self._calls_per) and self._calls_per[self._oi] == 0:
            self._oi += 1
        if self._oi < len(self._calls_per):
            si = self._sample_idx_at(self._oi)
            self._cur_list = self.by_sample.get(si, [])
        else:
            self._cur_list = []
        self._served = 0

    def _sample_idx_at(self, oi):
        """_order[oi] (= id(ids)) に対応する npz sample_idx。bind した dataset の
        走査順は precompute_soft_targets の走査順と同一 (どちらも dataset.samples 順) で、
        npz の sample_idx も dataset.samples 順に振られているので oi == sample_idx。"""
        return oi

    def soft_target(self, prefix_ids, gold_id):
        """1 target 位置の soft 分布 dict (CoOccurrenceTeacher と同一 I/F)。"""
        if gold_id < 5:
            return {gold_id: 1.0}
        if self._calls_per is None:
            raise RuntimeError("ExternalTagTeacher.bind(dataset, max_len) を先に呼ぶこと。")
        # 現 sample を配信し切ったら次の active sample へ。
        if self._served >= (self._calls_per[self._oi] if self._oi < len(self._calls_per) else 0):
            self._oi += 1
            self._advance_to_next_active()
        if self._oi >= len(self._calls_per):
            return {gold_id: 1.0}  # 理論上来ない (フォールバック)
        if self._served < len(self._cur_list):
            t, dist = self._cur_list[self._served]
            self._served += 1
            return dict(dist)
        self._served += 1
        return {gold_id: 1.0}

    def reset(self):
        """順送り状態をリセットし、最初の active sample に合わせる。"""
        self._oi = 0
        self._served = 0
        self._advance_to_next_active()


def compute_sample_soft(teacher, ids, tags_start, max_len):
    """1 サンプルの target 位置ごとの soft 分布を疎リストで返す (事前計算用)。

    返り値: list of (t, dict{id: prob})。t は collate の target index
    (= ids-position pos - 1)。loss_mode="tags" で実際に残る位置のみ作る
    (start_t = max(tags_start-1, 0) 未満は -100 でマスクされるため不要)。
    prefix = tags_start 以降で当該位置より前に出た本体タグ列 (build_soft_target_tensor
    と同一の prefix 規則)。teacher は決定的なので epoch 不変 = 1 度計算で足りる。
    """
    ids = ids[:max_len]
    n = len(ids)
    start_t = max(tags_start - 1, 0)
    out = []
    prefix = []
    for t in range(n - 1):  # collate の有効 target は index 0..n-2
        pos = t + 1
        gold = ids[pos]
        if pos >= tags_start and gold >= 5:
            # この位置は loss 対象 (本体タグ)。soft を作る。
            if t >= start_t:
                out.append((t, teacher.soft_target(prefix, gold)))
            prefix.append(gold)
        else:
            # 構造トークン / tags_start 未満。prefix には本体タグのみ積む。
            if pos >= tags_start and gold >= 5:
                prefix.append(gold)
    return out


def precompute_soft_targets(teacher, dataset, max_len):
    """dataset 全サンプルの soft 分布を事前計算し {id(ids_list): sparse} を返す。

    ids_list (= dataset.samples[i][0]) はオブジェクト同一性が安定なのでキーに使える。
    毎 epoch・毎バッチ再計算していた Python ループをこの 1 度に畳む (数値は不変)。
    """
    cache = {}
    for s in dataset.samples:
        ids, tags_start = s[0], s[1]
        cache[id(ids)] = compute_sample_soft(teacher, ids, tags_start, max_len)
    return cache


def build_soft_target_tensor(teacher, batch, tgt, max_len, soft_cache=None):
    """バッチの target 位置に対し teacher soft 分布 [B, L-1, V] を作る (疎→密)。

    soft_cache 指定時はサンプル単位の事前計算 (precompute_soft_targets) を引いて
    密テンソルへ展開するだけ (高速)。未指定時は従来どおりその場計算 (smoke 等)。
    teacher 未使用位置 (ignore) は all-zero 行 (KL 側で valid マスクし無視)。
    """
    B, Lm1 = tgt.shape
    soft = torch.zeros((B, Lm1, VOCAB_SIZE), dtype=torch.float32)
    for bi, sample in enumerate(batch):
        ids, tags_start = sample[0], sample[1]
        if soft_cache is not None:
            sparse = soft_cache.get(id(ids))
            if sparse is None:
                sparse = compute_sample_soft(teacher, ids, tags_start, max_len)
        else:
            sparse = compute_sample_soft(teacher, ids, tags_start, max_len)
        for t, dist in sparse:
            if t >= Lm1:
                continue
            for k, v in dist.items():
                soft[bi, t, k] = v
    return soft


# ==============================================================
# 評価: top-k tag recall (teacher forcing)
# ==============================================================
@torch.no_grad()
def eval_loss_and_recall(model, dataset, batch_size, max_len, loss_mode,
                         device, topk=10):
    """val loss と top-k tag recall を計算する。

    recall: 各サンプルの tags 部について、各 target タグ位置でモデルが
    予測する top-k logits に正解タグ id が入っていれば hit。
    sum(hit)/sum(target tags) を recall とする (teacher forcing)。
    random ベースライン = topk / VOCAB_SIZE も併せて返す。
    """
    model.eval()
    total_loss = 0.0
    total_tok = 0
    total_hit = 0
    total_tgt = 0
    rng = None
    for batch in make_batches(dataset, batch_size, False, rng):
        inp, tgt = collate(batch, max_len, loss_mode)
        inp = inp.to(device)
        tgt = tgt.to(device)
        logits = model(inp)  # [B, L-1, V]
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1),
            ignore_index=-100, reduction="sum")
        n_tok = (tgt != -100).sum().item()
        total_loss += loss.item()
        total_tok += n_tok
        # top-k recall は tags 部 (loss_mode に関係なく tags_start 以降) で測る
        topk_idx = logits.topk(topk, dim=-1).indices  # [B, L-1, topk]
        for bi, sample in enumerate(batch):
            ids, tags_start = sample[0], sample[1]
            ids = ids[:max_len]
            n = len(ids)
            for t in range(n - 1):
                pos = t + 1  # 予測される ids-position
                if pos < tags_start:
                    continue
                gold = ids[pos]
                if gold in (TOK_PAD, TOK_SEP, TOK_BOS):
                    continue
                total_tgt += 1
                if gold in topk_idx[bi, t].tolist():
                    total_hit += 1
    avg_loss = total_loss / max(total_tok, 1)
    recall = total_hit / max(total_tgt, 1)
    random_baseline = topk / VOCAB_SIZE
    return avg_loss, recall, random_baseline, total_tgt


@torch.no_grad()
def eval_loss_and_recall_by_source(model, dataset, batch_size, max_len, loss_mode,
                                   device, topk=10):
    """source ("synthetic" / "identity_cond") 別に val loss / recall を出す。

    混合 val を 1 度走査し、各サンプルの source でロス・recall を振り分ける。
    返り値: {source: {"val_loss", "recall", "n_target_tags", "n_samples"}} +
            全体集計 ("all")。loss は target 区間 (collate の -100 マスク後) のみ。
    """
    model.eval()
    agg = {}

    def _slot(s):
        if s not in agg:
            agg[s] = {"loss_sum": 0.0, "tok": 0, "hit": 0, "tgt": 0, "n": 0}
        return agg[s]

    for batch in make_batches(dataset, batch_size, False, None):
        inp, tgt = collate(batch, max_len, loss_mode)
        inp = inp.to(device)
        tgt = tgt.to(device)
        logits = model(inp)                       # [B, L-1, V]
        Lm1 = logits.shape[1]
        # サンプルごとに per-token CE を取り source 別に積算する。
        ce = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1),
            ignore_index=-100, reduction="none").reshape(len(batch), Lm1)
        valid = (tgt != -100)
        topk_idx = logits.topk(topk, dim=-1).indices  # [B, L-1, topk]
        for bi, sample in enumerate(batch):
            ids, tags_start, source = sample[0], sample[1], sample[2]
            slot = _slot(source)
            slot["n"] += 1
            v = valid[bi]
            slot["loss_sum"] += ce[bi][v].sum().item()
            slot["tok"] += int(v.sum().item())
            ids = ids[:max_len]
            n = len(ids)
            for t in range(min(n - 1, Lm1)):
                pos = t + 1
                if pos < tags_start:
                    continue
                gold = ids[pos]
                if gold in (TOK_PAD, TOK_SEP, TOK_BOS):
                    continue
                slot["tgt"] += 1
                if gold in topk_idx[bi, t].tolist():
                    slot["hit"] += 1

    out = {}
    tot = {"loss_sum": 0.0, "tok": 0, "hit": 0, "tgt": 0, "n": 0}
    for s, d in agg.items():
        for k in tot:
            tot[k] += d[k]
        out[s] = {
            "val_loss": d["loss_sum"] / max(d["tok"], 1),
            f"top{topk}_recall": d["hit"] / max(d["tgt"], 1),
            "n_target_tags": d["tgt"],
            "n_samples": d["n"],
        }
    out["all"] = {
        "val_loss": tot["loss_sum"] / max(tot["tok"], 1),
        f"top{topk}_recall": tot["hit"] / max(tot["tgt"], 1),
        "n_target_tags": tot["tgt"],
        "n_samples": tot["n"],
    }
    out["_random_baseline"] = topk / VOCAB_SIZE
    return out


def build_identity_prompt(tok, row, max_len):
    """identity_cond 行の retention 用 prompt を組む (target 区間直前まで)。

    prompt = <bos> identity_ids <sep> scene_text(greedy) <sep>
    build_sequence の identity_cond 分岐と同じ前半を再現する (2 個目 <sep> まで)。
    返り値: prompt_ids (list)。max_len を超える場合は None (retention 対象外)。
    """
    meta = row.get("meta", {})
    identity_tags = meta.get("identity_tags") or []
    identity_ids = [tok.tag_to_id_lookup(t) for t in identity_tags]
    scene_ids = encode_text_greedy(tok, row["text"])
    prompt = [TOK_BOS] + identity_ids + [TOK_SEP] + scene_ids + [TOK_SEP]
    if len(prompt) >= max_len:
        return None  # prompt だけで列が埋まる → 生成余地なし・retention 対象外
    return prompt


@torch.no_grad()
def eval_identity_retention(model, tok, val_rows, device, max_len=MAX_SEQ_LEN):
    """identity retention rate を計測する (A2 最重要新指標)。

    各 identity_cond val 行について prompt = <bos> identity <sep> scene_text <sep>
    から greedy デコードし、生成 target に identity_tags が何個含まれるかを測る:
      retention_i = |生成 target に出た identity_ids (集合)| / |入力 identity_ids (集合)|
    返り値: (mean_retention, n_cases, details(list of dict))。
    identity_ids は <unk> を除いた集合 (語彙内 identity のみ)。
    """
    model.eval()
    ret_sum = 0.0
    n = 0
    details = []
    for row in val_rows:
        if row.get("source") != "identity_cond":
            continue
        meta = row.get("meta", {})
        identity_tags = meta.get("identity_tags") or []
        identity_ids = set(tok.tag_to_id_lookup(t) for t in identity_tags)
        identity_ids.discard(TOK_UNK)  # 語彙外 identity は分母から除外
        if not identity_ids:
            continue
        prompt = build_identity_prompt(tok, row, max_len)
        if prompt is None:
            continue
        gen_ids = _greedy_generate(model, prompt, device, max_len)
        gen_set = set(gen_ids)
        hit = len(identity_ids & gen_set)
        ret = hit / len(identity_ids)
        ret_sum += ret
        n += 1
        details.append({
            "post_id": meta.get("post_id"),
            "n_identity": len(identity_ids),
            "n_retained": hit,
            "retention": ret,
            "gen_len": len(gen_ids),
        })
    mean_ret = ret_sum / max(n, 1)
    return mean_ret, n, details


@torch.no_grad()
def eval_generation_diversity(model, tok, val_rows, device, max_len=MAX_SEQ_LEN,
                              max_cases=200):
    """val prompt から greedy 生成し、生成多様性 (§10.3 同等) を測る。

    返り値 dict:
      corpus_unique_tags     : 全 val 生成にわたる本体 tag id のユニーク数。
      mean_per_seq_unique    : 系列内ユニーク率 (unique本体tag数/本体tag数) の平均。
      norm_entropy           : 生成本体タグ頻度分布の正規化エントロピー (H/log(ユニーク数))。
      n_cases / total_gen_tags。
    prompt は synthetic と同じ <bos> + encode_text_greedy(text) + <sep> (#1 と同条件)。
    """
    import math as _m
    model.eval()
    freq = {}
    per_seq_unique = []
    total = 0
    n = 0
    for row in val_rows:
        if n >= max_cases:
            break
        text_ids = encode_text_greedy(tok, row["text"])
        prompt = [TOK_BOS] + text_ids + [TOK_SEP]
        if len(prompt) >= max_len:
            continue
        gen = _greedy_generate(model, prompt, device, max_len)
        body = [g for g in gen if g >= 5]  # 本体タグのみ (specials 除外)
        if not body:
            n += 1
            continue
        per_seq_unique.append(len(set(body)) / len(body))
        for g in body:
            freq[g] = freq.get(g, 0) + 1
        total += len(body)
        n += 1
    corpus_unique = len(freq)
    if total > 0 and corpus_unique > 1:
        Z = sum(freq.values())
        H = -sum((c / Z) * _m.log(c / Z) for c in freq.values())
        norm_h = H / _m.log(corpus_unique)
    else:
        norm_h = 0.0
    mean_psu = (sum(per_seq_unique) / len(per_seq_unique)) if per_seq_unique else 0.0
    return {
        "corpus_unique_tags": corpus_unique,
        "mean_per_seq_unique": mean_psu,
        "norm_entropy": norm_h,
        "n_cases": n,
        "total_gen_tags": total,
    }


# ==============================================================
# C-2: 生成ベース set-metrics (実運用に近い物差し・training-spec C)
# ==============================================================
#
# 動機 (C 設計):
#   従来の eval_loss_and_recall は gold 接頭辞を毎位置で食わせる teacher-forcing
#   top-k recall = 「3 テンプレに合うか」を測る proxy。本関数は **greedy 自己回帰生成**
#   したタグ集合と gold タグ集合の set-overlap (実運用 = 自由文->タグ集合 に近い) を測る。
#   従来 recall は eval_loss_and_recall として**残す** (継続比較・非回帰アンカー)。
#
# 規約 (#1 / diversity / retention と厳密に揃える):
#   - prompt = [TOK_BOS] + encode_text_greedy(tok, row["text"]) + [TOK_SEP]  (:988-989)
#   - gen    = _greedy_generate(model, prompt, device, max_len)              (:992)
#   - gen_set = {g for g in gen if g >= 5}  specials 除外                     (:993)
#   - gold_set = {tag_to_id_lookup(t)} - {UNK}  (retention :943-944 と同規約)
#   集合指標なので順序は無視。recall@k のみ生成順の先頭 k タグを使う (順序情報を残す)。
@torch.no_grad()
def eval_generation_setmetrics(model, tok, rows, device, ks=(5, 10, 20),
                               max_len=MAX_SEQ_LEN, collect_per_sample=False):
    """greedy 生成タグ集合 vs gold タグ集合の set-metrics を計算する。

    per-sample (集合ベース・0 除算ガード付き):
      precision = |gen_set 交 gold_set| / |gen_set|
      recall    = |gen_set 交 gold_set| / |gold_set|
      f1        = 2PR / (P+R)
      jaccard   = |交| / |gen_set 和 gold_set|
      recall@k  = |{gen 先頭 k タグ (本体・順序維持・重複除去)} 交 gold_set| / |gold_set|

    集計:
      macro : 上記 per-sample 値のサンプル平均 (空集合等で値が定義できないサンプルは
              当該指標の平均母数から除外する)。
      micro : sum|交| / sum|gen| (precision)・sum|交| / sum|gold| (recall)・
              sum|交| / sum|和| (jaccard)・recall@k は sum hit@k / sum|gold|。
              micro_f1 は micro_precision / micro_recall の調和平均。

    返り値: dict (macro / micro / メタ情報)。prompt が max_len を超える行はスキップ。
    collect_per_sample=True のとき、入力 rows と同順 (1 行 1 要素・skip/未定義は NaN) の
    per-sample F1 / Jaccard / precision / recall 配列を "per_sample" キーで追加返却する
    (#1 と D5 で同じ rows・同じ skip 判定 = prompt 長のみ依存 → 行 index で paired 整合)。
    既定 False では返り値・挙動とも従来一致 (golden/既存テスト非回帰)。
    """
    model.eval()
    ks = tuple(sorted(set(int(k) for k in ks)))

    # macro 用アキュムレータ (指標ごとに有効サンプル数を別管理)。
    macro_sum = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "jaccard": 0.0}
    macro_cnt = {"precision": 0, "recall": 0, "f1": 0, "jaccard": 0}
    macro_rk_sum = {k: 0.0 for k in ks}
    macro_rk_cnt = {k: 0 for k in ks}

    # micro 用アキュムレータ。
    sum_inter = 0
    sum_gen = 0
    sum_gold = 0
    sum_union = 0
    sum_hit_at_k = {k: 0 for k in ks}

    n_cases = 0
    n_skipped = 0
    n_empty_gold = 0
    # per-sample 配列 (collect_per_sample 時のみ。rows と同順・1 行 1 要素・NaN=未定義/skip)。
    import math as _math
    ps_f1 = [] if collect_per_sample else None
    ps_jac = [] if collect_per_sample else None
    ps_prec = [] if collect_per_sample else None
    ps_rec = [] if collect_per_sample else None
    for row in rows:
        text_ids = encode_text_greedy(tok, row["text"])
        prompt = [TOK_BOS] + text_ids + [TOK_SEP]
        if len(prompt) >= max_len:
            n_skipped += 1
            if collect_per_sample:
                # skip 行も rows と同順を保つため NaN を 1 件積む (#1/D5 で skip 集合は同一)。
                ps_f1.append(float("nan")); ps_jac.append(float("nan"))
                ps_prec.append(float("nan")); ps_rec.append(float("nan"))
            continue
        gen = _greedy_generate(model, prompt, device, max_len)
        # 本体タグのみ・順序維持で重複除去 (recall@k の「先頭 k」を順序通り取るため)。
        gen_seq = []
        seen = set()
        for g in gen:
            if g >= 5 and g not in seen:
                seen.add(g)
                gen_seq.append(g)
        gen_set = set(gen_seq)
        gold_set = {tok.tag_to_id_lookup(t) for t in row["tags"]}
        gold_set.discard(TOK_UNK)
        n_cases += 1
        if not gold_set:
            # gold が空 (全タグ語彙外) のサンプルは recall/F1/Jaccard の母数から除外。
            n_empty_gold += 1
        inter = gen_set & gold_set
        ninter = len(inter)

        # micro 集計。
        sum_inter += ninter
        sum_gen += len(gen_set)
        sum_gold += len(gold_set)
        sum_union += len(gen_set | gold_set)

        # macro per-sample (0 除算ガード)。
        if len(gen_set) > 0:
            prec_i = ninter / len(gen_set)
            macro_sum["precision"] += prec_i
            macro_cnt["precision"] += 1
        else:
            prec_i = None
        if len(gold_set) > 0:
            rec_i = ninter / len(gold_set)
            macro_sum["recall"] += rec_i
            macro_cnt["recall"] += 1
        else:
            rec_i = None
        if prec_i is not None and rec_i is not None:
            denom = prec_i + rec_i
            f1_i = (2.0 * prec_i * rec_i / denom) if denom > 0 else 0.0
            macro_sum["f1"] += f1_i
            macro_cnt["f1"] += 1
        union_n = len(gen_set | gold_set)
        if union_n > 0:
            macro_sum["jaccard"] += ninter / union_n
            macro_cnt["jaccard"] += 1

        # recall@k (gen 先頭 k タグ)。
        for k in ks:
            topk_set = set(gen_seq[:k])
            hit_k = len(topk_set & gold_set)
            sum_hit_at_k[k] += hit_k
            if len(gold_set) > 0:
                macro_rk_sum[k] += hit_k / len(gold_set)
                macro_rk_cnt[k] += 1

        # per-sample 配列 (paired bootstrap/t 用)。macro と同じ定義域:
        #   F1 は prec_i/rec_i 双方定義時のみ実値・他は NaN。Jaccard は union>0 で実値。
        if collect_per_sample:
            f1_val = (f1_i if (prec_i is not None and rec_i is not None)
                      else float("nan"))
            jac_val = (ninter / union_n) if union_n > 0 else float("nan")
            ps_f1.append(float(f1_val)); ps_jac.append(float(jac_val))
            ps_prec.append(float(prec_i) if prec_i is not None else float("nan"))
            ps_rec.append(float(rec_i) if rec_i is not None else float("nan"))

    macro = {
        "precision": macro_sum["precision"] / max(macro_cnt["precision"], 1),
        "recall": macro_sum["recall"] / max(macro_cnt["recall"], 1),
        "f1": macro_sum["f1"] / max(macro_cnt["f1"], 1),
        "jaccard": macro_sum["jaccard"] / max(macro_cnt["jaccard"], 1),
    }
    for k in ks:
        macro[f"recall@{k}"] = macro_rk_sum[k] / max(macro_rk_cnt[k], 1)

    micro_p = sum_inter / max(sum_gen, 1)
    micro_r = sum_inter / max(sum_gold, 1)
    micro_f1 = (2.0 * micro_p * micro_r / (micro_p + micro_r)
                if (micro_p + micro_r) > 0 else 0.0)
    micro = {
        "precision": micro_p,
        "recall": micro_r,
        "f1": micro_f1,
        "jaccard": sum_inter / max(sum_union, 1),
    }
    for k in ks:
        micro[f"recall@{k}"] = sum_hit_at_k[k] / max(sum_gold, 1)

    out = {
        "macro": macro,
        "micro": micro,
        "ks": list(ks),
        "n_cases": n_cases,
        "n_skipped_prompt_overflow": n_skipped,
        "n_empty_gold": n_empty_gold,
        "sum_inter": sum_inter,
        "sum_gen": sum_gen,
        "sum_gold": sum_gold,
        "sum_union": sum_union,
    }
    if collect_per_sample:
        # rows と同順・1 行 1 要素 (len == len(rows))。NaN = skip/未定義。
        out["per_sample"] = {
            "n_rows": len(rows),
            "f1": ps_f1,
            "jaccard": ps_jac,
            "precision": ps_prec,
            "recall": ps_rec,
        }
    return out


# ==============================================================
# C-3: eval-only ハーネス (訓練せず採点のみ・training-spec C)
# ==============================================================
#
# 設計判断 (単一ソース維持):
#   greedy/encode を重複実装しないため別スクリプトを作らず train_bitnet.py に内包する
#   (#1 と完全に同じ _greedy_generate / encode_text_greedy / Tokenizer を使う)。
#   レポートは data/bitnet/eval_report_<name>.json に provenance 付きで書く。
def _file_sha256(path):
    """ファイルの sha256 (provenance 用)。存在しなければ None。"""
    import hashlib
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_rev():
    """現在の git rev (取得不能なら None)。"""
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def eval_only(args):
    """--eval-only: 与えた重みで legacy teacher-forcing recall + 生成 set-metrics を採点。

    - Legacy (非回帰アンカー): eval_loss_and_recall を pairs.val.jsonl で。
      = #1 既知値 ~0.777 を再現できることを確認する経路。
    - 新指標: eval_generation_setmetrics を ① pairs.val.jsonl ② eval_diverse_a ③ _b で。
      diverse jsonl は **ファイル存在ガード付き** (無ければスキップ・C-1 未着手でも動く)。
    - eval_generation_diversity も併記 (無料・情報量)。
    出力 = data/bitnet/eval_report_<name>.json (全数値 + provenance)。
    """
    from safetensors.torch import load_file  # noqa: F401 (存在確認用)
    import random

    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)

    data_dir = args.data_dir
    vocab_path = args.vocab or os.path.join(data_dir, "vocab.json")
    device = args.device or "cpu"  # eval は決定性重視で CPU 既定 (golden 経路に倣う)
    weights = args.weights
    if not weights:
        raise SystemExit("--eval-only には --weights <export_safetensors path> が必須")
    if not os.path.exists(weights):
        raise FileNotFoundError(f"{weights} が無い (--weights)")

    print(f"[eval-only] device={device} torch={torch.__version__} weights={weights}")
    tok = Tokenizer(vocab_path)
    print(f"[eval-only] vocab_size={tok.vocab_size()} tags={len(tok.tags)}")

    # dump_golden と同じ経路で export_safetensors 命名重みをロードする。
    model = _load_dense_from_export(weights, device)
    print(f"[eval-only] loaded {weights} (strict state_dict 一致)")

    val_path = os.path.join(data_dir, "pairs.val.jsonl")
    val_rows = load_pairs(val_path)
    val_ds = PairDataset(tok, val_rows, args.max_len)

    # ---- Legacy teacher-forcing top-k recall (#1 既知値 ~0.777 再現アンカー) ----
    tf_loss, tf_recall, tf_rand, tf_ntgt = eval_loss_and_recall(
        model, val_ds, args.batch_size, args.max_len, args.loss_mode,
        device, args.topk)
    print(f"[eval-only] LEGACY teacher-forcing: val_loss={tf_loss:.4f} "
          f"top{args.topk}_recall={tf_recall:.4f} "
          f"(rand={tf_rand:.4f}, {tf_recall/tf_rand:.1f}x, n_tgt={tf_ntgt})")

    # ---- 新: 生成ベース set-metrics on pairs.val.jsonl ----
    ks = tuple(int(k) for k in args.set_ks.split(","))
    set_val = eval_generation_setmetrics(model, tok, val_rows, device, ks, args.max_len)
    print(f"[eval-only] GEN set-metrics (pairs.val): "
          f"macro P={set_val['macro']['precision']:.4f} "
          f"R={set_val['macro']['recall']:.4f} "
          f"F1={set_val['macro']['f1']:.4f} "
          f"J={set_val['macro']['jaccard']:.4f} | "
          f"micro R={set_val['micro']['recall']:.4f} "
          f"(n_cases={set_val['n_cases']} skip={set_val['n_skipped_prompt_overflow']})")

    # ---- 新: diverse val (存在ガード付き) ----
    diverse_reports = {}
    persample_dump = {}  # tag -> per_sample dict (--dump-persample 時のみ)
    for tag, fname in (("eval_diverse_a", "pairs.eval_diverse_a.jsonl"),
                       ("eval_diverse_b", "pairs.eval_diverse_b.jsonl")):
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"[eval-only] {fname} 無し -> スキップ (C-1 未着手・存在ガード)")
            diverse_reports[tag] = {"skipped": True, "reason": "file not found",
                                    "path": fpath}
            continue
        rows = load_pairs(fpath)
        rep = eval_generation_setmetrics(
            model, tok, rows, device, ks, args.max_len,
            collect_per_sample=args.dump_persample)
        rep["skipped"] = False
        rep["file"] = fname
        rep["sha256"] = _file_sha256(fpath)
        if args.dump_persample and "per_sample" in rep:
            # npz には別保存し、JSON レポートからは配列を外す (JSON 肥大回避)。
            persample_dump[tag] = rep.pop("per_sample")
        diverse_reports[tag] = rep
        print(f"[eval-only] GEN set-metrics ({fname}): "
              f"macro R={rep['macro']['recall']:.4f} F1={rep['macro']['f1']:.4f} "
              f"(n_cases={rep['n_cases']})")

    # ---- 生成多様性 (併記) ----
    diversity = eval_generation_diversity(model, tok, val_rows, device, args.max_len)
    print(f"[eval-only] diversity: unique_tags={diversity['corpus_unique_tags']} "
          f"per_seq_unique={diversity['mean_per_seq_unique']:.3f} "
          f"norm_entropy={diversity['norm_entropy']:.3f}")

    # ---- identity retention (重みが identity_cond 評価対象なら・任意) ----
    # pairs.val.jsonl は synthetic のみなので retention は 0 件になり得る。
    # identity 重みの再採点用に pairs.identity.val.jsonl があれば併記する。
    id_val_path = os.path.join(data_dir, "pairs.identity.val.jsonl")
    retention_report = None
    if os.path.exists(id_val_path):
        id_rows = load_pairs(id_val_path)
        mean_ret, n_ret, _ = eval_identity_retention(
            model, tok, id_rows, device, args.max_len)
        retention_report = {"mean_retention": mean_ret, "n_cases": n_ret,
                            "file": "pairs.identity.val.jsonl"}
        if n_ret > 0:
            print(f"[eval-only] identity retention (pairs.identity.val)= "
                  f"{mean_ret:.4f} (n={n_ret})")

    # ---- レポート出力 ----
    name = args.eval_name or os.path.splitext(os.path.basename(weights))[0]
    report = {
        "schema": "dollama/eval_report/C-1",
        "provenance": {
            "weights": os.path.abspath(weights),
            "weights_sha256": _file_sha256(weights),
            "val_file": "pairs.val.jsonl",
            "val_sha256": _file_sha256(val_path),
            "vocab": os.path.abspath(vocab_path),
            "seed": seed,
            "device": device,
            "torch": torch.__version__,
            "git_rev": _git_rev(),
            "max_len": args.max_len,
            "topk": args.topk,
            "set_ks": list(ks),
        },
        "legacy_teacher_forcing": {
            "val_loss": tf_loss,
            f"top{args.topk}_recall": tf_recall,
            "random_baseline": tf_rand,
            "recall_vs_random_x": (tf_recall / tf_rand) if tf_rand else None,
            "n_target_tags": tf_ntgt,
            "note": "eval_loss_and_recall (gold prefix teacher forcing)・#1 継続比較アンカー",
        },
        "generation_setmetrics": {
            "pairs_val": set_val,
            "eval_diverse_a": diverse_reports.get("eval_diverse_a"),
            "eval_diverse_b": diverse_reports.get("eval_diverse_b"),
            "note": "greedy 自己回帰生成タグ集合 vs gold タグ集合の set-overlap (実運用寄り)",
        },
        "generation_diversity": diversity,
        "identity_retention": retention_report,
    }
    out_path = os.path.join(data_dir, f"eval_report_{name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"[eval-only] saved {out_path}")

    # --- per-sample 配列 npz (paired bootstrap/t 用・--dump-persample 時のみ) ---
    if args.dump_persample and persample_dump:
        import numpy as _np
        npz_path = os.path.join(data_dir, f"eval_persample_{name}.npz")
        save_kw = {}
        for tag, ps in persample_dump.items():
            save_kw[f"{tag}__f1"] = _np.asarray(ps["f1"], dtype=_np.float64)
            save_kw[f"{tag}__jaccard"] = _np.asarray(ps["jaccard"], dtype=_np.float64)
            save_kw[f"{tag}__precision"] = _np.asarray(ps["precision"], dtype=_np.float64)
            save_kw[f"{tag}__recall"] = _np.asarray(ps["recall"], dtype=_np.float64)
        # provenance を npz にも埋める (どの重み・seed の per-sample か追跡可能に)。
        save_kw["_weights"] = _np.asarray(os.path.abspath(weights))
        save_kw["_weights_sha256"] = _np.asarray(_file_sha256(weights) or "")
        save_kw["_seed"] = _np.asarray(seed)
        _np.savez_compressed(npz_path, **save_kw)
        print(f"[eval-only] saved per-sample arrays -> {npz_path} "
              f"(tags={list(persample_dump.keys())})")
    return report


# ==============================================================
# safetensors 出力 (bitnet.hpp::Layer レイアウト 1:1)
# ==============================================================
def export_safetensors(model, path, dtype):
    """bitnet.hpp の重みレイアウトで safetensors 保存する。

    テンソル名 (training-spec.md に記録):
      embed                       [4999,512]
      layers.{i}.attn_norm        [512]
      layers.{i}.wq/wk/wv/wo      [512,512]  (row-major [out,in] = nn.Linear.weight)
      layers.{i}.ffn_norm         [512]
      layers.{i}.w_gate/w_up      [1792,512]
      layers.{i}.w_down           [512,1792]
      final_norm                  [512]
    embed tied のため lm_head は別名出力しない。
    """
    from safetensors.torch import save_file

    sd = {}
    sd["embed"] = model.embed.weight.detach().to(dtype).contiguous().cpu()
    for i, layer in enumerate(model.layers):
        p = f"layers.{i}."
        sd[p + "attn_norm"] = layer.attn_norm.weight.detach().to(dtype).contiguous().cpu()
        sd[p + "wq"] = layer.wq.weight.detach().to(dtype).contiguous().cpu()
        sd[p + "wk"] = layer.wk.weight.detach().to(dtype).contiguous().cpu()
        sd[p + "wv"] = layer.wv.weight.detach().to(dtype).contiguous().cpu()
        sd[p + "wo"] = layer.wo.weight.detach().to(dtype).contiguous().cpu()
        sd[p + "ffn_norm"] = layer.ffn_norm.weight.detach().to(dtype).contiguous().cpu()
        sd[p + "w_gate"] = layer.w_gate.weight.detach().to(dtype).contiguous().cpu()
        sd[p + "w_up"] = layer.w_up.weight.detach().to(dtype).contiguous().cpu()
        sd[p + "w_down"] = layer.w_down.weight.detach().to(dtype).contiguous().cpu()
    sd["final_norm"] = model.final_norm.weight.detach().to(dtype).contiguous().cpu()
    save_file(sd, path)
    return list(sd.keys())


def sanity_reload(path):
    """保存した safetensors を読み戻して shape/dtype を検証する。"""
    from safetensors import safe_open

    expect = {"embed": (VOCAB_SIZE, D_MODEL), "final_norm": (D_MODEL,)}
    for i in range(N_LAYERS):
        p = f"layers.{i}."
        expect[p + "attn_norm"] = (D_MODEL,)
        expect[p + "wq"] = (D_MODEL, D_MODEL)
        expect[p + "wk"] = (D_MODEL, D_MODEL)
        expect[p + "wv"] = (D_MODEL, D_MODEL)
        expect[p + "wo"] = (D_MODEL, D_MODEL)
        expect[p + "ffn_norm"] = (D_MODEL,)
        expect[p + "w_gate"] = (FFN_DIM, D_MODEL)
        expect[p + "w_up"] = (FFN_DIM, D_MODEL)
        expect[p + "w_down"] = (D_MODEL, FFN_DIM)
    problems = []
    with safe_open(path, framework="pt") as f:
        keys = set(f.keys())
        if keys != set(expect.keys()):
            problems.append(f"key set mismatch: extra={keys - set(expect)}, "
                            f"missing={set(expect) - keys}")
        for name, shp in expect.items():
            if name not in keys:
                continue
            t = f.get_tensor(name)
            if tuple(t.shape) != shp:
                problems.append(f"{name} shape {tuple(t.shape)} != {shp}")
            if torch.isnan(t.float()).any() or torch.isinf(t.float()).any():
                problems.append(f"{name} has NaN/Inf")
    return problems


# ==============================================================
# golden dump (#6 C++ dense 推論の数値突合用)
# ==============================================================
#
# 設計判断 (ファイル形式):
#   ロジット・input_ids・生成 tag id 列はすべて **safetensors** で出力する。
#   生バイナリ + 手書きメタではなく safetensors を選んだ理由:
#     - #6 は src/io/safetensors.hpp を使うので、golden も同じローダーで読める
#       (dtype/shape/エンディアンの取り違えを構造的に排除・読み戻しサニティが楽)。
#     - safetensors は dtype/shape をヘッダに自己記述 -> メタ JSON 側で
#       エンディアン/レイアウトを別管理する必要がない (常に little-endian raw)。
#   ロジットは FP32 (誤差切り分けの基準)。input_ids / 生成 tag id 列は I32
#   (safetensors.hpp が I32 対応・C++ 側 token id は int で扱う)。
#
# manifest.json には各ケースの input_ids/text/shape/dtype/ファイル名と
# greedy 停止規則を人間可読で残す (C++ #6 実装者が読む唯一の索引)。
LOGITS_PATH = "logits_golden.safetensors"
GEN_PATH = "gen_golden.safetensors"
MANIFEST_PATH = "manifest.json"

# 生成 golden の greedy 停止規則 (C++ が完全再現できるよう厳密に明記)。
#   1. prompt = [<bos>] + encode_text_greedy(text) + [<sep>]。
#   2. 各ステップ: 現在列を BitNetDense に通し、最終位置 (列末) の logits の
#      argmax (FP32, tie は小さい id を採用 = torch.argmax の挙動) を次トークンとする。
#   3. 生成した id を列に追記し、生成トークン (生成 tag id 列) にも記録する。
#   4. 停止条件 (どれか満たしたら即停止):
#        a. 次トークン == <eos>(=2) -> <eos> は生成列に含めず停止。
#        b. 列長が MAX_SEQ_LEN(=64) に達した -> そこで打ち切り。
#   5. 既出タグの抑制・重複除去は **行わない** (素の greedy。C++ も同一)。
#   6. <pad>/<bos>/<sep>/<unk> が生成されても特別扱いせず、そのまま列に積む
#      (停止するのは <eos> のみ)。
GREEDY_STOP_RULE = {
    "prompt": "[<bos>] + encode_text_greedy(text) + [<sep>]",
    "step": "最終位置 logits の argmax (FP32, tie は小 id) を次トークン",
    "stop_eos": TOK_EOS,
    "stop_max_len": MAX_SEQ_LEN,
    "eos_excluded_from_output": True,
    "suppress_repeats": False,
    "special_tokens_pass_through": True,
    "note": "生成 tag id 列は <bos>/text/<sep> を含まない pure な生成部のみ。"
            "停止は <eos> 出力 or 列長 MAX_SEQ_LEN。",
}


def _make_fixed_input_ids(tok, val_rows, target_len):
    """決定的に長さ target_len ちょうどの input_ids を作る。

    val_rows を固定インデックスで走査し、build_sequence の自己回帰列を連結して
    target_len で切り出す (val は max seq ~30 のため 63 は複数行連結で確保)。
    再現可能・seq 範囲内 id のみ。
    """
    ids = []
    for r in val_rows:
        seq, _ = build_sequence(tok, r, MAX_SEQ_LEN)
        ids.extend(seq)
        if len(ids) >= target_len:
            break
    if len(ids) < target_len:
        raise RuntimeError(
            f"val 行を全部連結しても target_len={target_len} に届かない (got {len(ids)})")
    return ids[:target_len]


@torch.no_grad()
def _greedy_generate(model, prompt_ids, device, max_len=MAX_SEQ_LEN):
    """GREEDY_STOP_RULE に従い prompt から自己回帰 greedy デコードする。

    返り値: 生成した tag id 列 (prompt を含まない・<eos> を含まない)。
    """
    seq = list(prompt_ids)
    generated = []
    while len(seq) < max_len:
        inp = torch.tensor([seq], dtype=torch.long, device=device)
        logits = model(inp)            # [1, S, V] FP32
        next_logits = logits[0, -1]    # 列末位置 [V]
        nxt = int(torch.argmax(next_logits).item())  # tie は小 id
        if nxt == TOK_EOS:
            break                      # <eos> は出力に含めず停止
        seq.append(nxt)
        generated.append(nxt)
    return generated


@torch.no_grad()
def _sample_generate(model, prompt_ids, device, generator, max_len=MAX_SEQ_LEN,
                     temperature=1.0, top_k=0):
    """GREEDY_STOP_RULE と同じ停止規則で、argmax の代わりに temperature/top-k
    サンプリングする確率的デコーダ (F-0b best-of-N 用)。

    _greedy_generate の決定的単一出力に対し、同一 prompt でも generator の乱数で
    多様な候補が出る。**再現性は generator (torch.Generator) の seed で担保する**
    (rollout 側が (input, candidate) ごとに固定 seed の generator を渡す)。

    引数:
      generator   : torch.Generator — サンプリング乱数源 (seed 固定で再現的)。
      temperature : >0。1.0=素の分布 / <1=尖鋭化 / >1=平坦化。
      top_k       : >0 のとき上位 k logit のみ残して sampling (0=無効=全語彙)。

    返り値: 生成した tag id 列 (prompt を含まない・<eos> を含まない)。greedy と同じ。
    """
    if temperature <= 0.0:
        raise ValueError("temperature は正でなければならない (greedy は _greedy_generate)")
    seq = list(prompt_ids)
    generated = []
    while len(seq) < max_len:
        inp = torch.tensor([seq], dtype=torch.long, device=device)
        logits = model(inp)                        # [1, S, V] FP32
        next_logits = logits[0, -1].float() / temperature   # [V]
        if top_k and top_k > 0 and top_k < next_logits.numel():
            # 上位 k 以外を -inf にして分布から除外する。
            kth = torch.topk(next_logits, top_k).values[-1]
            next_logits = torch.where(
                next_logits < kth,
                torch.full_like(next_logits, float("-inf")),
                next_logits)
        probs = torch.softmax(next_logits, dim=-1)
        nxt = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
        if nxt == TOK_EOS:
            break                                  # <eos> は出力に含めず停止
        seq.append(nxt)
        generated.append(nxt)
    return generated


def dump_golden(args):
    """bitnet_dense_fp32.safetensors をロードした BitNetDense で golden を dump。

    (a) ロジット golden: 固定 input_ids (seq 8/32/63) の FP32 logits [seq, vocab]。
    (b) 生成 golden: 固定 text を greedy デコードした tag id 列。
    すべて学習時と同一の dense forward 経路 (BitNetDense) を通す。
    """
    from safetensors.torch import save_file, load_file
    import random

    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)

    data_dir = args.data_dir
    vocab_path = args.vocab or os.path.join(data_dir, "vocab.json")
    device = args.device or "cpu"  # golden は CPU 既定 (決定性重視・1080Ti FP16 native 無し)
    print(f"[golden] device={device} torch={torch.__version__}")

    tok = Tokenizer(vocab_path)
    print(f"[golden] vocab_size={tok.vocab_size()} tags={len(tok.tags)}")

    fp32_path = os.path.join(data_dir, "bitnet_dense_fp32.safetensors")
    if not os.path.exists(fp32_path):
        raise FileNotFoundError(
            f"{fp32_path} が無い。先に `python scripts/train_bitnet.py` で訓練して "
            f"FP32 重みを出力すること。")
    sd = load_file(fp32_path)
    model = BitNetDense().to(device)
    # export_safetensors 命名 (embed / layers.{i}.* / final_norm) を state_dict 命名へ写す。
    new_sd = {}
    new_sd["embed.weight"] = sd["embed"]
    new_sd["final_norm.weight"] = sd["final_norm"]
    for i in range(N_LAYERS):
        p = f"layers.{i}."
        new_sd[p + "attn_norm.weight"] = sd[p + "attn_norm"]
        new_sd[p + "ffn_norm.weight"] = sd[p + "ffn_norm"]
        for w in ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"):
            new_sd[p + w + ".weight"] = sd[p + w]
    model.load_state_dict(new_sd, strict=True)
    model.eval()
    print(f"[golden] loaded {fp32_path} (strict state_dict 一致)")

    val_rows = load_pairs(os.path.join(data_dir, "pairs.val.jsonl"))
    out_dir = os.path.join(data_dir, "golden")
    os.makedirs(out_dir, exist_ok=True)

    # ---- (a) ロジット golden -- seq 8 / 32 / 63 ----
    logit_seq_lens = [8, 32, 63]  # 境界含む (63 = max_seq 64 直下)
    logit_tensors = {}
    logit_cases = []
    for sl in logit_seq_lens:
        ids = _make_fixed_input_ids(tok, val_rows, sl)
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        logits_cpu = model(inp)[0].detach().float().contiguous().cpu()  # [seq, vocab] FP32
        re_logits = model(inp)[0].detach().float().contiguous().cpu()
        max_abs = (logits_cpu - re_logits).abs().max().item()
        if max_abs != 0.0:
            raise RuntimeError(f"自己整合失敗 seq={sl}: 再 forward 不一致 (max_abs={max_abs})")
        name = f"logits_s{sl}"
        ids_name = f"input_ids_s{sl}"
        logit_tensors[name] = logits_cpu
        logit_tensors[ids_name] = torch.tensor(ids, dtype=torch.int32)
        nan = bool(torch.isnan(logits_cpu).any() or torch.isinf(logits_cpu).any())
        logit_cases.append({
            "seq_len": sl, "input_ids": ids,
            "logits_tensor": name, "input_ids_tensor": ids_name,
            "logits_shape": [sl, VOCAB_SIZE], "logits_dtype": "F32",
            "input_ids_dtype": "I32", "has_nan_inf": nan,
        })
        print(f"[golden] (a) seq={sl}: logits {list(logits_cpu.shape)} "
              f"自己整合 max_abs={max_abs:.1e} nan/inf={nan}")
    logits_path = os.path.join(out_dir, LOGITS_PATH)
    save_file(logit_tensors, logits_path)
    print(f"[golden] saved {logits_path} ({len(logit_tensors)} tensors)")

    # ---- (b) 生成 golden -- 固定 text を greedy デコード ----
    gen_texts = [
        "a girl with long hair, looking at viewer, blue eyes.",
        "a boy with black hair, smile.",
        "1girl, twintails, school uniform, blush.",
        "金髪ツインテールの少女が笑っている",
        "two girls holding hands, blue sky.",
    ]
    gen_tensors = {}
    gen_cases = []
    for ci, text in enumerate(gen_texts):
        text_ids = encode_text_greedy(tok, text)
        prompt_ids = [TOK_BOS] + text_ids + [TOK_SEP]
        gen_ids = _greedy_generate(model, prompt_ids, device, MAX_SEQ_LEN)
        gen_ids2 = _greedy_generate(model, prompt_ids, device, MAX_SEQ_LEN)
        if gen_ids != gen_ids2:
            raise RuntimeError(f"生成が非決定的 case={ci}: {gen_ids} != {gen_ids2}")
        name = f"gen_ids_c{ci}"
        prompt_name = f"prompt_ids_c{ci}"
        gen_tensors[name] = (torch.tensor(gen_ids, dtype=torch.int32)
                             if gen_ids else torch.zeros((0,), dtype=torch.int32))
        gen_tensors[prompt_name] = torch.tensor(prompt_ids, dtype=torch.int32)
        gen_tags_readable = []
        for tid in gen_ids:
            if 5 <= tid < VOCAB_SIZE:
                gen_tags_readable.append(tok.tags[tid - 5])
            elif tid < 5:
                gen_tags_readable.append(tok.specials[tid])
        gen_cases.append({
            "case": ci, "text": text, "prompt_ids": prompt_ids,
            "prompt_ids_tensor": prompt_name, "gen_ids": gen_ids,
            "gen_ids_tensor": name, "gen_len": len(gen_ids),
            "gen_tags_readable": gen_tags_readable, "gen_ids_dtype": "I32",
        })
        print(f"[golden] (b) case={ci} text={text!r} prompt_len={len(prompt_ids)} "
              f"gen_len={len(gen_ids)} tags={gen_tags_readable[:6]}...")
    gen_path = os.path.join(out_dir, GEN_PATH)
    empties = [c["case"] for c in gen_cases if c["gen_len"] == 0]
    if empties:
        print(f"[golden] WARN: 空生成 cases={empties}")
    save_file(gen_tensors, gen_path)
    print(f"[golden] saved {gen_path} ({len(gen_tensors)} tensors)")

    # ---- manifest.json ----
    manifest = {
        "purpose": "dollama Phase 4 #6 -- C++ dense 推論 (src/infer/bitnet.hpp) の数値突合 golden",
        "generator": "scripts/train_bitnet.py --dump-golden",
        "source_weights": "data/bitnet/bitnet_dense_fp32.safetensors",
        "device": device, "torch": torch.__version__, "seed": seed,
        "arch": {
            "VOCAB_SIZE": VOCAB_SIZE, "D_MODEL": D_MODEL, "N_LAYERS": N_LAYERS,
            "N_HEADS": N_HEADS, "HEAD_DIM": HEAD_DIM, "FFN_DIM": FFN_DIM,
            "MAX_SEQ_LEN": MAX_SEQ_LEN, "ROPE_BASE": ROPE_BASE, "RMS_EPS": RMS_EPS,
        },
        "specials": {"PAD": TOK_PAD, "BOS": TOK_BOS, "EOS": TOK_EOS,
                     "SEP": TOK_SEP, "UNK": TOK_UNK},
        "format": {
            "container": "safetensors (little-endian raw, dtype/shape 自己記述)",
            "logits_file": LOGITS_PATH, "gen_file": GEN_PATH,
            "logits_dtype": "F32", "ids_dtype": "I32",
            "layout": "logits は row-major [seq, vocab]、id 列は 1D [n]",
            "reader": "src/io/safetensors.hpp で読める",
        },
        "logits_golden": {
            "note": "各ケース input_ids を BitNetDense に通した FP32 logits [seq, vocab]。"
                    "input_ids は val ペアの自己回帰列 prefix を連結して seq_len ちょうどに切出し。",
            "cases": logit_cases,
        },
        "generation_golden": {"greedy_stop_rule": GREEDY_STOP_RULE, "cases": gen_cases},
    }
    manifest_path = os.path.join(out_dir, MANIFEST_PATH)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"[golden] saved {manifest_path}")

    # ---- 読み戻しサニティ ----
    problems = []
    lt = load_file(logits_path)
    for c in logit_cases:
        t = lt[c["logits_tensor"]]
        if list(t.shape) != c["logits_shape"]:
            problems.append(f"{c['logits_tensor']} shape {list(t.shape)} != {c['logits_shape']}")
        if t.dtype != torch.float32:
            problems.append(f"{c['logits_tensor']} dtype {t.dtype} != float32")
        if torch.isnan(t).any() or torch.isinf(t).any():
            problems.append(f"{c['logits_tensor']} has NaN/Inf")
        it = lt[c["input_ids_tensor"]]
        if list(it.shape) != [c["seq_len"]]:
            problems.append(f"{c['input_ids_tensor']} shape {list(it.shape)} != [{c['seq_len']}]")
        if it.dtype != torch.int32:
            problems.append(f"{c['input_ids_tensor']} dtype {it.dtype} != int32")
        if it.tolist() != c["input_ids"]:
            problems.append(f"{c['input_ids_tensor']} ids 値が manifest と不一致")
    gt = load_file(gen_path)
    for c in gen_cases:
        t = gt[c["gen_ids_tensor"]]
        if t.dtype != torch.int32:
            problems.append(f"{c['gen_ids_tensor']} dtype {t.dtype} != int32")
        if t.tolist() != c["gen_ids"]:
            problems.append(f"{c['gen_ids_tensor']} ids 値が manifest と不一致")
    if problems:
        print("[golden] SANITY PROBLEMS:")
        for p in problems:
            print("   ", p)
        raise SystemExit(1)
    print("[golden] サニティ OK: 全 golden shape/dtype 一致・NaN/Inf なし・自己整合 OK")
    print(f"[golden] done. 出力先 {out_dir}/")


# ==============================================================
# golden dump (identity 条件付き版・A3 C++ 突合用)
# ==============================================================
#
# 設計判断 (#4 非回帰):
#   identity 版 golden は #4 の synthetic golden (logits_golden / gen_golden /
#   manifest.json) を**上書きしない**よう、すべて identity_ プレフィックスの別ファイルへ
#   書く (logits_golden_identity / gen_golden_identity / manifest_identity)。同じ
#   golden/ ディレクトリに synthetic と identity が併存する。重みも別物
#   (bitnet_dense_identity_fp32.safetensors) を使う。
ID_LOGITS_PATH = "logits_golden_identity.safetensors"
ID_GEN_PATH = "gen_golden_identity.safetensors"
ID_MANIFEST_PATH = "manifest_identity.json"


def _export_naming_to_state_dict(sd):
    """export_safetensors 命名 (embed / layers.{i}.* / final_norm) の dict を
    BitNetDense.state_dict() 命名 (embed.weight / layers.{i}.*.weight / final_norm.weight)
    へ写す (共通化・_load_dense_from_export と SFT 層状ロードで唯一のマッパを共有)。"""
    new_sd = {}
    new_sd["embed.weight"] = sd["embed"]
    new_sd["final_norm.weight"] = sd["final_norm"]
    for i in range(N_LAYERS):
        p = f"layers.{i}."
        new_sd[p + "attn_norm.weight"] = sd[p + "attn_norm"]
        new_sd[p + "ffn_norm.weight"] = sd[p + "ffn_norm"]
        for w in ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"):
            new_sd[p + w + ".weight"] = sd[p + w]
    return new_sd


def _load_export_into_model(model, fp32_path):
    """export_safetensors 命名の FP32 重みを既存 model へ strict ロードする (train/eval
    モードは変えない)。SFT の「正典 bitnet_dense から層状 warm-start」で使う。
    返り値: strict ロードしたキー数 (= state_dict のキー数)。"""
    from safetensors.torch import load_file
    sd = load_file(fp32_path)
    new_sd = _export_naming_to_state_dict(sd)
    # モデル param の dtype/device に合わせてから strict ロード (FP32 同士なら no-op)。
    ref = model.state_dict()
    cast = {k: new_sd[k].to(dtype=ref[k].dtype, device=ref[k].device)
            for k in new_sd}
    model.load_state_dict(cast, strict=True)
    return len(cast)


def _load_dense_from_export(fp32_path, device):
    """export_safetensors 命名の FP32 重みを BitNetDense にロードする (共通化)。"""
    from safetensors.torch import load_file
    sd = load_file(fp32_path)
    model = BitNetDense().to(device)
    model.load_state_dict(_export_naming_to_state_dict(sd), strict=True)
    model.eval()
    return model


def dump_golden_identity(args):
    """bitnet_dense_identity_fp32.safetensors で identity 条件付き golden を dump。

    A3 (C++ src/infer/bitnet.hpp の identity 条件付き推論) 突合用。
    (a) ロジット golden: 固定の 2-<sep> identity_cond 系列 (seq 8/32/63) の FP32 logits。
    (b) 生成 golden: identity prompt (<bos> identity <sep> scene_text(greedy) <sep>) を
        greedy デコードした tag id 列 (GREEDY_STOP_RULE 準拠)。
    #4 の synthetic golden は一切触らず identity_ 別ファイルへ書く。
    """
    from safetensors.torch import save_file, load_file
    import random

    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)

    data_dir = args.data_dir
    vocab_path = args.vocab or os.path.join(data_dir, "vocab.json")
    device = args.device or "cpu"
    print(f"[golden-id] device={device} torch={torch.__version__}")

    tok = Tokenizer(vocab_path)
    print(f"[golden-id] vocab_size={tok.vocab_size()} tags={len(tok.tags)}")

    fp32_path = os.path.join(data_dir, "bitnet_dense_identity_fp32.safetensors")
    if not os.path.exists(fp32_path):
        raise FileNotFoundError(
            f"{fp32_path} が無い。先に `--identity` で混合訓練して identity FP32 重みを "
            f"出力すること。")
    model = _load_dense_from_export(fp32_path, device)
    print(f"[golden-id] loaded {fp32_path} (strict state_dict 一致)")

    id_val_rows = load_pairs(os.path.join(data_dir, "pairs.identity.val.jsonl"))
    out_dir = os.path.join(data_dir, "golden")
    os.makedirs(out_dir, exist_ok=True)

    # ---- (a) ロジット golden -- identity_cond 2-<sep> 系列を seq 8/32/63 で ----
    logit_seq_lens = [8, 32, 63]
    logit_tensors = {}
    logit_cases = []
    for sl in logit_seq_lens:
        # identity_cond の build_sequence 列 (2-<sep>) を固定順に連結して seq_len 切出し。
        ids = []
        for r in id_val_rows:
            seq, _ = build_sequence(tok, r, MAX_SEQ_LEN)
            ids.extend(seq)
            if len(ids) >= sl:
                break
        if len(ids) < sl:
            raise RuntimeError(f"identity val を連結しても seq_len={sl} に届かない")
        ids = ids[:sl]
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        logits_cpu = model(inp)[0].detach().float().contiguous().cpu()
        re_logits = model(inp)[0].detach().float().contiguous().cpu()
        max_abs = (logits_cpu - re_logits).abs().max().item()
        if max_abs != 0.0:
            raise RuntimeError(f"自己整合失敗 seq={sl}: max_abs={max_abs}")
        name = f"logits_s{sl}"
        ids_name = f"input_ids_s{sl}"
        logit_tensors[name] = logits_cpu
        logit_tensors[ids_name] = torch.tensor(ids, dtype=torch.int32)
        nan = bool(torch.isnan(logits_cpu).any() or torch.isinf(logits_cpu).any())
        n_sep = sum(1 for x in ids if x == TOK_SEP)
        logit_cases.append({
            "seq_len": sl, "input_ids": ids,
            "logits_tensor": name, "input_ids_tensor": ids_name,
            "logits_shape": [sl, VOCAB_SIZE], "logits_dtype": "F32",
            "input_ids_dtype": "I32", "has_nan_inf": nan, "n_sep": n_sep,
        })
        print(f"[golden-id] (a) seq={sl}: logits {list(logits_cpu.shape)} "
              f"n_sep={n_sep} 自己整合 max_abs={max_abs:.1e} nan/inf={nan}")
    logits_path = os.path.join(out_dir, ID_LOGITS_PATH)
    save_file(logit_tensors, logits_path)
    print(f"[golden-id] saved {logits_path} ({len(logit_tensors)} tensors)")

    # ---- (b) 生成 golden -- identity prompt を greedy デコード ----
    gen_tensors = {}
    gen_cases = []
    # val の identity_cond 行から先頭 5 件を固定採用 (再現的・prompt が max_len 内のもの)。
    picked = 0
    for r in id_val_rows:
        if r.get("source") != "identity_cond":
            continue
        prompt_ids = build_identity_prompt(tok, r, MAX_SEQ_LEN)
        if prompt_ids is None:
            continue
        meta = r.get("meta", {})
        identity_tags = meta.get("identity_tags") or []
        identity_ids = [tok.tag_to_id_lookup(t) for t in identity_tags]
        gen_ids = _greedy_generate(model, prompt_ids, device, MAX_SEQ_LEN)
        gen_ids2 = _greedy_generate(model, prompt_ids, device, MAX_SEQ_LEN)
        if gen_ids != gen_ids2:
            raise RuntimeError(f"生成が非決定的 post_id={meta.get('post_id')}")
        ci = picked
        name = f"gen_ids_c{ci}"
        prompt_name = f"prompt_ids_c{ci}"
        idt_name = f"identity_ids_c{ci}"
        gen_tensors[name] = (torch.tensor(gen_ids, dtype=torch.int32)
                             if gen_ids else torch.zeros((0,), dtype=torch.int32))
        gen_tensors[prompt_name] = torch.tensor(prompt_ids, dtype=torch.int32)
        gen_tensors[idt_name] = (torch.tensor(identity_ids, dtype=torch.int32)
                                 if identity_ids else torch.zeros((0,), dtype=torch.int32))
        gen_tags_readable = []
        for tid in gen_ids:
            if 5 <= tid < VOCAB_SIZE:
                gen_tags_readable.append(tok.tags[tid - 5])
            elif tid < 5:
                gen_tags_readable.append(tok.specials[tid])
        id_set = set(identity_ids) - {TOK_UNK}
        retained = len(id_set & set(gen_ids))
        gen_cases.append({
            "case": ci, "post_id": meta.get("post_id"), "text": r["text"],
            "identity_tags": identity_tags, "identity_ids": identity_ids,
            "prompt_ids": prompt_ids, "prompt_ids_tensor": prompt_name,
            "gen_ids": gen_ids, "gen_ids_tensor": name, "gen_len": len(gen_ids),
            "identity_ids_tensor": idt_name,
            "gen_tags_readable": gen_tags_readable,
            "n_identity": len(id_set),
            "n_retained": retained,
            "gen_ids_dtype": "I32",
        })
        print(f"[golden-id] (b) case={ci} post_id={meta.get('post_id')} "
              f"prompt_len={len(prompt_ids)} gen_len={len(gen_ids)} "
              f"identity_retained={retained}/{len(id_set)} "
              f"tags={gen_tags_readable[:6]}...")
        picked += 1
        if picked >= 5:
            break
    gen_path = os.path.join(out_dir, ID_GEN_PATH)
    save_file(gen_tensors, gen_path)
    print(f"[golden-id] saved {gen_path} ({len(gen_tensors)} tensors)")

    # ---- manifest_identity.json ----
    manifest = {
        "purpose": "dollama Phase 4 A (A3) -- C++ identity 条件付き推論の数値突合 golden",
        "generator": "scripts/train_bitnet.py --dump-golden-identity",
        "source_weights": "data/bitnet/bitnet_dense_identity_fp32.safetensors",
        "device": device, "torch": torch.__version__, "seed": seed,
        "conditioning": {
            "scheme": "(a-1) prompt prefix + <sep>(id=3) 2 回流用 (dataset-spec §13.1)",
            "sequence": "<bos> [identity ids] <sep> [scene_text greedy ids] <sep> "
                        "[target ids] <eos>",
            "prompt_for_generation": "<bos> [identity ids] <sep> [scene_text greedy ids] <sep>",
            "note": "vocab/specials/arch は #4 と不変。<sep> が 2 個出るのは仕様 "
                    "(区間は位置で決まる)。",
        },
        "arch": {
            "VOCAB_SIZE": VOCAB_SIZE, "D_MODEL": D_MODEL, "N_LAYERS": N_LAYERS,
            "N_HEADS": N_HEADS, "HEAD_DIM": HEAD_DIM, "FFN_DIM": FFN_DIM,
            "MAX_SEQ_LEN": MAX_SEQ_LEN, "ROPE_BASE": ROPE_BASE, "RMS_EPS": RMS_EPS,
        },
        "specials": {"PAD": TOK_PAD, "BOS": TOK_BOS, "EOS": TOK_EOS,
                     "SEP": TOK_SEP, "UNK": TOK_UNK},
        "format": {
            "container": "safetensors (little-endian raw)",
            "logits_file": ID_LOGITS_PATH, "gen_file": ID_GEN_PATH,
            "logits_dtype": "F32", "ids_dtype": "I32",
            "reader": "src/io/safetensors.hpp で読める",
        },
        "logits_golden": {
            "note": "identity_cond 2-<sep> 系列 (val) を連結し seq_len ちょうどで切出した "
                    "input_ids を BitNetDense に通した FP32 logits [seq, vocab]。",
            "cases": logit_cases,
        },
        "generation_golden": {
            "greedy_stop_rule": GREEDY_STOP_RULE,
            "prompt_note": "prompt は identity prefix + scene greedy + <sep>。"
                           "identity_ids_c{ci} は retention 検証用の入力 identity 集合。",
            "cases": gen_cases,
        },
    }
    manifest_path = os.path.join(out_dir, ID_MANIFEST_PATH)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"[golden-id] saved {manifest_path}")

    # ---- 読み戻しサニティ ----
    problems = []
    lt = load_file(logits_path)
    for c in logit_cases:
        t = lt[c["logits_tensor"]]
        if list(t.shape) != c["logits_shape"]:
            problems.append(f"{c['logits_tensor']} shape {list(t.shape)} != {c['logits_shape']}")
        if t.dtype != torch.float32:
            problems.append(f"{c['logits_tensor']} dtype {t.dtype} != float32")
        if torch.isnan(t).any() or torch.isinf(t).any():
            problems.append(f"{c['logits_tensor']} has NaN/Inf")
        it = lt[c["input_ids_tensor"]]
        if it.dtype != torch.int32 or it.tolist() != c["input_ids"]:
            problems.append(f"{c['input_ids_tensor']} ids/dtype 不一致")
    gt = load_file(gen_path)
    for c in gen_cases:
        t = gt[c["gen_ids_tensor"]]
        if t.dtype != torch.int32 or t.tolist() != c["gen_ids"]:
            problems.append(f"{c['gen_ids_tensor']} ids/dtype 不一致")
    if not gen_cases:
        problems.append("identity_cond 生成ケースが 0 件 (val に identity_cond 行が無い?)")
    if problems:
        print("[golden-id] SANITY PROBLEMS:")
        for p in problems:
            print("   ", p)
        raise SystemExit(1)
    print("[golden-id] サニティ OK: shape/dtype 一致・NaN/Inf なし・自己整合 OK")
    print(f"[golden-id] done. 出力先 {out_dir}/ (synthetic golden は非回帰)")


# ==============================================================
# 訓練ループ
# ==============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/bitnet")
    # 搬送モード (torch import 前の _maybe_transport で処理・argparse には --help 用に定義)。
    ap.add_argument("--copy", default=None, metavar="SRC_DIR",
                    help="搬送: SRC_DIR(git 管理外アセット base) -> ローカル --data-dir へ "
                         "exact コピーのみ実行して終了 (torch 非依存・訓練しない)。")
    ap.add_argument("--publish", default=None, metavar="DST_DIR",
                    help="搬送: ローカル --data-dir -> DST_DIR へ exact コピーのみ実行して "
                         "終了 (torch 非依存・訓練しない・--copy の逆方向)。")
    ap.add_argument("--vocab", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--loss-mode", default="tags", choices=["tags", "all"],
                    help="loss を tags 部のみ or 全体で取る")
    ap.add_argument("--seed", type=int, default=20260620)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch・少数バッチで疎通確認")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dump-golden", action="store_true",
                    help="bitnet_dense_fp32.safetensors から #6 突合用 golden を dump (独立サブモード)")
    ap.add_argument("--identity", action="store_true",
                    help="A2: #1 (synthetic) と identity_cond を混合訓練し、source 別 val "
                         "loss / identity retention を計測。出力は bitnet_dense_identity*。"
                         "未指定なら #4 純 synthetic 訓練 (bitnet_dense*.safetensors を更新)")
    ap.add_argument("--dump-golden-identity", action="store_true",
                    help="A3: identity 条件付き系列の golden を dump (独立サブモード・"
                         "bitnet_dense_identity_fp32.safetensors を使う)")
    ap.add_argument("--distill", action="store_true",
                    help="D3: train に pairs.distill.train.jsonl (Qwen2 蒸留ペア) を混合し "
                         "過学習緩和を狙う。val は #1 と同一の pairs.val.jsonl で不変。"
                         "出力は bitnet_dense_distill* / train_stats_distill.json (別名・"
                         "#1/#4/#6 の重み・golden は無改変)。--identity 未指定時は "
                         "synthetic(4500)+distill を混合 (#1 系の蒸留版)。")
    # --- D5: soft-label KL 蒸留 (案A 共起 teacher) ---
    ap.add_argument("--distill-kl", action="store_true",
                    help="D5: soft-label KL 蒸留 (案A 共起 teacher)。新規ペアは作らず "
                         "cache/danbooru_posts.jsonl の共起から target 分布を軟化し "
                         "L=alpha*T^2*KL(student||teacher)+(1-alpha)*hardCE で学習。"
                         "出力は bitnet_dense_kl* / train_stats_kl.json (別名・"
                         "#1/#4/#6/distill/identity は無改変)。val は #1 と同一 (不変)。"
                         "synthetic(4500) のみを train とする (案A は target 軟化なので "
                         "ペア数は増やさない)。")
    ap.add_argument("--kl-alpha", type=float, default=0.5,
                    help="D5: KL 損失の重み alpha (0=従来 hardCE 完全一致・1=KL のみ)")
    ap.add_argument("--kl-temp", type=float, default=2.0,
                    help="D5: KL 蒸留温度 T (student/teacher logits を T で割って softmax・"
                         "勾配スケール補正に T^2 を乗ずる)。teacher は確率分布なので "
                         "teacher^(1/T) を再正規化して温度を反映する。")
    ap.add_argument("--cooc-main-mass", type=float, default=0.85,
                    help="D5 案A teacher: gold 次タグに置く主質量 (残余を共起候補へ)")
    ap.add_argument("--cooc-temp", type=float, default=2.0,
                    help="D5 案A teacher: 共起スコア softmax 温度 (大=平坦)")
    ap.add_argument("--cooc-topn", type=int, default=32,
                    help="D5 案A teacher: 残余を配る共起候補の上位件数")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="D5 (②): 訓練時 residual dropout 率 (eval で恒等=golden 非回帰)。"
                         "既定 0.0 で従来挙動。")
    # --- D6: 外部教師 (TIPO-200M) soft-label KL 蒸留 (案c) ---
    ap.add_argument("--distill-ext", action="store_true",
                    help="D6: 外部教師 (TIPO-200M) soft-label KL 蒸留。"
                         "dollma_d6_teacher_cache.py が出した COO npz を読んで "
                         "target 分布を軟化し L=alpha*T^2*KL+(1-alpha)*hardCE で学習。"
                         "TIPO 本体は訓練時にロードしない (npz を読むだけ)。"
                         "出力は bitnet_dense_d6* / train_stats_d6.json (別名・"
                         "#1/#4/#6/distill/identity/kl は無改変)。val は #1 と同一 (不変)。"
                         "--kl-alpha/--kl-temp を流用 (--cooc-* は案A 専用のまま)。")
    ap.add_argument("--ext-soft-cache", default=None,
                    help="D6: 外部教師 soft npz パスの接頭辞 "
                         "(既定 data/bitnet/cache/d6_teacher_soft.{train,val}.npz)。")
    # --- C: eval-only ハーネス (訓練せず採点のみ) ---
    ap.add_argument("--eval-only", action="store_true",
                    help="C: 訓練せず --weights の重みで legacy teacher-forcing recall + "
                         "生成 set-metrics を採点し eval_report_<name>.json を出力 "
                         "(独立サブモード・本番重み/golden は無改変)。")
    ap.add_argument("--weights", default=None,
                    help="C: --eval-only で採点する export_safetensors 命名重み "
                         "(例 data/bitnet/bitnet_dense_fp32.safetensors)。")
    ap.add_argument("--eval-name", default=None,
                    help="C: eval_report_<name>.json の name (既定 = 重みファイル basename)。")
    ap.add_argument("--set-ks", default="5,10,20",
                    help="C: set-metrics recall@k の k (カンマ区切り・既定 5,10,20)。")
    ap.add_argument("--dump-persample", action="store_true",
                    help="C (施策C seed sweep): --eval-only で diverse_a/b の per-sample "
                         "F1/Jaccard/precision/recall 配列を eval_persample_<name>.npz に保存 "
                         "(rows と同順・NaN=skip/未定義)。paired bootstrap/t 用。"
                         "既定 off で従来挙動 (golden/既存テスト非回帰)。")
    # 決裁 (2026-06-24 ユーザー): 施策B (入力多様化) のレシピ既定化を確定した。
    #   ただし本引数の default=None (= #1 凍結経路) は、A (同一性条件付き) の実ペア増と
    #   束ねる「次の出荷リトレイン」まで意図的に据え置く。ここで default を diverse ファイルに
    #   変えると指定時の別名出力分岐 (basename 由来サフィックス) に落ち、bitwise 非回帰アンカー
    #   および本番 bitnet_dense*/golden 据え置きという決裁の遅延条項を破ることになるため。
    #   A 時に diverse train ファイル (pairs.train.diverse_b2000.jsonl 等) を既定指定し、
    #   正典重みと C++ 推論 golden を 1 回で再生成する。詳細は docs/training-spec.md §14.8。
    # 施策 D (容量増のコード化): アーキ次元を CLI でパラメタ化する。
    #   --arch base (既定・現行と完全一致・非回帰) / d80m (n_layers/ffn 拡張 ~80M)。
    #   個別上書き --d-model/--n-layers/--ffn は arch を土台に被せる (直交)。
    #   default はいずれも None で、引数なし = 現行 base アーキ。
    ap.add_argument("--arch", default=None, choices=sorted(ARCH_CONFIGS),
                    help="アーキ config (base=現行32.98M / d80m=容量増79.9M)。"
                         "未指定は base (現行と非回帰)。")
    ap.add_argument("--d-model", type=int, default=None,
                    help="d_model 個別上書き (n_heads で割り切れること)。通常は据え置き。")
    ap.add_argument("--n-layers", type=int, default=None,
                    help="n_layers 個別上書き (施策 D の容量ノブ)。")
    ap.add_argument("--ffn", type=int, default=None,
                    help="ffn 次元個別上書き (施策 D の容量ノブ)。")
    ap.add_argument("--train-file", default=None,
                    help="B (施策B 入力多様化): plain #1 訓練の train ファイルを差し替える "
                         "(既定 None = data_dir/pairs.train.jsonl で従来挙動 完全不変)。"
                         "指定時は出力を別名 (basename 由来サフィックス) にして "
                         "本番 bitnet_dense*/golden を無改変に保つ。val は pairs.val.jsonl 不変。")
    # --- F-0b (G-2a): rejection-sampling SFT (RAFT) ---
    #   正典 bitnet_dense から層状に warm-start し、best-of-N 勝者ペアで低 LR・少 epoch の
    #   追加学習を行う。破滅的忘却回避が最優先ゆえ既存訓練 (lr 3e-4/30ep) より 1〜2 桁小さい
    #   LR・2〜4 epoch を推奨。出力は隔離別名 bitnet_dense_sft{,_fp32} / train_stats_sft
    #   (正典 bitnet_dense*/golden は G-3 合格まで無改変)。source=="rejection_sft" は
    #   build_sequence の synthetic 分岐 (1-<sep>) で読める。
    ap.add_argument("--sft-rejection", action="store_true",
                    help="F-0b (G-2a): 正典 bitnet_dense から層状 warm-start し "
                         "best-of-N 勝者ペア (--sft-data) で RAFT-SFT。出力は隔離 "
                         "bitnet_dense_sft{,_fp32} / train_stats_sft (正典無改変)。"
                         "identity/distill 系とは併用不可。")
    ap.add_argument("--sft-data", default=None,
                    help="F-0b: RAFT-SFT 勝者ペア jsonl (既定 data/rollouts/sft_bestofn.jsonl)。"
                         "schema は synthetic 経路互換 (source=rejection_sft)。")
    ap.add_argument("--sft-init", default=None,
                    help="F-0b: SFT 層状 warm-start 元の export 命名 FP32 safetensors "
                         "(既定 <data-dir>/bitnet_dense_fp32.safetensors = 正典)。")
    args = ap.parse_args()

    # --- 施策 D: アーキ次元を config 駆動で確定 (全サブモードより前・1 回だけ) ---
    #   引数なしなら base 維持 (= 現行と bitwise 非回帰)。golden/eval/train 全経路が
    #   この再束縛後のモジュール定数を読むため、ここで 1 回呼べば全員が config 連動になる。
    arch_info = apply_arch(arch=args.arch, d_model=args.d_model,
                           n_layers=args.n_layers, ffn=args.ffn)
    print(f"[arch] {arch_info['vocab']}v d_model={arch_info['d_model']} "
          f"n_layers={arch_info['n_layers']} n_heads={arch_info['n_heads']} "
          f"ffn={arch_info['ffn']} max_seq={arch_info['max_seq']} "
          f"-> param_count={arch_info['param_count']:,}")

    # --- F-0b: SFT は identity/distill 系と併用不可 (層状 warm-start は synthetic 経路のみ) ---
    if args.sft_rejection and (args.identity or args.distill or args.distill_kl
                               or args.distill_ext):
        raise SystemExit("[sft] --sft-rejection は --identity/--distill*/ と併用不可")

    # --- golden dump サブモード (訓練は行わず golden だけ出力) ---
    if args.dump_golden:
        dump_golden(args)
        return
    if args.dump_golden_identity:
        dump_golden_identity(args)
        return
    if args.eval_only:
        eval_only(args)
        return

    import random
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)

    data_dir = args.data_dir
    vocab_path = args.vocab or os.path.join(data_dir, "vocab.json")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} torch={torch.__version__} "
          f"cuda_avail={torch.cuda.is_available()}")
    if device == "cuda":
        print(f"[env] gpu={torch.cuda.get_device_name(0)} "
              f"cap={torch.cuda.get_device_capability(0)}")
    else:
        print("[env] CPU 訓練 (CUDA 不可)")

    tok = Tokenizer(vocab_path)
    print(f"[tok] vocab_size={tok.vocab_size()} tags={len(tok.tags)}")

    # --- データ読み込み (A2: --identity で #1 + identity_cond を混合) ---
    #   B: --train-file 指定時は plain #1 train を差し替える (val は不変)。
    #   未指定なら従来通り pairs.train.jsonl (既定挙動 完全不変)。
    train_file = args.train_file or os.path.join(data_dir, "pairs.train.jsonl")
    syn_train = load_pairs(train_file)
    syn_val = load_pairs(os.path.join(data_dir, "pairs.val.jsonl"))
    if args.train_file:
        print(f"[data] train-file 差し替え: {train_file} ({len(syn_train)} 件)")
    if args.identity:
        id_train = load_pairs(os.path.join(data_dir, "pairs.identity.train.jsonl"))
        id_val = load_pairs(os.path.join(data_dir, "pairs.identity.val.jsonl"))
        if args.smoke:
            syn_train, syn_val = syn_train[:64], syn_val[:32]
            id_train, id_val = id_train[:64], id_val[:32]
            args.epochs = 1
        # 混合 = 単純結合 + シャッフル (混合比は 1:1 相当・両ファイルとも 4500/500)。
        # シャッフルは make_batches 側でも毎 epoch 行うが、source の偏りを避けるため
        # 結合直後にも 1 度シャッフルしておく (seed 固定で再現的)。
        train_rows = syn_train + id_train
        if args.distill:
            # identity+distill 併用 (任意): 蒸留ペアも train に足す。val は不変。
            dis_tr = load_pairs(os.path.join(data_dir, "pairs.distill.train.jsonl"))
            if args.smoke:
                dis_tr = dis_tr[:64]
            train_rows = train_rows + dis_tr
        val_rows = syn_val + id_val
        rng_mix = random.Random(seed)
        rng_mix.shuffle(train_rows)
        n_syn_tr = sum(1 for r in train_rows if r.get("source") == "synthetic")
        n_id_tr = sum(1 for r in train_rows if r.get("source") == "identity_cond")
        n_syn_va = sum(1 for r in val_rows if r.get("source") == "synthetic")
        n_id_va = sum(1 for r in val_rows if r.get("source") == "identity_cond")
        print(f"[data] MIXED train={len(train_rows)} "
              f"(synthetic={n_syn_tr} identity_cond={n_id_tr}) "
              f"val={len(val_rows)} (synthetic={n_syn_va} identity_cond={n_id_va}) "
              f"loss_mode={args.loss_mode}")
    elif args.distill:
        # D3: synthetic(4500) + Qwen2 蒸留(3000) を混合。val は #1 と同一 (不変)。
        #   蒸留ペアは synthetic と同じ 1-<sep> 経路 (source=="llm_distill" は
        #   build_sequence の synthetic 分岐へ落ちる → 既存ロジックで読める)。
        distill_train = load_pairs(os.path.join(data_dir, "pairs.distill.train.jsonl"))
        if args.smoke:
            syn_train, syn_val = syn_train[:64], syn_val[:32]
            distill_train = distill_train[:64]
            args.epochs = 1
        train_rows = syn_train + distill_train
        val_rows = syn_val  # val は絶対不変 (#1 の pairs.val.jsonl 500 件)
        rng_mix = random.Random(seed)
        rng_mix.shuffle(train_rows)  # source の塊を崩す (seed 固定で再現的)
        n_syn_tr = sum(1 for r in train_rows if r.get("source") == "synthetic")
        n_dis_tr = sum(1 for r in train_rows if r.get("source") == "llm_distill")
        print(f"[data] DISTILL-MIXED train={len(train_rows)} "
              f"(synthetic={n_syn_tr} llm_distill={n_dis_tr}) "
              f"val={len(val_rows)} (synthetic・不変) loss_mode={args.loss_mode}")
    elif args.distill_ext:
        # D6: synthetic(4500) のみ。案c も target 分布を軟化するのでペア数は増やさない。
        #   val は #1 と同一 (固定 val 500・不変・#1 との突合用)。
        train_rows = syn_train
        val_rows = syn_val
        if args.smoke:
            train_rows = train_rows[:128]
            val_rows = val_rows[:64]
            args.epochs = 1
        print(f"[data] EXT-DISTILL train={len(train_rows)} val={len(val_rows)} "
              f"loss_mode={args.loss_mode} (synthetic only・target は案c TIPO 教師 npz で軟化)")
    elif args.distill_kl:
        # D5: synthetic(4500) のみ。案A は target 分布を軟化するのでペア数は増やさない。
        #   val は #1 と同一 (固定 val 500・不変・#1 との突合用)。
        train_rows = syn_train
        val_rows = syn_val
        if args.smoke:
            train_rows = train_rows[:128]
            val_rows = val_rows[:64]
            args.epochs = 1
        print(f"[data] KL-DISTILL train={len(train_rows)} val={len(val_rows)} "
              f"loss_mode={args.loss_mode} (synthetic only・target は案A 共起 teacher で軟化)")
    elif args.sft_rejection:
        # F-0b (G-2a): best-of-N 勝者ペアで RAFT-SFT。train = 勝者ペアのみ。
        #   val は #1 と同一 (pairs.val・追加学習で in-dist が崩れないかの monitor 用)。
        #   本ゲート (diverse set-F1 非退行) は別途 --eval-only で採点する。
        default_sft = os.path.join(
            os.path.dirname(data_dir.rstrip("/\\")) or ".",
            "rollouts", "sft_bestofn.jsonl")
        sft_path = args.sft_data or default_sft
        if not os.path.exists(sft_path):
            raise FileNotFoundError(f"{sft_path} が無い (--sft-data / G-1 の勝者ペア)")
        train_rows = load_pairs(sft_path)
        val_rows = syn_val
        if args.smoke:
            train_rows = train_rows[:32]
            val_rows = val_rows[:32]
            args.epochs = 1
        n_ja = sum(1 for r in train_rows if r.get("lang") == "ja")
        n_en = sum(1 for r in train_rows if r.get("lang") == "en")
        n_rej = sum(1 for r in train_rows if r.get("source") == "rejection_sft")
        print(f"[data] SFT-REJECTION train={len(train_rows)} "
              f"(rejection_sft={n_rej} ja={n_ja} en={n_en}) val={len(val_rows)} "
              f"loss_mode={args.loss_mode} sft_data={sft_path}")
    else:
        train_rows = syn_train
        val_rows = syn_val
        if args.smoke:
            train_rows = train_rows[:128]
            val_rows = val_rows[:64]
            args.epochs = 1
        print(f"[data] train={len(train_rows)} val={len(val_rows)} "
              f"loss_mode={args.loss_mode} (synthetic only)")

    train_ds = PairDataset(tok, train_rows, args.max_len)
    val_ds = PairDataset(tok, val_rows, args.max_len)

    # シーケンス長統計 (ログ用)
    lens = [len(s[0]) for s in train_ds.samples]
    print(f"[data] seq_len min={min(lens)} max={max(lens)} "
          f"mean={sum(lens)/len(lens):.1f}")

    model = BitNetDense(dropout=args.dropout).to(device)
    if args.dropout > 0.0:
        print(f"[model] dropout={args.dropout} (訓練時のみ・eval で恒等=golden 非回帰)")
    # F-0b (G-2a): SFT は乱数初期化でなく正典 bitnet_dense から層状 warm-start する。
    #   これで「追加学習」= 破滅的忘却回避の前提が成立する (乱数から学ぶ通常訓練と別物)。
    if args.sft_rejection:
        init_path = args.sft_init or os.path.join(
            data_dir, "bitnet_dense_fp32.safetensors")
        if not os.path.exists(init_path):
            raise FileNotFoundError(
                f"{init_path} が無い (--sft-init / 正典 warm-start 元)")
        n_loaded = _load_export_into_model(model, init_path)
        print(f"[sft] 層状 warm-start: {init_path} から {n_loaded} テンソルを strict ロード "
              f"(乱数初期化を上書き)")
    n_params = sum(p.numel() for p in model.parameters())
    # embed tied: lm_head は別パラメータでないので二重カウントなし。
    #   期待値は config 連動の compute_param_count() (C++ param_count() 同式)。
    expect_pc = compute_param_count()
    print(f"[model] params={n_params:,} (bitnet.hpp param_count={expect_pc:,})")
    assert n_params == expect_pc, (
        f"param 数不一致: model={n_params} != compute_param_count={expect_pc}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.95))

    # --- D5: 案A 共起 teacher を 1 度だけ構築 (KL 蒸留時のみ) ---
    teacher = None
    soft_cache = None
    if args.distill_ext:
        # D6: 外部教師 npz を読むだけ (TIPO 本体はロードしない)。
        #   --ext-soft-cache は train/val 共通の接頭辞 (例 ".../d6_teacher_soft")。
        #   未指定なら data/bitnet/cache/d6_teacher_soft を既定接頭辞とする。
        ext_prefix = args.ext_soft_cache or os.path.join(
            data_dir, "cache", "d6_teacher_soft")
        tr_npz = ext_prefix + ".train.npz"
        teacher = ExternalTagTeacher(tok, tr_npz)
        teacher.bind(train_ds, args.max_len)
        print(f"[teacher] 案c 外部教師 (TIPO-200M) npz 読込 {tr_npz} "
              f"(samples={teacher.n_samples}) | KL alpha={args.kl_alpha} T={args.kl_temp}")
        import time as _t
        _ts = _t.time()
        soft_cache = precompute_soft_targets(teacher, train_ds, args.max_len)
        print(f"[teacher] soft target 事前計算 {len(soft_cache)} サンプル "
              f"({_t.time()-_ts:.1f}s・以後 epoch で再計算しない)")
    elif args.distill_kl:
        posts_path = os.path.join(data_dir, "cache", "danbooru_posts.jsonl")
        if not os.path.exists(posts_path):
            raise FileNotFoundError(
                f"{posts_path} が無い (案A 共起 teacher の供給源)。dataset-spec §1.2 参照。")
        teacher = CoOccurrenceTeacher(
            tok, posts_path, main_mass=args.cooc_main_mass,
            t_cooc=args.cooc_temp, topn=args.cooc_topn)
        print(f"[teacher] 案A 共起 teacher 構築済 "
              f"(main_mass={args.cooc_main_mass} t_cooc={args.cooc_temp} "
              f"topn={args.cooc_topn}) | KL alpha={args.kl_alpha} T={args.kl_temp}")
        import time as _t
        _ts = _t.time()
        soft_cache = precompute_soft_targets(teacher, train_ds, args.max_len)
        print(f"[teacher] soft target 事前計算 {len(soft_cache)} サンプル "
              f"({_t.time()-_ts:.1f}s・以後 epoch で再計算しない)")

    n_train_batches = (len(train_ds) + args.batch_size - 1) // args.batch_size
    total_steps = n_train_batches * args.epochs

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(total_steps - args.warmup, 1)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * prog))

    rng = random.Random(seed)
    history = []
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        ep_loss = 0.0
        ep_tok = 0
        for batch in make_batches(train_ds, args.batch_size, True, rng):
            lr = lr_at(step)
            for g in opt.param_groups:
                g["lr"] = lr
            inp, tgt = collate(batch, args.max_len, args.loss_mode)
            inp = inp.to(device)
            tgt = tgt.to(device)
            logits = model(inp)
            if teacher is not None:
                # D5: L = alpha*T^2*KL(student||teacher) + (1-alpha)*hardCE。
                #   target 区間マスク (tgt!=-100) を hardCE/KL とも共有 (§9.2 厳守)。
                hard_ce = F.cross_entropy(
                    logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1),
                    ignore_index=-100)
                T = args.kl_temp
                soft_t = build_soft_target_tensor(
                    teacher, batch, tgt.cpu(), args.max_len,
                    soft_cache=soft_cache).to(device)  # [B,L-1,V]
                # teacher を温度 T で軟化: teacher^(1/T) を再正規化 (確率分布 teacher 版の温度)。
                valid = (tgt != -100)                       # [B,L-1]
                Bb, Lm1 = tgt.shape
                logp_s = F.log_softmax(logits / T, dim=-1)  # [B,L-1,V]
                soft_pow = soft_t.clamp_min(0).pow(1.0 / T)
                soft_norm = soft_pow / soft_pow.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                # KL(teacher||student) = sum teacher*(log teacher - log p_s)。
                # 定数 (teacher エントロピー) は勾配に効かないので学生側 cross 項のみで十分だが
                # 値の解釈性のため完全 KL を計算する (valid 位置のみ平均)。
                logt = torch.log(soft_norm.clamp_min(1e-12))
                kl_pos = (soft_norm * (logt - logp_s)).sum(dim=-1)  # [B,L-1]
                nval = valid.sum().clamp_min(1)
                kl = (kl_pos * valid).sum() / nval
                loss = args.kl_alpha * (T * T) * kl + (1.0 - args.kl_alpha) * hard_ce
            else:
                loss = F.cross_entropy(
                    logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1),
                    ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n_tok = (tgt != -100).sum().item()
            ep_loss += loss.item() * n_tok
            ep_tok += n_tok
            step += 1
            if args.smoke and step >= 4:
                break
        train_loss = ep_loss / max(ep_tok, 1)
        val_loss, recall, rand_base, n_tgt = eval_loss_and_recall(
            model, val_ds, args.batch_size, args.max_len, args.loss_mode,
            device, args.topk)
        elapsed = time.time() - t0
        hrec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_val_gap": val_loss - train_loss,  # D4 過学習主指標 (+ で val>train)
            f"top{args.topk}_recall": recall,
            "lr": lr,
            "elapsed_s": elapsed,
        }
        if args.identity:
            # source 別 val loss / recall を毎 epoch 出す (混合の偏りを監視)。
            by_src = eval_loss_and_recall_by_source(
                model, val_ds, args.batch_size, args.max_len, args.loss_mode,
                device, args.topk)
            syn = by_src.get("synthetic", {})
            idc = by_src.get("identity_cond", {})
            rk = f"top{args.topk}_recall"
            hrec["val_loss_synthetic"] = syn.get("val_loss")
            hrec["val_loss_identity_cond"] = idc.get("val_loss")
            hrec[f"top{args.topk}_recall_synthetic"] = syn.get(rk)
            hrec[f"top{args.topk}_recall_identity_cond"] = idc.get(rk)
            print(f"[ep {epoch:3d}] train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} top{args.topk}_recall={recall:.4f} "
                  f"| syn(vl={syn.get('val_loss'):.4f} r={syn.get(rk):.4f}) "
                  f"id(vl={idc.get('val_loss'):.4f} r={idc.get(rk):.4f}) "
                  f"lr={lr:.2e} t={elapsed:.1f}s")
        else:
            print(f"[ep {epoch:3d}] train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} top{args.topk}_recall={recall:.4f} "
                  f"(rand={rand_base:.4f}) lr={lr:.2e} t={elapsed:.1f}s")
        history.append(hrec)

    train_time = time.time() - t0

    # ---- 重み出力 ----
    # footgun 対策: --smoke は疎通確認専用なので本番の重み/stats を絶対に上書きしない。
    #   smoke 時は別名 (bitnet_dense_smoke*.safetensors / train_stats_smoke.json) へ書く。
    #   本番訓練モード (--smoke なし) の挙動・ファイル名は一切変えない。
    #   identity 版 (--identity) は #4 純 dense と別名 (bitnet_dense_identity*) に分ける:
    #     #4 の bitnet_dense_fp32.safetensors は #6 golden 突合に使われており壊せない。
    #     混合訓練したモデルは別物なので名前を分け、#4 と #6 を保全する (設計判断)。
    if args.sft_rejection:
        base = "bitnet_dense_sft"
        stats_base = "train_stats_sft"
    elif args.distill_ext:
        base = "bitnet_dense_d6"
        stats_base = "train_stats_d6"
    elif args.distill_kl:
        base = "bitnet_dense_kl"
        stats_base = "train_stats_kl"
    elif args.identity and args.distill:
        base = "bitnet_dense_identity_distill"
        stats_base = "train_stats_identity_distill"
    elif args.distill:
        base = "bitnet_dense_distill"
        stats_base = "train_stats_distill"
    elif args.identity and args.train_file:
        # [B-merge-at-A] 正典化まとめ焼き: B(--train-file=diverse_b2000 で syn を差し替え)
        #   と A(--identity で identity_cond を混合) の両方が真のとき merged 別名へ出す。
        #   単独 elif args.identity: より必ず前に置く (identity 分岐が先勝ちすると
        #   A 単体別名 bitnet_dense_identity* に落ちて衝突するため)。distill 系との
        #   優先順位 (distill_ext/distill_kl/identity+distill/distill) は不変。
        base = "bitnet_dense_merged"
        stats_base = "train_stats_merged"
    elif args.identity:
        base = "bitnet_dense_identity"
        stats_base = "train_stats_identity"
    elif args.train_file:
        # B: plain #1 と同設定で train だけ差し替えた版。本番 bitnet_dense*/golden は
        #   無改変に保つため、train-file basename 由来サフィックスで別名出力する。
        #   例 pairs.train.diverse_b.jsonl -> bitnet_dense_diverse_b{,_fp32}.safetensors /
        #      train_stats_diverse_b.json。
        bn = os.path.basename(args.train_file)
        # "pairs.train.<suffix>.jsonl" から <suffix> を取り出す (無ければ basename 全体を流用)。
        m = re.match(r"^pairs\.train\.(.+)\.jsonl$", bn)
        suffix = m.group(1) if m else os.path.splitext(bn)[0]
        base = "bitnet_dense_" + suffix
        stats_base = "train_stats_" + suffix
        print(f"[B] train-file 別名出力: base={base} stats_base={stats_base}")
    else:
        base = "bitnet_dense"
        stats_base = "train_stats"
    if args.smoke:
        fp16_path = os.path.join(data_dir, base + "_smoke.safetensors")
        fp32_path = os.path.join(data_dir, base + "_smoke_fp32.safetensors")
        stats_filename = stats_base + "_smoke.json"
        print(f"[smoke] 出力は smoke 別名へ ({base}*.safetensors / "
              f"{stats_base}.json は上書きしない)")
    else:
        fp16_path = os.path.join(data_dir, base + ".safetensors")
        fp32_path = os.path.join(data_dir, base + "_fp32.safetensors")
        stats_filename = stats_base + ".json"
    keys = export_safetensors(model, fp16_path, torch.float16)
    export_safetensors(model, fp32_path, torch.float32)
    print(f"[save] {fp16_path} ({len(keys)} tensors, FP16)")
    print(f"[save] {fp32_path} (FP32 golden)")

    # ---- 読み戻しサニティ ----
    prob16 = sanity_reload(fp16_path)
    prob32 = sanity_reload(fp32_path)
    if prob16 or prob32:
        print("[sanity] PROBLEMS:")
        for p in prob16 + prob32:
            print("   ", p)
    else:
        print("[sanity] OK: 全テンソル name/shape/dtype 整合・NaN/Inf なし")

    fp16_mb = os.path.getsize(fp16_path) / (1024 * 1024)
    fp32_mb = os.path.getsize(fp32_path) / (1024 * 1024)

    # ---- 最終 val recall (再計測・ログ確定) ----
    final_loss, final_recall, rand_base, n_tgt = eval_loss_and_recall(
        model, val_ds, args.batch_size, args.max_len, args.loss_mode,
        device, args.topk)

    result = {
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "final_val_loss": final_loss,
        f"final_top{args.topk}_recall": final_recall,
        "random_baseline_recall": rand_base,
        "recall_vs_random_x": (final_recall / rand_base) if rand_base else None,
        "val_target_tags": n_tgt,
        "train_time_s": round(train_time, 1),
    }

    # ---- D5/D6: val_loss 反転(底) epoch + 生成多様性 (§10.3 同等) ----
    if args.distill_kl or args.distill_ext:
        vls = [h["val_loss"] for h in history]
        argmin_ep = int(min(range(len(vls)), key=lambda i: vls[i])) if vls else None
        diversity = eval_generation_diversity(
            model, tok, val_rows, device, args.max_len)
        result["val_loss_inversion_epoch"] = argmin_ep
        result["val_loss_min"] = (min(vls) if vls else None)
        result["final_train_val_gap"] = (history[-1]["train_val_gap"]
                                         if history else None)
        result["generation_diversity"] = diversity
        result["kl_settings"] = {
            "kl_alpha": args.kl_alpha, "kl_temp": args.kl_temp,
            "cooc_main_mass": args.cooc_main_mass, "cooc_temp": args.cooc_temp,
            "cooc_topn": args.cooc_topn, "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "teacher": ("TIPO-200M (apache-2.0)・案c 外部教師 soft npz" if args.distill_ext
                        else "案A 共起 (cache/danbooru_posts.jsonl・prefix 条件付き soft label)"),
        }
        if args.distill_ext:
            result["kl_settings"]["ext_soft_cache"] = (
                (args.ext_soft_cache or os.path.join(data_dir, "cache",
                 "d6_teacher_soft")) + ".train.npz")
        print(f"[D5/D6] val_loss 反転(底) epoch={argmin_ep} (min={min(vls):.4f}) "
              f"final gap={result['final_train_val_gap']:.4f}")
        print(f"[D5/D6] 生成多様性: unique_tags={diversity['corpus_unique_tags']} "
              f"per_seq_unique={diversity['mean_per_seq_unique']:.3f} "
              f"norm_entropy={diversity['norm_entropy']:.3f} "
              f"(n_cases={diversity['n_cases']})")

    # ---- A2: source 別 val 指標 + identity retention rate ----
    by_source = None
    retention = None
    if args.identity:
        by_source = eval_loss_and_recall_by_source(
            model, val_ds, args.batch_size, args.max_len, args.loss_mode,
            device, args.topk)
        mean_ret, n_ret, ret_details = eval_identity_retention(
            model, tok, val_rows, device, args.max_len)
        retention = {
            "mean_retention": mean_ret,
            "n_cases": n_ret,
            "definition": "mean over identity_cond val ( |生成 target に出た identity 集合| "
                          "/ |入力 identity 集合(語彙内)| )。prompt=<bos> identity <sep> "
                          "scene_text(greedy) <sep>、greedy デコード (GREEDY_STOP_RULE 準拠)。",
        }
        result["by_source"] = by_source
        result["identity_retention"] = retention
        rk = f"top{args.topk}_recall"
        syn = by_source.get("synthetic", {})
        idc = by_source.get("identity_cond", {})
        print(f"[A2] source 別 val: "
              f"synthetic(val_loss={syn.get('val_loss'):.4f} {rk}={syn.get(rk):.4f} "
              f"n={syn.get('n_samples')}) "
              f"identity_cond(val_loss={idc.get('val_loss'):.4f} {rk}={idc.get(rk):.4f} "
              f"n={idc.get('n_samples')})")
        print(f"[A2] identity retention rate = {mean_ret:.4f} "
              f"(n_cases={n_ret}・1.0=生成 target が入力 identity を全部含む)")

    stats = {
        "seed": seed,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "mode": ("sft_rejection" if args.sft_rejection
                 else "kl_distill_ext" if args.distill_ext
                 else "kl_distill_cooc" if args.distill_kl
                 else "identity_distill_mixed" if (args.identity and args.distill)
                 else "distill_mixed" if args.distill
                 else "identity_mixed" if args.identity
                 else "synthetic_only"),
        "hyperparams": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup": args.warmup,
            "max_len": args.max_len,
            "loss_mode": args.loss_mode,
            "topk": args.topk,
        },
        "data": {
            "train": len(train_rows),
            "val": len(val_rows),
            "vocab_size": tok.vocab_size(),
            "mixed": bool(args.identity or args.distill),
            "n_synthetic_train": sum(1 for r in train_rows
                                     if r.get("source") == "synthetic"),
            "n_distill_train": sum(1 for r in train_rows
                                   if r.get("source") == "llm_distill"),
            "n_identity_train": sum(1 for r in train_rows
                                    if r.get("source") == "identity_cond"),
            "distill": bool(args.distill),
        },
        "model": {
            "params": n_params,
            "arch": (args.arch or "base"),
            "bitnet_hpp_param_count": compute_param_count(),
            "fp16_mb": round(fp16_mb, 3),
            "fp32_mb": round(fp32_mb, 3),
        },
        "result": result,
        "tensor_keys": keys,
        "history": history,
    }
    stats_path = os.path.join(data_dir, stats_filename)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(f"[save] {stats_path}")
    print(f"[done] final val_loss={final_loss:.4f} "
          f"top{args.topk}_recall={final_recall:.4f} "
          f"(rand={rand_base:.4f}, {final_recall/rand_base:.1f}x) "
          f"size FP16={fp16_mb:.1f}MB train_time={train_time:.1f}s")


if __name__ == "__main__":
    main()
