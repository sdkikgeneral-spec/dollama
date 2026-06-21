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
import json
import math
import os
import re
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================
# アーキ定数 (bitnet.hpp と厳密一致)
# ==============================================================
VOCAB_SIZE = 4999
D_MODEL = 512
N_LAYERS = 8
N_HEADS = 8
HEAD_DIM = D_MODEL // N_HEADS  # 64
FFN_DIM = 1792
MAX_SEQ_LEN = 64
ROPE_BASE = 10000.0
RMS_EPS = 1e-5

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
    """1 層: pre-RMSNorm attention + pre-RMSNorm SwiGLU FFN。bias なし。"""

    def __init__(self):
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
        h = h + self.wo(ctx)
        # --- FFN (SwiGLU) ---
        x = self.ffn_norm(h)
        inter = F.silu(self.w_gate(x)) * self.w_up(x)
        h = h + self.w_down(inter)
        return h


class BitNetDense(nn.Module):
    """bitnet.hpp の dense (FP) 等価モデル。embed tied lm_head。"""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.layers = nn.ModuleList([Block() for _ in range(N_LAYERS)])
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
        h = self.embed(tokens)  # [B,S,D]
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
    """1 ペアを自己回帰列に整形する。

    形式: <bos> text(タグ化) <sep> tags <eos>
      - text 側: tokenizer.hpp::encode_text の greedy 最長一致は再現が重いので、
        ここではデータの構造を利用する。text は tags 列から逆生成された自然文なので、
        text 中に現れる語彙タグを greedy 最長一致で拾う。tokenizer.hpp と同じく
        非語彙単語はスキップする。
      - tags 側: 正準順序の target タグ列をそのまま id 化 (encode の tag_to_id 相当)。
    返り値: (ids[list], tags_start_index)  tags_start_index は <sep> の次の位置。
    """
    # --- text 側: greedy 最長一致 (tokenizer.hpp::encode_text 相当) ---
    text_ids = encode_text_greedy(tok, row["text"])
    # --- tags 側 ---
    tag_ids = [tok.tag_to_id_lookup(t) for t in row["tags"]]

    ids = [TOK_BOS] + text_ids + [TOK_SEP] + tag_ids + [TOK_EOS]
    sep_pos = 1 + len(text_ids)  # <sep> の位置
    tags_start = sep_pos + 1     # tags 部の先頭位置
    if len(ids) > max_len:
        ids = ids[:max_len]
        ids[-1] = TOK_EOS
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
    """整形済みシーケンスを保持する単純なデータセット。"""

    def __init__(self, tok, rows, max_len):
        self.samples = []
        for r in rows:
            ids, tags_start = build_sequence(tok, r, max_len)
            self.samples.append((ids, tags_start))

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
    L = max(len(ids) for ids, _ in batch)
    L = min(L, max_len)
    inp = torch.full((B, L - 1), TOK_PAD, dtype=torch.long)
    tgt = torch.full((B, L - 1), -100, dtype=torch.long)  # ignore_index=-100
    for bi, (ids, tags_start) in enumerate(batch):
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
        for bi, (ids, tags_start) in enumerate(batch):
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
# 訓練ループ
# ==============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/bitnet")
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
    args = ap.parse_args()

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

    train_rows = load_pairs(os.path.join(data_dir, "pairs.train.jsonl"))
    val_rows = load_pairs(os.path.join(data_dir, "pairs.val.jsonl"))
    if args.smoke:
        train_rows = train_rows[:128]
        val_rows = val_rows[:64]
        args.epochs = 1
    print(f"[data] train={len(train_rows)} val={len(val_rows)} "
          f"loss_mode={args.loss_mode}")

    train_ds = PairDataset(tok, train_rows, args.max_len)
    val_ds = PairDataset(tok, val_rows, args.max_len)

    # シーケンス長統計 (ログ用)
    lens = [len(s[0]) for s in train_ds.samples]
    print(f"[data] seq_len min={min(lens)} max={max(lens)} "
          f"mean={sum(lens)/len(lens):.1f}")

    model = BitNetDense().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    # embed tied: lm_head は別パラメータでないので二重カウントなし。
    print(f"[model] params={n_params:,} (bitnet.hpp param_count=32,976,896)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.95))

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
        print(f"[ep {epoch:3d}] train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} top{args.topk}_recall={recall:.4f} "
              f"(rand={rand_base:.4f}) lr={lr:.2e} t={elapsed:.1f}s")
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            f"top{args.topk}_recall": recall,
            "lr": lr,
            "elapsed_s": elapsed,
        })

    train_time = time.time() - t0

    # ---- 重み出力 ----
    fp16_path = os.path.join(data_dir, "bitnet_dense.safetensors")
    fp32_path = os.path.join(data_dir, "bitnet_dense_fp32.safetensors")
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

    stats = {
        "seed": seed,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
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
        },
        "model": {
            "params": n_params,
            "bitnet_hpp_param_count": 32976896,
            "fp16_mb": round(fp16_mb, 3),
            "fp32_mb": round(fp32_mb, 3),
        },
        "result": {
            "final_train_loss": history[-1]["train_loss"] if history else None,
            "final_val_loss": final_loss,
            f"final_top{args.topk}_recall": final_recall,
            "random_baseline_recall": rand_base,
            "recall_vs_random_x": (final_recall / rand_base) if rand_base else None,
            "val_target_tags": n_tgt,
            "train_time_s": round(train_time, 1),
        },
        "tensor_keys": keys,
        "history": history,
    }
    stats_path = os.path.join(data_dir, "train_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(f"[save] {stats_path}")
    print(f"[done] final val_loss={final_loss:.4f} "
          f"top{args.topk}_recall={final_recall:.4f} "
          f"(rand={rand_base:.4f}, {final_recall/rand_base:.1f}x) "
          f"size FP16={fp16_mb:.1f}MB train_time={train_time:.1f}s")


if __name__ == "__main__":
    main()
