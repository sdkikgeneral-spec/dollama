# -*- coding: utf-8 -*-
"""dollama Phase 4 蒸留 D6 (案c) — 外部教師 (TIPO-200M / DanTagGen) soft target キャッシュ。

D5 (案A 共起 teacher) は過学習を抑制したが top10 recall を底上げしなかった
(soft 追従と one-hot val recall のトレードオフ・training-spec §11)。D6 は
**真の外部知識転移** を狙い、TIPO-200M / DanTagGen が学習した「条件付きタグ集合」を
自作 4999 vocab の疎 soft target に注入する。

設計の絶対制約 (承認済みプラン・案b-tagset):
  TIPO の vocab32013 は SentencePiece BPE で「1タグ=複数 subword」。next-subword
  logits を自作タグ単位4999分布に直接写像してはいけない (案b-logit は却下)。
  必ず TIPO に "タグ補完を生成" させ、出力された "タグ文字列" を自作 vocab に写像する:
    1. TIPO 出力をカンマで split (空白では割らない=`long hair` を壊さない)。
    2. 各タグ片を train_bitnet.Tokenizer.normalize で正規化 (import 再利用・二重実装禁止)。
    3. data/bitnet/vocab.json に完全一致引き。
    4. vocab 外タグは drop し in-vocab 質量で再正規化 (alias 表は作らない)。

teacher インターフェースは D5 CoOccurrenceTeacher と同一:
    soft_target(prefix_ids, gold_id) -> dict{vocab_id: prob}
  gold に main_mass、残余 (1-main_mass) を「TIPO が予測した次タグ候補」上位 topn へ
  温度付きで配る。共起カウントの代わりに TIPO 生成由来のタグ頻度を使う。

コスト圧縮 (per-sample 粒度):
  generate は ~0.9s/sample (FP32 batched, N=8)。per-position 個別生成は破綻するので、
  1 サンプルにつき TIPO へ seed タグ群を渡して N 回 generate → サンプル単位の
  タグ頻度分布を作り、そのサンプルの全 target 位置で共有する。実測外挿 ~1.3h/5000 件
  (overnight 余裕)。「TIPO が学習した条件付きタグ集合」を soft target に注入する D6 の
  目的は per-sample 共有でも達せる。

出力 (D6 別名のみ・本番重み/golden は触らない):
  data/bitnet/cache/d6_teacher_soft.{train,val}.npz  position 軸付き COO 疎形式
  data/bitnet/cache/d6_teacher_stats.json            OOV保持率・平均エントロピー
  --probe-only (T3/D6-0): teacher-alone top10 recall を D5 共起と A/B 出力 (訓練しない)。
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

# train_bitnet.py を import 再利用 (normalize / Tokenizer / dataset / 共起 teacher)。
# 二重実装禁止 — 同一ディレクトリから読む。
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import train_bitnet as tb  # noqa: E402


SEED = 20260620


# ==============================================================
# 教師プロンプト整形 (素の transformers + 自前 TIPO 構造化プロンプト)
# ==============================================================
# T1 で tipo-kgen は不要と判定済。TIPO/DanTagGen は LlamaForCausalLM なので
# 「tag: <seed>」継続プロンプトで comma 区切りタグを継続生成させ、最初の改行までを
# 補完タグ列として取る (probe で確認: 1行目が clean な comma-separated タグ列になる)。
def build_prompt(teacher_kind, seed_tags):
    """seed タグ列から教師継続プロンプトを作る。

    TIPO/DanTagGen とも tag セクション継続で「次に来るタグ」を comma 区切り生成する。
    seed_tags は自然文由来 greedy タグ + サンプルの先頭 target タグ群 (条件)。
    """
    seed = ", ".join(seed_tags)
    return f"tag: {seed}"


def first_line_tags(continuation):
    """生成継続テキストの 1 行目を comma で split したタグ片リスト。

    空白では割らない (`long hair` を壊さない)。前後空白のみ trim。
    """
    line = continuation.split("\n", 1)[0]
    return [t.strip() for t in line.split(",") if t.strip()]


# ==============================================================
# 教師生成 → サンプル単位タグ頻度 (vocab 写像済み)
# ==============================================================
class ExternalTeacherCache:
    """TIPO/DanTagGen 生成由来の per-sample 条件付きタグ頻度 teacher。

    各サンプルにつき seed タグから N 回 generate し、出力タグを自作 vocab に写像して
    頻度カウント tag_freq[vocab_id] を作る。soft_target は CoOccurrenceTeacher と同一
    インターフェースで、この per-sample 頻度を共起の代わりに使う (gold/prefix を除外し
    残余を温度付きで配る)。
    """

    def __init__(self, tok, main_mass=0.85, t_temp=2.0, topn=32):
        self.tok = tok
        self.main_mass = float(main_mass)
        self.t_temp = float(t_temp)
        self.topn = int(topn)
        self.cur_freq = {}  # 現在サンプルの {vocab_id: freq} (本体タグ id 5.. のみ)

    def map_tags_to_vocab(self, tag_pieces):
        """タグ片リスト → {vocab_id: count}。OOV は drop。

        normalize は train_bitnet.Tokenizer の static method を再利用 (二重実装禁止)。
        return: (kept_dict, n_total, n_kept)
        """
        kept = {}
        n_total = 0
        n_kept = 0
        for piece in tag_pieces:
            n_total += 1
            norm = tb.Tokenizer.normalize(piece)
            tid = self.tok.tag_to_id.get(norm)
            if tid is not None and tid >= 5:
                kept[tid] = kept.get(tid, 0) + 1
                n_kept += 1
        return kept, n_total, n_kept

    def set_current(self, tag_freq):
        """このサンプルの vocab 写像済みタグ頻度をセットする。"""
        self.cur_freq = tag_freq

    def soft_target(self, prefix_ids, gold_id):
        """1 target 位置の soft 分布 dict {id: prob} (CoOccurrenceTeacher と同一構造)。

        - gold が specials (<5) → 軟化しない (構造トークンは hard)。
        - gold に main_mass、残余 (1-main_mass) を「TIPO が予測した次タグ候補」
          (= cur_freq 中 gold/prefix 以外の上位 topn) へ温度付き softmax で配る。
        - 候補が無い場合は gold に全質量 (hard と同じ)。
        prefix と gold は候補から除外する。候補生成は gold 値に依存しない (除外のみ)。
        """
        if gold_id < 5:
            return {gold_id: 1.0}
        prefix_set = set(p for p in prefix_ids if p >= 5)
        score = {}
        for b, c in self.cur_freq.items():
            if b == gold_id or b < 5 or b in prefix_set:
                continue
            score[b] = float(c)
        if not score:
            return {gold_id: 1.0}
        items = sorted(score.items(), key=lambda kv: kv[1], reverse=True)[: self.topn]
        logs = [(math.log(s + 1.0) / self.t_temp) for _, s in items]
        mx = max(logs)
        exps = [math.exp(l - mx) for l in logs]
        Z = sum(exps)
        resid = 1.0 - self.main_mass
        out = {gold_id: self.main_mass}
        for (b, _), e in zip(items, exps):
            out[b] = out.get(b, 0.0) + resid * (e / Z)
        return out


# ==============================================================
# 教師モデル ロード + バッチ生成
# ==============================================================
TEACHER_HF = {
    "tipo": "KBlueLeaf/TIPO-200M",
    "dantaggen": "KBlueLeaf/DanTagGen-delta-rev2",
}


def load_teacher_model(teacher_kind, dtype_str, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_id = TEACHER_HF[teacher_kind]
    htok = AutoTokenizer.from_pretrained(hf_id)
    htok.padding_side = "left"  # batched generate は left pad
    if htok.pad_token is None:
        htok.pad_token = htok.eos_token
    dtype = torch.float32 if dtype_str == "fp32" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=dtype)
    model = model.to(device).eval()
    return htok, model


def generate_sample_freqs(rows, tok, teacher_kind, htok, model, device,
                          n_samples, max_new_tokens, prompt_batch, seed_tags_k,
                          temperature, top_p):
    """各 row につき seed タグから N 回 generate → vocab 写像済みタグ頻度を返す。

    return: (freqs[list of dict], oov_total, oov_kept)
      freqs[i] = {vocab_id: count}  (row i のサンプル単位条件付きタグ頻度)
      oov_total/oov_kept = 全生成タグ片の総数 / in-vocab に残った数 (OOV保持率算出用)。
    seed タグ = 自然文 text の greedy タグ + target 先頭 seed_tags_k タグ (条件)。
    """
    import torch

    helper = ExternalTeacherCache(tok)
    freqs = [None] * len(rows)
    oov_total = 0
    oov_kept = 0

    # seed タグを作る (各 row)。
    seeds = []
    for r in rows:
        text_tag_ids = tb.encode_text_greedy(tok, r.get("text", ""))
        text_tags = [tok.tags[i - 5] for i in text_tag_ids]  # id→正準タグ文字列
        head_target = r.get("tags", [])[:seed_tags_k]
        seed = text_tags + [t for t in head_target if t not in text_tags]
        if not seed:
            seed = ["1girl"]  # 空 seed 防止 (最低限の条件)
        seeds.append(seed)

    for bstart in range(0, len(rows), prompt_batch):
        bidx = list(range(bstart, min(bstart + prompt_batch, len(rows))))
        prompts = [build_prompt(teacher_kind, seeds[i]) for i in bidx]
        enc = htok(prompts, return_tensors="pt", padding=True)
        ids = enc.input_ids.to(device)
        am = enc.attention_mask.to(device)
        with torch.no_grad():
            out = model.generate(
                ids, attention_mask=am, do_sample=True,
                temperature=temperature, top_p=top_p,
                max_new_tokens=max_new_tokens,
                num_return_sequences=n_samples,
                pad_token_id=htok.eos_token_id,
            )
        # out shape: [len(bidx)*n_samples, gen_len]。row 順に n_samples 連続。
        plen = ids.shape[1]
        for k, ri in enumerate(bidx):
            agg = {}
            for j in range(n_samples):
                seq = out[k * n_samples + j]
                cont = htok.decode(seq[plen:], skip_special_tokens=True)
                pieces = first_line_tags(cont)
                kept, nt, nk = helper.map_tags_to_vocab(pieces)
                oov_total += nt
                oov_kept += nk
                for vid, c in kept.items():
                    agg[vid] = agg.get(vid, 0) + c
            freqs[ri] = agg
    return freqs, oov_total, oov_kept


# ==============================================================
# soft 分布 → position 軸付き COO npz
# ==============================================================
def build_coo_for_dataset(teacher, dataset, freqs, max_len):
    """各サンプルの target 位置ごとの soft 分布を COO 配列で返す。

    teacher: ExternalTeacherCache (set_current でサンプル頻度を切替えて使う)。
    freqs[i]: dataset.samples[i] に対応する vocab 写像済みタグ頻度 (None 可)。
    返り値: dict(rows=sample_idx, poss=target_index_t, cols=vocab_id, probs=prob)
      = D5 の (rows/cols/cnts) に position 軸 (poss) を加えた疎形式。各 (rows,poss) で
        prob の和は 1.0 (gold + 候補)。あわせて平均エントロピーを返す。
    """
    rows = []
    poss = []
    cols = []
    probs = []
    ent_sum = 0.0
    ent_n = 0
    for i, s in enumerate(dataset.samples):
        ids, tags_start = s[0], s[1]
        teacher.set_current(freqs[i] if freqs[i] is not None else {})
        sparse = tb.compute_sample_soft(teacher, ids, tags_start, max_len)
        for t, dist in sparse:
            H = 0.0
            for vid, p in dist.items():
                rows.append(i)
                poss.append(t)
                cols.append(vid)
                probs.append(p)
                if p > 0:
                    H -= p * math.log(p)
            ent_sum += H
            ent_n += 1
    return {
        "rows": np.array(rows, dtype=np.int32),
        "poss": np.array(poss, dtype=np.int32),
        "cols": np.array(cols, dtype=np.int32),
        "probs": np.array(probs, dtype=np.float32),
        "n_samples": np.array([len(dataset.samples)], dtype=np.int32),
        "max_len": np.array([max_len], dtype=np.int32),
    }, (ent_sum / max(ent_n, 1)), ent_n


def load_coo_npz(path):
    """書いた COO npz を読み戻す (テスト/再利用)。"""
    z = np.load(path)
    return {k: z[k] for k in z.files}


# ==============================================================
# probe (T3/D6-0): teacher-alone top10 recall A/B
# ==============================================================
def teacher_candidate_topk_recall(dataset, freqs, teacher_obj, max_len,
                                  is_external, topk=10):
    """teacher-alone top10 recall (真の予測力・A/B 用)。

    soft_target は gold に main_mass を注入するため、そのまま top-k を取ると gold が
    常に top-1 = recall 1.0 になり A/B にならない (degenerate)。そこで **gold を未知とした
    候補分布** で評価する: 各 target 位置で gold を未知 (センチネル=TOK_UNK) として
    teacher の候補分布を取り、その候補上位 top-k に **真の gold が入るか** を測る。
    = 「teacher が prefix だけから次タグ gold を予測できたか」。これが train_bitnet の
    eval recall (student の top10 recall) と直接比較できる teacher-alone recall。
    候補生成は gold 値に依存しない (gold/prefix を除外するのみ) ので、センチネル gold でも
    実 gold が候補に残り、正しく hit 判定できる。
    """
    SENTINEL = tb.VOCAB_SIZE  # =4999, vocab 範囲外・本体タグ id と衝突しない
    total = 0
    hit = 0
    informative = 0
    for i, s in enumerate(dataset.samples):
        ids, tags_start = s[0], s[1]
        if is_external:
            teacher_obj.set_current(freqs[i] if freqs[i] is not None else {})
        idslc = ids[:max_len]
        n = len(idslc)
        prefix = []
        for t in range(n - 1):
            pos = t + 1
            gold = idslc[pos]
            if pos < tags_start or gold < 5:
                if pos >= tags_start and gold >= 5:
                    prefix.append(gold)
                continue
            dist = teacher_obj.soft_target(prefix, SENTINEL)
            cands = {c: p for c, p in dist.items() if c != SENTINEL}
            ranked = sorted(cands.items(), key=lambda kv: kv[1], reverse=True)
            topk_ids = [vid for vid, _ in ranked[:topk]]
            total += 1
            if topk_ids:
                informative += 1
            if gold in topk_ids:
                hit += 1
            prefix.append(gold)
    return {
        "recall": hit / max(total, 1),
        "total": total,
        "informative_rate": informative / max(total, 1),
    }


# ==============================================================
# main
# ==============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/bitnet")
    ap.add_argument("--teacher", choices=["tipo", "dantaggen"], default="tipo")
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32",
                    help="教師 dtype。蒸留忠実度のため fp32 既定 (T1 推奨)。")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--main-mass", type=float, default=0.85)
    ap.add_argument("--temp", type=float, default=2.0,
                    help="候補配分 softmax 温度 (大きいほど平坦)。")
    ap.add_argument("--topn", type=int, default=32,
                    help="残余を配る候補上位件数。")
    ap.add_argument("--n-samples", type=int, default=8,
                    help="1 サンプルあたり教師 generate 回数 (num_return_sequences)。")
    ap.add_argument("--prompt-batch", type=int, default=16,
                    help="同時 generate するサンプル数 (VRAM 余裕で増やせる)。")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--seed-tags-k", type=int, default=6,
                    help="seed に含める target 先頭タグ数 (条件付け)。")
    ap.add_argument("--gen-temperature", type=float, default=1.0)
    ap.add_argument("--gen-top-p", type=float, default=0.95)
    ap.add_argument("--max-len", type=int, default=tb.MAX_SEQ_LEN)
    ap.add_argument("--probe-only", action="store_true",
                    help="T3/D6-0: 訓練せず val で teacher-alone recall を D5 と A/B。")
    ap.add_argument("--probe-limit", type=int, default=0,
                    help=">0 で val を先頭 N 件に絞る (時間外挿/スモーク用)。")
    ap.add_argument("--no-write-npz", action="store_true",
                    help="npz を書かない (probe で生成のみ確認したいとき)。")
    args = ap.parse_args()

    import torch
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    data_dir = args.data_dir
    cache_dir = os.path.join(data_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    vocab_path = os.path.join(data_dir, "vocab.json")
    tok = tb.Tokenizer(vocab_path)

    val_rows = tb.load_pairs(os.path.join(data_dir, "pairs.val.jsonl"))
    if args.probe_limit > 0:
        val_rows = val_rows[: args.probe_limit]
    val_ds = tb.PairDataset(tok, val_rows, args.max_len)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[d6] teacher={args.teacher} dtype={args.dtype} device={device} "
          f"main_mass={args.main_mass} temp={args.temp} topn={args.topn} "
          f"N={args.n_samples} seed_tags_k={args.seed_tags_k}")

    htok, model = load_teacher_model(args.teacher, args.dtype, device)

    # --- val 教師生成 (時間実測) ---
    t0 = time.time()
    val_freqs, oov_tot, oov_kept = generate_sample_freqs(
        val_rows, tok, args.teacher, htok, model, device,
        args.n_samples, args.max_new_tokens, args.prompt_batch,
        args.seed_tags_k, args.gen_temperature, args.gen_top_p)
    val_gen_s = time.time() - t0
    per_sample = val_gen_s / max(len(val_rows), 1)
    oov_keep_rate = oov_kept / max(oov_tot, 1)
    print(f"[d6] val 生成 {len(val_rows)} 件 / {val_gen_s:.1f}s "
          f"({per_sample:.3f}s/sample) OOV保持率={oov_keep_rate:.3f} "
          f"(kept {oov_kept}/{oov_tot})")
    est_full_h = per_sample * 5000 / 3600.0
    print(f"[d6] 全件 (train4500+val500) 外挿 ~ {est_full_h:.2f}h")

    teacher = ExternalTeacherCache(tok, args.main_mass, args.temp, args.topn)

    # --- COO 構築 + エントロピー (val) ---
    val_coo, val_mean_ent, val_ent_n = build_coo_for_dataset(
        teacher, val_ds, val_freqs, args.max_len)

    stats = {
        "teacher": args.teacher,
        "teacher_hf": TEACHER_HF[args.teacher],
        "dtype": args.dtype,
        "main_mass": args.main_mass,
        "temp": args.temp,
        "topn": args.topn,
        "n_samples": args.n_samples,
        "seed_tags_k": args.seed_tags_k,
        "seed": SEED,
        "val": {
            "n_rows": len(val_rows),
            "gen_seconds": round(val_gen_s, 2),
            "per_sample_seconds": round(per_sample, 4),
            "oov_keep_rate": round(oov_keep_rate, 4),
            "oov_kept": oov_kept,
            "oov_total": oov_tot,
            "mean_entropy_nats": round(val_mean_ent, 4),
            "n_soft_positions": val_ent_n,
        },
        "est_full_hours": round(est_full_h, 2),
    }

    if args.probe_only:
        # --- A/B: TIPO teacher-alone vs D5 共起 teacher-alone top10 recall ---
        tipo = teacher_candidate_topk_recall(
            val_ds, val_freqs, teacher, args.max_len, is_external=True, topk=10)

        # D5 共起 teacher (既存・同 val・同 prefix 規則)。
        posts_path = os.path.join(cache_dir, "danbooru_posts.jsonl")
        cooc_teacher = tb.CoOccurrenceTeacher(
            tok, posts_path, main_mass=args.main_mass,
            t_cooc=args.temp, topn=args.topn)
        cooc = teacher_candidate_topk_recall(
            val_ds, None, cooc_teacher, args.max_len, is_external=False, topk=10)

        winner = ("tipo" if tipo["recall"] > cooc["recall"] else
                  ("d5_cooc" if cooc["recall"] > tipo["recall"] else "tie"))
        stats["probe"] = {
            "topk": 10,
            "metric": "teacher-alone top10 recall (gold を未知とした候補分布の top10 に "
                      "真の gold が入る率・student eval recall と比較可能)",
            "tipo_teacher_alone_top10_recall": round(tipo["recall"], 4),
            "tipo_positions": tipo["total"],
            "tipo_informative_rate": round(tipo["informative_rate"], 4),
            "d5_cooc_teacher_alone_top10_recall": round(cooc["recall"], 4),
            "d5_cooc_positions": cooc["total"],
            "d5_cooc_informative_rate": round(cooc["informative_rate"], 4),
            "student_1_reference_top10_recall": 0.777,
            "winner": winner,
        }
        print("\n========== T3/D6-0 PROBE 結果 ==========")
        print(f"TIPO    teacher-alone top10 recall = {tipo['recall']:.4f} "
              f"(pos={tipo['total']}, informative={tipo['informative_rate']:.4f})")
        print(f"D5 共起 teacher-alone top10 recall = {cooc['recall']:.4f} "
              f"(pos={cooc['total']}, informative={cooc['informative_rate']:.4f})")
        print(f"(参考) #1 student dense top10 recall = 0.777")
        print(f"OOV 保持率 = {oov_keep_rate:.4f} / 平均エントロピー = "
              f"{val_mean_ent:.4f} nats")
        print(f"勝者: {winner}")
        print("========================================\n")
    else:
        # --- 本実行: train も生成して両 npz 書き出し ---
        train_rows = tb.load_pairs(os.path.join(data_dir, "pairs.train.jsonl"))
        train_ds = tb.PairDataset(tok, train_rows, args.max_len)
        t1 = time.time()
        tr_freqs, tr_oov_tot, tr_oov_kept = generate_sample_freqs(
            train_rows, tok, args.teacher, htok, model, device,
            args.n_samples, args.max_new_tokens, args.prompt_batch,
            args.seed_tags_k, args.gen_temperature, args.gen_top_p)
        train_gen_s = time.time() - t1
        train_coo, train_mean_ent, _ = build_coo_for_dataset(
            teacher, train_ds, tr_freqs, args.max_len)
        stats["train"] = {
            "n_rows": len(train_rows),
            "gen_seconds": round(train_gen_s, 2),
            "oov_keep_rate": round(tr_oov_kept / max(tr_oov_tot, 1), 4),
            "mean_entropy_nats": round(train_mean_ent, 4),
        }
        if not args.no_write_npz:
            tr_path = os.path.join(cache_dir, "d6_teacher_soft.train.npz")
            va_path = os.path.join(cache_dir, "d6_teacher_soft.val.npz")
            np.savez_compressed(tr_path, **train_coo)
            np.savez_compressed(va_path, **val_coo)
            print(f"[d6] npz 保存 {tr_path} / {va_path}")

    stats_path = os.path.join(cache_dir, "d6_teacher_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[d6] stats 保存 {stats_path}")


if __name__ == "__main__":
    main()
