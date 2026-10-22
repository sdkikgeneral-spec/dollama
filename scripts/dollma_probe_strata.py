# -*- coding: utf-8 -*-
"""dollma_probe_strata.py — Strata (ローカル推論エンジン・OpenAI 互換 API) の probe クライアント。

推論は Strata プロセス側。本スクリプトは 127.0.0.1 固定で HTTP を叩くだけ (host 引数は意図的に無い)。
RAM / VRAM / commit charge は gpu-benchmarker が別途測る。ここでは API 側の計時のみ。

サブコマンド:
  bench   prefill / decode tok/s・レイテンシ (任意で画像 1 枚入力)。
  tags    pairs.eval_diverse_a.jsonl から固定 seed で N 件抽出 → 日本語入力で danbooru タグ列を生成。
          生成後、train_bitnet.eval_generation_setmetrics (正典 LM が diverse_a 0.3332 を出した
          のと同一関数) で採点する (torch 必須 = py -3.12。無ければ preds.csv のみ出して score で後採点)。
  score   tags が出した preds.csv を同関数で採点 (py -3.12)。
  canon   同一 subset を正典 LM (bitnet_dense_fp32.safetensors) で採点 (py -3.12)。
  subset  subset の行 index / post_id / variant を JSON で出力 (別途採点用)。
  judge   「画像パス,ラベル」CSV の各画像に破綻有無を答えさせ、弱ラベルに対する AUC を算出。
  --selftest  ネットワーク・torch 不要の自己検査。

分母の注意: 正典 0.3332 は eval_diverse_a 全 1500 件 (ja+en) の macro F1。100 件 subset の値を
その 0.3332 と直接比較してはならない。比較は必ず `canon` が同一 subset で出す値と行う。

CSV ラベル規約 (judge): 破綻 = 陽性 (1)。bad / 悪 / ng / 1 / true → 1、good / 良 / ok / 0 / false → 0。

例 (py -3.12):
  py -3.12 scripts/dollma_probe_strata.py bench --model qwen3-vl --port 8080 --out-dir out/s
  py -3.12 scripts/dollma_probe_strata.py tags  --model qwen3-vl --port 8080 --out-dir out/s
  py -3.12 scripts/dollma_probe_strata.py canon --out-dir out/s   # 同一 subset を正典 LM で採点
"""

import argparse
import base64
import csv
import datetime
import json
import math
import mimetypes
import os
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

HOST = "127.0.0.1"  # 固定 (外部へ出さない)
DEFAULT_PAIRS = os.path.join(_REPO, "data", "bitnet", "pairs.eval_diverse_a.jsonl")
DEFAULT_DATA_DIR = os.path.join(_REPO, "data", "bitnet")
CANON_WEIGHTS = os.path.join(DEFAULT_DATA_DIR, "bitnet_dense_fp32.safetensors")
# 語彙外タグを gen 集合から落とさないための疑似 id 起点 (UNK は gen から除外されて precision が
# 嵩上げされるため、語彙外は一意な大きい id で「外れ」として数える)。
_OOV_ID_BASE = 10_000_000

TAGS_SYSTEM = (
    "You convert an image description into danbooru tags. "
    "Answer with JSON only: {\"tags\": [\"tag1\", \"tag2\", ...]}. "
    "Use lowercase English danbooru tags with spaces (not underscores), no explanations."
)
JUDGE_SYSTEM = (
    "You inspect an anime illustration for anatomical or rendering defects "
    "(extra/missing fingers, broken limbs, malformed face, etc). "
    "Answer with JSON only: {\"broken\": true|false, \"parts\": [\"hands\", ...], "
    "\"defect_score\": 0-10}. defect_score 10 = certainly broken, 0 = clean."
)


# ==============================================================
# 応答パーサ (純関数)
# ==============================================================
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)


def extract_json(text):
    """応答文字列から最初の JSON オブジェクト/配列を抜く。失敗なら None。決定的。

    ① 全体が JSON ② ```fence``` 内 ③ 最初の '{' / '[' からの括弧均衡走査、の順。
    """
    if text is None:
        return None
    s = text.strip()
    cands = [s]
    cands += [m.strip() for m in _FENCE_RE.findall(s)]
    for c in cands:
        try:
            return json.loads(c)
        except (ValueError, TypeError):
            pass
    for i, ch in enumerate(s):
        if ch not in "{[":
            continue
        frag = _balanced(s, i)
        if frag is None:
            continue
        try:
            return json.loads(frag)
        except ValueError:
            continue
    return None


def _balanced(s, start):
    """s[start] から括弧が閉じるまでの部分文字列 (文字列リテラル内の括弧は無視)。無ければ None。"""
    open_ch = s[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_str = False
    esc = False
    for j in range(start, len(s)):
        c = s[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return s[start:j + 1]
    return None


def normalize_tag(t):
    """タグ 1 個の正規化: 前後空白・箇条書き記号・引用符を除去、小文字化、'_' → ' '、空白圧縮。"""
    t = str(t).strip().strip("\"'`")
    t = re.sub(r"^[\-\*•\d]+[\.\)]\s+", "", t)  # "1. tag" / "- tag"
    t = re.sub(r"^[\-\*•]\s*", "", t)
    t = t.replace("_", " ").lower()
    t = re.sub(r"\s+", " ", t).strip().strip(",.;")
    return t


def _dedup(seq):
    seen = set()
    out = []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def parse_tags(text):
    """タグ列応答 → (タグ list・順序維持/重複除去, 経路 "json"|"freetext"|"empty")。

    JSON ({"tags":[..]} / [..] / "tags" が文字列) を試し、駄目ならカンマ/改行/セミコロン区切り。
    """
    obj = extract_json(text)
    tags = None
    if isinstance(obj, dict):
        for k in ("tags", "Tags", "tag", "danbooru_tags"):
            if k in obj:
                v = obj[k]
                tags = v if isinstance(v, list) else re.split(r"[,\n;]", str(v))
                break
    elif isinstance(obj, list) and all(isinstance(x, (str, int, float)) for x in obj):
        tags = obj
    if tags is not None:
        out = _dedup(normalize_tag(t) for t in tags)
        if out:
            return out, "json"
    if not text or not text.strip():
        return [], "empty"
    body = _FENCE_RE.sub(lambda m: m.group(1), text)
    # "tags:" のような前置きを剥がす。
    body = re.sub(r"^\s*(?:tags?|タグ)\s*[:：]\s*", "", body, flags=re.I)
    parts = re.split(r"[,\n;、，]", body)
    out = _dedup(normalize_tag(p) for p in parts)
    return out, ("freetext" if out else "empty")


_YES = re.compile(r"\b(yes|true|broken|defect(?:ive)?|y)\b|はい|破綻(?:して)?いる|あり", re.I)
_NO = re.compile(r"\b(no|false|clean|fine|none|n)\b|いいえ|破綻(?:して)?いない|なし", re.I)
_SCORE_SLASH = re.compile(r"(\d+(?:\.\d+)?)\s*/\s*(10|100|1)\b")
_SCORE_KEY = re.compile(r"(?:defect[_ ]?score|score|スコア)\s*[:：=]?\s*(\d+(?:\.\d+)?)", re.I)


def parse_judgement(text):
    """判定応答 → dict(broken: bool|None, parts: list, score: float|None [0,1], path)。

    score は「破綻らしさ」を 0..1 に正規化した値 (defect_score/10 など)。
    JSON 優先、失敗時は yes/no / "7/10" / "score: 7" を自由文から拾う。
    """
    res = {"broken": None, "parts": [], "score": None, "path": "none"}
    obj = extract_json(text)
    if isinstance(obj, dict):
        b = None
        for k in ("broken", "has_defect", "defect", "is_broken"):
            if k in obj:
                b = _to_bool(obj[k])
                break
        sc = None
        for k in ("defect_score", "score", "defect_level"):
            if k in obj:
                sc = _to_float(obj[k])
                break
        parts = obj.get("parts", obj.get("defect_parts", []))
        if isinstance(parts, str):
            parts = [p for p in re.split(r"[,、\n]", parts) if p.strip()]
        if not isinstance(parts, list):
            parts = []
        if b is not None or sc is not None:
            res["broken"] = b
            res["parts"] = [str(p).strip() for p in parts if str(p).strip()]
            res["score"] = _norm_score(sc)
            res["path"] = "json"
            _fill_broken_score(res)
            return res
    t = text or ""
    m = _SCORE_SLASH.search(t)
    if m:
        res["score"] = min(max(float(m.group(1)) / float(m.group(2)), 0.0), 1.0)
    else:
        m = _SCORE_KEY.search(t)
        if m:
            res["score"] = _norm_score(float(m.group(1)))
    # yes/no は先頭に近い方を採る (「No, but ... yes」を避ける)。
    my, mn = _YES.search(t), _NO.search(t)
    if my and mn:
        res["broken"] = my.start() < mn.start()
    elif my:
        res["broken"] = True
    elif mn:
        res["broken"] = False
    if res["broken"] is not None or res["score"] is not None:
        res["path"] = "freetext"
        _fill_broken_score(res)
    return res


def _fill_broken_score(res):
    """score 未取得なら broken から 1.0/0.0 を、broken 未取得なら score>=0.5 から補う。"""
    if res["score"] is None and res["broken"] is not None:
        res["score"] = 1.0 if res["broken"] else 0.0
    if res["broken"] is None and res["score"] is not None:
        res["broken"] = res["score"] >= 0.5


def _to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("true", "yes", "y", "1", "はい"):
        return True
    if s in ("false", "no", "n", "0", "いいえ"):
        return False
    return None


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _norm_score(x):
    """0..1 はそのまま、1 超 10 以下は /10、10 超は /100 (0..10 想定の緩い正規化)。"""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if x <= 1.0:
        return max(x, 0.0)
    if x <= 10.0:
        return x / 10.0
    return min(x / 100.0, 1.0)


# ==============================================================
# AUC (Mann-Whitney・同点は 0.5)
# ==============================================================
def auc(scores, labels):
    """labels: 1=陽性 (破綻)。同点は 0.5 カウント。片クラス欠如なら None。"""
    pos = [s for s, l in zip(scores, labels) if l == 1 and s is not None]
    neg = [s for s, l in zip(scores, labels) if l == 0 and s is not None]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def parse_label(s):
    """CSV ラベル → 0/1/None。破綻 = 1。"""
    t = str(s).strip().lower()
    if t in ("bad", "悪", "ng", "1", "true", "broken", "b"):
        return 1
    if t in ("good", "良", "ok", "0", "false", "clean", "g"):
        return 0
    return None


# ==============================================================
# 決定的抽出・set-F1 配線
# ==============================================================
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_subset(rows, n, seed, lang="ja"):
    """lang で絞った行 index から固定 seed で n 件を抽出 (昇順 index を返す・決定的)。

    元ファイルの行順にのみ依存 (set/dict 順序に依存しない)。lang="all" で絞らない。
    """
    cand = [i for i, r in enumerate(rows) if lang == "all" or r.get("lang") == lang]
    rng = random.Random(seed)
    if n >= len(cand):
        return sorted(cand)
    return sorted(rng.sample(cand, n))


def score_setf1(rows_subset, pred_tags, tok):
    """予測タグ列 (rows_subset と同順の list[list[str]]) を、正典評価と同一の
    train_bitnet.eval_generation_setmetrics で採点する (F1 を再実装しない)。

    生成 (greedy 生成と prompt 符号化) だけを Strata の予測へ差し替えるため、モジュール関数
    encode_text_greedy / _greedy_generate を一時的に置換する (scripts/test_dollma_eval_setmetrics.py
    と同じ手法)。語彙外タグは UNK で gen から落ちて precision が嵩上げされるのを避けるため、
    一意な疑似 id を割り当てて「外れ」として数える。
    返り値: eval_generation_setmetrics の dict (per_sample 付き) + n_oov_tags / n_pred_tags。
    """
    import train_bitnet as tb  # 遅延 import (torch 必須)

    assert len(rows_subset) == len(pred_tags)
    oov = {}
    n_oov = 0
    n_pred = 0
    id_lists = []
    for tags in pred_tags:
        ids = []
        for t in tags:
            n_pred += 1
            i = tok.tag_to_id_lookup(t)
            if i == tb.TOK_UNK:
                key = normalize_tag(t)
                if key not in oov:
                    oov[key] = _OOV_ID_BASE + len(oov)
                i = oov[key]
                n_oov += 1
            ids.append(i)
        id_lists.append(ids)

    counter = {"i": 0}

    def fake_encode(_tok, _text):
        # eval_generation_setmetrics は rows を 1 行ずつ順に呼ぶ。行番号を 1 トークンで運ぶ。
        v = counter["i"]
        counter["i"] += 1
        return [v]

    def fake_generate(_model, prompt_ids, _device, _max_len=None):
        return id_lists[prompt_ids[1]]

    class _M:
        def eval(self):
            return self

    orig_e, orig_g = tb.encode_text_greedy, tb._greedy_generate
    tb.encode_text_greedy, tb._greedy_generate = fake_encode, fake_generate
    try:
        rep = tb.eval_generation_setmetrics(_M(), tok, rows_subset, "cpu",
                                            collect_per_sample=True)
    finally:
        tb.encode_text_greedy, tb._greedy_generate = orig_e, orig_g
    rep["n_oov_tags"] = n_oov
    rep["n_pred_tags"] = n_pred
    return rep


# ==============================================================
# Strata クライアント (urllib・127.0.0.1 固定)
# ==============================================================
def _url(port, path):
    return f"http://{HOST}:{int(port)}{path}"


def image_data_uri(path):
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode("ascii")


def build_messages(system, user_text, image_path=None):
    if image_path:
        content = [{"type": "text", "text": user_text},
                   {"type": "image_url", "image_url": {"url": image_data_uri(image_path)}}]
    else:
        content = user_text
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": content})
    return msgs


def chat(port, model, messages, max_tokens=512, temperature=0.0, seed=None,
         timeout=600.0, stream=False):
    """chat.completions を叩く。dict(text, prompt_tokens, completion_tokens, t_total, t_first)。

    stream=True なら SSE を読み、初回トークン到達までを t_first として返す (prefill 近似)。
    """
    body = {"model": model, "messages": messages, "max_tokens": int(max_tokens),
            "temperature": float(temperature), "stream": bool(stream)}
    if seed is not None:
        body["seed"] = int(seed)
    if stream:
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(
        _url(port, "/v1/chat/completions"), data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    out = {"text": "", "prompt_tokens": None, "completion_tokens": None,
           "t_first": None, "t_total": None}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if not stream:
            data = json.loads(resp.read().decode("utf-8"))
            out["t_total"] = time.perf_counter() - t0
            out["text"] = data["choices"][0]["message"].get("content") or ""
            u = data.get("usage") or {}
            out["prompt_tokens"] = u.get("prompt_tokens")
            out["completion_tokens"] = u.get("completion_tokens")
            return out
        n_chunks = 0
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except ValueError:
                continue
            u = d.get("usage")
            if u:
                out["prompt_tokens"] = u.get("prompt_tokens")
                out["completion_tokens"] = u.get("completion_tokens")
            for ch in d.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    if out["t_first"] is None:
                        out["t_first"] = time.perf_counter() - t0
                    out["text"] += piece
                    n_chunks += 1
        out["t_total"] = time.perf_counter() - t0
        if out["completion_tokens"] is None:
            out["completion_tokens"] = n_chunks  # usage 非対応時は chunk 数で代用
            out["tokens_are_chunks"] = True
    return out


def rate_summary(r):
    """1 回分の計時 → prefill/decode tok/s (取れなければ None)。"""
    d = {}
    tf, tt = r.get("t_first"), r.get("t_total")
    pt, ct = r.get("prompt_tokens"), r.get("completion_tokens")
    d["prefill_tok_s"] = (pt / tf) if (pt and tf) else None
    d["decode_tok_s"] = ((ct - 1) / (tt - tf)) if (ct and ct > 1 and tf is not None
                                                    and tt > tf) else None
    return d


# ==============================================================
# 出力
# ==============================================================
def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_meta(out_dir, args, extra):
    meta = {
        "subcommand": args.cmd,
        "model": getattr(args, "model", None),
        "strata_version": getattr(args, "strata_version", None),
        "port": getattr(args, "port", None),
        "host": HOST,
        "seed": getattr(args, "seed", None),
        "datetime": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    meta.update(extra)
    p = os.path.join(out_dir, "run_meta.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return p


# ==============================================================
# サブコマンド
# ==============================================================
def cmd_bench(a):
    os.makedirs(a.out_dir, exist_ok=True)
    prompt = a.prompt
    if a.long_prompt_repeat > 1:
        prompt = (prompt + " ") * a.long_prompt_repeat
    rows = []
    for i in range(a.warmup + a.repeat):
        r = chat(a.port, a.model, build_messages(None, prompt, a.image),
                 max_tokens=a.max_tokens, temperature=0.0, seed=a.seed, stream=True)
        rs = rate_summary(r)
        tag = "warmup" if i < a.warmup else "measure"
        rows.append([i, tag, r["prompt_tokens"], r["completion_tokens"],
                     r["t_first"], r["t_total"], rs["prefill_tok_s"], rs["decode_tok_s"],
                     bool(a.image), r.get("tokens_are_chunks", False)])
        print(f"[bench] {i} {tag} ttft={r['t_first']} total={r['t_total']:.3f}s "
              f"prefill={rs['prefill_tok_s']} decode={rs['decode_tok_s']}")
    write_csv(os.path.join(a.out_dir, "bench.csv"),
              ["i", "phase", "prompt_tokens", "completion_tokens", "ttft_s", "total_s",
               "prefill_tok_s", "decode_tok_s", "with_image", "tokens_are_chunks"], rows)
    meas = [r for r in rows if r[1] == "measure"]

    def med(idx):
        v = [r[idx] for r in meas if r[idx] is not None]
        return statistics.median(v) if v else None

    summary = {"n_measure": len(meas), "median_ttft_s": med(4), "median_total_s": med(5),
               "median_prefill_tok_s": med(6), "median_decode_tok_s": med(7),
               "with_image": bool(a.image)}
    print("[bench] summary", json.dumps(summary))
    write_meta(a.out_dir, a, {"bench": summary, "max_tokens": a.max_tokens})
    return 0


def cmd_subset(a, rows=None):
    os.makedirs(a.out_dir, exist_ok=True)
    rows = rows if rows is not None else load_jsonl(a.pairs)
    idx = select_subset(rows, a.n, a.seed, a.lang)
    p = os.path.join(a.out_dir, "subset_ids.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"pairs": os.path.abspath(a.pairs), "seed": a.seed, "lang": a.lang,
                   "n": len(idx), "row_index": idx,
                   "ids": [[rows[i]["meta"]["post_id"], rows[i]["meta"]["variant"]]
                           for i in idx]}, f, ensure_ascii=False, indent=1)
    print(f"[subset] {len(idx)} 件 -> {p}")
    return 0


def cmd_tags(a):
    os.makedirs(a.out_dir, exist_ok=True)
    rows = load_jsonl(a.pairs)
    idx = select_subset(rows, a.n, a.seed, a.lang)
    cmd_subset(a, rows)
    out_rows = []
    for k, i in enumerate(idx):
        r = rows[i]
        t0 = time.perf_counter()
        try:
            res = chat(a.port, a.model, build_messages(TAGS_SYSTEM, r["text"]),
                       max_tokens=a.max_tokens, temperature=0.0, seed=a.seed)
            raw = res["text"]
            err = ""
        except (urllib.error.URLError, OSError, ValueError) as e:
            raw, err = "", repr(e)
        lat = time.perf_counter() - t0
        tags, path = parse_tags(raw)
        out_rows.append([i, r["meta"]["post_id"], r["meta"]["variant"], r["lang"], r["text"],
                         raw, "|".join(tags), path, f"{lat:.4f}", err])
        print(f"[tags] {k + 1}/{len(idx)} row={i} n_tags={len(tags)} path={path} {lat:.2f}s")
    write_csv(os.path.join(a.out_dir, "tags_preds.csv"),
              ["row_index", "post_id", "variant", "lang", "text", "raw", "tags", "parse_path",
               "latency_s", "error"], out_rows)
    write_meta(a.out_dir, a, {"n": len(idx), "pairs": os.path.abspath(a.pairs),
                              "lang": a.lang, "n_json": sum(1 for r in out_rows if r[7] == "json"),
                              "n_error": sum(1 for r in out_rows if r[9])})
    try:
        import torch  # noqa: F401
    except ImportError:
        print("[tags] torch 無し -> 採点は py -3.12 scripts/dollma_probe_strata.py score "
              f"--out-dir {a.out_dir} で行う")
        return 0
    return cmd_score(a)


def _load_tok(a):
    import train_bitnet as tb
    return tb.Tokenizer(os.path.join(a.data_dir, "vocab.json"))


def cmd_score(a):
    preds_path = os.path.join(a.out_dir, "tags_preds.csv")
    rows = load_jsonl(a.pairs)
    with open(preds_path, "r", encoding="utf-8", newline="") as f:
        pr = list(csv.DictReader(f))
    sub_rows = [rows[int(p["row_index"])] for p in pr]
    tags = [[t for t in p["tags"].split("|") if t] for p in pr]
    rep = score_setf1(sub_rows, tags, _load_tok(a))
    _dump_score(a, "strata_scored.json", rep, len(pr))
    return 0


def cmd_canon(a):
    """同一 subset を正典 LM で採点 (greedy 生成 + 同一 set-metrics 関数)。"""
    import train_bitnet as tb
    rows = load_jsonl(a.pairs)
    ids_path = os.path.join(a.out_dir, "subset_ids.json")
    if os.path.exists(ids_path):
        with open(ids_path, "r", encoding="utf-8") as f:
            idx = json.load(f)["row_index"]
    else:
        idx = select_subset(rows, a.n, a.seed, a.lang)
        cmd_subset(a, rows)
    sub = [rows[i] for i in idx]
    tok = tb.Tokenizer(os.path.join(a.data_dir, "vocab.json"))
    model = tb._load_dense_from_export(a.weights, "cpu")
    rep = tb.eval_generation_setmetrics(model, tok, sub, "cpu", collect_per_sample=True)
    _dump_score(a, "canon_scored.json", rep, len(sub))
    return 0


def _dump_score(a, name, rep, n):
    ps = rep.get("per_sample", {})
    out = {"n_rows": n, "macro": rep["macro"], "micro": rep["micro"],
           "n_cases": rep["n_cases"], "n_oov_tags": rep.get("n_oov_tags"),
           "n_pred_tags": rep.get("n_pred_tags"),
           "per_sample_f1": ps.get("f1"),
           "note": "分母 = この subset。全 1500 件の 0.3332 とは直接比較しない"}
    p = os.path.join(a.out_dir, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[score] {name}: macro F1={rep['macro']['f1']:.4f} (n={n}) -> {p}")
    # 両側が揃えば paired 差を出す。
    sp = os.path.join(a.out_dir, "strata_scored.json")
    cp = os.path.join(a.out_dir, "canon_scored.json")
    if os.path.exists(sp) and os.path.exists(cp):
        s = json.load(open(sp, encoding="utf-8"))["per_sample_f1"]
        c = json.load(open(cp, encoding="utf-8"))["per_sample_f1"]
        d = [x - y for x, y in zip(s, c) if not (x != x or y != y)]
        if d:
            print(f"[score] paired (strata - canon) mean={statistics.mean(d):+.4f} n={len(d)}")


def cmd_judge(a):
    os.makedirs(a.out_dir, exist_ok=True)
    items = []
    with open(a.csv, "r", encoding="utf-8", newline="") as f:
        for rec in csv.reader(f):
            if len(rec) < 2 or rec[0].strip().startswith("#"):
                continue
            lab = parse_label(rec[1])
            if lab is None:
                continue  # ヘッダ行・不明ラベルは読み飛ばす
            items.append((rec[0].strip(), lab))
    out_rows = []
    for k, (pth, lab) in enumerate(items):
        t0 = time.perf_counter()
        try:
            res = chat(a.port, a.model, build_messages(JUDGE_SYSTEM, a.prompt, pth),
                       max_tokens=a.max_tokens, temperature=0.0, seed=a.seed)
            raw, err = res["text"], ""
        except (urllib.error.URLError, OSError, ValueError) as e:
            raw, err = "", repr(e)
        lat = time.perf_counter() - t0
        j = parse_judgement(raw)
        out_rows.append([pth, lab, j["broken"], "|".join(j["parts"]), j["score"], j["path"],
                         raw, f"{lat:.4f}", err])
        print(f"[judge] {k + 1}/{len(items)} label={lab} score={j['score']} path={j['path']}")
    write_csv(os.path.join(a.out_dir, "judge_preds.csv"),
              ["image", "label", "broken", "parts", "score", "parse_path", "raw",
               "latency_s", "error"], out_rows)
    a_ = auc([r[4] for r in out_rows], [r[1] for r in out_rows])
    n_unparsed = sum(1 for r in out_rows if r[4] is None)
    print(f"[judge] AUC={a_} n={len(out_rows)} unparsed={n_unparsed}")
    write_meta(a.out_dir, a, {"n": len(out_rows), "auc": a_, "n_unparsed": n_unparsed,
                              "n_pos": sum(r[1] for r in out_rows), "csv": a.csv})
    return 0


# ==============================================================
# selftest / CLI
# ==============================================================
def selftest():
    t, p = parse_tags('{"tags": ["1girl", "Long_Hair", "1girl"]}')
    assert t == ["1girl", "long hair"] and p == "json", (t, p)
    t, p = parse_tags("tags: 1girl, blue_eyes\nsolo")
    assert t == ["1girl", "blue eyes", "solo"] and p == "freetext", (t, p)
    assert parse_judgement('{"broken": true, "defect_score": 8}')["score"] == 0.8
    assert parse_judgement("No defects. score: 2/10")["broken"] is False
    assert auc([0.9, 0.1, 0.5], [1, 0, 1]) == 1.0
    assert auc([0.5, 0.5], [1, 0]) == 0.5
    assert select_subset([{"lang": "ja"}] * 10, 3, 1) == select_subset([{"lang": "ja"}] * 10, 3, 1)
    print("selftest OK")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description="Strata (127.0.0.1) probe クライアント")
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")

    def common(p, net=True):
        p.add_argument("--out-dir", required=True)
        p.add_argument("--seed", type=int, default=20260620)
        if net:
            p.add_argument("--model", required=True)
            p.add_argument("--port", type=int, default=8080)
            p.add_argument("--strata-version", default=None,
                           help="run_meta.json に記録するだけ (Strata から自動取得しない)")

    def pairs(p):
        p.add_argument("--pairs", default=DEFAULT_PAIRS)
        p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
        p.add_argument("--n", type=int, default=100)
        p.add_argument("--lang", default="ja", choices=["ja", "en", "all"])

    b = sub.add_parser("bench")
    common(b)
    b.add_argument("--prompt", default="Describe an anime girl with long blue hair in detail.")
    b.add_argument("--long-prompt-repeat", type=int, default=1,
                   help="prompt を N 回連結して prefill を重くする")
    b.add_argument("--image", default=None, help="画像 1 枚入力 (任意)")
    b.add_argument("--max-tokens", type=int, default=256)
    b.add_argument("--warmup", type=int, default=1)
    b.add_argument("--repeat", type=int, default=5)

    t = sub.add_parser("tags")
    common(t)
    pairs(t)
    t.add_argument("--max-tokens", type=int, default=256)

    s = sub.add_parser("score")
    common(s, net=False)
    pairs(s)

    c = sub.add_parser("canon")
    common(c, net=False)
    pairs(c)
    c.add_argument("--weights", default=CANON_WEIGHTS)

    u = sub.add_parser("subset")
    common(u, net=False)
    pairs(u)

    j = sub.add_parser("judge")
    common(j)
    j.add_argument("--csv", required=True, help="「画像パス,ラベル」CSV")
    j.add_argument("--prompt", default="Does this illustration have anatomical defects?")
    j.add_argument("--max-tokens", type=int, default=256)
    return ap


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.cmd:
        ap.error("サブコマンドが必要 (bench/tags/score/canon/subset/judge) か --selftest")
    return {"bench": cmd_bench, "tags": cmd_tags, "score": cmd_score, "canon": cmd_canon,
            "subset": cmd_subset, "judge": cmd_judge}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
