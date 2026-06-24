#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dollama 入力多様化パイロット (施策 B / B-1) — diverse-train 構築スクリプト

施策 B のゴール: タグ集合を実 danbooru のまま固定し (tags-stay-real)、入力自然文
だけを人手 (Claude) 著述で多様化して、新 proxy (diverse-val 生成 set-F1) で
train 側の汎化効果を測る。混合方式は **Replace** (件数 4500 を維持)。著述件数 n は
CLI フラグで一般化済み (B-1-a パイロットは n=500、件数拡大は n=2000 等)。

施策 C (diverse-val 構築・scripts/dollma_make_eval_diverse.py) と同じ
「tags-stay-real・人手著述」機構を **train 向け** に流用する。Tokenizer / vocab /
正準順序ロジックは eval_diverse / make_pairs から import 流用し、二重実装しない。

**スーパーセット性 (件数拡大の要)**:
  _select_500 は post_id 安定ソート → random.Random(seed).shuffle → 先頭 n。
  seed を固定すれば n=2000 の先頭 500 件は n=500 の抽出と完全一致する
  (= 既存著述 part01-05 がそのまま再利用でき、追加著述は差分のみ)。
  --emit-prompts は --anchor で既存 prompts を渡すと、その post_id 列が今回の
  抽出の先頭に完全一致するか assert する (再利用可能性の機械保証)。

本スクリプトは 2 モードを持つ:

  --emit-prompts : pairs.train.jsonl から seed 決定的に n 件抽出し、
                   gold タグ列 + lang_hint のみのプロンプトバッチを出力 (段a)。
                   **散文は一切生成しない。** 散文 (段b) は main Claude が著述する。
                   --todo を渡すと、--anchor がカバーしない post_id (追加著述が
                   必要な分) だけを切り出した todo リストも出力する。
  --ingest       : 段a の prompts (gold 源) + main Claude が書いた texts を突合・
                   検証し、Replace 合成した train (著述n + synthetic(4500-n)=4500) を
                   凍結出力する (段c)。

不変条件 (厳守):
  - pairs.train.jsonl / pairs.val.jsonl / pairs.eval_diverse_*.jsonl /
    本番重み / golden は **読むだけ・無改変**。新ファイルはすべて別名。
  - tags-stay-real: gold = pairs.train.jsonl 行の実 danbooru タグをバイト一致コピー。
                    著述文からタグを抽出しない。
  - 凍結: 出力 jsonl は再現性アンカー。既存があればスキップ (--force でのみ再生成)。
    凍結アンカー (n=500 パイロットの diverse_train_prompts.jsonl /
    pairs.train.diverse_b.jsonl / stats.diverse_b.json / part01-05) は無改変。
    件数拡大版は別サフィックス (例 b2000) で出力する。

使い方:
  # 段a (n=500 パイロット・決定的)
  py -3.12 scripts/dollma_make_diverse_train.py --emit-prompts --n 500 --seed 20260620
  # 段a (n=2000 件数拡大・既存500をアンカーにスーパーセット検証 + todo 切り出し)
  py -3.12 scripts/dollma_make_diverse_train.py --emit-prompts --n 2000 --seed 20260620 \
      --prompts data/bitnet/diverse_train_prompts_b2000.jsonl \
      --anchor data/bitnet/diverse_train_prompts.jsonl \
      --todo data/bitnet/diverse_train_todo_b2000.jsonl
  # 段c (main Claude が diverse_train_texts.jsonl を書いた後)
  py -3.12 scripts/dollma_make_diverse_train.py --ingest
"""
import argparse
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# 二重実装禁止: Tokenizer / vocab ロード / リーク検査ユーティリティは
# dollma_make_eval_diverse から、正準順序ロジックは dollma_make_pairs から流用する。
import dollma_make_eval_diverse as ed
# ed.mp == dollma_make_pairs (正準順序 normalize_separator / classify_bucket 等)

# 流用するシンボル (新規にタグ整形・vocab ロードを書かないことの明示)
EvalTokenizer = ed.EvalTokenizer
load_vocab_set = ed.load_vocab_set
train_post_ids = ed.train_post_ids        # train (train+identity.train) post_id 集合
val_post_ids_and_rows = ed.val_post_ids_and_rows
_read_jsonl = ed._read_jsonl
TOK_UNK = ed.TOK_UNK

# 既定パス
DEF_DATA_DIR = ed.DEF_DATA_DIR
DEF_VOCAB = ed.DEF_VOCAB
DEF_TRAIN = ed.DEF_TRAIN                   # 抽出元 = pairs.train.jsonl (eval は val 由来)
DEF_VAL = ed.DEF_VAL
DEF_DIVERSE_A = os.path.join(DEF_DATA_DIR, "pairs.eval_diverse_a.jsonl")
DEF_DIVERSE_B = os.path.join(DEF_DATA_DIR, "pairs.eval_diverse_b.jsonl")
DEF_PROMPTS = os.path.join(DEF_DATA_DIR, "diverse_train_prompts.jsonl")
DEF_TEXTS = os.path.join(DEF_DATA_DIR, "diverse_train_texts.jsonl")
DEF_OUT_TRAIN = os.path.join(DEF_DATA_DIR, "pairs.train.diverse_b.jsonl")
DEF_STATS = os.path.join(DEF_DATA_DIR, "stats.diverse_b.json")

DEF_N = 500           # 既定の著述件数 (Replace なので train 件数は不変 4500)
DEF_TRAIN_TOTAL = 4500  # Replace 合成後の総件数 (件数アサート用)


# ------------------------------------------------------------
# 共通: 抽出元 train 行の読み込み + post_id キー化
# ------------------------------------------------------------
def _train_rows():
    """抽出元 pairs.train.jsonl の全行を順序保持で返す。"""
    return _read_jsonl(DEF_TRAIN)


def _post_id(row):
    return row.get("meta", {}).get("post_id")


def diverse_val_post_ids():
    """diverse-val (eval_diverse_a / eval_diverse_b) の post_id 集合を返す。

    段a の最重要 assert (diverse-val 汚染防止) のソース。読むだけ・無改変。
    """
    pids = set()
    for path in (DEF_DIVERSE_A, DEF_DIVERSE_B):
        if os.path.exists(path):
            for r in _read_jsonl(path):
                pid = _post_id(r)
                if pid is not None:
                    pids.add(pid)
    return pids


def _select_500(train_rows, n, seed):
    """seed 決定的に train_rows から n 件抽出する (関数名は履歴互換で据え置き)。

    再現性: post_id で安定ソートしてから random.Random(seed) でシャッフルし先頭 n。
    (jsonl の物理行順に依存しないよう、まず post_id でソートしてから振る)。

    スーパーセット性: シャッフル順は seed のみで決まり n に非依存。よって同一 seed なら
    n を増やしても先頭から積み増すだけ ⇒ n=2000 の先頭 500 = n=500 の抽出 (完全一致)。
    """
    eligible = [r for r in train_rows if _post_id(r) is not None]
    eligible.sort(key=lambda r: _post_id(r))  # 安定: 物理行順非依存
    rng = random.Random(seed)
    rng.shuffle(eligible)
    return eligible[:n]


# ------------------------------------------------------------
# 段a: プロンプトバッチ出力 (gold タグ列 + lang_hint のみ・散文は書かない)
# ------------------------------------------------------------
def emit_prompts(args):
    vocab_set, _freq = load_vocab_set(args.vocab)
    tok = EvalTokenizer(args.vocab)

    train_rows = _train_rows()
    train_pids = {_post_id(r) for r in train_rows if _post_id(r) is not None}
    val_pids, _val_rows = val_post_ids_and_rows()
    dv_pids = diverse_val_post_ids()

    sel = _select_500(train_rows, args.n, args.seed)
    if len(sel) < args.n:
        raise AssertionError(
            f"抽出件数不足: {len(sel)} < {args.n} (pairs.train.jsonl の件数を確認)")
    sel_pids = {_post_id(r) for r in sel}
    assert len(sel_pids) == len(sel), "抽出 post_id に重複"

    # --- 3 assert (リーク0・最重要) ---
    # ① 抽出 post_id は train に所属
    assert sel_pids <= train_pids, "抽出 post_id が train に非所属 (抽出元齟齬)"
    # ② val と非交差
    assert not (sel_pids & val_pids), \
        f"抽出が val と交差 (リーク): {sorted(sel_pids & val_pids)[:5]}"
    # ③ diverse-val (a/b) と非交差 (新 proxy の独立性を守る最重要 assert)
    assert not (sel_pids & dv_pids), \
        f"抽出が diverse-val と交差 (汚染): {sorted(sel_pids & dv_pids)[:5]}"

    # gold⊆vocab を段a でも assert (UNK 0)。train 行は synthetic 生成時に
    # vocab 射影済みのはずだが、tags-stay-real のコピー源として再確認する。
    for r in sel:
        for t in r["tags"]:
            if tok.tag_to_id_lookup(t) == TOK_UNK:
                raise AssertionError(
                    f"抽出 post {_post_id(r)} に vocab 外 gold '{t}'")

    # --- スーパーセット assert (件数拡大の要) ---
    # --anchor (既存の小さい prompts) を渡したら、その post_id 列が今回の抽出の
    # 先頭に完全一致 (順序込み) するか assert する。これが成立すれば、anchor の
    # 著述 (part01-05 等) はバイトのまま再利用でき、追加著述は差分だけで済む。
    anchor_pids = []
    if getattr(args, "anchor", None):
        if not os.path.exists(args.anchor):
            raise AssertionError(f"--anchor が存在しない: {args.anchor}")
        anchor_rows = _read_jsonl(args.anchor)
        anchor_pids = [a["post_id"] for a in anchor_rows]
        sel_pid_seq = [_post_id(r) for r in sel]
        head = sel_pid_seq[:len(anchor_pids)]
        assert head == anchor_pids, (
            "スーパーセット違反: 抽出先頭 post_id 列が anchor と不一致 "
            "(seed/n 変更時は anchor 再生成が必要)。"
            f" 先頭不一致例 anchor[0]={anchor_pids[0] if anchor_pids else None} "
            f"sel_head[0]={head[0] if head else None}")
        print(f"[diverse-train] スーパーセット assert OK: "
              f"抽出先頭 {len(anchor_pids)} 件が anchor ({os.path.basename(args.anchor)}) "
              f"と完全一致 → 既存著述は再利用可", file=sys.stderr)

    out_path = args.prompts
    skip_main = os.path.exists(out_path) and not args.force

    lines = []
    for r in sel:
        pid = _post_id(r)
        lines.append({
            "post_id": pid,
            # gold タグ列 = train 行の tags をバイト一致コピー (整形しない)。
            "gold_tags": list(r["tags"]),
            # lang_hint は抽出元行の既存 lang を踏襲。
            "lang_hint": r.get("lang", "ja"),
            "rating": r.get("meta", {}).get("rating", "g"),
        })

    if skip_main:
        print(f"[diverse-train] {out_path} 既存 → スキップ (--force で再生成)",
              file=sys.stderr)
    else:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for ln in lines:
                f.write(json.dumps(ln, ensure_ascii=False) + "\n")

    lang_c = {"ja": 0, "en": 0}
    for ln in lines:
        lang_c[ln["lang_hint"]] = lang_c.get(ln["lang_hint"], 0) + 1
    if not skip_main:
        print(f"[diverse-train] prompts 出力: {out_path} ({len(lines)} 件)",
              file=sys.stderr)
    print(f"[diverse-train]   lang_hint: ja={lang_c['ja']} en={lang_c['en']}",
          file=sys.stderr)
    print(f"[diverse-train]   3 assert OK "
          f"(抽出⊆train / 抽出∩val=0 / 抽出∩diverse-val=0), gold⊆vocab OK",
          file=sys.stderr)

    # --- 段b 向け todo リスト切り出し (追加著述が必要な分のみ) ---
    if getattr(args, "todo", None):
        anchor_pid_set = set(anchor_pids)
        todo = [ln for ln in lines if ln["post_id"] not in anchor_pid_set]
        os.makedirs(os.path.dirname(args.todo) or ".", exist_ok=True)
        with open(args.todo, "w", encoding="utf-8") as f:
            for ln in todo:
                f.write(json.dumps(ln, ensure_ascii=False) + "\n")
        tlang = {"ja": 0, "en": 0}
        for ln in todo:
            tlang[ln["lang_hint"]] = tlang.get(ln["lang_hint"], 0) + 1
        print(f"[diverse-train]   todo (追加著述要): {args.todo} "
              f"({len(todo)} 件 = 抽出{len(lines)} − anchor{len(anchor_pids)}・"
              f"ja={tlang['ja']} en={tlang['en']})", file=sys.stderr)

    return 0


# ------------------------------------------------------------
# 段c: 取り込み・検証・Replace 合成・凍結 (今回は実行しない)
# ------------------------------------------------------------
def _read_texts(texts_path):
    """diverse_train_texts.jsonl (+ _partNN 分割) を結合して読む。"""
    rows = []
    if os.path.exists(texts_path):
        rows.extend(_read_jsonl(texts_path))
    # _partNN 分割の結合 (diverse_train_texts_part01.jsonl ... を探す)
    base = texts_path[:-len(".jsonl")] if texts_path.endswith(".jsonl") else texts_path
    parts = []
    d = os.path.dirname(texts_path) or "."
    bn = os.path.basename(base)
    for name in sorted(os.listdir(d)):
        if name.startswith(bn + "_part") and name.endswith(".jsonl"):
            parts.append(os.path.join(d, name))
    for p in parts:
        rows.extend(_read_jsonl(p))
    return rows, parts


def ingest(args):
    vocab_set, _ = load_vocab_set(args.vocab)
    tok = EvalTokenizer(args.vocab)

    train_rows = _train_rows()
    train_pids = {_post_id(r) for r in train_rows if _post_id(r) is not None}
    val_pids, _val_rows = val_post_ids_and_rows()
    dv_pids = diverse_val_post_ids()

    if not os.path.exists(args.prompts):
        print(f"[diverse-train] prompts なし: {args.prompts} (先に --emit-prompts)",
              file=sys.stderr)
        return 1

    prompts = _read_jsonl(args.prompts)
    texts, parts = _read_texts(args.texts)
    if not texts:
        print(f"[diverse-train] texts なし: {args.texts} "
              f"(main Claude が散文を書く段b が未完)", file=sys.stderr)
        return 1

    # prompts を post_id で索引化 = gold の唯一のソース (tags-stay-real)。
    gold_by_pid = {pr["post_id"]: pr for pr in prompts}
    sel_pids = set(gold_by_pid.keys())

    errors = []
    written = {}   # post_id -> 著述行 (1 post 1 text)
    for i, tx in enumerate(texts):
        pid = tx.get("post_id")
        text = tx.get("text", "")
        lang = tx.get("lang")
        if pid not in gold_by_pid:
            errors.append(f"texts#{i}: prompts に無い post_id={pid}")
            continue
        pr = gold_by_pid[pid]

        # tags-stay-real: gold は段a プロンプトの gold_tags をバイト不変コピー。
        gold = list(pr["gold_tags"])

        # 検証: text 非空
        if not text or not text.strip():
            errors.append(f"texts#{i}: text 空 (post_id={pid})")
        # 検証: post_id が text に漏れない
        if str(pid) in text:
            errors.append(f"texts#{i}: post_id {pid} が text に漏出")
        # 検証: lang∈{ja,en}
        if lang not in ("ja", "en"):
            errors.append(f"texts#{i}: lang 不正 '{lang}' (post_id={pid})")
        # 検証: gold⊆vocab
        for t in gold:
            if tok.tag_to_id_lookup(t) == TOK_UNK:
                errors.append(f"texts#{i}: vocab 外 gold '{t}' (post_id={pid})")

        if pid in written:
            errors.append(f"texts#{i}: post_id {pid} 重複著述")
            continue

        written[pid] = {
            "text": text,
            "tags": gold,
            "lang": lang,
            "source": "llm_distill",
            "meta": {
                "post_id": pid,
                "rating": pr.get("rating", "g"),
                "n_tags": len(gold),
                "gen": "claude",
                "tmpl": -1,
            },
        }

    # 全 prompts に対応する著述が揃っているか
    missing = sel_pids - set(written.keys())
    if missing:
        errors.append(
            f"著述欠落 {len(missing)} 件 (例 {sorted(missing)[:5]})")

    # 著述行のリーク再確認 (段a と同じ 3 不変条件)。
    w_pids = set(written.keys())
    if not (w_pids <= train_pids):
        errors.append("著述 post_id が train に非所属")
    if w_pids & val_pids:
        errors.append(f"著述が val と交差: {sorted(w_pids & val_pids)[:5]}")
    if w_pids & dv_pids:
        errors.append(f"著述が diverse-val と交差: {sorted(w_pids & dv_pids)[:5]}")

    if errors:
        print(f"[diverse-train] 検証エラー {len(errors)} 件 (先頭10):",
              file=sys.stderr)
        for e in errors[:10]:
            print("   ", e, file=sys.stderr)
        return 2

    # --- Replace 合成: 著述n + synthetic(4500-n) = 4500 ---
    # synthetic = pairs.train.jsonl から段a抽出の n 件を除いた残り (バイトコピー)。
    synthetic = [r for r in train_rows if _post_id(r) not in sel_pids]
    authored = [written[pr["post_id"]] for pr in prompts]  # prompts 順で安定

    out_rows = authored + synthetic

    # --- 件数・重複アサート (厳守) ---
    assert len(authored) == args.n, \
        f"著述件数 {len(authored)} != {args.n}"
    assert len(synthetic) == DEF_TRAIN_TOTAL - args.n, \
        f"synthetic 残数 {len(synthetic)} != {DEF_TRAIN_TOTAL - args.n}"
    assert len(out_rows) == DEF_TRAIN_TOTAL, \
        f"合計 {len(out_rows)} != {DEF_TRAIN_TOTAL} (件数厳守)"
    out_pids = [_post_id(r) for r in out_rows]
    assert len(set(out_pids)) == len(out_pids), "合成後 post_id 重複"

    out_path = args.out_train
    if os.path.exists(out_path) and not args.force:
        print(f"[diverse-train] {out_path} 既存 → スキップ (--force で再生成)",
              file=sys.stderr)
    else:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[diverse-train] Replace 合成出力: {out_path} "
              f"(著述{len(authored)} + synthetic{len(synthetic)} = {len(out_rows)})",
              file=sys.stderr)

    # --- 統計 ---
    lang_c = {"ja": 0, "en": 0}
    for r in authored:
        lang_c[r["lang"]] = lang_c.get(r["lang"], 0) + 1
    uniq_text = len({r["text"] for r in authored})
    stats = {
        "version": 1,
        "policy": f"施策B B-1 / Replace / {len(authored)}件著述",
        "seed": args.seed,
        "n_authored": len(authored),
        "n_synthetic": len(synthetic),
        "n_total": len(out_rows),
        "authored_lang_ratio": {
            k: round(v / max(1, len(authored)), 4) for k, v in lang_c.items()},
        "authored_unique_text_ratio": round(uniq_text / max(1, len(authored)), 4),
        "authored_post_ids": sorted(sel_pids),
        "validation_ok": True,
        "validation": "tags-stay-real / gold⊆vocab / リーク0(train∋・val∩=0・"
                      "diverse-val∩=0) / text非空 / lang∈{ja,en} / 件数4500 / 重複0",
        "text_parts": [os.path.basename(p) for p in parts],
        "sources": {
            "extracted_from": "pairs.train.jsonl",
            "authored_by": "claude (human-in-the-loop)",
        },
    }
    stats_path = args.stats
    if os.path.exists(stats_path) and not args.force:
        print(f"[diverse-train] {stats_path} 既存 → スキップ", file=sys.stderr)
    else:
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=1)
        print(f"[diverse-train] stats: {stats_path}", file=sys.stderr)

    print(f"[diverse-train] 取り込み OK: 著述{len(authored)} "
          f"(ja={lang_c['ja']} en={lang_c['en']} uniq_text={uniq_text}) "
          f"+ synthetic{len(synthetic)} = {len(out_rows)} "
          f"(tags-stay-real / リーク0 / 件数4500 / 重複0 検証済)",
          file=sys.stderr)
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="dollama 施策B B-1 diverse-train 構築 (入力多様化)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit-prompts", action="store_true",
                      help="段a: train から抽出 → gold タグ+lang_hint のプロンプト出力 "
                           "(散文は生成しない)")
    mode.add_argument("--ingest", action="store_true",
                      help="段c: texts を取り込み検証・Replace 合成・凍結")
    ap.add_argument("--seed", type=int, default=20260620)
    ap.add_argument("--n", type=int, default=DEF_N,
                    help="著述件数 (Replace なので train 総数は 4500 不変)")
    ap.add_argument("--vocab", default=DEF_VOCAB)
    ap.add_argument("--prompts", default=DEF_PROMPTS)
    ap.add_argument("--texts", default=DEF_TEXTS)
    ap.add_argument("--out-train", default=DEF_OUT_TRAIN)
    ap.add_argument("--stats", default=DEF_STATS)
    ap.add_argument("--anchor", default=None,
                    help="段a: 既存の小さい prompts。抽出先頭 post_id 列がこれと"
                         "完全一致するか assert (スーパーセット性の機械保証)")
    ap.add_argument("--todo", default=None,
                    help="段a: --anchor がカバーしない post_id だけを切り出した "
                         "追加著述 todo リストの出力先")
    ap.add_argument("--force", action="store_true",
                    help="既存の凍結出力を上書き再生成")
    args = ap.parse_args()

    if args.emit_prompts:
        return emit_prompts(args)
    else:
        return ingest(args)


if __name__ == "__main__":
    raise SystemExit(main())
