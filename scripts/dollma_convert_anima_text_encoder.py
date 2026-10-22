#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dollma_convert_anima_text_encoder.py — Anima 自作実装トラック A1: Qwen3 0.6B TE の OpenVINO 変換。

Anima の text encoder は Qwen3-0.6B-Base の last_hidden_state (final RMSNorm 後・1 回 forward・KV cache 不要)。
これを OV IR (FP16・静的 [1,L]) に変換して NPU に載せる。scripts/dollma_convert_sdxl_text_encoders.py の型
(wrapper → ov.convert_model → 入出力名の明示 → reshape → save → torch 突合) に揃える。

=== IR の入出力 (C++ 側 src/infer/qwen3_encoder.hpp への申し送り・要確認事項) ===
  入力 0: input_ids       shape [1, L]  element_type i64   (★i32 を渡すと NPU で 0xC0000409 の恐れ・CLIP と同じ)
  入力 1: attention_mask  shape [1, L]  element_type i64   (1 = 有効 / 0 = pad・右 pad 前提・先頭 n 個が 1)
  出力 0: last_hidden_state shape [1, L, 1024] element_type f32
  ★有効長 n (= attention_mask の和) の先頭 n 行 [0:n] だけが意味を持つ。pad 位置の出力は捨てること。
  ★llm_adapter は source_attention_mask=None (pipeline 実装) で Qwen hidden を cross-attn に使う。
    つまり adapter へは [0:n] に切った hidden を渡す (pad 行を混ぜると結果が変わる)。
  ★右 pad + causal なので、有効 token の出力は「pad 無し (unpadded) の torch 出力」と一致する (--verify で確認)。
  pad token id は 151643 (<|endoftext|>)。特殊 token は付けない (pipeline は add_special_tokens=False)。

=== 推測で決めた箇所 (参照実装に明記なし) ===
  - mask 方式: 既定 "explicit" = wrapper 内で 4D 加算 mask [B,1,L,L] (causal + key padding・負値は -30000 の有限値) を
    自前で作り HF に渡す。FP16 実行時に finfo.min が -inf 化して NaN になるのを避ける設計判断。"native" は HF 任せ。
  - L の既定 128: A0 の prompt 長分布で再決定する前提の仮値 (--lengths で分布を出す)。
  - 保存は FP16 圧縮 (ov.save_model の compress_to_fp16)。FP32 IR は --save-fp32 で別名保存 (1e-5 級の変換忠実度確認用)。

=== 使い方 (研究機・anima venv: models/anima/.venv/Scripts/python.exe) ===
  --selftest                         重み不要の純関数テスト (+ OV があれば極小ランダム Qwen3 の変換テスト)
  --lengths                          E-1 prompt セットの Qwen3/T5 token 長分布と L の目安 (トークナイザのみ・重み不要)
  --convert [--seq-len 128]          重みロード → in-memory FP32 で変換忠実度確認 → FP16 IR 保存
  --tokenizers                       Qwen3 / T5 の OV tokenizer を変換・保存・HF と突合
  --verify                           保存済み IR を CPU(f32 hint) で torch と突合 (複数 mask)
  --bench --devices CPU GPU.0 NPU    デバイス別 compile 可否・compile 秒・推論中央値 (N=100)・torch 比誤差
  --all                              convert + tokenizers + verify
"""

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# constants.py::QWEN3_06B_CONFIG (diffusers_anima) の写し
QWEN3_06B_CONFIG = {
    "vocab_size": 151936,
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "max_position_embeddings": 32768,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "attention_bias": False,
}
HIDDEN = 1024
QWEN_PAD_ID = 151643          # <|endoftext|>
T5_EOS_ID = 1                 # text_encoding.AnimaPromptTokenizer: 末尾に eos を足す
T5_PAD_ID = 0
DEFAULT_SEQ_LEN = 128         # 仮値 (A0 の分布で再決定)
NEG_MASK = -30000.0           # FP16 でも有限 (65504 以内) な加算 mask 値
INPUT_NAMES = ["input_ids", "attention_mask"]
OUTPUT_NAMES = ["last_hidden_state"]


# ============================================================
# 純関数 (重み不要)
# ============================================================
def pad_ids(ids, seq_len, pad_id=QWEN_PAD_ID):
    """可変長 id 列 → (input_ids [1,L] i64, attention_mask [1,L] i64, n)。右 pad。L 超過は ValueError。"""
    n = len(ids)
    if n < 1:
        raise ValueError("空 prompt は不可 (pipeline は pad 1 個に置換する)")
    if n > seq_len:
        raise ValueError(f"token 数 {n} が静的長 L={seq_len} を超える (切り詰めは意味を変えるので拒否)")
    a = np.full((1, seq_len), pad_id, dtype=np.int64)
    m = np.zeros((1, seq_len), dtype=np.int64)
    a[0, :n] = np.asarray(ids, dtype=np.int64)
    m[0, :n] = 1
    return a, m, n


def valid_len(attention_mask):
    return int(np.asarray(attention_mask).sum())


def suggest_static_len(max_len, quantum=16, margin=1.25):
    """最大 token 長に余裕 margin を掛け quantum の倍数へ切り上げ。"""
    need = int(np.ceil(max_len * margin))
    return int(((need + quantum - 1) // quantum) * quantum)


def explicit_attention_mask(attention_mask, dtype=torch.float32, neg=NEG_MASK):
    """[B,L] (1/0) → 加算 mask [B,1,L,L]。許可 = causal (j<=i) かつ key j が有効。禁止位置は neg (有限値)。"""
    b, l = attention_mask.shape
    causal = torch.ones(l, l, dtype=torch.bool).tril()
    key_ok = attention_mask.to(torch.bool)[:, None, None, :]
    allowed = causal[None, None, :, :] & key_ok
    return (~allowed).to(dtype) * neg


def latency_stats(ms_list):
    xs = sorted(ms_list)
    n = len(xs)
    p95 = xs[min(n - 1, int(np.ceil(0.95 * n)) - 1)]
    return {"n": n, "median_ms": statistics.median(xs), "p95_ms": p95,
            "mean_ms": statistics.fmean(xs), "min_ms": xs[0], "max_ms": xs[-1]}


def compare_hidden(ref, got, n=None):
    """ref/got: [1,L,D] numpy。先頭 n 行 (有効 token) のみ比較。max/MAE/相対/token 別 cosine 最小。"""
    r = np.asarray(ref, dtype=np.float64)
    g = np.asarray(got, dtype=np.float64)
    if n is not None:
        r, g = r[:, :n], g[:, :n]
    d = np.abs(r - g)
    rm = float(np.abs(r).max()) or 1.0
    rf, gf = r.reshape(-1, r.shape[-1]), g.reshape(-1, g.shape[-1])
    num = (rf * gf).sum(-1)
    den = np.linalg.norm(rf, axis=-1) * np.linalg.norm(gf, axis=-1)
    cos = num / np.maximum(den, 1e-30)
    # token 別の相対誤差 (token0 は massive activation で桁が違うので rest と分けて見る)
    tok_ref = np.maximum(np.abs(r).max(axis=-1), 1e-30)           # [B,n]
    tok_rel = d.max(axis=-1) / tok_ref                            # [B,n]
    token0_rel = float(tok_rel[:, 0].max())
    rest_rel = float(tok_rel[:, 1:].max()) if tok_rel.shape[1] > 1 else None
    return {"max_abs_err": float(d.max()), "mae": float(d.mean()), "rel_max_err": float(d.max() / rm),
            "ref_max_abs": rm, "cos_min": float(cos.min()), "cos_mean": float(cos.mean()),
            "token0_rel_err": token0_rel, "rest_rel_err_max": rest_rel,
            "token0_max_abs": float(np.abs(r[:, 0]).max()),
            "rest_max_abs": float(np.abs(r[:, 1:]).max()) if r.shape[1] > 1 else None,
            "nonfinite": bool(not np.isfinite(g).all())}


def strip_model_prefix(sd):
    if sd and all(k.startswith("model.") for k in sd):
        return {k[6:]: v for k, v in sd.items()}
    return dict(sd)


def make_qwen3(config_dict=None, attn="sdpa"):
    """Qwen3Model を (重みなし・ランダム初期化で) 組む。config_dict 省略で Anima の 0.6B。"""
    from transformers import Qwen3Config, Qwen3Model
    cfg = Qwen3Config(**(config_dict or QWEN3_06B_CONFIG))
    cfg._attn_implementation = attn
    m = Qwen3Model(cfg)
    return m.eval()


def load_qwen3_weights(model, te_path):
    from safetensors.torch import load_file
    sd = strip_model_prefix(load_file(te_path, device="cpu"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"TE 重みが Qwen3-0.6B と不一致: missing={len(missing)} unexpected={len(unexpected)}")
    return model


class Qwen3TEWrapper(torch.nn.Module):
    """(input_ids i64 [1,L], attention_mask i64 [1,L]) → last_hidden_state [1,L,D]。"""

    def __init__(self, te, mask_mode="explicit"):
        super().__init__()
        if mask_mode not in ("explicit", "native"):
            raise ValueError(mask_mode)
        self.te = te
        self.mask_mode = mask_mode

    def forward(self, input_ids, attention_mask):
        if self.mask_mode == "explicit":
            m = explicit_attention_mask(attention_mask, dtype=torch.float32)
        else:
            m = attention_mask
        out = self.te(input_ids=input_ids, attention_mask=m, use_cache=False, return_dict=True)
        return out.last_hidden_state


def torch_reference(te, ids_np, mask_np):
    """pipeline と同じ呼び出し (有効 token のみ・pad 無し) の torch 出力。[1,n,D] numpy。"""
    n = valid_len(mask_np)
    with torch.no_grad():
        out = te(input_ids=torch.from_numpy(ids_np[:, :n]).long(),
                 attention_mask=torch.ones(1, n, dtype=torch.long))
    return out.last_hidden_state.float().numpy()


# ============================================================
# 変換
# ============================================================
def convert_te(te, seq_len, out_dir, mask_mode="explicit", save_fp32=False, tag=""):
    """wrapper → ov.convert_model → 名前付け → reshape [1,L] → FP16 IR 保存。戻り: (ov_model, info)。"""
    import openvino as ov
    wrapper = Qwen3TEWrapper(te.float().eval(), mask_mode).eval()
    ex_ids = torch.zeros(1, seq_len, dtype=torch.long)
    ex_mask = torch.ones(1, seq_len, dtype=torch.long)
    t0 = time.time()
    with torch.no_grad():
        ov_model = ov.convert_model(wrapper, example_input=(ex_ids, ex_mask))
    conv_sec = time.time() - t0

    # 入出力名を明示 (SDXL 変換と同じ流儀。C++ が名前で引けるように)
    for inp, nm in zip(ov_model.inputs, INPUT_NAMES):
        inp.get_tensor().set_names({nm})
    for out, nm in zip(ov_model.outputs, OUTPUT_NAMES):
        out.get_tensor().set_names({nm})
    # NPU 必須: 静的形状
    ov_model.reshape({"input_ids": [1, seq_len], "attention_mask": [1, seq_len]})

    io = {"inputs": [{"name": i.get_any_name(), "shape": list(i.shape),
                      "element_type": i.get_element_type().get_type_name()} for i in ov_model.inputs],
          "outputs": [{"index": k, "name": o.get_any_name(), "shape": list(o.shape),
                       "element_type": o.get_element_type().get_type_name()} for k, o in enumerate(ov_model.outputs)]}
    print("  入力:")
    for i in io["inputs"]:
        print(f"    name={i['name']!r} shape={i['shape']} element_type={i['element_type']}")
    print("  出力:")
    for o in io["outputs"]:
        print(f"    [{o['index']}] name={o['name']!r} shape={o['shape']} element_type={o['element_type']}")
    assert len(ov_model.outputs) == 1 and len(ov_model.inputs) == 2

    info = {"seq_len": seq_len, "mask_mode": mask_mode, "convert_sec": round(conv_sec, 2), **io}
    return ov_model, info


def save_ir(ov_model, out_dir, seq_len, fp16=True, name="model_ov"):
    import openvino as ov
    os.makedirs(out_dir, exist_ok=True)
    xml = os.path.join(out_dir, f"{name}.xml")
    t0 = time.time()
    ov.save_model(ov_model, xml, compress_to_fp16=fp16)
    sec = time.time() - t0
    bin_mb = os.path.getsize(xml.replace(".xml", ".bin")) / 1024 / 1024
    print(f"  -> {xml}  (bin {bin_mb:.1f} MB, {'FP16' if fp16 else 'FP32'}, {sec:.1f}s)")
    return xml, bin_mb


def compile_cpu_f32(model_or_path):
    import openvino as ov
    core = ov.Core()
    m = core.read_model(model_or_path) if isinstance(model_or_path, str) else model_or_path
    return core.compile_model(m, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})


def run_ov(compiled, ids_np, mask_np):
    req = compiled.create_infer_request()
    res = req.infer({"input_ids": ids_np.astype(np.int64), "attention_mask": mask_np.astype(np.int64)})
    return res[compiled.outputs[0]]


def verify_cases(seq_len, tok_ids=None):
    """突合ケース [(label, ids_np, mask_np)]。実 prompt (任意)・満杯・n=1・n=L/2 (乱数 id)。"""
    rng = np.random.default_rng(0)
    cases = []
    if tok_ids:
        i, m, _ = pad_ids(tok_ids, seq_len)
        cases.append((f"real(n={len(tok_ids)})", i, m))
    for label, n in (("full", seq_len), ("n=1", 1), ("half", max(2, seq_len // 2))):
        ids = rng.integers(0, 150000, size=n).tolist()
        i, m, _ = pad_ids(ids, seq_len)
        cases.append((f"{label}(n={n})", i, m))
    return cases


def verify_against_torch(compiled, te, cases, label):
    rows = []
    for name, ids, mask in cases:
        n = valid_len(mask)
        ref = torch_reference(te, ids, mask)               # [1,n,D] (pad 無し)
        got = run_ov(compiled, ids, mask)[:, :n]           # [1,n,D]
        c = compare_hidden(ref, got)
        rows.append({"case": name, **c})
        print(f"  [{label}] {name:<16} max_abs={c['max_abs_err']:.3e} MAE={c['mae']:.3e} "
              f"rel={c['rel_max_err']:.2e} cos_min={c['cos_min']:.8f} (ref max|x|={c['ref_max_abs']:.1f}) "
              f"| token0 rel={c['token0_rel_err']:.2e}"
              + (f" rest rel(max)={c['rest_rel_err_max']:.2e}" if c['rest_rel_err_max'] is not None else ""))
    return rows


# ============================================================
# トークナイザ
# ============================================================
def load_hf_tokenizers(tok_dir):
    from transformers import AutoTokenizer
    q = AutoTokenizer.from_pretrained(os.path.join(tok_dir, "prompt_tokenizer_qwen"))
    t = AutoTokenizer.from_pretrained(os.path.join(tok_dir, "prompt_tokenizer_t5"))
    return q, t


def pipeline_qwen_ids(qtok, text):
    """text_encoding.tokenize_with_weights 相当 (特殊 token なし・空は pad 1 個)。"""
    ids = qtok([text], add_special_tokens=False, truncation=False)["input_ids"][0]
    return list(ids) if len(ids) else [QWEN_PAD_ID]


def pipeline_t5_ids(ttok, text):
    """同上。末尾が eos でなければ eos を足す・空は [1]。"""
    ids = list(ttok([text], add_special_tokens=False, truncation=False)["input_ids"][0])
    if not ids:
        return [T5_EOS_ID]
    if ids[-1] != T5_EOS_ID:
        ids.append(T5_EOS_ID)
    return ids


def convert_tokenizers(tok_dir, out_dir, seq_len):
    """Qwen3: 静的 [1,L] 右 pad (input_ids + attention_mask 出力)。T5: 可変長・pad 無し (eos は C++ で付加)。"""
    import openvino as ov
    from openvino_tokenizers import convert_tokenizer
    q, t = load_hf_tokenizers(tok_dir)
    q.padding_side = "right"
    outs = {}
    for label, tok, kw in (
        ("qwen", q, dict(max_length=seq_len, use_max_padding=True, add_special_tokens=False)),
        ("t5", t, dict(add_special_tokens=False)),
    ):
        d = os.path.join(out_dir, label)
        os.makedirs(d, exist_ok=True)
        ov_tok = convert_tokenizer(tok, with_detokenizer=False, **kw)
        xml = os.path.join(d, "openvino_tokenizer.xml")
        ov.save_model(ov_tok, xml)
        outs[label] = xml
        print(f"  -> {xml}  (pad={tok.pad_token_id} eos={tok.eos_token_id})")
    return outs, (q, t)


def verify_tokenizers(xmls, hf, seq_len, texts, gated=None):
    """gated: FAIL 判定に使う label の集合 (None なら全件)。それ以外は情報扱い (行に "gated": False)。"""
    import openvino as ov
    core = ov.Core()
    cq = core.compile_model(core.read_model(xmls["qwen"]), "CPU")
    ct = core.compile_model(core.read_model(xmls["t5"]), "CPU")
    q, t = hf
    rows = []
    for label, s in texts:
        ref_q = pipeline_qwen_ids(q, s)
        ref_t = pipeline_t5_ids(t, s)
        oq = cq([s])
        ids_q = oq["input_ids"][0].tolist()
        n_q = int(oq["attention_mask"][0].sum())
        ov_q_ok = (len(ref_q) <= seq_len and ids_q[:n_q] == ref_q
                   and all(i == QWEN_PAD_ID for i in ids_q[n_q:])) if s else None
        ids_t = ct([s])["input_ids"][0].tolist()
        ov_t = (ids_t + [T5_EOS_ID]) if (not ids_t or ids_t[-1] != T5_EOS_ID) else ids_t
        ov_t_ok = (ov_t == ref_t) if s else None
        rows.append({"label": label, "qwen_n": len(ref_q), "t5_n": len(ref_t),
                     "qwen_match": ov_q_ok, "t5_match": ov_t_ok,
                     "gated": (gated is None) or (label in gated)})
        print(f"  [{label:<10}] qwen n={len(ref_q):>3} match={ov_q_ok}  t5 n={len(ref_t):>3} match={ov_t_ok}")
    return rows


def tokenizer_gate_ok(rows):
    """gated 行の qwen/t5 がすべて True なら OK (空 prompt などの None は gated に入れない前提)。"""
    return all(r["qwen_match"] is True and r["t5_match"] is True for r in rows if r.get("gated"))


EMPTY_PROMPT_RULE = ("空 prompt: Qwen は pad(151643) 1 個 (mask=1)、T5 は [eos=1] 1 個に置換 (pipeline の規則)。"
                     "OV tokenizer は空文字で n=0 を返すので C++ 側で特殊処理が必要。")

TOKENIZER_TEXTS_EXTRA = [
    ("tags-sym", "1girl, (masterpiece:1.2), [x], <b>"),
    ("japanese", "日本語のテスト"),
    ("newline", "a  b\nc"),   # T5 は改行/連続空白の扱いが HF と OV で異なりうる (情報目的)
    ("empty", ""),
]


def e1_texts():
    """E-1 の p1〜p4 を anima-base / anima-aesthetic の prefix 付きで (positive) + negative。"""
    import dollma_probe_anima as P
    import dollma_eval_image_grid as G
    out = []
    for mk in P.MODEL_SPECS:
        for pid in G.PROMPTS:
            out.append((f"{mk[6:10]}-{pid}-pos", P.build_prompt(mk, pid, G.PROMPTS)))
            out.append((f"{mk[6:10]}-{pid}-neg", P.build_negative(mk, pid, G.PROMPTS)))
    return out


def cmd_lengths(tok_dir):
    q, t = load_hf_tokenizers(tok_dir)
    rows = []
    for label, s in e1_texts():
        rows.append((label, len(pipeline_qwen_ids(q, s)), len(pipeline_t5_ids(t, s))))
    print(f"{'prompt':<18}{'qwen3':>7}{'t5(+eos)':>10}")
    for r in rows:
        print(f"{r[0]:<18}{r[1]:>7}{r[2]:>10}")
    mq = max(r[1] for r in rows)
    mt = max(r[2] for r in rows)
    print(f"\nmax: qwen3={mq} t5={mt}")
    print(f"L の目安 (max*1.25 を 16 の倍数へ): {suggest_static_len(mq)} "
          f"(2 の冪の候補: 64 / 128 / 256。E-1 の 4 題材より長いユーザー prompt は L 超過で失敗させるか CPU/GPU 版へ退避)")
    return rows


# ============================================================
# ベンチ (デバイス別 compile 可否・時間)
# ============================================================
def bench_device(xml, device, te, seq_len, n_iter, warmup, cases):
    import openvino as ov
    core = ov.Core()
    res = {"device": device}
    try:
        t0 = time.time()
        compiled = core.compile_model(core.read_model(xml), device)
        res["compile_sec"] = round(time.time() - t0, 2)
        res["compile_ok"] = True
    except Exception as e:  # noqa: BLE001 (デバイス固有の失敗を記録するのが目的)
        res["compile_ok"] = False
        res["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        print(f"  [{device}] compile 失敗: {res['error']}")
        return res

    name, ids, mask = cases[0]
    req = compiled.create_infer_request()
    feed = {"input_ids": ids.astype(np.int64), "attention_mask": mask.astype(np.int64)}
    try:
        for _ in range(warmup):
            req.infer(feed)
        ts = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            req.infer(feed)
            ts.append((time.perf_counter() - t0) * 1000.0)
        res["latency"] = latency_stats(ts)
        res["latency_case"] = name
    except Exception as e:  # noqa: BLE001
        res["infer_ok"] = False
        res["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        print(f"  [{device}] 推論失敗: {res['error']}")
        return res
    res["infer_ok"] = True
    res["errors_vs_torch"] = verify_against_torch(compiled, te, cases, device)
    L = res["latency"]
    print(f"  [{device}] compile {res['compile_sec']}s  median={L['median_ms']:.2f}ms p95={L['p95_ms']:.2f}ms "
          f"(N={L['n']}, {name})")
    return res


# ============================================================
# selftest
# ============================================================
def selftest():
    fails = []

    def check(name, cond):
        print(f"[selftest] {'PASS' if cond else 'FAIL'} {name}")
        if not cond:
            fails.append(name)

    # pad_ids
    i, m, n = pad_ids([5, 6, 7], 8)
    check("pad_ids 形状・dtype i64", i.shape == (1, 8) and i.dtype == np.int64 and m.dtype == np.int64)
    check("pad_ids 右 pad", i[0].tolist() == [5, 6, 7] + [QWEN_PAD_ID] * 5 and m[0].tolist() == [1, 1, 1, 0, 0, 0, 0, 0])
    check("valid_len", valid_len(m) == n == 3)
    for bad in ([], list(range(9))):
        try:
            pad_ids(bad, 8)
            check(f"pad_ids 異常入力 len={len(bad)} は ValueError", False)
        except ValueError:
            check(f"pad_ids 異常入力 len={len(bad)} は ValueError", True)

    check("suggest_static_len", suggest_static_len(50) == 64 and suggest_static_len(100) == 128
          and suggest_static_len(12) == 16)

    # explicit mask: causal + key padding
    am = torch.tensor([[1, 1, 1, 0, 0]])
    em = explicit_attention_mask(am)
    allowed = (em[0, 0] == 0)
    check("explicit mask 形状", tuple(em.shape) == (1, 1, 5, 5))
    check("explicit mask: causal", bool(allowed[0, 0]) and not bool(allowed[0, 1]) and bool(allowed[2, 0]))
    check("explicit mask: pad key は禁止", not bool(allowed[:, 3:].any()))
    check("explicit mask: 有限・FP16 範囲内", bool(torch.isfinite(em).all()) and float(em.min()) > -65504.0)
    check("explicit mask: 全行に許可 key が 1 つ以上 (NaN 回避)", bool(allowed.any(dim=-1).all()))

    # compare_hidden
    r = np.random.default_rng(0).standard_normal((1, 4, 8))
    c = compare_hidden(r, r.copy())
    check("compare_hidden 同一 → err 0・cos 1", c["max_abs_err"] == 0 and abs(c["cos_min"] - 1.0) < 1e-12)
    c2 = compare_hidden(r, r + 1e-3, n=2)
    check("compare_hidden n で先頭のみ", abs(c2["max_abs_err"] - 1e-3) < 1e-9)
    r_nan = r.copy()
    r_nan[0, 0, 0] = np.nan
    check("compare_hidden nonfinite 検出", compare_hidden(r, r_nan)["nonfinite"] is True)
    check("compare_hidden 有限なら nonfinite=False", compare_hidden(r, r)["nonfinite"] is False)
    r2 = r.copy()
    r2[0, 0] += 1.0
    c3 = compare_hidden(r, r2)
    check("compare_hidden token0 / rest 分離", c3["token0_rel_err"] > 0 and c3["rest_rel_err_max"] == 0.0)

    # latency_stats
    ls = latency_stats(list(range(1, 101)))
    check("latency_stats 中央値/p95", ls["median_ms"] == 50.5 and ls["p95_ms"] == 95 and ls["n"] == 100)

    # strip prefix
    check("strip_model_prefix", set(strip_model_prefix({"model.a": 1, "model.b": 2})) == {"a", "b"}
          and set(strip_model_prefix({"model.a": 1, "b": 2})) == {"model.a", "b"})

    # pipeline T5 / Qwen の空・eos 規則 (擬似 tokenizer)
    class FakeTok:
        def __init__(self, ids):
            self.ids = ids

        def __call__(self, texts, add_special_tokens=False, truncation=False):
            return {"input_ids": [list(self.ids)]}
    check("pipeline_qwen_ids 空 → [pad]", pipeline_qwen_ids(FakeTok([]), "") == [QWEN_PAD_ID])
    check("pipeline_t5_ids 空 → [eos]", pipeline_t5_ids(FakeTok([]), "") == [T5_EOS_ID])
    check("pipeline_t5_ids eos 付加", pipeline_t5_ids(FakeTok([7, 8]), "x") == [7, 8, T5_EOS_ID])
    check("pipeline_t5_ids eos 重複なし", pipeline_t5_ids(FakeTok([7, T5_EOS_ID]), "x") == [7, T5_EOS_ID])

    # 極小 Qwen3 の変換テスト (OV があれば)
    try:
        import openvino  # noqa: F401
        have_ov = True
    except ImportError:
        have_ov = False
    if have_ov:
        ok = tiny_conversion_test()
        check("極小 Qwen3: 変換 → CPU(f32) が torch と一致・mask が動的に効く", ok)
    else:
        print("[selftest] SKIP 極小 Qwen3 変換 (openvino 無し)")

    print("[selftest] ALL PASS" if not fails else f"[selftest] {len(fails)} FAIL: {fails}")
    return not fails


TINY_CFG = {"vocab_size": 512, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "max_position_embeddings": 256,
            "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "attention_bias": False}


def tiny_conversion_test(mask_mode="explicit", attn="sdpa", seq_len=16, tol=1e-4):
    """極小ランダム Qwen3 で wrapper→OV→CPU(f32) を torch(pad 無し) と突合。mask が baked でなく動的に効くことも見る。"""
    torch.manual_seed(0)
    te = make_qwen3(TINY_CFG, attn)
    ov_model, _ = convert_te(te, seq_len, None, mask_mode)
    compiled = compile_cpu_f32(ov_model)
    rng = np.random.default_rng(1)
    worst = 0.0
    for n in (1, 5, seq_len):
        ids = rng.integers(0, 500, size=n).tolist()
        i, m, _ = pad_ids(ids, seq_len)
        ref = torch_reference(te, i, m)
        got = run_ov(compiled, i, m)[:, :n]
        worst = max(worst, compare_hidden(ref, got)["max_abs_err"])
    print(f"    (極小変換 mask={mask_mode} attn={attn}: worst max_abs_err={worst:.2e})")
    return worst < tol


# ============================================================
# main
# ============================================================
def main():
    import dollma_probe_anima as P
    ap = argparse.ArgumentParser(description="Anima A1: Qwen3 0.6B TE → OV IR (静的 [1,L])")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--lengths", action="store_true")
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--tokenizers", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--all", action="store_true", help="convert + tokenizers + verify")
    ap.add_argument("--seq-len", dest="seq_len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--mask-mode", dest="mask_mode", choices=["explicit", "native"], default="explicit")
    ap.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--save-fp32", dest="save_fp32", action="store_true", help="FP32 IR も別名保存")
    ap.add_argument("--te-path", dest="te_path", default=P.TE_PATH)
    ap.add_argument("--tokenizer-dir", dest="tok_dir", default=P.TOKENIZER_DIR)
    ap.add_argument("--out-dir", dest="out_dir", default=os.path.join(P.MODELS, "anima-text-encoder"))
    ap.add_argument("--tokenizers-out", dest="tok_out", default=os.path.join(P.MODELS, "anima-tokenizers"))
    ap.add_argument("--devices", nargs="+", default=["CPU", "GPU.0", "NPU"])
    ap.add_argument("--n", type=int, default=100, help="ベンチの計測回数")
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.lengths:
        cmd_lengths(args.tok_dir)
        return
    if args.all:
        args.convert = args.tokenizers = args.verify = True
    if not (args.convert or args.tokenizers or args.verify or args.bench):
        ap.print_help()
        return

    L = args.seq_len
    xml = os.path.join(args.out_dir, "model_ov.xml")
    meta_path = os.path.join(args.out_dir, "export_meta.json")
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    te = None
    exit_code = 0

    def get_te():
        nonlocal te
        if te is None:
            t0 = time.time()
            te = load_qwen3_weights(make_qwen3(QWEN3_06B_CONFIG, args.attn), args.te_path).float().eval()
            print(f"[A1] TE 重みロード {time.time() - t0:.1f}s")
        return te

    q_ids_real = None
    if os.path.isdir(args.tok_dir):
        try:
            q, _ = load_hf_tokenizers(args.tok_dir)
            import dollma_eval_image_grid as G
            q_ids_real = pipeline_qwen_ids(q, P.build_prompt("anima-base-v1.0", "p2", G.PROMPTS))
        except Exception as e:  # noqa: BLE001
            print(f"[A1] 実 prompt の token 化に失敗 (乱数ケースのみで突合): {e}")

    if args.convert:
        print(f"\n{'=' * 60}\nQwen3 0.6B TE を OV IR に変換 (L={L}, mask={args.mask_mode}, attn={args.attn})\n{'=' * 60}")
        ov_model, info = convert_te(get_te(), L, args.out_dir, args.mask_mode)
        # 変換忠実度: in-memory FP32 (圧縮前) を CPU f32 で torch と突合
        print("  [convert] in-memory FP32 グラフの忠実度:")
        verify_against_torch(compile_cpu_f32(ov_model), get_te(), verify_cases(L, q_ids_real), "fp32-graph")
        xml, bin_mb = save_ir(ov_model, args.out_dir, L, fp16=True)
        if args.save_fp32:
            save_ir(ov_model, args.out_dir, L, fp16=False, name="model_ov_fp32")
        info.update({"ir_xml": xml, "ir_bin_mb": round(bin_mb, 1), "precision": "FP16 (compress_to_fp16)",
                     "pad_token_id": QWEN_PAD_ID,
                     "semantics": "last_hidden_state (final RMSNorm 後)。有効 token [0:n] のみ使用。右 pad・causal。"})
        meta["convert"] = info
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)

    if args.tokenizers:
        print(f"\n{'=' * 60}\nQwen3 / T5 OV tokenizer 変換 (Qwen は [1,{L}] 右 pad)\n{'=' * 60}")
        xmls, hf = convert_tokenizers(args.tok_dir, args.tok_out, L)
        e1 = e1_texts()      # p1〜p4 × pos/neg × 2 prefix = 16 本 → 不一致は FAIL
        texts = e1 + TOKENIZER_TEXTS_EXTRA
        rows = verify_tokenizers(xmls, hf, L, texts, gated={lb for lb, _ in e1})
        meta["tokenizers"] = rows
        meta["tokenizers_note"] = ("qwen: input_ids/attention_mask [1,L] i64・pad=151643・特殊 token なし。"
                                   "t5: 可変長 (pad 無し)・eos(=1) は C++ で末尾に付加・T5 weights は全 1.0。"
                                   + EMPTY_PROMPT_RULE)
        if not tokenizer_gate_ok(rows):
            print("[A1] FAIL: E-1 prompt 16 本のうち tokenizer 不一致あり")
            exit_code = 1
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)

    if args.verify or args.bench:
        if not os.path.isfile(xml):
            print(f"[A1] IR が無い: {xml} (--convert を先に)")
            sys.exit(1)
        cases = verify_cases(L, q_ids_real)

    if args.verify:
        print(f"\n{'=' * 60}\n保存済み FP16 IR を CPU(f32 hint) で torch と突合\n{'=' * 60}")
        meta["verify_fp16_ir_cpu_f32"] = verify_against_torch(compile_cpu_f32(xml), get_te(), cases, "fp16-ir/CPU-f32")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)

    if args.bench:
        print(f"\n{'=' * 60}\nデバイス別ベンチ (N={args.n}・warmup {args.warmup})\n{'=' * 60}")
        import openvino as ov
        avail = ov.Core().available_devices
        print(f"  available_devices: {avail}")
        res = [bench_device(xml, d, get_te(), L, args.n, args.warmup, cases) for d in args.devices]
        out = {"seq_len": L, "available_devices": avail, "results": res}
        bp = os.path.join(args.out_dir, f"bench_L{L}.json")
        with open(bp, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"  -> {bp}")

    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
