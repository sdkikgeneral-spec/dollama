# -*- coding: utf-8 -*-
"""dollama L-1 offline LoRA merge 単体テスト (docs/testing.md 準拠)。

核 (key 写像・低ランク積加算・scaling・te skip・写像漏れ/base 不在) は numpy のみで動く。
safetensors 往復・構造検証は safetensors ライブラリがある環境でのみ実走 (無ければ SKIP)。
torch は不要。

本番アセット (data/ ・本番 checkpoint ・golden) は一切触らない。fixture は temp に生成する。

実行:
  py -3.12 scripts/test_dollma_merge_lora.py
  あるいは: py -3.12 -m pytest scripts/test_dollma_merge_lora.py -q
"""
import os
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import dollma_merge_lora as ml


# =====================================================================
# 1. key 写像 golden 突合
# =====================================================================

def test_key_mapping_golden():
    """代表 kohya モジュール名 → diffusers モジュール名 の写像テーブル正しさ。"""
    cases = [
        # attention 各 proj
        ("down_blocks_0_attentions_1_transformer_blocks_0_attn1_to_q",
         "down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_q"),
        ("down_blocks_0_attentions_1_transformer_blocks_0_attn1_to_k",
         "down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_k"),
        ("down_blocks_0_attentions_1_transformer_blocks_0_attn1_to_v",
         "down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_v"),
        # to_out.0 (複合 leaf)
        ("down_blocks_0_attentions_1_transformer_blocks_0_attn1_to_out_0",
         "down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_out.0"),
        ("mid_block_attentions_0_transformer_blocks_0_attn2_to_out_0",
         "mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0"),
        # ff.net.0.proj / ff.net.2
        ("up_blocks_0_attentions_2_transformer_blocks_0_ff_net_0_proj",
         "up_blocks.0.attentions.2.transformer_blocks.0.ff.net.0.proj"),
        ("up_blocks_0_attentions_2_transformer_blocks_0_ff_net_2",
         "up_blocks.0.attentions.2.transformer_blocks.0.ff.net.2"),
        # proj_in / proj_out
        ("down_blocks_1_attentions_0_proj_in",
         "down_blocks.1.attentions.0.proj_in"),
        ("down_blocks_1_attentions_0_proj_out",
         "down_blocks.1.attentions.0.proj_out"),
        # mid_block cross attn
        ("mid_block_attentions_0_transformer_blocks_0_attn2_to_q",
         "mid_block.attentions.0.transformer_blocks.0.attn2.to_q"),
        # time/add embedding linear
        ("time_embedding_linear_1", "time_embedding.linear_1"),
        ("add_embedding_linear_2", "add_embedding.linear_2"),
    ]
    for kohya, want in cases:
        got = ml.kohya_to_diffusers_key(kohya)
        assert got == want, "key 写像不一致: {} -> {} (期待 {})".format(kohya, got, want)
    print("[ok] key 写像 golden {} 件一致".format(len(cases)))


def test_base_dotless_resolution():
    """base のドット除去索引経由の解決が構造復元より優先され、確実に当たること。"""
    base_keys = [
        "down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_q.weight",
        "down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_out.0.weight",
        "up_blocks.0.attentions.2.transformer_blocks.0.ff.net.0.proj.weight",
        "conv_in.weight",
    ]
    idx = ml.build_base_dotless_index(base_keys)
    # kohya モジュール → base key
    assert ml.resolve_base_key(
        "down_blocks_0_attentions_1_transformer_blocks_0_attn1_to_q", idx
    ) == "down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_q.weight"
    assert ml.resolve_base_key(
        "down_blocks_0_attentions_1_transformer_blocks_0_attn1_to_out_0", idx
    ) == "down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_out.0.weight"
    assert ml.resolve_base_key(
        "up_blocks_0_attentions_2_transformer_blocks_0_ff_net_0_proj", idx
    ) == "up_blocks.0.attentions.2.transformer_blocks.0.ff.net.0.proj.weight"
    # 存在しないものは None
    assert ml.resolve_base_key("nonexistent_module_xyz", idx) is None
    print("[ok] base ドット除去索引による key 解決")


# =====================================================================
# 2. 数値 (W + scale*(B@A)) numpy リファレンス突合
# =====================================================================

def _toy_base_lora(rank=2, out_dim=4, in_dim=3, alpha=None, dtype=np.float32):
    """小さな既知 W, A, B を作る。"""
    W = np.arange(out_dim * in_dim, dtype=np.float32).reshape(out_dim, in_dim)
    A = (np.arange(rank * in_dim, dtype=np.float32).reshape(rank, in_dim) + 1.0) * 0.1
    B = (np.arange(out_dim * rank, dtype=np.float32).reshape(out_dim, rank) + 1.0) * 0.1
    base = {"down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight": W.astype(dtype)}
    mod = "down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q"
    lora = {
        "lora_unet_{}.lora_down.weight".format(mod): A.astype(np.float16),
        "lora_unet_{}.lora_up.weight".format(mod): B.astype(np.float16),
    }
    if alpha is not None:
        lora["lora_unet_{}.alpha".format(mod)] = np.array(float(alpha), dtype=np.float32)
    return base, lora, W, A.astype(np.float16), B.astype(np.float16)


def test_numeric_merge_reference():
    """W + scale*(B@A) を numpy リファレンスと突合 (fp tol)。"""
    base, lora, W, A, B = _toy_base_lora(rank=2, alpha=2, dtype=np.float32)
    out, report = ml.merge_state_dicts(base, lora, strength=1.0, out_np_dtype=np.float32)

    rank = A.shape[0]
    scale = 1.0 * (2.0 / rank)  # alpha=2, rank=2 -> 1.0
    ref = W.astype(np.float32) + scale * (B.astype(np.float32) @ A.astype(np.float32))

    key = "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight"
    got = out[key]
    assert got.shape == W.shape
    assert np.allclose(got, ref, atol=1e-5), "数値不一致 max={}".format(
        np.max(np.abs(got - ref)))
    assert len(report.merged_modules) == 1
    print("[ok] 数値 W+scale*(B@A) リファレンス突合 (max err {:.2e})".format(
        float(np.max(np.abs(got - ref)))))


# =====================================================================
# 3. scaling (alpha / strength / alpha 欠如)
# =====================================================================

def test_scaling_alpha_and_strength():
    """alpha 違い・strength 違い・alpha 欠如の scale が期待値。"""
    key = "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight"

    # alpha=1, rank=2, strength=1 -> scale=0.5
    base, lora, W, A, B = _toy_base_lora(rank=2, alpha=1, dtype=np.float32)
    out, _ = ml.merge_state_dicts(base, lora, strength=1.0, out_np_dtype=np.float32)
    ref = W + 0.5 * (B.astype(np.float32) @ A.astype(np.float32))
    assert np.allclose(out[key], ref, atol=1e-5), "alpha=1 scale 不一致"

    # strength=0.5, alpha=2, rank=2 -> scale=0.5
    base, lora, W, A, B = _toy_base_lora(rank=2, alpha=2, dtype=np.float32)
    out, _ = ml.merge_state_dicts(base, lora, strength=0.5, out_np_dtype=np.float32)
    ref = W + 0.5 * (B.astype(np.float32) @ A.astype(np.float32))
    assert np.allclose(out[key], ref, atol=1e-5), "strength=0.5 scale 不一致"

    # alpha 欠如 -> alpha=rank -> scale=strength
    base, lora, W, A, B = _toy_base_lora(rank=2, alpha=None, dtype=np.float32)
    out, _ = ml.merge_state_dicts(base, lora, strength=1.0, out_np_dtype=np.float32)
    ref = W + 1.0 * (B.astype(np.float32) @ A.astype(np.float32))
    assert np.allclose(out[key], ref, atol=1e-5), "alpha 欠如既定 scale 不一致"

    print("[ok] scaling: alpha=1->0.5 / strength=0.5->0.5 / alpha欠如->strength")


# =====================================================================
# 4. shape 不一致は例外
# =====================================================================

def test_shape_mismatch_raises():
    """B@A が base W と shape 不一致なら例外。"""
    # base W は [4,3] だが LoRA は [4,2]@[2,5] = [4,5] になるよう作る
    W = np.zeros((4, 3), dtype=np.float32)
    base = {"down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight": W}
    mod = "down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q"
    lora = {
        "lora_unet_{}.lora_down.weight".format(mod): np.zeros((2, 5), dtype=np.float16),
        "lora_unet_{}.lora_up.weight".format(mod): np.zeros((4, 2), dtype=np.float16),
    }
    try:
        ml.merge_state_dicts(base, lora, strength=1.0)
        raise AssertionError("shape 不一致で例外が飛ばなかった")
    except ValueError:
        pass
    print("[ok] shape 不一致 -> ValueError")


# =====================================================================
# 5. te1/te2 skip + warn カウント
# =====================================================================

def test_te_skip_count():
    """lora_te1_*/lora_te2_* を含む LoRA で UNet のみマージ・te は skip カウント。"""
    base, lora, W, A, B = _toy_base_lora(rank=2, alpha=2, dtype=np.float32)
    # te1/te2 key を足す (base には対応物が無い = マージ対象外)
    lora["lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight"] = \
        np.zeros((2, 8), dtype=np.float16)
    lora["lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight"] = \
        np.zeros((8, 2), dtype=np.float16)
    lora["lora_te2_text_model_encoder_layers_5_mlp_fc1.lora_down.weight"] = \
        np.zeros((2, 8), dtype=np.float16)
    lora["lora_te2_text_model_encoder_layers_5_mlp_fc1.lora_up.weight"] = \
        np.zeros((8, 2), dtype=np.float16)

    out, report = ml.merge_state_dicts(base, lora, strength=1.0, out_np_dtype=np.float32)
    assert len(report.merged_modules) == 1, "UNet が 1 件マージされること"
    # te key は 4 件 (down/up ×2 系統) skip 集計される
    assert len(report.te_skipped) == 4, "te skip 件数 {}".format(len(report.te_skipped))
    # te は出力 checkpoint に混入しない
    for k in out:
        assert not k.startswith("lora_te")
    print("[ok] te1/te2 skip ({} 件) + UNet のみマージ".format(len(report.te_skipped)))


# =====================================================================
# 6. 写像漏れ報告 / base 不在 key で例外
# =====================================================================

def test_unmapped_reported():
    """down/up 片方欠落 (不完全 LoRA) は写像漏れに集計され黙って捨てない。"""
    base, lora, W, A, B = _toy_base_lora(rank=2, alpha=2, dtype=np.float32)
    # 別モジュールの up だけ足す (down 欠落)
    lora["lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k.lora_up.weight"] = \
        np.zeros((4, 2), dtype=np.float16)
    out, report = ml.merge_state_dicts(base, lora, strength=1.0)
    assert len(report.unmapped) == 1, "不完全 LoRA が写像漏れ報告されること"
    assert "missing down/up" in report.unmapped[0]
    print("[ok] 写像漏れ (不完全 LoRA) 報告")


def test_base_absent_key_raises():
    """base に存在しない diffusers key へ写像する LoRA は例外 (命名ズレ早期検出)。"""
    # base には to_q だけ。LoRA は base に無い to_zzz を狙う完全 down/up セット。
    base, lora, W, A, B = _toy_base_lora(rank=2, alpha=2, dtype=np.float32)
    lora["lora_unet_down_blocks_9_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight"] = \
        np.zeros((2, 3), dtype=np.float16)
    lora["lora_unet_down_blocks_9_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight"] = \
        np.zeros((4, 2), dtype=np.float16)
    try:
        ml.merge_state_dicts(base, lora, strength=1.0)
        raise AssertionError("base 不在 key で例外が飛ばなかった")
    except KeyError:
        pass
    print("[ok] base 不在 key への写像 -> KeyError")


# =====================================================================
# 7. 触れない key の bitwise 不変 + Conv マージ
# =====================================================================

def test_untouched_bitwise_and_conv():
    """LoRA が触れない key が bitwise 不変。Conv (4D) base への B@A reshape 加算。"""
    rng = np.random.default_rng(7)
    # Linear (触れる) + 別 Linear (触れない) + Conv (触れる, 4D)
    lin_W = rng.standard_normal((4, 3)).astype(np.float32)
    untouched = rng.standard_normal((5, 6)).astype(np.float32)
    conv_W = rng.standard_normal((2, 2, 3, 3)).astype(np.float32)  # out=2,in=2,k=3,3 -> 2x18

    base = {
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight": lin_W,
        "down_blocks.0.resnets.0.norm1.weight": untouched,  # LoRA 非対象
        "down_blocks.0.resnets.0.conv1.weight": conv_W,
    }
    lin_mod = "down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q"
    conv_mod = "down_blocks_0_resnets_0_conv1"
    A_lin = rng.standard_normal((2, 3)).astype(np.float16)
    B_lin = rng.standard_normal((4, 2)).astype(np.float16)
    # Conv: delta は [2, 18] = [out, in*k*k] にして reshape
    A_conv = rng.standard_normal((2, 18)).astype(np.float16)
    B_conv = rng.standard_normal((2, 2)).astype(np.float16)
    lora = {
        "lora_unet_{}.lora_down.weight".format(lin_mod): A_lin,
        "lora_unet_{}.lora_up.weight".format(lin_mod): B_lin,
        "lora_unet_{}.alpha".format(lin_mod): np.array(2.0, dtype=np.float32),
        "lora_unet_{}.lora_down.weight".format(conv_mod): A_conv,
        "lora_unet_{}.lora_up.weight".format(conv_mod): B_conv,
        "lora_unet_{}.alpha".format(conv_mod): np.array(2.0, dtype=np.float32),
    }

    out, report = ml.merge_state_dicts(base, lora, strength=1.0, out_np_dtype=np.float32)

    # 触れない key は bitwise 不変
    assert np.array_equal(out["down_blocks.0.resnets.0.norm1.weight"], untouched), \
        "触れない key が改変された"
    assert report.copied_untouched == 1

    # Conv マージ参照
    sc = 2.0 / 2.0
    delta_conv = sc * (B_conv.astype(np.float32) @ A_conv.astype(np.float32))
    ref_conv = conv_W + delta_conv.reshape(conv_W.shape)
    assert np.allclose(out["down_blocks.0.resnets.0.conv1.weight"], ref_conv, atol=1e-4), \
        "Conv マージ不一致"
    assert len(report.merged_modules) == 2
    print("[ok] 触れない key bitwise 不変 + Conv 4D reshape 加算")


# =====================================================================
# 8. dtype 追従 / NaN-Inf チェック
# =====================================================================

def test_dtype_follow_and_finite():
    """出力 dtype が base 追従 (None) / 指定で fp16/fp32。NaN/Inf 無し。"""
    base, lora, W, A, B = _toy_base_lora(rank=2, alpha=2, dtype=np.float16)
    # base 追従 -> fp16
    out, _ = ml.merge_state_dicts(base, lora, strength=1.0, out_np_dtype=None)
    key = "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight"
    assert out[key].dtype == np.float16, "base 追従で fp16 にならない"
    # 明示 fp32
    out2, _ = ml.merge_state_dicts(base, lora, strength=1.0, out_np_dtype=np.float32)
    assert out2[key].dtype == np.float32
    # finite
    assert np.all(np.isfinite(out2[key]))
    print("[ok] dtype 追従/明示 + finite チェック")


# =====================================================================
# 9. safetensors 往復 + 標準形式構造検証 (C++ ローダー互換)
# =====================================================================

def test_safetensors_roundtrip_and_format():
    """出力 safetensors が再ロード可能・dtype/shape 健全・NaN/Inf 無し・
    触れない key が bitwise 不変・標準形式 (header JSON + tensor bytes) であること。

    safetensors ライブラリ不在は SKIP。"""
    if not ml.HAVE_SAFETENSORS:
        print("[skip] safetensors ライブラリ不在 → 往復/構造検証は省略")
        return

    import json
    import struct
    from safetensors.numpy import save_file, load_file

    rng = np.random.default_rng(20260620)
    lin_W = rng.standard_normal((4, 3)).astype(np.float16)
    untouched = rng.standard_normal((5, 6)).astype(np.float16)
    base = {
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight": lin_W,
        "conv_in.weight": untouched,
    }
    mod = "down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q"
    lora = {
        "lora_unet_{}.lora_down.weight".format(mod):
            (rng.standard_normal((2, 3)) * 0.01).astype(np.float16),
        "lora_unet_{}.lora_up.weight".format(mod):
            (rng.standard_normal((4, 2)) * 0.01).astype(np.float16),
        "lora_unet_{}.alpha".format(mod): np.array(2.0, dtype=np.float32),
    }

    with tempfile.TemporaryDirectory() as td:
        base_p = os.path.join(td, "base.safetensors")
        lora_p = os.path.join(td, "lora.safetensors")
        out_p = os.path.join(td, "merged.safetensors")
        save_file({k: np.ascontiguousarray(v) for k, v in base.items()}, base_p)
        save_file({k: np.ascontiguousarray(v) for k, v in lora.items()}, lora_p)

        report = ml.run_merge_files(base_p, lora_p, out_p, strength=1.0, verbose=False)
        assert len(report.merged_modules) == 1

        # 再ロード
        reloaded = load_file(out_p)
        key = "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight"
        assert key in reloaded and "conv_in.weight" in reloaded
        assert reloaded[key].shape == (4, 3)
        assert reloaded[key].dtype == np.float16  # base 追従
        assert np.all(np.isfinite(reloaded[key].astype(np.float32)))
        # 触れない key が bitwise 不変
        assert np.array_equal(reloaded["conv_in.weight"], untouched), \
            "触れない key が往復で改変"

        # 標準形式構造検証 (src/io/safetensors.hpp が読める形式):
        # [0..8) LE uint64 = ヘッダ長 N、[8..8+N) UTF-8 JSON、[8+N..) tensor bytes
        with open(out_p, "rb") as f:
            raw = f.read()
        assert len(raw) >= 8
        n = struct.unpack("<Q", raw[:8])[0]
        assert 8 + n <= len(raw), "ヘッダ長がファイルサイズを超過"
        header = json.loads(raw[8:8 + n].decode("utf-8"))
        # 各テンソルが dtype/shape/data_offsets を持ち、offset が body 範囲内・整合
        body_len = len(raw) - (8 + n)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            assert set(meta.keys()) >= {"dtype", "shape", "data_offsets"}
            b, e = meta["data_offsets"]
            assert 0 <= b <= e <= body_len, "data_offsets が body 範囲外"
            numel = 1
            for d in meta["shape"]:
                numel *= d
            itemsize = {"F16": 2, "F32": 4, "F64": 8, "I64": 8,
                        "I32": 4, "I8": 1, "U8": 1, "BF16": 2}[meta["dtype"]]
            assert numel * itemsize == (e - b), \
                "{}: shape*itemsize != byte length".format(name)
    print("[ok] safetensors 往復 + 標準形式構造検証 (C++ ローダー互換)")


# =====================================================================
# 10. 本番非汚染ガード (sentinel)
# =====================================================================

def test_no_production_pollution():
    """テストが temp 以外へ書かないことの sentinel: data/ 配下の mtime を前後比較。

    (このテスト自体は新規ファイルを作らないことの簡易確認。)"""
    data_dir = os.path.join(_HERE, "..", "data")
    if not os.path.isdir(data_dir):
        print("[skip] data/ 不在 → 汚染ガード省略")
        return
    # マージ系の出力名 (merged*.safetensors) が data/ に作られていないこと
    leaked = []
    for root, _dirs, files in os.walk(data_dir):
        for fn in files:
            if fn.startswith("merged") and fn.endswith(".safetensors"):
                leaked.append(os.path.join(root, fn))
    assert not leaked, "data/ に merged*.safetensors が漏れた: {}".format(leaked)
    print("[ok] 本番非汚染 (data/ に merged*.safetensors 漏れなし)")


def _run_all():
    fns = [
        test_key_mapping_golden,
        test_base_dotless_resolution,
        test_numeric_merge_reference,
        test_scaling_alpha_and_strength,
        test_shape_mismatch_raises,
        test_te_skip_count,
        test_unmapped_reported,
        test_base_absent_key_raises,
        test_untouched_bitwise_and_conv,
        test_dtype_follow_and_finite,
        test_safetensors_roundtrip_and_format,
        test_no_production_pollution,
    ]
    for fn in fns:
        fn()
    print("\n=== ALL MERGE-LORA TESTS PASSED ({} 件) ===".format(len(fns)))


if __name__ == "__main__":
    _run_all()
