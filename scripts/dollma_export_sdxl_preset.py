"""
dollma_export_sdxl_preset.py
=============================
タスク 2-6d: アニメ特化 SDXL checkpoint を 1 preset ぶん DL / 変換し、
自作 C++ ランタイムがそのまま読める形 (`models/presets/<preset>/`) に落とす。

出力:
  unet_weights.safetensors      UNet state_dict 全キー fp16 (dollma_dump_unet_golden.py と同流儀)
  vae_weights.safetensors       VAE state_dict の decoder.*/post_quant_conv.* のみ fp16
                                 (dollma_dump_vae_golden.py と同流儀)
  text-encoder-l/model_ov.xml(.bin)  CLIP-L text encoder OV IR (dollma_convert_sdxl_text_encoders.py 流用)
  text-encoder-g/model_ov.xml(.bin)  CLIP-bigG text encoder OV IR (同上)
  preset.json                   由来・ライセンス・prediction_type・推奨 prompt prefix 等のメタ

使い方:
  python scripts/dollma_export_sdxl_preset.py --preset animagine-xl-4
  python scripts/dollma_export_sdxl_preset.py --preset noobai-xl \
      --source Laxhar/noobai-XL-1.1 --hf-file noobaiXLNAIXL_epsilonPred11.safetensors
      # 注意: 単一 .safetensors 経路では diffusers が prediction_type を base 既定
      # (epsilon) から生成するため実際の checkpoint 値を検査できない。ファイル名に
      # v_pred/ztsnr テンソルが無いことを raw キー走査で確認したうえで、
      # 明示的に --assume-epsilon を付けない限り SystemExit(2) で停止する。

制約 (自作 C++ ランタイム前提):
  - prediction_type は epsilon のみ対応 (v-pred 非対応)。
  - VAE scaling_factor は 0.13025 のみ対応 (src/infer/diffusion.cu の kScalingFactor
    がハードコードのため、これと異なる checkpoint は自作ランタイムで読めない)。
  - tokenizer (vocab.json/merges.txt) は base SDXL と同一である必要がある (C++ 側は
    base の OV tokenizer をそのまま使い回す設計のため)。
"""

import argparse
import hashlib
import json
import os
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dollma_convert_sdxl_text_encoders import convert_text_encoders  # noqa: E402

BASE_ID = "stabilityai/stable-diffusion-xl-base-1.0"
EXPECTED_VAE_SCALING_FACTOR = 0.13025  # src/infer/diffusion.cu:41 kScalingFactor と一致必須


# ============================================================
# preset 既定値 (source は HF repo id、hf_file を指定すると単一ファイル取得)
# ============================================================
PRESETS = {
    "animagine-xl-4": {
        "source": "cagliostrolab/animagine-xl-4.0",
        "hf_file": None,
        "license": "CreativeML Open RAIL++-M",
        "prompt_prefix": "masterpiece, high score, great score, absurdres",
        "negative_prefix": "lowres, bad anatomy, bad hands, text, error, "
                            "missing fingers, extra digit, fewer digits, cropped, "
                            "worst quality, low quality, low score, bad score, average score",
    },
    "illustrious-xl": {
        "source": "OnomaAIResearch/Illustrious-xl-early-release-v0",
        "hf_file": None,
        "license": "Fair AI Public License 1.0-SD",
        "prompt_prefix": "masterpiece, best quality",
        "negative_prefix": "lowres, bad anatomy, bad hands, text, error, "
                            "missing fingers, extra digit, fewer digits, cropped, "
                            "worst quality, low quality",
    },
    "noobai-xl": {
        # v-pred 版 (名前に vpred/v-pred を含む) は使わない。eps 版のみ。
        "source": "Laxhar/noobai-XL-1.1",
        "hf_file": None,
        "license": "Fair AI Public License 1.0-SD",
        "prompt_prefix": "masterpiece, best quality, newest",
        "negative_prefix": "lowres, bad anatomy, bad hands, text, error, "
                            "missing fingers, extra digit, fewer digits, cropped, "
                            "worst quality, low quality, very displeasing",
    },
}


def sha256_file(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS.keys()))
    ap.add_argument("--source", default=None,
                    help="HF repo id、または .safetensors ファイルパス (省略時は preset 既定)")
    ap.add_argument("--hf-file", default=None,
                    help="--source が HF repo かつ単一ファイル repo のとき、repo 内のファイル名")
    ap.add_argument("--out", default=None,
                    help="出力先ディレクトリ (既定 models/presets/<preset>)")
    ap.add_argument("--assume-epsilon", action="store_true",
                    help="単一 .safetensors 経路で prediction_type を検査できないとき、"
                         "raw キーに v_pred/ztsnr が無いことを確認した上で明示的に epsilon と仮定する")
    return ap.parse_args()


# ============================================================
# 単一ファイル checkpoint の v-pred / ztsnr raw キー検査
# ============================================================
def check_single_file_not_vpred(path: str):
    """diffusers の from_single_file は scheduler を base config から生成するため
    pipe.scheduler.config.prediction_type は checkpoint 実体を反映しない (常に base 既定)。
    そのため raw safetensors キーを直接走査し、v_pred/ztsnr を示唆するキーが無いか調べる。
    確証は得られない (v-pred モデルが疑わしいキーを持たない場合もある) ので、
    呼び出し側は --assume-epsilon を明示しない限り停止する。
    """
    suspicious = []
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            lk = k.lower()
            if "v_pred" in lk or "vpred" in lk or "ztsnr" in lk:
                suspicious.append(k)
    return suspicious


def resolve_snapshot_dir(source: str) -> Path:
    """diffusers 用のサブフォルダだけを取得し、実際に読んだ snapshot の commit ディレクトリを返す。

    revision は huggingface_hub.snapshot_download が返すローカルパスの basename
    (= 解決済み commit hash) を正典とする。model_info().sha は「repo の現在の HEAD」
    であり実際にダウンロードした snapshot と一致する保証がないため使わない。
    """
    from huggingface_hub import snapshot_download
    allow_patterns = [
        "model_index.json",
        "scheduler/*", "text_encoder/*", "text_encoder_2/*",
        "tokenizer/*", "tokenizer_2/*", "unet/*", "vae/*",
    ]
    local_dir = snapshot_download(repo_id=source, allow_patterns=allow_patterns)
    return Path(local_dir)


def load_pipeline(source: str, hf_file: str | None, assume_epsilon: bool):
    """source が .safetensors 単体ファイルなら from_single_file、
    HF repo id なら from_pretrained (hf_file 指定時は単一ファイル DL 後 from_single_file)。

    戻り値: (pipe, revision, sha256, single_file_path または None)
    single_file_path が None でないとき、prediction_type の pipe.scheduler.config は
    信用できない (呼び出し側で別途 raw キー検査すること)。
    """
    from diffusers import StableDiffusionXLPipeline

    revision = None
    sha256 = None
    single_file_path = None

    is_local_file = source.endswith(".safetensors") and os.path.isfile(source)

    if is_local_file:
        if hf_file is not None:
            print(f"  警告: ローカルファイル指定時は --hf-file は無視される (hf_file={hf_file!r})")
        print(f"  ローカル単一ファイルからロード: {source}")
        single_file_path = source
        sha256 = sha256_file(Path(source))
        pipe = StableDiffusionXLPipeline.from_single_file(source, torch_dtype=torch.float32)
    elif hf_file is not None:
        from huggingface_hub import hf_hub_download
        print(f"  HF 単一ファイル DL: {source} / {hf_file}")
        local_path = hf_hub_download(repo_id=source, filename=hf_file)
        single_file_path = local_path
        sha256 = sha256_file(Path(local_path))
        # キャッシュ配置は .../snapshots/<commit_hash>/<filename> なので
        # 親ディレクトリ名が実際に読んだ commit hash。
        revision = Path(local_path).parent.name
        pipe = StableDiffusionXLPipeline.from_single_file(local_path, torch_dtype=torch.float32)
    else:
        print(f"  HF repo からロード (from_pretrained, サブフォルダのみ snapshot_download): {source}")
        snapshot_dir = resolve_snapshot_dir(source)
        revision = snapshot_dir.name
        pipe = StableDiffusionXLPipeline.from_pretrained(str(snapshot_dir), torch_dtype=torch.float32)

    if single_file_path is not None:
        suspicious = check_single_file_not_vpred(single_file_path)
        if suspicious:
            print(f"エラー: 単一ファイル checkpoint に v_pred/ztsnr を示唆するキーが見つかった: "
                  f"{suspicious[:10]}。v-pred checkpoint の可能性が高いため中止する。")
            raise SystemExit(1)
        if not assume_epsilon:
            print("エラー: 単一 .safetensors 経路では diffusers が prediction_type を "
                  "base 既定 (epsilon) から生成するため、checkpoint 実体の prediction_type を "
                  "確定できない。raw キーに v_pred/ztsnr は無かったが、確証は得られないため "
                  "--assume-epsilon を明示しない限り停止する。")
            raise SystemExit(2)
        print("  raw キー検査: v_pred/ztsnr 示唆キーなし + --assume-epsilon 指定 -> epsilon とみなす")

    return pipe, revision, sha256, single_file_path


def dump_unet_weights(unet):
    full_sd = unet.state_dict()
    weights = OrderedDict()
    for k, v in full_sd.items():
        weights[k] = v.to(torch.float16).contiguous()
    return weights


def dump_vae_weights(vae):
    full_sd = vae.state_dict()
    weights = OrderedDict()
    for k, v in full_sd.items():
        if k.startswith("decoder.") or k.startswith("post_quant_conv."):
            weights[k] = v.to(torch.float16).contiguous()
    return weights


def base_unet_key_shape_signature():
    """base SDXL UNet のキー名+shape 署名 (meta device で 0 コスト化・実重み DL/確保なし)。"""
    from diffusers import UNet2DConditionModel
    config = UNet2DConditionModel.load_config(BASE_ID, subfolder="unet")
    with torch.device("meta"):
        model = UNet2DConditionModel.from_config(config)
    sd = model.state_dict()
    return {k: tuple(v.shape) for k, v in sd.items()}


def verify_key_shape_match(preset_weights: dict, base_sig: dict):
    preset_sig = {k: tuple(v.shape) for k, v in preset_weights.items()}
    base_keys = set(base_sig.keys())
    preset_keys = set(preset_sig.keys())
    missing = base_keys - preset_keys
    extra = preset_keys - base_keys
    shape_mismatch = [k for k in (base_keys & preset_keys) if base_sig[k] != preset_sig[k]]
    ok = (len(missing) == 0 and len(extra) == 0 and len(shape_mismatch) == 0)
    return ok, missing, extra, shape_mismatch


def base_tokenizer_hashes():
    """base SDXL tokenizer/tokenizer_2 の vocab.json/merges.txt sha256。"""
    from huggingface_hub import hf_hub_download
    out = {}
    for sub in ("tokenizer", "tokenizer_2"):
        for fn in ("vocab.json", "merges.txt"):
            p = hf_hub_download(repo_id=BASE_ID, filename=f"{sub}/{fn}")
            out[f"{sub}/{fn}"] = sha256_file(Path(p))
    return out


def preset_tokenizer_hashes(pipe):
    """preset の tokenizer/tokenizer_2 の save_pretrained 経由で vocab.json/merges.txt を取り出し sha256。"""
    import tempfile
    out = {}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pipe.tokenizer.save_pretrained(td / "tokenizer")
        pipe.tokenizer_2.save_pretrained(td / "tokenizer_2")
        for sub in ("tokenizer", "tokenizer_2"):
            for fn in ("vocab.json", "merges.txt"):
                p = td / sub / fn
                if p.is_file():
                    out[f"{sub}/{fn}"] = sha256_file(p)
                else:
                    out[f"{sub}/{fn}"] = None
    return out


def main():
    args = parse_args()
    preset_cfg = PRESETS[args.preset]
    source = args.source or preset_cfg["source"]
    hf_file = args.hf_file if args.hf_file is not None else preset_cfg["hf_file"]

    out_dir = Path(args.out) if args.out else Path("models") / "presets" / args.preset
    out_l = out_dir / "text-encoder-l"
    out_g = out_dir / "text-encoder-g"
    unet_path = out_dir / "unet_weights.safetensors"
    vae_path = out_dir / "vae_weights.safetensors"
    meta_path = out_dir / "preset.json"

    print(f"{'='*70}\npreset={args.preset}  source={source}  hf_file={hf_file}\n{'='*70}")

    t0 = datetime.now(timezone.utc)
    pipe, revision, sha256, single_file_path = load_pipeline(source, hf_file, args.assume_epsilon)

    # prediction_type 検査。v-pred の checkpoint は変換しない (自作 scheduler は eps 前提)。
    # 単一ファイル経路では load_pipeline 内で既に検査済み (raw キー + --assume-epsilon)。
    if single_file_path is None:
        pred_type = pipe.scheduler.config.prediction_type
        print(f"  prediction_type = {pred_type}")
        if pred_type != "epsilon":
            print(f"エラー: prediction_type={pred_type} は epsilon ではない。"
                  f" v-pred checkpoint は非対応のため変換を中止する。")
            raise SystemExit(1)
    else:
        pred_type = "epsilon"  # load_pipeline の raw キー検査 + --assume-epsilon で確定済み

    # VAE scaling_factor 検査。自作 C++ (src/infer/diffusion.cu kScalingFactor) がハードコードのため
    # これと異なる checkpoint は自作ランタイムで正しく decode できない。
    vae_scaling_factor = float(pipe.vae.config.scaling_factor)
    print(f"  vae_scaling_factor = {vae_scaling_factor}")
    if abs(vae_scaling_factor - EXPECTED_VAE_SCALING_FACTOR) > 1e-8:
        print(f"エラー: vae_scaling_factor={vae_scaling_factor} != "
              f"{EXPECTED_VAE_SCALING_FACTOR} (src/infer/diffusion.cu:41 kScalingFactor ハードコード)。"
              f" 自作ランタイムで正しく decode できないため中止する。")
        raise SystemExit(1)

    # tokenizer 一致検査。C++ 側は base の OV tokenizer をそのまま使い回すため、
    # preset の tokenizer 語彙が base と異なると誤トークナイズになる。
    print("\ntokenizer (vocab.json/merges.txt) の base 一致を検査中...")
    base_tok_hashes = base_tokenizer_hashes()
    preset_tok_hashes = preset_tokenizer_hashes(pipe)
    tok_mismatch = {k: (base_tok_hashes.get(k), preset_tok_hashes.get(k))
                    for k in base_tok_hashes
                    if base_tok_hashes.get(k) != preset_tok_hashes.get(k)}
    tok_match = (len(tok_mismatch) == 0)
    if not tok_match:
        print(f"エラー: tokenizer が base と一致しない: {tok_mismatch}")
        print("  C++ は base の OV tokenizer を使い回すため、preset 独自語彙は非対応。中止する。")
        raise SystemExit(1)
    print("  -> tokenizer 一致 OK (vocab.json/merges.txt sha256 が base と同一)")

    # --- UNet をメモリ上に fp16 化 (base とのキー名+shape 照合を保存前に行う) ---
    print("\nUNet 重みを fp16 化中 (保存前にキー名+shape 照合を実施)...")
    unet_weights = dump_unet_weights(pipe.unet)
    base_sig = base_unet_key_shape_signature()
    key_match, missing, extra, shape_mismatch = verify_key_shape_match(unet_weights, base_sig)
    print(f"  base keys={len(base_sig)}  preset keys={len(unet_weights)}  "
          f"missing={len(missing)}  extra={len(extra)}  shape_mismatch={len(shape_mismatch)}")
    if not key_match:
        print("エラー: UNet キー名/shape が base と一致しない。ローダーが読めない可能性が高いため、"
              "何も書き込まずに中止する。")
        if missing:
            print(f"  missing (先頭10): {sorted(missing)[:10]}")
        if extra:
            print(f"  extra   (先頭10): {sorted(extra)[:10]}")
        if shape_mismatch:
            print(f"  shape_mismatch (先頭10): {sorted(shape_mismatch)[:10]}")
        raise SystemExit(1)
    print("  -> キー名+shape 一致 OK")

    # --- ここまでの検査を全通過したので、初めてディスクに書き込む ---
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nUNet 重み保存中...")
    save_file(unet_weights, unet_path)
    unet_mb = unet_path.stat().st_size / 1e6
    print(f"  -> {unet_path} ({unet_mb:.1f} MB, {len(unet_weights)} keys)")

    print("\nVAE (decoder) 重み保存中...")
    vae_weights = dump_vae_weights(pipe.vae)
    save_file(vae_weights, vae_path)
    vae_mb = vae_path.stat().st_size / 1e6
    print(f"  -> {vae_path} ({vae_mb:.1f} MB, {len(vae_weights)} keys)")

    print("\nText encoder を OV IR に変換中 (数値突合あり)...")
    _, _, te_errs = convert_text_encoders(
        pipe.text_encoder, pipe.text_encoder_2, out_l, out_g,
        verify=True, tok_l=pipe.tokenizer, tok_g=pipe.tokenizer_2)
    print(f"  CLIP-L  penultimate err = {te_errs['clip_l_penultimate']:.2e}")
    print(f"  bigG    penultimate err = {te_errs['bigg_penultimate']:.2e}")
    print(f"  bigG    pooled      err = {te_errs['bigg_pooled']:.2e}")

    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()

    # --- メタ ---
    meta = {
        "name": args.preset,
        "source": source,
        "hf_file": hf_file,
        "revision": revision,
        "sha256": sha256,
        "license": preset_cfg["license"],
        "prediction_type": pred_type,
        "vae_scaling_factor": vae_scaling_factor,
        "prompt_prefix": preset_cfg["prompt_prefix"],
        "negative_prefix": preset_cfg["negative_prefix"],
        "converted_at": t0.isoformat(),
        "unet_bytes": unet_path.stat().st_size,
        "unet_sha256": sha256_file(unet_path),
        "vae_bytes": vae_path.stat().st_size,
        "vae_sha256": sha256_file(vae_path),
        "text_encoder_l_err": te_errs["clip_l_penultimate"],
        "text_encoder_g_penultimate_err": te_errs["bigg_penultimate"],
        "text_encoder_g_pooled_err": te_errs["bigg_pooled"],
        "key_shape_match_vs_base": key_match,
        "tokenizer_match_vs_base": tok_match,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n-> {meta_path}")

    print(f"\n{'='*70}\n完了: preset={args.preset}  所要 {elapsed:.1f}s\n{'='*70}")
    print(f"  unet: {unet_mb:.1f} MB  sha256={meta['unet_sha256'][:16]}...")
    print(f"  vae : {vae_mb:.1f} MB  sha256={meta['vae_sha256'][:16]}...")
    print(f"  te-l: {out_l / 'model_ov.bin'}")
    print(f"  te-g: {out_g / 'model_ov.bin'}")


if __name__ == "__main__":
    main()
