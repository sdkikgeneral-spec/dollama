# -*- coding: utf-8 -*-
"""dollma_collect_rollouts.py — Phase 4 施策 F (品質 FB ループ) / F-0a:
品質フィードバック閉路を 1 周回す **rollout 収集ハーネス**。

閉路 (1 rollout):
  1. LM (BitNetDense) が input_text から danbooru タグプロンプトを greedy 生成。
  2. 既存 txt2img HTTP サーバ (`dollama --http`) に投げて SDXL 画像生成。
  3. ScorerNet (OV・B-3c で検証済 model_ov_fp32.xml) で採点 → logits [1,9]。
  4. dollma_reward.reward_from_scorer で anatomy 8 軸 → スカラ報酬。
  5. JSONL に 1 行追記: {input_text, prompt, seed, axes:[8], quality:null, reward}。

**実走は研究機 (RTX5080 + OpenVINO + 実 SDXL 重み + ScorerNet IR + BitNet 重み) でのみ。**
gpu-benchmarker が 50-100 rollout を回し、収集後に「プロンプト間で reward がばらつくか/
best と worst に学習可能なギャップがあるか」(= anatomy-only が学習信号になるか) を測る。
**本ハーネスはその計装まで** (JSONL に reward と axes を残す)。fine-tune 本体は F-0b。

開発機 (OV/torch/重み不在) では実走資産が無いので **明示 [SKIP] で早期 return** する
(NotImplementedError は投げない)。純ヘルパ (sigmoid / logits→軸 / 行スキーマ / 入力プール
サンプリング) は OV/SDXL/torch なしで単体テストできる。

実走に必要な env / 資産 (研究機・gpu-benchmarker 向け):
  - models/scorer-net/model_ov_fp32.xml / .bin  (dollma_convert_scorer.py で生成済)
  - data/bitnet/bitnet_dense_fp32.safetensors   (scripts/train_bitnet.py で訓練済・本線 #1)
  - data/bitnet/vocab.json                       (BitNet トークナイザ語彙)
  - env DOLLAMA_OV_TOKENIZERS_DLL                (本 txt2img(NPU) 有効化に必須・
                                                  未設定だと段2 golden に落ち prompt 無視)
  - build/src/dollama.exe                        (--http サーバ)
実走例 (研究機):
  set DOLLAMA_OV_TOKENIZERS_DLL=<path>/openvino_tokenizers.dll
  py -3.14 scripts/dollma_collect_rollouts.py --run --n 80 \
      --out data/rollouts/rollouts.jsonl --seed 20260630
"""

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from dollma_reward import reward_from_scorer  # noqa: E402  (純関数・依存なし)

# 実走資産パス (研究機で存在を確認・開発機では不在→[SKIP])。
# BitNet 重み/vocab は本線 #1 (dense hard CE 6ep) = data/bitnet/ 配下が実体。
# 同一性版 bitnet_dense_identity_fp32.safetensors も同ディレクトリにあるが既定にしない
# (アーキ差の可能性)。CLI (--bitnet-weights / --vocab) で上書き可 (定数是正 + CLI の二段構え)。
SCORER_IR = os.path.join(ROOT, "models", "scorer-net", "model_ov_fp32.xml")
BITNET_WEIGHTS = os.path.join(ROOT, "data", "bitnet", "bitnet_dense_fp32.safetensors")
VOCAB_PATH = os.path.join(ROOT, "data", "bitnet", "vocab.json")
DOLLAMA_EXE = os.path.join(ROOT, "build", "src", "dollama.exe")

# ScorerNet I/O 規約 (dollma_convert_scorer.py / golden_scorernet_meta.json と厳密一致)。
SCORER_IMG_SIZE = 512          # 入力 [1,3,512,512] NCHW [0,1]
SCORER_N_OUT = 9               # 出力 [1,9] logit: index0=quality / index1..8=anatomy 8 軸
N_AXIS = 8

# quality head は B-3b で凍結中 → 報酬・JSONL では None で記録する (前方互換)。
# True にすると index0 を sigmoid して quality として報酬へ混ぜる (将来 quality 凍結解除時)。
QUALITY_ENABLED = False


# ============================================================
# LM 入力プール (user 自然文 → LM が danbooru タグへ変換)
#   reward のばらつき (信号ゲート) は input_text の多様性が源。素直な単独被写体から
#   崩れやすい多人数/複雑ポーズ/手の本数まで散らし、anatomy の分散を意図的に作る。
# ============================================================
INPUT_TEXTS = [
    "a girl with long hair smiling",
    "a boy in a school uniform standing",
    "a girl with twintails looking at the viewer",
    "a single girl with a ponytail and glasses",
    "a girl wearing a kimono, upper body",
    "two girls hugging in a dynamic pose",
    "a girl holding two swords in an action pose",
    "a crowd of people with many hands",
    "a girl doing a handstand, full body",
    "a boy reaching toward the camera, foreshortening",
    "a girl playing the piano with both hands",
    "a girl holding a bow and arrow at full draw",
    "a girl with short hair and cat ears",
    "a boy with messy hair, portrait",
    "a girl in a dress, standing, wavy hair",
    "a girl with braided hair and blue eyes",
    "two boys fighting with crossed arms",
    "a girl juggling multiple objects",
    "a girl doing the splits, hands on the floor",
    "a girl with a side ponytail wearing a sweater",
]


# ============================================================
# 純ヘルパ (OV/SDXL/torch 非依存・単体テスト対象)
# ============================================================
def sigmoid(x):
    """数値安定な sigmoid (ScorerNet は logit を出すので確率へ変換するのに使う)。"""
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def axes_from_logits(logits9):
    """ScorerNet logits [9] (index0=quality / index1..8=anatomy) から anatomy 8 軸の
    sigmoid 確率 list[float] を取り出す。reward_from_scorer は確率を受けるので必ず通す。"""
    if len(logits9) != SCORER_N_OUT:
        raise ValueError(f"ScorerNet 出力長 {len(logits9)} != {SCORER_N_OUT}")
    return [sigmoid(float(logits9[1 + i])) for i in range(N_AXIS)]


def quality_from_logits(logits9, enabled=QUALITY_ENABLED):
    """quality (index0) を sigmoid 確率で返す。凍結中 (enabled=False) は None
    (JSONL に quality:null を残し、報酬へも混ぜない前方互換)。"""
    if not enabled:
        return None
    return sigmoid(float(logits9[0]))


def rollout_row(input_text, prompt, seed, axes, quality, reward):
    """rollouts.jsonl の 1 行スキーマ。F-0a 信号ゲートはこの reward/axes を集計する。

    quality は **常にフィールドを持つ** (現状 None=quality head 凍結中・将来の凍結解除に備える)。
    """
    return {
        "input_text": input_text,
        "prompt": prompt,
        "seed": int(seed),
        "axes": [float(a) for a in axes],
        "quality": (None if quality is None else float(quality)),
        "reward": float(reward),
    }


def build_input_texts(n, seed):
    """rollout 用に input_text を seed 決定的に n 件サンプリングする (重複許容・並び決定的)。

    n が プールより多ければ巡回して埋める。reward のばらつきはここの多様性が源なので、
    まず一巡ぶんは重複なしで全題材を当て、残りをランダム補充する。
    """
    import random
    rng = random.Random(seed)
    pool = list(INPUT_TEXTS)
    rng.shuffle(pool)
    out = []
    while len(out) < n:
        if len(out) % len(pool) == 0:
            rng.shuffle(pool)
        out.append(pool[len(out) % len(pool)])
    return out[:n]


# ============================================================
# BitNet state_dict リマップ (export 命名 → PyTorch state_dict 命名)
#   export_safetensors は bitnet.hpp レイアウト命名 (embed / layers.{i}.wq / final_norm)
#   で保存するため、BitNetDense.state_dict() 命名 (embed.weight / layers.{i}.wq.weight /
#   final_norm.weight) と一致しない。dump_golden と同じ写しを通してから strict ロードする。
# ============================================================
def _remap_bitnet_state_dict(sd, n_layers):
    """export_safetensors 命名の state_dict を BitNetDense の state_dict 命名へ写す。"""
    new_sd = {}
    new_sd["embed.weight"] = sd["embed"]
    new_sd["final_norm.weight"] = sd["final_norm"]
    for i in range(n_layers):
        p = f"layers.{i}."
        new_sd[p + "attn_norm.weight"] = sd[p + "attn_norm"]
        new_sd[p + "ffn_norm.weight"] = sd[p + "ffn_norm"]
        for w in ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"):
            new_sd[p + w + ".weight"] = sd[p + w]
    return new_sd


# ============================================================
# 研究機 実走部 (--run でのみ呼ばれる。重い import は関数内に隔離)
# ============================================================
def _research_assets_status(args):
    """実走資産の有無を (ok, reasons) で返す。開発機では ok=False → [SKIP]。

    BitNet 重み/vocab は args 経由 (定数是正 + CLI 上書きの二段構え) で参照する。
    """
    reasons = []
    try:
        import openvino  # noqa: F401
    except Exception:
        reasons.append("openvino 不在")
    try:
        import torch  # noqa: F401
    except Exception:
        reasons.append("torch 不在")
    if not os.path.isfile(SCORER_IR):
        reasons.append(f"ScorerNet IR 不在: {SCORER_IR}")
    if not os.path.isfile(args.bitnet_weights):
        reasons.append(f"BitNet 重み不在: {args.bitnet_weights}")
    if not os.path.isfile(args.vocab):
        reasons.append(f"vocab 不在: {args.vocab}")
    return (len(reasons) == 0), reasons


def preprocess_for_scorer(path):
    """生成 PNG → ScorerNet 入力 [1,3,512,512] f32 [0,1] NCHW。

    train_scorer の前処理と厳密一致: RGB 化 → 512 リサイズ → /255 → CHW。
    RGBA (透過) は WD14 と同じく白背景合成 (matting OFF 運用なら通常 RGB)。
    """
    import numpy as np
    from PIL import Image

    img = Image.open(path)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img.convert("RGBA")).convert("RGB")
    else:
        img = img.convert("RGB")
    img = img.resize((SCORER_IMG_SIZE, SCORER_IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0      # [512,512,3] [0,1]
    arr = arr.transpose(2, 0, 1)                          # CHW
    return np.ascontiguousarray(arr[np.newaxis, ...], dtype=np.float32)  # [1,3,512,512]


def run_on_research_machine(args, input_texts):
    """研究機でのみ: LM 生成 → SDXL(HTTP) → ScorerNet(OV) → reward → JSONL。"""
    import numpy as np  # noqa: F401
    import openvino as ov
    import torch
    from safetensors.torch import load_file

    # gen_scorer_corpus の HTTP/サーバ lifecycle ヘルパを再利用 (配管の二重実装回避)。
    import dollma_gen_scorer_corpus as gc
    # BitNet (LM) — greedy 生成・トークナイザ・列構築を train_bitnet から借りる。
    import train_bitnet as tb

    # --- LM ロード (重み/vocab は args 経由) ---
    device = args.lm_device
    tok = tb.Tokenizer(args.vocab)
    model = tb.BitNetDense().to(device)
    sd = load_file(args.bitnet_weights)
    # export_safetensors 命名 (embed / layers.{i}.* / final_norm) を state_dict 命名へ写す
    # (dump_golden と同経路)。写さずに strict ロードすると Missing/Unexpected key で落ちる。
    remapped = _remap_bitnet_state_dict(sd, tb.N_LAYERS)
    model.load_state_dict(remapped, strict=True)
    model.eval()
    print(f"[rollout] LM ロード完了 device={device} vocab={tok.vocab_size()} "
          f"weights={args.bitnet_weights}", file=sys.stderr)

    def lm_prompt(input_text):
        """input_text → danbooru タグ列文字列 (BitNet greedy)。"""
        prompt_ids = [tb.TOK_BOS] + tb.encode_text_greedy(tok, input_text) + [tb.TOK_SEP]
        gen_ids = tb._greedy_generate(model, prompt_ids, device)
        # id → タグ名 (specials id<5 は除外・tag idは tok.tags[id-5])。
        tags = [tok.tags[i - 5] for i in gen_ids if i >= 5]
        return ", ".join(tags)

    # --- ScorerNet (OV CPU FP32 = golden 経路) ---
    tok_dll = gc.resolve_tokenizers_dll(os.environ.get("DOLLAMA_OV_TOKENIZERS_DLL"))
    core = ov.Core()
    scorer = core.compile_model(core.read_model(SCORER_IR), args.scorer_device)
    print(f"[rollout] ScorerNet ロード完了 device={args.scorer_device}", file=sys.stderr)

    # --- サーバ env (本 txt2img 有効化 + matting OFF) ---
    env = dict(os.environ)
    env["DOLLAMA_OV_TOKENIZERS_DLL"] = tok_dll
    env["DOLLAMA_MATTING_WEIGHTS"] = args.no_matting_marker

    proc = None
    base_url = args.server_url
    if base_url:
        gc.wait_for_health(base_url, timeout=args.health_timeout)
    else:
        base_url = f"http://127.0.0.1:{args.port}"
        cmd = [args.exe, "--http", "--port", str(args.port), "--steps", str(args.steps)]
        print(f"[rollout] サーバ起動: {' '.join(cmd)}", file=sys.stderr)
        proc = gc.subprocess.Popen(cmd, cwd=ROOT, env=env)
        gc.wait_for_health(base_url, timeout=args.health_timeout, proc=proc)
        print("[rollout] サーバ ready", file=sys.stderr)

    img_dir = args.img_dir
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rewards = []
    try:
        with open(args.out, "w", encoding="utf-8") as fout:
            for i, input_text in enumerate(input_texts):
                # 1) LM 生成
                prompt = lm_prompt(input_text)
                if not prompt.strip():
                    print(f"[rollout] #{i:04d} 空 prompt (input='{input_text}') スキップ",
                          file=sys.stderr)
                    continue

                # 2) SDXL 生成 (HTTP・品質ネガは注入しない=素の anatomy を測る)
                body = gc.build_request_body({"prompt": prompt}, args.steps, "1024x1024")
                b64 = gc.post_generation(base_url, body, timeout=args.gen_timeout)
                img_path = os.path.join(img_dir, f"{i:06d}.png")
                gc.save_b64_png(b64, img_path)

                # 3) ScorerNet 採点 (logits [9])
                x = preprocess_for_scorer(img_path)
                logits = scorer(x)[scorer.output(0)][0]  # [9]
                axes = axes_from_logits(logits)
                quality = quality_from_logits(logits, QUALITY_ENABLED)

                # 4) 報酬
                reward = reward_from_scorer(axes, quality)
                rewards.append(reward)

                # 5) JSONL 行 (seed = server 内 wall-clock・非再現 → provenance に明記)
                row = rollout_row(input_text, prompt, args.seed, axes, quality, reward)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()

                if i < 5 or (i + 1) % 10 == 0:
                    print(f"[rollout] #{i:04d} reward={reward:+.4f} "
                          f"worst={max(axes):.3f} input='{input_text[:32]}' "
                          f"prompt='{prompt[:48]}'", file=sys.stderr)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except Exception:
                proc.kill()

    # --- 信号ゲート用サマリ (収集後に gpu-benchmarker が深掘りする最小統計) ---
    if rewards:
        rmin, rmax = min(rewards), max(rewards)
        rmean = sum(rewards) / len(rewards)
        var = sum((r - rmean) ** 2 for r in rewards) / len(rewards)
        provenance = {
            "stage": "F-0a rollout 収集",
            "n_rollouts": len(rewards),
            "reward_min": round(rmin, 4),
            "reward_max": round(rmax, 4),
            "reward_mean": round(rmean, 4),
            "reward_std": round(var ** 0.5, 4),
            "best_minus_worst": round(rmax - rmin, 4),  # 学習可能ギャップの粗指標
            "quality_enabled": QUALITY_ENABLED,
            "scorer_ir": os.path.relpath(SCORER_IR, ROOT).replace(os.sep, "/"),
            "bitnet_weights": os.path.relpath(args.bitnet_weights, ROOT).replace(os.sep, "/"),
            "out": args.out,
            "seed_note": "画像 seed = server 内 wall-clock(秒) ゆえ非再現",
        }
        print("[rollout] PROVENANCE " + json.dumps(provenance, ensure_ascii=False),
              file=sys.stderr)
        print(f"[rollout] 完了: {len(rewards)} 件 / reward [{rmin:+.3f}, {rmax:+.3f}] "
              f"std={var ** 0.5:.3f} → {args.out}")
    else:
        print("[rollout] 収集 0 件 (全 prompt 空?) → JSONL は空")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="F-0a 品質 FB ループ rollout 収集ハーネス (研究機専用・開発機は [SKIP])")
    ap.add_argument("--n", type=int, default=80, help="rollout 件数 (信号ゲートは 50-100 推奨)")
    ap.add_argument("--out", default="data/rollouts/rollouts.jsonl", help="JSONL 出力パス")
    ap.add_argument("--img-dir", dest="img_dir", default="data/rollouts/img")
    ap.add_argument("--seed", type=int, default=20260630, help="input_text サンプリング seed")
    ap.add_argument("--bitnet-weights", dest="bitnet_weights", default=BITNET_WEIGHTS,
                    help="BitNet 本線重み safetensors (既定=本線 #1 dense fp32)")
    ap.add_argument("--vocab", dest="vocab", default=VOCAB_PATH,
                    help="BitNet トークナイザ語彙 vocab.json")
    ap.add_argument("--exe", default=DOLLAMA_EXE, help="dollama 実行ファイル (--http)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--server-url", dest="server_url", default=None,
                    help="既存サーバ URL (指定時は subprocess 起動せず利用)")
    ap.add_argument("--lm-device", dest="lm_device", default="cpu",
                    help="BitNet LM 推論デバイス (cpu / cuda)")
    ap.add_argument("--scorer-device", dest="scorer_device", default="CPU",
                    help="ScorerNet OV デバイス (golden は CPU FP32)")
    ap.add_argument("--health-timeout", dest="health_timeout", type=float, default=240.0)
    ap.add_argument("--gen-timeout", dest="gen_timeout", type=float, default=180.0)
    ap.add_argument("--no-matting-marker", dest="no_matting_marker", default="__no_matting__")
    ap.add_argument("--run", dest="run", action="store_true",
                    help="研究機で実走する (RTX5080+OV+重み必須)。未指定は資産確認のみ。")
    args = ap.parse_args(argv)

    input_texts = build_input_texts(args.n, args.seed)

    ok, reasons = _research_assets_status(args)
    if not args.run:
        # 既定: 計画だけ表示 (実走しない)。
        print(f"[PLAN] rollout {args.n} 件 / out={args.out} / seed={args.seed}")
        print(f"[PLAN] BitNet 重み: {args.bitnet_weights}")
        print(f"[PLAN] 実走資産: {'揃っている' if ok else '不足'}"
              + ("" if ok else f" ({'; '.join(reasons)})"))
        print("[PLAN] 研究機で実走するには --run を付ける (gpu-benchmarker 担当)。")
        for t in input_texts[:3]:
            print(f"  例 input_text: {t}")
        return

    if not ok:
        # 開発機など資産不在 → 明示 [SKIP] で早期 return (例外は投げない)。
        print(f"[SKIP] 実走資産が揃っていないため rollout 収集をスキップ: {'; '.join(reasons)}")
        print("[SKIP] 研究機 (RTX5080 + OpenVINO + 実 SDXL/ScorerNet/BitNet 重み) で実走すること。")
        return

    run_on_research_machine(args, input_texts)


if __name__ == "__main__":
    main()
