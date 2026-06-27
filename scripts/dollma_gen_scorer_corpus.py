# -*- coding: utf-8 -*-
"""dollama Phase 4 Model B / B-3a / E-1 — 画像コーパス生成 + WD14 タグ付け driver。

**このスクリプトの実走部 (--run) は研究機 (RTX5080 + OpenVINO, 実 SDXL 重み配置済) でのみ走る。**
非研究機では `--dry-run` (既定) で配管だけ検証し、実生成・実推論はしない。
純ヘルパ (DLL 解決フォールバック・リクエスト body 組立・b64→PNG 保存・rating 抽出・
prompt 反映 sanity・jsonl 行スキーマ・build_jobs) は OV/SDXL/HTTP なしで単体テストできる。

役割 (2 段):
  1. SDXL 生成: 既存 txt2img HTTP サーバ (`dollama --http` の Txt2ImgGenerator/NPU) を subprocess
     で起動し、ジョブ毎に `/v1/images/generations` を叩いて画像を得る。良/悪サンプルが偏らない
     よう、品質ネガティブ注入 ON/OFF や崩れやすいプロンプトを混ぜて品質・解剖の分散を作る。
  2. WD14 タグ付け: 生成画像を WD14 SwinV2 (OV・CPU) でタグ付けし sigmoid スコア [N_TAGS] を得て
     `scorer_wd14.jsonl` (1 行 = {image, wd14_scores, prompt, rating, inject_quality_negatives}) に書く。

  → 出力 `scorer_wd14.jsonl` を `dollma_make_scorer_labels.py --wd14` に食わせると、OV/SDXL 非依存
     パートで 8 軸 soft target + 品質スカラ (provider) に変換され `scorer.{train,val}.jsonl` になる。
     生成と推論はここ (研究機)、ラベル化は非研究機でも可、という分業。

注意 (CLAUDE.md / E-1 前提):
  - 段1 本 txt2img(NPU) を有効化するには env `DOLLAMA_OV_TOKENIZERS_DLL` をサーバ subprocess に
    渡す (未設定だと段2 golden に落ちて prompt が無視されコーパスが無意味化する)。
  - matting OFF (不透明 RGB): env `DOLLAMA_MATTING_WEIGHTS` を存在しないパスにして make_matter→nullptr。
  - 画像 seed は server 内 time(秒)・非再現 (provenance に明記)。size は 1024×1024 固定。
  - WD14 は CPU が最速 (probe8)。前処理は dollma_label_image.preprocess を流用。
"""

import argparse
import base64
import glob
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

# 良サンプル品質ネガ 9 語 (character-bible §6 default_quality_negatives と厳密一致)。
# src/core/character.hpp::default_quality_negatives() の写し。変更時は両方を同時に直す。
QUALITY_NEGATIVES = (
    "bad hands, malformed hands, mutated hands, fused fingers, "
    "bad anatomy, deformed, lowres, worst quality, jpeg artifacts"
)


# ------------------------------------------------------------
# 崩れやすさを散らすプロンプト設計 (品質・解剖の分散を意図的に作る)
#   GOOD: 素直な単独被写体・品質ネガ ON で崩れにくくする。
#   HARD: 多人数 / 複雑ポーズ / 手の本数 / 小物把持 / foreshortening 等・品質ネガ OFF で崩れを誘発。
# 題材を髪型・上半身/全身・手・複数キャラ・小物・縮図 (foreshortening) に散らす。
# ------------------------------------------------------------
GOOD_PROMPTS = [
    "1girl, solo, long hair, simple background, upper body",
    "1girl, solo, short hair, standing, white background",
    "1boy, solo, looking at viewer, portrait, simple background",
    "1girl, solo, ponytail, school uniform, upper body, simple background",
    "1girl, solo, twintails, smile, white background, looking at viewer",
    "1girl, solo, long hair, sitting, indoors, simple background",
    "1boy, solo, short hair, jacket, upper body, plain background",
    "1girl, solo, braid, blue eyes, portrait, simple background",
    "1girl, solo, bob cut, hood, upper body, white background",
    "1girl, solo, wavy hair, dress, standing, simple background",
    "1boy, solo, messy hair, closed mouth, portrait, white background",
    "1girl, solo, hime cut, kimono, upper body, simple background",
    "1girl, solo, bangs, hoodie, looking at viewer, plain background",
    "1girl, solo, side ponytail, sweater, upper body, white background",
    "1boy, solo, undercut, collared shirt, portrait, simple background",
    "1girl, solo, long hair, glasses, upper body, white background",
    "1girl, solo, drill hair, frills, portrait, simple background",
    "1girl, solo, short hair, cat ears, upper body, plain background",
    "1boy, solo, ponytail, scarf, looking at viewer, simple background",
    "1girl, solo, straight hair, choker, upper body, white background",
    "1girl, solo, curly hair, beret, portrait, simple background",
    "1girl, solo, low twintails, ribbon, upper body, white background",
    "1boy, solo, swept bangs, turtleneck, portrait, simple background",
    "1girl, solo, half updo, earrings, upper body, plain background",
    "1girl, solo, blunt bangs, cardigan, looking at viewer, white background",
]
HARD_PROMPTS = [
    "2girls, complex pose, dynamic angle, intertwined arms",
    "1girl, holding two swords, spread fingers, action pose, foreshortening",
    "multiple girls, group, many hands, crowd",
    "1girl, full body, contorted pose, hands on hips, dynamic angle",
    "3girls, group hug, overlapping limbs, many hands",
    "1boy, reaching toward viewer, foreshortening, open hand, spread fingers",
    "2boys fighting, crossed arms, dynamic motion, extreme perspective",
    "1girl, holding bow and arrow, drawn bowstring, full body, twisted torso",
    "crowd of people, many faces, overlapping bodies, busy composition",
    "1girl, doing a handstand, full body, upside down, splayed fingers",
    "2girls, back to back, holding hands, intertwined fingers, dynamic pose",
    "1boy, punching toward camera, extreme foreshortening, clenched fist",
    "1girl, juggling, multiple objects, many hands raised, motion blur",
    "4girls, dance formation, overlapping arms, complex group pose",
    "1girl, holding a large gun, both hands on grip, action pose, low angle",
    "1boy, mid-cartwheel, full body, limbs spread, dynamic angle",
    "2girls wrestling, tangled limbs, grabbing, dynamic motion",
    "1girl, playing piano, both hands on keys, foreshortening, side view",
    "1girl, holding chopsticks and bowl, eating, hands close together",
    "1boy, climbing rope, full body, arms overhead, gripping hands",
    "3boys, team pose, arms over shoulders, many hands, group",
    "1girl, archery pose, full draw, twisted spine, foreshortened arm",
    "1girl, holding umbrella and bag, both hands full, walking, full body",
    "2girls, piggyback ride, overlapping limbs, holding on, dynamic pose",
    "1girl, doing splits, full body, extreme leg angle, hands on floor",
]


def build_jobs(n, seed):
    """生成ジョブ (prompt, inject_quality_negatives, rating) のリストを seed 決定的に作る。

    良/悪を交互に混ぜて品質・解剖の分散を作る (i 偶数=良・奇数=悪)。実生成はしない (リストだけ)。
    後方互換: 既存の good/hard 交互 + seed 決定構造 + inject_quality_negatives ラベルを維持。
    """
    import random
    rng = random.Random(seed)
    jobs = []
    for i in range(n):
        good = (i % 2 == 0)
        pool = GOOD_PROMPTS if good else HARD_PROMPTS
        jobs.append({
            "prompt": rng.choice(pool),
            "inject_quality_negatives": good,   # 良サンプルは品質ネガ ON
            "rating": "g",
        })
    return jobs


# ============================================================
# 純ヘルパ (OV/SDXL/HTTP 非依存・単体テスト対象)
# ============================================================
def resolve_tokenizers_dll(explicit=None):
    """openvino_tokenizers DLL のパスを解決する。

    優先順:
      1. explicit (env DOLLAMA_OV_TOKENIZERS_DLL) が実在ファイルならそれ。
      2. 無ければ python pkg `openvino_tokenizers` の lib/ から自動検出。
    どちらでも見つからなければ **明確に RuntimeError** (段2 golden 落ち=コーパス無意味化を防ぐ)。
    """
    if explicit:
        if os.path.isfile(explicit):
            return explicit
        raise RuntimeError(
            f"DOLLAMA_OV_TOKENIZERS_DLL に指定されたパスが存在しません: {explicit}")
    # pkg から自動検出
    try:
        import openvino_tokenizers
    except ImportError as e:
        raise RuntimeError(
            "openvino_tokenizers DLL を自動検出できません: python パッケージ "
            "'openvino_tokenizers' が import できず、env DOLLAMA_OV_TOKENIZERS_DLL も未指定。"
            " (未解決だと段1 本txt2img が段2 golden に落ち prompt が無視されコーパスが無意味化する)"
        ) from e
    lib_dir = os.path.join(os.path.dirname(openvino_tokenizers.__file__), "lib")
    cand = (glob.glob(os.path.join(lib_dir, "*tokenizers*.dll"))
            or glob.glob(os.path.join(lib_dir, "*tokenizers*.so"))
            or glob.glob(os.path.join(lib_dir, "*tokenizers*.dylib")))
    if not cand:
        raise RuntimeError(
            f"openvino_tokenizers pkg は見つかりましたが lib/ に DLL がありません: {lib_dir}")
    return cand[0]


def build_request_body(job, steps, size):
    """ジョブ → /v1/images/generations の JSON body を組み立てる。

    良サンプル (inject_quality_negatives=True) のみ negative_prompt に品質ネガ 9 語を流す。
    """
    body = {
        "prompt": job["prompt"],
        "steps": int(steps),
        "size": size,
        "response_format": "b64_json",
    }
    if job.get("inject_quality_negatives"):
        body["negative_prompt"] = QUALITY_NEGATIVES
    return body


def save_b64_png(b64, path):
    """b64_json 文字列を PNG バイト列にデコードして path に保存し、バイト数を返す。"""
    data = base64.b64decode(b64)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


def extract_rating(wd14_scores, names, categories):
    """WD14 スコア [N_TAGS] から rating (selected_tags.csv category==9) の最大スコア name を返す。

    rating タグが 1 つも無ければ None。general/sensitive/questionable/explicit のいずれか。
    """
    best, best_s = None, -1.0
    for i, c in enumerate(categories):
        if c == 9 and i < len(wd14_scores):
            s = float(wd14_scores[i])
            if s > best_s:
                best_s, best = s, names[i]
    return best


def wd14_row(image, wd14_scores, prompt, rating, inject):
    """scorer_wd14.jsonl の 1 行スキーマ (make_scorer_labels の入力境界)。"""
    return {
        "image": image,
        "wd14_scores": [float(v) for v in wd14_scores],
        "prompt": prompt,
        "rating": rating,
        "inject_quality_negatives": bool(inject),
    }


def prompt_overlap_tags(prompt, wd14_scores, names, thresh=0.35):
    """prompt の語と WD14 高スコアタグ (>=thresh) の重なり集合 (prompt 反映 sanity check 用)。

    prompt のカンマ区切りタグを underscore 表記 (`long hair`→`long_hair`) と素の語の両方に展開し、
    WD14 name (selected_tags.csv 表記=underscore) と小文字一致を取る。
    """
    ptoks = set()
    for part in prompt.split(","):
        t = part.strip().lower()
        if not t:
            continue
        ptoks.add(t.replace(" ", "_"))   # underscore 表記 (WD14 name 形式)
        for w in t.split():              # 素の単語 (1girl, solo など)
            ptoks.add(w)
    hit = set()
    for i, n in enumerate(names):
        if i < len(wd14_scores) and float(wd14_scores[i]) >= thresh:
            if n.lower() in ptoks:
                hit.add(n)
    return hit


# ============================================================
# HTTP / サーバ lifecycle (実走時のみ)
# ============================================================
def wait_for_health(base_url, timeout, proc=None, interval=2.0):
    """/health が {"status":"ok"} を返すまでポーリング (接続拒否=準備中としてリトライ)。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"サーバプロセスが /health ready 前に終了しました (exit={proc.returncode})")
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=3) as resp:
                if resp.status == 200:
                    j = json.loads(resp.read().decode("utf-8"))
                    if j.get("status") == "ok":
                        return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass  # 接続拒否=まだ起動中。リトライ。
        time.sleep(interval)
    raise TimeoutError(f"/health が {timeout}s 以内に ready になりませんでした: {base_url}")


def post_generation(base_url, body, timeout):
    """/v1/images/generations に POST し b64_json を返す。"""
    req = urllib.request.Request(
        base_url + "/v1/images/generations",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if not data.get("data") or "b64_json" not in data["data"][0]:
        raise RuntimeError(f"生成レスポンスに b64_json がありません: {str(data)[:200]}")
    return data["data"][0]["b64_json"]


# ------------------------------------------------------------
# 研究機実走部 (--run でのみ呼ばれる)。重い import は関数内に隔離。
# ------------------------------------------------------------
def run_on_research_machine(args, jobs):
    """研究機でのみ実行: SDXL 生成 (HTTP) → WD14 タグ付け → scorer_wd14.jsonl 出力。"""
    # 研究機専用の重い依存はここで import (本機テストの top-level import を汚さない)。
    import numpy as np  # noqa: F401  (preprocess が使う)
    import openvino as ov
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    from dollma_label_image import preprocess, load_tags, MODEL_XML, TAGS_CSV

    # --- 段1 本txt2img(NPU) 有効化に必須の env を解決 (未解決なら明確に失敗) ---
    tok_dll = resolve_tokenizers_dll(os.environ.get("DOLLAMA_OV_TOKENIZERS_DLL"))
    print(f"[corpus] tokenizers DLL: {tok_dll}", file=sys.stderr)

    # サーバ subprocess に渡す env (現環境 + tokenizers DLL + matting OFF)。
    env = dict(os.environ)
    env["DOLLAMA_OV_TOKENIZERS_DLL"] = tok_dll
    env["DOLLAMA_MATTING_WEIGHTS"] = args.no_matting_marker  # 存在しないパス → 不透明 RGB

    # --- サーバ lifecycle (--server-url 指定時は起動せず既存利用) ---
    proc = None
    base_url = args.server_url
    if base_url:
        print(f"[corpus] 既存サーバを利用: {base_url}", file=sys.stderr)
        wait_for_health(base_url, timeout=args.health_timeout)
    else:
        base_url = f"http://127.0.0.1:{args.port}"
        cmd = [args.exe, "--http", "--port", str(args.port), "--steps", str(args.steps)]
        print(f"[corpus] サーバ起動: {' '.join(cmd)} (cwd={ROOT})", file=sys.stderr)
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env)
        print(f"[corpus] /health を最大 {args.health_timeout}s ポーリング "
              f"(NPU compile + 5.1GB ロードで ~60s)", file=sys.stderr)
        wait_for_health(base_url, timeout=args.health_timeout, proc=proc)
        print("[corpus] サーバ ready", file=sys.stderr)

    img_dir = args.img_dir
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    t_start = time.time()
    n_sanity = max(1, min(args.sanity_n, len(jobs)))
    sanity_overlap_total = 0
    n_good = sum(1 for j in jobs if j["inject_quality_negatives"])
    try:
        # --- WD14 OV (CPU) を 1 回ロード ---
        names, categories = load_tags(TAGS_CSV)
        core = ov.Core()
        compiled = core.compile_model(core.read_model(MODEL_XML), args.wd14_device)
        print(f"[corpus] WD14 ロード完了 (device={args.wd14_device}, N_TAGS={len(names)})",
              file=sys.stderr)

        with open(args.out, "w", encoding="utf-8") as fout:
            for i, job in enumerate(jobs):
                # 1) SDXL 生成 (HTTP)
                body = build_request_body(job, args.steps, "1024x1024")
                b64 = post_generation(base_url, body, timeout=args.gen_timeout)
                img_path = os.path.join(img_dir, f"{i:06d}.png")
                nbytes = save_b64_png(b64, img_path)

                # 2) WD14 タグ付け (CPU)
                x = preprocess(img_path)
                out = compiled(x)[compiled.output(0)][0]  # [N_TAGS] sigmoid 済み
                scores = [float(v) for v in out]
                rating = extract_rating(scores, names, categories)

                # jsonl 行を即書き (長時間ジョブでも途中まで残す)
                rel = os.path.relpath(img_path, ROOT).replace(os.sep, "/")
                row = wd14_row(rel, scores, job["prompt"], rating,
                               job["inject_quality_negatives"])
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()

                # 3) prompt 反映 sanity check (最初の数枚)
                if i < n_sanity:
                    hits = prompt_overlap_tags(job["prompt"], scores, names)
                    sanity_overlap_total += len(hits)
                    print(f"[corpus] #{i:06d} bytes={nbytes} rating={rating} "
                          f"prompt-overlap({len(hits)})={sorted(hits)[:8]} "
                          f"prompt='{job['prompt'][:48]}'", file=sys.stderr)
                    # sanity 窓を撮り終えた直後に、相関ゼロ (=golden 落ち疑い) なら中断。
                    if i == n_sanity - 1 and sanity_overlap_total == 0:
                        raise RuntimeError(
                            f"prompt 反映 sanity check 失敗: 最初の {n_sanity} 枚で WD14 タグと "
                            "prompt 主要語の重なりが 0 でした。段1 本txt2img が段2 golden に落ちて "
                            "prompt が無視された疑いがあります (DOLLAMA_OV_TOKENIZERS_DLL を確認)。"
                            " コーパスが無意味になるため中断します。")
                elif (i + 1) % 25 == 0:
                    print(f"[corpus] 進捗 {i + 1}/{len(jobs)}", file=sys.stderr)
    finally:
        if proc is not None:
            print("[corpus] サーバ終了", file=sys.stderr)
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()

    elapsed = time.time() - t_start
    # provenance (再現可能性のため stderr に残す・seed は wall-clock 非再現)
    provenance = {
        "stage": "B-3a / E-1 scorer corpus",
        "n_images": len(jobs),
        "n_good": n_good,
        "n_hard": len(jobs) - n_good,
        "elapsed_sec": round(elapsed, 1),
        "sec_per_image": round(elapsed / len(jobs), 2) if jobs else 0.0,
        "out": args.out,
        "img_dir": img_dir,
        "size": "1024x1024",
        "steps": args.steps,
        "seed_note": "画像 seed = server 内 wall-clock time(秒) ゆえ非再現 (provenance に明記)",
        "matting": "OFF (DOLLAMA_MATTING_WEIGHTS=存在しないパス → 不透明 RGB)",
        "generator": "Txt2ImgGenerator (NPU CLIP-L/bigG + RTX5080 SDXL)",
        "wd14_model": os.path.relpath(MODEL_XML, ROOT).replace(os.sep, "/"),
        "wd14_device": args.wd14_device,
        "sanity_overlap_total": sanity_overlap_total,
        "tokenizers_dll": tok_dll,
    }
    print("[corpus] PROVENANCE " + json.dumps(provenance, ensure_ascii=False),
          file=sys.stderr)
    print(f"[corpus] 完了: {len(jobs)} 枚 / {elapsed:.1f}s → {args.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="SDXL 画像コーパス生成 + WD14 タグ付け driver (研究機専用・既定 dry-run)")
    ap.add_argument("--n", type=int, default=8, help="生成枚数 (既定 8=スモーク・本番は --n 200)")
    ap.add_argument("--seed", type=int, default=20260620)
    ap.add_argument("--out", default="data/scorer/scorer_wd14.jsonl",
                    help="WD14 スコアベクタ出力 jsonl (make_scorer_labels の入力)")
    ap.add_argument("--img-dir", dest="img_dir", default="data/scorer/img",
                    help="生成 PNG の保存先")
    ap.add_argument("--exe", default=os.path.join(ROOT, "build", "src", "dollama.exe"),
                    help="dollama 実行ファイル (--http サーバ起動用)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--server-url", dest="server_url", default=None,
                    help="既存サーバ URL (指定時は subprocess 起動せず利用)")
    ap.add_argument("--wd14-device", dest="wd14_device", default="CPU",
                    help="WD14 OV デバイス (probe8: CPU が最速)")
    ap.add_argument("--health-timeout", dest="health_timeout", type=float, default=240.0,
                    help="/health ready 待ちタイムアウト秒 (NPU compile+5.1GB ロードで ~60s)")
    ap.add_argument("--gen-timeout", dest="gen_timeout", type=float, default=180.0,
                    help="1 枚生成の HTTP タイムアウト秒 (生成 ~11s/枚)")
    ap.add_argument("--sanity-n", dest="sanity_n", type=int, default=3,
                    help="prompt 反映 sanity check の対象枚数 (先頭から)")
    ap.add_argument("--no-matting-marker", dest="no_matting_marker", default="__no_matting__",
                    help="DOLLAMA_MATTING_WEIGHTS に渡す存在しないパス (make_matter→nullptr)")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true", default=True,
                    help="(既定) 生成計画だけ出力し実走しない")
    ap.add_argument("--run", dest="dry_run", action="store_false",
                    help="研究機で実走する (RTX5080+OV 必須・実 SDXL 重み配置前提)")
    args = ap.parse_args(argv)

    jobs = build_jobs(args.n, args.seed)

    if args.dry_run:
        n_good = sum(1 for j in jobs if j["inject_quality_negatives"])
        print(f"[DRY-RUN] 生成計画 {len(jobs)} 枚 (良 {n_good} / 悪 {len(jobs) - n_good})")
        print(f"[DRY-RUN] 出力予定: {args.out} / 画像: {args.img_dir}")
        print(f"[DRY-RUN] prompt プール: good {len(GOOD_PROMPTS)} / hard {len(HARD_PROMPTS)}")
        print("[DRY-RUN] 実走は研究機で --run を付けて呼ぶこと (RTX5080+OV 必須)。")
        for j in jobs[:3]:
            print("  例:", json.dumps(j, ensure_ascii=False))
        return

    # --run: 研究機実走
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    run_on_research_machine(args, jobs)


if __name__ == "__main__":
    main()
