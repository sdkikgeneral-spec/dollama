# -*- coding: utf-8 -*-
"""dollma Phase 4 Model B / B-3a — 品質スコアラ教師ラベル構築 (配管のみ・OV/SDXL 非依存パート)。

ScorerNet (純 conv backbone・入力 [1,3,512,512]・出力 [1,1+8]) を蒸留訓練するための
**教師ラベル** を作る。教師は 2 系統:

  (A) 解剖 8 軸 soft target — quality_gate.hpp の catalog/AnomalyAxis を Python に写し、
      WD14 sigmoid スコアベクタを「軸ごとに該当タグ sigmoid の最大値」で連続値化する。
      QualityGate (C++) は閾値 hit だが、蒸留教師は **閾値なしの連続 soft 値** にする
      (これが Stage1 との差・plan/dataset-spec §16 確定方針)。
  (B) 品質スカラ — 外部美的モデルで採点。**選定 = Eugeoter/waifu-scorer-v4-beta (apache-2.0)**
      (一次情報 = HF API metadata + README とも apache-2.0・matting ISNet=Apache と同基準で許諾的・
      アニメ特化・CLIP+MLP 形態)。本ファイルでは provider 境界と出力正規化のみ確定し、
      重み実体 DL・CLIP image encode・実採点は研究機/コーパス生成後に差す (実走しない)。

**このスクリプトは OV/SDXL/GPU を一切要さない**。WD14 実推論は呼ばない。入力は
「研究機で WD14 を回した結果のスコアベクタ (jsonl/npy)」を受け取る境界にする。
SDXL 生成と WD14 実行は研究機前提で、本ファイルでは手順をコメント化するに留め実走しない。

軸の出力順は AnomalyAxis (quality_gate.hpp) に 1:1:
  0 Hands / 1 Limbs / 2 Head / 3 Eyes / 4 Ears / 5 Mouth / 6 Digits / 7 GlobalAnatomy

再現性: seed 固定・provenance (生成プロンプト・教師モデル名/版/ライセンス・seed) を出力に残す。
依存は標準ライブラリのみ (numpy も必須にしない)。dataset-spec §8 の分割規約に倣う。
"""

import argparse
import csv
import json
import os
import random
import sys
import time


# ============================================================
# 1. AnomalyAxis / catalog — quality_gate.hpp の写し (唯一のソースは C++ だが Python 側で再現)
#    src/infer/quality_gate.hpp::catalog() / enum AnomalyAxis と **必ず一致** させること。
#    変更時は両方を同時に直す (C++ が本体・本ファイルは蒸留教師用の鏡)。
# ============================================================

# AnomalyAxis 列挙の順序 = ScorerNet 出力 axis[8] の順序。C++ enum と 1:1。
AXIS_ORDER = [
    "Hands",          # 0
    "Limbs",          # 1
    "Head",           # 2
    "Eyes",           # 3
    "Ears",           # 4
    "Mouth",          # 5
    "Digits",         # 6
    "GlobalAnatomy",  # 7
]
AXIS_INDEX = {name: i for i, name in enumerate(AXIS_ORDER)}
N_AXIS = len(AXIS_ORDER)  # 8

# 異常タグ名 → 軸。quality_gate.hpp::catalog() の 18 エントリと完全一致。
# タグ名は **アンダースコア区切り** (WD14 selected_tags.csv の生 name 表記に合わせる)。
CATALOG = [
    # Hands
    ("bad_hands", "Hands"),
    # Limbs
    ("extra_arms", "Limbs"),
    ("missing_limb", "Limbs"),
    ("disembodied_limb", "Limbs"),
    ("severed_limb", "Limbs"),
    ("amputee", "Limbs"),
    ("double_amputee", "Limbs"),
    # Head
    ("multiple_heads", "Head"),
    ("disembodied_head", "Head"),
    ("severed_head", "Head"),
    ("extra_faces", "Head"),
    # Eyes / Ears / Mouth
    ("extra_eyes", "Eyes"),
    ("extra_ears", "Ears"),
    ("extra_mouth", "Mouth"),
    # Digits
    ("fewer_digits", "Digits"),
    # GlobalAnatomy
    ("bad_anatomy", "GlobalAnatomy"),
    ("bad_proportions", "GlobalAnatomy"),
    ("deformed", "GlobalAnatomy"),
]


# ============================================================
# 2. selected_tags.csv ローダ (tag 名 → 出力 index)
#    quality_gate.hpp::load_name_to_index と同流儀: header 1 行 + データ行 i = WD14 出力 index i。
# ============================================================
def load_name_to_index(csv_path):
    """selected_tags.csv (tag_id,name,category,count) から name → 出力 index dict を作る。

    データ行 i (header 除く) が WD14 出力ベクタの index i に対応する。
    重複 name は先勝ち (quality_gate.hpp と同じ)。
    """
    name_to_index = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = True
        index = 0
        for row in reader:
            if header:
                header = False
                continue
            if len(row) < 2:
                index += 1
                continue
            name = row[1]
            if name not in name_to_index:
                name_to_index[name] = index
            index += 1
    return name_to_index


def resolve_catalog(name_to_index):
    """CATALOG を index 解決する。CSV に無いタグは黙って skip (語彙差異に頑健)。

    戻り値: [(name, axis_name, index), ...]  (解決できたものだけ)
    """
    resolved = []
    for name, axis in CATALOG:
        idx = name_to_index.get(name)
        if idx is None:
            continue  # CSV に存在しないタグはスキップ
        resolved.append((name, axis, idx))
    return resolved


# ============================================================
# 3. 軸集約: WD14 スコアベクタ → 8 軸連続 soft target
#    「軸ごとに該当タグ sigmoid の最大値」(max-sigmoid 集約・閾値なし連続値)。
# ============================================================
def aggregate_axes(wd14_scores, resolved):
    """WD14 sigmoid スコアベクタ [N_TAGS] → axis[8] float in [0,1] (AnomalyAxis 順)。

    各軸 = その軸に属する解決済みタグの sigmoid スコアの **最大値**。
    該当タグが 1 つも無い軸は 0.0。範囲外 index は安全に skip。
    """
    axes = [0.0] * N_AXIS
    n = len(wd14_scores)
    for name, axis, idx in resolved:
        if idx >= n:
            continue  # CSV と入力長の不整合に備える (quality_gate.hpp と同方針)
        s = float(wd14_scores[idx])
        ai = AXIS_INDEX[axis]
        if s > axes[ai]:
            axes[ai] = s
    return axes


# ============================================================
# 4. 品質スカラ (B) — プラガブルなインターフェース
#    選定 = waifu_scorer_v4 (apache-2.0)。境界はプラガブルに保つ:
#    - 入力レコードが quality を持てばそれを採用 (passthrough)。
#    - 美的モデル provider に委譲 (waifu_scorer_v4)。重み/embed 未配置なら None (後埋め)。
#    - anatomy_proxy は **保留方針により既定では使わない** (ユーザー: 縮退に落とさない)。
#      引数で明示した時のみ縮退代理を出す (PL/ユーザー判断後の運用フック)。
# ============================================================
class QualityProvider:
    """品質スカラの供給元 (抽象)。研究機で外部美的モデルを差すための境界。"""

    name = "abstract"
    version = "none"
    license = "none"

    def score(self, record):
        raise NotImplementedError


class PassthroughQualityProvider(QualityProvider):
    """レコードに既に quality があればそれを通す。無ければ None (後埋め)。"""

    name = "passthrough"
    version = "1"
    license = "n/a"

    def score(self, record):
        q = record.get("quality")
        return float(q) if q is not None else None


class AnatomyProxyQualityProvider(QualityProvider):
    """縮退案: 外部美的モデルが使えない間の **代理スコア** (1 - max(8 軸))。

    **ユーザー方針 = 縮退保留**: 許諾的ライセンスのアニメ美的モデルが見つからない場合は
    これに落とさず quality=null で止める。美的モデル選定後の今は既定で使わず、
    将来 PL/ユーザーが明示判断したときの運用フックとして残す (provenance に proxy 印)。
    """

    name = "anatomy_proxy"
    version = "1"
    license = "n/a"

    def score(self, record):
        axes = record.get("axis")
        if not axes:
            return None
        return float(max(0.0, 1.0 - max(axes)))


# ------------------------------------------------------------
# WaifuScorer V4-beta (Eugeoter/waifu-scorer-v4-beta・apache-2.0) — 選定された美的モデル
#
# 選定根拠 (一次情報・HF API + README で確認):
#   - ライセンス: HF API metadata + README とも **apache-2.0** (matting ISNet=Apache と同基準で許諾的)。
#   - アニメ特化: README「a aesthetic scorer model designed to score anime-like images」。
#   - 形態: CLIP+MLP (improved-aesthetic-predictor 系譜)。model.safetensors の MLP は
#       768(=CLIP ViT-L/14 image embed) → 2048 → 512 → 256 → 128 → 32 → 1 (BatchNorm 入り)。
#   - 出力レンジ: 生スコア **0〜10** (README「Score ranges from 0 to 10. Higher is better」)。
#   - 重み入手元: huggingface.co/Eugeoter/waifu-scorer-v4-beta/model.safetensors (研究機で DL)。
#
# 採点には CLIP ViT-L/14 **image** encoder が要る (dollama の CLIP-L は text encoder。image は別)。
# 本 provider は「CLIP image embedding [768] → MLP → 0..10 → /10 で [0,1] 化」の **配管と正規化のみ**
# を確定し、CLIP image encode と MLP 重みは研究機/コーパス生成後に差す (本機では DL も実走もしない)。
# ------------------------------------------------------------
class WaifuScorerV4Provider(QualityProvider):
    """Eugeoter/waifu-scorer-v4-beta (apache-2.0) を使う品質スカラ provider。

    本開発機 (実画像なし・SDXL 未生成・重み未 DL) では **実走しない**。確定するのは:
      - 出力正規化 normalize_score: 生スコア 0..10 → [0,1] クランプ (本機乾式でも研究機でも同関数)。
      - score 境界: record の 'clip_image_embed':[768] と重みが揃わなければ None (後埋め)。
    研究機では ① CLIP ViT-L/14 image encoder で画像→[768] embed ② MLP forward → 生スコア
    → normalize_score、を _forward_mlp に結線する。
    """

    name = "waifu_scorer_v4"
    version = "v4-beta"
    license = "apache-2.0"
    repo = "Eugeoter/waifu-scorer-v4-beta"
    embed_dim = 768          # CLIP ViT-L/14 image embedding 次元
    raw_score_max = 10.0     # README: スコアは 0〜10

    def __init__(self, weights_path=None):
        # weights_path = model.safetensors のパス (研究機で配置)。本機では None のまま。
        self.weights_path = weights_path
        self._mlp = None  # 実 MLP は研究機で lazy ロード (本機では DL しない)

    @staticmethod
    def normalize_score(raw):
        """生スコア (0〜10) → [0,1] にクランプ正規化。乾式でも実走でも同じ純関数。"""
        return float(min(1.0, max(0.0, float(raw) / WaifuScorerV4Provider.raw_score_max)))

    def score(self, record):
        # 配管: CLIP image embedding と重みが揃わなければ採点不能 → None (後埋め)。
        emb = record.get("clip_image_embed")
        if emb is None or self.weights_path is None:
            return None
        raw = self._forward_mlp(emb)  # 研究機実走部 (MLP forward → 生スコア 0..10)
        return self.normalize_score(raw)

    def _forward_mlp(self, emb):
        # !!! 研究機実走部 (本機では到達しない: weights_path/embed が揃わない) !!!
        raise NotImplementedError(
            "WaifuScorer MLP forward は研究機実走部 (本機は配管のみ)。実走時に下記を結線:\n"
            "  - safetensors_load(self.weights_path) で MLP 重みをロード\n"
            "  - emb[768] を 768→2048→512→256→128→32→1 (BatchNorm/ReLU) に通し生スコアを得る\n"
            "  - CLIP image embedding は研究機の CLIP ViT-L/14 image encoder で採る (DL/実走は研究機)")


# ------------------------------------------------------------
# AnimeAesthetic (deepghs/anime_aesthetic・openrail) — 7 段順序クラス美的モデル
#
# 採用判断 (PL/ユーザー relay・本機では一次情報で形態のみ確認):
#   - ライセンス: openrail (HF API metadata + README とも)。openrail は商用/再配布/改変可で
#     制限は付属の行動禁止条項のみ → 「教師でラベル生成 → 自作 ScorerNet に蒸留」用途は抵触なし。
#     教師モデル自体は再配布しない (生成ラベルのみ保持) のでクリーン。
#     ※ openrail は許諾的 (Apache/MIT) とは別系統。**ライセンス採否は最終的にユーザー確認事項**。
#   - 形態: ONNX (swinv2pv3 backbone img_size=448 / caformer_s36 もあり)。meta.json で確認。
#   - 出力: 7 段 **順序クラス** ロジット/確率。labels =
#       [masterpiece, best, great, good, normal, low, worst]  (index 0=最良 .. 6=最悪)。
#   - 重み入手元: huggingface.co/deepghs/anime_aesthetic/<backbone>/model.onnx (研究機で DL)。
#
# 連続 quality スカラ [0,1] への写像 (= 期待順序スコア):
#   各クラス index k (0..6) に均等値 v_k = (6 - k) / 6 を割当 (masterpiece=1.0 .. worst=0.0・線形等間隔)。
#   quality = Σ_k p_k · v_k  (softmax 確率との期待値)。p が確率分布なら結果は必ず [0,1]。
# 本 provider は **この写像と前処理仕様のみ** を確定し、ONNX 推論本体は研究機で差す (本機 DL/実走なし)。
# ------------------------------------------------------------
class AnimeAestheticDeepghsProvider(QualityProvider):
    """deepghs/anime_aesthetic (openrail) を使う品質スカラ provider。

    本開発機 (ONNX/OV 無・重み未 DL・実画像なし) では **推論しない**。確定するのは:
      - expected_score: 7 クラス確率 → 期待順序スコア [0,1] (純計算・本機でも実走可)。
      - score 境界: record の 'aesthetic_probs':[7] があればそれを写像。無ければ None (後埋め)。
    研究機では ① ONNX (swinv2pv3 448) で画像→7 クラスロジット ② softmax → aesthetic_probs を
    record に積む ③ ここで expected_score で [0,1] 化、を結線する。
    """

    name = "anime_aesthetic_deepghs"
    version = "swinv2pv3_v0_448_ls0.2"
    license = "openrail"
    repo = "deepghs/anime_aesthetic"
    # 順序クラス (index 0=最良 masterpiece .. 6=最悪 worst)。meta.json labels と一致。
    labels = ["masterpiece", "best", "great", "good", "normal", "low", "worst"]
    n_classes = 7
    img_size = 448             # swinv2pv3 backbone (meta.json img_size)
    # 前処理仕様 (WD14 SwinV2 と同系: 0-255 float・NHWC・正規化なし)。
    # ※ 研究機で ONNX の実 IO に厳密一致させること (CLIP の i32/i64 教訓・wd14.hpp 前処理規約)。
    preprocess = "img 0-255 float32, NHWC [1,448,448,3], 正規化なし (研究機で ONNX IO に厳密一致)"

    @staticmethod
    def class_values(n=7):
        """各クラス index k → 均等値 v_k = (n-1-k)/(n-1)。masterpiece=1.0 .. worst=0.0。"""
        last = n - 1
        return [(last - k) / last for k in range(n)]

    @classmethod
    def expected_score(cls, probs):
        """7 クラス確率 [7] → 期待順序スコア Σ p_k·v_k ∈ [0,1] (純計算)。

        probs は softmax 済み確率を想定 (合計≈1)。長さ不一致は ValueError。
        合計が 0 (全ゼロ) の場合は None (採点不能)。
        """
        if probs is None or len(probs) != cls.n_classes:
            raise ValueError(f"aesthetic_probs は長さ {cls.n_classes} が必要 (実 {None if probs is None else len(probs)})")
        vals = cls.class_values(cls.n_classes)
        total = sum(float(p) for p in probs)
        if total <= 0.0:
            return None
        # 念のため正規化 (確率でなくロジット由来の生値が来ても期待値が [0,1] に収まる)。
        s = sum(float(p) * v for p, v in zip(probs, vals)) / total
        return float(min(1.0, max(0.0, s)))

    def score(self, record):
        probs = record.get("aesthetic_probs")
        if probs is None:
            return None  # 7 クラス確率未配置 (本機) → 後埋め境界
        return self.expected_score(probs)


def make_quality_provider(kind, weights_path=None):
    if kind == "passthrough":
        return PassthroughQualityProvider()
    if kind == "waifu_scorer_v4":
        return WaifuScorerV4Provider(weights_path=weights_path)
    if kind == "anime_aesthetic_deepghs":
        return AnimeAestheticDeepghsProvider()
    if kind == "anatomy_proxy":
        return AnatomyProxyQualityProvider()
    raise ValueError(f"unknown quality provider: {kind}")


# ============================================================
# 5. 入力読み込み — WD14 スコアベクタ (研究機で WD14 を回した結果) を受ける境界
#    形式: jsonl 1 行 = {image, wd14_scores:[...], prompt?, rating?, quality?, clip_image_embed?, aesthetic_probs?, meta?}
#    npy 経路は研究機で巨大スコアベクタを扱う場合用 (sidecar index jsonl + .npy 行列)。
# ============================================================
def load_wd14_records_jsonl(path):
    """WD14 スコア入力 jsonl を読む。

    各行に必須: "image" (画像参照パス), "wd14_scores" (sigmoid 済み [N_TAGS] float リスト)。
    任意: "prompt", "rating", "quality", "clip_image_embed" [768], "aesthetic_probs" [7] (研究機で採る)。
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "image" not in rec or "wd14_scores" not in rec:
                raise ValueError("WD14 入力行に 'image' と 'wd14_scores' が必要")
            records.append(rec)
    return records


# ============================================================
# 6. サンプル構築 — 1 入力レコード → scorer サンプル (image, quality, axis[8], provenance)
# ============================================================
def build_sample(record, resolved, quality_provider):
    """WD14 入力レコード → scorer 教師サンプル dict (schema は dataset-spec §16)。"""
    axes = aggregate_axes(record["wd14_scores"], resolved)
    sample = {
        "image": record["image"],
        "quality": None,          # B 系統 (provider 由来 or 後埋め)
        "axis": axes,             # A 系統 (解剖 8 軸 soft target・AnomalyAxis 順)
        "meta": {
            "prompt": record.get("prompt"),
            "rating": record.get("rating"),
            "n_tags_in": len(record["wd14_scores"]),
        },
    }
    # anatomy_proxy だけは axis を見るので sample を渡す。他は元 record を渡す
    # (clip_image_embed / quality を見るため)。
    if isinstance(quality_provider, AnatomyProxyQualityProvider):
        sample["quality"] = quality_provider.score(sample)
    else:
        sample["quality"] = quality_provider.score(record)
    return sample


# ============================================================
# 7. 分割 (seed 固定・dataset-spec §8 踏襲) + provenance
# ============================================================
def split_train_val(samples, val_ratio, seed):
    """seed 固定でシャッフルし train/val に分ける。image 参照単位でリークしないよう
    同一 image が train/val に跨らないことも assert する (dataset-spec §8 踏襲)。"""
    rng = random.Random(seed)
    order = list(range(len(samples)))
    rng.shuffle(order)
    n_val = int(round(len(samples) * val_ratio))
    val_idx = set(order[:n_val])
    train, val = [], []
    train_imgs, val_imgs = set(), set()
    for i, s in enumerate(samples):
        if i in val_idx:
            val.append(s)
            val_imgs.add(s["image"])
        else:
            train.append(s)
            train_imgs.add(s["image"])
    leak = train_imgs & val_imgs
    if leak:
        raise AssertionError(f"image リーク (train/val 跨り) {len(leak)} 件: {list(leak)[:3]}")
    return train, val


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def axis_distribution(samples):
    """軸ごとの平均/最大 soft 値・>0 件数を集計 (stats 用)。"""
    n = len(samples)
    dist = {}
    for ai, name in enumerate(AXIS_ORDER):
        vals = [s["axis"][ai] for s in samples]
        nz = sum(1 for v in vals if v > 0.0)
        dist[name] = {
            "mean": round(sum(vals) / n, 6) if n else 0.0,
            "max": round(max(vals), 6) if vals else 0.0,
            "nonzero": nz,
        }
    return dist


def build_provenance(args, quality_provider, n_in, n_train, n_val):
    """再現可能性のための provenance。生成プロンプトはサンプル側 meta に持つので
    ここには教師モデル名/版/ライセンス・seed・出典・状態を残す。"""
    qmeta = {
        "provider": quality_provider.name,
        "version": quality_provider.version,
        "license": getattr(quality_provider, "license", "none"),
    }
    if isinstance(quality_provider, WaifuScorerV4Provider):
        qmeta["repo"] = quality_provider.repo
        qmeta["form"] = "CLIP ViT-L/14 image embed[768] -> MLP -> raw 0..10 -> /10 -> [0,1]"
        qmeta["selected_reason"] = (
            "apache-2.0 (許諾的・ISNet と同基準) + アニメ特化 + CLIP+MLP。"
            "重み/CLIP image encode は研究機で配置・採点 (本ファイルは配管/正規化のみ)")
        qmeta["weights_path"] = quality_provider.weights_path  # None なら未配置 (本機)
    if isinstance(quality_provider, AnimeAestheticDeepghsProvider):
        qmeta["repo"] = quality_provider.repo
        qmeta["labels"] = quality_provider.labels
        qmeta["form"] = "ONNX swinv2pv3 448 -> 7 段順序クラス確率 -> 期待値 Sum p_k*(6-k)/6 -> [0,1]"
        qmeta["preprocess"] = quality_provider.preprocess
        qmeta["selected_reason"] = (
            "openrail (商用/再配布/改変可・行動制限のみ・教師非再配布でクリーン) + アニメ特化 + ONNX。"
            "7 クラス確率は研究機 ONNX 推論で採り record に積む (本ファイルは写像/前処理仕様のみ)")
    return {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "counts": {"input": n_in, "train": n_train, "val": n_val},
        "axis_order": AXIS_ORDER,
        "teachers": {
            "anatomy": {
                "name": "wd14+quality_gate.catalog (max-sigmoid 軸集約)",
                "source": "WD14 SwinV2 v3 selected_tags.csv (研究機推論・本ファイルは非実行)",
                "catalog_entries": len(CATALOG),
                "note": "閾値なし連続 soft target (QualityGate の閾値 hit と異なる)",
            },
            "quality": qmeta,
        },
        "inputs": {
            "wd14_scores": os.path.basename(args.wd14),
            "selected_tags_csv": os.path.basename(args.selected_tags),
        },
        "status": "DRY: WD14/SDXL/美的モデル 未実走・配管のみ (dataset-spec §16)",
    }


# ============================================================
# 8. main
# ============================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="品質スコアラ教師ラベル構築 (B-3a 配管・OV/SDXL 非依存)")
    ap.add_argument("--wd14", required=True,
                    help="WD14 スコアベクタ jsonl (研究機で WD14 を回した結果)")
    ap.add_argument("--selected-tags", dest="selected_tags", required=True,
                    help="WD14 selected_tags.csv (tag 名→index 解決用)")
    ap.add_argument("--out-dir", dest="out_dir", default="data/scorer",
                    help="出力先ディレクトリ")
    ap.add_argument("--quality-provider", dest="quality_provider",
                    default="waifu_scorer_v4",
                    choices=["passthrough", "waifu_scorer_v4", "anime_aesthetic_deepghs", "anatomy_proxy"],
                    help="品質スカラ (B) の供給元。既定 = 選定済み waifu_scorer_v4 (apache-2.0)。"
                         "anatomy_proxy は縮退 (ユーザー方針=保留・明示時のみ)")
    ap.add_argument("--quality-weights", dest="quality_weights", default=None,
                    help="美的モデル重み (model.safetensors) パス。研究機で配置。本機は未指定 (=None)")
    ap.add_argument("--val-ratio", dest="val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20260620)
    args = ap.parse_args(argv)

    name_to_index = load_name_to_index(args.selected_tags)
    resolved = resolve_catalog(name_to_index)
    print(f"catalog 解決: {len(resolved)}/{len(CATALOG)} 件 (CSV={os.path.basename(args.selected_tags)})")

    provider = make_quality_provider(args.quality_provider, weights_path=args.quality_weights)
    records = load_wd14_records_jsonl(args.wd14)
    print(f"WD14 入力: {len(records)} 件")

    samples = [build_sample(r, resolved, provider) for r in records]
    train, val = split_train_val(samples, args.val_ratio, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(os.path.join(args.out_dir, "scorer.train.jsonl"), train)
    write_jsonl(os.path.join(args.out_dir, "scorer.val.jsonl"), val)

    prov = build_provenance(args, provider, len(records), len(train), len(val))
    prov["axis_dist_train"] = axis_distribution(train) if train else {}
    n_q = sum(1 for s in samples if s["quality"] is not None)
    prov["quality_filled"] = {"with_quality": n_q, "missing": len(samples) - n_q}
    with open(os.path.join(args.out_dir, "scorer_stats.json"), "w", encoding="utf-8") as f:
        json.dump(prov, f, ensure_ascii=False, indent=2)

    print(f"train {len(train)} / val {len(val)} → {args.out_dir}")
    print(f"quality 充足: {n_q}/{len(samples)} (provider={provider.name}, license={getattr(provider,'license','n/a')})")


if __name__ == "__main__":
    main()
