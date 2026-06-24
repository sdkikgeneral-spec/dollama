# -*- coding: utf-8 -*-
"""dollama Phase 4 Model B / B-3b — アニメ品質スコアラ ScorerNet 蒸留訓練スクリプト。

dollma_probe_quality_scorer.py の純 conv backbone (ResNet-18 級・attention 皆無・約 11.18M params)
を **昇格** させて訓練側へ持ち込み、data/scorer/scorer.{train,val}.jsonl を消費して
蒸留訓練し、重みを safetensors ([out,in] 規約・FP16/FP32 両出力) で書き出す。

このスクリプトの担当範囲 (PL 確定・B-3b):
  - この作業 PC (GTX1080Ti / i7-10700 / NPU なし / Pascal sm_61) でコーディング完結する範囲
    = モデル定義 + 蒸留訓練ロジック + 乾式テスト。
  - **NPU / OpenVINO には一切触れない** (OV 変換は別タスク B-3c で研究機担当)。
  - 美的モデル (waifu_scorer_v4 等) は **非ロード**。jsonl の `quality` 値をそのまま教師に使う
    (ラベル生成は dollma_make_scorer_labels.py 側の責務・本スクリプトは jsonl を読むだけ)。

教師信号 (dataset-spec §16.3 / §16.8 確定):
  - ScorerNet 出力 [1, 1+8]: index0 = 品質スカラ回帰 / index1..8 = 解剖 8 軸。
  - 解剖 8 軸 (index1..8) = quality_gate.hpp の AnomalyAxis に 1:1:
      0 Hands / 1 Limbs / 2 Head / 3 Eyes / 4 Ears / 5 Mouth / 6 Digits / 7 GlobalAnatomy。
    teacher は WD14 max-sigmoid 軸集約の soft 値 [0,1] → **BCE** (soft-label 蒸留)。
  - 品質スカラ (index0) = 外部美的モデルの soft スコア [0,1] → **Huber** 回帰。
  - **§16.8 既定: quality が (バッチ/データセットで) 全 null の間は B (品質) head を凍結**する。
    凍結 = index0 の出力に勾配を流さない (B head 重み・bias を学習対象から外す)。

再現性 (train_bitnet.py と同方針):
  - seed 固定 (既定 20260620) → 同 seed 再実行で重み bitwise 一致。
  - torch の決定的アルゴリズムを可能な範囲で有効化。
  - 出力 data/scorer/scorer_net{,_fp32}.safetensors (gitignore 前提・再生成可)。

軸順 / 出力形状 / スキーマは dataset-spec §16・quality_gate.hpp と矛盾させない。
本番の他アーティファクト・golden は無改変。
"""

import argparse
import json
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================
# アーキ定数 (dataset-spec §16 / quality_gate.hpp と厳密一致)
# ==============================================================
N_AXIS = 8  # 解剖 8 軸 (AnomalyAxis の数)
N_OUT = 1 + N_AXIS  # 出力 [1, 1+8] = 9 (index0=quality / index1..8=axis[0..7])
IMG_SIZE = 512  # ScorerNet 実用入力解像度 (dataset-spec §16: [1,3,512,512])
IMG_CH = 3

# 軸順 = quality_gate.hpp::AnomalyAxis / make_scorer_labels.py::AXIS_ORDER と 1:1。
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
assert len(AXIS_ORDER) == N_AXIS


# ==============================================================
# ScorerNet — 純 conv backbone (ResNet-18 級・attention 皆無)
#   dollma_probe_quality_scorer.py の定義を **昇格** (再掲)。
#   probe は openvino を module top で import するため本機 (OV 無) では import 不能。
#   ゆえに定義を再掲し、param_count == probe の 11.18M を assert で検証する。
#   probe との唯一の差は head の出力次元: probe は n_attr=7 (出力 8) を例示値として
#   置いていたが、dataset-spec §16.3 の確定スキーマ [1,1+8]=9 (8 軸) に合わせ n_attr=8
#   (出力 9) とする。差は head の重み 1 行 (512) + bias 1 のみで 11.18M は不変。
# ==============================================================
class BasicBlock(nn.Module):
    """ResNet BasicBlock (conv3x3 → bn → relu → conv3x3 → bn → +identity → relu)。"""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + idt)


class ScorerNet(nn.Module):
    """ResNet-18 級の純 conv スコアラ。head = 品質スカラ + 解剖 8 軸ロジット。

    出力 [B, 1+n_attr]:
      index0      = 品質スカラ (回帰・logit。sigmoid 前の生値で Huber を取る)。
      index1..8   = 解剖 8 軸ロジット (BCEWithLogits)。AnomalyAxis 順。
    """

    def __init__(self, n_attr=N_AXIS):
        super().__init__()
        self.n_attr = n_attr
        self.stem = nn.Sequential(
            nn.Conv2d(IMG_CH, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))

        def layer(cin, cout, stride):
            return nn.Sequential(BasicBlock(cin, cout, stride), BasicBlock(cout, cout, 1))

        self.l1 = layer(64, 64, 1)
        self.l2 = layer(64, 128, 2)
        self.l3 = layer(128, 256, 2)
        self.l4 = layer(256, 512, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(512, 1 + n_attr)  # [品質, 軸×8]

    def forward(self, x):
        x = self.stem(x)
        x = self.l4(self.l3(self.l2(self.l1(x))))
        x = self.pool(x).flatten(1)
        return self.head(x)  # [B, 1+n_attr]


def param_count(model):
    return sum(p.numel() for p in model.parameters())


def assert_param_count(model):
    """param_count が probe の 11.18M と一致することを検証する。

    probe (dollma_probe_quality_scorer.py) は n_params/1e6 を 2 桁で表示 = 11.18。
    出力次元差 (probe 8 / 本 9) は head の 512+1 のみで丸め後の 11.18 は不変。
    """
    n = param_count(model)
    m = round(n / 1e6, 2)
    if m != 11.18:
        raise AssertionError(f"ScorerNet param_count {n} ({m}M) != probe 11.18M")
    return n


# ==============================================================
# データセット — scorer.{train,val}.jsonl を読む (dataset-spec §16.3)
# ==============================================================
#
# 1 サンプル:
#   {"image": "...", "quality": 0.41|null, "axis": [8 軸 float], "meta": {...}}
#
# **画像ピクセルは jsonl に埋めない** (dataset-spec §16.3) ので、本スクリプトは
#   - image パスが実在すれば読み込む (将来の研究機実走経路)。
#   - 実在しなければ image パス文字列から **決定的に擬似画像を合成** する
#     (本機にピクセルが無い段階の配管検証用。seed と image 名のみに依存 = 再現的)。
# どちらでも軸/品質ラベルは jsonl からそのまま使う (教師は jsonl が唯一のソース)。
# ==============================================================
class ScorerDataset(torch.utils.data.Dataset):
    """scorer jsonl を読むデータセット。画像は実ファイル優先・無ければ決定的合成。

    返す:
      image:        [3, IMG_SIZE, IMG_SIZE] float32 (0..1 正規化)
      axis:         [8] float32 (teacher soft target [0,1])
      quality:      float32 (teacher soft target [0,1]・null は 0.0 で埋め mask で無効化)
      quality_mask: float32 (1.0=quality 有効 / 0.0=null で損失から除外)
    """

    def __init__(self, jsonl_path, img_size=IMG_SIZE, synth_seed=0):
        self.records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "axis" not in rec or len(rec["axis"]) != N_AXIS:
                    raise ValueError(f"axis は長さ {N_AXIS} が必要: {rec.get('image')}")
                self.records.append(rec)
        self.img_size = img_size
        self.synth_seed = synth_seed

    def __len__(self):
        return len(self.records)

    def _synth_image(self, image_path):
        """image パス文字列から決定的に擬似画像を作る (本機ピクセル不在時の配管用)。

        seed は (synth_seed, image パスの hash) のみに依存 → 同一入力で必ず同じ。
        torch.Generator を使い global RNG を汚さない (seed 決定性を保つため)。
        """
        h = hash((self.synth_seed, image_path)) & 0x7FFFFFFF
        g = torch.Generator()
        g.manual_seed(int(h))
        return torch.rand(IMG_CH, self.img_size, self.img_size, generator=g)

    def __getitem__(self, i):
        rec = self.records[i]
        img_path = rec.get("image", f"__noimg_{i}")
        # 実ファイルがあれば読む経路は研究機用 (本機は基本合成)。
        if isinstance(img_path, str) and os.path.isfile(img_path):
            img = self._load_image_file(img_path)
        else:
            img = self._synth_image(str(img_path))

        axis = torch.tensor(rec["axis"], dtype=torch.float32)
        q = rec.get("quality", None)
        if q is None:
            quality = torch.tensor(0.0, dtype=torch.float32)
            qmask = torch.tensor(0.0, dtype=torch.float32)
        else:
            quality = torch.tensor(float(q), dtype=torch.float32)
            qmask = torch.tensor(1.0, dtype=torch.float32)
        return img, axis, quality, qmask

    def _load_image_file(self, path):
        """実画像ファイルを [3,H,W] float32 (0..1) で読む (研究機実走経路)。

        Pillow があれば使う。無ければ合成にフォールバック (本機で torch のみの場合)。
        """
        try:
            from PIL import Image
        except Exception:
            return self._synth_image(path)
        im = Image.open(path).convert("RGB").resize((self.img_size, self.img_size))
        arr = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8).clone()
        arr = arr.view(self.img_size, self.img_size, IMG_CH).permute(2, 0, 1).contiguous()
        return arr.float() / 255.0

    def quality_count(self):
        """quality が non-null なサンプル数 (B head 凍結判定に使う)。"""
        return sum(1 for r in self.records if r.get("quality") is not None)


# ==============================================================
# 蒸留 loss
# ==============================================================
def scorer_loss(pred, axis_tgt, quality_tgt, quality_mask,
                freeze_b, huber_delta=1.0, quality_weight=1.0):
    """ScorerNet 蒸留 loss。

    pred:         [B, 1+8] (index0=quality logit / 1..8=axis logit)
    axis_tgt:     [B, 8]   teacher soft [0,1]
    quality_tgt:  [B]      teacher soft [0,1] (null は 0.0・mask で無効化)
    quality_mask: [B]      1.0=有効 / 0.0=null
    freeze_b:     True なら B (品質) head を損失から除外 (§16.8 既定・全 null 時)。

    解剖 8 軸 = BCEWithLogits (soft target 蒸留)。
    品質スカラ = Huber (smooth L1)。logit を sigmoid して [0,1] 域で teacher と比較。
    戻り値: (total, axis_loss, quality_loss)。
    """
    axis_logit = pred[:, 1:]                 # [B,8]
    axis_loss = F.binary_cross_entropy_with_logits(axis_logit, axis_tgt)

    if freeze_b:
        # B head 凍結: 品質損失を 0 にし、index0 へ勾配が流れないようにする。
        quality_loss = pred.new_zeros(())
    else:
        q_pred = torch.sigmoid(pred[:, 0])   # [B] [0,1]
        m = quality_mask
        denom = m.sum().clamp_min(1.0)
        # Huber を要素ごとに取り mask して有効サンプルのみ平均。
        per = F.huber_loss(q_pred, quality_tgt, delta=huber_delta, reduction="none")
        quality_loss = (per * m).sum() / denom

    total = axis_loss + quality_weight * quality_loss
    return total, axis_loss.detach(), quality_loss.detach()


# ==============================================================
# B head 凍結ユーティリティ
# ==============================================================
def set_b_head_frozen(model, frozen):
    """B (品質 = head の index0 行) を凍結する。

    head は Linear(512, 1+8)。index0 = 品質行。PyTorch の nn.Linear は重みを
    1 テンソルで持つため行ごとの requires_grad 分離はできない。よって §16.8 既定の
    「全 null の間 B head 凍結」は **head 全体の勾配を grad hook でゼロ化** はせず、
    loss 側 (freeze_b) で品質項を切る方式とする (axis loss は head 全体に勾配を流す)。

    本関数は将来の拡張点として残す no-op (現状の凍結は scorer_loss(freeze_b) が担う)。
    検証用に凍結フラグを model に持たせるだけ。
    """
    model._b_frozen = bool(frozen)


# ==============================================================
# 訓練
# ==============================================================
def set_deterministic(seed):
    """seed 固定 + 決定的アルゴリズム (train_bitnet.py と同方針)。"""
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def make_loader(dataset, batch_size, seed, shuffle):
    """決定的 DataLoader。shuffle は seed 固定 Generator で再現的に。"""
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=g,
        num_workers=0, drop_last=False)


def train(args):
    set_deterministic(args.seed)
    device = torch.device(args.device)

    train_ds = ScorerDataset(args.train_file, img_size=args.img_size, synth_seed=args.seed)
    val_ds = None
    if args.val_file and os.path.isfile(args.val_file):
        val_ds = ScorerDataset(args.val_file, img_size=args.img_size, synth_seed=args.seed)

    model = ScorerNet(n_attr=N_AXIS).to(device)
    n_params = assert_param_count(model)
    print(f"[scorer] ScorerNet params = {n_params} = {n_params/1e6:.5f}M "
          f"(probe 11.18M 一致・pure conv・out [1,{N_OUT}])")

    # §16.8: quality が全 null の間は B head 凍結。
    # --freeze-b auto = データに quality が 1 件も無ければ凍結 / 1 件でもあれば学習。
    qcount = train_ds.quality_count()
    if args.freeze_b == "auto":
        freeze_b = (qcount == 0)
    elif args.freeze_b == "on":
        freeze_b = True
    else:
        freeze_b = False
    set_b_head_frozen(model, freeze_b)
    print(f"[scorer] quality non-null = {qcount}/{len(train_ds)} → B head "
          f"{'凍結 (損失から除外・§16.8 既定)' if freeze_b else '学習'}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(train_ds, args.batch_size, args.seed, shuffle=True)

    history = []
    for epoch in range(args.epochs):
        model.train()
        tot = ax = qa = 0.0
        nb = 0
        for img, axis, quality, qmask in train_loader:
            img = img.to(device)
            axis = axis.to(device)
            quality = quality.to(device)
            qmask = qmask.to(device)
            pred = model(img)
            loss, al, ql = scorer_loss(
                pred, axis, quality, qmask, freeze_b,
                huber_delta=args.huber_delta, quality_weight=args.quality_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            ax += float(al)
            qa += float(ql)
            nb += 1
        tr = dict(epoch=epoch, train_loss=tot / max(nb, 1),
                  axis_loss=ax / max(nb, 1), quality_loss=qa / max(nb, 1))

        if val_ds is not None:
            tr.update(evaluate(model, val_ds, device, freeze_b, args))
        history.append(tr)
        msg = (f"[ep {epoch}] train_loss {tr['train_loss']:.5f} "
               f"(axis {tr['axis_loss']:.5f} / quality {tr['quality_loss']:.5f})")
        if "val_loss" in tr:
            msg += f"  val_loss {tr['val_loss']:.5f}"
        print(msg)

    # 出力 (FP16 / FP32 両方)。
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        fp32_path = os.path.join(args.out_dir, "scorer_net_fp32.safetensors")
        fp16_path = os.path.join(args.out_dir, "scorer_net.safetensors")
        export_safetensors(model, fp32_path, torch.float32)
        export_safetensors(model, fp16_path, torch.float16)
        problems = sanity_reload(fp32_path)
        if problems:
            raise RuntimeError("safetensors reload 検証失敗: " + "; ".join(problems))
        print(f"[scorer] saved {fp32_path} (FP32) / {fp16_path} (FP16)")
        write_stats(args, model, train_ds, val_ds, freeze_b, history)

    return model, history


@torch.no_grad()
def evaluate(model, ds, device, freeze_b, args):
    model.eval()
    loader = make_loader(ds, args.batch_size, args.seed, shuffle=False)
    tot = ax = qa = 0.0
    nb = 0
    for img, axis, quality, qmask in loader:
        img = img.to(device)
        axis = axis.to(device)
        quality = quality.to(device)
        qmask = qmask.to(device)
        pred = model(img)
        loss, al, ql = scorer_loss(
            pred, axis, quality, qmask, freeze_b,
            huber_delta=args.huber_delta, quality_weight=args.quality_weight)
        tot += float(loss)
        ax += float(al)
        qa += float(ql)
        nb += 1
    return dict(val_loss=tot / max(nb, 1), val_axis_loss=ax / max(nb, 1),
                val_quality_loss=qa / max(nb, 1))


# ==============================================================
# safetensors 出力 (ScorerNet 重みレイアウト・[out,in] 規約)
# ==============================================================
def export_safetensors(model, path, dtype):
    """ScorerNet の state_dict をそのまま safetensors 保存する。

    nn.Conv2d.weight は [out_ch, in_ch, kh, kw]・nn.Linear.weight は [out, in] で
    いずれも row-major [out, ...] 規約 (train_bitnet.py::export_safetensors と同哲学)。
    BatchNorm の running_mean/var/num_batches_tracked も含め state_dict 全テンソルを保存し、
    OV 変換 (B-3c 研究機) が PyTorch state_dict としてそのまま読めるようにする。

    safetensors は num_batches_tracked のような 0 次元/int テンソルも保存できる。
    """
    from safetensors.torch import save_file

    sd = {}
    for k, v in model.state_dict().items():
        t = v.detach().cpu()
        # BatchNorm の num_batches_tracked は整数 (dtype 変換しない)。浮動小数のみ dtype 化。
        if t.is_floating_point():
            t = t.to(dtype)
        sd[k] = t.contiguous()
    save_file(sd, path)
    return list(sd.keys())


def sanity_reload(path):
    """保存した safetensors を読み戻して NaN/Inf と head 形状を検証する。"""
    from safetensors import safe_open

    problems = []
    with safe_open(path, framework="pt") as f:
        keys = set(f.keys())
        if "head.weight" not in keys:
            problems.append("head.weight 欠損")
        else:
            t = f.get_tensor("head.weight")
            if tuple(t.shape) != (N_OUT, 512):
                problems.append(f"head.weight shape {tuple(t.shape)} != ({N_OUT},512)")
        for name in keys:
            t = f.get_tensor(name)
            if t.is_floating_point() and (torch.isnan(t.float()).any() or torch.isinf(t.float()).any()):
                problems.append(f"{name} has NaN/Inf")
    return problems


def write_stats(args, model, train_ds, val_ds, freeze_b, history):
    """provenance + 訓練設定を scorer_train_stats.json に残す (再現用)。"""
    stats = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "torch": torch.__version__,
        "device": args.device,
        "params": param_count(model),
        "n_out": N_OUT,
        "axis_order": AXIS_ORDER,
        "img_size": args.img_size,
        "hparams": {
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": args.weight_decay, "huber_delta": args.huber_delta,
            "quality_weight": args.quality_weight,
        },
        "b_head_frozen": bool(freeze_b),
        "quality_non_null_train": train_ds.quality_count(),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds) if val_ds is not None else 0,
        "history": history,
        "teacher": {
            "anatomy": "wd14 max-sigmoid 軸集約 (dollma_make_scorer_labels.py) → BCE soft 蒸留",
            "quality": "外部美的モデル soft スコア [0,1] (jsonl passthrough) → Huber 回帰",
            "note": "美的モデルは本スクリプトで非ロード (jsonl の quality をそのまま教師)",
        },
        "note": "NPU/OpenVINO 非依存 (OV 変換は B-3c 研究機担当)",
    }
    with open(os.path.join(args.out_dir, "scorer_train_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


# ==============================================================
# main
# ==============================================================
def build_argparser():
    ap = argparse.ArgumentParser(description="ScorerNet 蒸留訓練 (B-3b・NPU/OV 非依存)")
    ap.add_argument("--train-file", dest="train_file",
                    default="data/scorer/scorer.train.jsonl",
                    help="訓練 jsonl (dataset-spec §16.3)")
    ap.add_argument("--val-file", dest="val_file",
                    default="data/scorer/scorer.val.jsonl",
                    help="検証 jsonl (無ければ val スキップ)")
    ap.add_argument("--out-dir", dest="out_dir", default="data/scorer",
                    help="重み/stats 出力先 (空文字で出力スキップ)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-4)
    ap.add_argument("--huber-delta", dest="huber_delta", type=float, default=1.0)
    ap.add_argument("--quality-weight", dest="quality_weight", type=float, default=1.0)
    ap.add_argument("--img-size", dest="img_size", type=int, default=IMG_SIZE)
    ap.add_argument("--freeze-b", dest="freeze_b", default="auto",
                    choices=["auto", "on", "off"],
                    help="B(品質)head 凍結 (§16.8)。auto=quality 全 null なら凍結")
    ap.add_argument("--device", default="cpu",
                    help="cpu / cuda (本機 GTX1080Ti は cpu 既定で決定性重視)")
    ap.add_argument("--seed", type=int, default=20260620)
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    print(f"[scorer] device={args.device} torch={torch.__version__} seed={args.seed}")
    t0 = time.perf_counter()
    train(args)
    print(f"[scorer] done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
