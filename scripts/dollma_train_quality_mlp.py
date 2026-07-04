# -*- coding: utf-8 -*-
"""dollama Phase 4 Model B / Package C — 自作 quality MLP を CLIP 空間で waifu 蒸留。

Package A で判明した「生ピクセル ResNet では quality 蒸留が負相関/不安定」
(val corr -0.25〜+0.40) を、CLIP 空間へ移して立て直す。教師 waifu も
CLIP ViT-L embed → MLP なので、生徒 (自作 MLP) を同じ CLIP 空間に置けば
蒸留は素直に通るはず。

入力: data/scorer/scorer.{train,val}.jsonl (Package B で clip_image_embed[768] 追記済)。
  - 特徴 = clip_image_embed[768] (L2 正規化済)。
  - 教師 = quality (Step① renorm 済み [0,1]・std0.298)。
出力: data/scorer/quality_mlp{,_fp32}.safetensors ([out,in] 規約・FP16/FP32)。
provenance: data/scorer/quality_mlp_stats.json。

主指標: val pearson corr(sigmoid(pred), teacher quality)。ゲート = 明確に正 (目標 +0.5+)。

再現性: seed 固定・決定的 (train_scorer.py / train_bitnet.py と同方針)。
train/val は既存 jsonl split をそのまま使う (162/18)。

これは dollama 自作モデル (waifu 直乗せでなく自作 MLP へ蒸留)。出荷推論は
CLIP embed → 本 MLP → sigmoid で quality スカラ [0,1] を得る。
"""

import argparse
import json
import os
import random
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 768


# ==============================================================
# 自作 quality MLP (CLIP embed[768] → quality logit スカラ)
# ==============================================================
class QualityMLP(nn.Module):
    """CLIP image embed[768] → 品質スカラ logit (sigmoid 前)。

    hidden はリストで段階的に容量を上げられる (実験軸: 最小 [256,64] から)。
    各隠れ層 = Linear → ReLU → Dropout。出力 = Linear(・,1)。
    """

    def __init__(self, hidden=(256, 64), dropout=0.1, in_dim=EMBED_DIM):
        super().__init__()
        self.hidden = list(hidden)
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # [B]


def param_count(model):
    return sum(p.numel() for p in model.parameters())


# ==============================================================
# データ (clip_image_embed[768] + quality をメモリに読む・小標本)
# ==============================================================
def load_split(path):
    embeds, quals = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            e = r.get("clip_image_embed")
            if e is None or len(e) != EMBED_DIM:
                raise ValueError(f"clip_image_embed[{EMBED_DIM}] が無い: {r.get('image')}")
            q = r.get("quality")
            if q is None:
                continue  # 教師なしは学習対象外
            embeds.append(e)
            quals.append(float(q))
    x = torch.tensor(embeds, dtype=torch.float32)
    y = torch.tensor(quals, dtype=torch.float32)
    return x, y


# ==============================================================
# 相関 (pearson) — 主指標
# ==============================================================
def pearson(a, b):
    a = a.detach().float()
    b = b.detach().float()
    am, bm = a.mean(), b.mean()
    av, bv = a - am, b - bm
    denom = (av.pow(2).sum().sqrt() * bv.pow(2).sum().sqrt())
    if float(denom) == 0.0:
        return 0.0
    return float((av * bv).sum() / denom)


# ==============================================================
# 訓練
# ==============================================================
def set_deterministic(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(x))
    corr = pearson(pred, y)
    return dict(
        val_corr=corr,
        pred_std=float(pred.std(unbiased=False)),
        pred_min=float(pred.min()), pred_max=float(pred.max()),
        pred_mean=float(pred.mean()),
        loss=float(F.mse_loss(pred, y)),
    )


def oof_cv_corr(x, y, hidden, dropout, wd, lr, epochs, k=5, seed=20260620):
    """全 180 の k-fold out-of-fold 予測で robust な corr を測る。

    val=18 の pearson は分散が巨大 (n=18 真値0 の 95%CI ~= ±0.47) ゆえ主指標には脆い。
    train+val を pool した out-of-fold 予測で corr を取ると n=180 の安定推定になる。
    戻り値: (oof_corr, oof_pred_std, oof_pred_min, oof_pred_max)。
    """
    n = len(y)
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    oof = torch.zeros(n)
    for f in range(k):
        va_idx = perm[f::k]
        mask = torch.ones(n, dtype=torch.bool)
        mask[va_idx] = False
        tr_idx = mask.nonzero().squeeze(-1)
        set_deterministic(seed + f)
        m = QualityMLP(hidden=hidden, dropout=dropout)
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
        for _ in range(epochs):
            m.train()
            F.mse_loss(torch.sigmoid(m(x[tr_idx])), y[tr_idx]).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        m.eval()
        with torch.no_grad():
            oof[va_idx] = torch.sigmoid(m(x[va_idx]))
    return (pearson(oof, y), float(oof.std(unbiased=False)),
            float(oof.min()), float(oof.max()))


def train(args):
    set_deterministic(args.seed)
    device = torch.device(args.device)

    xtr, ytr = load_split(args.train_file)
    xva, yva = load_split(args.val_file)
    xtr, ytr, xva, yva = (t.to(device) for t in (xtr, ytr, xva, yva))
    print(f"[qmlp] train {tuple(xtr.shape)} / val {tuple(xva.shape)}  "
          f"teacher std train={float(ytr.std(unbiased=False)):.4f} val={float(yva.std(unbiased=False)):.4f}")

    hidden = tuple(int(h) for h in args.hidden.split(",") if h != "")
    model = QualityMLP(hidden=hidden, dropout=args.dropout).to(device)
    n_params = param_count(model)
    print(f"[qmlp] QualityMLP hidden={hidden} params={n_params} ({n_params/1e6:.4f}M)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # フルバッチ (162 サンプル・小標本ゆえ mini-batch 不要・決定的)。
    g = torch.Generator()
    g.manual_seed(args.seed)
    history = []
    best = None
    for epoch in range(args.epochs):
        model.train()
        pred = torch.sigmoid(model(xtr))
        if args.loss == "huber":
            loss = F.huber_loss(pred, ytr, delta=args.huber_delta)
        else:
            loss = F.mse_loss(pred, ytr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        ev = evaluate(model, xva, yva)
        tr_corr = pearson(torch.sigmoid(model(xtr)).detach(), ytr)
        rec = dict(epoch=epoch, train_loss=float(loss.detach()),
                   train_corr=tr_corr, **ev)
        history.append(rec)
        if best is None or ev["val_corr"] > best["val_corr"]:
            best = rec
        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs - 1:
            print(f"[ep {epoch:>4}] loss {float(loss):.5f} "
                  f"train_corr {tr_corr:+.4f} val_corr {ev['val_corr']:+.4f} "
                  f"pred_std {ev['pred_std']:.4f} range[{ev['pred_min']:.3f},{ev['pred_max']:.3f}]")

    final = history[-1]
    print(f"\n[qmlp] final  val_corr {final['val_corr']:+.4f} | "
          f"best@ep{best['epoch']} val_corr {best['val_corr']:+.4f}")
    print(f"[qmlp] pred std {final['pred_std']:.4f} range [{final['pred_min']:.3f},{final['pred_max']:.3f}] "
          f"(teacher val range [{float(yva.min()):.3f},{float(yva.max()):.3f}] std {float(yva.std(unbiased=False)):.4f})")

    # robust 主指標: train+val pool の 5-fold out-of-fold corr (n=180・val=18 のノイズ回避)。
    oof = None
    if args.cv_report:
        xall = torch.cat([xtr, xva]).cpu()
        yall = torch.cat([ytr, yva]).cpu()
        oc, ostd, omin, omax = oof_cv_corr(
            xall, yall, hidden, args.dropout, args.weight_decay, args.lr,
            args.epochs, k=args.cv_folds, seed=args.seed)
        oof = dict(oof_corr=oc, folds=args.cv_folds, n=len(yall),
                   pred_std=ostd, pred_min=omin, pred_max=omax)
        print(f"[qmlp] OOF {args.cv_folds}-fold corr (n={len(yall)}) {oc:+.4f} "
              f"pred_std {ostd:.4f} range[{omin:.3f},{omax:.3f}]  <- robust 主指標")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        fp32 = os.path.join(args.out_dir, "quality_mlp_fp32.safetensors")
        fp16 = os.path.join(args.out_dir, "quality_mlp.safetensors")
        export_safetensors(model, fp32, torch.float32)
        export_safetensors(model, fp16, torch.float16)
        problems = sanity_reload(fp32, model)
        if problems:
            raise RuntimeError("safetensors reload 検証失敗: " + "; ".join(problems))
        print(f"[qmlp] saved {fp32} / {fp16}")
        write_stats(args, model, hidden, n_params, history, best, final,
                    float(ytr.std(unbiased=False)), float(yva.std(unbiased=False)),
                    (float(yva.min()), float(yva.max())), oof)

    return model, history, best


# ==============================================================
# safetensors 出力 ([out,in] 規約・FP16/FP32)
# ==============================================================
def export_safetensors(model, path, dtype):
    from safetensors.torch import save_file
    sd = {}
    for k, v in model.state_dict().items():
        t = v.detach().cpu()
        if t.is_floating_point():
            t = t.to(dtype)
        sd[k] = t.contiguous()
    save_file(sd, path)
    return list(sd.keys())


def sanity_reload(path, ref_model):
    from safetensors import safe_open
    problems = []
    with safe_open(path, framework="pt") as f:
        keys = set(f.keys())
        ref_keys = set(ref_model.state_dict().keys())
        if keys != ref_keys:
            problems.append(f"key 集合不一致 {keys ^ ref_keys}")
        for name in keys:
            t = f.get_tensor(name)
            if t.is_floating_point() and (torch.isnan(t.float()).any() or torch.isinf(t.float()).any()):
                problems.append(f"{name} has NaN/Inf")
    return problems


def write_stats(args, model, hidden, n_params, history, best, final,
                teacher_std_train, teacher_std_val, teacher_val_range, oof=None):
    stats = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": "QualityMLP (dollama 自作・CLIP embed[768] → quality logit)",
        "note": "Package C: 生ピクセル ScorerNet の quality 蒸留 (負相関/不安定) を CLIP 空間で立て直す",
        "seed": args.seed,
        "torch": torch.__version__,
        "device": args.device,
        "arch": {"in_dim": EMBED_DIM, "hidden": list(hidden), "dropout": args.dropout,
                 "out": 1, "activation": "ReLU"},
        "param_count": n_params,
        "hparams": {"epochs": args.epochs, "lr": args.lr,
                    "weight_decay": args.weight_decay, "loss": args.loss,
                    "huber_delta": args.huber_delta},
        "teacher": {"source": "jsonl quality (Step① renorm 済み [0,1])",
                    "std_train": round(teacher_std_train, 4),
                    "std_val": round(teacher_std_val, 4),
                    "val_range": [round(teacher_val_range[0], 4), round(teacher_val_range[1], 4)]},
        "result": {
            "final_val_corr": round(final["val_corr"], 4),
            "best_val_corr": round(best["val_corr"], 4),
            "best_epoch": best["epoch"],
            "final_pred_std": round(final["pred_std"], 4),
            "final_pred_range": [round(final["pred_min"], 4), round(final["pred_max"], 4)],
            "final_train_corr": round(final["train_corr"], 4),
        },
        "oof_cv": (dict(oof_corr=round(oof["oof_corr"], 4), folds=oof["folds"], n=oof["n"],
                        pred_std=round(oof["pred_std"], 4),
                        pred_range=[round(oof["pred_min"], 4), round(oof["pred_max"], 4)],
                        note="train+val pool の out-of-fold corr = robust 主指標 (val=18 のノイズ回避)")
                   if oof else None),
        "vs_raw_pixel": "生ピクセル ScorerNet quality 蒸留は val corr -0.25〜+0.40 で不安定・逆相関だった",
        "history_tail": history[-5:],
        "inference": "CLIP ViT-L embed[768] (L2 正規化) → QualityMLP → sigmoid → quality [0,1]",
    }
    with open(os.path.join(args.out_dir, "quality_mlp_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


# ==============================================================
# main
# ==============================================================
def build_argparser():
    ap = argparse.ArgumentParser(description="自作 quality MLP を CLIP 空間で waifu 蒸留 (Package C)")
    ap.add_argument("--train-file", dest="train_file", default="data/scorer/scorer.train.jsonl")
    ap.add_argument("--val-file", dest="val_file", default="data/scorer/scorer.val.jsonl")
    ap.add_argument("--out-dir", dest="out_dir", default="data/scorer",
                    help="空文字で出力スキップ (スイープ時)")
    ap.add_argument("--hidden", default="256,64", help="隠れ層カンマ区切り (最小 256,64)")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-3)
    ap.add_argument("--loss", default="mse", choices=["mse", "huber"])
    ap.add_argument("--huber-delta", dest="huber_delta", type=float, default=0.1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cv-report", dest="cv_report", action="store_true", default=True,
                    help="robust 主指標として全 180 の out-of-fold corr を計算 (既定 on)")
    ap.add_argument("--no-cv-report", dest="cv_report", action="store_false")
    ap.add_argument("--cv-folds", dest="cv_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260620)
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    print(f"[qmlp] device={args.device} torch={torch.__version__} seed={args.seed}")
    t0 = time.perf_counter()
    train(args)
    print(f"[qmlp] done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
