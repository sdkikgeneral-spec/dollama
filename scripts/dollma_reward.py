# -*- coding: utf-8 -*-
"""dollma_reward.py — Phase 4 施策 F (品質 FB ループ) / F-0a:
ScorerNet の出力から LM 強化学習用の **スカラ報酬** を計算する純関数。

閉路: LM がタグプロンプト生成 → SDXL が画像生成 → ScorerNet が採点 → 本関数で報酬化
      → (F-0b 以降で) 報酬を使い LM を fine-tune。
本モジュールは「採点 → 報酬」の一段だけを担い、**OV/torch 非依存・配列演算のみ**で書く
(開発機でも単体テスト可。dollama の哲学「採点グルーは OV ガード外でテスト実走」と同方針)。

報酬設計 (PL 確定・anatomy のみ):
  ScorerNet の anatomy 8 軸 (AnomalyAxis 順 0 Hands / 1 Limbs / 2 Head / 3 Eyes /
  4 Ears / 5 Mouth / 6 Digits / 7 GlobalAnatomy) は各 [0,1] sigmoid 確率で **高い=異常**。
  - **worst-axis 主**: reward = -max_i(axes[i])。2D キャラは手 1 本の破綻で全損するため、
    mean ではなく max 集約が正しい (src/server/scoring_postprocess.hpp の
    collect_anomalous_axes が「異常度の高い軸を拾う」のと同じ哲学)。
  - **mean 従**: 同点 (worst が等しい) プロンプト同士を識別するため、mean を極小の重みで
    混ぜる。worst の支配は崩さず、worst が同点のとき「他軸の異常が少ない方」を高報酬にする。
    凸結合なので境界は正確に保たれる: 全 0 軸 → 0.0 / 全 1 軸 → -1.0
    (両端では max==mean ゆえ重みに依らず一致する)。

quality head の前方互換 (重要):
  ScorerNet index0 = 美的 quality だが **B-3b で凍結中** (anatomy 8 軸のみ訓練・quality=null)。
  よって現状の報酬は anatomy のみ。美的品質モデル waifu_scorer_v4 は **apache-2.0 で採用可** で、
  quality 有効化は単なる実行ギャップ (ライセンスの壁ではない)。将来 quality head が
  有効化されたら本関数の `quality` 引数 (現状 None) が非 None になり、美的スコアを報酬へ
  混ぜる経路 (下記 QUALITY_WEIGHT) が発火する。今は anatomy 主・将来 quality 混合。
"""

# 同点処理用の mean の重み (極小)。worst の支配を崩さない範囲で他軸の異常量を反映する。
# 凸結合 reward = -((1-w)*worst + w*mean) のため、全 0/全 1 の境界は w に依らず正確。
MEAN_TIE_WEIGHT = 0.05

# 将来 quality head 凍結解除時に美的スコアを報酬へ混ぜる重み (現状は未発火)。
# quality は [0,1] で高い=良。anatomy_reward は [-1,0] で 0=良。両者の「良=大」方向を
# 揃えるため quality を (quality-1.0) で [-1,0] へ写してから凸結合する。
QUALITY_WEIGHT = 0.3


def reward_from_scorer(axes, quality=None):
    """ScorerNet の anatomy 8 軸 (sigmoid 確率) から LM 強化学習用スカラ報酬を返す。

    引数:
      axes    : list[float] — anatomy 軸ごとの異常確率 [0,1] (高い=異常)。AnomalyAxis 順。
                通常は長さ 8。空リストは ValueError。
      quality : float | None — 美的 quality 確率 [0,1] (高い=良)。
                **現状は常に None** (B-3b で quality head 凍結中)。
                非 None のとき美的スコアを報酬へ混ぜる前方互換経路が発火する (将来用)。

    返り値:
      float — 報酬。値域 [-1, 0] (anatomy のみのとき)。0 が最良 (異常なし)、-1 が最悪 (全破綻)。

    集約:
      anatomy_reward = -((1-MEAN_TIE_WEIGHT)*max(axes) + MEAN_TIE_WEIGHT*mean(axes))
        worst-axis 主・mean 従。境界: 全 0 → 0.0 / 全 1 → -1.0 (max==mean ゆえ正確)。
      quality が非 None のとき:
        reward = (1-QUALITY_WEIGHT)*anatomy_reward + QUALITY_WEIGHT*(quality-1.0)
        (美的が良いほど報酬を 0 方向へ押し上げる。anatomy のみは quality=None 経路)。

    集約器は差し替え可能 (worst-axis = 2D 全損という設計判断の表明であって不可侵ではない)。
    """
    if not axes:
        raise ValueError("axes が空です (ScorerNet anatomy 8 軸を渡すこと)")

    worst = max(axes)
    mean = sum(axes) / len(axes)

    # worst-axis 主・mean 従の凸結合。全 0/全 1 の境界は重みに依らず正確。
    anatomy_reward = -((1.0 - MEAN_TIE_WEIGHT) * worst + MEAN_TIE_WEIGHT * mean)

    if quality is None:
        # 現状の本線: anatomy のみ (quality head 凍結中)。
        return anatomy_reward

    # 将来用 (quality 凍結解除時): 美的スコアを混ぜる。quality を [-1,0] へ写して凸結合。
    return (1.0 - QUALITY_WEIGHT) * anatomy_reward + QUALITY_WEIGHT * (quality - 1.0)
