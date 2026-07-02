// BackendImageGenerator — IDiffusionBackend を IImageGenerator へ橋渡しする汎用アダプタ。
//
// ----------------------------------------------------------------
// 設計意図:
//   従来の Txt2ImgGenerator は「SDXL の encode + 拡散」と「解像度 reject / seed 決定 /
//   採点ログ / matting PNG 化」という 2 種の責務を 1 クラスに抱えていた。プラグイン化に
//   あたり前者を IDiffusionBackend (アーキ固有) へ切り出し、後者 (アーキ非依存の共通後処理)
//   を本クラスに集約する。新アーキ backend を足しても本クラスは無改修で使い回せる。
//
//   本ヘッダは IDiffusionBackend / IMatter / IScorer (いずれも純 cpp interface) 越しに
//   呼ぶため CUDA/OV 非依存に保たれ、ガード不要で任意の cpp TU から include できる。
//   set_matter / set_scorer はここに集約する (従来 Txt2ImgGenerator の member を移設)。
// ----------------------------------------------------------------
#pragma once

#include <cstdint>
#include <ctime>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "server/diffusion_backend.hpp"
#include "server/generator.hpp"
#include "server/matter_runner.hpp"
#include "server/matting_postprocess.hpp"
#include "server/scorer_runner.hpp"
#include "server/scoring_postprocess.hpp"

namespace dollama
{

// IDiffusionBackend を保持し、共通後処理 (解像度 reject / seed / 採点ログ / matting PNG 化)
// を被せて IImageGenerator を実装する。
class BackendImageGenerator : public IImageGenerator
{
public:
    // backend を受け取り所有する。nullptr は構築失敗として拒否する
    //   (呼び出し側 = build_image_generator は make_backend の nullptr を先に検知して
    //    段2/3 へ落ちるため、本クラスへ nullptr が渡ることは通常ない)。
    explicit BackendImageGenerator(std::unique_ptr<IDiffusionBackend> backend)
        : backend_(std::move(backend))
    {
        if (!backend_)
        {
            throw std::runtime_error(
                "BackendImageGenerator: backend が nullptr (make_backend 失敗)");
        }
    }

    BackendImageGenerator(const BackendImageGenerator&)            = delete;
    BackendImageGenerator& operator=(const BackendImageGenerator&) = delete;

    // M-6: マッティング器を後付け注入する (build_image_generator が DI 後に 1 回呼ぶ)。
    void set_matter(std::unique_ptr<IMatter> m) override
    {
        matter_ = std::move(m);
    }

    // B-5-3: スコアラを後付け注入する (build_image_generator が DI 後に 1 回呼ぶ)。
    void set_scorer(std::unique_ptr<IScorer> s) override
    {
        scorer_ = std::move(s);
    }

    // 1 リクエスト分を backend で生成し PNG バイト列を返す。
    GenResult generate(const GenRequest& req) override
    {
        // --- 解像度: 当面 1024x1024 固定。1024 以外が来たら reject する ---
        //   (従来 Txt2ImgGenerator と同方針。clamp で黙って差し替えず明示的に弾く。)
        if ((req.width != 0 && req.width != 1024) ||
            (req.height != 0 && req.height != 1024))
        {
            throw std::runtime_error(
                "BackendImageGenerator: width/height は 1024 のみ対応 (SDXL 1024x1024 固定)");
        }

        // --- ステップ数: req.steps をそのまま (1 未満は 20) ---
        const int steps = (req.steps > 0) ? req.steps : 20;

        // --- seed: GenRequest に seed フィールドが無いため内部で決める (時刻ベース) ---
        const uint64_t seed = static_cast<uint64_t>(std::time(nullptr));

        // --- 拡散 backend 実行 (CFG は backend 側の既定に委譲: cfg=0 を渡す) ---
        //   backend が cfg<=0 のとき自前の既定 (SDXL は 7.5) を使う契約。
        std::vector<uint8_t> rgb;
        int w = 0, h = 0;
        backend_->generate(req.prompt, req.negative_prompt, steps, seed,
                           /*cfg=*/0.0f, /*w=*/1024, /*h=*/1024, rgb, w, h);

        // --- B-5-3: スコアラ注入時のみ生キャラ rgb を採点 (matting 合成前)。失敗しても
        //     生成は止めない (score_image_safe が例外/nullptr を握る)。結果は現状ログのみ
        //     (消費者 F 未着手ゆえ GenResult 添付/API 露出はしない=死にコード回避)。
        ScoringOutcome sc = score_image_safe(scorer_.get(), rgb, w, h);
        if (sc.scored)
        {
            std::clog << "[scorer] quality=" << sc.result.quality << " anomaly:";
            auto flags = collect_anomalous_axes(sc.result);
            if (flags.empty())
            {
                std::clog << " none";
            }
            for (const auto& f : flags)
            {
                std::clog << ' ' << axis_name(f.axis) << '(' << f.score << ')';
            }
            std::clog << '\n';
        }

        // --- HWC uint8 RGB → PNG (M-6: matting ON なら透過 PNG・OFF/無なら不透明) ---
        GenResult out;
        out.png_bytes = encode_png_maybe_transparent(
            req.matting ? matter_.get() : nullptr, rgb, w, h);
        return out;
    }

    // モデル識別子 (GET /v1/models 用)。backend に委譲する。
    std::string model_id() const override
    {
        return backend_->model_id();
    }

private:
    std::unique_ptr<IDiffusionBackend> backend_;
    std::unique_ptr<IMatter>           matter_; // M-6: set_matter で注入 (既定 nullptr = 不透明)
    std::unique_ptr<IScorer>           scorer_; // B-5-3: set_scorer で注入 (既定 nullptr = 不採点)
};

} // namespace dollama
