// HTTP サーバー (OpenAI Images 互換) 単体テスト + 往復ベンチ
//
// testing.md 形式: 各検証は if (!cond) { cerr; return false; }、
// main で ok = test_xxx() && ok 集約、成功は PASSED / 最後に ALL PASSED。
//
// 純 C++ (CUDA / OpenVINO 非依存)。StubGenerator を注入し httplib::Server を
// バックグラウンドスレッドで 127.0.0.1 の OS 自動割当ポート (bind_to_any_port)
// で起動 → wait_until_ready → httplib::Client で自己リクエスト → 検証。
//
// G-8k F1: 最後に「生成の直列化ゲート」を持つ。cpp-httplib はスレッドプールから
// リクエストを並行ディスパッチするが、生成経路はグローバル共有の bump アリーナ
// (device_arena) を握るため同時 2 本を許さない。api.cpp のファネルで直列化されて
// いることを、計装スタブの同時実行数カウンタで実測して落とす。
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "server/api.hpp"
#include "server/base64.hpp"
#include "server/generator.hpp"
#include "server/png.hpp"
#include "server/stub_generator.hpp"

namespace dollama
{

using json = nlohmann::json;

// テスト用にサーバを bind_to_any_port で起動し、ポートと制御を保持する。
// 生成器は外から注入する (直列化ゲートでは計装版スタブを差し込むため)。
struct ServerFixture
{
    httplib::Server svr;
    std::thread th;
    int port = 0;

    explicit ServerFixture(IImageGenerator& gen)
    {
        register_routes(svr, gen);
        port = svr.bind_to_any_port("127.0.0.1");
        th = std::thread([this] { svr.listen_after_bind(); });
        svr.wait_until_ready();
    }

    ~ServerFixture()
    {
        svr.stop();
        if (th.joinable())
        {
            th.join();
        }
    }
};

// ----------------------------------------------------------------
// PNG エンコード / サイズ読み取りの往復
// ----------------------------------------------------------------
static bool test_png_roundtrip()
{
    const int w = 17, h = 9;
    std::vector<uint8_t> rgb(static_cast<size_t>(w) * h * 3);
    for (size_t i = 0; i < rgb.size(); ++i)
    {
        rgb[i] = static_cast<uint8_t>(i & 0xFF);
    }
    std::vector<uint8_t> png = encode_png_rgb8(rgb, w, h);

    // シグネチャ確認
    const uint8_t sig[8] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    for (int i = 0; i < 8; ++i)
    {
        if (png[i] != sig[i])
        {
            std::cerr << "[png] シグネチャ不一致 byte " << i << "\n";
            return false;
        }
    }
    int rw = 0, rh = 0;
    if (!read_png_size(png, rw, rh) || rw != w || rh != h)
    {
        std::cerr << "[png] IHDR サイズ不一致 got " << rw << "x" << rh << "\n";
        return false;
    }
    std::cout << "  [png] roundtrip OK (" << png.size() << " bytes)\n";
    return true;
}

// ----------------------------------------------------------------
// base64 エンコード / デコードの往復
// ----------------------------------------------------------------
static bool test_base64_roundtrip()
{
    for (size_t len = 0; len < 64; ++len)
    {
        std::vector<uint8_t> in(len);
        for (size_t i = 0; i < len; ++i)
        {
            in[i] = static_cast<uint8_t>((i * 37 + 11) & 0xFF);
        }
        const std::string enc = base64_encode(in);
        const std::vector<uint8_t> dec = base64_decode(enc);
        if (dec != in)
        {
            std::cerr << "[base64] 往復不一致 len=" << len << "\n";
            return false;
        }
    }
    std::cout << "  [base64] roundtrip OK (len 0..63)\n";
    return true;
}

// ----------------------------------------------------------------
// GET / → 200 + text/html + HTML 内容 (簡易 Web UI)
// ----------------------------------------------------------------
static bool test_index_page(ServerFixture& fx)
{
    httplib::Client cli("127.0.0.1", fx.port);
    auto res = cli.Get("/");
    if (!res || res->status != 200)
    {
        std::cerr << "[index] status != 200 (got "
                  << (res ? res->status : -1) << ")\n";
        return false;
    }
    const std::string ctype = res->get_header_value("Content-Type");
    if (ctype.find("text/html") == std::string::npos)
    {
        std::cerr << "[index] Content-Type に text/html が無い: " << ctype << "\n";
        return false;
    }
    if (res->body.find("<!DOCTYPE html") == std::string::npos)
    {
        std::cerr << "[index] body に <!DOCTYPE html が無い\n";
        return false;
    }
    if (res->body.find("/v1/images/generations") == std::string::npos)
    {
        std::cerr << "[index] body に /v1/images/generations が無い\n";
        return false;
    }
    std::cout << "  [index] 200 OK text/html (" << res->body.size() << " bytes)\n";
    return true;
}

// ----------------------------------------------------------------
// GET /health → 200 + status ok
// ----------------------------------------------------------------
static bool test_health(ServerFixture& fx)
{
    httplib::Client cli("127.0.0.1", fx.port);
    auto res = cli.Get("/health");
    if (!res || res->status != 200)
    {
        std::cerr << "[health] status != 200\n";
        return false;
    }
    auto j = json::parse(res->body, nullptr, false);
    if (j.is_discarded() || j.value("status", "") != "ok")
    {
        std::cerr << "[health] body 不正: " << res->body << "\n";
        return false;
    }
    std::cout << "  [health] 200 OK\n";
    return true;
}

// ----------------------------------------------------------------
// GET /v1/models → 200 + data[0].id
// ----------------------------------------------------------------
static bool test_models(ServerFixture& fx)
{
    httplib::Client cli("127.0.0.1", fx.port);
    auto res = cli.Get("/v1/models");
    if (!res || res->status != 200)
    {
        std::cerr << "[models] status != 200\n";
        return false;
    }
    auto j = json::parse(res->body, nullptr, false);
    if (j.is_discarded() || !j.contains("data") || !j["data"].is_array() ||
        j["data"].empty() || !j["data"][0].contains("id"))
    {
        std::cerr << "[models] body 不正: " << res->body << "\n";
        return false;
    }
    std::cout << "  [models] 200 OK id=" << j["data"][0]["id"].get<std::string>() << "\n";
    return true;
}

// ----------------------------------------------------------------
// POST /v1/images/generations 正常系 → 200 + PNG シグネチャ + IHDR サイズ一致
// + 往復レイテンシ計測
// ----------------------------------------------------------------
static bool test_generations_ok(ServerFixture& fx)
{
    httplib::Client cli("127.0.0.1", fx.port);
    json req = {
        {"prompt", "1girl, silver hair, magical girl"},
        {"n", 1},
        {"size", "256x192"},
        {"response_format", "b64_json"},
    };

    const auto t0 = std::chrono::steady_clock::now();
    auto res = cli.Post("/v1/images/generations", req.dump(), "application/json");
    const auto t1 = std::chrono::steady_clock::now();

    if (!res || res->status != 200)
    {
        std::cerr << "[gen] status != 200 (got "
                  << (res ? res->status : -1) << ")\n";
        return false;
    }
    auto j = json::parse(res->body, nullptr, false);
    if (j.is_discarded() || !j.contains("data") || !j["data"].is_array() ||
        j["data"].empty() || !j["data"][0].contains("b64_json"))
    {
        std::cerr << "[gen] body 不正\n";
        return false;
    }

    const std::string b64 = j["data"][0]["b64_json"].get<std::string>();
    const std::vector<uint8_t> png = base64_decode(b64);

    const uint8_t sig[8] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    if (png.size() < 8)
    {
        std::cerr << "[gen] PNG が短すぎる\n";
        return false;
    }
    for (int i = 0; i < 8; ++i)
    {
        if (png[i] != sig[i])
        {
            std::cerr << "[gen] PNG シグネチャ不一致\n";
            return false;
        }
    }
    int rw = 0, rh = 0;
    if (!read_png_size(png, rw, rh) || rw != 256 || rh != 192)
    {
        std::cerr << "[gen] IHDR サイズが要求 size と不一致 got "
                  << rw << "x" << rh << "\n";
        return false;
    }

    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  [gen] 200 OK PNG " << png.size()
              << " bytes (256x192) 往復 " << ms << " ms\n";
    return true;
}

// ----------------------------------------------------------------
// POST /v1/images/generations 異常系: prompt 欠落 → 400
// ----------------------------------------------------------------
static bool test_generations_missing_prompt(ServerFixture& fx)
{
    httplib::Client cli("127.0.0.1", fx.port);
    json req = {{"n", 1}};
    auto res = cli.Post("/v1/images/generations", req.dump(), "application/json");
    if (!res || res->status != 400)
    {
        std::cerr << "[gen-missing] status != 400 (got "
                  << (res ? res->status : -1) << ")\n";
        return false;
    }
    auto j = json::parse(res->body, nullptr, false);
    if (j.is_discarded() || !j.contains("error") ||
        j["error"].value("type", "") != "invalid_request_error")
    {
        std::cerr << "[gen-missing] エラー JSON 形式不正: " << res->body << "\n";
        return false;
    }
    std::cout << "  [gen-missing] 400 invalid_request_error OK\n";
    return true;
}

// ----------------------------------------------------------------
// POST /v1/images/generations 異常系: 不正 JSON → 400
// ----------------------------------------------------------------
static bool test_generations_bad_json(ServerFixture& fx)
{
    httplib::Client cli("127.0.0.1", fx.port);
    auto res = cli.Post("/v1/images/generations", "{ not json", "application/json");
    if (!res || res->status != 400)
    {
        std::cerr << "[gen-badjson] status != 400 (got "
                  << (res ? res->status : -1) << ")\n";
        return false;
    }
    std::cout << "  [gen-badjson] 400 OK\n";
    return true;
}

// ================================================================
// G-8k F1: 生成の直列化ゲート
// ================================================================

// 直列化ゲート用の計装スタブ生成器。
//
//   - generate() の入口で in_flight を +1・出口で -1 し、観測した同時実行数の
//     **最大値** を残す。
//   - 中で意図的に数十 ms スリープして「重なる窓」を能動的に開ける。窓が無ければ
//     並行に呼ばれても衝突が観測できず、ゲートが空撃ちになるため。
//
// 直列化が壊れていれば max_in_flight が 2 になる。api.cpp のファネルでロックが
// 効いていれば 1 のまま。「200 が 2 本返った」では直列化を証明しないので、
// 同時実行数そのものを測る。
class SerialProbeGenerator : public IImageGenerator
{
public:
    // 生成 1 回あたりの疑似負荷 (ms)。この幅だけ「重なる窓」が開く。
    static constexpr int kBusyMs = 120;

    // ★ この下限は落とさないこと。窓幅が HTTP 往復オーバーヘッド (本開発機で 15-40ms)
    //   を十分上回っていないと、下の [2] レイテンシゲートが判別力を失って空撃ちになる。
    //   実測 (負のコントロール): kBusyMs=1 + ロック削除で回すと、同時実行 max=2 を
    //   捕まえられない回が出るうえ [2] も通ってしまい 8 回中 2 回が偽 PASS になった。
    //   コンパイル時に止めるのが唯一確実な防ぎ方なので static_assert で固定する。
    static_assert(kBusyMs >= 50,
                  "kBusyMs が小さすぎる: 重なる窓が HTTP オーバーヘッドに埋もれ、"
                  "直列化ゲートが空撃ちになる");

    GenResult generate(const GenRequest& req) override
    {
        // 入口: fetch_add の戻り + 1 が「自分を含む現在の同時実行数」。
        const int now = in_flight_.fetch_add(1) + 1;
        // 最大値を CAS で更新 (compare_exchange 失敗時 prev は現在値に更新される)。
        int prev = max_in_flight_.load();
        while (now > prev && !max_in_flight_.compare_exchange_weak(prev, now))
        {
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(kBusyMs));

        GenResult r;
        try
        {
            r = inner_.generate(req);
        }
        catch (...)
        {
            // 例外時もカウンタを戻す (以降の観測を汚さない)。
            in_flight_.fetch_sub(1);
            throw;
        }
        calls_.fetch_add(1);
        in_flight_.fetch_sub(1);
        return r;
    }

    std::string model_id() const override
    {
        return inner_.model_id();
    }

    int max_in_flight() const
    {
        return max_in_flight_.load();
    }

    int calls() const
    {
        return calls_.load();
    }

    // 観測用カウンタだけを 0 に戻す。
    // in_flight_ は **意図的にリセットしない** — これは「今まさに generate の中に
    // いる本数」を表す live カウンタで、リセットは実態の捏造になる。呼び出し時点で
    // 非 0 ならカウンタが漏れている (= 直列化ゲートは FAIL 方向に倒れる) ので安全側。
    void reset_counters()
    {
        max_in_flight_.store(0);
        calls_.store(0);
    }

private:
    StubGenerator    inner_;
    std::atomic<int> in_flight_{0};
    std::atomic<int> max_in_flight_{0};
    std::atomic<int> calls_{0};
};

// POST /v1/images/generations を 2 本同時に投げ、以下をハードゲートする。
//   [1] 生成器の中で同時実行が 1 を超えなかったこと            ← 本命 (直列化の否定形)
//   [2] 遅い側のレイテンシが「競合なし基準 + 0.8*kBusyMs」以上   ← 窓が開いた正の証拠
//   [3] 2 応答の PNG が、それぞれの prompt の単発参照とバイト一致すること
//
// [2] が要る理由: [1] は否定形なので「2 本がそもそも重ならなかった」場合も max=1 で
// PASS してしまう (kBusyMs が小さくされた / スレッドプールが 1 になった / httplib が
// accept を直列化した、等)。そうなると api.cpp のロックを消してもゲートが緑のままに
// なる。重なった上で直列化されたなら負け側は勝ち側の生成を丸ごと待つが、重ならなければ
// 単発と同じ時間で返る — この差で「窓が開いた」ことを積極的に確認する。
// 総経過時間は判別に使えない (直列化されて重なった場合も、そもそも重ならなかった
// 場合も等しく約 2*kBusyMs になる) ので、必ず **個別** レイテンシの最大値で見る。
// しきい値は単発実測 (base_lat) 相対にする — 詳細は (5) のコメント。
//
// [3] で 2 本の prompt を別にするのは、StubGenerator が prompt ハッシュから決定的に
// 生成する = 同一 prompt では PNG バイト一致がトートロジーになるため。別 prompt なら
// 別バイトになり、応答のクロス配送 (req0 の body が req1 に届く) も同時に検出できる。
static bool test_generations_serialized(ServerFixture& fx, SerialProbeGenerator& gen)
{
    constexpr int kN = 2;
    // 2 本は **別 prompt** にする (理由は上のコメント [3])。
    const char* const prompts[kN] = {"serialization gate-A", "serialization gate-B"};

    auto make_req = [&](int i) -> std::string
    {
        const json j = {
            {"prompt", prompts[i]},
            {"n", 1},
            {"size", "64x48"},
            {"response_format", "b64_json"},
        };
        return j.dump();
    };

    // b64_json を取り出して PNG バイト列に戻す小物 (失敗は空を返す)。
    auto extract_png = [](const std::string& body) -> std::vector<uint8_t>
    {
        auto j = json::parse(body, nullptr, false);
        if (j.is_discarded() || !j.contains("data") || !j["data"].is_array() ||
            j["data"].empty() || !j["data"][0].contains("b64_json"))
        {
            return {};
        }
        return base64_decode(j["data"][0]["b64_json"].get<std::string>());
    };

    // --- (1) prompt ごとの単発参照 = 「壊れていない」正解 PNG ---
    //   ここで測る「競合なし 1 本」のレイテンシが、下の [2] のベースラインになる
    //   (= kBusyMs + この機体の HTTP 往復オーバーヘッド)。
    std::vector<std::vector<uint8_t>> refs(kN);
    double base_lat = 1.0e9; // 競合なしレイテンシの最小値 (小さい方=保守的なしきい値)
    for (int i = 0; i < kN; ++i)
    {
        httplib::Client cli("127.0.0.1", fx.port);
        cli.set_read_timeout(10, 0);
        const auto b0 = std::chrono::steady_clock::now();
        auto res = cli.Post("/v1/images/generations", make_req(i), "application/json");
        const auto b1 = std::chrono::steady_clock::now();
        const double bl = std::chrono::duration<double, std::milli>(b1 - b0).count();
        if (bl < base_lat)
        {
            base_lat = bl;
        }
        if (!res || res->status != 200)
        {
            std::cerr << "[serial] 基準リクエスト " << i << " が 200 でない (got "
                      << (res ? res->status : -1) << ")\n";
            return false;
        }
        refs[i] = extract_png(res->body);
        if (refs[i].empty())
        {
            std::cerr << "[serial] 基準応答 " << i << " から PNG を取り出せない\n";
            return false;
        }
    }
    // ハーネス健全性: 参照 2 本が別バイトでなければ [3] がトートロジーに戻る。
    if (refs[0] == refs[1])
    {
        std::cerr << "[serial] 2 prompt の参照 PNG が同一バイト = クロス配送を"
                     "検出できない (ハーネス異常)\n";
        return false;
    }
    gen.reset_counters();

    // --- (2) 2 本を「本当に重ねて」投げる ---
    std::atomic<int>  arrived{0};
    std::atomic<bool> rendezvous_failed{false};
    std::vector<int>         statuses(kN, -1);
    std::vector<std::string> bodies(kN);
    std::vector<double>      latency_ms(kN, 0.0);

    auto worker = [&](int i)
    {
        httplib::Client cli("127.0.0.1", fx.port);
        cli.set_read_timeout(10, 0);

        // ランデブー: 全スレッドが POST の直前で揃うまで待つ。
        // (これは「クライアント 2 本が POST 直前で揃った」ことしか保証しない。
        //  サーバ側で実際に窓が開いたかは下の [2] レイテンシゲートで確認する。)
        arrived.fetch_add(1);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (arrived.load() < kN)
        {
            if (std::chrono::steady_clock::now() > deadline)
            {
                rendezvous_failed.store(true);
                break;
            }
            std::this_thread::yield();
        }

        const auto s0 = std::chrono::steady_clock::now();
        auto res = cli.Post("/v1/images/generations", make_req(i), "application/json");
        const auto s1 = std::chrono::steady_clock::now();

        // 各スレッドが別要素だけを触る (再確保は起きない) ので排他は不要。
        latency_ms[i] = std::chrono::duration<double, std::milli>(s1 - s0).count();
        if (res)
        {
            statuses[i] = res->status;
            bodies[i]   = res->body;
        }
    };

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> th;
    th.reserve(kN);
    for (int i = 0; i < kN; ++i)
    {
        th.emplace_back(worker, i);
    }
    for (auto& t : th)
    {
        t.join();
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double max_lat = 0.0;
    for (int i = 0; i < kN; ++i)
    {
        if (latency_ms[i] > max_lat)
        {
            max_lat = latency_ms[i];
        }
    }

    // --- (3) ハーネス自体の健全性 (ここが崩れるとゲートが無意味になる) ---
    if (rendezvous_failed.load())
    {
        std::cerr << "[serial] ランデブー不成立 (ハーネス異常)\n";
        return false;
    }
    if (gen.calls() != kN)
    {
        std::cerr << "[serial] 生成器の呼び出し回数が " << gen.calls()
                  << " (期待 " << kN << ")\n";
        return false;
    }

    // --- (4) 本命: 生成器の中で同時実行が 1 を超えていないこと ---
    if (gen.max_in_flight() > 1)
    {
        std::cerr << "[serial] 生成が直列化されていない: 同時実行 max="
                  << gen.max_in_flight() << " (期待 1)\n";
        return false;
    }

    // --- (5) 窓が実際に開いた正の証拠: 遅い側が「競合なし + 0.8*kBusyMs」以上待たされたこと ---
    //   重なって直列化 → 負け側 ≈ base_lat + kBusyMs (勝ち側の生成を丸ごと待つ)。
    //   重ならなかった   → 両方 ≈ base_lat。
    //   しきい値を **実測 base_lat 相対** にするのが要点。素朴に 1.5*kBusyMs のような
    //   絶対値にすると、HTTP オーバーヘッドが kBusyMs に対して大きい機体・設定で
    //   「重なっていないのに超えてしまう」偽 PASS が出る (kBusyMs を小さくされた場合が
    //   まさにこれ)。base_lat を引けばオーバーヘッドが相殺され、残るのは
    //   「他方の生成を待ったか否か」だけになる。
    //   下限ゲートなので遅いマシンで偽 FAIL しない (遅さは base_lat にも乗る)。
    const double min_loss_lat = base_lat + 0.8 * SerialProbeGenerator::kBusyMs;
    if (max_lat < min_loss_lat)
    {
        std::cerr << "[serial] 2 本が重なっていない = 窓が開いていない: 個別レイテンシ max="
                  << max_lat << " ms < " << min_loss_lat << " ms (競合なし基準 "
                  << base_lat << " + 0.8*" << SerialProbeGenerator::kBusyMs << ")。"
                  << "この状態では直列化を検査できていない (ゲートが空撃ち)\n";
        return false;
    }

    // --- (6) 応答が壊れていないこと (クロス配送・状態漏れの検出) ---
    for (int i = 0; i < kN; ++i)
    {
        if (statuses[i] != 200)
        {
            std::cerr << "[serial] リクエスト " << i << " が 200 でない (got "
                      << statuses[i] << ")\n";
            return false;
        }
        const std::vector<uint8_t> png = extract_png(bodies[i]);
        if (png.empty())
        {
            std::cerr << "[serial] リクエスト " << i << " から PNG を取り出せない\n";
            return false;
        }
        if (png != refs[i])
        {
            std::cerr << "[serial] リクエスト " << i << " (prompt=" << prompts[i]
                      << ") の PNG が単発参照と不一致 (" << png.size() << " vs "
                      << refs[i].size() << " bytes)"
                      << (png == refs[1 - i] ? " ← 応答のクロス配送" : "") << "\n";
            return false;
        }
    }

    std::cout << "  [serial] 直列化 OK 同時実行 max=" << gen.max_in_flight()
              << " calls=" << gen.calls()
              << " / 個別レイテンシ " << latency_ms[0] << "+" << latency_ms[1]
              << " ms (max " << max_lat << " >= " << min_loss_lat
              << " = 競合なし " << base_lat << " +0.8*busy → 窓 OK)"
              << " / PNG " << refs[0].size() << "," << refs[1].size()
              << " bytes 各参照と一致 / 総 " << total_ms << " ms (busy "
              << SerialProbeGenerator::kBusyMs << " ms/回)\n";
    return true;
}

} // namespace dollama

int main()
{
    using namespace dollama;
    bool ok = true;

    std::cout << "=== test_http ===\n";

    // 配管単体 (サーバ不要)
    ok = test_png_roundtrip() && ok;
    ok = test_base64_roundtrip() && ok;

    // サーバ起動して自己リクエスト
    {
        StubGenerator gen;
        ServerFixture fx(gen);
        std::cout << "  [server] bound port " << fx.port << "\n";
        ok = test_index_page(fx) && ok;
        ok = test_health(fx) && ok;
        ok = test_models(fx) && ok;
        ok = test_generations_ok(fx) && ok;
        ok = test_generations_missing_prompt(fx) && ok;
        ok = test_generations_bad_json(fx) && ok;
    } // ここで svr.stop() → join

    // G-8k F1: 生成の直列化ゲート (計装スタブを注入した別サーバで実施)
    {
        SerialProbeGenerator gen;
        ServerFixture fx(gen);
        std::cout << "  [server] bound port " << fx.port << " (直列化ゲート)\n";
        ok = test_generations_serialized(fx, gen) && ok;
    } // ここで svr.stop() → join

    if (ok)
    {
        std::cout << "ALL PASSED\n";
        return 0;
    }
    std::cerr << "FAILED\n";
    return 1;
}
