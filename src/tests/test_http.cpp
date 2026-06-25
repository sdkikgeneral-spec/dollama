// HTTP サーバー (OpenAI Images 互換) 単体テスト + 往復ベンチ
//
// testing.md 形式: 各検証は if (!cond) { cerr; return false; }、
// main で ok = test_xxx() && ok 集約、成功は PASSED / 最後に ALL PASSED。
//
// 純 C++ (CUDA / OpenVINO 非依存)。StubGenerator を注入し httplib::Server を
// バックグラウンドスレッドで 127.0.0.1 の OS 自動割当ポート (bind_to_any_port)
// で起動 → wait_until_ready → httplib::Client で自己リクエスト → 検証。
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
#include "server/png.hpp"
#include "server/stub_generator.hpp"

namespace dollama
{

using json = nlohmann::json;

// テスト用にサーバを bind_to_any_port で起動し、ポートと制御を保持する。
struct ServerFixture
{
    httplib::Server svr;
    StubGenerator gen;
    std::thread th;
    int port = 0;

    ServerFixture()
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
        ServerFixture fx;
        std::cout << "  [server] bound port " << fx.port << "\n";
        ok = test_index_page(fx) && ok;
        ok = test_health(fx) && ok;
        ok = test_models(fx) && ok;
        ok = test_generations_ok(fx) && ok;
        ok = test_generations_missing_prompt(fx) && ok;
        ok = test_generations_bad_json(fx) && ok;
    } // ここで svr.stop() → join

    if (ok)
    {
        std::cout << "ALL PASSED\n";
        return 0;
    }
    std::cerr << "FAILED\n";
    return 1;
}
