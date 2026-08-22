// OpenAI Images 互換 HTTP API 実装
//
// エンドポイント:
//   GET  /                       — ブラウザで開く簡易 Web UI (HTML 1 枚)
//   POST /v1/images/generations — txt2img (JSON → GenRequest → 生成 → base64 PNG)
//   POST /v1/images/edits       — img2img (現フェーズは 501 Not Implemented)
//   GET  /health                — {"status":"ok"}
//   GET  /v1/models             — {"data":[{"id":...,"object":"model"}]}
//
// 配管: ルーティング = cpp-httplib、JSON = nlohmann/json、base64 = 自作小物。
// 生成本体は IImageGenerator 越し (CUDA/OpenVINO 非依存の純 C++)。
#include "server/api.hpp"

#include <chrono>
#include <exception>
#include <iostream>
#include <mutex>
#include <string>

#include <nlohmann/json.hpp>

#include "server/base64.hpp"
#include "server/generator.hpp"

namespace dollama
{

namespace
{

using json = nlohmann::json;

// OpenAI 形式のエラー JSON を組み立てて res に書き込む
void write_error(httplib::Response& res, int status, const std::string& message,
                 const std::string& type)
{
    json err;
    err["error"] = {
        {"message", message},
        {"type", type},
        {"code", nullptr},
    };
    res.status = status;
    res.set_content(err.dump(), "application/json");
}

// "1024x1024" を幅・高さに分解する。失敗時 false。
bool parse_size(const std::string& s, int& w, int& h)
{
    const auto x = s.find('x');
    if (x == std::string::npos)
    {
        return false;
    }
    try
    {
        w = std::stoi(s.substr(0, x));
        h = std::stoi(s.substr(x + 1));
    }
    catch (...)
    {
        return false;
    }
    return (w > 0 && h > 0);
}

// G-8k F1: 生成 (IImageGenerator::generate) を直列化するためのロック。
//
// なぜ必要か:
//   cpp-httplib は既定でスレッドプールから複数リクエストを並行ディスパッチする。
//   一方、生成経路は **プロセス唯一のグローバル可変状態を 3 つ** 握っており、
//   どれもロックを持たない。したがって生成の同時 2 本は成立しない。
//
//   [1] bump アリーナ (src/kernels/device_arena) — G-8k で UNet / VAE / conv2d の
//       一時バッファをここへ集約した。状態 (owner / cur / offset / req_live) は無ロック。
//         - 所有権違反を検出できた場合 = 生成 A の step 間 (アリーナ静止の瞬間) に
//           生成 B が正規の所有権移譲で割り込み、A の次 forward の check_thread が
//           throw → A が途中 step で 500 になる。
//         - すり抜けた場合             = check_thread → カーソル bump が非アトミックな
//           ため両者が静止判定を通過し、同一カーソルから重複払い出し = 画像の破壊。
//   [2] cuBLAS ハンドル (src/kernels/gemm.cu の g_cublas_handle) — 遅延生成が
//       非アトミック。同時 2 本で cublasCreate が 2 回走りハンドルが 1 本リークし、
//       ポインタ書き込み自体が data race。定常状態でも cuBLAS は同一ハンドルの
//       同時使用を保障しない (本 repo に cublasSetStream は 0 箇所 = 全部デフォルト
//       ストリーム)。gemm.cu 自身のコメントが「単一 stream・単一スレッド前提」と明言。
//   [3] GroupNorm multi-block の常駐バッファ (src/kernels/groupnorm.cu の g_mb_buf) —
//       grow-only 再確保。A が返却ポインタでカーネル実行中に、B がより大きい要求で
//       入ると cudaFree → A が解放済み VRAM に書く。epilogue 経路限定だが
//       出荷 --fast は epilogue を含意するので実運用で踏む。
//
//   ★ 履歴の訂正: HTTP 並行生成は G-8k の退行ではない。[2] は fc97b76 (2026-06-22 /
//     2-6 S3-B)、[3] は 42ec1be (2026-07-11 / G-4k S1a) から在る。G-8k は 3 つ目の
//     共有状態を足しただけで、並行生成は少なくとも 2-6 から一貫して不成立だった。
//
//   ★ このロックを外せる条件: **3 つすべて** が個別にスレッド安全化されたとき。
//     アリーナ単体をロック化しても [2][3] が残るので、このロックは外せない。
//     「アリーナを直したからもう要らない」と判断すると [2][3] の競合が復活する。
//
// なぜファイルスコープか (生成器のメンバにしない理由):
//   上記 3 つはいずれも生成器インスタンスではなく **プロセス** で共有される資源だから。
//   生成器インスタンス単位のロックでは、生成器を 2 つ構築した瞬間に守れなくなる。
//   ファネルは本 TU の handle_generations 1 箇所だけなので、ここに 1 本置けば足りる。
std::mutex g_generate_mutex;

void handle_generations(IImageGenerator& gen, const httplib::Request& req,
                        httplib::Response& res)
{
    json body;
    try
    {
        body = json::parse(req.body);
    }
    catch (const std::exception&)
    {
        write_error(res, 400, "リクエストボディが正しい JSON ではありません",
                    "invalid_request_error");
        return;
    }

    if (!body.is_object())
    {
        write_error(res, 400, "リクエストボディは JSON オブジェクトである必要があります",
                    "invalid_request_error");
        return;
    }

    // prompt 必須
    if (!body.contains("prompt") || !body["prompt"].is_string() ||
        body["prompt"].get<std::string>().empty())
    {
        write_error(res, 400, "'prompt' は必須の文字列フィールドです",
                    "invalid_request_error");
        return;
    }

    GenRequest gr;
    gr.prompt = body["prompt"].get<std::string>();

    if (body.contains("negative_prompt") && body["negative_prompt"].is_string())
    {
        gr.negative_prompt = body["negative_prompt"].get<std::string>();
    }
    if (body.contains("n") && body["n"].is_number_integer())
    {
        gr.n = body["n"].get<int>();
    }
    if (body.contains("steps") && body["steps"].is_number_integer())
    {
        gr.steps = body["steps"].get<int>();
    }
    if (body.contains("size") && body["size"].is_string())
    {
        int w = 0, h = 0;
        if (!parse_size(body["size"].get<std::string>(), w, h))
        {
            write_error(res, 400, "'size' は \"幅x高さ\" 形式である必要があります",
                        "invalid_request_error");
            return;
        }
        gr.width = w;
        gr.height = h;
    }

    // L-2: ランタイム LoRA (任意項目)。loras: [{ "name": str, "strength": float=1.0 }]。
    // 未指定 = 空 = 従来経路 (無改変)。形式不正は 400 で明示的に弾く (黙って無視しない)。
    if (body.contains("loras"))
    {
        if (!body["loras"].is_array())
        {
            write_error(res, 400, "'loras' は配列である必要があります",
                        "invalid_request_error");
            return;
        }
        for (const auto& item : body["loras"])
        {
            if (!item.is_object() || !item.contains("name") || !item["name"].is_string() ||
                item["name"].get<std::string>().empty())
            {
                write_error(res, 400,
                            "'loras' の各要素は {\"name\": 文字列, \"strength\": 数値} "
                            "である必要があります",
                            "invalid_request_error");
                return;
            }
            LoraSpec spec;
            spec.name = item["name"].get<std::string>();
            if (item.contains("strength"))
            {
                if (!item["strength"].is_number())
                {
                    write_error(res, 400, "'loras[].strength' は数値である必要があります",
                                "invalid_request_error");
                    return;
                }
                spec.strength = item["strength"].get<float>();
            }
            gr.loras.push_back(spec);
        }
    }

    // response_format: url は未対応
    if (body.contains("response_format") && body["response_format"].is_string())
    {
        const std::string rf = body["response_format"].get<std::string>();
        if (rf != "b64_json")
        {
            write_error(res, 400, "'response_format' は 'b64_json' のみ対応しています",
                        "invalid_request_error");
            return;
        }
    }

    // 生成 (失敗は 500。ただしリクエスト起因 = std::invalid_argument は 400:
    //  L-2 の LoRA 名未解決など、呼び出し側入力の誤りをサーバ異常と区別する)
    GenResult result;
    try
    {
        // 直列化 (理由は g_generate_mutex の定義コメント)。ロックのスコープは
        // gen.generate() の 1 文だけに絞る — JSON 解析・base64 化・応答組み立ては
        // 並行のままでよい。生成器の内側から本ファネルへ再入する経路は無いので
        // 非再帰の std::mutex で足りる (IImageGenerator::generate の呼び出し元は
        // 本関数と CLI 経路の 2 箇所のみ・CLI 経路は --http と排他)。
        {
            std::lock_guard<std::mutex> lock(g_generate_mutex);
            result = gen.generate(gr);
        }
    }
    catch (const std::invalid_argument& e)
    {
        write_error(res, 400, std::string("リクエストが不正です: ") + e.what(),
                    "invalid_request_error");
        return;
    }
    catch (const std::exception& e)
    {
        write_error(res, 500, std::string("生成に失敗しました: ") + e.what(),
                    "server_error");
        return;
    }

    const std::string b64 = base64_encode(result.png_bytes);
    const auto now = std::chrono::system_clock::now();
    const long long created =
        std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();

    json out;
    out["created"] = created;
    out["data"] = json::array();
    out["data"].push_back({{"b64_json", b64}});

    res.status = 200;
    res.set_content(out.dump(), "application/json");
}

// ブラウザで開く簡易 Web UI (外部依存なし・インライン CSS/JS の単一ページ)。
// prompt / negative_prompt を入力し /v1/images/generations を fetch して
// 返ってきた b64_json を <img> に表示するだけの最小構成。
const char* const kIndexHtml = R"HTML(<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>dollama</title>
<style>
  body { font-family: sans-serif; max-width: 720px; margin: 24px auto; padding: 0 16px; color: #222; }
  h1 { font-size: 1.4rem; }
  label { display: block; margin-top: 12px; font-weight: bold; }
  textarea { width: 100%; box-sizing: border-box; font-size: 1rem; padding: 6px; }
  button { margin-top: 16px; padding: 8px 20px; font-size: 1rem; cursor: pointer; }
  button:disabled { opacity: 0.5; cursor: default; }
  #status { margin-top: 12px; min-height: 1.2em; }
  #status.error { color: #c00; }
  #result { margin-top: 16px; }
  #result img { max-width: 100%; border: 1px solid #ccc; }
  .note { margin-top: 8px; font-size: 0.85rem; color: #666; }
</style>
</head>
<body>
<h1>dollama 画像生成</h1>
<p class="note">注記: この PC では生成器がスタブにフォールバックするため、本物の絵は出ません (UI と PNG 表示の確認用です)。</p>

<label for="prompt">prompt</label>
<textarea id="prompt" rows="3" placeholder="1girl, silver hair, magical girl">1girl, silver hair, magical girl</textarea>

<label for="negative">negative_prompt</label>
<textarea id="negative" rows="2" placeholder="lowres, bad anatomy"></textarea>

<button id="gen">生成</button>

<div id="status"></div>
<div id="result"></div>

<script>
const btn = document.getElementById("gen");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");

function setStatus(msg, isError)
{
  statusEl.textContent = msg;
  statusEl.className = isError ? "error" : "";
}

btn.addEventListener("click", async () =>
{
  const prompt = document.getElementById("prompt").value;
  const negative_prompt = document.getElementById("negative").value;
  if (!prompt.trim())
  {
    setStatus("prompt を入力してください", true);
    return;
  }

  btn.disabled = true;
  setStatus("生成中...", false);
  resultEl.innerHTML = "";

  try
  {
    const resp = await fetch("/v1/images/generations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, negative_prompt, response_format: "b64_json" })
    });

    if (resp.status !== 200)
    {
      let detail = "HTTP " + resp.status;
      try
      {
        const errJson = await resp.json();
        if (errJson && errJson.error && errJson.error.message)
        {
          detail += ": " + errJson.error.message;
        }
      }
      catch (e) {}
      setStatus("生成に失敗しました (" + detail + ")", true);
      return;
    }

    const data = await resp.json();
    if (!data || !Array.isArray(data.data) || data.data.length === 0 || !data.data[0].b64_json)
    {
      setStatus("レスポンスに画像データがありません", true);
      return;
    }

    const b64 = data.data[0].b64_json;
    const img = document.createElement("img");
    img.src = "data:image/png;base64," + b64;
    resultEl.appendChild(img);
    setStatus("完了", false);
  }
  catch (e)
  {
    setStatus("通信エラー: " + e, true);
  }
  finally
  {
    btn.disabled = false;
  }
});
</script>
</body>
</html>
)HTML";

} // namespace

void register_routes(httplib::Server& svr, IImageGenerator& gen)
{
    // ブラウザで開く簡易 Web UI
    svr.Get("/",
        [](const httplib::Request&, httplib::Response& res)
        {
            res.status = 200;
            res.set_content(kIndexHtml, "text/html; charset=utf-8");
        });

    svr.Get("/health",
        [](const httplib::Request&, httplib::Response& res)
        {
            json j;
            j["status"] = "ok";
            res.status = 200;
            res.set_content(j.dump(), "application/json");
        });

    svr.Get("/v1/models",
        [&gen](const httplib::Request&, httplib::Response& res)
        {
            json j;
            j["data"] = json::array();
            j["data"].push_back({{"id", gen.model_id()}, {"object", "model"}});
            res.status = 200;
            res.set_content(j.dump(), "application/json");
        });

    svr.Post("/v1/images/generations",
        [&gen](const httplib::Request& req, httplib::Response& res)
        {
            handle_generations(gen, req, res);
        });

    svr.Post("/v1/images/edits",
        [](const httplib::Request&, httplib::Response& res)
        {
            write_error(res, 501, "img2img (edits) は未実装です",
                        "not_implemented_error");
        });

    // 例外ハンドラ (ハンドラ内で漏れた例外を 500 にする)
    svr.set_exception_handler(
        [](const httplib::Request&, httplib::Response& res, std::exception_ptr ep)
        {
            std::string msg = "内部エラー";
            try
            {
                if (ep)
                {
                    std::rethrow_exception(ep);
                }
            }
            catch (const std::exception& e)
            {
                msg = e.what();
            }
            json err;
            err["error"] = {
                {"message", msg},
                {"type", "server_error"},
                {"code", nullptr},
            };
            res.status = 500;
            res.set_content(err.dump(), "application/json");
        });
}

int start_server(IImageGenerator& gen, const std::string& host, int port)
{
    httplib::Server svr;
    register_routes(svr, gen);

    std::cout << "[http] listening on " << host << ":" << port << "\n";
    if (!svr.listen(host.c_str(), port))
    {
        std::cerr << "[http] bind に失敗しました: " << host << ":" << port << "\n";
        return 1;
    }
    return 0;
}

} // namespace dollama
