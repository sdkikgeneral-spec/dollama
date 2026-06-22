using System.Net.Http.Json;
using System.Text.Json;

namespace Dollama.Ui.Services;

// C++ 生成サーバー (cpp-httplib, OpenAI Images 互換) を叩く型付きクライアント。
//
// このプロセス (Blazor Server) からサーバー間通信で C++ を叩くため、
// ブラウザは C++ サーバーを直接見ない = CORS は発生しない。
// BaseAddress は Program.cs で appsettings の "Dollama:BaseUrl" から設定する。
public sealed class DollamaClient
{
    private readonly HttpClient _http;
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNameCaseInsensitive = true,
    };

    public DollamaClient(HttpClient http)
    {
        _http = http;
    }

    // 生成リクエストを投げ、PNG バイト列を返す。
    // C++ 側が 4xx/5xx を返したらメッセージを拾って GenerationException を投げる。
    public async Task<byte[]> GenerateAsync(GenerationRequest req, CancellationToken ct = default)
    {
        using var resp = await _http.PostAsJsonAsync("/v1/images/generations", req, ct);

        if (!resp.IsSuccessStatusCode)
        {
            throw new GenerationException(await ExtractErrorAsync(resp, ct));
        }

        var body = await resp.Content.ReadFromJsonAsync<GenerationResponse>(JsonOpts, ct);
        var b64 = body?.Data.FirstOrDefault()?.B64Json;
        if (string.IsNullOrEmpty(b64))
        {
            throw new GenerationException("サーバー応答に画像データ (b64_json) が含まれていません");
        }

        return Convert.FromBase64String(b64);
    }

    // GET /health が {"status":"ok"} を返すかどうか。接続インジケータ用。
    public async Task<bool> HealthAsync(CancellationToken ct = default)
    {
        try
        {
            using var resp = await _http.GetAsync("/health", ct);
            if (!resp.IsSuccessStatusCode)
            {
                return false;
            }
            using var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync(ct));
            return doc.RootElement.TryGetProperty("status", out var s) &&
                   s.GetString() == "ok";
        }
        catch
        {
            // サーバー未起動・接続拒否などは「未接続」として握りつぶす
            return false;
        }
    }

    // OpenAI 形式エラー { "error": { "message", ... } } から message を取り出す。
    private static async Task<string> ExtractErrorAsync(HttpResponseMessage resp, CancellationToken ct)
    {
        try
        {
            var env = await resp.Content.ReadFromJsonAsync<ErrorEnvelope>(JsonOpts, ct);
            if (!string.IsNullOrEmpty(env?.Error?.Message))
            {
                return env!.Error!.Message;
            }
        }
        catch
        {
            // JSON でない応答はステータスだけ返す
        }
        return $"HTTP {(int)resp.StatusCode} {resp.ReasonPhrase}";
    }
}

// 生成系のエラーを UI で扱いやすくするための例外
public sealed class GenerationException : Exception
{
    public GenerationException(string message) : base(message) { }
}
