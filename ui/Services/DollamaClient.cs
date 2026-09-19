using System.Net.Http.Json;
using System.Text.Json;

namespace Dollama.Ui.Services;

// C++ 生成サーバー (cpp-httplib, OpenAI Images 互換) を叩くクライアント。
//
// このプロセス (Blazor Server) からサーバー間通信で C++ を叩くため、
// ブラウザは C++ サーバーを直接見ない = CORS は発生しない。
//
// 2-6f: 複数エンドポイント切替に伴い、固定 BaseAddress の型付き HttpClient から
// IHttpClientFactory + EndpointRegistry.Current の組へ変更。リクエストごとに
// EndpointUrl.Combine(Registry.Current.Url, path) で絶対 URL を組み立てる。
public sealed class DollamaClient
{
    private readonly IHttpClientFactory _factory;
    private readonly EndpointRegistry _registry;
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNameCaseInsensitive = true,
    };

    public DollamaClient(IHttpClientFactory factory, EndpointRegistry registry)
    {
        _factory = factory;
        _registry = registry;
    }

    // 生成リクエストを投げ、PNG バイト列を返す。
    // C++ 側が 4xx/5xx を返したらメッセージを拾って GenerationException を投げる。
    public async Task<byte[]> GenerateAsync(GenerationRequest req, CancellationToken ct = default)
    {
        var http = _factory.CreateClient("dollama");
        var url = EndpointUrl.Combine(_registry.Current.Url, "/v1/images/generations");
        using var resp = await http.PostAsJsonAsync(url, req, ct);

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

    // GET /health が {"status":"ok"} を返すかどうか (現在選択中のエンドポイント)。接続インジケータ用。
    public async Task<bool> HealthAsync(CancellationToken ct = default)
    {
        var status = await ProbeAsync(_registry.Current, ct);
        return status.Reachable;
    }

    // 指定エンドポイントの到達可否とモデル ID を 1 回で調べる。
    // /health が 200 かつ status=="ok" のときだけ /v1/models を叩いて data[0].id を拾う。
    // タイムアウトは切替 UI を待たせないよう短め (3 秒) に切る。
    // 失敗はすべて握りつぶし Reachable=false, ModelId=null を返す (HealthAsync と同じ流儀)。
    public async Task<EndpointStatus> ProbeAsync(EndpointRegistry.Endpoint endpoint, CancellationToken ct = default)
    {
        try
        {
            using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            cts.CancelAfter(TimeSpan.FromSeconds(3));

            var http = _factory.CreateClient("dollama");
            var healthUrl = EndpointUrl.Combine(endpoint.Url, "/health");
            using var healthResp = await http.GetAsync(healthUrl, cts.Token);
            if (!healthResp.IsSuccessStatusCode)
            {
                return new EndpointStatus(false, null);
            }
            using var healthDoc = JsonDocument.Parse(await healthResp.Content.ReadAsStringAsync(cts.Token));
            var ok = healthDoc.RootElement.TryGetProperty("status", out var s) && s.GetString() == "ok";
            if (!ok)
            {
                return new EndpointStatus(false, null);
            }

            string? modelId = null;
            try
            {
                var modelsUrl = EndpointUrl.Combine(endpoint.Url, "/v1/models");
                using var modelsResp = await http.GetAsync(modelsUrl, cts.Token);
                if (modelsResp.IsSuccessStatusCode)
                {
                    using var modelsDoc = JsonDocument.Parse(await modelsResp.Content.ReadAsStringAsync(cts.Token));
                    if (modelsDoc.RootElement.TryGetProperty("data", out var data) &&
                        data.ValueKind == JsonValueKind.Array && data.GetArrayLength() > 0 &&
                        data[0].TryGetProperty("id", out var id))
                    {
                        modelId = id.GetString();
                    }
                }
            }
            catch
            {
                // /v1/models が無い・落ちている場合でも到達性自体は /health で確定済みなので無視。
            }

            return new EndpointStatus(true, modelId);
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            // 呼び出し側の ct が原因のキャンセルは呼び出し側の意図なので再 throw する。
            // (内部の 3 秒タイムアウト起因のキャンセルはここに来ない = 下の catch で Reachable=false)
            throw;
        }
        catch
        {
            // サーバー未起動・接続拒否・内部タイムアウト (3秒) などは「未到達」として握りつぶす
            return new EndpointStatus(false, null);
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

// ProbeAsync の結果。Reachable=false のとき ModelId は常に null。
public sealed record EndpointStatus(bool Reachable, string? ModelId);
