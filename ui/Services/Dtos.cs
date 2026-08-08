using System.Text.Json.Serialization;

namespace Dollama.Ui.Services;

// C++ 生成サーバー (OpenAI Images 互換 API) との往復で使う DTO 群。
// フィールド名は src/server/api.cpp のパース実装に厳密に合わせる (snake_case)。

// POST /v1/images/generations のリクエストボディ
public sealed class GenerationRequest
{
    [JsonPropertyName("prompt")]
    public string Prompt { get; set; } = "";

    // 空文字なら送らない (省略時 C++ 側は "" 扱い)
    [JsonPropertyName("negative_prompt")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? NegativePrompt { get; set; }

    [JsonPropertyName("steps")]
    public int Steps { get; set; } = 20;

    // "幅x高さ" 形式 (例 "1024x1024")
    [JsonPropertyName("size")]
    public string Size { get; set; } = "1024x1024";

    // C++ 側は b64_json のみ対応
    [JsonPropertyName("response_format")]
    public string ResponseFormat { get; set; } = "b64_json";

    // ランタイム LoRA 指定 (L-2)。空/未選択のときは null にして "loras" キー自体を
    // 出さない (WhenWritingNull)。これにより従来経路 (LoRA なし生成) は bit-exact 維持。
    [JsonPropertyName("loras")]
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public List<LoraSpec>? Loras { get; set; }
}

// LoRA 1 件の指定。src/server/api.cpp の loras:[{name,strength}] に厳密対応。
// C# の PascalCase 既定に流されないよう JsonPropertyName を明示する (snake_case 契約)。
public sealed class LoraSpec
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = "";

    // 強度。C++ 側既定は 1.0。UI では 0.0–1.5 に制約する (UI 側の安全弁)。
    [JsonPropertyName("strength")]
    public float Strength { get; set; } = 1.0f;
}

// レスポンス { "created": <unix秒>, "data": [ { "b64_json": "..." } ] }
public sealed class GenerationResponse
{
    [JsonPropertyName("created")]
    public long Created { get; set; }

    [JsonPropertyName("data")]
    public List<ImageData> Data { get; set; } = new();
}

public sealed class ImageData
{
    [JsonPropertyName("b64_json")]
    public string B64Json { get; set; } = "";
}

// エラー { "error": { "message", "type", "code" } }
public sealed class ErrorEnvelope
{
    [JsonPropertyName("error")]
    public ErrorBody? Error { get; set; }
}

public sealed class ErrorBody
{
    [JsonPropertyName("message")]
    public string Message { get; set; } = "";

    [JsonPropertyName("type")]
    public string Type { get; set; } = "";
}
