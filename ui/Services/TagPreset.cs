using System.Text.Json.Serialization;

namespace Dollama.Ui.Services;

// 名前付きタグプリセット。kind で「プロンプト用」「ネガティブ用」を型分けする。
// 同一 kind 内で name は一意 (prompt と negative で同名は別物として共存可)。
public sealed class TagPreset
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = "";

    // "prompt" または "negative"
    [JsonPropertyName("kind")]
    public string Kind { get; set; } = "";

    [JsonPropertyName("tags")]
    public List<string> Tags { get; set; } = new();
}
