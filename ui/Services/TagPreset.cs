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

    // サムネイル画像のファイル名 (data/thumbs/ 配下・例 "prompt_お気に入り.png")。
    // nullable で既存 presets.json (thumbnail なし) と後方互換。
    [JsonPropertyName("thumbnail")]
    public string? Thumbnail { get; set; }
}
