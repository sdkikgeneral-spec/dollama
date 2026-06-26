using System.Text.Json.Serialization;

namespace Dollama.Ui.Services;

// タグパレットの 1 カテゴリ。wwwroot/tag-palette.json の各要素に対応する。
// JSON スキーマ: [{ "category": string, "tags": [string, ...] }, ...]
// このファイルは prompt-engineer 管轄 (UI 側は読むだけ・内容に依存しない)。
public sealed class TagCategory
{
    [JsonPropertyName("category")]
    public string Category { get; set; } = "";

    [JsonPropertyName("tags")]
    public List<string> Tags { get; set; } = new();
}
