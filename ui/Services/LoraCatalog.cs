using System.Text.Json;
using System.Text.Json.Serialization;

namespace Dollama.Ui.Services;

// LoRA カタログ (静的 JSON) を保持するシングルトン。
// C++ サーバーに LoRA 一覧を返す GET が無いため、UI 側で wwwroot/loras.json を持つ。
//
// TagPaletteCatalog を忠実に踏襲: ファイル不在・壊れ JSON でも例外で起動を
// 止めず、空リストで復帰する。ビルド・起動はこのファイルの内容に依存しない。
// カタログの name が実在しない場合は生成時に C++ サーバーが 400 を返す (許容)。
public sealed class LoraCatalog
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNameCaseInsensitive = true,
    };

    // 起動時に読み込んだ LoRA 一覧 (読み取り専用で公開)。
    public IReadOnlyList<LoraCatalogEntry> Entries { get; }

    public LoraCatalog(IHostEnvironment env)
    {
        // ContentRootPath = ui/ 直下。wwwroot/loras.json に解決。
        var path = Path.Combine(env.ContentRootPath, "wwwroot", "loras.json");
        Entries = Load(path);
    }

    // ファイル不在・壊れ JSON は空リスト (起動を止めない)。
    private static IReadOnlyList<LoraCatalogEntry> Load(string path)
    {
        try
        {
            if (!File.Exists(path))
            {
                return new List<LoraCatalogEntry>();
            }
            var json = File.ReadAllText(path);
            return JsonSerializer.Deserialize<List<LoraCatalogEntry>>(json, JsonOpts)
                   ?? new List<LoraCatalogEntry>();
        }
        catch
        {
            // 壊れ JSON 等は空扱い。
            return new List<LoraCatalogEntry>();
        }
    }
}

// カタログの 1 項目。wwwroot/loras.json の各要素に対応する。
// JSON スキーマ: [{ "name": string, "label": string, "defaultStrength": number }, ...]
// name は C++ 側で <name>.safetensors (models/loras) に解決される。
public sealed class LoraCatalogEntry
{
    // サーバーへ渡す LoRA 名 (拡張子なし)。
    [JsonPropertyName("name")]
    public string Name { get; set; } = "";

    // UI 表示名 (日本語可)。
    [JsonPropertyName("label")]
    public string Label { get; set; } = "";

    // 初期強度。省略時は 1.0。
    [JsonPropertyName("defaultStrength")]
    public float DefaultStrength { get; set; } = 1.0f;
}
