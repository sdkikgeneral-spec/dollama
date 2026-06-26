using System.Text.Json;

namespace Dollama.Ui.Services;

// キュレーション済みタグパレット (静的 JSON) を保持するシングルトン。
// wwwroot/tag-palette.json (prompt-engineer 管轄) を起動時に 1 回読む。
//
// PresetStore.Load と同じ堅牢さ: ファイル不在・壊れ JSON でも例外で起動を
// 止めず、空リストで復帰する。ビルド・起動はこのファイルの内容に依存しない。
public sealed class TagPaletteCatalog
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNameCaseInsensitive = true,
    };

    // 起動時に読み込んだカテゴリ一覧 (読み取り専用で公開)。
    public IReadOnlyList<TagCategory> Categories { get; }

    public TagPaletteCatalog(IHostEnvironment env)
    {
        // ContentRootPath = ui/ 直下。wwwroot/tag-palette.json に解決。
        var path = Path.Combine(env.ContentRootPath, "wwwroot", "tag-palette.json");
        Categories = Load(path);
    }

    // ファイル不在・壊れ JSON は空リスト (起動を止めない)。
    private static IReadOnlyList<TagCategory> Load(string path)
    {
        try
        {
            if (!File.Exists(path))
            {
                return new List<TagCategory>();
            }
            var json = File.ReadAllText(path);
            return JsonSerializer.Deserialize<List<TagCategory>>(json, JsonOpts)
                   ?? new List<TagCategory>();
        }
        catch
        {
            // 壊れ JSON 等は空扱い。
            return new List<TagCategory>();
        }
    }
}
