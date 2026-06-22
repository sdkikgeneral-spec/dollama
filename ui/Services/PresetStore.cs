using System.Text.Json;

namespace Dollama.Ui.Services;

// タグプリセットを ui/data/presets.json に永続化するスレッドセーフなストア (シングルトン)。
// kind ("prompt" / "negative") で型分けして扱う。
//
// Blazor Server のコンポーネントから直接 DI で呼ぶ (HTTP 不要)。
// ファイル不在・壊れ JSON でも例外で起動を止めず、空リストで復帰する。
public sealed class PresetStore
{
    private readonly string _path;
    private readonly object _gate = new();
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = true,
        // 日本語タグ名をエスケープせずそのまま出力する
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    public PresetStore(IHostEnvironment env)
    {
        // ContentRootPath = ui/ 直下。data/presets.json に解決。
        _path = Path.Combine(env.ContentRootPath, "data", "presets.json");
    }

    // 指定 kind のプリセットを名前順で返す。
    public IReadOnlyList<TagPreset> All(string kind)
    {
        lock (_gate)
        {
            return Load()
                .Where(p => string.Equals(p.Kind, kind, StringComparison.OrdinalIgnoreCase))
                .OrderBy(p => p.Name, StringComparer.OrdinalIgnoreCase)
                .ToList();
        }
    }

    // プリセットを保存する。同一 kind の同名は上書き。名前・kind 空は弾く。
    public void Save(TagPreset preset)
    {
        var name = preset.Name?.Trim() ?? "";
        var kind = preset.Kind?.Trim() ?? "";
        if (name.Length == 0)
        {
            throw new ArgumentException("プリセット名は必須です");
        }
        if (kind.Length == 0)
        {
            throw new ArgumentException("kind は必須です");
        }
        preset.Name = name;
        preset.Kind = kind;

        lock (_gate)
        {
            var list = Load();
            list.RemoveAll(p => SameKey(p, kind, name));
            list.Add(preset);
            Persist(list);
        }
    }

    // kind + 名前でプリセットを削除する。
    public void Delete(string kind, string name)
    {
        lock (_gate)
        {
            var list = Load();
            if (list.RemoveAll(p => SameKey(p, kind, name)) > 0)
            {
                Persist(list);
            }
        }
    }

    private static bool SameKey(TagPreset p, string kind, string name) =>
        string.Equals(p.Kind, kind, StringComparison.OrdinalIgnoreCase) &&
        string.Equals(p.Name, name, StringComparison.OrdinalIgnoreCase);

    // --- 内部: ファイル I/O (呼び出し側で _gate ロック済み前提) ---

    private List<TagPreset> Load()
    {
        try
        {
            if (!File.Exists(_path))
            {
                return new List<TagPreset>();
            }
            var json = File.ReadAllText(_path);
            return JsonSerializer.Deserialize<List<TagPreset>>(json, JsonOpts)
                   ?? new List<TagPreset>();
        }
        catch
        {
            // 壊れ JSON 等は空扱い (起動・操作を止めない)
            return new List<TagPreset>();
        }
    }

    private void Persist(List<TagPreset> list)
    {
        var dir = Path.GetDirectoryName(_path)!;
        Directory.CreateDirectory(dir);
        File.WriteAllText(_path, JsonSerializer.Serialize(list, JsonOpts));
    }
}
