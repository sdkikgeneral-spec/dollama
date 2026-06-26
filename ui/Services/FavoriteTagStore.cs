using System.Text.Json;

namespace Dollama.Ui.Services;

// お気に入りタグを ui/data/favorites.json に永続化するスレッドセーフなストア (シングルトン)。
// PresetStore のパターンを踏襲する:
//  - _gate ロックでスレッドセーフ
//  - UnsafeRelaxedJsonEscaping で日本語タグを非エスケープ出力
//  - ファイル不在・壊れ JSON でも例外で起動を止めず、空リストで復帰
//
// 書き込みは「一時ファイルに書いて File.Move(overwrite:true)」のアトミック置換で行い、
// 並行アクセス時にも presets.json と同様に半端な破損ファイルを残さない。
public sealed class FavoriteTagStore
{
    private readonly string _path;
    private readonly object _gate = new();
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = true,
        // 日本語タグ名をエスケープせずそのまま出力する
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    public FavoriteTagStore(IHostEnvironment env)
    {
        // ContentRootPath = ui/ 直下。data/favorites.json に解決。
        var dataDir = Path.Combine(env.ContentRootPath, "data");
        _path = Path.Combine(dataDir, "favorites.json");
    }

    // お気に入りタグを登録順で返す。
    public IReadOnlyList<string> All()
    {
        lock (_gate)
        {
            return Load();
        }
    }

    // タグを 1 つ追加する。前後空白除去 + 小文字寄せで正規化し、空・重複は無視。
    public void Add(string tag)
    {
        var norm = (tag ?? "").Trim().ToLowerInvariant();
        if (norm.Length == 0)
        {
            return;
        }
        lock (_gate)
        {
            var list = Load();
            if (list.Contains(norm))
            {
                return; // 重複は無視
            }
            list.Add(norm);
            Persist(list);
        }
    }

    // タグを削除する (正規化して照合)。
    public void Remove(string tag)
    {
        var norm = (tag ?? "").Trim().ToLowerInvariant();
        if (norm.Length == 0)
        {
            return;
        }
        lock (_gate)
        {
            var list = Load();
            if (list.RemoveAll(t => string.Equals(t, norm, StringComparison.Ordinal)) > 0)
            {
                Persist(list);
            }
        }
    }

    // --- 内部: ファイル I/O (呼び出し側で _gate ロック済み前提) ---

    private List<string> Load()
    {
        try
        {
            if (!File.Exists(_path))
            {
                return new List<string>();
            }
            var json = File.ReadAllText(_path);
            return JsonSerializer.Deserialize<List<string>>(json, JsonOpts)
                   ?? new List<string>();
        }
        catch
        {
            // 壊れ JSON 等は空扱い (起動・操作を止めない)。
            return new List<string>();
        }
    }

    // アトミック書込: 一時ファイルへ書いてから File.Move で置換する。
    // 書込中にプロセス/別スレッドが落ちても、既存の正しいファイルが壊れない。
    private void Persist(List<string> list)
    {
        var dir = Path.GetDirectoryName(_path)!;
        Directory.CreateDirectory(dir);
        var json = JsonSerializer.Serialize(list, JsonOpts);
        var tmp = _path + ".tmp";
        File.WriteAllText(tmp, json);
        File.Move(tmp, _path, overwrite: true);
    }
}
