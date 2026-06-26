using System.Text.Json;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;

namespace Dollama.Ui.Services;

// タグプリセットを ui/data/presets.json に永続化するスレッドセーフなストア (シングルトン)。
// kind ("prompt" / "negative") で型分けして扱う。
//
// Blazor Server のコンポーネントから直接 DI で呼ぶ (HTTP 不要)。
// ファイル不在・壊れ JSON でも例外で起動を止めず、空リストで復帰する。
//
// サムネイルは別ファイル PNG として data/thumbs/ 配下に保存し、presets.json には
// ファイル名のみを持つ (thumbnail フィールド)。縮小は SixLabors.ImageSharp で
// 128px 上限・アスペクト比維持。
public sealed class PresetStore
{
    private readonly string _path;
    private readonly string _thumbsDir;
    private readonly object _gate = new();
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = true,
        // 日本語タグ名をエスケープせずそのまま出力する
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    // サムネイルの最大辺 (px)。これを上限にアスペクト比を維持して縮小する。
    private const int ThumbMaxPx = 128;

    public PresetStore(IHostEnvironment env)
    {
        // ContentRootPath = ui/ 直下。data/presets.json に解決。
        var dataDir = Path.Combine(env.ContentRootPath, "data");
        _path = Path.Combine(dataDir, "presets.json");
        _thumbsDir = Path.Combine(dataDir, "thumbs");
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

    // プリセットを保存する (サムネなし)。既存呼び出し互換のため null 委譲。
    public void Save(TagPreset preset)
    {
        Save(preset, null);
    }

    // プリセットを保存する。同一 kind の同名は上書き。名前・kind 空は弾く。
    // thumbnailPng != null のとき 128px 上限へ縮小し data/thumbs/ に PNG 保存して
    // preset.Thumbnail にファイル名を設定する。null のときはサムネ処理をしない
    // (既存サムネがあれば preset.Thumbnail はそのまま温存される)。
    public void Save(TagPreset preset, byte[]? thumbnailPng)
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
            if (thumbnailPng is not null)
            {
                var file = SafeFile(kind, name);
                Directory.CreateDirectory(_thumbsDir);
                var thumbPath = Path.Combine(_thumbsDir, file);

                // 128px 上限・アスペクト比維持で縮小して PNG 保存。
                using (var image = Image.Load(thumbnailPng))
                {
                    image.Mutate(x => x.Resize(new ResizeOptions
                    {
                        Size = new Size(ThumbMaxPx, ThumbMaxPx),
                        Mode = ResizeMode.Max,
                    }));
                    image.SaveAsPng(thumbPath);
                }
                preset.Thumbnail = file;
            }

            var list = Load();
            list.RemoveAll(p => SameKey(p, kind, name));
            list.Add(preset);
            Persist(list);
        }
    }

    // kind + 名前でプリセットを削除する。対応するサムネ PNG も削除する。
    public void Delete(string kind, string name)
    {
        lock (_gate)
        {
            var list = Load();
            if (list.RemoveAll(p => SameKey(p, kind, name)) > 0)
            {
                Persist(list);
            }

            // 孤児サムネを残さないよう PNG も消す (存在しなくても落ちない)。
            var thumbPath = Path.Combine(_thumbsDir, SafeFile(kind, name));
            try
            {
                if (File.Exists(thumbPath))
                {
                    File.Delete(thumbPath);
                }
            }
            catch
            {
                // 削除失敗 (ロック等) は致命でない。無視して続行。
            }
        }
    }

    // サムネ実体があれば静的公開 URL ("/thumb/{file}?v={ticks}") を返す。
    // ?v=LastWrite で上書き時にブラウザキャッシュを確実に無効化する。無ければ null。
    public string? ThumbnailUrl(TagPreset p)
    {
        var file = p.Thumbnail;
        if (string.IsNullOrEmpty(file))
        {
            return null;
        }
        var thumbPath = Path.Combine(_thumbsDir, file);
        if (!File.Exists(thumbPath))
        {
            return null;
        }
        var ticks = File.GetLastWriteTimeUtc(thumbPath).Ticks;
        return $"/thumb/{Uri.EscapeDataString(file)}?v={ticks}";
    }

    // kind + name から data/thumbs/ 配下に閉じた安全なファイル名 "{kind}_{name}.png" を作る。
    // 不正文字を _ 置換・".." を潰し・先頭末尾ドット除去でパストラバーサルを防ぐ。
    // 日本語名は保持する。生成パスが必ず _thumbsDir 配下に閉じることを最後に検証する。
    internal string SafeFile(string kind, string name)
    {
        var raw = $"{kind}_{name}";
        var invalid = Path.GetInvalidFileNameChars();
        var chars = raw.Select(c => Array.IndexOf(invalid, c) >= 0 ? '_' : c).ToArray();
        var sanitized = new string(chars);

        // ".." を潰す (連続も含めて残らなくなるまで繰り返す)。
        while (sanitized.Contains(".."))
        {
            sanitized = sanitized.Replace("..", "_");
        }
        // パス区切りも念のため除去 (invalid に含まれない環境への保険)。
        sanitized = sanitized.Replace('/', '_').Replace('\\', '_');
        // 先頭末尾のドット・空白を除去。
        sanitized = sanitized.Trim().Trim('.');
        if (sanitized.Length == 0)
        {
            sanitized = "_";
        }

        var file = sanitized + ".png";

        // 最終防御: 生成パスが必ず _thumbsDir 配下であることを検証する。
        var full = Path.GetFullPath(Path.Combine(_thumbsDir, file));
        var baseDir = Path.GetFullPath(_thumbsDir) + Path.DirectorySeparatorChar;
        if (!full.StartsWith(baseDir, StringComparison.Ordinal))
        {
            // 想定外 (区切り混入等)。ファイル名部分のみに切り詰めて再構成。
            file = Path.GetFileName(file);
            if (string.IsNullOrEmpty(file))
            {
                file = "_.png";
            }
        }
        return file;
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
