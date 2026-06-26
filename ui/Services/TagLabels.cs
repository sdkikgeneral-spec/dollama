using System.Text.Json;

namespace Dollama.Ui.Services;

// danbooru タグ (英語) → 日本語ラベルの対応辞書を保持するシングルトン。
// wwwroot/tag-labels.ja.json (prompt-engineer 管轄) を起動時に 1 回読む。
//
// TagPaletteCatalog と同じ堅牢さ: ファイル不在・壊れ JSON でも例外で起動を
// 止めず、空辞書で復帰する (= 全件英語フォールバック)。
// 内部に保持するタグ値・C++ へ送る prompt は英語のまま不変で、本サービスは
// あくまで「表示専用」の写像である。
public sealed class TagLabels
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        // 辞書 (キー=タグ値) なので大文字小文字判定は実質効かないが害はない。
        PropertyNameCaseInsensitive = true,
    };

    // 英語タグ → 日本語ラベルの辞書 (読み取り専用で扱う)。
    private readonly Dictionary<string, string> _labels;

    public TagLabels(IHostEnvironment env)
    {
        // ContentRootPath = ui/ 直下。wwwroot/tag-labels.ja.json に解決。
        var path = Path.Combine(env.ContentRootPath, "wwwroot", "tag-labels.ja.json");
        _labels = Load(path);
    }

    // ファイル不在・壊れ JSON は空辞書 (起動を止めない・全件英語フォールバック)。
    private static Dictionary<string, string> Load(string path)
    {
        try
        {
            if (!File.Exists(path))
            {
                return new Dictionary<string, string>();
            }
            var json = File.ReadAllText(path);
            return JsonSerializer.Deserialize<Dictionary<string, string>>(json, JsonOpts)
                   ?? new Dictionary<string, string>();
        }
        catch
        {
            // 壊れ JSON 等は空扱い。
            return new Dictionary<string, string>();
        }
    }

    // 表示用ラベルを返す。lang=="ja" かつ辞書にヒットすれば日本語ラベル、
    // それ以外 (en / 辞書外 / null) は英語タグをそのまま返す (フォールバック)。
    public string Display(string tag, string lang)
    {
        if (tag is null)
        {
            return "";
        }
        if (lang == "ja" && _labels.TryGetValue(tag, out var ja))
        {
            return ja;
        }
        return tag;
    }
}
