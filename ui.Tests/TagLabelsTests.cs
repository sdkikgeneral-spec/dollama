using Dollama.Ui.Services;
using Xunit;

namespace Dollama.Ui.Tests;

// TagLabels を一時ディレクトリ上の自前 tag-labels.ja.json で検証する。
// 本番 wwwroot/tag-labels.ja.json (prompt-engineer 管轄) には依存しない。
// ContentRootPath/wwwroot/tag-labels.ja.json を読むため wwwroot を作って書き込む。
public sealed class TagLabelsTests : IDisposable
{
    private readonly string _root;
    private readonly string _wwwroot;
    private readonly string _labelsJson;

    public TagLabelsTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "dollama_labels_test_" + Guid.NewGuid().ToString("N"));
        _wwwroot = Path.Combine(_root, "wwwroot");
        Directory.CreateDirectory(_wwwroot);
        _labelsJson = Path.Combine(_wwwroot, "tag-labels.ja.json");
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_root))
            {
                Directory.Delete(_root, recursive: true);
            }
        }
        catch
        {
            // 掃除失敗はテスト本質でない。無視。
        }
    }

    private TagLabels Build()
    {
        return new TagLabels(new FakeHostEnvironment { ContentRootPath = _root });
    }

    private void WriteSampleDict()
    {
        File.WriteAllText(_labelsJson, """
        {
          "long hair": "ロングヘア",
          "smile": "笑顔"
        }
        """);
    }

    // (1) 正常辞書ロードで lang="ja" のヒットは日本語ラベルを返す。
    [Fact]
    public void Display_Ja_Hit_ReturnsJapanese()
    {
        WriteSampleDict();
        var labels = Build();
        Assert.Equal("ロングヘア", labels.Display("long hair", "ja"));
        Assert.Equal("笑顔", labels.Display("smile", "ja"));
    }

    // (2) lang="en" は辞書にあっても英語タグをそのまま返す。
    [Fact]
    public void Display_En_ReturnsEnglish()
    {
        WriteSampleDict();
        var labels = Build();
        Assert.Equal("long hair", labels.Display("long hair", "en"));
        Assert.Equal("smile", labels.Display("smile", "en"));
    }

    // (3) 辞書外タグは lang="ja" でも英語フォールバック。
    [Fact]
    public void Display_Ja_Miss_FallsBackToEnglish()
    {
        WriteSampleDict();
        var labels = Build();
        Assert.Equal("blush", labels.Display("blush", "ja"));
    }

    // (4) ファイル不在のときは全件英語フォールバック (起動を止めない)。
    [Fact]
    public void Display_FileAbsent_AllEnglish()
    {
        Assert.False(File.Exists(_labelsJson));
        var labels = Build();
        Assert.Equal("long hair", labels.Display("long hair", "ja"));
        Assert.Equal("smile", labels.Display("smile", "ja"));
    }

    // (5) 壊れ JSON のときは全件英語フォールバック (例外を投げない)。
    [Fact]
    public void Display_CorruptJson_AllEnglish()
    {
        File.WriteAllText(_labelsJson, "{ this is not valid json )))");
        var labels = Build();
        Assert.Equal("long hair", labels.Display("long hair", "ja"));
        Assert.Equal("smile", labels.Display("smile", "ja"));
    }
}
