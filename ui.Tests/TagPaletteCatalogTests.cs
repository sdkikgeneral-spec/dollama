using Dollama.Ui.Services;
using Xunit;

namespace Dollama.Ui.Tests;

// TagPaletteCatalog を一時ディレクトリ上の自前 tag-palette.json で検証する。
// 本番 wwwroot/tag-palette.json には依存しない (内容非依存を担保)。
// ContentRootPath/wwwroot/tag-palette.json を読むため wwwroot を作って書き込む。
public sealed class TagPaletteCatalogTests : IDisposable
{
    private readonly string _root;
    private readonly string _wwwroot;
    private readonly string _paletteJson;

    public TagPaletteCatalogTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "dollama_palette_test_" + Guid.NewGuid().ToString("N"));
        _wwwroot = Path.Combine(_root, "wwwroot");
        Directory.CreateDirectory(_wwwroot);
        _paletteJson = Path.Combine(_wwwroot, "tag-palette.json");
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

    private TagPaletteCatalog Build()
    {
        return new TagPaletteCatalog(new FakeHostEnvironment { ContentRootPath = _root });
    }

    // (1) 正常 JSON でカテゴリ数・タグ数が一致する (PropertyNameCaseInsensitive 込み)。
    [Fact]
    public void Load_ValidJson_MatchesCounts()
    {
        File.WriteAllText(_paletteJson, """
        [
          { "Category": "人数", "Tags": ["1girl", "1boy", "solo"] },
          { "category": "表情", "tags": ["smile", "blush"] }
        ]
        """);

        var cat = Build();
        Assert.Equal(2, cat.Categories.Count);
        Assert.Equal("人数", cat.Categories[0].Category);
        Assert.Equal(3, cat.Categories[0].Tags.Count);
        Assert.Equal("表情", cat.Categories[1].Category);
        Assert.Equal(2, cat.Categories[1].Tags.Count);
        Assert.Equal("smile", cat.Categories[1].Tags[0]);
    }

    // (2) ファイル不在のとき空 (起動を止めない)。
    [Fact]
    public void Load_FileAbsent_IsEmpty()
    {
        Assert.False(File.Exists(_paletteJson));
        var cat = Build();
        Assert.Empty(cat.Categories);
    }

    // (3) 壊れ JSON のとき空 (例外を投げない)。
    [Fact]
    public void Load_CorruptJson_IsEmpty()
    {
        File.WriteAllText(_paletteJson, "{ not a category array )))");
        var cat = Build();
        Assert.Empty(cat.Categories);
    }
}
