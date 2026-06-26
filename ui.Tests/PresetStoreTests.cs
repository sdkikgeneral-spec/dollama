using System.Text.Json;
using Dollama.Ui.Services;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Hosting;
using SixLabors.ImageSharp;
using Xunit;

namespace Dollama.Ui.Tests;

// ContentRootPath を一時ディレクトリに差し込むための最小 IHostEnvironment フェイク。
// PresetStore は ContentRootPath しか見ないので他は適当な既定で十分。
internal sealed class FakeHostEnvironment : IHostEnvironment
{
    public string ApplicationName { get; set; } = "Dollama.Ui.Tests";
    public string EnvironmentName { get; set; } = "Test";
    public string ContentRootPath { get; set; } = "";
    public IFileProvider ContentRootFileProvider { get; set; } = new NullFileProvider();
}

// PresetStore のサムネイル機能を一時ディレクトリ上で検証する。
// 各テストは IDisposable で一時ディレクトリを掃除する。
public sealed class PresetStoreTests : IDisposable
{
    private readonly string _root;
    private readonly string _dataDir;
    private readonly string _thumbsDir;
    private readonly string _presetsJson;
    private readonly PresetStore _store;

    public PresetStoreTests()
    {
        // テストごとに固有の一時ディレクトリを ContentRootPath として使う。
        _root = Path.Combine(Path.GetTempPath(), "dollama_ui_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_root);
        _dataDir = Path.Combine(_root, "data");
        _thumbsDir = Path.Combine(_dataDir, "thumbs");
        _presetsJson = Path.Combine(_dataDir, "presets.json");
        _store = new PresetStore(new FakeHostEnvironment { ContentRootPath = _root });
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

    // 任意サイズの単色 PNG バイト列を作る (テスト入力用)。
    private static byte[] MakePng(int w, int h)
    {
        using var image = new Image<SixLabors.ImageSharp.PixelFormats.Rgba32>(w, h);
        using var ms = new MemoryStream();
        image.SaveAsPng(ms);
        return ms.ToArray();
    }

    // presets.json を生の JSON 文字列としてそのまま書き込む (後方互換テスト用)。
    private void WriteRawPresets(string json)
    {
        Directory.CreateDirectory(_dataDir);
        File.WriteAllText(_presetsJson, json);
    }

    // (1) サムネ付き Save → thumbs/*.png 生成・Thumbnail 設定・presets.json に thumbnail 出力。
    [Fact]
    public void Save_WithThumbnail_WritesPngAndSetsThumbnailField()
    {
        var preset = new TagPreset { Name = "fav", Kind = "prompt", Tags = new() { "1girl" } };
        _store.Save(preset, MakePng(64, 48));

        Assert.False(string.IsNullOrEmpty(preset.Thumbnail));
        var thumbPath = Path.Combine(_thumbsDir, preset.Thumbnail!);
        Assert.True(File.Exists(thumbPath));

        // presets.json に thumbnail フィールドが出力されている。
        var json = File.ReadAllText(_presetsJson);
        Assert.Contains("\"thumbnail\"", json);
        Assert.Contains(preset.Thumbnail!, json);

        // 読み戻しても Thumbnail が残る。
        var loaded = _store.All("prompt").Single();
        Assert.Equal(preset.Thumbnail, loaded.Thumbnail);
    }

    // (2) 256px 入力 → 保存サムネを ImageSharp で復号し最大辺 ≤ 128。
    [Fact]
    public void Save_LargeInput_ResizesToMax128()
    {
        var preset = new TagPreset { Name = "big", Kind = "prompt" };
        _store.Save(preset, MakePng(256, 200));

        var thumbPath = Path.Combine(_thumbsDir, preset.Thumbnail!);
        using var img = Image.Load(thumbPath);
        Assert.True(Math.Max(img.Width, img.Height) <= 128,
            $"最大辺 {Math.Max(img.Width, img.Height)} が 128 を超えています");
        // アスペクト比が維持されている (256x200 → 128x100)。
        Assert.Equal(128, img.Width);
        Assert.Equal(100, img.Height);
    }

    // (3) Delete → json エントリと thumb PNG の両方が消える。
    [Fact]
    public void Delete_RemovesEntryAndThumbPng()
    {
        var preset = new TagPreset { Name = "todel", Kind = "prompt" };
        _store.Save(preset, MakePng(64, 64));
        var thumbPath = Path.Combine(_thumbsDir, preset.Thumbnail!);
        Assert.True(File.Exists(thumbPath));

        _store.Delete("prompt", "todel");

        Assert.Empty(_store.All("prompt"));
        Assert.False(File.Exists(thumbPath));
    }

    // (4) 後方互換: thumbnail 無し旧 presets.json を置き All() が読め Thumbnail==null。
    [Fact]
    public void Load_LegacyJsonWithoutThumbnail_IsBackwardCompatible()
    {
        WriteRawPresets("""
        [
          { "name": "old", "kind": "prompt", "tags": ["1girl", "solo"] }
        ]
        """);

        var loaded = _store.All("prompt").Single();
        Assert.Equal("old", loaded.Name);
        Assert.Equal(2, loaded.Tags.Count);
        Assert.Null(loaded.Thumbnail);
    }

    // (5) ファイル名安全化: name に /・..・空・日本語・重複を入れても thumbs ディレクトリ外に書かれない。
    [Theory]
    [InlineData("prompt", "a/b")]
    [InlineData("prompt", "../escape")]
    [InlineData("prompt", "..")]
    [InlineData("prompt", "..\\win")]
    [InlineData("negative", "お気に入り")]
    [InlineData("prompt", "  .leading.")]
    public void Save_UnsafeName_StaysWithinThumbsDir(string kind, string name)
    {
        var preset = new TagPreset { Name = name, Kind = kind };
        _store.Save(preset, MakePng(32, 32));

        var thumbFull = Path.GetFullPath(Path.Combine(_thumbsDir, preset.Thumbnail!));
        var baseFull = Path.GetFullPath(_thumbsDir) + Path.DirectorySeparatorChar;
        Assert.StartsWith(baseFull, thumbFull, StringComparison.Ordinal);
        Assert.True(File.Exists(thumbFull));
        // ファイル名に区切り・親参照が残っていない。
        Assert.DoesNotContain("..", preset.Thumbnail!);
        Assert.DoesNotContain('/', preset.Thumbnail!);
        Assert.DoesNotContain('\\', preset.Thumbnail!);
    }

    // (5b) 日本語名はファイル名に保持される。
    [Fact]
    public void Save_JapaneseName_PreservedInFileName()
    {
        var preset = new TagPreset { Name = "お気に入り", Kind = "prompt" };
        _store.Save(preset, MakePng(32, 32));
        Assert.Contains("お気に入り", preset.Thumbnail!);
    }

    // (6) 上書き: 同 kind+name 再保存でサムネ更新 (?v=/LastWrite が変わる)。
    [Fact]
    public async Task Save_Overwrite_UpdatesThumbnailVersion()
    {
        var p1 = new TagPreset { Name = "dup", Kind = "prompt" };
        _store.Save(p1, MakePng(64, 64));
        var url1 = _store.ThumbnailUrl(_store.All("prompt").Single());

        // LastWrite の解像度差を避けるため少し待つ。
        await Task.Delay(50);

        var p2 = new TagPreset { Name = "dup", Kind = "prompt", Tags = new() { "new" } };
        _store.Save(p2, MakePng(80, 80));

        var all = _store.All("prompt");
        Assert.Single(all); // 同 kind+name は 1 件に上書き
        var url2 = _store.ThumbnailUrl(all.Single());

        Assert.NotNull(url1);
        Assert.NotNull(url2);
        Assert.NotEqual(url1, url2); // ?v= (LastWrite ticks) が変わる
    }

    // (7) Save(preset, null) → thumb 無し・タグは保存・Thumbnail==null。
    [Fact]
    public void Save_NullThumbnail_SavesTagsWithoutThumb()
    {
        var preset = new TagPreset { Name = "notthumb", Kind = "prompt", Tags = new() { "x", "y" } };
        _store.Save(preset, null);

        var loaded = _store.All("prompt").Single();
        Assert.Equal(2, loaded.Tags.Count);
        Assert.Null(loaded.Thumbnail);
        // thumbs ディレクトリ自体が作られていない (または空)。
        if (Directory.Exists(_thumbsDir))
        {
            Assert.Empty(Directory.GetFiles(_thumbsDir));
        }
    }

    // (8) ThumbnailUrl: サムネ無し=null / 有り=/thumb/...?v=...。
    [Fact]
    public void ThumbnailUrl_NullWhenAbsent_UrlWhenPresent()
    {
        // サムネ無しは null。
        var noThumb = new TagPreset { Name = "n", Kind = "prompt" };
        Assert.Null(_store.ThumbnailUrl(noThumb));

        // 実体があれば /thumb/...?v=...。
        var withThumb = new TagPreset { Name = "y", Kind = "prompt" };
        _store.Save(withThumb, MakePng(40, 40));
        var url = _store.ThumbnailUrl(_store.All("prompt").Single());
        Assert.NotNull(url);
        Assert.StartsWith("/thumb/", url);
        Assert.Contains("?v=", url);

        // Thumbnail フィールドはあるが実体ファイルが無いケースは null。
        var dangling = new TagPreset { Name = "d", Kind = "prompt", Thumbnail = "prompt_missing.png" };
        Assert.Null(_store.ThumbnailUrl(dangling));
    }
}
