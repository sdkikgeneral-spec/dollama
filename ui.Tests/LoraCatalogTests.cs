using System.Text.Json;
using Dollama.Ui.Services;
using Xunit;

namespace Dollama.Ui.Tests;

// LoraCatalog を一時ディレクトリ上の自前 loras.json で検証する。
// 本番 wwwroot/loras.json には依存しない (TagPaletteCatalogTests と同じ作法)。
// ContentRootPath/wwwroot/loras.json を読むため wwwroot を作って書き込む。
public sealed class LoraCatalogTests : IDisposable
{
    private readonly string _root;
    private readonly string _wwwroot;
    private readonly string _lorasJson;

    public LoraCatalogTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "dollama_lora_test_" + Guid.NewGuid().ToString("N"));
        _wwwroot = Path.Combine(_root, "wwwroot");
        Directory.CreateDirectory(_wwwroot);
        _lorasJson = Path.Combine(_wwwroot, "loras.json");
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

    private LoraCatalog Build()
    {
        return new LoraCatalog(new FakeHostEnvironment { ContentRootPath = _root });
    }

    // (1) 正常 JSON で件数・値が一致する (PropertyNameCaseInsensitive 込み)。
    [Fact]
    public void Load_ValidJson_MatchesEntries()
    {
        File.WriteAllText(_lorasJson, """
        [
          { "Name": "anime_style_v1", "Label": "アニメ画風 v1", "DefaultStrength": 0.8 },
          { "name": "detail_enhancer", "label": "ディテール強調", "defaultStrength": 0.6 }
        ]
        """);

        var cat = Build();
        Assert.Equal(2, cat.Entries.Count);
        Assert.Equal("anime_style_v1", cat.Entries[0].Name);
        Assert.Equal("アニメ画風 v1", cat.Entries[0].Label);
        Assert.Equal(0.8f, cat.Entries[0].DefaultStrength, 3);
        Assert.Equal("detail_enhancer", cat.Entries[1].Name);
        Assert.Equal(0.6f, cat.Entries[1].DefaultStrength, 3);
    }

    // defaultStrength 省略時は既定 1.0。
    [Fact]
    public void Load_MissingStrength_DefaultsToOne()
    {
        File.WriteAllText(_lorasJson, """
        [ { "name": "no_strength", "label": "強度なし" } ]
        """);

        var cat = Build();
        Assert.Single(cat.Entries);
        Assert.Equal(1.0f, cat.Entries[0].DefaultStrength, 3);
    }

    // (2) ファイル不在のとき空 (起動を止めない)。
    [Fact]
    public void Load_FileAbsent_IsEmpty()
    {
        Assert.False(File.Exists(_lorasJson));
        var cat = Build();
        Assert.Empty(cat.Entries);
    }

    // (3) 壊れ JSON のとき空 (例外を投げない)。
    [Fact]
    public void Load_CorruptJson_IsEmpty()
    {
        File.WriteAllText(_lorasJson, "{ not a lora array )))");
        var cat = Build();
        Assert.Empty(cat.Entries);
    }
}

// GenerationRequest への LoRA マッピングとシリアライズ契約を検証する。
// PL ガード1 (空選択で loras キーを出さない) を固定する。
public sealed class LoraRequestMappingTests
{
    // シリアライズは snake_case・日本語非エスケープの本番方針に合わせる必要はないが、
    // ここでは既定オプションでキー名 (JsonPropertyName) を確認する。
    private static readonly JsonSerializerOptions Opts = new();

    // 選択あり → loras キー + snake_case (name/strength) が出る。
    [Fact]
    public void Serialize_WithLoras_EmitsSnakeCaseKeys()
    {
        var req = new GenerationRequest
        {
            Prompt = "1girl",
            Loras = new List<LoraSpec>
            {
                new LoraSpec { Name = "anime_style_v1", Strength = 0.8f },
                new LoraSpec { Name = "detail_enhancer", Strength = 1.5f },
            },
        };

        var json = JsonSerializer.Serialize(req, Opts);

        Assert.Contains("\"loras\"", json);
        Assert.Contains("\"name\":\"anime_style_v1\"", json);
        Assert.Contains("\"strength\":0.8", json);
        Assert.Contains("\"name\":\"detail_enhancer\"", json);
        Assert.Contains("\"strength\":1.5", json);

        // PascalCase に流れていないこと。
        Assert.DoesNotContain("\"Name\"", json);
        Assert.DoesNotContain("\"Strength\"", json);
        Assert.DoesNotContain("\"Loras\"", json);
    }

    // 空選択 (null) → loras キーが一切出ない (従来経路無改変・ガード1)。
    [Fact]
    public void Serialize_NoLoras_OmitsLorasKey()
    {
        var req = new GenerationRequest
        {
            Prompt = "1girl",
            Loras = null,
        };

        var json = JsonSerializer.Serialize(req, Opts);

        Assert.DoesNotContain("loras", json);
    }

    // ラウンドトリップで name/strength が保たれる。
    [Fact]
    public void RoundTrip_PreservesNameAndStrength()
    {
        var req = new GenerationRequest
        {
            Prompt = "1girl",
            Loras = new List<LoraSpec>
            {
                new LoraSpec { Name = "x", Strength = 0.35f },
            },
        };

        var json = JsonSerializer.Serialize(req, Opts);
        var back = JsonSerializer.Deserialize<GenerationRequest>(json, Opts);

        Assert.NotNull(back);
        Assert.NotNull(back!.Loras);
        Assert.Single(back.Loras!);
        Assert.Equal("x", back.Loras![0].Name);
        Assert.Equal(0.35f, back.Loras[0].Strength, 3);
    }
}
