using Dollama.Ui.Services;
using Microsoft.Extensions.Configuration;
using Xunit;

namespace Dollama.Ui.Tests;

// EndpointRegistry を一時ディレクトリ上で検証する。
// FakeHostEnvironment は PresetStoreTests.cs で定義済み (internal・同一名前空間で共有)。
public sealed class EndpointRegistryTests : IDisposable
{
    private readonly string _root;
    private readonly string _dataDir;
    private readonly string _endpointsJson;

    public EndpointRegistryTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "dollama_ui_test_ep_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_root);
        _dataDir = Path.Combine(_root, "data");
        _endpointsJson = Path.Combine(_dataDir, "endpoints.json");
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

    private FakeHostEnvironment Env() => new() { ContentRootPath = _root };

    private static IConfiguration BuildConfig(Dictionary<string, string?> values) =>
        new ConfigurationBuilder().AddInMemoryCollection(values).Build();

    // (1) appsettings のみ (JSON なし) → 設定由来 1 件が Current になる。
    [Fact]
    public void ConfigOnly_NoJson_UsesConfigEndpoint()
    {
        var config = BuildConfig(new Dictionary<string, string?>
        {
            ["Dollama:Endpoints:0:Name"] = "a",
            ["Dollama:Endpoints:0:Url"] = "http://host-a:8080",
        });
        var registry = new EndpointRegistry(config, Env());

        var ep = Assert.Single(registry.All);
        Assert.Equal("a", ep.Name);
        Assert.Equal("http://host-a:8080", ep.Url);
        Assert.True(ep.FromConfig);
        Assert.Equal("a", registry.Current.Name);
    }

    // (2) JSON に config と同名のエントリがあれば URL は JSON 優先。ただし FromConfig は維持 (削除不可のまま)。
    [Fact]
    public void JsonMerge_NameDuplicate_JsonUrlWins()
    {
        Directory.CreateDirectory(_dataDir);
        File.WriteAllText(_endpointsJson, """[{"name":"local","url":"http://json-wins:9999"}]""");

        var config = BuildConfig(new Dictionary<string, string?>
        {
            ["Dollama:Endpoints:0:Name"] = "local",
            ["Dollama:Endpoints:0:Url"] = "http://config-loses:8080",
        });
        var registry = new EndpointRegistry(config, Env());

        var ep = Assert.Single(registry.All);
        Assert.Equal("local", ep.Name);
        Assert.Equal("http://json-wins:9999", ep.Url);
        Assert.True(ep.FromConfig); // 名前が config と衝突する限り削除不可のまま
    }

    // (3) 壊れ JSON → 例外を投げず、設定由来のみで空復帰する。
    [Fact]
    public void CorruptJson_FallsBackToConfigOnly()
    {
        Directory.CreateDirectory(_dataDir);
        File.WriteAllText(_endpointsJson, "{not valid json");

        var config = BuildConfig(new Dictionary<string, string?>
        {
            ["Dollama:Endpoints:0:Name"] = "a",
            ["Dollama:Endpoints:0:Url"] = "http://host-a:8080",
        });
        var registry = new EndpointRegistry(config, Env());

        var ep = Assert.Single(registry.All);
        Assert.Equal("a", ep.Name);
    }

    // (4) Add → 永続化 → 新しい Registry インスタンスで再読込しても残っている。
    [Fact]
    public void Add_PersistsAcrossNewInstance()
    {
        var config = BuildConfig(new Dictionary<string, string?>
        {
            ["Dollama:Endpoints:0:Name"] = "a",
            ["Dollama:Endpoints:0:Url"] = "http://host-a:8080",
        });
        var registry = new EndpointRegistry(config, Env());

        Assert.True(registry.Add("b", "http://host-b:8080"));
        Assert.True(File.Exists(_endpointsJson));

        var reloaded = new EndpointRegistry(config, Env());
        var b = Assert.Single(reloaded.All, e => e.Name == "b");
        Assert.Equal("http://host-b:8080", b.Url);
        Assert.False(b.FromConfig);
    }

    // (6) Dollama:Endpoints に項目はあるが Name/Url が全部空白 → フィルタで全落ちしても
    //     BaseUrl (無ければ既定値) へフォールバックし、Current アクセスで例外にならない。
    [Fact]
    public void ConfigEndpoints_AllBlank_FallsBackToBaseUrl()
    {
        var config = BuildConfig(new Dictionary<string, string?>
        {
            ["Dollama:Endpoints:0:Name"] = "  ",
            ["Dollama:Endpoints:0:Url"] = "",
            ["Dollama:BaseUrl"] = "http://fallback-host:8080",
        });
        var registry = new EndpointRegistry(config, Env());

        var ep = Assert.Single(registry.All);
        Assert.Equal("local", ep.Name);
        Assert.Equal("http://fallback-host:8080", ep.Url);
        Assert.True(ep.FromConfig);
        Assert.Equal("local", registry.Current.Name);
    }

    // (5) FromConfig な項目は Remove しても消えず false を返す。
    [Fact]
    public void Remove_FromConfigEndpoint_ReturnsFalseAndStays()
    {
        var config = BuildConfig(new Dictionary<string, string?>
        {
            ["Dollama:Endpoints:0:Name"] = "local",
            ["Dollama:Endpoints:0:Url"] = "http://127.0.0.1:8080",
        });
        var registry = new EndpointRegistry(config, Env());

        Assert.False(registry.Remove("local"));
        Assert.Single(registry.All);
        Assert.Equal("local", registry.Current.Name);
    }

    // (7) 不正な URL / 空入力は Add が false を返し、一覧にも JSON にも足さない。
    [Theory]
    [InlineData("", "http://host-b:8080")]      // 名前が空
    [InlineData("   ", "http://host-b:8080")]   // 名前が空白のみ
    [InlineData("b", "")]                       // URL が空
    [InlineData("b", "host-b:8080")]            // scheme 無し (絶対 URL でない)
    [InlineData("b", "/v1/images")]             // 相対 URL
    [InlineData("b", "ftp://host-b/x")]         // http/https 以外の scheme
    [InlineData("b", "file:///C:/x")]           // ローカルファイル scheme
    public void Add_InvalidInput_RejectedAndNotPersisted(string name, string url)
    {
        var registry = new EndpointRegistry(DefaultConfig(), Env());

        Assert.False(registry.Add(name, url));
        Assert.Single(registry.All);                 // 設定由来の 1 件のまま
        Assert.False(File.Exists(_endpointsJson));   // 書き込みも起きない
    }

    // (8) 同名を再 Add すると重複せず URL が上書きされる。
    [Fact]
    public void Add_SameName_OverwritesUrlWithoutDuplicating()
    {
        var registry = new EndpointRegistry(DefaultConfig(), Env());

        Assert.True(registry.Add("b", "http://host-b:8080"));
        Assert.True(registry.Add("b", "http://host-b:9999"));

        var b = Assert.Single(registry.All, e => e.Name == "b");
        Assert.Equal("http://host-b:9999", b.Url);
        Assert.Equal(2, registry.All.Count);  // a (config) + b のみ
    }

    // (9) JSON 由来なら Remove できる。選択中を消したら Current は残りの先頭へ戻る。
    [Fact]
    public void Remove_JsonEndpoint_RemovesAndFallsBackCurrent()
    {
        var registry = new EndpointRegistry(DefaultConfig(), Env());
        Assert.True(registry.Add("b", "http://host-b:8080"));
        Assert.True(registry.Select("b"));
        Assert.Equal("b", registry.Current.Name);

        Assert.True(registry.Remove("b"));

        Assert.Single(registry.All);
        Assert.Equal("a", registry.Current.Name);   // 残った先頭 (設定由来) へ戻る

        // 永続化側からも消えている (新インスタンスで再読込しても復活しない)
        var reloaded = new EndpointRegistry(DefaultConfig(), Env());
        Assert.DoesNotContain(reloaded.All, e => e.Name == "b");
    }

    // (10) 存在しない名前の Select / Remove は例外でなく false で、状態も動かさない。
    [Fact]
    public void SelectAndRemove_UnknownName_ReturnFalseAndKeepState()
    {
        var registry = new EndpointRegistry(DefaultConfig(), Env());

        Assert.False(registry.Select("nope"));
        Assert.False(registry.Remove("nope"));
        Assert.Equal("a", registry.Current.Name);
        Assert.Single(registry.All);
    }

    private static IConfiguration DefaultConfig() => BuildConfig(new Dictionary<string, string?>
    {
        ["Dollama:Endpoints:0:Name"] = "a",
        ["Dollama:Endpoints:0:Url"] = "http://host-a:8080",
    });
}
