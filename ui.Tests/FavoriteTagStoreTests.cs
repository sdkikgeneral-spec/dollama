using Dollama.Ui.Services;
using Xunit;

namespace Dollama.Ui.Tests;

// FavoriteTagStore を一時ディレクトリ上で検証する。
// PresetStoreTests と同じ FakeHostEnvironment を流用 (ContentRootPath 差し込み)。
// 各テストは IDisposable で一時ディレクトリを掃除する。
public sealed class FavoriteTagStoreTests : IDisposable
{
    private readonly string _root;
    private readonly string _dataDir;
    private readonly string _favoritesJson;
    private readonly FavoriteTagStore _store;

    public FavoriteTagStoreTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "dollama_fav_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_root);
        _dataDir = Path.Combine(_root, "data");
        _favoritesJson = Path.Combine(_dataDir, "favorites.json");
        _store = new FavoriteTagStore(new FakeHostEnvironment { ContentRootPath = _root });
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

    // favorites.json を生の JSON 文字列としてそのまま書き込む (壊れ JSON テスト用)。
    private void WriteRaw(string json)
    {
        Directory.CreateDirectory(_dataDir);
        File.WriteAllText(_favoritesJson, json);
    }

    // (1) Add は前後空白除去 + 小文字寄せで正規化される。
    [Fact]
    public void Add_NormalizesTrimAndLowercase()
    {
        _store.Add("  Long Hair  ");
        var all = _store.All();
        Assert.Single(all);
        Assert.Equal("long hair", all[0]);
    }

    // (2) Add は空 (空白のみ含む) を無視する。
    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("\t")]
    public void Add_IgnoresEmpty(string tag)
    {
        _store.Add(tag);
        Assert.Empty(_store.All());
    }

    // (3) Add は重複 (正規化後で同一) を無視する。
    [Fact]
    public void Add_IgnoresDuplicate()
    {
        _store.Add("smile");
        _store.Add("SMILE");
        _store.Add(" smile ");
        Assert.Single(_store.All());
    }

    // (4) Remove は正規化して照合し削除する。
    [Fact]
    public void Remove_DeletesNormalized()
    {
        _store.Add("1girl");
        _store.Add("solo");
        _store.Remove("  1GIRL ");

        var all = _store.All();
        Assert.Single(all);
        Assert.Equal("solo", all[0]);
    }

    // (5) All は登録順で返り、ファイルへ永続化されて新インスタンスから読める (往復)。
    [Fact]
    public void All_RoundTripsThroughFile()
    {
        _store.Add("a");
        _store.Add("b");
        _store.Add("c");

        // 別インスタンスで読み直しても順序込みで一致する。
        var reopened = new FavoriteTagStore(new FakeHostEnvironment { ContentRootPath = _root });
        Assert.Equal(new[] { "a", "b", "c" }, reopened.All());
    }

    // (6) ファイル不在のとき All は空。
    [Fact]
    public void All_EmptyWhenFileAbsent()
    {
        Assert.False(File.Exists(_favoritesJson));
        Assert.Empty(_store.All());
    }

    // (7) 壊れ JSON のとき All は空 (例外を投げない)。
    [Fact]
    public void All_EmptyWhenJsonCorrupt()
    {
        WriteRaw("{ this is not a json array ]]");
        Assert.Empty(_store.All());
    }

    // (8) アトミック書込: 書込前に正しい favorites.json があるとき、Add が完了するまで
    //     既存ファイルが破壊されない (.tmp 経由で置換され、最終的に有効な JSON が残る)。
    [Fact]
    public void Add_AtomicWrite_KeepsValidFile()
    {
        WriteRaw("[ \"existing\" ]");
        _store.Add("new");

        // .tmp が残っていない (Move で置換済み)。
        Assert.False(File.Exists(_favoritesJson + ".tmp"));

        // 既存 + 追加の両方が読める = 既存を破壊せず追記された。
        var all = _store.All();
        Assert.Equal(new[] { "existing", "new" }, all);
    }
}
