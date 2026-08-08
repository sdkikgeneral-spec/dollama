using System.Reflection;
using Dollama.Ui.Services;
using Xunit;

namespace Dollama.Ui.Tests;

// PreviewHistory / PreviewItem (直近生成のミニ履歴 / docs/ui-brushup-plan.md §5 P3-5) の検証。
//
// 履歴はセッション内メモリのみ・永続化しない。並び替え規則を razor に書かず純クラスへ出し、
// ここで全数に近い形で固定する (GenerateGate と同じ流儀)。
//
// ★ 本ファイルの最重要項目は末尾の「breadth の錠前」= public メンバ集合の完全一致検査。
//   Remove / Reorder / Pin / Clear / Save を足した瞬間にテストが赤くなり、
//   「直近 4 枚を見せて戻すだけ」という P3-5 の境界を越えようとした手が止まる。
public sealed class PreviewHistoryTests
{
    // ────────────────────────────────────────────────
    // テスト用の item 生成。CreatedAt は既定で **Id と逆順** に振ってある
    // (並び順が CreatedAt ではなく投入順で決まることを常に踏むため)。
    // ────────────────────────────────────────────────
    // (Id は int.MinValue / MaxValue も渡ってくるので、時刻の計算は剰余で潰しておく)
    private static PreviewItem Item(int id) => Item(id, DateTimeOffset.UnixEpoch.AddMinutes(100 - (id % 1000)));

    private static PreviewItem Item(int id, DateTimeOffset createdAt)
        => new(id, $"b64-{id}", $"本番 {id}²", $"dollama_{id}.png", createdAt);

    // 1..count を順に Add した履歴 (= 新しい順に count, count-1, ...)。
    private static IReadOnlyList<PreviewItem> Build(int count)
    {
        IReadOnlyList<PreviewItem> history = Array.Empty<PreviewItem>();
        for (var i = 1; i <= count; i++)
        {
            history = PreviewHistory.Add(history, Item(i));
        }
        return history;
    }

    private static int[] Ids(IReadOnlyList<PreviewItem> items) => items.Select(i => i.Id).ToArray();

    // ════════════════════════════════════════════════
    // (1) 容量
    // ════════════════════════════════════════════════

    // 上限は 4 枚。直接固定する (「4」は右ペイン下 1 行に収まる枚数として決めた値)。
    [Fact]
    public void Capacity_IsFour()
    {
        Assert.Equal(4, PreviewHistory.Capacity);
    }

    // 投入 n 件 → 長さは min(n, 4)。
    [Theory]
    [InlineData(0, 0)]
    [InlineData(1, 1)]
    [InlineData(2, 2)]
    [InlineData(3, 3)]
    [InlineData(4, 4)]
    [InlineData(5, 4)]
    [InlineData(6, 4)]
    [InlineData(20, 4)]
    public void Add_LengthIsMinOfCountAndCapacity(int count, int expected)
    {
        Assert.Equal(expected, Build(count).Count);
    }

    // 並びは常に新しい順 (直近が先頭)。
    [Theory]
    [InlineData(1, new[] { 1 })]
    [InlineData(2, new[] { 2, 1 })]
    [InlineData(3, new[] { 3, 2, 1 })]
    [InlineData(4, new[] { 4, 3, 2, 1 })]
    [InlineData(5, new[] { 5, 4, 3, 2 })]
    [InlineData(6, new[] { 6, 5, 4, 3 })]
    [InlineData(20, new[] { 20, 19, 18, 17 })]
    public void Add_KeepsNewestFirst(int count, int[] expectedIds)
    {
        Assert.Equal(expectedIds, Ids(Build(count)));
    }

    // 5 件目で最古 (Id=1) が落ちる。落ちたものは Find でも引けない。
    [Fact]
    public void Add_FifthDropsTheOldest()
    {
        var four = Build(4);
        Assert.NotNull(PreviewHistory.Find(four, 1));

        var five = PreviewHistory.Add(four, Item(5));

        Assert.Equal(new[] { 5, 4, 3, 2 }, Ids(five));
        Assert.Null(PreviewHistory.Find(five, 1));
    }

    // 上限を超えた列を渡されても (別経路で壊れた状態でも) 4 枚に切り詰めて返す。
    [Fact]
    public void Add_TruncatesAnOverlongInput()
    {
        var overlong = new List<PreviewItem> { Item(6), Item(5), Item(4), Item(3), Item(2), Item(1) };

        var next = PreviewHistory.Add(overlong, Item(7));

        Assert.Equal(new[] { 7, 6, 5, 4 }, Ids(next));
        Assert.Equal(6, overlong.Count); // 入力は切り詰めない
    }

    // ════════════════════════════════════════════════
    // (2) 非 Mutate (P2-1 で AddOne の Mutate を是正した前例あり)
    // ════════════════════════════════════════════════

    // 元リストの Count・要素・順序が一切変わらないこと。上限未満と上限到達の両方で見る。
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(4)]   // ← 切り詰めが起きる側
    [InlineData(6)]
    public void Add_DoesNotMutateTheInput(int count)
    {
        var current = new List<PreviewItem>();
        for (var i = count; i >= 1; i--)
        {
            current.Add(Item(i));
        }
        var before = current.ToArray();

        PreviewHistory.Add(current, Item(99));

        Assert.Equal(before.Length, current.Count);
        for (var i = 0; i < before.Length; i++)
        {
            Assert.Same(before[i], current[i]); // 要素も順序も同一 (参照ごと)
        }
    }

    // 常に新しいインスタンスを返す (同一参照を返さない)。
    // 同じ参照を返すと Blazor の差分描画から見て「変化なし」に見え、画面が更新されない。
    [Fact]
    public void Add_AlwaysReturnsANewInstance()
    {
        IReadOnlyList<PreviewItem> current = new List<PreviewItem> { Item(1) };

        var a = PreviewHistory.Add(current, Item(2));
        var b = PreviewHistory.Add(current, Item(2));

        Assert.NotSame(current, a);
        Assert.NotSame(current, b);
        Assert.NotSame(a, b);
        Assert.Equal(Ids(a), Ids(b));
    }

    // 空履歴からの 1 枚目 (実装の分岐で最も踏まれる経路)。
    [Fact]
    public void Add_WorksFromAnEmptyHistory()
    {
        foreach (IReadOnlyList<PreviewItem> empty in new IReadOnlyList<PreviewItem>[]
                 {
                     Array.Empty<PreviewItem>(),
                     new List<PreviewItem>(),
                 })
        {
            var next = PreviewHistory.Add(empty, Item(1));

            Assert.Equal(new[] { 1 }, Ids(next));
            Assert.Empty(empty);
        }
    }

    // 既存要素はコピーせず参照のまま持ち回る (base64 文字列を複製しない)。
    [Fact]
    public void Add_CarriesExistingItemsByReference()
    {
        var first = Item(1);
        var history = PreviewHistory.Add(Array.Empty<PreviewItem>(), first);

        var next = PreviewHistory.Add(history, Item(2));

        Assert.Same(first, next[1]);
    }

    // 防御: null 列 / null 要素 / null item でも落ちない (描画経路で落とせない)。
    [Fact]
    public void Add_IsDefensiveAgainstNulls()
    {
        var fromNull = PreviewHistory.Add(null!, Item(1));
        Assert.Equal(new[] { 1 }, Ids(fromNull));

        var withNullElement = new List<PreviewItem> { Item(2), null!, Item(1) };
        var cleaned = PreviewHistory.Add(withNullElement, Item(3));
        Assert.Equal(new[] { 3, 2, 1 }, Ids(cleaned));

        var nullItem = PreviewHistory.Add(Build(2), null!);
        Assert.Equal(new[] { 2, 1 }, Ids(nullItem));

        Assert.Empty(PreviewHistory.Add(null!, null!));
    }

    // 同じ Id を 2 度入れても弾かない (重複排除はしない = 余計な規則を持ち込まない)。
    [Fact]
    public void Add_DoesNotDeduplicateIds()
    {
        var history = PreviewHistory.Add(Build(1), Item(1));

        Assert.Equal(new[] { 1, 1 }, Ids(history));
    }

    // ════════════════════════════════════════════════
    // (3) 選択 (Find / IsSelected)
    // ════════════════════════════════════════════════

    [Theory]
    [InlineData(4, true)]
    [InlineData(3, true)]
    [InlineData(2, true)]
    [InlineData(1, true)]
    [InlineData(0, false)]
    [InlineData(5, false)]
    [InlineData(-1, false)]
    [InlineData(int.MaxValue, false)]
    [InlineData(int.MinValue, false)]
    public void Find_HitsOnlyExistingIds(int id, bool expectHit)
    {
        var found = PreviewHistory.Find(Build(4), id);

        if (expectHit)
        {
            Assert.NotNull(found);
            Assert.Equal(id, found!.Id);
        }
        else
        {
            Assert.Null(found);
        }
    }

    // 空リスト / null リストでも例外を投げず null。
    [Fact]
    public void Find_IsSafeOnEmptyAndNull()
    {
        Assert.Null(PreviewHistory.Find(Array.Empty<PreviewItem>(), 1));
        Assert.Null(PreviewHistory.Find(new List<PreviewItem>(), 1));
        Assert.Null(PreviewHistory.Find(null!, 1));
    }

    // 見つけたものは同一参照で返す (再表示で base64 を作り直さない)。
    [Fact]
    public void Find_ReturnsTheSameInstance()
    {
        var target = Item(7);
        var history = PreviewHistory.Add(Build(2), target);

        Assert.Same(target, PreviewHistory.Find(history, 7));
    }

    // Id が重複していたら先頭 (= より新しい方) を返す。決定的であることが要点。
    [Fact]
    public void Find_PrefersTheHeadWhenIdsCollide()
    {
        var older = Item(1, DateTimeOffset.UnixEpoch);
        var newer = Item(1, DateTimeOffset.UnixEpoch.AddDays(1));
        var history = PreviewHistory.Add(PreviewHistory.Add(Array.Empty<PreviewItem>(), older), newer);

        Assert.Same(newer, PreviewHistory.Find(history, 1));
        Assert.Same(newer, PreviewHistory.Find(history, 1)); // 2 回呼んでも同じ
    }

    // null 要素が混ざっていても Find は落ちない。
    [Fact]
    public void Find_SkipsNullElements()
    {
        var items = new List<PreviewItem> { null!, Item(1) };

        Assert.Equal(1, PreviewHistory.Find(items, 1)!.Id);
        Assert.Null(PreviewHistory.Find(items, 2));
    }

    [Theory]
    [InlineData(3, 3, true)]
    [InlineData(3, 4, false)]
    [InlineData(0, 0, true)]
    [InlineData(-1, -1, true)]
    [InlineData(int.MaxValue, int.MaxValue, true)]
    [InlineData(1, 0, false)]
    public void IsSelected_ComparesIdOnly(int itemId, int selectedId, bool expected)
    {
        Assert.Equal(expected, PreviewHistory.IsSelected(Item(itemId), selectedId));
    }

    // 未選択を表す番兵 (razor 側は 0 や -1 を使う) で誰も選択中にならないこと。
    [Fact]
    public void IsSelected_IsFalseForEveryItemWhenNothingIsSelected()
    {
        foreach (var item in Build(4))
        {
            Assert.False(PreviewHistory.IsSelected(item, 0));
            Assert.False(PreviewHistory.IsSelected(item, -1));
        }
        Assert.False(PreviewHistory.IsSelected(null!, 1));
    }

    // 履歴のうち選択中は高々 1 枚 (Id が一意に振られている限り)。
    [Fact]
    public void IsSelected_MarksAtMostOneItem()
    {
        var history = Build(4);

        foreach (var selected in new[] { 1, 2, 3, 4 })
        {
            Assert.Equal(1, history.Count(i => PreviewHistory.IsSelected(i, selected)));
        }
    }

    // ════════════════════════════════════════════════
    // (4) 決定性 (CreatedAt に依存しない)
    // ════════════════════════════════════════════════

    // CreatedAt をどう振っても並びは投入順だけで決まる。
    // (時刻でソートすると、時計の巻き戻しや同一秒で順序が入れ替わる)
    [Fact]
    public void Order_DoesNotDependOnCreatedAt()
    {
        var stamps = new[]
        {
            DateTimeOffset.MaxValue,
            DateTimeOffset.MinValue,
            DateTimeOffset.UnixEpoch,
            DateTimeOffset.UnixEpoch,           // 同一時刻
            DateTimeOffset.UnixEpoch.AddDays(-1) // 巻き戻し
        };

        IReadOnlyList<PreviewItem> history = Array.Empty<PreviewItem>();
        for (var i = 0; i < stamps.Length; i++)
        {
            history = PreviewHistory.Add(history, Item(i + 1, stamps[i]));
        }

        Assert.Equal(new[] { 5, 4, 3, 2 }, Ids(history));
    }

    // 同じ入力なら何度呼んでも同じ出力 (隠れた状態を持たない)。
    [Fact]
    public void Add_IsPureForTheSameInput()
    {
        var current = Build(3);
        var item = Item(9);

        var a = PreviewHistory.Add(current, item);
        var b = PreviewHistory.Add(current, item);

        Assert.Equal(Ids(a), Ids(b));
        for (var i = 0; i < a.Count; i++)
        {
            Assert.Same(a[i], b[i]);
        }
    }

    // 同じ手順を 2 回踏めば同じ履歴になる (静的クラスが状態を溜めていないこと)。
    [Fact]
    public void History_IsReproducibleAcrossRuns()
    {
        Assert.Equal(Ids(Build(7)), Ids(Build(7)));
    }

    // ════════════════════════════════════════════════
    // (5) breadth の錠前 — ここが P3-5 の境界の実効装置
    // ════════════════════════════════════════════════

    // PreviewHistory の public メンバはこの 4 つで確定。
    // Remove / Reorder / Pin / Clear / Save / Load を足すと即座に赤くなる。
    private static readonly string[] ExpectedSurface =
    {
        "Field:Capacity",
        "Method:Add",
        "Method:Find",
        "Method:IsSelected",
    };

    private static string[] PublicSurfaceOf(Type type)
        => type.GetMembers(BindingFlags.Public | BindingFlags.Static | BindingFlags.Instance | BindingFlags.DeclaredOnly)
               .Select(m => $"{m.MemberType}:{m.Name}")
               .OrderBy(s => s, StringComparer.Ordinal)
               .ToArray();

    [Fact]
    public void PublicSurface_IsExactlyCapacityAddFindIsSelected()
    {
        var actual = PublicSurfaceOf(typeof(PreviewHistory));

        Assert.Equal(
            ExpectedSurface.OrderBy(s => s, StringComparer.Ordinal).ToArray(),
            actual);
    }

    // 入れ子型で API を増やす抜け道も塞ぐ。
    [Fact]
    public void PreviewHistory_HasNoNestedTypes()
    {
        Assert.Empty(typeof(PreviewHistory).GetNestedTypes(BindingFlags.Public | BindingFlags.NonPublic));
    }

    // 検査器の自己検査 (変異検査の常設版):
    // Clear() を 1 本足した写しに対しては、同じ抽出が 5 件になり期待集合と食い違う。
    // = 上の検査が空振りでないことの証明。
    private static class PreviewHistoryWithClear
    {
        public const int Capacity = 4;
        public static IReadOnlyList<PreviewItem> Add(IReadOnlyList<PreviewItem> current, PreviewItem item) => current;
        public static PreviewItem? Find(IReadOnlyList<PreviewItem> items, int id) => null;
        public static bool IsSelected(PreviewItem item, int selectedId) => false;
        public static IReadOnlyList<PreviewItem> Clear() => Array.Empty<PreviewItem>();
    }

    [Fact]
    public void SurfaceCheck_WouldFailIfClearWereAdded()
    {
        var mutated = PublicSurfaceOf(typeof(PreviewHistoryWithClear));

        Assert.Contains("Method:Clear", mutated);
        Assert.Equal(5, mutated.Length);
        Assert.NotEqual(ExpectedSurface.OrderBy(s => s, StringComparer.Ordinal).ToArray(), mutated);
    }

    // PreviewItem が持つのは 5 つの値だけ。
    [Fact]
    public void PreviewItem_ExposesExactlyFiveProperties()
    {
        var props = typeof(PreviewItem)
            .GetProperties(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
            .Select(p => p.Name)
            .OrderBy(n => n, StringComparer.Ordinal)
            .ToArray();

        Assert.Equal(
            new[] { "Badge", "Base64", "CreatedAt", "DownloadName", "Id" },
            props);
    }

    // ★ PL 裁定: 履歴に byte[] を持たせない。
    //   (毎描画の base64 化を避けるため base64 だけを持ち、生バイトは選択時に 1 回 decode する)
    //   ここが赤い = 4 枚分の PNG 生バイトが常駐し始めた、という設計逸脱の合図。
    [Fact]
    public void PreviewItem_HoldsNoRawBytes()
    {
        var types = typeof(PreviewItem)
            .GetProperties(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
            .Select(p => p.PropertyType)
            .Concat(typeof(PreviewItem)
                .GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.DeclaredOnly)
                .Where(f => !f.Name.Contains("k__BackingField", StringComparison.Ordinal))
                .Select(f => f.FieldType))
            .ToArray();

        foreach (var t in types)
        {
            Assert.False(t.IsArray, $"配列を持たせない (byte[] 常駐の入口): {t.Name}");
        }

        // 許す型は int / string / DateTimeOffset の 3 つだけ。
        var allowed = new[] { typeof(int), typeof(string), typeof(DateTimeOffset) };
        foreach (var t in types)
        {
            Assert.Contains(t, allowed);
        }
    }
}
