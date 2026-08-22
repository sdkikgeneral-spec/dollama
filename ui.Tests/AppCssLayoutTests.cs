using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace Dollama.Ui.Tests;

// P3-4 (中央列 400 → 460px + 右ペインの余白調整) と P3-2 (レスポンシブ再構成) の回帰止め。
// docs/ui-brushup-plan.md §5 「P3 バッチ E 実装メモ」参照。
//
// ★ 守りたい性質は 3 つ。どれも「壊れても画面はそれらしく描画されてしまう」種類なので機械で見る。
//   ① **デスクトップ非退行** — 縦積み専用の道具 (order / position: sticky) が
//      メディアクエリの外へ漏れないこと。漏れると 3 ペイン時のレイアウトが変わる。
//   ② **縦積みの導線** — 並びが 中央 → 右 → 左 (生成 → 結果 → パレット) であること。
//      **値の一致ではなく大小関係**で見る (後で番号を詰め直しても壊れない)。
//   ③ **貼り付いた行が読めること** — sticky 行が地 (--panel) と上境界を持つこと。
//      透明だと裏を流れるチップ列と重なって判読できない。
//
// ★ 加えて「新しい規則が P1-6 の実効 px ベースライン表 (AppCssScaleBaselineTests) を
//   1 行も動かしていない」ことも直接固定する。余白・字送り・角丸に触れずに
//   レイアウトを変えるのが本バッチの制約だった。
//
// 既存の CSS パーサ (Blocks / MediaBlocks / TopLevelCss / ParseDeclarations …) は
// AppCssTokenTests / TelemetryRoleTests の partial からそのまま共有する。
// bUnit は入れない (依存追加ゼロの既存方針どおり CSS をテキストとして読む)。
public sealed partial class AppCssTokenTests
{
    // 3 ペインのトラック定義。**変えたのは中央の上限だけ** (400 → 460) という
    // P3-4 の約束を、左固定幅と中央下限も含めて固定する。
    private const int LeftColumnPx = 320;
    private const int CenterColumnMinPx = 340;
    private const int CenterColumnMaxPx = 460;

    // 縦積みブロックが触ってよいセレクタ。ここを増やすとレスポンシブの責務が広がるので、
    // 集合そのものを固定して増減にレビューが入るようにする。
    private static readonly string[] StackedSelectors =
    {
        ".main", ".gen", ".canvas", ".sidebar", ".gen-actions",
    };

    // ────────────────────────────────────────────────
    // (15) P3-4 / P3-2 をまとめて見る本番判定
    // ────────────────────────────────────────────────

    [Fact]
    public void Layout_SatisfiesEveryRule()
    {
        var problems = LayoutProblems(File.ReadAllText(AppCssPath));
        Assert.True(problems.Count == 0, "レイアウト規則違反: " + string.Join(" / ", problems));
    }

    // ────────────────────────────────────────────────
    // (16) P3-4 中央列の上限と右ペイン
    // ────────────────────────────────────────────────

    [Fact]
    public void MainGrid_KeepsThreeTracksWithTheCenterCapRaised()
    {
        var tracks = GridTracks(TopLevelMainGrid());

        Assert.True(tracks.Count == 3, $"3 トラック構成であること (実際 {tracks.Count} 本): {string.Join(" | ", tracks)}");

        // 左は固定幅のまま (追加先トグル 3 ラベルが 1 行に収まる幅)
        Assert.Equal($"{LeftColumnPx}px", tracks[0]);

        // 中央は minmax。**下限は据え置きで上限だけが 460px**
        var m = Regex.Match(tracks[1], @"^minmax\(\s*(\d+)px\s*,\s*(\d+)px\s*\)$");
        Assert.True(m.Success, $"中央トラックは minmax(下限px, 上限px) であること: {tracks[1]}");
        Assert.Equal(CenterColumnMinPx, int.Parse(m.Groups[1].Value));
        Assert.Equal(CenterColumnMaxPx, int.Parse(m.Groups[2].Value));

        // 右は 1fr のまま (固定幅にすると画面幅の余りを受け止められない)
        Assert.Equal("1fr", tracks[2]);
    }

    [Fact]
    public void MainGrid_CollapsesToASingleColumnOnlyInsideTheStackedQuery()
    {
        // 縦積みは 1 カラム
        var stacked = Blocks(StackedInner())
            .Where(b => SelectorParts(b.Selector).Contains(".main", StringComparer.Ordinal))
            .ToList();

        Assert.True(stacked.Count == 1, $"縦積みブロックの .main は 1 件であること (実際 {stacked.Count} 件)");
        Assert.Equal("1fr", DeclarationValue(stacked[0].Body, "grid-template-columns"));

        // 1 カラム化がメディアクエリの外に漏れていないこと (漏れると 3 ペインが消える)
        Assert.Equal(3, GridTracks(TopLevelMainGrid()).Count);
    }

    [Fact]
    public void PreviewFrame_IsCappedOnDesktopOnlyAndTouchesTheStackedBreakpoint()
    {
        var caps = DesktopPreviewQueries(File.ReadAllText(AppCssPath));
        Assert.True(caps.Count == 1, $"プレビュー枠の頭打ちを持つ min-width クエリは 1 つであること (実際 {caps.Count} 件)");

        // 縦積みの境界と隙間なく接すること (1100 + 1)。どちらにも入らない幅を作らない。
        Assert.Equal(StackedBreakpointPx() + 1, MinWidthPx(caps[0].Condition));

        var cap = Blocks(caps[0].Inner)
            .Single(b => SelectorParts(b.Selector).Contains(".preview", StringComparer.Ordinal));

        // 頭打ちは max-width (幅を止めるだけ)。width / flex を書き換えると枠の伸縮が変わる。
        var body = Normalize(cap.Body);
        Assert.Contains("max-width:", body);
        Assert.DoesNotContain("width: 100%", body);
        Assert.DoesNotContain("flex:", body);
        Assert.DoesNotContain("aspect-ratio", body);
    }

    [Fact]
    public void PreviewFrame_KeepsStretchingAtTopLevelSoNarrowPanesAreUnchanged()
    {
        // 頭打ちは「広いときだけ効く上限」。素の伸び (flex: 1 / width: 100% / height: 100%) は
        // トップレベルに残っていること — ここが消えると狭いペインで枠が内容幅まで潰れる。
        var preview = Blocks(TopLevelCss(File.ReadAllText(AppCssPath)))
            .Single(b => Normalize(b.Selector) == ".preview");

        var body = Normalize(preview.Body);
        Assert.Contains("flex: 1", body);
        Assert.Contains("width: 100%", body);
        Assert.Contains("height: 100%", body);
        // 上限そのものはトップレベルに置かない (縦積みは幅いっぱいの結果ペインのままにする)
        Assert.DoesNotContain("max-width", body);
    }

    // ────────────────────────────────────────────────
    // (17) P3-2 縦積みの並びと sticky
    // ────────────────────────────────────────────────

    [Fact]
    public void StackedOrder_IsCenterThenRightThenLeft()
    {
        var order = StackedOrders();

        Assert.True(order.ContainsKey(".gen"), "縦積みで中央 (.gen) の order が無い");
        Assert.True(order.ContainsKey(".canvas"), "縦積みで右 (.canvas) の order が無い");
        Assert.True(order.ContainsKey(".sidebar"), "縦積みで左 (.sidebar) の order が無い");

        // ★ 値ではなく**大小関係**を見る (詰め直しても壊れない)。
        //   生成 (中央) → 結果 (右) → パレット (左) の導線。
        Assert.True(
            order[".gen"] < order[".canvas"],
            $"生成コントロールは結果より前 (実際 .gen={order[".gen"]} / .canvas={order[".canvas"]})");
        Assert.True(
            order[".canvas"] < order[".sidebar"],
            $"結果はパレットより前 (実際 .canvas={order[".canvas"]} / .sidebar={order[".sidebar"]})");
    }

    [Fact]
    public void Order_IsDeclaredOnlyInsideTheStackedQuery()
    {
        var leaked = DeclarationsOutsideStacked(File.ReadAllText(AppCssPath))
            .Where(x => x.Property == "order")
            .Select(x => $"{x.Selector} {{ order: {x.Value} }}")
            .ToList();

        Assert.True(
            leaked.Count == 0,
            "order が縦積みブロックの外にある (3 ペインの並びが変わる): " + string.Join(" / ", leaked));
    }

    [Fact]
    public void StickyRow_ExistsOnlyInsideTheStackedQuery()
    {
        var leaked = DeclarationsOutsideStacked(File.ReadAllText(AppCssPath))
            .Where(x => x.Property == "position" && x.Value == "sticky")
            .Select(x => x.Selector)
            .ToList();

        Assert.True(
            leaked.Count == 0,
            "position: sticky が縦積みブロックの外にある (3 ペインで生成ボタン行が貼り付く): "
            + string.Join(" / ", leaked));
    }

    [Fact]
    public void StickyRow_IsTheGenerateActionsRowPinnedToTheBottom()
    {
        var sticky = Blocks(StackedInner())
            .Where(b => DeclarationValue(b.Body, "position") == "sticky")
            .ToList();

        Assert.True(sticky.Count == 1, $"縦積みの sticky 規則は 1 つであること (実際 {sticky.Count} 件)");
        Assert.Equal(".gen-actions", Normalize(sticky[0].Selector));

        // 下端に貼る (top で貼ると生成ボタンが上に居座り、結果が押し下がる)
        Assert.Equal("0", DeclarationValue(sticky[0].Body, "bottom"));
        Assert.Null(DeclarationValue(sticky[0].Body, "top"));
    }

    [Fact]
    public void StickyRow_HasPanelSurfaceAndTopBorder()
    {
        var sticky = StickyBlock();
        var body = Normalize(sticky.Body);

        // 地が無いと裏を流れるチップ列と重なって読めない。器 (.gen) と同じ面にする。
        Assert.Contains("background: var(--panel)", body);
        // 上境界が無いと「どこから貼り付いた行か」の切れ目が消える
        Assert.Contains("border-top:", body);
        Assert.Contains("var(--border)", body);
    }

    [Fact]
    public void StickyRow_ReleasesTheScrollContainerSoItCanActuallyStick()
    {
        // ★ 急所: .gen が overflow-y: auto のままだと、sticky の基準は .gen 自身の
        //   スクロールポートになる。縦積みの .gen は自分ではスクロールしないので、
        //   sticky は**永久に発動しない** (CSS 的には正しいのに効かない)。
        var gen = Blocks(StackedInner())
            .Single(b => Normalize(b.Selector) == ".gen");

        Assert.Equal("visible", DeclarationValue(gen.Body, "overflow-y"));

        // デスクトップ側は従来どおり自前スクロール (ここが visible になると 3 ペインが崩れる)
        var desktopGen = Blocks(TopLevelCss(File.ReadAllText(AppCssPath)))
            .Single(b => Normalize(b.Selector) == ".gen");
        Assert.Equal("auto", DeclarationValue(desktopGen.Body, "overflow-y"));
    }

    [Fact]
    public void StackedQuery_TouchesOnlyTheLayoutSelectors()
    {
        var selectors = Blocks(StackedInner())
            .Select(b => Normalize(b.Selector))
            .OrderBy(x => x, StringComparer.Ordinal)
            .ToList();

        Assert.Equal(
            StackedSelectors.OrderBy(x => x, StringComparer.Ordinal).ToList(),
            selectors);
    }

    // ────────────────────────────────────────────────
    // (18) バッチ D の成果 (P1-6 の実効 px ベースライン / P3-1 のボタン) 非退行
    // ────────────────────────────────────────────────

    [Fact]
    public void NewLayoutRules_StayOutOfTheP1_6BaselineTables()
    {
        // 新しく足した規則は寸法 (font-size / border-radius) を 1 つも持たない
        Assert.DoesNotContain(ScaleActual.Value, e => e.Selector == ".gen-actions");
        Assert.Equal(1, ScaleActual.Value.Count(e => e.Selector == ".preview"));   // border-radius のみ

        // 余白 (padding / margin / gap) も増やしていない。
        // ※ 走査器はメディアクエリの中身も素のセレクタとして拾うので、
        //   縦積みブロックに余白を足すと必ずここが動く。
        Assert.Equal(1, SpacingActual.Value.Count(e => e.Selector == ".gen-actions")); // gap のみ
        Assert.Equal(2, SpacingActual.Value.Count(e => e.Selector == ".main"));        // gap + padding
        Assert.Equal(2, SpacingActual.Value.Count(e => e.Selector == ".gen"));         // gap + padding
        Assert.Equal(1, SpacingActual.Value.Count(e => e.Selector == ".canvas"));      // padding
        Assert.Equal(2, SpacingActual.Value.Count(e => e.Selector == ".sidebar"));     // gap + padding-right
        Assert.DoesNotContain(SpacingActual.Value, e => e.Selector == ".preview");
    }

    [Fact]
    public void DesktopButtonRow_IsUnchanged()
    {
        // 3 ペイン時の生成ボタン行は「横並び + 12px の間隔」だけ。
        // ここに位置や地が生えると P3-1 のボタン表 (.gen-actions .btn) の前提が変わる。
        var row = Blocks(TopLevelCss(File.ReadAllText(AppCssPath)))
            .Single(b => Normalize(b.Selector) == ".gen-actions");

        var body = Normalize(row.Body);
        Assert.Contains("display: flex", body);
        Assert.Contains("gap: var(--sp-3)", body);
        Assert.DoesNotContain("position", body);
        Assert.DoesNotContain("order", body);
        Assert.DoesNotContain("background", body);
        Assert.DoesNotContain("border", body);
    }

    [Fact]
    public void DesktopPanes_KeepTheirOwnScrollContainers()
    {
        var top = Blocks(TopLevelCss(File.ReadAllText(AppCssPath)));

        foreach (var selector in new[] { ".sidebar", ".gen" })
        {
            var block = top.Single(b => Normalize(b.Selector) == selector);
            Assert.Equal("auto", DeclarationValue(block.Body, "overflow-y"));
            Assert.Equal("0", DeclarationValue(block.Body, "min-height"));
        }
    }

    // ══════════════════════════════════════════════════
    //  (19) 検査装置そのものの自己検査 + 変異検査
    //
    //  上の検査は「現行 app.css に対して緑」なので、判定器が壊れていても
    //  (何も拾わない・@media を素通しする) 緑のまま素通りしうる。
    //  合成 CSS と**本物の app.css に 1 箇所だけ変異を入れたもの**を食わせて、
    //  「噛んでいること」と「壊すと赤くなること」を確かめる。
    // ══════════════════════════════════════════════════

    // 出荷している形を最小構成で写したもの (これを 1 箇所ずつ壊したのが下の変異ケース)。
    //
    // ※ `calc(100vh - 176px)` の **176 という数値はこの判定器の検査対象ではない**
    //    (見るのは「頭打ちが min-width クエリ 1 つにあり、境界が縦積みと接し、寸法を持たない」こと)。
    //    ここを 176 にしてあるのは出荷 app.css の写しとしての忠実さのためだけで、
    //    P3-5 でサムネ列 1 行 (72px) を足したぶん **104 → 176** に更新した。
    //    数値そのものは ui.Tests/PreviewHistoryRazorTests が
    //    「N = 104 (E の基準) + .thumb の height」として別途固定する。
    private const string LayoutCssShipped = """
        .main { display: grid; grid-template-columns: 320px minmax(340px, 460px) 1fr; }
        .sidebar { overflow-y: auto; }
        .gen { overflow-y: auto; }
        .canvas { position: relative; }
        .preview { flex: 1; width: 100%; height: 100%; }
        .gen-actions { display: flex; gap: var(--sp-3); }
        @media (min-width: 1101px) { .preview { max-width: calc(100vh - 176px); } }
        @media (max-width: 1100px)
        {
            .main { grid-template-columns: 1fr; }
            .gen { order: 1; overflow-y: visible; }
            .canvas { order: 2; min-height: 320px; }
            .sidebar { order: 3; overflow-y: visible; }
            .gen-actions { position: sticky; bottom: 0; z-index: 1; background: var(--panel); border-top: 1px solid var(--border); }
        }
        """;

    [Fact]
    public void LayoutAnalyzer_AcceptsTheShippedShape()
    {
        Assert.Empty(LayoutProblems(LayoutCssShipped));
    }

    public static IEnumerable<object[]> BrokenLayoutCssData() => new[]
    {
        // ── デスクトップ非退行 (最重要) ──
        new object[]
        {
            "sticky がメディアクエリの外",
            LayoutCssShipped + "\n.gen-actions { position: sticky; bottom: 0; }",
            "[sticky-scope]",
        },
        new object[]
        {
            "order がメディアクエリの外",
            LayoutCssShipped + "\n.gen { order: 1; }",
            "[order-scope]",
        },
        // ── 縦積みの並び ──
        new object[]
        {
            "従来どおり左が先頭 (DOM 順のまま)",
            LayoutCssShipped.Replace(".sidebar { order: 3;", ".sidebar { order: 0;"),
            "[order]",
        },
        new object[]
        {
            "結果が生成コントロールより前",
            LayoutCssShipped.Replace(".canvas { order: 2;", ".canvas { order: 0;"),
            "[order]",
        },
        new object[]
        {
            "左ペインの order が抜けた",
            LayoutCssShipped.Replace(".sidebar { order: 3; ", ".sidebar { "),
            "[order]",
        },
        // ── sticky 行 ──
        new object[]
        {
            "sticky 行そのものが無い",
            LayoutCssShipped.Replace("position: sticky; bottom: 0; z-index: 1; ", ""),
            "[sticky]",
        },
        new object[]
        {
            "貼り付く辺が指定されていない",
            LayoutCssShipped.Replace("bottom: 0; ", ""),
            "[sticky]",
        },
        new object[]
        {
            "上端に貼り付いている",
            LayoutCssShipped.Replace("bottom: 0;", "top: 0;"),
            "[sticky]",
        },
        new object[]
        {
            "地が無い (裏の要素と重なって読めない)",
            LayoutCssShipped.Replace("background: var(--panel); ", ""),
            "[surface]",
        },
        new object[]
        {
            "上境界が無い",
            LayoutCssShipped.Replace("border-top: 1px solid var(--border); ", ""),
            "[surface]",
        },
        new object[]
        {
            "スクロール容器を解除していない (sticky が永久に発動しない)",
            LayoutCssShipped.Replace(".gen { order: 1; overflow-y: visible; }", ".gen { order: 1; }"),
            "[scroll]",
        },
        // ── 3 ペインのトラック ──
        new object[]
        {
            "中央の上限が 400px のまま",
            LayoutCssShipped.Replace("minmax(340px, 460px)", "minmax(340px, 400px)"),
            "[grid]",
        },
        new object[]
        {
            "トラックが 2 本に減った",
            LayoutCssShipped.Replace("320px minmax(340px, 460px) 1fr", "320px 1fr"),
            "[grid]",
        },
        new object[]
        {
            "右トラックが固定幅になった",
            LayoutCssShipped.Replace("minmax(340px, 460px) 1fr", "minmax(340px, 460px) 600px"),
            "[grid]",
        },
        new object[]
        {
            "縦積みが 1 カラムでない",
            LayoutCssShipped.Replace(".main { grid-template-columns: 1fr; }", ".main { grid-template-columns: 1fr 1fr; }"),
            "[stack]",
        },
        // ── 右ペインの余白調整 ──
        new object[]
        {
            "枠の頭打ちが消えた",
            LayoutCssShipped.Replace("@media (min-width: 1101px) { .preview { max-width: calc(100vh - 176px); } }", ""),
            "[canvas]",
        },
        new object[]
        {
            "頭打ちの境界が縦積みと重なる",
            LayoutCssShipped.Replace("min-width: 1101px", "min-width: 1000px"),
            "[canvas]",
        },
        new object[]
        {
            "素の伸びまで殺した (狭いペインで枠が潰れる)",
            LayoutCssShipped.Replace(".preview { flex: 1; width: 100%; height: 100%; }", ".preview { height: 100%; }"),
            "[canvas]",
        },
        // ── ベースライン表の防衛 ──
        new object[]
        {
            "縦積みブロックに余白が生えた",
            LayoutCssShipped.Replace("position: sticky;", "position: sticky; padding: 8px;"),
            "[sizing]",
        },
    };

    [Theory]
    [MemberData(nameof(BrokenLayoutCssData))]
    public void LayoutAnalyzer_RejectsBrokenShapes(string label, string css, string code)
    {
        var problems = LayoutProblems(css);

        Assert.True(problems.Count > 0, $"変異「{label}」を検出できていない (検査が空振り)");
        Assert.Contains(problems, p => p.StartsWith(code, StringComparison.Ordinal));
    }

    // 合成 CSS だけだと「本物の app.css を読めていない」可能性が残るので、
    // **出荷ファイルそのものに 1 箇所だけ変異を入れて**赤くなることも確かめる。
    public static IEnumerable<object[]> MutatedRealCssData() => new[]
    {
        // ★ PL 指定の実証: sticky をメディアクエリの外へ出したら赤くなること
        new object[] { "sticky を外へ持ち出す", "sticky-out", "[sticky-scope]" },
        new object[] { "order を外へ持ち出す", "order-out", "[order-scope]" },
        new object[] { "中央の上限を 400px へ戻す", "center-400", "[grid]" },
        new object[] { "スクロール容器を解除しない", "keep-scroll", "[scroll]" },
        new object[] { "sticky 行の上境界を消す", "no-edge", "[surface]" },
        new object[] { "枠の頭打ちを消す", "no-cap", "[canvas]" },
    };

    [Theory]
    [MemberData(nameof(MutatedRealCssData))]
    public void RealAppCss_TurnsRedWhenMutated(string label, string mutation, string code)
    {
        var css = File.ReadAllText(AppCssPath);

        var mutated = mutation switch
        {
            // メディアクエリの外に同じ規則を置く = 3 ペインでもボタン行が貼り付く
            "sticky-out" => css + "\n.gen-actions\n{\n    position: sticky;\n    bottom: 0;\n}\n",
            "order-out" => css + "\n.gen\n{\n    order: 1;\n}\n",
            "center-400" => css.Replace($"minmax({CenterColumnMinPx}px, {CenterColumnMaxPx}px)",
                                        $"minmax({CenterColumnMinPx}px, 400px)"),
            "keep-scroll" => css.Replace("overflow-y: visible", "overflow-y: auto"),
            "no-edge" => css.Replace("border-top: 1px solid var(--border);", ""),
            "no-cap" => Regex.Replace(css, @"@media \(min-width: 1101px\)\s*\{[^{}]*\{[^{}]*\}\s*\}", ""),
            _ => throw new ArgumentOutOfRangeException(nameof(mutation)),
        };

        Assert.NotEqual(css, mutated);

        var problems = LayoutProblems(mutated);
        Assert.True(problems.Count > 0, $"変異「{label}」を検出できていない (検査が空振り)");
        Assert.Contains(problems, p => p.StartsWith(code, StringComparison.Ordinal));
    }

    [Fact]
    public void GridTrackParser_SplitsOnTopLevelWhitespaceOnly()
    {
        // minmax(…) の中のカンマ・空白で切ってしまうと 3 トラックが 4 本に見える
        Assert.Equal(
            new[] { "320px", "minmax(340px, 460px)", "1fr" },
            GridTracks("320px minmax(340px, 460px) 1fr"));

        Assert.Equal(new[] { "1fr" }, GridTracks("  1fr  "));
        Assert.Equal(new[] { "minmax(0, 1fr)", "2fr" }, GridTracks("minmax(0, 1fr) 2fr"));
    }

    [Fact]
    public void OutsideStackedScanner_SeesTopLevelAndOtherQueriesButNotTheStackedOne()
    {
        const string css = """
            .a { order: 1; }
            @media (min-width: 1101px) { .b { order: 2; } }
            @media (max-width: 1100px) { .c { order: 3; } }
            """;

        var outside = DeclarationsOutsideStacked(css)
            .Where(x => x.Property == "order")
            .Select(x => x.Selector)
            .OrderBy(x => x, StringComparer.Ordinal)
            .ToList();

        // 縦積みブロック (.c) だけが除外され、他は全部見える
        Assert.Equal(new[] { ".a", ".b" }, outside);
    }

    // ══════════════════════════════════════════════════
    //  以下 P3 バッチ E 用ヘルパー
    // ══════════════════════════════════════════════════

    // レイアウト規則を 1 箇所で判定する。先頭のコードで「どの規則を破ったか」が分かる形
    // (変異検査が宛先を指定できる)。
    private static List<string> LayoutProblems(string css)
    {
        var problems = new List<string>();
        var top = Blocks(TopLevelCss(css));
        var media = MediaBlocks(css);

        // ① 3 ペインのトラック (P3-4)
        var grid = top
            .Where(b => SelectorParts(b.Selector).Contains(".main", StringComparer.Ordinal))
            .Select(b => DeclarationValue(b.Body, "grid-template-columns"))
            .Where(v => v is not null)
            .ToList();

        if (grid.Count != 1)
        {
            problems.Add($"[grid] メディアクエリ外の .main の grid-template-columns が {grid.Count} 件 (1 件であること)");
        }
        else
        {
            var tracks = GridTracks(grid[0]!);
            if (tracks.Count != 3)
            {
                problems.Add($"[grid] 3 トラック構成でない ({tracks.Count} 本): {grid[0]}");
            }
            else
            {
                if (tracks[0] != $"{LeftColumnPx}px")
                {
                    problems.Add($"[grid] 左トラックが {LeftColumnPx}px でない: {tracks[0]}");
                }

                var m = Regex.Match(tracks[1], @"^minmax\(\s*(\d+)px\s*,\s*(\d+)px\s*\)$");
                if (!m.Success)
                {
                    problems.Add($"[grid] 中央トラックが minmax(下限px, 上限px) でない: {tracks[1]}");
                }
                else
                {
                    var min = int.Parse(m.Groups[1].Value);
                    var max = int.Parse(m.Groups[2].Value);

                    if (min != CenterColumnMinPx)
                    {
                        problems.Add($"[grid] 中央トラックの下限が動いた ({min}px / 据え置きは {CenterColumnMinPx}px)");
                    }

                    if (max != CenterColumnMaxPx)
                    {
                        problems.Add($"[grid] 中央トラックの上限が {max}px (P3-4 は {CenterColumnMaxPx}px)");
                    }
                }

                if (!tracks[2].EndsWith("fr", StringComparison.Ordinal))
                {
                    problems.Add($"[grid] 右トラックが可変 (fr) でない: {tracks[2]}");
                }
            }
        }

        // ② 縦積みブロック (max-width) は 1 つだけ
        var stacked = media.Where(x => x.Condition.Contains("max-width", StringComparison.Ordinal)).ToList();
        if (stacked.Count != 1)
        {
            problems.Add($"[stack] max-width のメディアクエリが {stacked.Count} 件 (1 件であること)");
        }
        else
        {
            var inner = Blocks(stacked[0].Inner);

            var stackedGrid = inner
                .Where(b => SelectorParts(b.Selector).Contains(".main", StringComparer.Ordinal))
                .Select(b => DeclarationValue(b.Body, "grid-template-columns"))
                .ToList();

            if (stackedGrid.Count != 1 || stackedGrid[0] != "1fr")
            {
                problems.Add("[stack] 縦積みが 1 カラム (grid-template-columns: 1fr) でない: "
                             + string.Join(" / ", stackedGrid));
            }

            // ③ 並びは 中央 < 右 < 左 (値ではなく関係)
            var order = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var b in inner)
            {
                var value = DeclarationValue(b.Body, "order");
                if (value is not null && int.TryParse(value, out var n))
                {
                    order[Normalize(b.Selector)] = n;
                }
            }

            foreach (var selector in new[] { ".gen", ".canvas", ".sidebar" })
            {
                if (!order.ContainsKey(selector))
                {
                    problems.Add($"[order] 縦積みで {selector} の order が無い (並びが DOM 順に戻る)");
                }
            }

            if (order.ContainsKey(".gen") && order.ContainsKey(".canvas") && order[".gen"] >= order[".canvas"])
            {
                problems.Add($"[order] 生成 → 結果 の順でない (.gen={order[".gen"]} / .canvas={order[".canvas"]})");
            }

            if (order.ContainsKey(".canvas") && order.ContainsKey(".sidebar") && order[".canvas"] >= order[".sidebar"])
            {
                problems.Add($"[order] 結果 → パレット の順でない (.canvas={order[".canvas"]} / .sidebar={order[".sidebar"]})");
            }

            // ④ sticky 行
            var sticky = inner.Where(b => DeclarationValue(b.Body, "position") == "sticky").ToList();
            if (sticky.Count != 1)
            {
                problems.Add($"[sticky] 縦積みの position: sticky が {sticky.Count} 件 (生成ボタン行の 1 件であること)");
            }
            else
            {
                if (Normalize(sticky[0].Selector) != ".gen-actions")
                {
                    problems.Add($"[sticky] 貼り付けているのが生成ボタン行でない: {sticky[0].Selector}");
                }

                if (DeclarationValue(sticky[0].Body, "bottom") != "0")
                {
                    problems.Add("[sticky] 画面下端 (bottom: 0) に貼り付いていない");
                }

                if (DeclarationValue(sticky[0].Body, "top") is not null)
                {
                    problems.Add("[sticky] 上端にも貼り付けている (下端専用であること)");
                }

                var body = Normalize(sticky[0].Body);
                if (!body.Contains("background:", StringComparison.Ordinal))
                {
                    problems.Add("[surface] sticky 行に地が無い (裏を流れる要素と重なって読めない)");
                }

                if (!body.Contains("border-top:", StringComparison.Ordinal))
                {
                    problems.Add("[surface] sticky 行に上境界が無い (貼り付いた行の切れ目が消える)");
                }
            }

            // ⑤ sticky の基準をビューポートへ戻すため .gen のスクロール容器を解除する
            var stackedGen = inner.Where(b => Normalize(b.Selector) == ".gen").ToList();
            if (stackedGen.Count != 1 || DeclarationValue(stackedGen[0].Body, "overflow-y") != "visible")
            {
                problems.Add("[scroll] 縦積みで .gen の overflow-y を解除していない (sticky が発動しない)");
            }

            // ⑥ 余白・字送り・角丸に触らない (P1-6 の実効 px ベースライン表を動かさない)
            foreach (var b in inner)
            {
                foreach (var d in ParseDeclarations(b.Body).Where(d => IsLayoutForbiddenProperty(d.Property)))
                {
                    problems.Add($"[sizing] 縦積みブロックが寸法を持っている: {b.Selector} {{ {d.Property}: {d.Value} }}");
                }
            }
        }

        // ⑦ 縦積み専用の道具がブロックの外に漏れていないこと (デスクトップ非退行)
        foreach (var d in DeclarationsOutsideStacked(css))
        {
            if (d.Property == "order")
            {
                problems.Add($"[order-scope] order がメディアクエリの外にある: {d.Selector} {{ order: {d.Value} }}");
            }

            if (d.Property == "position" && d.Value == "sticky")
            {
                problems.Add($"[sticky-scope] position: sticky がメディアクエリの外にある: {d.Selector}");
            }
        }

        // ⑧ 右ペインの余白調整 (P3-4): 枠の頭打ちはデスクトップ側だけ
        var caps = DesktopPreviewQueries(css);
        if (caps.Count != 1)
        {
            problems.Add($"[canvas] プレビュー枠の頭打ちを持つ min-width クエリが {caps.Count} 件 (1 件であること)");
        }
        else
        {
            var min = MinWidthPx(caps[0].Condition);
            if (stacked.Count == 1 && min != StackedBreakpointPx(stacked[0].Condition) + 1)
            {
                problems.Add($"[canvas] 頭打ちの境界 ({min}px) が縦積みの境界と隙間なく接していない");
            }

            var cap = Blocks(caps[0].Inner)
                .Single(b => SelectorParts(b.Selector).Contains(".preview", StringComparer.Ordinal));

            foreach (var d in ParseDeclarations(cap.Body).Where(d => IsLayoutForbiddenProperty(d.Property)))
            {
                problems.Add($"[sizing] 枠の頭打ちが寸法を持っている: {d.Property}: {d.Value}");
            }
        }

        // トップレベルの .preview は素の伸びを保つ (頭打ちは上限としてだけ効かせる)
        var preview = top.Where(b => Normalize(b.Selector) == ".preview").ToList();
        if (preview.Count != 1)
        {
            problems.Add($"[canvas] メディアクエリ外の .preview 宣言が {preview.Count} 件 (1 件であること)");
        }
        else
        {
            var body = Normalize(preview[0].Body);
            if (!body.Contains("flex: 1", StringComparison.Ordinal) || !body.Contains("width: 100%", StringComparison.Ordinal))
            {
                problems.Add("[canvas] .preview の素の伸び (flex: 1 / width: 100%) が消えている");
            }
        }

        return problems;
    }

    // 余白・字送り・角丸。P3 バッチ E の新規則が触ってはいけないもの
    // (触ると P1-6 の実効 px ベースライン表の書き換えが要る = 裁定事項)。
    private static bool IsLayoutForbiddenProperty(string property)
        => IsSpacingProperty(property) || property is "font-size" or "border-radius";

    private readonly record struct ScopedDeclaration(string Selector, string Property, string Value);

    // 縦積みブロック (max-width のメディアクエリ) **以外**の全宣言。
    // トップレベルも他のメディアクエリも含む = 「縦積み専用の道具が漏れていないか」を見る。
    private static List<ScopedDeclaration> DeclarationsOutsideStacked(string css)
    {
        var list = new List<ScopedDeclaration>();

        var scopes = new List<string> { TopLevelCss(css) };
        scopes.AddRange(MediaBlocks(css)
            .Where(m => !m.Condition.Contains("max-width", StringComparison.Ordinal))
            .Select(m => m.Inner));

        foreach (var scope in scopes)
        {
            foreach (var b in Blocks(scope))
            {
                foreach (var d in ParseDeclarations(b.Body))
                {
                    list.Add(new ScopedDeclaration(Normalize(b.Selector), d.Property, d.Value));
                }
            }
        }

        return list;
    }

    // プレビュー枠の頭打ちを持つ min-width クエリ (P3-4 の右ペイン余白調整)。
    private static List<(string Condition, string Inner)> DesktopPreviewQueries(string css)
        => MediaBlocks(css)
            .Where(m => !m.Condition.Contains("max-width", StringComparison.Ordinal))
            .Where(m => Blocks(m.Inner).Any(b =>
                SelectorParts(b.Selector).Contains(".preview", StringComparer.Ordinal)
                && DeclarationValue(b.Body, "max-width") is not null))
            .ToList();

    private static int MinWidthPx(string condition)
    {
        var m = Regex.Match(condition, @"min-width\s*:\s*(\d+)px");
        Assert.True(m.Success, $"min-width の条件が読めない: {condition}");
        return int.Parse(m.Groups[1].Value);
    }

    private static int StackedBreakpointPx(string condition)
    {
        var m = Regex.Match(condition, @"max-width\s*:\s*(\d+)px");
        Assert.True(m.Success, $"max-width の条件が読めない: {condition}");
        return int.Parse(m.Groups[1].Value);
    }

    private static int StackedBreakpointPx() => StackedBreakpointPx(StackedQuery().Condition);

    // 出荷 app.css の縦積みメディアクエリ (max-width)。1 つであることも同時に固定する。
    private static (string Condition, string Inner) StackedQuery()
    {
        var stacked = MediaBlocks(File.ReadAllText(AppCssPath))
            .Where(m => m.Condition.Contains("max-width", StringComparison.Ordinal))
            .ToList();

        Assert.True(stacked.Count == 1, $"max-width のメディアクエリは 1 つであること (実際 {stacked.Count} 件)");
        return stacked[0];
    }

    private static string StackedInner() => StackedQuery().Inner;

    private static CssBlock StickyBlock()
        => Blocks(StackedInner()).Single(b => DeclarationValue(b.Body, "position") == "sticky");

    private static Dictionary<string, int> StackedOrders()
    {
        var order = new Dictionary<string, int>(StringComparer.Ordinal);

        foreach (var b in Blocks(StackedInner()))
        {
            var value = DeclarationValue(b.Body, "order");
            if (value is not null)
            {
                Assert.True(int.TryParse(value, out var n), $"order が整数でない: {b.Selector} {{ order: {value} }}");
                order[Normalize(b.Selector)] = n;
            }
        }

        return order;
    }

    private static string TopLevelMainGrid()
    {
        var main = Blocks(TopLevelCss(File.ReadAllText(AppCssPath)))
            .Where(b => Normalize(b.Selector) == ".main")
            .ToList();

        Assert.True(main.Count == 1, $"メディアクエリ外の .main 宣言は 1 つであること (実際 {main.Count} 件)");

        var value = DeclarationValue(main[0].Body, "grid-template-columns");
        Assert.True(value is not null, ".main に grid-template-columns が無い");
        return value!;
    }

    // grid-template-columns の値をトラックへ分ける。
    // minmax(340px, 460px) の中のカンマ・空白では切らない (括弧の深さを見る)。
    private static List<string> GridTracks(string value)
    {
        var tracks = new List<string>();
        var sb = new StringBuilder();
        var depth = 0;

        foreach (var c in Normalize(value))
        {
            if (c == '(')
            {
                depth++;
            }
            else if (c == ')')
            {
                depth--;
            }

            if (char.IsWhiteSpace(c) && depth == 0)
            {
                if (sb.Length > 0)
                {
                    tracks.Add(sb.ToString());
                    sb.Clear();
                }

                continue;
            }

            sb.Append(c);
        }

        if (sb.Length > 0)
        {
            tracks.Add(sb.ToString());
        }

        return tracks;
    }
}
