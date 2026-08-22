using System.Text;
using System.Text.RegularExpressions;
using Xunit;

namespace Dollama.Ui.Tests;

// P3-3 (テレメトリ役割ラベルの露出 / docs/ui-brushup-plan.md §4.4-3) の回帰止め。
//
// ★ 守りたい性質は 3 つ。どれも「壊れても画面はそれらしく動いてしまう」種類なので機械で見る。
//   ① Role ("CLIP enc" / "SDXL UNet" …) が title 属性**以外**の DOM に出ていること
//      (title だけに戻ると、パイプラインの見せ場がまたホバーへ埋まる)
//   ② 出す / 出さないの制御が CSS だけで完結していること
//      (razor に「今ラベルを出すか」の状態を持たせない = テレメトリの真実源は _sample のまま)
//   ③ ラベルを足したせいでバーのレイアウトが動かないこと
//      (既定 display: none で通常フローに入らない + 幅に余裕がある時だけ出す)
//
// ★ ②③ は「メディアクエリの中か外か」を区別しないと検査できない。既存の Blocks() は
//   入れ子を扱わず @media の中の規則を素のセレクタとして拾ってしまうので、
//   ここで @media を条件と中身に分ける小さなスキャナを足している
//   (スキャナ自身が噛んでいることは末尾の自己検査・変異検査で確かめる)。
//
// bUnit は入れない (依存追加ゼロの既存方針どおり CSS/razor をテキストとして読む)。
public sealed partial class AppCssTokenTests
{
    private const string RoleSelector = ".tm-role";

    // 「幅に余裕がある時だけ」と言える下限。これ未満のブレークポイントで出すのは
    // 実質「常時表示」= 狭い画面でトップバーが溢れる。
    private const int MinRoleBreakpointPx = 1200;

    // .tm-role に書いてはいけないプロパティ。
    // 寸法を持った瞬間に (a) バーの位置・幅が動きうる (b) P1-6 の実効 px ベースライン
    // (AppCssScaleBaselineTests) の書き換えが要る、の両方に触れる。
    // 字は .tm-item の --fs-xs を、間隔は .tm-item の gap を継承させるのが設計。
    private static readonly string[] RoleForbiddenProperties =
    {
        "font-size", "padding", "margin", "gap", "width", "height",
        "min-width", "max-width", "border", "position",
        "top", "right", "bottom", "left", "flex", "line-height",
    };

    // ────────────────────────────────────────────────
    // (12) P3-3 役割ラベル: CSS 側
    // ────────────────────────────────────────────────

    // 3 つの規則をまとめて見る本番判定。個別の理由は下の 3 件が読める形で落ちる。
    [Fact]
    public void RoleLabel_SatisfiesEveryVisibilityRule()
    {
        var problems = RoleLabelProblems(File.ReadAllText(AppCssPath));
        Assert.True(problems.Count == 0, "役割ラベルの表示規則違反: " + string.Join(" / ", problems));
    }

    [Fact]
    public void RoleLabel_IsHiddenByDefaultSoItNeverEntersTheLayout()
    {
        var top = TopLevelRoleBlocks(File.ReadAllText(AppCssPath));

        Assert.True(top.Count == 1, $"メディアクエリ外の {RoleSelector} 宣言は 1 つであること (実際 {top.Count} 件)");

        var body = Normalize(top[0].Body);
        // display: none = 通常フローに 1px も入らない。バー (64px 固定) の幅も位置も不変。
        Assert.Contains("display: none", body);
        // 数値 (--dev-*) より一段落とした説明文として出す
        Assert.Contains("color: var(--muted)", body);
        // 折り返すとトップバーの行が伸びる
        Assert.Contains("white-space: nowrap", body);
    }

    [Fact]
    public void RoleLabel_IsRevealedOnlyByAMinWidthMediaQuery()
    {
        var css = File.ReadAllText(AppCssPath);

        var media = MediaBlocks(css)
            .Where(m => RoleBlocks(m.Inner).Count > 0)
            .ToList();

        Assert.True(media.Count == 1, $"{RoleSelector} を出すメディアクエリは 1 つであること (実際 {media.Count} 件)");

        // 「幅に余裕がある時だけ」= min-width。max-width で出すと狭い側で溢れる。
        Assert.DoesNotContain("max-width", media[0].Condition);
        var m = Regex.Match(media[0].Condition, @"min-width\s*:\s*(\d+)px");
        Assert.True(m.Success, $"min-width の条件で出すこと (実際: {media[0].Condition})");
        Assert.True(
            int.Parse(m.Groups[1].Value) >= MinRoleBreakpointPx,
            $"ブレークポイントが小さすぎる (実質常時表示): {m.Groups[1].Value}px");

        var inner = Normalize(RoleBlocks(media[0].Inner).Single().Body);
        Assert.Contains("display:", inner);
        Assert.DoesNotContain("display: none", inner);
    }

    [Fact]
    public void RoleLabel_DeclaresNothingThatCouldMoveTheBars()
    {
        var offenders = new List<string>();

        foreach (var b in AllRoleBlocks(File.ReadAllText(AppCssPath)))
        {
            foreach (var d in ParseDeclarations(b.Body))
            {
                if (IsForbiddenForRoleLabel(d.Property))
                {
                    offenders.Add($"{d.Property}: {d.Value}");
                }
            }
        }

        Assert.True(
            offenders.Count == 0,
            $"{RoleSelector} が寸法を持つとバーが動く / P1-6 のベースライン表の書き換えが要る: "
            + string.Join(" / ", offenders));
    }

    // 上の裏取り。P1-6 の実効 px 表 (A)(A2) は app.css 全体を走査するので、
    // 役割ラベルが寸法を持っていれば「表に無い宣言」として別の場所でも赤くなるはず。
    // ここでは「表を書き換えずに済んでいる」ことを直接固定する。
    [Fact]
    public void RoleLabel_StaysOutOfTheP1_6BaselineTables()
    {
        Assert.DoesNotContain(ScaleActual.Value, e => e.Selector == RoleSelector);
        Assert.DoesNotContain(SpacingActual.Value, e => e.Selector == RoleSelector);
    }

    // P2-10 の回帰止め。役割ラベルは既定グラデにもグローにも触らない。
    [Fact]
    public void RoleLabel_DoesNotDisturbTheDefaultGradientOrGeneratingGlow()
    {
        var css = File.ReadAllText(AppCssPath);
        var blocks = Blocks(css);

        // 既定グラデ (未知デバイスのフォールバック) が生きていること
        var fill = blocks.Where(b => Normalize(b.Selector) == ".tm-fill").ToList();
        Assert.True(fill.Count == 1, $".tm-fill の宣言は 1 つであること (実際 {fill.Count} 件)");
        Assert.Contains("linear-gradient(90deg, var(--accent), var(--dev-npu))", Normalize(fill[0].Body));

        // グローの土台 (.tm-bar の overflow: hidden とバー幅) が不変であること
        var bar = blocks.Where(b => Normalize(b.Selector) == ".tm-bar").ToList();
        Assert.True(bar.Count == 1, $".tm-bar の宣言は 1 つであること (実際 {bar.Count} 件)");
        Assert.Contains("overflow: hidden", Normalize(bar[0].Body));
        Assert.Contains("width: 64px", Normalize(bar[0].Body));

        // 生成中の規則がラベルへ広がっていないこと (グローの宛先は .tm-bar のまま)
        var generating = blocks
            .Where(b => b.Selector.Contains(".generating") && b.Selector.Contains(RoleSelector))
            .ToList();
        Assert.True(generating.Count == 0, "生成中の規則が役割ラベルへ伸びている: "
            + string.Join(" / ", generating.Select(g => g.Selector)));

        // P2-6 の --edge 合成にも割り込まないこと
        foreach (var b in AllRoleBlocks(css))
        {
            Assert.DoesNotContain("box-shadow", Normalize(b.Body));
            Assert.DoesNotContain("--edge", Normalize(b.Body));
        }
    }

    // 新トークンを足さないこと (PL 裁定)。--fs-*/--sp-*/--r-* の 12 段は
    // ScaleTokens_AreExactlyTheTwelveStepsOfSection4_2 が縛っているが、
    // 配色側には「増やさない」向きの検査が無かったので P3-3 で足す。
    [Fact]
    public void Root_HasNoTokenBeyondTheReviewedSet()
    {
        string[] expected =
        {
            // 面 / 境界 / 文字
            "--bg", "--panel", "--panel-2", "--border", "--border-strong", "--text", "--muted",
            // アクセント系
            "--accent", "--on-accent", "--accent-weak", "--accent-border", "--focus-ring",
            // 状態色
            "--ok", "--ng", "--ng-soft", "--ng-weak",
            // P2-2 / P2-6
            "--overlay", "--edge", "--edge-off", "--edge-target",
            // テレメトリのデバイス色 (P1-5)
            "--dev-cpu", "--dev-npu", "--dev-igpu", "--dev-gpu",
            // 寸法 (P1-6 / §4.2)
            "--fs-xs", "--fs-sm", "--fs-md", "--fs-lg",
            "--sp-1", "--sp-2", "--sp-3", "--sp-4", "--sp-6",
            "--r-sm", "--r-md", "--r-pill",
        };

        Assert.Equal(
            expected.OrderBy(x => x, StringComparer.Ordinal).ToList(),
            RootTokens().Keys.OrderBy(x => x, StringComparer.Ordinal).ToList());
    }

    // ────────────────────────────────────────────────
    // (13) P3-3 役割ラベル: razor 側
    // ────────────────────────────────────────────────

    [Fact]
    public void RoleText_IsRenderedAsElementContentNotOnlyInTitle()
    {
        var razor = File.ReadAllText(GenerateRazorPath);

        // 描画される文字列として出ていること (title 属性だけに戻ったら赤)
        Assert.Contains("@d.Role", TextContent(razor));
        Assert.Contains("<span class=\"tm-role\">@d.Role</span>", razor);

        // 狭い幅での受け皿として title も残すこと (ラベルは >=1500px でしか出ない)
        var item = ElementTags(razor).Single(t => t.Contains("class=\"tm-item", StringComparison.Ordinal));
        Assert.Contains("@d.Role", item);
    }

    // ラベルは行の末尾。役割名 (長さがまちまち) を先頭側に置くと、
    // 4 つ並んだバーの開始位置が item ごとにずれて「4 HW が並んで動く」絵が崩れる。
    [Fact]
    public void RoleLabel_ComesLastSoTheBarsStayAligned()
    {
        var razor = File.ReadAllText(GenerateRazorPath);

        var dev = razor.IndexOf("class=\"tm-dev\"", StringComparison.Ordinal);
        var bar = razor.IndexOf("class=\"tm-bar\"", StringComparison.Ordinal);
        var pct = razor.IndexOf("class=\"tm-pct\"", StringComparison.Ordinal);
        var role = razor.IndexOf("class=\"tm-role\"", StringComparison.Ordinal);

        Assert.True(dev > 0 && bar > dev && pct > bar && role > pct,
            $"tm-item の並びは dev → bar → pct → role であること (dev={dev} bar={bar} pct={pct} role={role})");
    }

    [Fact]
    public void RoleLabel_AddsNoStateToRazor()
    {
        var razor = File.ReadAllText(GenerateRazorPath);

        // クラスは静的 1 語のみ (式で出し分けていない = 表示制御は CSS 側)
        var label = ElementTags(razor)
            .Single(t => (ClassAttribute(t) ?? "").Contains("tm-role", StringComparison.Ordinal));
        Assert.Equal("tm-role", ClassAttribute(label));
        Assert.Empty(SplitClassTokens(ClassAttribute(label)!).Dynamic);

        // tm-item から tm-role までの区間に条件分岐が無いこと (@if で包んでいない)
        var from = razor.IndexOf("class=\"tm-item", StringComparison.Ordinal);
        var to = razor.IndexOf("class=\"tm-role\"", StringComparison.Ordinal);
        Assert.True(from > 0 && to > from);
        Assert.DoesNotContain("@if", razor[from..to]);

        // @code 側に役割ラベル由来の状態・分岐が 1 文字も無いこと
        var code = razor[razor.IndexOf("@code {", StringComparison.Ordinal)..];
        Assert.DoesNotContain("Role", code);
        Assert.DoesNotContain("tm-role", code);
    }

    // ══════════════════════════════════════════════════
    //  (14) 検査装置そのものの自己検査 + 変異検査
    //
    //  上の検査は「現行 app.css / Generate.razor に対して緑」なので、スキャナが
    //  壊れていても (何も拾わない・@media を素通しする) 緑のまま素通りしうる。
    //  合成入力を食わせて「噛んでいること」と「壊すと赤くなること」を確かめる。
    // ══════════════════════════════════════════════════

    // 出荷している形 (これを 1 箇所ずつ壊したものが下の変異ケース)。
    private const string RoleCssShipped = """
        .tm-role { display: none; color: var(--muted); white-space: nowrap; }
        @media (min-width: 1500px) { .tm-role { display: block; } }
        """;

    [Fact]
    public void RoleLabelAnalyzer_AcceptsTheShippedShape()
    {
        Assert.Empty(RoleLabelProblems(RoleCssShipped));
    }

    public static IEnumerable<object[]> BrokenRoleCssData() => new[]
    {
        // 既定で見えている = 全幅で通常フローに入りバーが動く
        new object[]
        {
            "常時表示",
            RoleCssShipped.Replace("display: none;", "display: block;"),
            "[default]",
        },
        // 出す規則がメディアクエリの外にある (幅に関係なく出る)
        new object[]
        {
            "メディアクエリ無し",
            ".tm-role { display: none; color: var(--muted); white-space: nowrap; }\n.tm-role { display: block; }",
            "[media]",
        },
        // 狭い側で出す (max-width) = 余裕が無いときに出す
        new object[]
        {
            "max-width で露出",
            RoleCssShipped.Replace("min-width: 1500px", "max-width: 700px"),
            "[media]",
        },
        // ブレークポイントが小さすぎる = 実質常時表示
        new object[]
        {
            "低すぎるブレークポイント",
            RoleCssShipped.Replace("min-width: 1500px", "min-width: 400px"),
            "[breakpoint]",
        },
        // メディアクエリの中でも display: none のまま = どの幅でも出ない
        new object[]
        {
            "出ないメディアクエリ",
            RoleCssShipped.Replace("display: block;", "display: none;"),
            "[media]",
        },
        // 寸法を持った (バーが動く / P1-6 のベースライン表が動く)
        new object[]
        {
            "寸法を持った",
            RoleCssShipped.Replace("white-space: nowrap;", "white-space: nowrap; font-size: 10px; margin-left: 6px;"),
            "[sizing]",
        },
        // ラベルの規則ごと消えた
        new object[]
        {
            "規則が無い",
            ".tm-item { gap: 6px; }",
            "[default]",
        },
    };

    [Theory]
    [MemberData(nameof(BrokenRoleCssData))]
    public void RoleLabelAnalyzer_RejectsBrokenShapes(string label, string css, string code)
    {
        var problems = RoleLabelProblems(css);

        Assert.True(problems.Count > 0, $"変異「{label}」を検出できていない (検査が空振り)");
        Assert.Contains(problems, p => p.StartsWith(code, StringComparison.Ordinal));
    }

    // @media を条件と中身に分けられること。既存の Blocks() は入れ子を扱わないので、
    // 「メディアクエリの中の規則が素のセレクタとして混ざる」ことも同時に示しておく
    // (このズレこそがスキャナを別に用意した理由)。
    [Fact]
    public void MediaScanner_SplitsAtRuleBlocksFromTopLevel()
    {
        const string css = """
            .a { color: red; }
            @media (min-width: 900px)
            {
                .a { color: blue; }
                .b { color: green; }
            }
            .c { color: black; }
            """;

        var media = MediaBlocks(css);
        Assert.Single(media);
        Assert.Equal("(min-width: 900px)", media[0].Condition);
        Assert.Equal(new[] { ".a", ".b" }, Blocks(media[0].Inner).Select(b => b.Selector).ToArray());

        // トップレベルからはメディアクエリの中身がきれいに落ちていること
        Assert.Equal(new[] { ".a", ".c" }, Blocks(TopLevelCss(css)).Select(b => b.Selector).ToArray());

        // 素の Blocks() だと中身が混ざる = 幅の条件が消えて見える
        Assert.Equal(3, Blocks(css).Count(b => b.Selector is ".a" or ".b"));
    }

    [Fact]
    public void TextContent_SeparatesAttributeTextFromRenderedText()
    {
        // title 属性にしか無い → 描画テキストには出ない
        Assert.DoesNotContain(
            "@d.Role",
            TextContent("<div class=\"tm-item\" title=\"@d.Device @d.Role\"><span>@d.Device</span></div>"));

        // 要素の中身として出ている → 描画テキストに出る
        Assert.Contains(
            "@d.Role",
            TextContent("<div title=\"@d.Device @d.Role\"><span class=\"tm-role\">@d.Role</span></div>"));

        // razor コメントは描画されないので数えない
        Assert.DoesNotContain("@d.Role", TextContent("<div>@* @d.Role はここでは出ない *@</div>"));

        // ラムダの `>` や入れ子の引用符で切れないこと (ElementTags と同じ土台)
        Assert.Contains(
            "見出し",
            TextContent("<button class=\"btn @(x ? \"on\" : \"\")\" @onclick=\"() => F(a, b)\">見出し</button>"));
    }

    // ══════════════════════════════════════════════════
    //  以下 P3-3 用ヘルパー
    // ══════════════════════════════════════════════════

    private static bool IsForbiddenForRoleLabel(string property)
        => RoleForbiddenProperties.Any(f =>
            property.Equals(f, StringComparison.Ordinal)
            || property.StartsWith(f + "-", StringComparison.Ordinal));

    private static List<CssBlock> RoleBlocks(string css)
        => Blocks(css)
            .Where(b => SelectorParts(b.Selector).Contains(RoleSelector, StringComparer.Ordinal))
            .ToList();

    private static List<CssBlock> TopLevelRoleBlocks(string css) => RoleBlocks(TopLevelCss(css));

    private static List<CssBlock> AllRoleBlocks(string css)
        => RoleBlocks(TopLevelCss(css))
            .Concat(MediaBlocks(css).SelectMany(m => RoleBlocks(m.Inner)))
            .ToList();

    // 役割ラベルの表示規則を 1 箇所で判定する。
    // 先頭のコードで「どの規則を破ったか」が分かる形にしてある (変異検査が宛先を指定できる)。
    private static List<string> RoleLabelProblems(string css)
    {
        var problems = new List<string>();

        // ① 既定は非表示 (通常フローに入らない)
        var top = TopLevelRoleBlocks(css);
        if (top.Count != 1)
        {
            problems.Add($"[default] メディアクエリ外の {RoleSelector} 宣言が {top.Count} 件 (1 件であること)");
        }
        else if (DeclarationValue(top[0].Body, "display") != "none")
        {
            problems.Add($"[default] 既定が display: none でない (通常フローに入りバーが動く)");
        }

        // ② 出すのは min-width のメディアクエリ 1 つだけ
        var media = MediaBlocks(css).Where(m => RoleBlocks(m.Inner).Count > 0).ToList();
        if (media.Count != 1)
        {
            problems.Add($"[media] {RoleSelector} を出すメディアクエリが {media.Count} 件 (1 件であること)");
        }
        else
        {
            var condition = media[0].Condition;
            var m = Regex.Match(condition, @"min-width\s*:\s*(\d+)px");

            if (!m.Success || condition.Contains("max-width", StringComparison.Ordinal))
            {
                problems.Add($"[media] 露出条件が min-width でない: {condition}");
            }
            else if (int.Parse(m.Groups[1].Value) < MinRoleBreakpointPx)
            {
                problems.Add($"[breakpoint] ブレークポイントが小さすぎる (実質常時表示): {m.Groups[1].Value}px");
            }

            var display = DeclarationValue(RoleBlocks(media[0].Inner)[0].Body, "display");
            if (display is null or "none")
            {
                problems.Add($"[media] メディアクエリの中でもラベルが出ない (display: {display ?? "未指定"})");
            }
        }

        // ③ 寸法を持たない
        foreach (var b in AllRoleBlocks(css))
        {
            foreach (var d in ParseDeclarations(b.Body).Where(d => IsForbiddenForRoleLabel(d.Property)))
            {
                problems.Add($"[sizing] {RoleSelector} が寸法を持っている: {d.Property}: {d.Value}");
            }
        }

        return problems;
    }

    // 宣言ブロックから 1 プロパティの実効値 (後勝ち) を取る。無ければ null。
    private static string? DeclarationValue(string body, string property)
        => ParseDeclarations(body)
            .Where(d => d.Property == property)
            .Select(d => d.Value)
            .LastOrDefault();

    // @media ブロックを (条件, 中身) へ分ける。Blocks() は入れ子を扱わないので、
    // 「どの幅で効く規則か」を見るにはこちらが要る。
    private static List<(string Condition, string Inner)> MediaBlocks(string css)
    {
        css = StripComments(css);

        var list = new List<(string, string)>();
        var i = css.IndexOf("@media", StringComparison.Ordinal);

        while (i >= 0)
        {
            var open = css.IndexOf('{', i);
            if (open < 0)
            {
                break;
            }

            var end = MatchingBrace(css, open);
            if (end < 0)
            {
                break;
            }

            list.Add((Normalize(css[(i + "@media".Length)..open]), css[(open + 1)..end]));
            i = css.IndexOf("@media", end, StringComparison.Ordinal);
        }

        return list;
    }

    // @media ブロックを丸ごと落とした CSS (= トップレベルの規則だけ)。
    private static string TopLevelCss(string css)
    {
        css = StripComments(css);

        var sb = new StringBuilder();
        var cursor = 0;
        var i = css.IndexOf("@media", StringComparison.Ordinal);

        while (i >= 0)
        {
            var open = css.IndexOf('{', i);
            var end = open < 0 ? -1 : MatchingBrace(css, open);
            if (end < 0)
            {
                break;
            }

            sb.Append(css[cursor..i]);
            cursor = end + 1;
            i = css.IndexOf("@media", cursor, StringComparison.Ordinal);
        }

        sb.Append(css[cursor..]);
        return sb.ToString();
    }

    // css[open] の '{' に対応する '}' の位置。見つからなければ -1。
    private static int MatchingBrace(string css, int open)
    {
        var depth = 0;

        for (var j = open; j < css.Length; j++)
        {
            if (css[j] == '{')
            {
                depth++;
            }
            else if (css[j] == '}')
            {
                depth--;
                if (depth == 0)
                {
                    return j;
                }
            }
        }

        return -1;
    }

    // razor から「画面に出る文字」だけを取り出す (開始タグ・終了タグ・razor コメントを落とす)。
    // title 属性にしか無い文字列と、要素の中身として出ている文字列を区別するのに使う。
    private static string TextContent(string razor)
    {
        razor = Regex.Replace(razor, @"@\*.*?\*@", "", RegexOptions.Singleline);

        var sb = new StringBuilder();

        for (var i = 0; i < razor.Length; i++)
        {
            if (razor[i] == '<' && i + 1 < razor.Length && char.IsLetter(razor[i + 1]))
            {
                // 開始タグ (属性値のラムダ `=>` や入れ子の引用符をまたぐ)
                i = ScanTag(razor, i) - 1;
                continue;
            }

            if (razor[i] == '<' && i + 1 < razor.Length && razor[i + 1] == '/')
            {
                var gt = razor.IndexOf('>', i);
                if (gt < 0)
                {
                    break;
                }

                i = gt;
                continue;
            }

            sb.Append(razor[i]);
        }

        return sb.ToString();
    }
}
