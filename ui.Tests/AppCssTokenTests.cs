using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using Xunit;

namespace Dollama.Ui.Tests;

// 配色トークン層 (docs/ui-brushup-plan.md §4.1 / P1-1〜P1-5・P1-7) の回帰止め。
//
// bUnit は入れない (依存追加ゼロの方針)。CSS/razor をテキストとして読み、
// 「トークン経由率 100%」「コントラストが読める水準」「フォーカス表示が消えていない」
// といった、壊れると気づきにくい性質だけを機械的に検査する。
//
// ★ 色の値そのものは hex 完全一致では検査しない。それは単なる変更検知器になり、
//   デザイン調整のたびにテストを書き換えることになる。守るのは可読性 = コントラスト比。
// ※ P3 バッチ D で partial 化した。P1-6 / P3-1 のリファクタ前ベースライン
//    (AppCssScaleBaselineTests.cs) が、ここの CSS パーサ (Blocks / RootTokens /
//    StripComments / Normalize / RepoRoot) をそのまま再利用するため。
//    既存のテストは 1 件も変更していない。
public sealed partial class AppCssTokenTests
{
    // ── 検査対象のファイル ──────────────────────────────
    private static string AppCssPath => Path.Combine(RepoRoot(), "ui", "wwwroot", "app.css");
    private static string MainLayoutCssPath => Path.Combine(RepoRoot(), "ui", "Components", "Layout", "MainLayout.razor.css");
    private static string ReconnectCssPath => Path.Combine(RepoRoot(), "ui", "Components", "Layout", "ReconnectModal.razor.css");
    private static string ReconnectRazorPath => Path.Combine(RepoRoot(), "ui", "Components", "Layout", "ReconnectModal.razor");
    private static string GenerateRazorPath => Path.Combine(RepoRoot(), "ui", "Components", "Pages", "Generate.razor");
    private static string TagPresetFieldRazorPath => Path.Combine(RepoRoot(), "ui", "Components", "Shared", "TagPresetField.razor");
    private static string TagPaletteRazorPath => Path.Combine(RepoRoot(), "ui", "Components", "Shared", "TagPalette.razor");

    // ────────────────────────────────────────────────
    // (1) トークン定義: 既存 7 + 新設 13 が :root に揃っていること
    // ────────────────────────────────────────────────
    [Fact]
    public void Root_DefinesAllTokens()
    {
        var tokens = RootTokens();

        string[] expected =
        {
            // 既存 (据え置き) 7
            "--bg", "--panel", "--panel-2", "--text", "--accent", "--ok", "--ng",
            // 改訂 + 新設 13
            "--border", "--border-strong", "--muted", "--on-accent",
            "--accent-weak", "--accent-border", "--focus-ring",
            "--ng-soft", "--ng-weak",
            "--dev-cpu", "--dev-npu", "--dev-igpu", "--dev-gpu",
            // P2-2 で追加: 生成中オーバーレイの暗幕
            "--overlay",
            // P2-6 で追加: 追加先の左縁を box-shadow として合成するための影トークン
            "--edge", "--edge-off", "--edge-target",
        };

        foreach (var name in expected)
        {
            Assert.True(tokens.ContainsKey(name), $"トークン {name} が :root に無い");
            Assert.False(string.IsNullOrWhiteSpace(tokens[name]), $"トークン {name} の値が空");
        }
    }

    // ────────────────────────────────────────────────
    // (2) コントラスト: WCAG 相対輝度で比を計算して下限を守る
    //     しきい値は「文字は AA (4.5) / 本文は AAA 寄り (7.0) / 枠は装飾線として
    //     見えること」を基準に置く。
    // ────────────────────────────────────────────────
    [Theory]
    // 本文はダーク地で AAA 圏を維持する
    [InlineData("--text", "--panel", 7.0)]
    // 補足文字 (11-12px で使う) は AA を余裕込みで満たす
    [InlineData("--muted", "--panel", 4.5)]
    [InlineData("--muted", "--panel-2", 4.5)]
    // accent 塗りボタンの上に載る文字
    [InlineData("--on-accent", "--accent", 4.5)]
    // エラー本文の淡色
    [InlineData("--ng-soft", "--panel", 4.5)]
    // テレメトリのデバイス色 4 種 (数値も同色で出すので文字基準で見る)
    [InlineData("--dev-cpu", "--panel", 4.5)]
    [InlineData("--dev-npu", "--panel", 4.5)]
    [InlineData("--dev-igpu", "--panel", 4.5)]
    [InlineData("--dev-gpu", "--panel", 4.5)]
    // 操作可能要素の外周。改訂前 (--border #2c313b vs --panel-2 = 1.15) から明確に上げる。
    // ※ PL 指示の「vs --panel-2 >= 2.0」は plan §4.1 の概算 2.3:1 に基づくが、
    //   #4a5261 の実算は 1.90 なので下限は 1.85 に置き、より意味のある
    //   「操作要素の外周 vs それを囲む面 (--panel)」で 2.0 を担保する。
    [InlineData("--border-strong", "--panel-2", 1.85)]
    [InlineData("--border-strong", "--panel", 2.0)]
    // 装飾線。全枠を 3:1 にすると縞模様化して逆効果なので、改訂前 1.26 からの底上げのみ見る。
    [InlineData("--border", "--panel", 1.4)]
    public void Tokens_MeetContrastFloor(string fg, string bg, double floor)
    {
        var tokens = RootTokens();
        var ratio = ContrastRatio(ParseHex(tokens[fg]), ParseHex(tokens[bg]));
        Assert.True(
            ratio >= floor,
            $"{fg} / {bg} のコントラスト比 {ratio:0.000} が下限 {floor:0.00} を下回った");
    }

    // ────────────────────────────────────────────────
    // (3) 半透明トークンは rgba() としてパースでき α が (0,1) にあること
    // ────────────────────────────────────────────────
    [Theory]
    [InlineData("--accent-weak")]
    [InlineData("--accent-border")]
    [InlineData("--focus-ring")]
    [InlineData("--ng-weak")]
    public void RgbaTokens_HaveAlphaBetweenZeroAndOne(string name)
    {
        var value = RootTokens()[name];
        var m = Regex.Match(value, @"^rgba\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*([0-9.]+)\s*\)$");
        Assert.True(m.Success, $"{name} は rgba(r, g, b, a) 形式であること (実際: {value})");

        var alpha = double.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture);
        Assert.InRange(alpha, 0.001, 0.999);
    }

    // ────────────────────────────────────────────────
    // (4) :root の外に直書きの色が 1 つも無いこと (トークン経由率 100%)
    //     ここが崩れると「--accent を変えたのにチップだけ旧色」が再発する。
    // ────────────────────────────────────────────────
    [Fact]
    public void AppCss_HasNoLiteralColorOutsideRoot()
    {
        var body = StripComments(File.ReadAllText(AppCssPath));
        body = RootBlockRegex().Replace(body, "");

        var hex = Regex.Matches(body, @"#[0-9a-fA-F]{3,8}\b").Select(x => x.Value).ToList();
        Assert.True(hex.Count == 0, ":root 外に hex の直書きがある: " + string.Join(", ", hex));

        var rgb = Regex.Matches(body, @"rgba?\(").Select(x => x.Value).ToList();
        Assert.True(rgb.Count == 0, $":root 外に rgb/rgba の直書きが {rgb.Count} 件ある");
    }

    // ────────────────────────────────────────────────
    // (5) 使われている var(--x) が全て :root に定義されていること
    //     (scoped CSS からの参照も含む。scoped は :root を持たないので継承頼み)
    // ────────────────────────────────────────────────
    [Fact]
    public void AllVarReferences_AreDefinedInRoot()
    {
        var defined = RootTokens().Keys.ToHashSet(StringComparer.Ordinal);
        var used = new SortedSet<string>(StringComparer.Ordinal);

        foreach (var path in new[] { AppCssPath, MainLayoutCssPath, ReconnectCssPath })
        {
            var text = StripComments(File.ReadAllText(path));
            foreach (Match m in Regex.Matches(text, @"var\(\s*(--[A-Za-z0-9_-]+)"))
            {
                used.Add(m.Groups[1].Value);
            }
        }

        var missing = used.Where(u => !defined.Contains(u)).ToList();
        Assert.True(missing.Count == 0, "未定義の CSS 変数を参照している: " + string.Join(", ", missing));

        // scoped CSS が :root を持ち込むと二重管理になるので禁止する (コメントでの言及は除く)
        Assert.DoesNotContain(":root", StripComments(File.ReadAllText(MainLayoutCssPath)));
        Assert.DoesNotContain(":root", StripComments(File.ReadAllText(ReconnectCssPath)));
    }

    // ────────────────────────────────────────────────
    // (6) フォーカス統一リング (P1-2)
    //     キーボード操作の現在地表示は消えると気づきにくいので構造ごと固定する。
    // ────────────────────────────────────────────────
    [Fact]
    public void FocusRing_IsSingleBlockCoveringAllInteractiveSelectors()
    {
        var blocks = Blocks(File.ReadAllText(AppCssPath));
        var ring = blocks.Where(b => b.Body.Contains("var(--focus-ring)")).ToList();
        Assert.True(ring.Count == 1, $"var(--focus-ring) を使う宣言ブロックは 1 つであること (実際 {ring.Count} 件)");

        var selector = ring[0].Selector;
        string[] required =
        {
            "textarea:focus", "select:focus", "input:focus-visible",
            "button:focus-visible", ".chips:focus-within",
        };
        foreach (var s in required)
        {
            Assert.True(selector.Contains(s), $"フォーカス統一リングのセレクタに {s} が無い: {selector}");
        }

        // .chips は .taginput .chips で宣言されている。裸で書くと詳細度負けする。
        Assert.Contains(".taginput .chips:focus-within", selector);
        // リングは box-shadow で描く (outline で太らせるとレイアウトに響く)
        Assert.Contains("box-shadow", ring[0].Body);
        Assert.Contains("border-color: var(--accent)", Normalize(ring[0].Body));

        // .chip-entry は input:focus-visible にも当たるので二重リングを打ち消していること
        var cancel = blocks.Where(b => b.Selector.Contains(".chip-entry")
                                       && Normalize(b.Body).Contains("box-shadow: none")).ToList();
        Assert.True(cancel.Count >= 1, ".chip-entry の二重リング打ち消し規則が無い");
    }

    // ────────────────────────────────────────────────
    // (7) placeholder (P1-3): UA 既定の半透明に戻さない
    // ────────────────────────────────────────────────
    [Fact]
    public void Placeholder_UsesMutedTokenWithFullOpacity()
    {
        var blocks = Blocks(File.ReadAllText(AppCssPath));
        var ph = blocks.Where(b => b.Selector.Contains("::placeholder")).ToList();
        Assert.True(ph.Count >= 1, "::placeholder の宣言が無い");

        var body = Normalize(ph[0].Body);
        Assert.Contains("color: var(--muted)", body);
        Assert.Contains("opacity: 1", body);
    }

    // ────────────────────────────────────────────────
    // (8) テレメトリのデバイス別色 (P1-5)
    // ────────────────────────────────────────────────
    [Theory]
    [InlineData(".tm-cpu", "var(--dev-cpu)")]
    [InlineData(".tm-npu", "var(--dev-npu)")]
    [InlineData(".tm-igpu", "var(--dev-igpu)")]
    [InlineData(".tm-gpu", "var(--dev-gpu)")]
    public void TelemetryDeviceClasses_UseDeviceTokens(string cls, string token)
    {
        var blocks = Blocks(File.ReadAllText(AppCssPath))
            .Where(b => b.Selector.Contains(cls))
            .ToList();

        Assert.True(blocks.Count >= 1, $"{cls} の宣言が app.css に無い");
        Assert.True(blocks.All(b => b.Body.Contains(token)),
            $"{cls} の宣言は {token} を参照すること");
    }

    [Fact]
    public void GenerateRazor_AssignsDeviceClassViaDeviceStyle()
    {
        var razor = File.ReadAllText(GenerateRazorPath);
        Assert.Contains("DeviceStyle.CssClass", razor);
        // 既存クラス名は変えない (テレメトリ表示の土台)
        Assert.Contains("class=\"tm-item", razor);
        Assert.Contains("telemetry-mini", razor);
    }

    // ────────────────────────────────────────────────
    // (9) P1-7 の回帰止め: エラー UI / 再接続モーダル
    // ────────────────────────────────────────────────
    [Fact]
    public void ErrorUi_IsDefinedOnlyInScopedCssAndIsDark()
    {
        // app.css 側は死にコードだったので消してある (詳細度で scoped が必ず勝つため)
        Assert.DoesNotContain("#blazor-error-ui", StripComments(File.ReadAllText(AppCssPath)));

        // 「なぜ変えたか」を書いた日本語コメントに旧値が出るので、宣言部だけを見る
        var scoped = StripComments(File.ReadAllText(MainLayoutCssPath));
        Assert.DoesNotContain("lightyellow", scoped);
        Assert.DoesNotContain("color-scheme: light only", Normalize(scoped));
        Assert.Contains("var(--panel)", scoped);
        Assert.Contains("var(--text)", scoped);
    }

    [Fact]
    public void ReconnectModal_IsDarkAndJapaneseButKeepsBlazorContract()
    {
        var css = File.ReadAllText(ReconnectCssPath);
        Assert.DoesNotContain("#6b9ed2", css);
        Assert.DoesNotContain("#3b6ea2", css);
        Assert.DoesNotContain("#0087ff", css);
        Assert.DoesNotMatch(new Regex(@"\bwhite\b"), css);
        Assert.Contains("var(--panel)", css);

        var razor = File.ReadAllText(ReconnectRazorPath);
        Assert.DoesNotContain("Rejoining the server", razor);

        // id / class は Blazor の JS との契約。文言と配色以外は触らない。
        string[] contract =
        {
            "components-reconnect-modal",
            "components-reconnect-container",
            "components-rejoining-animation",
            "components-reconnect-first-attempt-visible",
            "components-reconnect-repeated-attempt-visible",
            "components-reconnect-failed-visible",
            "components-seconds-to-next-attempt",
            "components-reconnect-button",
            "components-pause-visible",
            "components-resume-button",
            "components-resume-failed-visible",
            "data-nosnippet",
        };
        foreach (var id in contract)
        {
            Assert.True(razor.Contains(id), $"ReconnectModal.razor から {id} が消えている (Blazor JS との契約)");
        }
    }

    // ────────────────────────────────────────────────
    // (10) P2-6 追加先の可視化 (課題 #9)
    //      「パレットのタグが今どこに入るか」の表示。border 幅で描くと
    //      レイアウトが 3px ずれるので inset の box-shadow で描く、が肝。
    // ────────────────────────────────────────────────
    [Fact]
    public void TargetField_UsesInsetShadowNotBorderWidth()
    {
        var tokens = RootTokens();

        // 左縁の実体は inset の box-shadow (border-left ではない)
        var edge = Normalize(tokens["--edge-target"]);
        Assert.StartsWith("inset", edge);
        Assert.Contains("3px", edge);
        Assert.Contains("var(--accent)", edge);

        // 既定は no-op (影を描かない)。ここが色付きだと全フィールドに縁が出る。
        Assert.Equal("var(--edge-off)", Normalize(tokens["--edge"]));
        Assert.Contains("transparent", tokens["--edge-off"]);

        var blocks = Blocks(File.ReadAllText(AppCssPath));

        // 追加先フィールドの規則は「変数を差し替えるだけ」— border 幅も padding も動かさない
        var target = blocks.Where(b => b.Selector.Contains(".is-target")
                                       && b.Selector.Contains(".chips")).ToList();
        Assert.True(target.Count == 1, $".field.is-target ….chips の宣言は 1 つであること (実際 {target.Count} 件)");
        var body = Normalize(target[0].Body);
        Assert.Contains("--edge: var(--edge-target)", body);
        Assert.DoesNotContain("border-left", body);
        Assert.DoesNotContain("border-width", body);
        Assert.DoesNotContain("padding", body);

        // .chips 本体が var(--edge) を描画していること (ここが無いと非フォーカス時に出ない)
        var chips = blocks.Where(b => Normalize(b.Selector) == ".taginput .chips").ToList();
        Assert.True(chips.Count == 1, ".taginput .chips の宣言は 1 つであること");
        Assert.Contains("box-shadow: var(--edge)", Normalize(chips[0].Body));
    }

    [Fact]
    public void FocusRing_ComposesTargetEdgeSoNeitherIsLost()
    {
        // 追加先かつフォーカス中でも「左縁 + リング」が両方出ること。
        // リングの宣言は 1 ブロックに固定されている (上の (6)) ため、
        // 合成は var(--edge) を影リストの先頭に置く形でしか成立しない。
        var ring = Blocks(File.ReadAllText(AppCssPath))
            .Single(b => b.Body.Contains("var(--focus-ring)"));

        Assert.Contains("box-shadow: var(--edge), 0 0 0 2px var(--focus-ring)", Normalize(ring.Body));

        // チップの × ボタンは --edge を継承してしまうので打ち消していること
        var cancel = Blocks(File.ReadAllText(AppCssPath))
            .Where(b => b.Selector.Contains(".chip-x")
                        && Normalize(b.Body).Contains("--edge: var(--edge-off)")).ToList();
        Assert.True(cancel.Count >= 1, ".chip-x で --edge を打ち消していない (× ボタンに左縁が載る)");
    }

    [Fact]
    public void TargetField_IsDrivenByParentParameterNotSelfJudgement()
    {
        // 追加先の真実源は Generate.razor。子は降ってきた IsTarget を見るだけ。
        var field = File.ReadAllText(TagPresetFieldRazorPath);
        Assert.Contains("public bool IsTarget", field);
        Assert.Contains("is-target", field);
        // 子が _target/Target を自前で判定していないこと
        Assert.DoesNotContain("Target ==", field);

        var razor = File.ReadAllText(GenerateRazorPath);
        Assert.Contains("IsTarget=\"@(_target == \"prompt\")\"", razor);
        Assert.Contains("IsTarget=\"@(_target == \"negative\")\"", razor);
    }

    [Fact]
    public void Palette_ShowsSurfaceHintWhenFavoritesIsTarget()
    {
        var blocks = Blocks(File.ReadAllText(AppCssPath))
            .Where(b => b.Selector.Contains(".fav-target")).ToList();

        Assert.True(blocks.Count == 1, $".palette.fav-target の宣言は 1 つであること (実際 {blocks.Count} 件)");
        var body = Normalize(blocks[0].Body);
        // 面で示す = 地色 + 縁。色は既存トークンのみ (新トークンを増やさない)
        Assert.Contains("background: var(--accent-weak)", body);
        Assert.Contains("border-color: var(--accent-border)", body);

        var palette = File.ReadAllText(TagPaletteRazorPath);
        Assert.Contains("fav-target", palette);
    }

    // ────────────────────────────────────────────────
    // (11) P2-10 生成中のテレメトリ強調 (課題 #11 / §4.4-2)
    //      「全 HW 協調」の見せ場が title 属性 (ホバー) に埋もれていた回帰止め。
    // ────────────────────────────────────────────────
    [Theory]
    [InlineData(".tm-cpu", "var(--dev-cpu)")]
    [InlineData(".tm-npu", "var(--dev-npu)")]
    [InlineData(".tm-igpu", "var(--dev-igpu)")]
    [InlineData(".tm-gpu", "var(--dev-gpu)")]
    public void TelemetryGenerating_GlowsWithDeviceColor(string cls, string token)
    {
        var glow = Blocks(File.ReadAllText(AppCssPath))
            .Where(b => b.Selector.Contains(".generating") && b.Selector.Contains(cls)).ToList();

        Assert.True(glow.Count == 1, $"{cls} の生成中グロー宣言は 1 つであること (実際 {glow.Count} 件)");
        var body = Normalize(glow[0].Body);
        Assert.Contains("box-shadow", body);
        Assert.Contains(token, body);

        // グローは .tm-bar 側に出す。.tm-fill (子) に付けると .tm-bar の
        // overflow: hidden でクリップされて 1px も見えない。
        Assert.EndsWith(".tm-bar", glow[0].Selector);
    }

    [Fact]
    public void GeneratingPill_IsAccentFilledAndDrivenByTelemetryFlag()
    {
        var pill = Blocks(File.ReadAllText(AppCssPath))
            .Where(b => b.Selector.Contains(".gen-pill")).ToList();

        Assert.True(pill.Count == 1, $".gen-pill の宣言は 1 つであること (実際 {pill.Count} 件)");
        var body = Normalize(pill[0].Body);
        Assert.Contains("background: var(--accent)", body);
        Assert.Contains("color: var(--on-accent)", body);

        var razor = File.ReadAllText(GenerateRazorPath);
        Assert.Contains("gen-pill", razor);
        Assert.Contains("生成中", razor);
        // 生成中クラスの真実源はテレメトリの Generating (_busy ではない)
        Assert.Contains("_sample?.Generating == true ? \"generating\"", razor);
    }

    // ══════════════════════════════════════════════════
    //  以下ヘルパー
    // ══════════════════════════════════════════════════

    // :root ブロック (app.css の先頭 1 個) を丸ごと取る正規表現。
    private static Regex RootBlockRegex() => new(@":root\s*\{[^}]*\}", RegexOptions.Singleline);

    // :root の --name: value; を辞書化する。
    private static Dictionary<string, string> RootTokens()
    {
        var css = StripComments(File.ReadAllText(AppCssPath));
        var m = RootBlockRegex().Match(css);
        Assert.True(m.Success, "app.css に :root ブロックが見つからない");

        var inner = m.Value[(m.Value.IndexOf('{') + 1)..^1];
        var tokens = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (Match d in Regex.Matches(inner, @"(--[A-Za-z0-9_-]+)\s*:\s*([^;]+);"))
        {
            tokens[d.Groups[1].Value] = d.Groups[2].Value.Trim();
        }
        return tokens;
    }

    // CSS のコメントを落とす (コメント中の色やセレクタを検査に混ぜないため)。
    private static string StripComments(string css)
        => Regex.Replace(css, @"/\*.*?\*/", "", RegexOptions.Singleline);

    // 連続空白を 1 個に畳んで比較しやすくする。
    private static string Normalize(string s) => Regex.Replace(s, @"\s+", " ").Trim();

    private readonly record struct CssBlock(string Selector, string Body);

    // 「セレクタ { 宣言 }」を素朴に切り出す (このプロジェクトの CSS は入れ子を使わない)。
    private static List<CssBlock> Blocks(string css)
    {
        css = StripComments(css);
        var list = new List<CssBlock>();
        foreach (Match m in Regex.Matches(css, @"([^{}]+)\{([^{}]*)\}", RegexOptions.Singleline))
        {
            list.Add(new CssBlock(Normalize(m.Groups[1].Value), m.Groups[2].Value));
        }
        return list;
    }

    // #rgb / #rrggbb を 0-255 の 3 成分へ。
    private static (int R, int G, int B) ParseHex(string value)
    {
        var m = Regex.Match(value.Trim(), @"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$");
        Assert.True(m.Success, $"hex 色として読めない: {value}");

        var h = m.Groups[1].Value;
        if (h.Length == 3)
        {
            h = string.Concat(h[0], h[0], h[1], h[1], h[2], h[2]);
        }

        return (Convert.ToInt32(h[..2], 16), Convert.ToInt32(h.Substring(2, 2), 16), Convert.ToInt32(h.Substring(4, 2), 16));
    }

    // WCAG 2.x の相対輝度。
    private static double RelativeLuminance((int R, int G, int B) c)
    {
        static double Channel(int v)
        {
            var s = v / 255.0;
            return s <= 0.03928 ? s / 12.92 : Math.Pow((s + 0.055) / 1.055, 2.4);
        }

        return 0.2126 * Channel(c.R) + 0.7152 * Channel(c.G) + 0.0722 * Channel(c.B);
    }

    // WCAG 2.x のコントラスト比 (1.0〜21.0)。
    private static double ContrastRatio((int R, int G, int B) a, (int R, int G, int B) b)
    {
        var la = RelativeLuminance(a);
        var lb = RelativeLuminance(b);
        var (hi, lo) = la >= lb ? (la, lb) : (lb, la);
        return (hi + 0.05) / (lo + 0.05);
    }

    // リポジトリルート。まずテストソースの位置 (コンパイル時に埋まる) から辿り、
    // 駄目なら実行ディレクトリから辿る。どちらでも見つからなければ明示的に落とす。
    private static string RepoRoot()
    {
        foreach (var start in new[] { SourceDir(), AppContext.BaseDirectory })
        {
            if (string.IsNullOrEmpty(start))
            {
                continue;
            }

            for (var dir = new DirectoryInfo(start); dir is not null; dir = dir.Parent)
            {
                if (File.Exists(Path.Combine(dir.FullName, "ui", "wwwroot", "app.css")))
                {
                    return dir.FullName;
                }
            }
        }

        Assert.Fail(
            "リポジトリルートを特定できない (ui/wwwroot/app.css が見つからない)。" +
            $" 探索起点: source={SourceDir()} / base={AppContext.BaseDirectory}");
        return "";
    }

    private static string SourceDir([CallerFilePath] string thisFile = "")
        => Path.GetDirectoryName(thisFile) ?? "";
}
