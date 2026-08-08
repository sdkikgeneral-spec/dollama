using System.Text.RegularExpressions;
using Xunit;

namespace Dollama.Ui.Tests;

// P1-6 (タイポ/スペーシング/角丸トークン化) と P3-1 (ボタン 5 系統 → .btn 系へ統合) の
// **リファクタ前ベースライン**。
//
// ★ なぜ実装より先に書くか
//   どちらも「見た目は変えずに宣言を畳む」変更なので、リファクタ後にテストを書くと
//   期待値が実装のコピーになり検査価値がゼロになる。先に現行 app.css の実効値を凍結し、
//   畳んだ後もそれが一致することを機械的に見る。
//
// ★ 何を凍結するか
//   (A)  全 font-size / border-radius の「セレクタ → 実効値」表 (67 件)
//   (A2) 全 padding / margin / gap の「セレクタ → 実効値」表 (65 件)
//   (B)  ボタン系 8 群 × 4 状態 (base / :hover / :disabled / :focus-visible) の宣言集合
//   (C)  旧クラス名が「今は存在する」こと (P3-1 後に 0 件へ反転させる)
//
// ★ やらないこと
//   完全な CSS カスケード再現はしない。(B) の比較対象は上記 4 状態の宣言集合に限定し、
//   合成は「同一クラス数前提の後勝ち解決」で近似する。
//
// bUnit 等の依存は追加しない (既存方針どおり CSS/razor をテキストとして読む)。
public sealed partial class AppCssTokenTests
{
    private static string PresetSidebarRazorPath
        => Path.Combine(RepoRoot(), "ui", "Components", "Shared", "PresetSidebar.razor");

    // ══════════════════════════════════════════════════
    //  (A) P1-6 用: font-size / border-radius の実効値ベースライン
    // ══════════════════════════════════════════════════

    // 現行 app.css の全宣言 (font-size 36 + border-radius 29 = 65)。
    // 値は var() 解決後の実効値で書く (現状は全て直書きなので px / % がそのまま並ぶ)。
    // P1-6 で var(--fs-sm) / var(--r-sm) へ置換しても、解決結果がこの表と一致すれば緑。
    //
    // ★ P3-1 (ボタン統合) で**セレクタだけ**を張り替えた行がある (値は据え置き)。
    //   ボタン 7 系統の border-radius 6 宣言が .btn-ghost / .gen-actions .btn /
    //   .btn-icon / .lang-toggle .btn の 4 つに畳まれたので border-radius は 31 → 29。
    //   font-size は 6 消えて 6 生えたので 36 のまま。
    private static readonly ScaleEntry[] ScaleBaseline =
    {
        // ── ベース / トップバー ──
        new("html, body",                     "font-size",     "14px"),
        new(".brand",                         "font-size",     "16px"),
        new(".brand-sub",                     "font-size",     "12px"),
        new(".lang-toggle",                   "border-radius", "6px"),
        new(".lang-toggle .btn",              "font-size",     "12px"),   // ← .lang-toggle button
        new(".lang-toggle .btn",              "border-radius", "0"),      // P3-1: ghost の --r-sm を打ち消す (実効 0 のまま)
        new(".conn",                          "font-size",     "12px"),
        new(".conn .dot",                     "border-radius", "50%"),
        new(".conn .btn",                     "font-size",     "11px"),   // ← .conn-retry

        // ── 共通ボタン (P3-1) ──
        new(".btn-primary",                   "font-size",     "15px"),   // ← .generate (§4.2 で 16 へ寄せ)
        new(".btn-ghost",                     "border-radius", "6px"),    // ← .preset-btn / .fav-plus / .preview-save / .conn-retry
        new(".btn-icon",                      "border-radius", "4px"),    // ← .ps-del
        new(".btn-icon",                      "font-size",     "13px"),   // ← .ps-del (§4.2 で 14 へ寄せ)

        // ── テレメトリ ──
        new(".tm-hint",                       "font-size",     "12px"),
        new(".tm-item",                       "font-size",     "11px"),
        new(".tm-bar",                        "border-radius", "3px"),
        new(".gen-pill",                      "border-radius", "999px"),
        new(".gen-pill",                      "font-size",     "11px"),

        // ── 3 ペインの器 ──
        new(".gen",                           "border-radius", "10px"),
        new(".canvas",                        "border-radius", "10px"),
        new(".field > span",                  "font-size",     "12px"),
        new("textarea, select, input[type=\"range\"]", "border-radius", "6px"),

        // ── 生成ボタン行まわり ──
        new(".gen-actions .btn",              "border-radius", "8px"),    // ← .generate (§4.2 で 10 へ寄せ・2 ボタン共用)
        new(".gen-actions .btn-ghost",        "font-size",     "14px"),   // ← .generate.secondary
        new(".gen-reason",                    "font-size",     "12px"),
        new(".error",                         "border-radius", "6px"),
        new(".error",                         "font-size",     "13px"),   // §4.2 で 12 or 14 への寄せを許容

        // ── プレビュー (右ペイン) ──
        new(".gen-mode",                      "border-radius", "999px"),
        new(".gen-mode",                      "font-size",     "11px"),
        new(".canvas-save",                   "font-size",     "11px"),   // ← .preview-save
        new(".preview",                       "border-radius", "8px"),    // §4.2 で 10 への寄せを許容
        new(".preview-elapsed",               "font-size",     "13px"),
        new(".placeholder",                   "font-size",     "13px"),
        new(".spinner",                       "border-radius", "50%"),

        // ── タグパレット (左ペイン) ──
        new(".palette, .preset-sidebar",      "border-radius", "10px"),
        new(".palette-head",                  "font-size",     "13px"),
        new(".palette-target label",          "font-size",     "12px"),
        new(".palette-target label",          "border-radius", "6px"),
        new(".palette-cat",                   "border-radius", "6px"),
        new(".palette-cat > summary",         "font-size",     "12px"),
        new(".palette-empty",                 "font-size",     "11px"),
        new(".palette-tag",                   "border-radius", "999px"),
        new(".palette-tag",                   "font-size",     "12px"),

        // ── お気に入り ──
        new(".fav-chip",                      "border-radius", "999px"),
        new(".fav-add",                       "font-size",     "12px"),
        new(".fav-x",                         "font-size",     "13px"),
        new(".fav-entry",                     "border-radius", "6px"),
        new(".fav-entry",                     "font-size",     "12px"),

        // ── プリセット一覧 (左ペイン) ──
        new(".ps-card",                       "border-radius", "6px"),
        new(".ps-card img",                   "border-radius", "4px"),
        new(".ps-noimg",                      "border-radius", "4px"),
        new(".ps-noimg",                      "font-size",     "9px"),    // スケール外の最小値 (要棚卸し)
        new(".ps-name",                       "font-size",     "12px"),

        // ── プリセット保存バー (中央) ──
        new(".preset-name",                   "border-radius", "6px"),
        new(".preset-msg",                    "font-size",     "12px"),

        // ── チップ入力 ──
        new(".taginput .chips",               "border-radius", "6px"),
        new(".tag-chip",                      "border-radius", "999px"),
        new(".tag-chip",                      "font-size",     "13px"),
        new(".chip-x",                        "font-size",     "14px"),
        new(".chip-x",                        "border-radius", "50%"),

        // ── LoRA ──
        new(".lora-head",                     "font-size",     "12px"),
        new(".lora-empty",                    "font-size",     "11px"),
        new(".lora-chip",                     "border-radius", "999px"),
        new(".lora-chip",                     "font-size",     "12px"),
        new(".lora-val",                      "font-size",     "11px"),
    };

    // §4.2 が明示的に認めている「中間値の寄せ」。
    //   font-size 13 → 12 or 14 / font-size 15 → 14 or 16 / border-radius 8 → 10
    // P1-6 で寄せた箇所を理由付きで 1 件ずつ積んである。**それ以外の値変更は通さない**
    // (許可の形は ScaleDriftAllowList_OnlyContainsRoundingsPermittedBySection4_2 が縛る)。
    private static readonly ScaleDrift[] AllowedScaleDrifts =
    {
        // ── font-size 13px (7 件) ──
        new(".error", "font-size", "13px", "14px",
            "エラーは読ませる文なので本文サイズ (--fs-md) へ"),
        new(".preview-elapsed", "font-size", "13px", "14px",
            "減光した前回画像の上に載る経過秒。可読側 (--fs-md) へ寄せる"),
        new(".placeholder", "font-size", "13px", "14px",
            "広い空プレビューの案内文。本文サイズ (--fs-md) へ"),
        new(".palette-head", "font-size", "13px", "12px",
            "§4.2 のセクションヘッダ型 (--fs-sm + letter-spacing + --muted) へ"),
        new(".fav-x", "font-size", "13px", "14px",
            "× グリフの寸法を .chip-x (--fs-md) と統一する"),
        // P3-1 で宛先セレクタのみ .ps-del → .btn-icon へ張り替え (寄せの中身は不変)
        new(".btn-icon", "font-size", "13px", "14px",
            "× グリフの寸法を .chip-x / .fav-x (--fs-md) と統一する"),
        new(".tag-chip", "font-size", "13px", "12px",
            "パレットのタグ (--fs-sm) からチップ化しても字の大きさが変わらないように"),

        // ── font-size 15px (1 件) ── ※ P3-1 で宛先を .generate → .btn-primary へ張り替え
        new(".btn-primary", "font-size", "15px", "16px",
            "主 CTA。14px に寄せると下書きボタンと同寸になり階層が消えるので大きい側へ"),

        // ── border-radius 8px (2 件) ── ※ P3-1 で宛先を .generate → .gen-actions .btn へ張り替え
        new(".gen-actions .btn", "border-radius", "8px", "10px",
            "§4.2「現 8/10 を 10 に寄せる」。器 (.gen / .canvas) と半径をそろえる"),
        new(".preview", "border-radius", "8px", "10px",
            "§4.2「現 8/10 を 10 に寄せる」。外側の器 .canvas (--r-md) と半径をそろえる"),
    };

    // 寄せた件数の固定。増減するときは必ずレビューが要る (PL 裁定)。
    private const int ExpectedScaleDrifts = 10;

    public static IEnumerable<object[]> ScaleBaselineData()
        => ScaleBaseline.Select(e => new object[] { e.Selector, e.Property, e.Value });

    // 表の 1 行 = 1 テストケース。落ちたときにどのセレクタの何 px が動いたか即分かる形にする。
    [Theory]
    [MemberData(nameof(ScaleBaselineData))]
    public void EffectiveScaleValue_MatchesBaseline(string selector, string property, string expected)
    {
        var actual = ScaleActual.Value
            .Where(x => x.Selector == selector && x.Property == property)
            .Select(x => x.Value)
            .ToList();

        Assert.True(
            actual.Count == 1,
            $"{selector} の {property} 宣言は 1 つであること (実際 {actual.Count} 件)");

        var drift = AllowedScaleDrifts.FirstOrDefault(d => d.Selector == selector && d.Property == property);
        if (drift is null)
        {
            Assert.Equal(expected, actual[0]);
            return;
        }

        // 許可リストは「元の値」も一致していないと意味がない (表の書き換え隠蔽を防ぐ)
        Assert.Equal(expected, drift.From);
        Assert.Equal(drift.To, actual[0]);
    }

    // 表に無い宣言が増えていない / 表の行が消えていないこと。
    // P1-6 は「宣言を var() へ置換する」だけで件数は動かない想定。件数が動いたら
    // 意図的な統合なので、この数字を書き換えるときに必ずレビューが入る。
    // (P3-1 = ボタン統合でちょうど border-radius が 2 宣言分だけ畳まれた)
    [Fact]
    public void ScaleBaseline_CoversEveryDeclarationInAppCss()
    {
        var actual = ScaleActual.Value;

        Assert.Equal(36, actual.Count(x => x.Property == "font-size"));
        Assert.Equal(29, actual.Count(x => x.Property == "border-radius"));
        Assert.Equal(65, actual.Count);
        Assert.Equal(65, ScaleBaseline.Length);

        var baseline = ScaleBaseline.Select(e => (e.Selector, e.Property)).ToHashSet();
        var extra = actual
            .Where(a => !baseline.Contains((a.Selector, a.Property)))
            .Select(a => $"{a.Selector} {{ {a.Property} }}")
            .ToList();
        Assert.True(extra.Count == 0, "ベースライン表に無い宣言が増えている: " + string.Join(" / ", extra));

        var found = actual.Select(a => (a.Selector, a.Property)).ToHashSet();
        var missing = baseline.Where(b => !found.Contains(b)).Select(b => $"{b.Item1} {{ {b.Item2} }}").ToList();
        Assert.True(missing.Count == 0, "ベースライン表にあるのに app.css から消えた宣言: " + string.Join(" / ", missing));
    }

    // ★ P1-6 で反転させた検査 (旧 ScaleDriftAllowList_IsEmptyBeforeP1_6)。
    //   「空であること」→「レビュー済みの 10 件ちょうどであること」。
    //   件数を定数で固定しているので、寄せを 1 件足すだけでも赤くなる。
    [Fact]
    public void ScaleDriftAllowList_MatchesTheReviewedRoundings()
    {
        Assert.Equal(ExpectedScaleDrifts, AllowedScaleDrifts.Length);

        // 同じ宛先を二重に積まない (後勝ちで検査が骨抜きになるのを防ぐ)
        var keys = AllowedScaleDrifts.Select(d => (d.Selector, d.Property)).ToList();
        Assert.Equal(keys.Count, keys.Distinct().Count());

        // 宛先がベースライン表に実在し、From が表の値と一致すること
        foreach (var d in AllowedScaleDrifts)
        {
            var row = ScaleBaseline.SingleOrDefault(e => e.Selector == d.Selector && e.Property == d.Property);
            Assert.True(row is not null, $"許可リストの宛先がベースライン表に無い: {d.Selector} {{ {d.Property} }}");
            Assert.Equal(row!.Value, d.From);
        }
    }

    // 許可リストが野放しにならないよう、中身の形も縛る。空でも将来のために効かせておく。
    [Fact]
    public void ScaleDriftAllowList_OnlyContainsRoundingsPermittedBySection4_2()
    {
        foreach (var d in AllowedScaleDrifts)
        {
            Assert.False(string.IsNullOrWhiteSpace(d.Reason), $"{d.Selector} の寄せに理由が書かれていない");

            var permitted =
                (d.Property == "font-size" && d.From == "13px" && d.To is "12px" or "14px") ||
                (d.Property == "font-size" && d.From == "15px" && d.To is "14px" or "16px") ||
                (d.Property == "border-radius" && d.From == "8px" && d.To == "10px");

            Assert.True(
                permitted,
                $"§4.2 が認めていない寄せ: {d.Selector} {{ {d.Property}: {d.From} → {d.To} }}");
        }
    }

    // ★ P1-6 で反転させた検査 (旧 ScaleTokens_DoNotExistYet)。
    //   「まだ無いこと」→「§4.2 の 12 トークンがちょうど揃っていること」。
    //   段を勝手に増やす (--fs-xxs 等) と赤くなる = トークン新設は PL 裁定事項。
    [Fact]
    public void ScaleTokens_AreExactlyTheTwelveStepsOfSection4_2()
    {
        var tokens = RootTokens();

        string[] expected =
        {
            "--fs-xs", "--fs-sm", "--fs-md", "--fs-lg",
            "--sp-1", "--sp-2", "--sp-3", "--sp-4", "--sp-6",
            "--r-sm", "--r-md", "--r-pill",
        };

        foreach (var name in expected)
        {
            Assert.True(tokens.ContainsKey(name), $"寸法トークン {name} が :root に無い");
        }

        var defined = tokens.Keys.Where(IsScaleTokenName).OrderBy(x => x, StringComparer.Ordinal).ToList();
        Assert.Equal(
            expected.OrderBy(x => x, StringComparer.Ordinal).ToList(),
            defined);
    }

    // スケール名が 4 段 / 5 段 / 3 段から漏れないこと。
    // app.css・scoped CSS のどこかで var(--fs-xxs) のような未定義段を参照し始めたら赤くする。
    [Theory]
    [InlineData("--fs-", new[] { "xs", "sm", "md", "lg" })]
    [InlineData("--sp-", new[] { "1", "2", "3", "4", "6" })]
    [InlineData("--r-", new[] { "sm", "md", "pill" })]
    public void ReferencedScaleSteps_StayWithinTheDeclaredSet(string prefix, string[] steps)
    {
        var allowed = steps.Select(s => prefix + s).ToHashSet(StringComparer.Ordinal);
        var used = new SortedSet<string>(StringComparer.Ordinal);

        foreach (var path in new[] { AppCssPath, MainLayoutCssPath, ReconnectCssPath })
        {
            var text = StripComments(File.ReadAllText(path));
            foreach (Match m in Regex.Matches(text, @"var\(\s*(--[A-Za-z0-9_-]+)"))
            {
                if (m.Groups[1].Value.StartsWith(prefix, StringComparison.Ordinal))
                {
                    used.Add(m.Groups[1].Value);
                }
            }
        }

        var outside = used.Where(u => !allowed.Contains(u)).ToList();
        Assert.True(outside.Count == 0, $"{prefix}* の段から外れた参照: " + string.Join(", ", outside));
    }

    // トークンそのものの健全性。段が入れ替わったり単位が壊れたりすると
    // 実効 px 表 (A)(A2) は「全部一致しない」形で落ちるので、原因が読める検査を別に置く。
    [Fact]
    public void ScaleTokens_AreMonotonicAndWithinSaneRanges()
    {
        var tokens = RootTokens();

        // タイポ: xs < sm < md < lg かつ 10–18px
        var fs = new[] { "--fs-xs", "--fs-sm", "--fs-md", "--fs-lg" }.Select(n => Px(tokens[n])).ToList();
        for (var i = 0; i < fs.Count; i++)
        {
            Assert.InRange(fs[i], 10, 18);
            if (i > 0)
            {
                Assert.True(fs[i - 1] < fs[i], $"--fs-* が単調増加でない: {string.Join(" / ", fs)}");
            }
        }

        // スペーシング: 4 の倍数かつ単調増加
        var sp = new[] { "--sp-1", "--sp-2", "--sp-3", "--sp-4", "--sp-6" }.Select(n => Px(tokens[n])).ToList();
        for (var i = 0; i < sp.Count; i++)
        {
            Assert.True(sp[i] % 4 == 0, $"--sp-* が 4px グリッドから外れた: {sp[i]}px");
            if (i > 0)
            {
                Assert.True(sp[i - 1] < sp[i], $"--sp-* が単調増加でない: {string.Join(" / ", sp)}");
            }
        }

        // 名前が刻み数を表しているので --sp-N = N * 4px であること
        Assert.Equal(new[] { 4, 8, 12, 16, 24 }, sp);

        // 角丸: sm < md、pill は 999px
        Assert.True(Px(tokens["--r-sm"]) < Px(tokens["--r-md"]), "--r-sm < --r-md であること");
        Assert.Equal(999, Px(tokens["--r-pill"]));
    }

    // ペイン見出しはセクションヘッダ型 (§4.2): 小さく (--fs-sm) 薄く (--muted) 広く (letter-spacing)。
    [Fact]
    public void PaneHeading_IsSectionHeaderStyle()
    {
        var head = Blocks(File.ReadAllText(AppCssPath))
            .Where(b => Normalize(b.Selector) == ".palette-head")
            .ToList();

        Assert.True(head.Count == 1, $".palette-head の宣言は 1 つであること (実際 {head.Count} 件)");

        var body = Normalize(head[0].Body);
        Assert.Contains("font-size: var(--fs-sm)", body);
        Assert.Contains("letter-spacing: 0.06em", body);
        Assert.Contains("color: var(--muted)", body);

        // 見出しクラスはタグパレットとプリセット一覧の両ペインで共有されている
        foreach (var path in new[] { TagPaletteRazorPath, PresetSidebarRazorPath })
        {
            Assert.Contains("class=\"palette-head\"", File.ReadAllText(path));
        }
    }

    private static bool IsScaleTokenName(string name)
        => name.StartsWith("--fs-", StringComparison.Ordinal)
           || name.StartsWith("--sp-", StringComparison.Ordinal)
           || name.StartsWith("--r-", StringComparison.Ordinal);

    private static int Px(string value)
    {
        var m = Regex.Match(value.Trim(), @"^(\d+)px$");
        Assert.True(m.Success, $"px 値として読めない: {value}");
        return int.Parse(m.Groups[1].Value);
    }

    // ══════════════════════════════════════════════════
    //  (A2) P1-6 用: padding / margin / gap の実効値ベースライン
    //
    //  (A) と同じ流儀。**置換前に緑を確認してから** var(--sp-*) を入れる。
    //  スペーシングには「寄せ」の許可リストを設けない — §4.2 が認めているのは
    //  font-size / border-radius の中間値だけで、余白は 1px でも動かさない。
    // ══════════════════════════════════════════════════

    // 4px グリッド。ここにぴったり一致する成分だけがトークン化の対象になる。
    private static readonly string[] SpacingGrid = { "4px", "8px", "12px", "16px", "24px" };

    // 現行 app.css の全宣言 (padding 33 + padding-right 1 + padding-bottom 1 +
    // margin 1 + margin-top 2 + margin-right 1 + margin-left 1 + gap 25 = 65)。
    // 末尾コメントの ★ は「全成分が 4px グリッド (0 は中立) = トークン化の対象」の印。
    private static readonly ScaleEntry[] SpacingBaseline =
    {
        // ── ベース / トップバー ──
        new("html, body",                     "margin",         "0"),
        new(".topbar",                        "gap",            "18px"),
        new(".topbar",                        "padding",        "8px 18px"),
        new(".lang-toggle",                   "gap",            "0"),
        new(".lang-toggle .btn",              "padding",        "4px 10px"),     // ← .lang-toggle button
        new(".conn",                          "gap",            "6px"),
        new(".conn .btn",                     "padding",        "2px 8px"),      // ← .conn-retry
        new(".conn .btn",                     "margin-left",    "4px"),          // ★ ← .conn-retry

        // ── 共通ボタン (P3-1) ──
        new(".btn-icon",                      "padding",        "0"),            // ← .ps-del

        // ── テレメトリ ──
        new(".telemetry-mini",                "gap",            "14px"),
        new(".tm-item",                       "gap",            "6px"),
        new(".gen-pill",                      "padding",        "2px 10px"),

        // ── 3 ペインの器 ──
        new(".main",                          "gap",            "16px"),         // ★
        new(".main",                          "padding",        "16px"),         // ★
        new(".sidebar",                       "gap",            "12px"),         // ★
        new(".sidebar",                       "padding-right",  "4px"),          // ★
        new(".gen",                           "gap",            "12px"),         // ★
        new(".gen",                           "padding",        "16px"),         // ★
        new(".canvas",                        "padding",        "16px"),         // ★
        new(".field",                         "gap",            "4px"),          // ★
        new(".row",                           "gap",            "12px"),         // ★
        new("textarea, select, input[type=\"range\"]", "padding", "8px"),        // ★

        // ── 生成ボタン行まわり ──
        new(".gen-actions",                   "gap",            "12px"),         // ★
        new(".gen-actions .btn",              "padding",        "12px"),         // ★ ← .generate
        new(".gen-reason",                    "margin-top",     "-4px"),         // 負値 (グリッド外)
        new(".error",                         "padding",        "8px 10px"),

        // ── プレビュー (右ペイン) ──
        new(".gen-mode",                      "padding",        "3px 10px"),
        new(".canvas-save",                   "padding",        "3px 10px"),     // ← .preview-save
        new(".preview-overlay",               "gap",            "12px"),         // ★

        // ── タグパレット (左ペイン) ──
        new(".palette, .preset-sidebar",      "padding",        "12px"),         // ★
        new(".palette, .preset-sidebar",      "gap",            "8px"),          // ★
        new(".palette-target",                "gap",            "6px"),
        new(".palette-target label",          "gap",            "4px"),          // ★
        new(".palette-target label",          "padding",        "4px 4px"),      // ★
        new(".palette-cat",                   "padding",        "0 8px"),        // ★
        new(".palette-cat > summary",         "padding",        "6px 2px"),
        new(".palette-tags",                  "gap",            "4px"),          // ★
        new(".palette-tags",                  "padding",        "4px 0 8px"),    // ★
        new(".palette-empty",                 "padding",        "4px 0 8px"),    // ★
        new(".palette-tag",                   "padding",        "2px 8px"),

        // ── お気に入り ──
        new(".fav-add",                       "padding",        "2px 4px 2px 10px"),
        new(".fav-x",                         "padding",        "0 6px 0 2px"),
        new(".fav-bar",                       "gap",            "6px"),
        new(".fav-bar",                       "padding-bottom", "8px"),          // ★
        new(".fav-entry",                     "padding",        "4px 8px"),      // ★
        new(".fav-bar .btn",                  "padding",        "0 10px"),       // ← .fav-plus

        // ── プリセット一覧 (左ペイン) ──
        new(".ps-list",                       "gap",            "4px"),          // ★
        new(".ps-list",                       "padding",        "4px 0 8px"),    // ★
        new(".ps-card",                       "gap",            "8px"),          // ★
        new(".ps-card",                       "padding",        "4px 6px"),

        // ── プリセット保存バー (中央) ──
        new(".presetbar",                     "gap",            "6px"),
        new(".preset-name",                   "padding",        "6px 8px"),
        new(".presetbar .btn",                "padding",        "6px 10px"),     // ← .preset-btn

        // ── チップ入力 ──
        new(".taginput .chips",               "gap",            "6px"),
        new(".taginput .chips",               "padding",        "6px"),
        new(".tag-chip",                      "gap",            "4px"),          // ★
        new(".tag-chip",                      "padding",        "2px 4px 2px 10px"),
        new(".chip-x",                        "padding",        "0 4px"),        // ★
        new(".chip-entry",                    "padding",        "4px"),          // ★

        // ── LoRA ──
        new(".lora-chips",                    "gap",            "6px"),
        new(".lora-chips",                    "margin-top",     "4px"),          // ★
        new(".lora-head",                     "margin-right",   "4px"),          // ★
        new(".lora-item",                     "gap",            "6px"),
        new(".lora-chip",                     "padding",        "2px 10px"),
        new(".lora-strength",                 "gap",            "4px"),          // ★
    };

    public static IEnumerable<object[]> SpacingBaselineData()
        => SpacingBaseline.Select(e => new object[] { e.Selector, e.Property, e.Value });

    [Theory]
    [MemberData(nameof(SpacingBaselineData))]
    public void EffectiveSpacingValue_MatchesBaseline(string selector, string property, string expected)
    {
        var actual = SpacingActual.Value
            .Where(x => x.Selector == selector && x.Property == property)
            .Select(x => x.Value)
            .ToList();

        Assert.True(
            actual.Count == 1,
            $"{selector} の {property} 宣言は 1 つであること (実際 {actual.Count} 件)");

        // 余白に「寄せ」は無い。var(--sp-*) へ畳んでも実効 px は完全一致であること。
        Assert.Equal(expected, actual[0]);
    }

    [Fact]
    public void SpacingBaseline_CoversEveryDeclarationInAppCss()
    {
        var actual = SpacingActual.Value;

        Assert.Equal(33, actual.Count(x => x.Property == "padding"));
        Assert.Equal(25, actual.Count(x => x.Property == "gap"));
        Assert.Equal(65, actual.Count);
        Assert.Equal(65, SpacingBaseline.Length);

        var baseline = SpacingBaseline.Select(e => (e.Selector, e.Property)).ToHashSet();
        var extra = actual
            .Where(a => !baseline.Contains((a.Selector, a.Property)))
            .Select(a => $"{a.Selector} {{ {a.Property} }}")
            .ToList();
        Assert.True(extra.Count == 0, "ベースライン表に無い余白宣言が増えている: " + string.Join(" / ", extra));

        var found = actual.Select(a => (a.Selector, a.Property)).ToHashSet();
        var missing = baseline.Where(b => !found.Contains(b)).Select(b => $"{b.Item1} {{ {b.Item2} }}").ToList();
        Assert.True(missing.Count == 0, "ベースライン表にあるのに app.css から消えた余白宣言: " + string.Join(" / ", missing));
    }

    // P1-6 の置換ルールそのものの検査。
    //   「全成分が 4px グリッド (0 は中立) の宣言だけを var(--sp-*) へ畳む。
    //     6px / 10px / -4px のような非グリッド値を含む宣言は 1 成分も触らない」
    // ルールを満たしているのに px 直書きが残っていれば取りこぼし = 赤。
    // ※ 逆向き (非グリッド値をトークンで書く) はトークン定義側が 4 の倍数しか持たないので
    //   ScaleTokens_AreMonotonicAndWithinSaneRanges が押さえる。
    [Fact]
    public void SpacingTokens_CoverEveryFullyOnGridDeclaration()
    {
        var tokens = RootTokens();
        var leftovers = new List<string>();

        foreach (var raw in SpacingRaw.Value)
        {
            var parts = ResolveMetricVars(raw.Value, tokens)
                .Split(' ', StringSplitOptions.RemoveEmptyEntries);

            var onGrid = parts.All(p => p == "0" || SpacingGrid.Contains(p, StringComparer.Ordinal));
            if (!onGrid || parts.All(p => p == "0"))
            {
                // 非グリッド値を含む宣言と、0 だけの宣言 (トークン不要) は対象外
                continue;
            }

            if (Regex.IsMatch(raw.Value, @"\d+px"))
            {
                leftovers.Add($"{raw.Selector} {{ {raw.Property}: {raw.Value} }}");
            }
        }

        Assert.True(
            leftovers.Count == 0,
            "4px グリッドに乗っているのに var(--sp-*) へ畳まれていない余白: " + string.Join(" / ", leftovers));
    }

    // ══════════════════════════════════════════════════
    //  (B) P3-1 用: ボタン系の 4 状態ベースライン
    // ══════════════════════════════════════════════════

    // ボタン 8 群 × 4 状態。
    // Chain = その状態を作っている app.css のセレクタを**カスケード順**に並べたもの
    //         (後ろが後勝ち)。P3-1 後はここを `.btn` + バリアント (+ 器の規則) へ
    //         差し替えるだけで、Declarations (期待値) は据え置きのまま緑になるはず、
    //         というのが本検査の趣旨。**★ P3-1 で書き換えたのは Chain だけ**で、
    //         期待宣言の literal は 1 文字も動かしていない (動かすなら AllowedButtonDiffs)。
    // Declarations = その状態の最終宣言集合 ("prop: value")。順序は問わない (ソートして比較)。
    private static readonly ButtonStateSpec[] ButtonBaseline =
    {
        // ── 1. 生成 (primary) ─────────────────────────
        new("generate(primary)", "base", new[] { ".btn", ".btn-primary", ".gen-actions .btn" }, new[]
        {
            "flex: 1",
            "background: var(--accent)",
            "color: var(--on-accent)",
            "border: none",
            "border-radius: 8px",
            "padding: 12px",
            "font-weight: 700",
            "font-size: 15px",
            "cursor: pointer",
        }),
        // primary には hover 規則が無い (accent 塗りのまま変化しない)。
        // ★ P3-1 でも空のまま = hover は .btn ではなく .btn-ghost / .btn-icon 側に置いた。
        //   .btn:hover にすると accent 塗りの上で文字色が動いて**見た目が変わる**ため。
        new("generate(primary)", "hover", Array.Empty<string>(), Array.Empty<string>()),
        new("generate(primary)", "disabled", new[] { ".btn:disabled" }, new[]
        {
            "opacity: 0.6",
            "cursor: default",
        }),
        new("generate(primary)", "focus-visible", new[] { "button:focus-visible" }, FocusRingDecls),

        // ── 2. 下書き (secondary) ─────────────────────
        // .btn (1) → .btn-ghost (1) → .gen-actions .btn / .gen-actions .btn-ghost (2) = 後勝ち。
        new("generate.secondary", "base",
            new[] { ".btn", ".btn-ghost", ".gen-actions .btn", ".gen-actions .btn-ghost" }, new[]
        {
            "flex: 1",
            "background: var(--panel-2)",
            "color: var(--text)",
            "border: 1px solid var(--border-strong)",
            "border-radius: 8px",
            "padding: 12px",
            "font-weight: 600",
            "font-size: 14px",
            "cursor: pointer",
        }),
        new("generate.secondary", "hover",
            new[] { ".btn-ghost:hover:not(:disabled)", ".gen-actions .btn-ghost:hover:not(:disabled)" }, new[]
        {
            "border-color: var(--accent)",
            "color: var(--accent)",
        }),
        new("generate.secondary", "disabled", new[] { ".btn:disabled" }, new[]
        {
            "opacity: 0.6",
            "cursor: default",
        }),
        new("generate.secondary", "focus-visible", new[] { "button:focus-visible" }, FocusRingDecls),

        // ── 3. プリセット保存バーのボタン ─────────────
        new(".preset-btn", "base", new[] { ".btn", ".btn-ghost", ".presetbar .btn" }, new[]
        {
            "background: var(--panel-2)",
            "color: var(--text)",
            "border: 1px solid var(--border-strong)",
            "border-radius: 6px",
            "padding: 6px 10px",
            "font: inherit",
            "cursor: pointer",
        }),
        // 枠だけ動かす hover は .btn-ghost 側にそのまま載っている (文字色は動かさない)。
        new(".preset-btn", "hover", new[] { ".btn-ghost:hover:not(:disabled)" }, new[]
        {
            "border-color: var(--accent)",
        }),
        new(".preset-btn", "disabled", new[] { ".btn:disabled", ".presetbar .btn:disabled" }, new[]
        {
            "opacity: 0.45",
            "cursor: default",
        }),
        new(".preset-btn", "focus-visible", new[] { "button:focus-visible" }, FocusRingDecls),

        // ── 4. お気に入り追加 (+) ─────────────────────
        // .preset-btn とは padding だけが違う (0 10px vs 6px 10px)。P3-1 の統合対象そのもの。
        // 統合後は「.btn .btn-ghost + 器の padding」だけの差になっている。
        new(".fav-plus", "base", new[] { ".btn", ".btn-ghost", ".fav-bar .btn" }, new[]
        {
            "background: var(--panel-2)",
            "color: var(--text)",
            "border: 1px solid var(--border-strong)",
            "border-radius: 6px",
            "padding: 0 10px",
            "font: inherit",
            "cursor: pointer",
        }),
        new(".fav-plus", "hover", new[] { ".btn-ghost:hover:not(:disabled)" }, new[]
        {
            "border-color: var(--accent)",
        }),
        new(".fav-plus", "disabled", new[] { ".btn:disabled", ".fav-bar .btn:disabled" }, new[]
        {
            "opacity: 0.45",
            "cursor: default",
        }),
        new(".fav-plus", "focus-visible", new[] { "button:focus-visible" }, FocusRingDecls),

        // ── 5. 言語トグル (セグメント) ────────────────
        // 器 (.lang-toggle) が枠と角丸を持つので、ghost の枠は border: none で、
        // 角丸は border-radius: 0 で打ち消す (後者は AllowedButtonDiffs #1)。
        new(".lang-toggle button", "base", new[] { ".btn", ".btn-ghost", ".lang-toggle .btn" }, new[]
        {
            "appearance: none",
            "border: none",
            "background: var(--panel-2)",
            "color: var(--muted)",
            "font-size: 12px",
            "padding: 4px 10px",
            "cursor: pointer",
            "transition: background 0.12s, color 0.12s",
        }),
        // hover 作法がここだけ「文字色替え」(他は枠色替え)。文字色は器側で据え置き、
        // ghost の border-color が上から載る (border: none なので描画には出ない = 差分 #2)。
        new(".lang-toggle button", "hover",
            new[] { ".btn-ghost:hover:not(:disabled)", ".lang-toggle .btn:hover" }, new[]
        {
            "color: var(--text)",
        }),
        // ★ 元から :disabled 規則が無い群。P3-1 後は .btn:disabled が構文上は当たるが、
        //   この 3 群 (lang-toggle / ps-del / preview-save) は disabled 属性を持たない
        //   ことを ButtonsWithoutDisabledRule_AreNeverRenderedDisabled が機械保証する。
        //   よって表示は不変で、Chain は空のまま = 意図的差分ではない。
        new(".lang-toggle button", "disabled", Array.Empty<string>(), Array.Empty<string>()),
        new(".lang-toggle button", "focus-visible", new[] { "button:focus-visible" }, FocusRingDecls),

        // ── 6. プリセット削除 (アイコンボタン) ────────
        new(".ps-del", "base", new[] { ".btn", ".btn-icon" }, new[]
        {
            "width: 20px",
            "height: 20px",
            "line-height: 18px",
            "text-align: center",
            "background: transparent",
            "color: var(--muted)",
            "border: 1px solid var(--border-strong)",
            "border-radius: 4px",
            "cursor: pointer",
            "font-size: 13px",
            "padding: 0",
            "flex-shrink: 0",
        }),
        // 削除だけ hover が ng 系 (赤) — 破壊操作なので P3-1 でも残した (.btn-icon の作法)。
        new(".ps-del", "hover", new[] { ".btn-icon:hover:not(:disabled)" }, new[]
        {
            "border-color: var(--ng)",
            "color: var(--ng-soft)",
        }),
        new(".ps-del", "disabled", Array.Empty<string>(), Array.Empty<string>()),
        new(".ps-del", "focus-visible", new[] { "button:focus-visible" }, FocusRingDecls),

        // ── 7. 画像保存 (button ではなく a[download]) ──
        new(".preview-save", "base", new[] { ".btn", ".btn-ghost", ".canvas-save" }, new[]
        {
            "position: absolute",
            "top: 10px",
            "right: 10px",
            "z-index: 1",
            "background: var(--panel-2)",
            "color: var(--text)",
            "border: 1px solid var(--border-strong)",
            "border-radius: 6px",
            "padding: 3px 10px",
            "font-size: 11px",
            "font-weight: 600",
            "line-height: 1.6",
            "text-decoration: none",
            "white-space: nowrap",
            "cursor: pointer",
        }),
        // 統合後は :not(:disabled) 付きの共通セレクタに当たる。a は決して :disabled に
        // ならないので :not(:disabled) は常に真 = 挙動不変。
        new(".preview-save", "hover",
            new[] { ".btn-ghost:hover:not(:disabled)", ".canvas-save:hover:not(:disabled)" }, new[]
        {
            "border-color: var(--accent)",
            "color: var(--accent)",
        }),
        new(".preview-save", "disabled", Array.Empty<string>(), Array.Empty<string>()),
        // a 要素なので button:focus-visible には当たらない。リング統一ブロックの
        // .preview-save:focus-visible を P3-1 で .btn:focus-visible へ張り替えた
        // (a に .btn を付ける以上、リング側にも明示的な受け皿が要る)。
        new(".preview-save", "focus-visible", new[] { ".btn:focus-visible" }, FocusRingDecls),

        // ── 8. 再接続 (トップバーの極小ボタン) ────────
        new(".conn-retry", "base", new[] { ".btn", ".btn-ghost", ".conn .btn" }, new[]
        {
            "appearance: none",
            "background: var(--panel-2)",
            "color: var(--text)",
            "border: 1px solid var(--border-strong)",
            "border-radius: 6px",
            "font-size: 11px",
            "font-weight: 600",
            "padding: 2px 8px",
            "margin-left: 4px",
            "cursor: pointer",
            "white-space: nowrap",
            "transition: border-color 0.12s, color 0.12s",
        }),
        new(".conn-retry", "hover",
            new[] { ".btn-ghost:hover:not(:disabled)", ".conn .btn:hover:not(:disabled)" }, new[]
        {
            "border-color: var(--accent)",
            "color: var(--accent)",
        }),
        // disabled の opacity がここだけ 0.6 (preset-btn / fav-plus は 0.45)。
        // P3-1 では 0.6 を共通 (.btn:disabled) に据え、0.45 の 2 群を器側で残した =
        // どちらの見た目も変えずに宣言だけ畳んでいる (許可リストの消費ゼロ)。
        new(".conn-retry", "disabled", new[] { ".btn:disabled" }, new[]
        {
            "opacity: 0.6",
            "cursor: default",
        }),
        new(".conn-retry", "focus-visible", new[] { "button:focus-visible" }, FocusRingDecls),
    };

    // フォーカス統一リング (P1-2) は 1 ブロックで全ボタンに当たる。8 群で同じ値を共有する。
    private static string[] FocusRingDecls =>
    new[]
    {
        "outline: none",
        "border-color: var(--accent)",
        "box-shadow: var(--edge), 0 0 0 2px var(--focus-ring)",
    };

    // P3-1 で「意図的に変える」差分。PL 裁定により **上限 6 件**。
    //
    // ★ 実際に使ったのは 2 件だけで、どちらも言語トグルのセグメントに .btn-ghost が
    //   載ったことで**宣言が 1 つ増えた**もの。2 件とも「描画結果は同じ」= 見た目は不変。
    //   0.45/0.6 の 2 段や hover の枠のみ/枠+文字の別は、器側の規則として残すことで
    //   差分 0 のまま畳んだ (安易に統一して枠を消費しない方針)。
    private static readonly ButtonDiff[] AllowedButtonDiffs =
    {
        // (1) ghost の角丸をセグメントで打ち消す。従来は border-radius 宣言が無く実効 0、
        //     打ち消し後も 0 なので描画は同じ。宣言が 1 つ増えただけ。
        //     打ち消さないと .lang-toggle (角丸 + overflow: hidden) の中でセグメントが
        //     丸まり、区切りに隙間が見える = そちらが本物の見た目変化になる。
        new(".lang-toggle button", "base", "border-radius", "", "0",
            "器 (.lang-toggle) が角丸を持つので ghost の --r-sm を 0 で打ち消す。実効値は従来と同じ 0"),

        // (2) hover の枠色。セグメントは border: none なので border-color は 1px も描かれない。
        //     文字色は器側 (.lang-toggle .btn:hover) が var(--text) のまま勝つ。
        new(".lang-toggle button", "hover", "border-color", "", "var(--accent)",
            "共通 hover (.btn-ghost) の枠色が載るが、セグメントは border: none なので描画に出ない"),
    };

    private const int MaxAllowedButtonDiffs = 6;

    // 実際に使った件数。増やすときは必ずレビューが要る (PL 裁定)。
    private const int ExpectedButtonDiffs = 2;

    public static IEnumerable<object[]> ButtonBaselineData()
        => ButtonBaseline.Select(s => new object[] { s.Button, s.State });

    [Theory]
    [MemberData(nameof(ButtonBaselineData))]
    public void ButtonState_MatchesBaselineDeclarations(string button, string state)
    {
        var spec = ButtonBaseline.Single(s => s.Button == button && s.State == state);

        // 実際の CSS からセレクタ連鎖を辿って後勝ち合成する
        var actual = MergeSelectorChain(spec.Chain);

        // 期待値 = 凍結した宣言集合 + 許可された意図的差分
        var expected = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var d in spec.Declarations)
        {
            var i = d.IndexOf(':');
            expected[d[..i].Trim()] = d[(i + 1)..].Trim();
        }

        // (A) の許可リストで寄せた値はここにも波及する (同じ app.css を別角度から見ているだけ)。
        // ★ ベースライン表の literal は書き換えない — 寄せの正当性は AllowedScaleDrifts 側
        //   (§4.2 の形 + 件数固定) で 1 箇所に閉じ込めておきたいため。
        //   連鎖上のセレクタが持つ値と From が一致するときだけ適用するので、
        //   後勝ちで上書きされている値 (例: .generate.secondary の font-size) は動かない。
        foreach (var drift in AllowedScaleDrifts.Where(d => spec.Chain.Contains(d.Selector, StringComparer.Ordinal)))
        {
            if (expected.TryGetValue(drift.Property, out var current) && current == drift.From)
            {
                expected[drift.Property] = drift.To;
            }
        }

        foreach (var diff in AllowedButtonDiffs.Where(d => d.Button == button && d.State == state))
        {
            if (diff.Before.Length > 0)
            {
                Assert.True(
                    expected.TryGetValue(diff.Property, out var before) && before == diff.Before,
                    $"許可リストの Before がベースラインと違う: {button} {state} {diff.Property}");
            }

            if (diff.After.Length == 0)
            {
                expected.Remove(diff.Property);
            }
            else
            {
                expected[diff.Property] = diff.After;
            }
        }

        Assert.Equal(Render(expected), Render(actual));
    }

    // ★ P3-1 で反転させた検査 (旧 ButtonDiffAllowList_IsEmptyBeforeP3_1)。
    //   「空であること」→「レビュー済みの 2 件ちょうどであること」。
    //   件数を定数で固定しているので、差分を 1 件足すだけでも赤くなる。
    [Fact]
    public void ButtonDiffAllowList_MatchesTheReviewedDiffs()
    {
        Assert.Equal(ExpectedButtonDiffs, AllowedButtonDiffs.Length);
        Assert.True(
            AllowedButtonDiffs.Length <= MaxAllowedButtonDiffs,
            $"意図的差分は {MaxAllowedButtonDiffs} 件まで (PL 裁定)。実際 {AllowedButtonDiffs.Length} 件");

        // 同じ (群, 状態, プロパティ) を二重に積まない (後勝ちで検査が骨抜きになるのを防ぐ)
        var keys = AllowedButtonDiffs.Select(d => (d.Button, d.State, d.Property)).ToList();
        Assert.Equal(keys.Count, keys.Distinct().Count());

        // 許可リストの宛先が実在する状態であること (書き間違いで検査が素通りするのを防ぐ)
        foreach (var d in AllowedButtonDiffs)
        {
            Assert.False(string.IsNullOrWhiteSpace(d.Reason), $"{d.Button} {d.State} の差分に理由が無い");
            Assert.Contains(ButtonBaseline, s => s.Button == d.Button && s.State == d.State);
        }
    }

    // ══════════════════════════════════════════════════
    //  (C) 旧クラス名の在庫 — P3-1 で反転させる検査
    // ══════════════════════════════════════════════════

    // ★ P3-1 で反転させた検査 (旧 LegacyButtonFamily_OwnsExactlyTheseSelectorsBeforeP3_1)。
    //   「この 7 系統がこれらのセレクタを持つ」→「7 系統とも app.css から消えている」。
    //   接頭辞一致なので `.generate` を 1 つでも書き戻すと赤くなる。
    public static IEnumerable<object[]> LegacyButtonFamilyData() => new[]
    {
        new object[] { ".generate" },
        new object[] { ".preset-btn" },
        new object[] { ".fav-plus" },
        new object[] { ".lang-toggle button" },
        new object[] { ".ps-del" },
        new object[] { ".preview-save" },
        new object[] { ".conn-retry" },
    };

    [Theory]
    [MemberData(nameof(LegacyButtonFamilyData))]
    public void LegacyButtonFamily_IsFullyRetiredFromAppCss(string prefix)
    {
        var actual = SelectorPartsStartingWith(prefix);

        Assert.True(
            actual.Count == 0,
            $"P3-1 で退役したはずの {prefix} 系のセレクタが app.css に残っている: " + string.Join(" / ", actual));
    }

    // ★ P3-1 で反転させた検査 (旧 BtnClasses_DoNotExistYet)。
    //   「まだ無いこと」→「4 クラスがこの規則だけを持つこと」。
    //   .btn-lg のような 5 つめのバリアントを足すと赤くなる = 語彙を増やすのは裁定事項。
    //   ★ 器側の規則 (.gen-actions .btn 等) は**この在庫に含めない** — 接頭辞一致の
    //     判定 (SelectorPartsStartingWith) が `.btn` で始まる部品しか拾わないため。
    //     器側は寸法だけを持つ「置き場所の責務」で、ボタンの語彙ではない。
    public static IEnumerable<object[]> BtnFamilyData() => new[]
    {
        new object[] { ".btn", new[]
        {
            ".btn",
            ".btn:disabled",
            ".btn:focus-visible",
        }},
        new object[] { ".btn-primary", new[]
        {
            ".btn-primary",
        }},
        new object[] { ".btn-ghost", new[]
        {
            ".btn-ghost",
            ".btn-ghost:hover:not(:disabled)",
        }},
        new object[] { ".btn-icon", new[]
        {
            ".btn-icon",
            ".btn-icon:hover:not(:disabled)",
        }},
    };

    [Theory]
    [MemberData(nameof(BtnFamilyData))]
    public void BtnFamily_OwnsExactlyTheseSelectors(string prefix, string[] expected)
    {
        var actual = SelectorPartsStartingWith(prefix);

        Assert.Equal(
            string.Join("\n", expected.OrderBy(x => x, StringComparer.Ordinal)),
            string.Join("\n", actual));
    }

    // ★ P3-1 で反転させた検査 (旧 LegacyButtonClass_AppearsExactlyOnceInRazorBeforeP3_1)。
    //   旧クラス名 7 種が razor から**クラストークンとして** 1 つも出てこないこと。
    //   文字列一致ではなく class 属性を解析してトークン比較するので、
    //   `class="foo preset-btn"` のような並べ方でも捕まえる。
    public static IEnumerable<object[]> RetiredButtonClassData() => new[]
    {
        new object[] { "generate" },
        new object[] { "secondary" },
        new object[] { "conn-retry" },
        new object[] { "preview-save" },
        new object[] { "preset-btn" },
        new object[] { "fav-plus" },
        new object[] { "ps-del" },
    };

    [Theory]
    [MemberData(nameof(RetiredButtonClassData))]
    public void RetiredButtonClass_IsGoneFromEveryRazor(string cls)
    {
        var hits = new List<string>();

        foreach (var path in RazorFiles())
        {
            foreach (var token in ClassTokens(File.ReadAllText(path)))
            {
                if (token == cls)
                {
                    hits.Add(Path.GetFileName(path));
                }
            }
        }

        Assert.True(hits.Count == 0, $"退役したクラス {cls} が razor に残っている: " + string.Join(" / ", hits));
    }

    // 器 (.lang-toggle) 自体のクラスは残る — 消えたのは中の button の装飾セレクタだけ。
    [Fact]
    public void LangToggleContainer_KeepsItsClass()
    {
        var razor = File.ReadAllText(GenerateRazorPath);
        var count = Regex.Matches(razor, Regex.Escape("class=\"lang-toggle\"")).Count;

        Assert.True(count == 1, $"器 .lang-toggle のクラスは 1 箇所であること (実際 {count} 箇所)");
    }

    // ★ P3-1 で反転させた検査 (旧 LangToggleButtons_HaveNoStaticClassBeforeP3_1)。
    //   子孫セレクタ頼みだったセグメントに .btn が生え、状態クラス (.on) は式のまま残る。
    [Fact]
    public void LangToggleButtons_CarryBtnClassAlongsideTheirStateClass()
    {
        var razor = File.ReadAllText(GenerateRazorPath);

        Assert.Contains("class=\"btn btn-ghost @(_tagLang == \"ja\" ? \"on\" : \"\")\"", razor);
        Assert.Contains("class=\"btn btn-ghost @(_tagLang == \"en\" ? \"on\" : \"\")\"", razor);
    }

    // ══════════════════════════════════════════════════
    //  (C2) P3-1 の張り替え漏れ全数検査
    //
    //  (C) が「app.css から旧セレクタが消えたか」を見るのに対し、こちらは razor 側。
    //  張り替え漏れ = クラスが 1 つ落ちるだけで素の <button> に戻り、UA 既定の
    //  灰色ボタンが 1 つだけ紛れ込む — 目視では気づきにくいので機械で全数を見る。
    // ══════════════════════════════════════════════════

    private static readonly string[] BtnVariants = { "btn-primary", "btn-ghost", "btn-icon" };

    // 「.btn を持つなら必ずバリアントちょうど 1 つ」「バリアントを持つなら必ず .btn」。
    //   0 個 → 素の button に落ちて崩壊 / 2 個 → 後勝ちで見た目が不定。
    [Fact]
    public void EveryBtnElement_HasTheBaseClassAndExactlyOneVariant()
    {
        var bad = new List<string>();
        var variantCount = new Dictionary<string, int>(StringComparer.Ordinal);
        var total = 0;

        foreach (var path in AllRazorFiles())
        {
            foreach (var tag in ElementTags(File.ReadAllText(path)))
            {
                var tokens = SplitClassTokens(ClassAttribute(tag) ?? "").Static;
                var variants = tokens.Where(t => BtnVariants.Contains(t, StringComparer.Ordinal)).ToList();
                var hasBase = tokens.Contains("btn", StringComparer.Ordinal);

                if (!hasBase && variants.Count == 0)
                {
                    continue;
                }

                total++;
                var where = $"{Path.GetFileName(path)}: {Squeeze(tag)}";

                if (!hasBase)
                {
                    bad.Add($"{where} … バリアントだけで .btn が無い");
                }

                if (variants.Count != 1)
                {
                    bad.Add($"{where} … バリアントが {variants.Count} 個 ({string.Join("+", variants)})");
                }

                foreach (var v in variants)
                {
                    variantCount[v] = variantCount.GetValueOrDefault(v) + 1;
                }
            }
        }

        Assert.True(bad.Count == 0, string.Join("\n", bad));

        // 張り替えた 9 箇所ちょうど (生成 2 / 言語トグル 2 / 再接続 / 画像保存 /
        // プリセット保存 / お気に入り + / プリセット削除)。増減はレビュー対象。
        Assert.Equal(9, total);
        Assert.Equal(1, variantCount.GetValueOrDefault("btn-primary"));
        Assert.Equal(7, variantCount.GetValueOrDefault("btn-ghost"));
        Assert.Equal(1, variantCount.GetValueOrDefault("btn-icon"));
    }

    // 統合の前後で「どの要素がどのバリアントになったか」を 1 箇所に凍結する。
    // 目印は razor 中の一意な文字列 (ハンドラ名 / 属性) で、クラス名には依存しない。
    [Theory]
    [InlineData("Generate.razor",       "GenerateAsync(false)", "btn-primary")]
    [InlineData("Generate.razor",       "GenerateAsync(true)",  "btn-ghost")]
    [InlineData("Generate.razor",       "SetTagLang(\"ja\")",   "btn-ghost")]
    [InlineData("Generate.razor",       "SetTagLang(\"en\")",   "btn-ghost")]
    [InlineData("Generate.razor",       "ReconnectAsync",       "btn-ghost")]
    [InlineData("Generate.razor",       "download=",            "btn-ghost")]
    [InlineData("TagPresetField.razor", "SaveCurrent",          "btn-ghost")]
    [InlineData("TagPalette.razor",     "AddFavorite",          "btn-ghost")]
    [InlineData("PresetSidebar.razor",  "Delete(grp.Kind, p)",  "btn-icon")]
    public void ReplacedButton_MapsToTheExpectedVariant(string file, string marker, string variant)
    {
        var path = AllRazorFiles().Single(p => Path.GetFileName(p) == file);
        var tags = ElementTags(File.ReadAllText(path))
            .Where(t => t.Contains(marker, StringComparison.Ordinal))
            .ToList();

        Assert.True(tags.Count == 1, $"{file} で {marker} を持つ開始タグは 1 つであること (実際 {tags.Count} 個)");

        var tokens = SplitClassTokens(ClassAttribute(tags[0]) ?? "").Static;
        Assert.Contains("btn", tokens);
        Assert.Contains(variant, tokens);
    }

    // 画像保存だけは <a download>。button ではないので、リング統一ブロック側に
    // 受け皿 (.btn:focus-visible) が要る (P2-3 で同じ罠を踏んでいる)。
    [Fact]
    public void FocusRing_AcceptsBtnClassBecauseItIsAlsoUsedOnAnAnchor()
    {
        var saveTag = ElementTags(File.ReadAllText(GenerateRazorPath))
            .Single(t => t.Contains("download=", StringComparison.Ordinal));

        Assert.StartsWith("<a", saveTag);
        Assert.Contains("btn", SplitClassTokens(ClassAttribute(saveTag) ?? "").Static);

        // フォーカスリングの宣言は 1 ブロックのまま (P1-2 の契約) で、そこに .btn が居ること
        var ring = Blocks(File.ReadAllText(AppCssPath))
            .Where(b => b.Body.Contains("var(--focus-ring)")).ToList();

        Assert.True(ring.Count == 1, $"var(--focus-ring) を使うブロックは 1 つであること (実際 {ring.Count} 件)");
        Assert.Contains(".btn:focus-visible", SelectorParts(ring[0].Selector));
    }

    // :disabled 規則が無かった 3 群 (言語トグル / 削除 / 画像保存) に .btn:disabled が
    // 構文上は当たるようになった。表示が変わらない根拠は「disabled にならない」ことなので、
    // 散文ではなく razor で機械保証する。
    [Fact]
    public void ButtonsWithoutDisabledRule_AreNeverRenderedDisabled()
    {
        var disabled = new List<string>();

        foreach (var path in AllRazorFiles())
        {
            foreach (var tag in ElementTags(File.ReadAllText(path)))
            {
                var tokens = SplitClassTokens(ClassAttribute(tag) ?? "").Static;
                if (!tokens.Contains("btn", StringComparer.Ordinal))
                {
                    continue;
                }

                // 元から :disabled 規則を持たない 3 群 (計 4 要素)
                var noDisabledRule =
                    tag.Contains("SetTagLang", StringComparison.Ordinal)
                    || tag.Contains("Delete(grp.Kind, p)", StringComparison.Ordinal)
                    || tag.Contains("download=", StringComparison.Ordinal);

                if (tag.Contains("disabled=", StringComparison.Ordinal))
                {
                    Assert.False(noDisabledRule,
                        "元は :disabled 規則が無かった要素に disabled 属性が付いた " +
                        $"(.btn:disabled が新たに見た目を変える): {Path.GetFileName(path)}: {Squeeze(tag)}");
                    disabled.Add(Squeeze(tag));
                }
            }
        }

        // disabled を出しうるのは生成 2 + 再接続 + プリセット保存 + お気に入り + の 5 つ。
        // どれも元から :disabled 規則を持っていた群 (0.6 / 0.45) なので、
        // .btn:disabled への集約で見た目は変わらない。
        Assert.Equal(5, disabled.Count);
    }

    // 同詳細度 (0,3,0) で衝突する組の**記述順**を固定する。
    // 詳細度モデルを持たない (B) の後勝ち合成では捕まえられない、けれど実際に
    // 見た目が壊れる並びなので、順序そのものを検査する。
    //   ・.btn-ghost:hover の border-color vs 言語トグルの区切り線 (border-left)
    //     → 共通ボタンが先。後ろへ動かすと区切り線が hover で accent に化ける
    //   ・.lang-toggle .btn:hover の color vs .on の color
    //     → .on が後。逆にすると選択中セグメントの文字が hover で暗色に落ちる
    [Theory]
    [InlineData(".btn-ghost:hover:not(:disabled)", ".lang-toggle .btn + .btn")]
    [InlineData(".lang-toggle .btn:hover", ".lang-toggle .btn.on")]
    public void SameSpecificityRules_KeepTheOrderThatDecidesTheWinner(string earlier, string later)
    {
        var blocks = AppCssBlocks.Value;

        var i = blocks.FindIndex(b => SelectorParts(b.Selector).Contains(earlier, StringComparer.Ordinal));
        var j = blocks.FindIndex(b => SelectorParts(b.Selector).Contains(later, StringComparer.Ordinal));

        Assert.True(i >= 0, $"{earlier} が app.css に無い");
        Assert.True(j >= 0, $"{later} が app.css に無い");
        Assert.True(i < j, $"同詳細度なので {earlier} は {later} より前に書くこと (後勝ちで {later} を勝たせる)");
    }

    // P2-6 の --edge 合成に .btn が割り込んでいないこと。
    // .btn 側が box-shadow を持つと、追加先の左縁かフォーカスリングのどちらかが消える。
    [Fact]
    public void BtnRules_DoNotJoinTheEdgeShadowComposition()
    {
        var offenders = Blocks(File.ReadAllText(AppCssPath))
            .Where(b => SelectorParts(b.Selector).Any(p => p.StartsWith(".btn", StringComparison.Ordinal)))
            .Where(b => Normalize(b.Body).Contains("box-shadow") || Normalize(b.Body).Contains("--edge"))
            .ToList();

        // 唯一の例外がフォーカス統一リング (共有ブロック・P1-2/P2-6 側の検査が中身を縛る)
        Assert.True(
            offenders.Count == 1 && offenders[0].Body.Contains("var(--focus-ring)"),
            "リング統一ブロック以外の .btn 規則が box-shadow / --edge に触っている: "
            + string.Join(" / ", offenders.Select(o => o.Selector)));

        // --edge を打ち消しているチップの × は .btn 化していないこと
        // (.btn 化すると打ち消しの効く範囲が変わり、追加先フィールドの × に左縁が乗る)
        var chipX = ElementTags(File.ReadAllText(Path.Combine(RepoRoot(), "ui", "Components", "Shared", "TagInput.razor")))
            .Where(t => t.Contains("chip-x", StringComparison.Ordinal))
            .ToList();

        Assert.True(chipX.Count == 1, $".chip-x の要素は 1 つであること (実際 {chipX.Count} 個)");
        Assert.DoesNotContain("btn", SplitClassTokens(ClassAttribute(chipX[0]) ?? "").Static);
    }

    // ══════════════════════════════════════════════════
    //  (D) 検査装置そのものの自己検査
    //
    //  ベースラインは「現行 app.css に対して緑」なので、抽出器が壊れていても
    //  (例: 何も拾わない・var() を解決しない) 緑のまま素通りしうる。
    //  合成 CSS を食わせて「噛んでいること」と「P1-6 後も同じ答えを出すこと」を確かめる。
    //  ★ ここが緑でないと (A)(B) の緑には意味が無い。
    // ══════════════════════════════════════════════════

    // P1-6 後を先取りした合成 CSS。直書き px を var(--fs-*) / var(--r-*) 経由へ畳んでも
    // 抽出結果 (実効 px) が変わらないこと = ベースライン表が置換に耐えること。
    private const string SyntheticBefore = """
        :root { --accent: #6ea8fe; }
        .a { font-size: 12px; border-radius: 6px; }
        .b { font-size: 15px; color: var(--accent); }
        """;

    private const string SyntheticAfter = """
        :root { --accent: #6ea8fe; --fs-sm: 12px; --fs-lg: 15px; --r-sm: 6px; }
        .a { font-size: var(--fs-sm); border-radius: var(--r-sm); }
        .b { font-size: var(--fs-lg); color: var(--accent); }
        """;

    [Fact]
    public void Scanner_ResolvesTokensSoP1_6ReplacementIsInvisible()
    {
        var before = ScanScaleDeclarations(Blocks(SyntheticBefore), SyntheticTokens(SyntheticBefore));
        var after = ScanScaleDeclarations(Blocks(SyntheticAfter), SyntheticTokens(SyntheticAfter));

        // 抽出できていること (空を返して素通りしていないこと)
        Assert.Equal(3, before.Count);

        static string Dump(List<ScaleEntry> x)
            => string.Join("\n", x.Select(e => $"{e.Selector} {{ {e.Property}: {e.Value} }}"));

        Assert.Equal(".a { font-size: 12px }\n.a { border-radius: 6px }\n.b { font-size: 15px }", Dump(before));

        // 直書き → var() へ畳んでも実効値は不変
        Assert.Equal(Dump(before), Dump(after));
    }

    // 余白側の自己検査。ショートハンドの一部だけが var() になっても実効値が読めること
    // (= 「4px 0 8px」→「var(--sp-1) 0 var(--sp-2)」の置換がベースライン表に対して透明であること)。
    [Fact]
    public void Scanner_ResolvesSpacingTokensInsideShorthand()
    {
        const string before = """
            :root { --accent: #6ea8fe; }
            .a { padding: 4px 0 8px; gap: 12px; margin-left: 4px; }
            """;
        const string after = """
            :root { --accent: #6ea8fe; --sp-1: 4px; --sp-2: 8px; --sp-3: 12px; }
            .a { padding: var(--sp-1) 0 var(--sp-2); gap: var(--sp-3); margin-left: var(--sp-1); }
            """;

        static string Dump(string css)
            => string.Join("\n", ScanDeclarations(Blocks(css), SyntheticTokens(css), IsSpacingProperty)
                .Select(e => $"{e.Selector} {{ {e.Property}: {e.Value} }}"));

        Assert.Equal(
            ".a { padding: 4px 0 8px }\n.a { gap: 12px }\n.a { margin-left: 4px }",
            Dump(before));
        Assert.Equal(Dump(before), Dump(after));
    }

    [Fact]
    public void Scanner_DetectsChangedPixelValue()
    {
        // 12px → 13px の 1 文字差を確実に見分けること (= ベースラインが変更検知として働く)
        var mutated = ScanScaleDeclarations(
            Blocks(SyntheticBefore.Replace("12px", "13px")),
            SyntheticTokens(SyntheticBefore));

        Assert.Equal("13px", mutated.Single(e => e.Selector == ".a" && e.Property == "font-size").Value);
    }

    [Theory]
    // 寸法トークンは展開する (P1-6 の置換を吸収する)
    [InlineData("var(--fs-sm)", "12px")]
    [InlineData("var(--r-sm)", "6px")]
    // var の連鎖も辿る (--edge: var(--edge-off) のような形)
    [InlineData("var(--chain)", "8px")]
    // 色トークンは展開しない — 配色を変えただけで寸法表が壊れる偽陽性を避けるため
    [InlineData("var(--accent)", "var(--accent)")]
    [InlineData("1px solid var(--accent)", "1px solid var(--accent)")]
    // 未定義はそのまま (勝手に空へ潰さない)
    [InlineData("var(--nope)", "var(--nope)")]
    // 直書きは素通り
    [InlineData("6px 10px", "6px 10px")]
    public void ResolveMetricVars_ExpandsOnlyPureLengths(string input, string expected)
    {
        var tokens = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["--fs-sm"] = "12px",
            ["--r-sm"] = "6px",
            ["--chain"] = "var(--chain2)",
            ["--chain2"] = "8px",
            ["--accent"] = "#6ea8fe",
        };

        Assert.Equal(expected, ResolveMetricVars(input, tokens));
    }

    [Fact]
    public void MergeSelectorChain_AppliesLastWins()
    {
        // .btn + バリアント 1 枚を重ねたときの後勝ち解決 (P3-1 後の比較モデル)
        const string css = """
            .btn { background: var(--panel-2); padding: 6px; border: none; }
            .btn-ghost { background: transparent; color: var(--text); }
            """;

        var merged = MergeSelectorChain(
            new[] { ".btn", ".btn-ghost" },
            Blocks(css),
            new Dictionary<string, string>(StringComparer.Ordinal));

        Assert.Equal(
            "background: transparent\nborder: none\ncolor: var(--text)\npadding: 6px",
            Render(merged));
    }

    [Fact]
    public void MergeSelectorChain_PicksSelectorOutOfCommaList()
    {
        // フォーカス統一リングのように「1 ブロックに複数セレクタ」の形から拾えること
        const string css = "input:focus-visible, button:focus-visible { outline: none; }";

        var merged = MergeSelectorChain(
            new[] { "button:focus-visible" },
            Blocks(css),
            new Dictionary<string, string>(StringComparer.Ordinal));

        Assert.Equal("outline: none", Render(merged));
    }

    // razor の開始タグ切り出し。属性値に `>` (ラムダの =>) や入れ子の引用符
    // (@(x ? "on" : "")) が入るので、素朴な正規表現では途中で切れる。
    // ここが壊れると (C2) の全数検査が「1 件も見ていないのに緑」になる。
    [Fact]
    public void TagScanner_SurvivesLambdaArrowsAndNestedQuotes()
    {
        const string razor = """
            <div class="wrap">
                <button type="button" class="btn btn-ghost @(x == "a" ? "on" : "")"
                        @onclick="() => Delete(k, p)" disabled="@(!ok)">×</button>
                <a class="btn btn-ghost canvas-save" download="@name">保存</a>
            </div>
            """;

        var tags = ElementTags(razor);

        Assert.Equal(3, tags.Count);
        Assert.StartsWith("<div", tags[0]);
        // ラムダの `>` で切れていないこと (切れていれば disabled= が拾えない)
        Assert.Contains("disabled=\"@(!ok)\"", tags[1]);
        Assert.EndsWith(">", tags[1]);
        Assert.StartsWith("<a", tags[2]);

        // 入れ子の引用符をまたいで class 属性を最後まで読めていること
        Assert.Equal("btn btn-ghost @(x == \"a\" ? \"on\" : \"\")", ClassAttribute(tags[1]));
        Assert.Equal("btn btn-ghost canvas-save", ClassAttribute(tags[2]));
    }

    [Fact]
    public void ClassTokenizer_SeparatesStaticNamesFromExpressionLiterals()
    {
        var mixed = SplitClassTokens("btn btn-ghost @(x == \"a\" ? \"on\" : null)");
        Assert.Equal(new[] { "btn", "btn-ghost" }, mixed.Static);
        // 式の中の文字列リテラルは「動的に付きうるクラス名」として拾う
        Assert.Equal(new[] { "a", "on" }, mixed.Dynamic);

        // @Foo.Bar(x) 形式 (括弧の前に識別子が来る) でも静的トークンに混ざらないこと
        var call = SplitClassTokens("tm-item @DeviceStyle.CssClass(d.Device)");
        Assert.Equal(new[] { "tm-item" }, call.Static);
        Assert.Empty(call.Dynamic);

        // 空・空白のみは 0 トークン
        Assert.Empty(SplitClassTokens("   ").Static);
    }

    // 変異検査: バリアントを 1 つ落とした / 2 つ付けた razor を食わせると
    // (C2) の判定が本当に赤くなること (検査が空振りでないことの証明)。
    [Fact]
    public void BtnInvariant_DetectsMissingAndDoubledVariants()
    {
        static (bool HasBase, int Variants) Check(string tag)
        {
            var tokens = SplitClassTokens(ClassAttribute(tag) ?? "").Static;
            return (tokens.Contains("btn", StringComparer.Ordinal),
                    tokens.Count(t => BtnVariants.Contains(t, StringComparer.Ordinal)));
        }

        Assert.Equal((true, 1), Check("<button class=\"btn btn-ghost\">a</button>"));
        // バリアント落ち (素の button に落ちて UA 既定の見た目になる)
        Assert.Equal((true, 0), Check("<button class=\"btn\">a</button>"));
        // .btn 落ち (cursor / disabled / focus の作法が全部消える)
        Assert.Equal((false, 1), Check("<button class=\"btn-ghost\">a</button>"));
        // 二重 (後勝ちで不定)
        Assert.Equal((true, 2), Check("<button class=\"btn btn-ghost btn-icon\">a</button>"));
    }

    [Fact]
    public void SelectorPartsStartingWith_DoesNotBleedIntoNeighbouringNames()
    {
        const string css = """
            .btn { color: var(--text); }
            .btn:hover { color: var(--accent); }
            .btn-ghost { color: var(--muted); }
            .btnx { color: var(--ng); }
            .toolbar .btn { padding: 0; }
            """;

        // `.btn-ghost` / `.btnx` は別クラス。`.toolbar .btn` は接頭辞一致しない。
        Assert.Equal(
            new[] { ".btn", ".btn:hover" },
            SelectorPartsStartingWith(".btn", Blocks(css)));
    }

    // 合成 CSS から :root トークンを読む (本物の app.css を読む RootTokens とは別口)。
    private static Dictionary<string, string> SyntheticTokens(string css)
    {
        var tokens = new Dictionary<string, string>(StringComparer.Ordinal);
        var root = Blocks(css).Single(b => b.Selector == ":root");

        foreach (var d in ParseDeclarations(root.Body))
        {
            tokens[d.Property] = d.Value;
        }

        return tokens;
    }

    // ══════════════════════════════════════════════════
    //  以下ヘルパー (P1-6 / P3-1 後もそのまま使える形にしておく)
    // ══════════════════════════════════════════════════

    private sealed record ScaleEntry(string Selector, string Property, string Value);

    private sealed record ScaleDrift(string Selector, string Property, string From, string To, string Reason);

    private sealed record ButtonStateSpec(string Button, string State, string[] Chain, string[] Declarations);

    private sealed record ButtonDiff(string Button, string State, string Property, string Before, string After, string Reason);

    // app.css を毎回パースし直すと 100 ケース分で無駄なので 1 回で済ませる。
    // ※ 下の Lazy から参照するので、宣言順はここが先 (静的フィールドの初期化順)。
    private static readonly Lazy<List<CssBlock>> AppCssBlocks =
        new(() => Blocks(File.ReadAllText(AppCssPath)));

    private static readonly Lazy<List<ScaleEntry>> ScaleActual = new(ScanScaleDeclarations);

    private static readonly Lazy<List<ScaleEntry>> SpacingActual =
        new(() => ScanDeclarations(AppCssBlocks.Value, RootTokens(), IsSpacingProperty));

    // トークンを解決しない「生の値」。var(--sp-*) へ畳めたかどうかを見るのに使う。
    private static readonly Lazy<List<ScaleEntry>> SpacingRaw =
        new(() => ScanDeclarations(
            AppCssBlocks.Value,
            new Dictionary<string, string>(StringComparer.Ordinal),
            IsSpacingProperty));

    private static List<ScaleEntry> ScanScaleDeclarations()
        => ScanScaleDeclarations(AppCssBlocks.Value, RootTokens());

    // 合成 CSS でも呼べるよう blocks / tokens を引数に取る
    // (下の「自己検査」節が、この抽出器が本当に噛んでいることを検証する)。
    private static List<ScaleEntry> ScanScaleDeclarations(
        List<CssBlock> blocks,
        Dictionary<string, string> tokens)
        => ScanDeclarations(blocks, tokens, p => p is "font-size" or "border-radius");

    // 余白系プロパティ。ロングハンド (padding-right 等) と row-/column-gap も取りこぼさない。
    private static bool IsSpacingProperty(string property)
        => Regex.IsMatch(property, @"^(margin|padding)(-(top|right|bottom|left))?$")
           || Regex.IsMatch(property, @"^(row-|column-)?gap$");

    private static List<ScaleEntry> ScanDeclarations(
        List<CssBlock> blocks,
        Dictionary<string, string> tokens,
        Func<string, bool> wanted)
    {
        var list = new List<ScaleEntry>();

        foreach (var b in blocks)
        {
            if (b.Selector.StartsWith(":root", StringComparison.Ordinal))
            {
                continue;
            }

            foreach (var d in ParseDeclarations(b.Body))
            {
                if (wanted(d.Property))
                {
                    list.Add(new ScaleEntry(b.Selector, d.Property, ResolveMetricVars(d.Value, tokens)));
                }
            }
        }

        return list;
    }

    // 「セレクタ連鎖を後勝ちで合成する」= 同一クラス数を前提にした近似。
    // 完全なカスケード (詳細度・順序) は再現しない — 比較対象を 4 状態に限っているため。
    private static Dictionary<string, string> MergeSelectorChain(string[] chain)
        => MergeSelectorChain(chain, AppCssBlocks.Value, RootTokens());

    private static Dictionary<string, string> MergeSelectorChain(
        string[] chain,
        List<CssBlock> allBlocks,
        Dictionary<string, string> tokens)
    {
        var merged = new Dictionary<string, string>(StringComparer.Ordinal);

        foreach (var selector in chain)
        {
            var blocks = allBlocks
                .Where(b => SelectorParts(b.Selector).Contains(selector, StringComparer.Ordinal))
                .ToList();

            Assert.True(blocks.Count >= 1, $"セレクタ {selector} が app.css に無い");

            foreach (var b in blocks)
            {
                foreach (var d in ParseDeclarations(b.Body))
                {
                    merged[d.Property] = ResolveMetricVars(d.Value, tokens);
                }
            }
        }

        return merged;
    }

    // app.css 中の全セレクタ部品のうち、指定の接頭辞で始まるものを列挙する。
    // 接頭辞の直後は英数字・ハイフン・アンダースコアであってはならない
    // (`.preview-save` が `.preview-saved` を拾わないようにする)。
    private static List<string> SelectorPartsStartingWith(string prefix)
        => SelectorPartsStartingWith(prefix, AppCssBlocks.Value);

    private static List<string> SelectorPartsStartingWith(string prefix, List<CssBlock> blocks)
    {
        var found = new SortedSet<string>(StringComparer.Ordinal);

        foreach (var b in blocks)
        {
            foreach (var part in SelectorParts(b.Selector))
            {
                if (part.Length < prefix.Length || !part.StartsWith(prefix, StringComparison.Ordinal))
                {
                    continue;
                }

                if (part.Length > prefix.Length)
                {
                    var c = part[prefix.Length];
                    if (char.IsLetterOrDigit(c) || c == '-' || c == '_')
                    {
                        continue;
                    }
                }

                found.Add(part);
            }
        }

        return found.ToList();
    }

    private static string[] SelectorParts(string selector)
        => selector.Split(',').Select(Normalize).Where(s => s.Length > 0).ToArray();

    private readonly record struct CssDeclaration(string Property, string Value);

    // 宣言ブロックの中身を "prop: value" へ分解する (このプロジェクトの CSS に
    // セミコロンを含む値は無い)。
    private static List<CssDeclaration> ParseDeclarations(string body)
    {
        var list = new List<CssDeclaration>();

        foreach (var raw in body.Split(';'))
        {
            var s = Normalize(raw);
            var i = s.IndexOf(':');
            if (i <= 0)
            {
                continue;
            }

            list.Add(new CssDeclaration(Normalize(s[..i]), Normalize(s[(i + 1)..])));
        }

        return list;
    }

    // var(--x) のうち「純粋な寸法トークン」だけを実値へ展開する。
    // 色トークン (var(--accent) 等) は展開しない — 配色を触っただけで寸法・ボタンの表が
    // 壊れると偽陽性の山になるため。P1-6 の var(--fs-sm) / var(--r-sm) はここで解決される。
    private static string ResolveMetricVars(string value, Dictionary<string, string> tokens)
        => VarReferenceRegex().Replace(value, m =>
        {
            var resolved = ResolveTokenChain(tokens, m.Groups[1].Value, 0);
            return resolved is not null && IsMetric(resolved) ? resolved : m.Value;
        });

    private static Regex VarReferenceRegex()
        => new(@"var\(\s*(--[A-Za-z0-9_-]+)\s*(?:,[^()]*)?\)");

    // --a: var(--b) のような連鎖を辿って最終値を返す。未定義・循環は null。
    private static string? ResolveTokenChain(Dictionary<string, string> tokens, string name, int depth)
    {
        if (depth > 8 || !tokens.TryGetValue(name, out var value))
        {
            return null;
        }

        var m = Regex.Match(value.Trim(), @"^var\(\s*(--[A-Za-z0-9_-]+)\s*\)$");
        return m.Success ? ResolveTokenChain(tokens, m.Groups[1].Value, depth + 1) : value.Trim();
    }

    private static bool IsMetric(string value)
        => Regex.IsMatch(value.Trim(), @"^-?\d+(\.\d+)?(px|%|em|rem|vh|vw)?$");

    // 宣言集合を「1 行 1 宣言・プロパティ名順」で文字列化する。
    // Assert.Equal の差分がそのまま読めるようにするため。
    private static string Render(Dictionary<string, string> decls)
        => string.Join("\n", decls.OrderBy(kv => kv.Key, StringComparer.Ordinal).Select(kv => $"{kv.Key}: {kv.Value}"));

    private static string[] RazorFiles() => new[]
    {
        GenerateRazorPath,
        TagPresetFieldRazorPath,
        TagPaletteRazorPath,
        PresetSidebarRazorPath,
        ReconnectRazorPath,
    };

    // ui/ 配下の razor 全部。張り替え漏れの「全数」検査は列挙漏れがあると
    // 意味が無いので、手書きの一覧ではなくディレクトリから引く。
    private static string[] AllRazorFiles()
    {
        var files = Directory
            .GetFiles(Path.Combine(RepoRoot(), "ui"), "*.razor", SearchOption.AllDirectories)
            .OrderBy(x => x, StringComparer.Ordinal)
            .ToArray();

        // 既知の 5 つを含んでいること (glob が空振りしていないことの担保)
        foreach (var known in RazorFiles())
        {
            Assert.Contains(known, files);
        }

        return files;
    }

    // ── razor の開始タグ / class 属性の読み取り ─────────────────
    //
    // razor の属性値には
    //   ・ラムダの `=>` (タグ終端の `>` と紛らわしい)
    //   ・`@(x ? "on" : "")` のような入れ子の引用符
    // が入る。素朴な `<button[^>]*>` では途中で切れて検査が空振りするので、
    // 引用符と `@(…)` の括弧深さを見ながら 1 文字ずつ進む。

    private static List<string> ElementTags(string razor)
    {
        var tags = new List<string>();

        for (var i = 0; i < razor.Length; i++)
        {
            if (razor[i] != '<' || i + 1 >= razor.Length || !char.IsLetter(razor[i + 1]))
            {
                continue;
            }

            var end = ScanTag(razor, i);
            tags.Add(razor[i..end]);
            i = end - 1;
        }

        return tags;
    }

    // razor[start] の '<' から始まる開始タグの終端 (次の位置) を返す。
    private static int ScanTag(string razor, int start)
    {
        var quote = '\0';
        var paren = 0;

        for (var j = start + 1; j < razor.Length; j++)
        {
            var c = razor[j];

            if (quote != '\0')
            {
                if (paren > 0)
                {
                    // @( … ) の中。引用符は式の文字列リテラルなので属性の終端に数えない
                    if (c == '(')
                    {
                        paren++;
                    }
                    else if (c == ')')
                    {
                        paren--;
                    }
                }
                else if (c == '@' && j + 1 < razor.Length && razor[j + 1] == '(')
                {
                    paren = 1;
                    j++;
                }
                else if (c == quote)
                {
                    quote = '\0';
                }
            }
            else if (c is '"' or '\'')
            {
                quote = c;
            }
            else if (c == '>')
            {
                return j + 1;
            }
        }

        return razor.Length;
    }

    // 開始タグから class 属性の中身を返す (無ければ null)。
    private static string? ClassAttribute(string tag)
    {
        var m = Regex.Match(tag, @"(?<![-\w])class\s*=\s*(""|')");
        if (!m.Success)
        {
            return null;
        }

        var quote = m.Groups[1].Value[0];
        var i = m.Index + m.Length;
        var paren = 0;

        for (var j = i; j < tag.Length; j++)
        {
            var c = tag[j];

            if (paren > 0)
            {
                if (c == '(')
                {
                    paren++;
                }
                else if (c == ')')
                {
                    paren--;
                }
            }
            else if (c == '@' && j + 1 < tag.Length && tag[j + 1] == '(')
            {
                paren = 1;
                j++;
            }
            else if (c == quote)
            {
                return tag[i..j];
            }
        }

        return tag[i..];
    }

    // class 属性の中身をクラストークンへ分解する。
    //   Static  … 空白区切りでそのまま書かれている名前 (btn / btn-ghost …)
    //   Dynamic … @(…) / @Foo(…) の中の文字列リテラル (条件付きで付くクラス名)
    private static (List<string> Static, List<string> Dynamic) SplitClassTokens(string value)
    {
        var statics = new List<string>();
        var dynamics = new List<string>();

        for (var i = 0; i < value.Length; i++)
        {
            if (char.IsWhiteSpace(value[i]))
            {
                continue;
            }

            if (value[i] == '@')
            {
                var end = ScanExpression(value, i);
                foreach (Match lit in Regex.Matches(value[i..end], "\"([^\"]*)\""))
                {
                    var s = lit.Groups[1].Value.Trim();
                    if (s.Length > 0)
                    {
                        dynamics.Add(s);
                    }
                }

                i = end - 1;
                continue;
            }

            var j = i;
            while (j < value.Length && !char.IsWhiteSpace(value[j]))
            {
                j++;
            }

            statics.Add(value[i..j]);
            i = j - 1;
        }

        return (statics, dynamics);
    }

    // value[start] の '@' から始まる razor 式の終端 (次の位置) を返す。
    // `@(…)` は括弧の対応で、`@Foo.Bar(x)` は括弧を含めつつ空白まで。
    private static int ScanExpression(string value, int start)
    {
        var paren = 0;
        var seenParen = false;

        for (var j = start + 1; j < value.Length; j++)
        {
            var c = value[j];

            if (c == '(')
            {
                paren++;
                seenParen = true;
            }
            else if (c == ')')
            {
                paren--;
                if (paren == 0)
                {
                    return j + 1;
                }
            }
            else if (char.IsWhiteSpace(c) && paren == 0 && seenParen)
            {
                return j;
            }
            else if (char.IsWhiteSpace(c) && paren == 0 && !seenParen)
            {
                return j;
            }
        }

        return value.Length;
    }

    // razor 全体の class トークン (静的 + 式の中のリテラル) を平らに列挙する。
    private static IEnumerable<string> ClassTokens(string razor)
    {
        foreach (var tag in ElementTags(razor))
        {
            var value = ClassAttribute(tag);
            if (value is null)
            {
                continue;
            }

            var split = SplitClassTokens(value);
            foreach (var t in split.Static.Concat(split.Dynamic))
            {
                yield return t;
            }
        }
    }

    // 失敗メッセージ用にタグを 1 行へ潰す。
    private static string Squeeze(string tag)
    {
        var s = Normalize(tag);
        return s.Length <= 120 ? s : s[..117] + "…";
    }
}
