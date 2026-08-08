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
//   (A) 全 font-size / border-radius の「セレクタ → 実効値」表 (67 件)
//   (B) ボタン系 8 群 × 4 状態 (base / :hover / :disabled / :focus-visible) の宣言集合
//   (C) 旧クラス名が「今は存在する」こと (P3-1 後に 0 件へ反転させる)
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

    // 現行 app.css の全宣言 (font-size 36 + border-radius 31 = 67)。
    // 値は var() 解決後の実効値で書く (現状は全て直書きなので px / % がそのまま並ぶ)。
    // P1-6 で var(--fs-sm) / var(--r-sm) へ置換しても、解決結果がこの表と一致すれば緑。
    private static readonly ScaleEntry[] ScaleBaseline =
    {
        // ── ベース / トップバー ──
        new("html, body",                     "font-size",     "14px"),
        new(".brand",                         "font-size",     "16px"),
        new(".brand-sub",                     "font-size",     "12px"),
        new(".lang-toggle",                   "border-radius", "6px"),
        new(".lang-toggle button",            "font-size",     "12px"),
        new(".conn",                          "font-size",     "12px"),
        new(".conn .dot",                     "border-radius", "50%"),
        new(".conn-retry",                    "border-radius", "6px"),
        new(".conn-retry",                    "font-size",     "11px"),

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
        new(".generate",                      "border-radius", "8px"),
        new(".generate",                      "font-size",     "15px"),   // §4.2 で 14 or 16 への寄せを許容
        new(".generate.secondary",            "font-size",     "14px"),
        new(".gen-reason",                    "font-size",     "12px"),
        new(".error",                         "border-radius", "6px"),
        new(".error",                         "font-size",     "13px"),   // §4.2 で 12 or 14 への寄せを許容

        // ── プレビュー (右ペイン) ──
        new(".gen-mode",                      "border-radius", "999px"),
        new(".gen-mode",                      "font-size",     "11px"),
        new(".preview-save",                  "border-radius", "6px"),
        new(".preview-save",                  "font-size",     "11px"),
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
        new(".fav-plus",                      "border-radius", "6px"),

        // ── プリセット一覧 (左ペイン) ──
        new(".ps-card",                       "border-radius", "6px"),
        new(".ps-card img",                   "border-radius", "4px"),
        new(".ps-noimg",                      "border-radius", "4px"),
        new(".ps-noimg",                      "font-size",     "9px"),    // スケール外の最小値 (要棚卸し)
        new(".ps-name",                       "font-size",     "12px"),
        new(".ps-del",                        "border-radius", "4px"),
        new(".ps-del",                        "font-size",     "13px"),

        // ── プリセット保存バー (中央) ──
        new(".preset-name",                   "border-radius", "6px"),
        new(".preset-btn",                    "border-radius", "6px"),
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
    // P1-6 実装時に、寄せた箇所だけをここへ理由付きで積む。**それ以外の値変更は通さない**。
    private static readonly ScaleDrift[] AllowedScaleDrifts =
    {
        // 例) new(".error", "font-size", "13px", "14px", "§4.2: 13px は最寄りへ寄せる"),
    };

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
    [Fact]
    public void ScaleBaseline_CoversEveryDeclarationInAppCss()
    {
        var actual = ScaleActual.Value;

        Assert.Equal(36, actual.Count(x => x.Property == "font-size"));
        Assert.Equal(31, actual.Count(x => x.Property == "border-radius"));
        Assert.Equal(67, actual.Count);
        Assert.Equal(67, ScaleBaseline.Length);

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

    // 許可リストは今は空。P1-6 実装時にここを「寄せた件数」へ書き換える (= 反転させる検査)。
    [Fact]
    public void ScaleDriftAllowList_IsEmptyBeforeP1_6()
    {
        Assert.Empty(AllowedScaleDrifts);
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

    // P1-6 の着手前であることの確認 (= P1-6 後に反転させる検査)。
    [Fact]
    public void ScaleTokens_DoNotExistYet()
    {
        var tokens = RootTokens();
        var scale = tokens.Keys
            .Where(k => k.StartsWith("--fs-") || k.StartsWith("--r-") || k.StartsWith("--sp-"))
            .ToList();

        Assert.True(
            scale.Count == 0,
            "P1-6 は未着手のはずだが :root にスケールトークンがある: " + string.Join(", ", scale));
    }

    // ══════════════════════════════════════════════════
    //  (B) P3-1 用: ボタン系の 4 状態ベースライン
    // ══════════════════════════════════════════════════

    // ボタン 8 群 × 4 状態。
    // Chain = その状態を作っている app.css のセレクタを**カスケード順**に並べたもの
    //         (後ろが後勝ち)。P3-1 後はここを `.btn` + バリアントへ差し替えるだけで、
    //         Declarations (期待値) は据え置きのまま緑になるはず、というのが本検査の趣旨。
    // Declarations = その状態の最終宣言集合 ("prop: value")。順序は問わない (ソートして比較)。
    private static readonly ButtonStateSpec[] ButtonBaseline =
    {
        // ── 1. 生成 (primary) ─────────────────────────
        new("generate(primary)", "base", new[] { ".generate" }, new[]
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
        // primary には hover 規則が無い (accent 塗りのまま変化しない) — P3-1 で .btn:hover が
        // 付くと**見た目が変わる**ので、そのときは意図的差分として許可リストへ積むこと。
        new("generate(primary)", "hover", Array.Empty<string>(), Array.Empty<string>()),
        new("generate(primary)", "disabled", new[] { ".generate:disabled" }, new[]
        {
            "opacity: 0.6",
            "cursor: default",
        }),
        new("generate(primary)", "focus-visible", new[] { "button:focus-visible" }, FocusRingDecls),

        // ── 2. 下書き (secondary) ─────────────────────
        // .generate (1 クラス) の上に .generate.secondary (2 クラス) が乗る = 後勝ち。
        new("generate.secondary", "base", new[] { ".generate", ".generate.secondary" }, new[]
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
        new("generate.secondary", "hover", new[] { ".generate.secondary:hover:not(:disabled)" }, new[]
        {
            "border-color: var(--accent)",
            "color: var(--accent)",
        }),
        new("generate.secondary", "disabled", new[] { ".generate:disabled" }, new[]
        {
            "opacity: 0.6",
            "cursor: default",
        }),
        new("generate.secondary", "focus-visible", new[] { "button:focus-visible" }, FocusRingDecls),

        // ── 3. プリセット保存バーのボタン ─────────────
        new(".preset-btn", "base", new[] { ".preset-btn" }, new[]
        {
            "background: var(--panel-2)",
            "color: var(--text)",
            "border: 1px solid var(--border-strong)",
            "border-radius: 6px",
            "padding: 6px 10px",
            "font: inherit",
            "cursor: pointer",
        }),
        new(".preset-btn", "hover", new[] { ".preset-btn:hover:not(:disabled)" }, new[]
        {
            "border-color: var(--accent)",
        }),
        new(".preset-btn", "disabled", new[] { ".preset-btn:disabled" }, new[]
        {
            "opacity: 0.45",
            "cursor: default",
        }),
        new(".preset-btn", "focus-visible", new[] { "button:focus-visible" }, FocusRingDecls),

        // ── 4. お気に入り追加 (+) ─────────────────────
        // .preset-btn とは padding だけが違う (0 10px vs 6px 10px)。P3-1 の統合対象そのもの。
        new(".fav-plus", "base", new[] { ".fav-plus" }, new[]
        {
            "background: var(--panel-2)",
            "color: var(--text)",
            "border: 1px solid var(--border-strong)",
            "border-radius: 6px",
            "padding: 0 10px",
            "font: inherit",
            "cursor: pointer",
        }),
        new(".fav-plus", "hover", new[] { ".fav-plus:hover:not(:disabled)" }, new[]
        {
            "border-color: var(--accent)",
        }),
        new(".fav-plus", "disabled", new[] { ".fav-plus:disabled" }, new[]
        {
            "opacity: 0.45",
            "cursor: default",
        }),
        new(".fav-plus", "focus-visible", new[] { "button:focus-visible" }, FocusRingDecls),

        // ── 5. 言語トグル (セグメント) ────────────────
        new(".lang-toggle button", "base", new[] { ".lang-toggle button" }, new[]
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
        // hover 作法がここだけ「文字色替え」(他は枠色替え)。P3-1 の統一で変わるなら許可リストへ。
        new(".lang-toggle button", "hover", new[] { ".lang-toggle button:hover" }, new[]
        {
            "color: var(--text)",
        }),
        new(".lang-toggle button", "disabled", Array.Empty<string>(), Array.Empty<string>()),
        new(".lang-toggle button", "focus-visible", new[] { "button:focus-visible" }, FocusRingDecls),

        // ── 6. プリセット削除 (アイコンボタン) ────────
        new(".ps-del", "base", new[] { ".ps-del" }, new[]
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
        // 削除だけ hover が ng 系 (赤) — 破壊操作なので P3-1 でも残す想定。
        new(".ps-del", "hover", new[] { ".ps-del:hover" }, new[]
        {
            "border-color: var(--ng)",
            "color: var(--ng-soft)",
        }),
        new(".ps-del", "disabled", Array.Empty<string>(), Array.Empty<string>()),
        new(".ps-del", "focus-visible", new[] { "button:focus-visible" }, FocusRingDecls),

        // ── 7. 画像保存 (button ではなく a[download]) ──
        new(".preview-save", "base", new[] { ".preview-save" }, new[]
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
        // hover が :not(:disabled) を持たない (a には disabled が無いため)。
        new(".preview-save", "hover", new[] { ".preview-save:hover" }, new[]
        {
            "border-color: var(--accent)",
            "color: var(--accent)",
        }),
        new(".preview-save", "disabled", Array.Empty<string>(), Array.Empty<string>()),
        // a 要素なので button:focus-visible には当たらない。リング統一ブロックへ個別に
        // .preview-save:focus-visible が足してある (P2-3)。
        new(".preview-save", "focus-visible", new[] { ".preview-save:focus-visible" }, FocusRingDecls),

        // ── 8. 再接続 (トップバーの極小ボタン) ────────
        new(".conn-retry", "base", new[] { ".conn-retry" }, new[]
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
        new(".conn-retry", "hover", new[] { ".conn-retry:hover:not(:disabled)" }, new[]
        {
            "border-color: var(--accent)",
            "color: var(--accent)",
        }),
        // disabled の opacity がここだけ 0.6 (preset-btn / fav-plus は 0.45)。統一するなら許可リストへ。
        new(".conn-retry", "disabled", new[] { ".conn-retry:disabled" }, new[]
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
    // 例: hover 作法を枠色替えへ統一する / disabled の opacity を 1 つに揃える など。
    private static readonly ButtonDiff[] AllowedButtonDiffs =
    {
        // 例) new(".lang-toggle button", "hover", "color", "var(--text)", "",
        //         "hover 作法を枠色替えへ統一 (課題 #15)"),
    };

    private const int MaxAllowedButtonDiffs = 6;

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

    // 許可リストは今は空。P3-1 実装時にここを実件数へ書き換える (= 反転させる検査)。
    [Fact]
    public void ButtonDiffAllowList_IsEmptyBeforeP3_1()
    {
        Assert.Empty(AllowedButtonDiffs);
        Assert.True(
            AllowedButtonDiffs.Length <= MaxAllowedButtonDiffs,
            $"意図的差分は {MaxAllowedButtonDiffs} 件まで (PL 裁定)。実際 {AllowedButtonDiffs.Length} 件");

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

    // 「そのボタンに今どんな規則が存在するか」の在庫。
    // 状態表 (B) は宣言の中身を見るが、こちらは**規則の有無そのもの**を凍結する。
    // 例えば .ps-del に :disabled が無いことは現行仕様であり、P3-1 の .btn:disabled が
    // 効くようになるなら意図的差分として扱う必要がある。
    public static IEnumerable<object[]> LegacyButtonFamilyData() => new[]
    {
        new object[] { ".generate", new[]
        {
            ".generate",
            ".generate.secondary",
            ".generate.secondary:hover:not(:disabled)",
            ".generate:disabled",
        }},
        new object[] { ".preset-btn", new[]
        {
            ".preset-btn",
            ".preset-btn:disabled",
            ".preset-btn:hover:not(:disabled)",
        }},
        new object[] { ".fav-plus", new[]
        {
            ".fav-plus",
            ".fav-plus:disabled",
            ".fav-plus:hover:not(:disabled)",
        }},
        new object[] { ".lang-toggle button", new[]
        {
            ".lang-toggle button",
            ".lang-toggle button + button",
            ".lang-toggle button.on",
            ".lang-toggle button:hover",
        }},
        new object[] { ".ps-del", new[]
        {
            ".ps-del",
            ".ps-del:hover",
        }},
        new object[] { ".preview-save", new[]
        {
            ".preview-save",
            ".preview-save:focus-visible",
            ".preview-save:hover",
        }},
        new object[] { ".conn-retry", new[]
        {
            ".conn-retry",
            ".conn-retry:disabled",
            ".conn-retry:hover:not(:disabled)",
        }},
    };

    [Theory]
    [MemberData(nameof(LegacyButtonFamilyData))]
    public void LegacyButtonFamily_OwnsExactlyTheseSelectorsBeforeP3_1(string prefix, string[] expected)
    {
        var actual = SelectorPartsStartingWith(prefix);

        Assert.Equal(
            string.Join("\n", expected.OrderBy(x => x, StringComparer.Ordinal)),
            string.Join("\n", actual));
    }

    // 新クラスはまだ無い (P3-1 で追加され、この検査は「存在すること」へ反転する)。
    [Fact]
    public void BtnClasses_DoNotExistYet()
    {
        var css = StripComments(File.ReadAllText(AppCssPath));
        foreach (var cls in new[] { ".btn", ".btn-primary", ".btn-ghost", ".btn-icon" })
        {
            Assert.DoesNotContain(cls, css);
        }

        foreach (var path in RazorFiles())
        {
            Assert.DoesNotContain("class=\"btn", File.ReadAllText(path));
        }
    }

    // razor 側の旧クラス名。P3-1 では `class="btn btn-ghost"` 等へ張り替えるので、
    // ここが「今は 1 箇所ずつ存在する」ことを凍結しておく。
    public static IEnumerable<object[]> LegacyRazorClassData() => new[]
    {
        new object[] { "Generate.razor",        "class=\"generate\"" },
        new object[] { "Generate.razor",        "class=\"generate secondary\"" },
        new object[] { "Generate.razor",        "class=\"conn-retry\"" },
        new object[] { "Generate.razor",        "class=\"preview-save\"" },
        new object[] { "Generate.razor",        "class=\"lang-toggle\"" },
        new object[] { "TagPresetField.razor",  "class=\"preset-btn\"" },
        new object[] { "TagPalette.razor",      "class=\"fav-plus\"" },
        new object[] { "PresetSidebar.razor",   "class=\"ps-del\"" },
    };

    [Theory]
    [MemberData(nameof(LegacyRazorClassData))]
    public void LegacyButtonClass_AppearsExactlyOnceInRazorBeforeP3_1(string file, string literal)
    {
        var path = RazorFiles().Single(p => Path.GetFileName(p) == file);
        var text = File.ReadAllText(path);
        var count = Regex.Matches(text, Regex.Escape(literal)).Count;

        Assert.True(count == 1, $"{file} の {literal} は 1 箇所であること (実際 {count} 箇所)");
    }

    // 言語トグルの button は class を持たず、子孫セレクタだけで装飾されている。
    // P3-1 で .btn 化するならここに class が生えるので、現状を明示的に固定しておく。
    [Fact]
    public void LangToggleButtons_HaveNoStaticClassBeforeP3_1()
    {
        var razor = File.ReadAllText(GenerateRazorPath);

        Assert.Contains("class=\"@(_tagLang == \"ja\" ? \"on\" : \"\")\"", razor);
        Assert.Contains("class=\"@(_tagLang == \"en\" ? \"on\" : \"\")\"", razor);
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
    private static readonly Lazy<List<ScaleEntry>> ScaleActual = new(ScanScaleDeclarations);

    private static readonly Lazy<List<CssBlock>> AppCssBlocks =
        new(() => Blocks(File.ReadAllText(AppCssPath)));

    private static List<ScaleEntry> ScanScaleDeclarations()
        => ScanScaleDeclarations(AppCssBlocks.Value, RootTokens());

    // 合成 CSS でも呼べるよう blocks / tokens を引数に取る
    // (下の「自己検査」節が、この抽出器が本当に噛んでいることを検証する)。
    private static List<ScaleEntry> ScanScaleDeclarations(
        List<CssBlock> blocks,
        Dictionary<string, string> tokens)
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
                if (d.Property is "font-size" or "border-radius")
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
}
