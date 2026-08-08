using System.Globalization;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using Dollama.Ui.Services;
using Xunit;

namespace Dollama.Ui.Tests;

// PreviewLabel (プレビュー右上のモードバッジ文言) の検証。
// docs/ui-brushup-plan.md §5 P3-5 の前段 = Generate.razor の private static ModeBadge と、
// その呼び出し側の文字列補間を純クラスへ昇格させたリファクタ。
//
// ★ 手順が急所 (P3 バッチ D と同じ手): 下の期待値表は **移設前の ModeBadge を反射で呼んで
//   採取した実測値**で、移設前に緑であることを確認してから PreviewLabel へ移した。
//   後から書くと「移した実装に合わせた期待値」になり、リファクタの検査価値が消える。
//   移設前の一時テスト (LegacyModeBadge_* / LegacyComposition_*) は移設で用済みになったので、
//   代わりに「旧実装が残っていないこと」「razor が新クラスを呼んでいること」を常設で見る。
public sealed class PreviewLabelTests
{
    // ────────────────────────────────────────────────
    // 期待値表 (移設前の Generate.ModeBadge から採取した実測値)
    //   size → バッジ。読めなければ size をそのまま返すのが旧実装の挙動で、移設後も同じ。
    // ────────────────────────────────────────────────
    private static readonly (string Size, string Badge)[] BadgeRows =
    {
        ("1024x1024", "1024²"),        // 既定 (本番)
        ("768x768", "768²"),           // 下書き解像度
        ("768X768", "768²"),           // 大文字 X 区切り
        ("512x512", "512²"),           // 小さい方
        (" 1024 x 1024 ", "1024²"),    // 空白入り
        ("1024x768", "1024²"),         // 非正方でも幅だけ見る
        ("1024", "1024²"),             // 区切りなし (幅だけ)
        ("0x0", "0²"),                 // 0 も数字として通る
        ("abc", "abc"),                // パース不能 → そのまま
        ("abcxdef", "abcxdef"),        // パース不能 (区切りあり)
        ("x1024", "x1024"),            // 幅が空 → そのまま
        ("1024px", "1024px"),          // 数字の後ろに文字 → そのまま
        ("99999999999x1", "99999999999x1"), // int 範囲外 → そのまま
    };

    public static TheoryData<string, string> BadgeTable
    {
        get
        {
            var data = new TheoryData<string, string>();
            foreach (var (size, badge) in BadgeRows)
            {
                data.Add(size, badge);
            }
            return data;
        }
    }

    // 退化入力: 旧実装は空バッジ (合成すると "本番 " と末尾スペース) か、null なら例外だった。
    // ★ ここだけは移設で**意図的に変えた** (安全側へ堅牢化した) 唯一の差分。
    //   呼び出し側の size は固定 select 由来なので実運用では到達しない防御域。
    private static readonly string?[] DegenerateRows = { null, "", "   ", "\t" };

    public static TheoryData<string?> DegenerateSizes
    {
        get
        {
            var data = new TheoryData<string?>();
            foreach (var size in DegenerateRows)
            {
                data.Add(size);
            }
            return data;
        }
    }

    // ────────────────────────────────────────────────
    // (1) 読める size: 移設前の実測表と 1 文字も変わっていないこと
    //     (draft × size の全組合せ = 表 13 行 × 2)
    // ────────────────────────────────────────────────
    [Theory]
    [MemberData(nameof(BadgeTable))]
    public void Build_MatchesPreMoveBaseline(string size, string badge)
    {
        Assert.Equal($"下書き {badge}", PreviewLabel.Build(true, size));
        Assert.Equal($"本番 {badge}", PreviewLabel.Build(false, size));
    }

    // ────────────────────────────────────────────────
    // (2) 退化入力: 例外を投げず、モード名だけを返す (末尾スペースを残さない)
    // ────────────────────────────────────────────────
    [Theory]
    [MemberData(nameof(DegenerateSizes))]
    public void Build_DegenerateSizeYieldsModeNameOnly(string? size)
    {
        Assert.Equal("下書き", PreviewLabel.Build(true, size));
        Assert.Equal("本番", PreviewLabel.Build(false, size));
    }

    // 読めない size に前後空白が付いていたら落とす。
    // 旧実装は素通しだったので "本番   abc  " のように空白がそのまま出ていた。
    // これも退化域なので移設で直してある (読める size = 表の 13 行には影響しない)。
    [Theory]
    [InlineData("  abc  ", "abc")]
    [InlineData("\tabc\t", "abc")]
    [InlineData(" 12ab x 34 ", "12ab x 34")]
    public void Build_TrimsAnUnreadableSize(string size, string expected)
    {
        Assert.Equal($"本番 {expected}", PreviewLabel.Build(false, size));
        Assert.Equal($"下書き {expected}", PreviewLabel.Build(true, size));
    }

    // 旧実装は null で NullReferenceException (size.Split) を投げていた。移設で潰した穴。
    [Fact]
    public void Build_DoesNotThrowOnNull()
    {
        var ex = Record.Exception(() => PreviewLabel.Build(false, null));
        Assert.Null(ex);
    }

    // ────────────────────────────────────────────────
    // (3) 退化検知: どんな size でも下書きと本番が必ず区別できること
    //     (バッジだけ出して前置きを落とすと「速い絵か本番の絵か」が見分けられなくなる)
    // ────────────────────────────────────────────────
    [Fact]
    public void Build_DraftAndProductionAreAlwaysDistinguishable()
    {
        foreach (var size in AllSizes())
        {
            var d = PreviewLabel.Build(true, size);
            var p = PreviewLabel.Build(false, size);

            Assert.NotEqual(d, p);
            // 語は実装の定数を借りず literal で書く (定数を書き換える変異も検出するため)
            Assert.StartsWith("下書き", d, StringComparison.Ordinal);
            Assert.StartsWith("本番", p, StringComparison.Ordinal);
            Assert.False(string.IsNullOrWhiteSpace(d));
            Assert.False(string.IsNullOrWhiteSpace(p));
        }
    }

    // 公開 API は Build 1 本だけ。モード名の定数を public にすると
    // 「語だけ借りて自前で組み立てる」呼び出し側が生まれ、唯一の合流点でなくなる。
    [Fact]
    public void PublicSurface_IsExactlyBuild()
    {
        var members = typeof(PreviewLabel)
            .GetMembers(BindingFlags.Public | BindingFlags.Static | BindingFlags.Instance | BindingFlags.DeclaredOnly)
            .Select(m => $"{m.MemberType}:{m.Name}")
            .OrderBy(s => s, StringComparer.Ordinal)
            .ToArray();

        Assert.Equal(new[] { "Method:Build" }, members);
    }

    // ────────────────────────────────────────────────
    // (4) カルチャ非依存: 数字の整形も判定もロケールでぶれないこと
    //     (P2-4 で LoRA スライダのカルチャ依存 parse を踏んだ前例がある)
    // ────────────────────────────────────────────────
    [Theory]
    [InlineData("ar-SA")]   // 既定カレンダー = ヒジュラ暦・アラビア数字圏
    [InlineData("th-TH")]   // 既定カレンダー = 仏暦
    [InlineData("de-DE")]   // 小数点/桁区切りが逆
    [InlineData("sv-SE")]   // 負符号が U+2212
    [InlineData("ja-JP")]
    [InlineData("")]        // Invariant
    public void Build_IsCultureInvariant(string culture)
    {
        var original = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = new CultureInfo(culture);

            Assert.Equal("本番 1024²", PreviewLabel.Build(false, "1024x1024"));
            Assert.Equal("下書き 768²", PreviewLabel.Build(true, "768x768"));
            // 桁区切り付きは数値として読めない = そのまま出す (どのロケールでも同じ)
            Assert.Equal("本番 1,024x1,024", PreviewLabel.Build(false, "1,024x1,024"));
            // ★ 負値は実運用では来ないが、**カルチャ差が観測できる唯一の入り口**。
            //   正の整数は .NET が桁を置換しないのでどのロケールでも同じ字面になり、
            //   ここを負値で見ないと「CurrentCulture で整形する」変異を検出できない
            //   (sv-SE の負符号は U+2212 = ASCII の '-' ではない)。
            Assert.Equal("本番 -1024²", PreviewLabel.Build(false, "-1024x-1024"));
        }
        finally
        {
            CultureInfo.CurrentCulture = original;
        }
    }

    // ────────────────────────────────────────────────
    // (5) 例外を投げない / 決定的であること (描画中に呼ぶので落とせない)
    // ────────────────────────────────────────────────
    [Fact]
    public void Build_NeverThrowsAndIsDeterministic()
    {
        string?[] hostile =
        {
            null, "", " ", "\t", "\n", "x", "xx", "XxX", "－", "1024²", "1024x1024x1024",
            "-1x-1", "2147483648x1", new string('9', 400), "1024\0x1024", "🙂x🙂",
        };

        foreach (var size in hostile)
        {
            foreach (var draft in new[] { true, false })
            {
                var first = PreviewLabel.Build(draft, size);
                Assert.NotNull(first);
                Assert.False(string.IsNullOrWhiteSpace(first));
                // 同じ入力なら常に同じ出力
                Assert.Equal(first, PreviewLabel.Build(draft, size));
            }
        }
    }

    // ────────────────────────────────────────────────
    // (6) 移設の後始末: 旧実装が razor 側に残っていない / razor が新クラスを呼んでいる
    //     ここが赤いと「二重実装」または「純クラスを作っただけで未結線」を意味する。
    // ────────────────────────────────────────────────
    [Fact]
    public void Generate_NoLongerDeclaresItsOwnModeBadge()
    {
        var type = Array.Find(
            typeof(PreviewLabel).Assembly.GetTypes(),
            t => t.Name == "Generate" && t.Namespace == "Dollama.Ui.Components.Pages");

        Assert.True(type is not null, "Dollama.Ui.Components.Pages.Generate が見つからない");
        Assert.Null(type!.GetMethod("ModeBadge", BindingFlags.NonPublic | BindingFlags.Static));

        var src = File.ReadAllText(GenerateRazorPath);
        Assert.DoesNotContain("ModeBadge", src);
    }

    // ★ P3-5 (ミニ履歴) で結線の形が変わった。
    //   旧: `_lastMode = PreviewLabel.Build(draft, sendSize);` を直接代入 (この literal を検査していた)
    //   新: 履歴 1 件 (`PreviewItem`) の Badge として組み立て、funnel (`ShowResult`) が
    //       `_lastMode = item.Badge;` を代入する。本流と履歴のどちらから表示しても
    //       **同じ文言・同じ組み立て**になるのが P3-5 の要件だったため。
    //   検査は弱めず**強めて**引き継ぐ: 呼び出しが razor 全体で**ちょうど 1 箇所**
    //   (= 唯一の合流点であることの直接固定) を足した。「4 フィールドが 1 箇所でしか
    //   代入されない」ことは ui.Tests/PreviewHistoryRazorTests の F-1 が別途縛る。
    [Fact]
    public void Generate_BuildsTheModeLabelThroughPreviewLabel()
    {
        var src = File.ReadAllText(GenerateRazorPath);

        Assert.Contains("PreviewLabel.Build(draft, sendSize)", src);
        Assert.Single(Regex.Matches(src, Regex.Escape("PreviewLabel.Build(")));
    }

    // ────────────────────────────────────────────────
    // ヘルパ
    // ────────────────────────────────────────────────
    private static IEnumerable<string?> AllSizes()
    {
        foreach (var (size, _) in BadgeRows)
        {
            yield return size;
        }
        foreach (var size in DegenerateRows)
        {
            yield return size;
        }
    }

    private static string GenerateRazorPath
        => Path.Combine(RepoRoot(), "ui", "Components", "Pages", "Generate.razor");

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

        Assert.Fail("リポジトリルートを特定できない (ui/wwwroot/app.css が見つからない)");
        return "";
    }

    private static string SourceDir([CallerFilePath] string thisFile = "")
        => Path.GetDirectoryName(thisFile) ?? "";
}
