using System.Globalization;
using Dollama.Ui.Services;
using Xunit;

namespace Dollama.Ui.Tests;

// 生成画像の保存ファイル名 (docs/ui-brushup-plan.md §5 P2-3)。
//
// download 属性の値はブラウザがそのままファイル名にするので、
// 「パス区切り・OS の禁止文字が絶対に混ざらない」ことが本命の検査項目。
public sealed class DownloadNameTests
{
    private static readonly DateTime At = new(2026, 8, 8, 14, 25, 30, DateTimeKind.Local);

    // ── 通常系: 計画書の例どおりの形 ──────────────────
    [Fact]
    public void ForPng_BuildsPlannedShape()
    {
        Assert.Equal("dollama_20260808_142530_1024x1024.png", DownloadName.ForPng(At, "1024x1024"));
    }

    [Theory]
    [InlineData("512x512", "dollama_20260808_142530_512x512.png")]
    [InlineData("768x768", "dollama_20260808_142530_768x768.png")]
    [InlineData("832x1216", "dollama_20260808_142530_832x1216.png")]
    public void ForPng_UsesGivenSize(string size, string expected)
    {
        Assert.Equal(expected, DownloadName.ForPng(At, size));
    }

    // 大文字 X・前後や区切りの空白は正規化して受け入れる
    [Theory]
    [InlineData("1024X1024")]
    [InlineData(" 1024x1024 ")]
    [InlineData("1024 x 1024")]
    public void ForPng_NormalizesSizeSeparatorAndSpaces(string size)
    {
        Assert.Equal("dollama_20260808_142530_1024x1024.png", DownloadName.ForPng(At, size));
    }

    // ── 不正な size は unknown へフォールバック ──────────
    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("abc")]
    [InlineData("1024")]              // 高さが無い
    [InlineData("1024x")]
    [InlineData("x1024")]
    [InlineData("1024x1024x1024")]
    [InlineData("-1024x1024")]
    [InlineData("10.24x1024")]
    [InlineData("1000000x1000000")]   // 桁数が非常識
    public void ForPng_FallsBackToUnknownForUnparsableSize(string? size)
    {
        Assert.Equal("dollama_20260808_142530_unknown.png", DownloadName.ForPng(At, size));
    }

    // ── ★本命: パス区切り・禁止文字が混入しない ──────────
    [Theory]
    [InlineData("../../etc/passwd")]
    [InlineData("..\\..\\windows\\system32")]
    [InlineData("1024/1024")]
    [InlineData("1024\\1024")]
    [InlineData("1024x1024/../evil")]
    [InlineData("a:b")]
    [InlineData("<script>")]
    [InlineData("\"quoted\"")]
    [InlineData("nul")]
    [InlineData("1024x1024\n")]
    [InlineData("1024x1024\0")]
    [InlineData("こんにちは")]
    public void ForPng_NeverLeaksPathSeparatorsOrInvalidChars(string size)
    {
        var name = DownloadName.ForPng(At, size);

        Assert.DoesNotContain('/', name);
        Assert.DoesNotContain('\\', name);
        Assert.DoesNotContain("..", name);
        foreach (var c in Path.GetInvalidFileNameChars())
        {
            Assert.DoesNotContain(c, name);
        }

        // ファイル名以外の要素 (ディレクトリ) が生えていないこと
        Assert.Equal(name, Path.GetFileName(name));
    }

    // 何を渡しても英数字・アンダースコア・ドットしか出てこない (許可リストの担保)
    [Fact]
    public void ForPng_OutputIsAlwaysAsciiSafe()
    {
        string?[] sizes =
        {
            null, "", "1024x1024", "  768 X 768  ", "../x", "1024;rm -rf /", "%00", "🐈",
            "1024x1024\t", "'; DROP TABLE", "..", ".", "size=1024x1024",
        };

        foreach (var s in sizes)
        {
            var name = DownloadName.ForPng(At, s);
            Assert.Matches(@"^[A-Za-z0-9_.]+$", name);
            Assert.StartsWith("dollama_", name);
            Assert.EndsWith(".png", name);
        }
    }

    // ── 日時: 秒まで入り・ゼロ埋めされ・カルチャに依存しない ──
    [Fact]
    public void ForPng_ZeroPadsTimestamp()
    {
        var at = new DateTime(2026, 1, 2, 3, 4, 5, DateTimeKind.Local);
        Assert.Equal("dollama_20260102_030405_512x512.png", DownloadName.ForPng(at, "512x512"));
    }

    [Theory]
    [InlineData("ar-SA")]   // 既定カレンダーがヒジュラ暦
    [InlineData("th-TH")]   // 既定カレンダーが仏暦
    [InlineData("ja-JP")]
    [InlineData("de-DE")]
    public void ForPng_IsCultureInvariant(string culture)
    {
        var original = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = new CultureInfo(culture);
            Assert.Equal("dollama_20260808_142530_1024x1024.png", DownloadName.ForPng(At, "1024x1024"));
        }
        finally
        {
            CultureInfo.CurrentCulture = original;
        }
    }

    // 生成のたびに違う名前になる (上書き事故を避ける)
    [Fact]
    public void ForPng_DiffersBySecond()
    {
        var a = DownloadName.ForPng(At, "1024x1024");
        var b = DownloadName.ForPng(At.AddSeconds(1), "1024x1024");
        Assert.NotEqual(a, b);
    }

    // 例外を投げない (razor の描画中に呼ぶので落とせない)
    [Fact]
    public void ForPng_NeverThrows()
    {
        var ex = Record.Exception(() =>
        {
            DownloadName.ForPng(DateTime.MinValue, null);
            DownloadName.ForPng(DateTime.MaxValue, new string('9', 10_000));
        });
        Assert.Null(ex);
    }
}
