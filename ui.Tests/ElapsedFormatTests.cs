using System.Globalization;
using Dollama.Ui.Services;
using Xunit;

namespace Dollama.Ui.Tests;

// 生成中オーバーレイの経過秒表示 (docs/ui-brushup-plan.md §5 P2-2)。
//
// タイマー本体 (PeriodicTimer + StateHasChanged) は razor 側にあり自動テストの対象外。
// テストするのは切り出した純ロジック「TimeSpan → mm:ss」だけ。
public sealed class ElapsedFormatTests
{
    // ── 通常系: 0 秒から分をまたぐまで ──────────────────
    [Theory]
    [InlineData(0, "00:00")]
    [InlineData(1, "00:01")]
    [InlineData(9, "00:09")]
    [InlineData(12, "00:12")]   // 計画書の例 (§4.3)
    [InlineData(59, "00:59")]
    [InlineData(60, "01:00")]
    [InlineData(61, "01:01")]
    [InlineData(119, "01:59")]
    [InlineData(600, "10:00")]
    [InlineData(3599, "59:59")]
    public void Mmss_FormatsSecondsAsTwoDigitPairs(int seconds, string expected)
    {
        Assert.Equal(expected, ElapsedFormat.Mmss(TimeSpan.FromSeconds(seconds)));
    }

    // ── 60 分を超えても mm は繰り上がらず分として伸びる ──
    [Theory]
    [InlineData(3600, "60:00")]
    [InlineData(3661, "61:01")]
    [InlineData(5999, "99:59")]
    public void Mmss_KeepsCountingMinutesPastOneHour(int seconds, string expected)
    {
        Assert.Equal(expected, ElapsedFormat.Mmss(TimeSpan.FromSeconds(seconds)));
    }

    // ── 上限: 2 桁の mm に収まらない長さは 99:59 で頭打ち ──
    [Theory]
    [InlineData(6000)]      // ちょうど 100 分
    [InlineData(6001)]
    [InlineData(100000)]
    public void Mmss_SaturatesAtMax(int seconds)
    {
        Assert.Equal("99:59", ElapsedFormat.Mmss(TimeSpan.FromSeconds(seconds)));
    }

    [Fact]
    public void Mmss_SaturatesAtMax_ForTimeSpanMaxValue()
    {
        // long へキャストする前に頭打ちするのでオーバーフローしない
        Assert.Equal("99:59", ElapsedFormat.Mmss(TimeSpan.MaxValue));
    }

    // ── 下限: 負値は 00:00 (時計のずれで負になっても崩さない) ──
    [Theory]
    [InlineData(-1)]
    [InlineData(-60)]
    [InlineData(-100000)]
    public void Mmss_ClampsNegativeToZero(int seconds)
    {
        Assert.Equal("00:00", ElapsedFormat.Mmss(TimeSpan.FromSeconds(seconds)));
    }

    [Fact]
    public void Mmss_ClampsNegativeToZero_ForTimeSpanMinValue()
    {
        Assert.Equal("00:00", ElapsedFormat.Mmss(TimeSpan.MinValue));
    }

    // ── 端数は切り捨て (カウントアップ表示なので繰り上げない) ──
    [Theory]
    [InlineData(0.9, "00:00")]
    [InlineData(12.9, "00:12")]
    [InlineData(59.999, "00:59")]
    public void Mmss_TruncatesFraction(double seconds, string expected)
    {
        Assert.Equal(expected, ElapsedFormat.Mmss(TimeSpan.FromSeconds(seconds)));
    }

    // ── 形式の不変条件: 常に mm:ss の 5 文字・数字とコロンのみ ──
    [Fact]
    public void Mmss_AlwaysReturnsFiveCharsOfDigitsAndColon()
    {
        foreach (var s in new[] { -5, 0, 1, 59, 60, 3599, 3600, 5999, 6000, 999999 })
        {
            var text = ElapsedFormat.Mmss(TimeSpan.FromSeconds(s));
            Assert.Equal(5, text.Length);
            Assert.Equal(':', text[2]);
            Assert.All(new[] { text[0], text[1], text[3], text[4] }, c => Assert.True(char.IsDigit(c)));
        }
    }

    // ── 秒の桁は 0-59 に閉じる (60 秒表示が出ない) ──
    [Fact]
    public void Mmss_SecondsFieldNeverReachesSixty()
    {
        for (var s = 0; s < 4000; s++)
        {
            var ss = int.Parse(ElapsedFormat.Mmss(TimeSpan.FromSeconds(s))[3..], CultureInfo.InvariantCulture);
            Assert.InRange(ss, 0, 59);
        }
    }

    // ── カルチャ非依存 (アラビア数字・コロン区切りが変わらない) ──
    [Theory]
    [InlineData("ar-SA")]
    [InlineData("fa-IR")]
    [InlineData("de-DE")]
    [InlineData("ja-JP")]
    public void Mmss_IsCultureInvariant(string culture)
    {
        var original = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = new CultureInfo(culture);
            Assert.Equal("01:05", ElapsedFormat.Mmss(TimeSpan.FromSeconds(65)));
        }
        finally
        {
            CultureInfo.CurrentCulture = original;
        }
    }

    // ── 単調非減少 (経過が増えて表示が戻らない) ──
    [Fact]
    public void Mmss_IsMonotonicOverTime()
    {
        var previous = "00:00";
        for (var s = 0; s <= 6100; s += 7)
        {
            var current = ElapsedFormat.Mmss(TimeSpan.FromSeconds(s));
            Assert.True(string.CompareOrdinal(current, previous) >= 0,
                $"{s} 秒で表示が戻った: {previous} → {current}");
            previous = current;
        }
    }
}
