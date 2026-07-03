using Dollama.Ui.Services;
using Xunit;

namespace Dollama.Ui.Tests;

// DraftPreview.ResolveDraftSize の解像度決定ロジックを検証する。
// 判定軸は幅 W のみ: W>768 は 768x768 へ・W<=768 は据え置き・パース不能/空は 768x768。
public sealed class DraftPreviewTests
{
    // (1) 通常ケース: 幅 768 超は下書き 768x768 へ落とす。
    // (2) 768 以下は据え置き (512 は不採用なので下げない)。
    // (3) パース不能 / 空は安全側で 768x768。
    [Theory]
    [InlineData("1024x1024", "768x768")]   // 幅 1024 > 768 → 落とす
    [InlineData("768x768", "768x768")]     // 幅 768 = 768 → 据え置き (結果も 768x768)
    [InlineData("512x512", "512x512")]     // 幅 512 < 768 → 据え置き (下げない)
    [InlineData("1536x1536", "768x768")]   // 幅 1536 > 768 → 落とす
    [InlineData("1024x768", "768x768")]    // 幅 1024 > 768 → 落とす (高さは無関係)
    [InlineData("abc", "768x768")]         // パース不能 → 安全側
    [InlineData("", "768x768")]            // 空 → 安全側
    public void ResolveDraftSize_ReturnsExpected(string productionSize, string expected)
    {
        Assert.Equal(expected, DraftPreview.ResolveDraftSize(productionSize));
    }
}
