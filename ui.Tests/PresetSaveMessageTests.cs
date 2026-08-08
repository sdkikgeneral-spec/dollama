using System.Globalization;
using System.Runtime.CompilerServices;
using Dollama.Ui.Services;
using Xunit;

namespace Dollama.Ui.Tests;

// PresetSaveMessage (プリセット保存後の一行メッセージ) の検証。
// 出典: docs/ui-brushup-plan.md §3 課題 #19 / §4.3 / §5 P2-7。
//
// PresetStore.Save は同一 kind の同名を黙って上書きする。UI がそれを言わないと
// 「保存したのに増えていない」に見えるので、上書きかどうかを文言に必ず出す。
// 入力は真偽 2 つ (overwrite × hasThumb) なので **4 通りを全数検査**する。
public sealed class PresetSaveMessageTests
{
    // ────────────────────────────────────────────────
    // (1) 4 通り全数
    // ────────────────────────────────────────────────
    [Theory]
    [InlineData(false, false, "「base-girl」を保存しました")]
    [InlineData(true, false, "「base-girl」を上書き保存しました")]
    [InlineData(false, true, "「base-girl」をサムネ付きで保存しました")]
    [InlineData(true, true, "「base-girl」をサムネ付きで上書き保存しました")]
    public void Build_CoversAllFourCombinations(bool overwrite, bool hasThumb, string expected)
    {
        Assert.Equal(expected, PresetSaveMessage.Build("base-girl", overwrite, hasThumb));
    }

    // ────────────────────────────────────────────────
    // (2) 上書きかどうかが文言で必ず区別できること (課題 #19 の核心)。
    //     ここが同じ文言に退化すると「黙って上書き」に逆戻りする。
    // ────────────────────────────────────────────────
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Build_OverwriteAndNewAreAlwaysDistinguishable(bool hasThumb)
    {
        var fresh = PresetSaveMessage.Build("neg-common", overwrite: false, hasThumb);
        var over = PresetSaveMessage.Build("neg-common", overwrite: true, hasThumb);

        Assert.NotEqual(fresh, over);
        Assert.Contains("上書き", over);
        Assert.DoesNotContain("上書き", fresh);
    }

    // ────────────────────────────────────────────────
    // (3) サムネの有無も文言で区別できること (従来からの挙動を維持)
    // ────────────────────────────────────────────────
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Build_ThumbnailPresenceIsVisible(bool overwrite)
    {
        var without = PresetSaveMessage.Build("x", overwrite, hasThumb: false);
        var with = PresetSaveMessage.Build("x", overwrite, hasThumb: true);

        Assert.NotEqual(without, with);
        Assert.Contains("サムネ", with);
        Assert.DoesNotContain("サムネ", without);
    }

    // ────────────────────────────────────────────────
    // (4) 名前の扱い: 必ず「」で囲んで出す・前後空白は落とす
    //     (PresetStore.Save も name を Trim して保存するので表示と実体が一致する)
    // ────────────────────────────────────────────────
    [Theory]
    [InlineData("base-girl", "base-girl")]
    [InlineData("  base-girl  ", "base-girl")]
    [InlineData("\tねこ耳\n", "ねこ耳")]
    [InlineData("a b", "a b")]                 // 語中の空白は保持
    public void Build_TrimsNameAndWrapsInBrackets(string input, string shown)
    {
        var msg = PresetSaveMessage.Build(input, overwrite: false, hasThumb: false);

        Assert.Contains($"「{shown}」", msg);
        Assert.StartsWith("「", msg);
    }

    // 日本語名・記号入りでもそのまま出す (ファイル名の正規化はストア側の責務)。
    [Theory]
    [InlineData("ねこ耳ロング")]
    [InlineData("preset #1 (v2)")]
    [InlineData("a/b\\c")]
    public void Build_KeepsNameVerbatim(string name)
    {
        Assert.Contains($"「{name}」", PresetSaveMessage.Build(name, overwrite: true, hasThumb: true));
    }

    // ────────────────────────────────────────────────
    // (5) 例外を投げない・空文字でも文言として成立すること
    //     (保存側は空名を弾くので到達しないが、文言生成で落ちてはいけない)
    // ────────────────────────────────────────────────
    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData(null)]
    public void Build_IsSafeForEmptyOrNullName(string? name)
    {
        var msg = PresetSaveMessage.Build(name!, overwrite: false, hasThumb: false);

        Assert.Contains("「」", msg);
        Assert.Contains("保存しました", msg);
    }

    // ────────────────────────────────────────────────
    // (6) 形式の不変条件: 常に非空・「」を 1 組だけ含み・「保存しました」で終わる
    // ────────────────────────────────────────────────
    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void Build_AlwaysEndsWithSavedAndHasSingleBracketPair(bool overwrite, bool hasThumb)
    {
        var msg = PresetSaveMessage.Build("p", overwrite, hasThumb);

        Assert.False(string.IsNullOrWhiteSpace(msg));
        Assert.EndsWith("保存しました", msg);
        Assert.Equal(1, msg.Count(c => c == '「'));
        Assert.Equal(1, msg.Count(c => c == '」'));
    }

    // ────────────────────────────────────────────────
    // (7) カルチャ非依存 (文言に数値・日付を混ぜていないことの担保)
    // ────────────────────────────────────────────────
    [Theory]
    [InlineData("de-DE")]
    [InlineData("ar-SA")]
    [InlineData("ja-JP")]
    [InlineData("")]
    public void Build_IsCultureInvariant(string culture)
    {
        var expected = PresetSaveMessage.Build("p", overwrite: true, hasThumb: true);

        var prev = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = new CultureInfo(culture);
            Assert.Equal(expected, PresetSaveMessage.Build("p", overwrite: true, hasThumb: true));
        }
        finally
        {
            CultureInfo.CurrentCulture = prev;
        }
    }

    // ────────────────────────────────────────────────
    // (8) 純関数であること (同じ入力で常に同じ出力)
    // ────────────────────────────────────────────────
    [Fact]
    public void Build_IsDeterministic()
    {
        var a = PresetSaveMessage.Build("same", overwrite: true, hasThumb: false);
        var b = PresetSaveMessage.Build("same", overwrite: true, hasThumb: false);

        Assert.Equal(a, b);
    }

    // ────────────────────────────────────────────────
    // (9) 呼び出し側 (TagPresetField.razor) の配線をテキストで固定する。
    //     bUnit は入れない方針なのでレンダリングは組まず、
    //     「壊れると気づきにくい 3 点」だけを機械検査する。
    // ────────────────────────────────────────────────
    [Fact]
    public void TagPresetField_HidesSaveBarWhenEmptyAndDetectsOverwriteWithoutTouchingStore()
    {
        var razor = File.ReadAllText(Path.Combine(
            RepoRoot(), "ui", "Components", "Shared", "TagPresetField.razor"));

        // ① 保存バーの非表示条件はタグ 0 個 (課題 #20)
        Assert.Contains("@if (Tags.Count > 0)", razor);

        // ② 文言はこの純クラス経由 (razor に三項演算子を戻さない)
        Assert.Contains("PresetSaveMessage.Build(", razor);
        Assert.DoesNotContain("を保存しました\"", razor);

        // ③ 同名判定は PresetStore.All(kind) の読み取りだけで行う (store は不可侵)
        Assert.Contains("Presets.All(Kind)", razor);
    }

    // リポジトリルート (ui/wwwroot/app.css を目印に遡る)。
    private static string RepoRoot([CallerFilePath] string thisFile = "")
    {
        for (var dir = new DirectoryInfo(Path.GetDirectoryName(thisFile)!); dir is not null; dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "ui", "wwwroot", "app.css")))
            {
                return dir.FullName;
            }
        }

        Assert.Fail($"リポジトリルートを特定できない (探索起点: {thisFile})");
        return "";
    }
}
