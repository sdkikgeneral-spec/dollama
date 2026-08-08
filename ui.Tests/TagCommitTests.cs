using Dollama.Ui.Services;
using Xunit;

namespace Dollama.Ui.Tests;

// TagCommit (チップ入力の未確定テキスト → タグ列の確定) の純ロジック検証。
// 出典: docs/ui-brushup-plan.md §5 P2-1 / §7「draft 確定の正規化 (trim/lowercase/
// カンマ展開/重複) をロジック単体で検証」。
public sealed class TagCommitTests
{
    // ────────────────────────────────────────────────
    // (1) Normalize: 既存経路 (FavoriteTagStore.Add / Generate.AddTag) と同じ規則
    // ────────────────────────────────────────────────
    [Theory]
    [InlineData("1girl", "1girl")]
    [InlineData("  1girl  ", "1girl")]
    [InlineData("1GIRL", "1girl")]
    [InlineData("Long Hair", "long hair")]   // 語中の空白は保持する
    [InlineData("\t smile \r\n", "smile")]
    [InlineData("", "")]
    [InlineData("   ", "")]
    [InlineData(null, "")]                   // null でも例外を投げない
    public void Normalize_TrimsAndLowercases(string? raw, string expected)
    {
        Assert.Equal(expected, TagCommit.Normalize(raw));
    }

    // ────────────────────────────────────────────────
    // (2) Split: カンマ展開・空除去・入力内重複除去・順序保持
    // ────────────────────────────────────────────────
    [Fact]
    public void Split_ReturnsEmptyForBlankInput()
    {
        Assert.Empty(TagCommit.Split(null));
        Assert.Empty(TagCommit.Split(""));
        Assert.Empty(TagCommit.Split("   "));
        Assert.Empty(TagCommit.Split(",, , ,"));
    }

    [Fact]
    public void Split_SingleTagIsNormalized()
    {
        Assert.Equal(new[] { "1girl" }, TagCommit.Split("  1Girl "));
    }

    [Fact]
    public void Split_ExpandsCommasAndKeepsOrder()
    {
        Assert.Equal(
            new[] { "1girl", "long hair", "smile" },
            TagCommit.Split("1girl, Long Hair ,SMILE"));
    }

    [Fact]
    public void Split_DropsEmptySegments()
    {
        Assert.Equal(new[] { "a", "b" }, TagCommit.Split(",a,,  , b,"));
    }

    [Fact]
    public void Split_RemovesDuplicatesWithinInputCaseInsensitively()
    {
        // 先勝ち (最初の出現順を保つ)
        Assert.Equal(new[] { "1girl", "smile" }, TagCommit.Split("1girl, 1GIRL , smile, 1girl"));
    }

    // ────────────────────────────────────────────────
    // (3) Merge: 新リスト再代入の流儀 (既存参照を Mutate しない)
    // ────────────────────────────────────────────────
    [Fact]
    public void Merge_AppendsDraftTagsToCurrent()
    {
        var current = new List<string> { "1girl" };
        var next = TagCommit.Merge(current, "smile, Blush");

        Assert.Equal(new[] { "1girl", "smile", "blush" }, next);
    }

    [Fact]
    public void Merge_NeverMutatesCurrentAndAlwaysReturnsNewInstance()
    {
        var current = new List<string> { "1girl" };
        var next = TagCommit.Merge(current, "smile");

        Assert.NotSame(current, next);              // 常に別インスタンス
        Assert.Equal(new[] { "1girl" }, current);   // 元リストは不変
    }

    [Fact]
    public void Merge_WithBlankDraftReturnsCopyOfCurrent()
    {
        var current = new List<string> { "1girl", "smile" };

        foreach (var draft in new string?[] { null, "", "   ", " , , " })
        {
            var next = TagCommit.Merge(current, draft);
            Assert.NotSame(current, next);
            Assert.Equal(current, next);
        }
    }

    [Fact]
    public void Merge_SkipsTagsAlreadyPresent()
    {
        var current = new List<string> { "1girl", "smile" };
        var next = TagCommit.Merge(current, "smile, 1girl, blush");

        Assert.Equal(new[] { "1girl", "smile", "blush" }, next);
    }

    [Fact]
    public void Merge_DuplicateCheckIsCaseInsensitiveButKeepsExistingSpelling()
    {
        // プリセット由来の表記 ("Long Hair") は確定操作で書き換えない。
        var current = new List<string> { "Long Hair" };
        var next = TagCommit.Merge(current, "long hair, smile");

        Assert.Equal(new[] { "Long Hair", "smile" }, next);
    }

    [Fact]
    public void Merge_FromEmptyCurrentEqualsSplit()
    {
        var next = TagCommit.Merge(new List<string>(), "1girl, smile , 1GIRL");
        Assert.Equal(TagCommit.Split("1girl, smile , 1GIRL"), next);
        Assert.Equal(new[] { "1girl", "smile" }, next);
    }

    // 「確定で何個増えたか」は Count 差で判定できること
    // (TagInput は Count が変わったときだけ TagsChanged を通知する)。
    [Fact]
    public void Merge_CountDeltaTellsWhetherAnythingWasAdded()
    {
        var current = new List<string> { "1girl" };

        Assert.Equal(current.Count, TagCommit.Merge(current, "1GIRL").Count);   // 重複のみ → 増えない
        Assert.Equal(current.Count, TagCommit.Merge(current, "   ").Count);     // 空白のみ → 増えない
        Assert.Equal(current.Count + 1, TagCommit.Merge(current, "smile").Count);
    }
}
