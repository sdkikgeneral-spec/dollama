using Dollama.Ui.Services;
using Xunit;

namespace Dollama.Ui.Tests;

// GenerateGate (生成ボタンの活性条件と理由テキスト) の検証。
// 出典: docs/ui-brushup-plan.md §3 課題 #1/#2/#21 / §4.3 / §5 P2-1・P2-8・P2-9。
//
// 入力は 4 つの真偽 (busy / connected / タグ有無 / draft 空) なので **16 組合せを全数検査**する。
// razor に条件式を散らさない代わりに、ここで表として固定するのが本テストの役目。
public sealed class GenerateGateTests
{
    // ────────────────────────────────────────────────
    // (1) 16 組合せ全数 (busy × connected × promptTagCount(0/1) × draftEmpty)
    // ────────────────────────────────────────────────
    [Theory]
    // busy=false / connected=false … 未接続が理由 (タグや draft の状態によらず)
    [InlineData(false, false, 0, true, false, GenerateGate.ReasonDisconnected)]
    [InlineData(false, false, 0, false, false, GenerateGate.ReasonDisconnected)]
    [InlineData(false, false, 1, true, false, GenerateGate.ReasonDisconnected)]
    [InlineData(false, false, 1, false, false, GenerateGate.ReasonDisconnected)]
    // busy=false / connected=true … プロンプトの有無で決まる
    [InlineData(false, true, 0, true, false, GenerateGate.ReasonNoPrompt)]
    [InlineData(false, true, 0, false, true, null)]   // ★ 罠の核心: draft だけでも押せる
    [InlineData(false, true, 1, true, true, null)]
    [InlineData(false, true, 1, false, true, null)]
    // busy=true … 常に押せない。文言が「生成中…」なので理由は出さない
    [InlineData(true, false, 0, true, false, null)]
    [InlineData(true, false, 0, false, false, null)]
    [InlineData(true, false, 1, true, false, null)]
    [InlineData(true, false, 1, false, false, null)]
    [InlineData(true, true, 0, true, false, null)]
    [InlineData(true, true, 0, false, false, null)]
    [InlineData(true, true, 1, true, false, null)]
    [InlineData(true, true, 1, false, false, null)]
    public void Evaluate_CoversAllSixteenCombinations(
        bool busy, bool connected, int promptTagCount, bool draftEmpty,
        bool expectedCan, string? expectedReason)
    {
        var (can, reason) = GenerateGate.Evaluate(busy, connected, promptTagCount, draftEmpty);

        Assert.Equal(expectedCan, can);
        Assert.Equal(expectedReason, reason);
    }

    // ────────────────────────────────────────────────
    // (2) 罠の核心を単独でも固定しておく:
    //     「draft に文字があるがタグ 0」は **enabled**。
    //     ここを disabled にすると生成前確定 (P2-1) の経路へ永久に入れない。
    // ────────────────────────────────────────────────
    [Fact]
    public void Evaluate_DraftTextAloneEnablesGeneration()
    {
        var (can, reason) = GenerateGate.Evaluate(busy: false, connected: true, promptTagCount: 0, draftEmpty: false);

        Assert.True(can);
        Assert.Null(reason);
    }

    // ────────────────────────────────────────────────
    // (3) 理由の優先順位: busy が最優先 (未接続でも理由は出さない)
    // ────────────────────────────────────────────────
    [Fact]
    public void Evaluate_BusyHidesReasonEvenWhenDisconnected()
    {
        var (can, reason) = GenerateGate.Evaluate(busy: true, connected: false, promptTagCount: 0, draftEmpty: true);

        Assert.False(can);
        Assert.Null(reason);
    }

    // 未接続はプロンプト空より優先して出す (先に直すべきはサーバー接続のため)
    [Fact]
    public void Evaluate_DisconnectedTakesPriorityOverEmptyPrompt()
    {
        var (_, reason) = GenerateGate.Evaluate(busy: false, connected: false, promptTagCount: 0, draftEmpty: true);

        Assert.Equal(GenerateGate.ReasonDisconnected, reason);
    }

    // ────────────────────────────────────────────────
    // (4) タグ数は 0 超かどうかだけを見る (負値は 0 扱い・多数でも同じ)
    // ────────────────────────────────────────────────
    [Theory]
    [InlineData(-1, false)]
    [InlineData(0, false)]
    [InlineData(1, true)]
    [InlineData(42, true)]
    public void Evaluate_TagCountIsOnlyCheckedAgainstZero(int count, bool expectedCan)
    {
        var (can, _) = GenerateGate.Evaluate(busy: false, connected: true, promptTagCount: count, draftEmpty: true);
        Assert.Equal(expectedCan, can);
    }

    // ────────────────────────────────────────────────
    // (5) 理由テキストは空でなく、互いに異なること (画面にそのまま出る文言)
    // ────────────────────────────────────────────────
    [Fact]
    public void Reasons_AreDistinctAndNonEmpty()
    {
        Assert.False(string.IsNullOrWhiteSpace(GenerateGate.ReasonDisconnected));
        Assert.False(string.IsNullOrWhiteSpace(GenerateGate.ReasonNoPrompt));
        Assert.NotEqual(GenerateGate.ReasonDisconnected, GenerateGate.ReasonNoPrompt);
    }
}
