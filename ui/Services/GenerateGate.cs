namespace Dollama.Ui.Services;

// 生成ボタンを押せるか / 押せないなら理由は何か を決める純ロジック
// (docs/ui-brushup-plan.md §5 P2-1 / P2-8 / P2-9 の唯一の合流点)。
//
// 「押せるのに何も起きない」「押せないのに理由が出ない」という無言失敗を根絶するため、
// disabled 条件と理由テキストを razor の条件式に散らさず 1 箇所へ集約する。
// Ctrl+Enter (P2-8) も同じ Evaluate を通すので、ボタンとショートカットが食い違わない。
public static class GenerateGate
{
    // 未接続 (GET /health が通っていない) ときの理由。再接続ボタンと対で出す。
    public const string ReasonDisconnected = "C++ サーバーに接続していません";

    // プロンプトが空 (確定タグ 0 かつ未確定 draft も空) のときの理由。
    public const string ReasonNoPrompt = "プロンプトにタグを 1 つ以上追加してください";

    // busy           : 生成中か (2 ボタンのどちらかが走っている)
    // connected      : C++ サーバーに接続できているか (_connected)
    // promptTagCount : 確定済みプロンプトタグの個数
    // draftEmpty     : プロンプト欄の未確定テキストが空か
    //
    // ★ draft に文字があれば確定タグ 0 でも押せる。
    //   ここを「タグ 0 なら無条件 disabled」にすると、生成前確定 (P2-1) の経路へ
    //   永久に入れず「二度押しが要る」という別の無言失敗に化ける。
    //
    // 理由は優先順位付きで 1 つだけ返す:
    //   ① busy         → null (ボタン文言が「生成中…」なので理由は出さない)
    //   ② 未接続       → ReasonDisconnected
    //   ③ タグ 0 かつ draft 空 → ReasonNoPrompt
    //   ④ それ以外     → null (押せる)
    public static (bool CanGenerate, string? Reason) Evaluate(
        bool busy, bool connected, int promptTagCount, bool draftEmpty)
    {
        var hasPrompt = promptTagCount > 0 || !draftEmpty;
        var canGenerate = !busy && connected && hasPrompt;

        if (busy)
        {
            return (canGenerate, null);
        }
        if (!connected)
        {
            return (canGenerate, ReasonDisconnected);
        }
        if (!hasPrompt)
        {
            return (canGenerate, ReasonNoPrompt);
        }
        return (canGenerate, null);
    }
}
