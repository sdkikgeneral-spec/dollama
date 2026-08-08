namespace Dollama.Ui.Services;

// チップ入力の「未確定テキスト (draft)」をタグ列へ確定する純ロジック
// (docs/ui-brushup-plan.md §5 P2-1)。
//
// razor に文字列処理を散らさず、テスト可能な純クラスへ切り出す (DraftPreview と同じ流儀)。
// 名前は「下書き(高速プレビュー)モード」の DraftPreview と紛らわしくないよう、
// TagDraft ではなく TagCommit にしてある (draft = 未確定入力 / Draft = 生成モードの下書き)。
//
// 正規化規則は既存経路 (FavoriteTagStore.Add / Generate.AddTag) と同一:
// **前後空白を落として小文字へ寄せる**。タグ値は英語 danbooru タグのまま保持する。
public static class TagCommit
{
    // タグ 1 個を正規化する。null は "" 扱い。語中の空白 ("long hair") は保持する。
    public static string Normalize(string? raw)
    {
        return (raw ?? "").Trim().ToLowerInvariant();
    }

    // 未確定テキストをカンマで展開してタグ列にする。
    //  - 各要素を Normalize
    //  - 空要素は捨てる (",,a," → ["a"])
    //  - 入力内の重複は先勝ちで除去 ("a, A" → ["a"])
    //  - 入力順は保つ
    // null / 空 / 空白のみ → 空リスト。例外は投げない。
    public static IReadOnlyList<string> Split(string? draft)
    {
        var tags = new List<string>();
        if (string.IsNullOrWhiteSpace(draft))
        {
            return tags;
        }

        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var part in draft.Split(','))
        {
            var tag = Normalize(part);
            if (tag.Length == 0 || !seen.Add(tag))
            {
                continue;
            }
            tags.Add(tag);
        }
        return tags;
    }

    // 既存タグ列 current に draft を確定して連結した **新しいリスト** を返す。
    // current は書き換えない (UI の「新リスト再代入」流儀に合わせ、常に別インスタンスを返す)。
    //
    // 重複判定は正規化後で行うが、既存要素は原文のまま温存する
    // (プリセット由来の表記を確定操作で書き換えないため)。
    public static List<string> Merge(IReadOnlyList<string> current, string? draft)
    {
        var next = new List<string>();
        var seen = new HashSet<string>(StringComparer.Ordinal);

        if (current is not null)
        {
            foreach (var tag in current)
            {
                next.Add(tag);
                seen.Add(Normalize(tag));
            }
        }

        foreach (var tag in Split(draft))
        {
            if (seen.Add(tag))
            {
                next.Add(tag);
            }
        }

        return next;
    }
}
