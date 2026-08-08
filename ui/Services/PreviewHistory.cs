namespace Dollama.Ui.Services;

// 直近生成のミニ履歴 1 件分 (docs/ui-brushup-plan.md §5 P3-5)。
//
// ★ byte[] を持たせない (PL 裁定)。履歴が持つのは base64 文字列だけで、
//   生バイトが要るのは「プリセット保存の瞬間」だけなので選択時に 1 回 decode する。
//   ・byte[] のみ保持 → 毎描画で base64 化が走る。生成中は経過秒タイマーで 1 秒ごとに
//     再描画されるため、4 枚分の割り当てを毎秒回すことになる
//   ・両方保持    → 常駐が 2.33 倍 (base64 は 4/3 倍・生バイト 1 倍)
//
// Id は razor 側が単調増加のカウンタで振る (履歴は状態を持たない純ロジックのため)。
// CreatedAt は表示 (title 属性) 用であって並び順の根拠ではない = 並びは投入順だけで決まる。
public sealed record PreviewItem(
    int Id,
    string Base64,
    string Badge,
    string DownloadName,
    DateTimeOffset CreatedAt);

// ミニ履歴の並び替え規則そのもの。状態を持たない静的クラス
// (TagCommit / GenerateGate と同じ流儀・履歴の実体は Generate.razor のフィールド)。
//
// ★ できることは 4 つだけと決めてある: 直近 Capacity 枚を新しい順に並べる / クリックで
//   再表示する / 選択中が分かる / 空なら出さない。
//   削除・並べ替え・ピン留め・永続化 (localStorage / ファイル / presets.json) は
//   **やらない** (設定ごとの復元はプリセットの役割で、ここは「さっきの絵に戻る」だけ)。
//   この境界は ui.Tests/PreviewHistoryTests の public メンバ集合検査が機械的に守る。
public static class PreviewHistory
{
    // 保持する枚数の上限。右ペイン下に 1 行で並ぶ枚数でもある。
    public const int Capacity = 4;

    // current の先頭に item を足した **新しいリスト** を返す (新しい順)。
    // 上限を超えた分は末尾 (= 最も古い) から落とす。
    //
    // current は書き換えない。UI 側は「新しいリストを再代入する」流儀なので、
    // ここで Mutate すると Blazor の差分描画から見て変化が無いのに中身だけ変わる
    // (P2-1 で AddOne の Mutate を是正したのと同じ罠)。常に別インスタンスを返す。
    //
    // 防御: current が null / null 要素を含む / 既に上限超過でも例外は投げない。
    public static IReadOnlyList<PreviewItem> Add(IReadOnlyList<PreviewItem> current, PreviewItem item)
    {
        var next = new List<PreviewItem>(Capacity);

        if (item is not null)
        {
            next.Add(item);
        }

        if (current is not null)
        {
            foreach (var old in current)
            {
                if (next.Count >= Capacity)
                {
                    break;
                }
                if (old is null)
                {
                    continue;
                }
                next.Add(old);
            }
        }

        return next;
    }

    // Id で 1 件引く。見つからなければ null (呼び出し側は「何も起きない」を選べる)。
    // Id が重複していたら先頭 = より新しい方を返す (決定的)。
    public static PreviewItem? Find(IReadOnlyList<PreviewItem> items, int id)
    {
        if (items is null)
        {
            return null;
        }

        foreach (var item in items)
        {
            if (item is not null && item.Id == id)
            {
                return item;
            }
        }
        return null;
    }

    // サムネに選択中の縁取りを付けるか。null 安全 (描画中に呼ぶので落とせない)。
    public static bool IsSelected(PreviewItem item, int selectedId)
    {
        return item is not null && item.Id == selectedId;
    }
}
