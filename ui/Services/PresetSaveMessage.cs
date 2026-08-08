namespace Dollama.Ui.Services;

// プリセット保存後に出す一行メッセージの文言 (docs/ui-brushup-plan.md §5 P2-7 / 課題 #19)。
//
// PresetStore.Save は同一 kind の同名を**黙って上書き**する (store 側の仕様は正しい)。
// UI がそれを一言も言わないと「保存したのに増えていない」に見えるため、
// 上書きだったかどうかを文言に出す。
//
// 判定 (同名が既にあるか) は呼び出し側が PresetStore.All(kind) を読んで行い、
// ここは**文言の組み立てだけ**を担う純ロジック (razor に三項演算子を積まないため)。
// 入力は真偽 2 つなので 4 通りを全数テストできる。
public static class PresetSaveMessage
{
    // name      : 保存したプリセット名 (前後空白は落として表示する)
    // overwrite : 同一 kind に同名が既にあり、上書きになったか
    // hasThumb  : サムネイル (直近の生成画像) 付きで保存したか
    public static string Build(string name, bool overwrite, bool hasThumb)
    {
        var label = (name ?? "").Trim();
        var verb = overwrite ? "上書き保存しました" : "保存しました";

        return hasThumb
            ? $"「{label}」をサムネ付きで{verb}"
            : $"「{label}」を{verb}";
    }
}
