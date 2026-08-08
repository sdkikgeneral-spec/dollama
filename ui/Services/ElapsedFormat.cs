using System.Globalization;

namespace Dollama.Ui.Services;

// 生成中オーバーレイに出す経過時間の表示整形 (docs/ui-brushup-plan.md §4.3 プレビュー / P2-2)。
//
// タイマー (PeriodicTimer) そのものは Generate.razor 側に置き、テストできるのは
// 「TimeSpan → 画面文字列」の写像だけなので、そこを純ロジックとして切り出す
// (DraftPreview / GenerateGate と同じ流儀)。
public static class ElapsedFormat
{
    // 表示の下限 (負値・ゼロ)。
    public const string Zero = "00:00";

    // 表示の上限。mm:ss の 2 桁分に収まらない長さは頭打ちにする
    // (生成が 100 分を超えることは想定しないが、桁あふれで表示が崩れないようにする)。
    public const string Max = "99:59";

    // 経過時間を mm:ss (例 "00:12") にする。
    //   - 負値      → "00:00" (時計のずれで負になっても崩さない)
    //   - 端数      → 切り捨て (12.9 秒は "00:12"。カウントアップは切り捨てが自然)
    //   - 100 分以上 → "99:59" で頭打ち (2 桁の mm に収める)
    // 例外は投げない。
    public static string Mmss(TimeSpan elapsed)
    {
        var totalSeconds = elapsed.TotalSeconds;

        if (double.IsNaN(totalSeconds) || totalSeconds <= 0)
        {
            return Zero;
        }

        // 100 分 (6000 秒) 以上は頭打ち。double のまま比較して long への変換であふれさせない。
        if (totalSeconds >= 100 * 60)
        {
            return Max;
        }

        var seconds = (long)totalSeconds; // 端数切り捨て
        var mm = seconds / 60;
        var ss = seconds % 60;

        return string.Create(
            CultureInfo.InvariantCulture, $"{mm:00}:{ss:00}");
    }
}
