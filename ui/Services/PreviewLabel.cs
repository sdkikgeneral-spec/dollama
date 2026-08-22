using System.Globalization;

namespace Dollama.Ui.Services;

// プレビュー右上に出す「どのモードで・どの解像度で焼いた絵か」の一行文言
// (docs/ui-brushup-plan.md §5 P3-5 / §4.3 プレビュー)。
//
// 元は Generate.razor の private static ModeBadge + 呼び出し側の文字列補間だった。
// P3-5 のミニ履歴が **同じバッジを履歴の各サムネにも付ける**ため、razor 内の
// private メソッドのままだと「本流と履歴で別々に組み立てる」二重実装になる。
// 唯一の合流点として純クラスへ昇格させた (DraftPreview / PresetSaveMessage と同じ流儀)。
//
// ★ 移設は「移設前の ModeBadge に対してテストを書いて緑にする → 移す」の順で行った。
//   期待値表 (ui.Tests/PreviewLabelTests.cs の BadgeTable) は旧実装から採取した実測値で、
//   読める size については移設前後で 1 文字も変えていない。
//   変えたのは退化入力 (null / 空 / 空白のみ) だけ = 旧実装は null で落ち、空だと
//   "本番 " と末尾スペースが出ていた。ここは安全側へ堅牢化した (下記 Build のコメント)。
public static class PreviewLabel
{
    // モード名。画面にそのまま出る語。
    // ★ public にしない: 外へ出すと「語だけ借りて自前で組み立てる」呼び出し側が生まれ、
    //   唯一の合流点という設計が崩れる。公開 API は Build 1 本だけ。
    private const string DraftMode = "下書き";
    private const string ProductionMode = "本番";

    // draft : 下書き (高速プレビュー) モードで焼いたか
    // size  : 実際に C++ サーバーへ送った "WxH" (下書きなら DraftPreview.ResolveDraftSize 済みの値)
    //
    // 例: Build(false, "1024x1024") → "本番 1024²"
    //     Build(true,  "768x768")   → "下書き 768²"
    //
    // 退化入力 (null / 空 / 空白のみ / 読めない size) でも例外は投げない。
    // 描画中に呼ばれるので落とせない (旧 ModeBadge は null で NullReferenceException だった)。
    // バッジが空になるときはモード名だけを返す (末尾スペースを残さない)。
    public static string Build(bool draft, string? size)
    {
        var mode = draft ? DraftMode : ProductionMode;
        var badge = Badge(size);
        return badge.Length == 0 ? mode : $"{mode} {badge}";
    }

    // "WxH" の幅を取り出して「768²」形式にする。読めなければ前後空白を落として素通し。
    // カルチャ非依存 (InvariantCulture 固定): 桁区切りや負符号の表記でぶれない。
    private static string Badge(string? size)
    {
        if (string.IsNullOrWhiteSpace(size))
        {
            return "";
        }

        // 'x' は大文字小文字どちらでも区切りとして扱う (DraftPreview.ResolveDraftSize と同じ規則)。
        var parts = size.Split('x', 'X');
        if (parts.Length >= 1 &&
            int.TryParse(parts[0].Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var width))
        {
            return width.ToString(CultureInfo.InvariantCulture) + "²";
        }

        return size.Trim();
    }
}
