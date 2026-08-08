using System.Globalization;
using System.Text.RegularExpressions;

namespace Dollama.Ui.Services;

// 生成画像を保存するときのファイル名を決める純ロジック
// (docs/ui-brushup-plan.md §4.3 プレビュー / P2-3・課題 #8)。
//
// 保存は `<a download="…" href="data:image/png;base64,…">` だけで完結し JS interop は不要。
// ただし download 属性の値はブラウザが**そのままファイル名に使う**ので、
// パス区切りや OS の禁止文字が絶対に混ざらないことをここで保証する。
public static partial class DownloadName
{
    // ファイル名の接頭辞。
    public const string Prefix = "dollama";

    // size が "WxH" として読めないときの代替 (無効な文字を持ち込まないための固定語)。
    public const string UnknownSize = "unknown";

    // 生成画像の保存ファイル名。
    //   例: ForPng(2026-08-08 14:25:30, "1024x1024") → "dollama_20260808_142530_1024x1024.png"
    //
    // - 日時は yyyyMMdd_HHmmss (InvariantCulture 固定・和暦や仏暦のロケールでも崩れない)
    // - size は数字と 'x' だけに正規化する ("1024 X 1024" → "1024x1024")
    // - 読めない size (null / 空 / "abc" / "../x" 等) は "unknown"
    // 例外は投げない。戻り値に含まれるのは英数字・アンダースコア・ドットのみ。
    public static string ForPng(DateTime at, string? size)
    {
        var stamp = at.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
        return $"{Prefix}_{stamp}_{NormalizeSize(size)}.png";
    }

    // "WxH" を数字と 'x' だけの形へ正規化する。読めなければ UnknownSize。
    // ★ 許可リスト方式 (危険文字の除去ではなく「数字と x 以外は全部 unknown」) にしてある。
    //   除去方式だと新種の文字を見落としたときに素通りする。
    private static string NormalizeSize(string? size)
    {
        if (string.IsNullOrWhiteSpace(size))
        {
            return UnknownSize;
        }

        var m = SizeRegex().Match(size);
        if (!m.Success)
        {
            return UnknownSize;
        }

        return $"{m.Groups[1].Value}x{m.Groups[2].Value}";
    }

    // 幅・高さとも 1〜5 桁の数字。前後と区切りの空白は許す。
    [GeneratedRegex(@"^\s*(\d{1,5})\s*[xX]\s*(\d{1,5})\s*$")]
    private static partial Regex SizeRegex();
}
