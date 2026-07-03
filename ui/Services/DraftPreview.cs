namespace Dollama.Ui.Services;

// 下書き (高速プレビュー) モードの解像度決定ロジック。
// 本番と同じ SDXL 重み・同じステップ数のまま、解像度だけ下げて当たり付けを速くする。
// プレビュー解像度は 768x768 固定 (512x512 は崩れるので不採用)。
// C++ サーバーは無改修で、既存 POST /v1/images/generations に size を変えて投げるだけ。
public static class DraftPreview
{
    // 下書きプレビューの固定解像度。
    public const string DraftSize = "768x768";

    // 本番選択サイズ productionSize から下書き用サイズを決める。
    // 判定軸は幅 W のみ:
    //   - W > 768  → 下書きへ落とす (DraftSize)
    //   - W <= 768 → そのまま据え置き (512 は不採用なので下げない)
    //   - パース不能 / 空 / null → 安全側で DraftSize
    // 例外は投げない (int.TryParse で安全に判定)。
    public static string ResolveDraftSize(string? productionSize)
    {
        if (string.IsNullOrWhiteSpace(productionSize))
        {
            return DraftSize;
        }

        // "WxH" を 'x' (大文字小文字どちらでも) で分割し、幅 W のみ見る。
        var parts = productionSize.Split('x', 'X');
        if (parts.Length < 1 || !int.TryParse(parts[0].Trim(), out var width))
        {
            return DraftSize;
        }

        // 幅が 768 を超えるときだけ下書きへ落とす。768 以下はそのまま。
        return width > 768 ? DraftSize : productionSize;
    }
}
