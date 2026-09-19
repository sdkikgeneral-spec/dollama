namespace Dollama.Ui.Services;

// エンドポイントのベース URL とパスを結合する純関数。
// 末尾 "/" の有無 (baseUrl 側) と先頭 "/" の有無 (path 側) の 4 通りを吸収し、
// 常に「baseUrl の末尾スラッシュなし」+「/」+「path の先頭スラッシュなし」に正規化する。
public static class EndpointUrl
{
    public static string Combine(string baseUrl, string path)
    {
        var b = (baseUrl ?? "").TrimEnd('/');
        var p = (path ?? "").TrimStart('/');
        return $"{b}/{p}";
    }
}
