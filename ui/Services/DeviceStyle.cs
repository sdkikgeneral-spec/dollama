namespace Dollama.Ui.Services;

// テレメトリ・ミニメーターのデバイス別配色クラスを決める純ロジック。
//
// razor に三項演算子を並べず、テスト可能な純クラスへ切り出す (DraftPreview と同じ流儀)。
// デバイス名の出所は Telemetry/TelemetryBroadcaster.cs の 4 種
// ("CPU" / "NPU" / "iGPU" / "RTX5080")。
// 返すクラス名は wwwroot/app.css の .tm-cpu / .tm-npu / .tm-igpu / .tm-gpu に対応し、
// それぞれ --dev-cpu / --dev-npu / --dev-igpu / --dev-gpu で fill と % を塗り分ける。
public static class DeviceStyle
{
    // デバイス名 → app.css のデバイス色クラス名。
    // 大文字小文字は区別しない (前後の空白も無視する)。
    // 未知のデバイス名・null・空文字は "" を返し、.tm-fill 既定のグラデへフォールバックする
    // (将来 C++ 側が新しいデバイスを push しても表示が壊れないようにするため)。
    public static string CssClass(string? device)
    {
        if (string.IsNullOrWhiteSpace(device))
        {
            return "";
        }

        switch (device.Trim().ToLowerInvariant())
        {
        case "cpu":
            return "tm-cpu";
        case "npu":
            return "tm-npu";
        case "igpu":
            return "tm-igpu";
        case "rtx5080":
            return "tm-gpu";
        default:
            return "";
        }
    }
}
