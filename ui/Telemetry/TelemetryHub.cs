using Microsoft.AspNetCore.SignalR;

namespace Dollama.Ui.Telemetry;

// クライアント (Generate.razor) が接続する SignalR ハブ。
// サーバー → クライアントの一方向 push ("Update" メソッドで TelemetrySample を送る) だけなので
// ハブ自体にメソッドは無い。実際の送信は TelemetryBroadcaster が IHubContext 経由で行う。
public sealed class TelemetryHub : Hub
{
}
