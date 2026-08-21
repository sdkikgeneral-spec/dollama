using Microsoft.AspNetCore.SignalR;

namespace Dollama.Ui.Telemetry;

// 約 500ms 周期で HW テレメトリを全クライアントへ push する常駐サービス。
//
// ★ 現状はスタブ: C++ 側に /telemetry エンドポイントが無いため、
//    時刻ベースの擬似波形 (デバイスごとに位相をずらした sin) を生成して送る。
//    生成中 (GenerationActivity.IsGenerating) は GPU を高負荷側に張り付かせ、
//    NPU/iGPU/CPU も役割に応じて変化させる。
//
// ★ TODO(実 HW 配線): C++ サーバーに GET /telemetry が実装されたら、
//    このループを「DollamaClient で /telemetry をポーリング → 実測値を中継」に置き換える。
//    SignalR の push 構造とクライアント側はそのまま再利用できる。
public sealed class TelemetryBroadcaster : BackgroundService
{
    private readonly IHubContext<TelemetryHub> _hub;
    private readonly GenerationActivity _activity;

    private const int PeriodMs = 500;

    public TelemetryBroadcaster(IHubContext<TelemetryHub> hub, GenerationActivity activity)
    {
        _hub = hub;
        _activity = activity;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        // 経過時間 t [秒] を進めながら擬似波形を作る (Random は使わず決定的に)
        double t = 0;
        using var timer = new PeriodicTimer(TimeSpan.FromMilliseconds(PeriodMs));

        while (await timer.WaitForNextTickAsync(stoppingToken))
        {
            t += PeriodMs / 1000.0;
            var generating = _activity.IsGenerating;
            await _hub.Clients.All.SendAsync("Update", BuildSample(t, generating), stoppingToken);
        }
    }

    // idle 時はゆるい揺らぎ、生成中は役割に応じて負荷が上がるスタブ波形。
    private static TelemetrySample BuildSample(double t, bool generating)
    {
        // (デバイス, 役割, 位相, idle中央値, 生成中の上乗せ)
        (string dev, string role, double phase, double idle, double busy)[] specs =
        {
            ("CPU",     "Tag LM",    0.0, 12, 35),
            ("NPU",     "CLIP enc",  1.6, 4,  55),
            ("iGPU",    "VAE enc",   3.1, 6,  40),
            ("RTX5080", "SDXL UNet", 4.7, 8,  88),
        };

        var devices = new List<DeviceLoad>(specs.Length);
        foreach (var s in specs)
        {
            // 0..1 のゆらぎ
            double wobble = 0.5 + 0.5 * Math.Sin(t * 1.3 + s.phase);
            double baseline = generating ? s.busy : s.idle;
            // 生成中は ±15% 程度、idle 時は ±4% 程度に揺らす
            double amp = generating ? 15 : 4;
            double pct = baseline + (wobble - 0.5) * 2 * amp;
            devices.Add(new DeviceLoad
            {
                Device = s.dev,
                Role = s.role,
                Percent = Math.Clamp(pct, 0, 100),
            });
        }

        return new TelemetrySample { Devices = devices, Generating = generating };
    }
}
