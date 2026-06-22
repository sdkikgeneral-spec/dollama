namespace Dollama.Ui.Telemetry;

// SignalR で push する 1 フレーム分の HW テレメトリ。
// 4 デバイス (CPU / NPU / iGPU / RTX5080) の使用率と役割ラベル。
public sealed class TelemetrySample
{
    public List<DeviceLoad> Devices { get; set; } = new();

    // この瞬間に生成が走っているか (UI 側で強調表示するのに使う)
    public bool Generating { get; set; }
}

public sealed class DeviceLoad
{
    // "CPU" / "NPU" / "iGPU" / "RTX5080"
    public string Device { get; set; } = "";

    // パイプライン上の役割 ("Qwen2 LLM" / "CLIP enc" / "VAE enc" / "SDXL UNet")
    public string Role { get; set; } = "";

    // 使用率 0-100
    public double Percent { get; set; }
}
