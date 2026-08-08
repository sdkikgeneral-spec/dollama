using Dollama.Ui.Services;
using Xunit;

namespace Dollama.Ui.Tests;

// DeviceStyle.CssClass のデバイス名 → CSS クラス写像を検証する。
// 期待値は wwwroot/app.css の .tm-cpu / .tm-npu / .tm-igpu / .tm-gpu と対応する。
public sealed class DeviceStyleTests
{
    // (1) TelemetryBroadcaster が push する 4 デバイスが正しいクラスに写ること。
    [Theory]
    [InlineData("CPU", "tm-cpu")]
    [InlineData("NPU", "tm-npu")]
    [InlineData("iGPU", "tm-igpu")]
    [InlineData("RTX5080", "tm-gpu")]
    public void CssClass_MapsKnownDevices(string device, string expected)
    {
        Assert.Equal(expected, DeviceStyle.CssClass(device));
    }

    // (2) 大文字小文字は区別しない。前後の空白も無視する。
    [Theory]
    [InlineData("cpu", "tm-cpu")]
    [InlineData("Cpu", "tm-cpu")]
    [InlineData("npu", "tm-npu")]
    [InlineData("IGPU", "tm-igpu")]
    [InlineData("igpu", "tm-igpu")]
    [InlineData("rtx5080", "tm-gpu")]
    [InlineData("  RTX5080  ", "tm-gpu")]
    public void CssClass_IsCaseInsensitive(string device, string expected)
    {
        Assert.Equal(expected, DeviceStyle.CssClass(device));
    }

    // (3) 未知・null・空・空白は "" (= 既定グラデへフォールバック)。
    //     将来 C++ 側が新デバイスを push しても表示が壊れないことの担保。
    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("GPU")]
    [InlineData("RTX4090")]
    [InlineData("dGPU")]
    public void CssClass_UnknownFallsBackToEmpty(string? device)
    {
        Assert.Equal("", DeviceStyle.CssClass(device));
    }
}
