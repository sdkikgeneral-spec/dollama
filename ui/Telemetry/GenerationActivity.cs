namespace Dollama.Ui.Telemetry;

// 生成中かどうかを Broadcaster と共有するシングルトン。
// Generate.razor が生成の開始/終了で Enter()/Exit() を呼び、
// TelemetryBroadcaster がその状態を見て「生成中は GPU が跳ねる」スタブ波形にする。
//
// 実 HW 配線 (C++ /telemetry) が入ったらこのフラグは不要になり、
// Broadcaster が C++ から取得した実測値を中継する形に置き換わる。
public sealed class GenerationActivity
{
    private int _active;

    public bool IsGenerating => Volatile.Read(ref _active) > 0;

    public void Enter() => Interlocked.Increment(ref _active);

    public void Exit() => Interlocked.Decrement(ref _active);
}
