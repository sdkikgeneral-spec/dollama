namespace Dollama.Ui.Services;

// タグパレットから親 (Generate.razor) へタグ追加を通知するためのペイロード。
// Target は "prompt" / "negative" のどちらの列へ足すか。
public sealed record TagAdd(string Tag, string Target);
