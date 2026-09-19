using Dollama.Ui.Components;
using Dollama.Ui.Services;
using Dollama.Ui.Telemetry;
using Microsoft.Extensions.FileProviders;

var builder = WebApplication.CreateBuilder(args);

// Blazor Server (インタラクティブサーバーコンポーネント)
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

// SignalR (テレメトリ push)
builder.Services.AddSignalR();

// C++ 生成サーバーを叩く名前付き HttpClient。
// 2-6f: BaseAddress を起動時に固定せず、DollamaClient がリクエストごとに
// EndpointRegistry.Current の URL で絶対 URL を組み立てる (複数エンドポイント切替)。
builder.Services.AddHttpClient("dollama", c =>
{
    c.Timeout = TimeSpan.FromMinutes(5); // 本生成は 84s 規模になり得るため長め
});

// 複数 dollama エンドポイントの切替 (appsettings "Dollama:Endpoints" + ui/data/endpoints.json)
builder.Services.AddSingleton<Dollama.Ui.Services.EndpointRegistry>();
builder.Services.AddScoped<Dollama.Ui.Services.DollamaClient>();

// 生成中フラグ (Broadcaster と Generate.razor で共有) と テレメトリ常駐サービス
builder.Services.AddSingleton<GenerationActivity>();
builder.Services.AddHostedService<TelemetryBroadcaster>();

// プリセット永続化ストア (ui/data/presets.json)
builder.Services.AddSingleton<Dollama.Ui.Services.PresetStore>();

// キュレーション済みタグパレット (wwwroot/tag-palette.json・起動時に 1 回読む)
builder.Services.AddSingleton<Dollama.Ui.Services.TagPaletteCatalog>();

// LoRA カタログ (wwwroot/loras.json・起動時に 1 回読む)
builder.Services.AddSingleton<Dollama.Ui.Services.LoraCatalog>();

// お気に入りタグの永続化ストア (ui/data/favorites.json)
builder.Services.AddSingleton<Dollama.Ui.Services.FavoriteTagStore>();

// タグの日本語ラベル辞書 (wwwroot/tag-labels.ja.json・起動時に 1 回読む)
builder.Services.AddSingleton<Dollama.Ui.Services.TagLabels>();

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
    app.UseHsts();
}
app.UseStatusCodePagesWithReExecute("/not-found", createScopeForStatusCodePages: true);

app.UseAntiforgery();

// プリセットのサムネイル PNG を /thumb で静的公開する (ui/data/thumbs/)。
// 起動時に必ずディレクトリを作る (無いと PhysicalFileProvider が落ちる)。
// MapStaticAssets (wwwroot 配信) とは別 RequestPath なので順序衝突しない。
var thumbsDir = Path.Combine(app.Environment.ContentRootPath, "data", "thumbs");
Directory.CreateDirectory(thumbsDir);
app.UseStaticFiles(new StaticFileOptions
{
    FileProvider = new PhysicalFileProvider(thumbsDir),
    RequestPath = "/thumb",
});

app.MapStaticAssets();
app.MapHub<TelemetryHub>("/hubs/telemetry");
app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();
